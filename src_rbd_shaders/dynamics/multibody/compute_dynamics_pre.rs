//! Fused FK + body-jacobians + velocity-propagation + CRBA mass-matrix
//! kernel.
//!
//! Replaces a 4-dispatch chain
//! (`gpu_mb_forward_kinematics` → `gpu_mb_body_jacobians` →
//! `gpu_mb_update_velocities` → `gpu_mb_mass_matrix_with_coriolis`) with a
//! single workgroup-parallel kernel: each phase reuses the same `(multibody,
//! batch)` workgroup and a workgroup-shared scratch slot, removing three
//! WebGPU dispatches per `compute_dynamics` call.
//!
//! Phases (32 lanes per workgroup):
//!   1. Forward kinematics — link walk (lane 0 only; others idle at barrier).
//!      Writes workspace `local_to_parent`, `local_to_world`, `shift02`,
//!      `shift23` and pushes the link world pose into the shared `poses`
//!      buffer for downstream consumers (mprops update, broad phase).
//!   2. Body jacobians — per-link sequential outer loop, 32-lane parallel
//!      column work; matches `gpu_mb_body_jacobians`.
//!   3. Velocity propagation — link walk (lane 0). Mirrors
//!      `gpu_mb_update_velocities`.
//!   4. CRBA + Coriolis mass-matrix assembly — per-link sequential outer
//!      loop, 32-lane parallel inner work. Mirrors
//!      `gpu_mb_mass_matrix_with_coriolis`.
//!
//! Bindings: 14 storage + 8 uniform — exactly at the 14-storage WebGPU limit
//! requested by the testbed. The follow-up `gpu_mb_gravity_and_lu` kernel
//! finishes the dynamics pipeline (gravity rhs + LU factor + LU solve).

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

#[cfg(feature = "dim3")]
use glamx::{Mat3, Vec3};

use parry::math::VectorExt;
use crate::dynamics::body::{LocalMassProperties, Velocity};
use crate::dynamics::joint::SPATIAL_DIM;
use crate::utils::{Slice, SliceMut};
use crate::utils::linalg::{
    MAX_MB_DOFS, MatSlice, copy_from_par, fill_par, gemm_inertia_lhs_par,
    gemm_omega_skew_tr_cross_buf_par, gemm_skew_tr_lhs_cross_buf_par, gemm_skew_tr_lhs_par,
    gemm_tr_par, quadform_spatial_par,
};
#[cfg(feature = "dim3")]
use crate::utils::linalg::gemm_skew_lhs_cross_buf_par;
use crate::{ANG_DIM, AngVector, DIM, Pose, Vector, gcross_av};

use super::jacobian::{joint_jacobian_accumulate_par, joint_jacobian_column};
use super::mass_matrix::link_world_inertia;
use super::types::{MultibodyInfo, MultibodyLinkStatic, MultibodyLinkWorkspace};
use super::utils::body_to_parent;

const LANES: u32 = 32;

// TODO: refactor into multiple functions (but single kernel) to share between the coriolis and non-coriolis versions.
/// Fused FK + body-jacobians + velocity propagation + CRBA-with-Coriolis.
#[spirv_bindgen]
#[spirv(compute(threads(32, 1, 1)))]
pub fn gpu_mb_compute_dynamics_pre(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    links_workspace: &mut [MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] links_local_mprops: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] poses: &mut [Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] body_jacobians: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] mass_matrices: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] coriolis_packed: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] dof_state: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 10)] dt_uniform: &f32,
    #[spirv(uniform, descriptor_set = 0, binding = 11)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 12)] links_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 13)] jacobians_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 14)] mass_matrix_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 15)] coriolis_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 16)] i_coriolis_dt_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 17)] dof_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 18)] colliders_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 19)] coriolis_w_section_offset: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 20)] i_coriolis_dt_section_offset: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 21)] dof_damping_section_offset: &u32,
    // Dummy workgroup cell forces the khal CPU dispatch to use the coroutine
    // path (for parity with the original kernels that needed it). Cheap on
    // GPU — unused.
    #[spirv(workgroup)] _cpu_marker: &mut u32,
) {
    let batch_id = wg_id.y as usize;
    let mb_idx = wg_id.x;
    let lane = lid.x;
    // Padding multibody slots have `num_links == 0` and `ndofs == 0` so all
    // per-link / per-DOF loops below iterate zero times. No early-return —
    // WGSL's naga frontend can't prove a storage-loaded comparison is
    // uniform across the workgroup, so any subsequent `workgroupBarrier()`
    // would be flagged "called from non-uniform control flow". See
    // `gpu_mb_lu_decompose` for the rationale.
    let _ = num_multibodies;

    let dt = *dt_uniform;

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let links_start = batch_id * *links_batch_capacity as usize;
    let jac_start = batch_id * *jacobians_batch_capacity as usize;
    let mm_start = batch_id * *mass_matrix_batch_capacity as usize;
    let cor_start = batch_id * *coriolis_batch_capacity as usize;
    let icdt_start = batch_id * *i_coriolis_dt_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;
    let coll_start = batch_id * *colliders_batch_capacity as usize;

    // Section offsets within the packed buffers.
    let cor_w_off = *coriolis_w_section_offset as usize;
    let cor_icdt_off = *i_coriolis_dt_section_offset as usize;
    let damp_off = *dof_damping_section_offset as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let num_links = mb.num_links;
    let ndofs = mb.ndofs;
    let first_link_global = links_start + mb.first_link as usize;
    let mb_jac_base = jac_start + mb.jacobian_offset as usize;
    let mb_mm_base = mm_start + mb.mass_matrix_offset as usize;
    let mb_cor_base = cor_start + mb.coriolis_offset as usize;
    let mb_cor_w_base = cor_w_off + mb_cor_base;
    let mb_icdt_base = cor_icdt_off + icdt_start + mb.i_coriolis_dt_offset as usize;
    let damping_base = dof_start + mb.first_dof as usize;
    let gen_base = damping_base;

    let stat_slice = Slice(links_static, first_link_global);
    let mut ws_slice = SliceMut(links_workspace, first_link_global);
    let local_mprops_slice = Slice(links_local_mprops, first_link_global);
    let mut poses_slice = SliceMut(poses, coll_start);
    let damping_slice = Slice(dof_state, damp_off + damping_base);
    let vel_slice = Slice(dof_state, gen_base);

    // ============== Phase 1: Forward Kinematics ==============
    // Sequential parent-before-child link walk; lane 0 only, others idle at
    // the trailing barrier.
    if lane == 0 {
        // Root pose.
        let stat0 = stat_slice.read(0);
        let root_pose = if mb.root_is_dynamic == 0 {
            poses_slice.read(stat0.rb_id as usize)
        } else {
            let ws_ref = ws_slice.at(0);
            let pose = body_to_parent(&stat0, ws_ref.joint_rot, &ws_ref.coords);
            poses_slice.write(stat0.rb_id as usize, pose);
            pose
        };
        let link0 = ws_slice.at_mut(0);
        link0.local_to_parent = root_pose;
        link0.local_to_world = root_pose;

        for k in 1..num_links {
            let k_usize = k as usize;
            let stat = stat_slice.at(k_usize);
            let local_to_parent;
            let parent_to_world;
            {
                let ws_ref = ws_slice.at(k_usize);
                let parent_ref = ws_slice.at(stat.parent_link_id as usize);
                parent_to_world = parent_ref.local_to_world;
                local_to_parent = body_to_parent(&stat, ws_ref.joint_rot, &ws_ref.coords);
            }
            let local_to_world = parent_to_world * local_to_parent;

            let parent_lmp = local_mprops_slice.read(stat.parent_link_id as usize);
            let lmp = local_mprops_slice.read(k_usize);
            let world_com = local_to_world * lmp.com;
            let parent_com_world = parent_to_world * parent_lmp.com;
            let child_anchor_world = local_to_world * stat.data.local_frame_b.translation;
            let shift02 = child_anchor_world - parent_com_world;
            let shift23 = world_com - child_anchor_world;

            let link_mut = ws_slice.at_mut(k_usize);
            link_mut.local_to_parent = local_to_parent;
            link_mut.local_to_world = local_to_world;
            link_mut.shift02 = shift02;
            link_mut.shift23 = shift23;
            poses_slice.write(stat.rb_id as usize, local_to_world);
        }
    }
    workgroup_memory_barrier_with_group_sync();

    // ============== Phase 2: Body Jacobians ==============
    // NOTE: fixed number of iterations for uniform control flow.
    // TODO(PERF): on non-web platforms we could just use `mb.num_links` as the upper bound.
    for k in 0..MAX_MB_DOFS as u32 {
        let mut parent_to_world = Pose::default();
        let link_j = MatSlice::dense(
            mb_jac_base + (k as usize) * SPATIAL_DIM * (ndofs as usize),
            SPATIAL_DIM as u32,
            ndofs,
        );


        if k < num_links {
            let link_infos = stat_slice.at(k as usize);
            let link = ws_slice.at(k as usize);

            if k != 0 {
                let parent_j = MatSlice::dense(
                    mb_jac_base
                        + (link_infos.parent_link_id as usize)
                        * SPATIAL_DIM
                        * (ndofs as usize),
                    SPATIAL_DIM as u32,
                    ndofs,
                );
                let parent_link = ws_slice.at(link_infos.parent_link_id as usize);
                parent_to_world = parent_link.local_to_world;

                copy_from_par(body_jacobians, link_j, parent_j, lane, LANES);
                let link_j_v = link_j.fixed_rows(0, DIM);
                let parent_j_w = parent_j.fixed_rows(DIM, ANG_DIM);
                gemm_skew_tr_lhs_par(
                    body_jacobians,
                    link_j_v,
                    1.0,
                    link.shift02,
                    parent_j_w,
                    1.0,
                    lane,
                    LANES,
                );
            } else {
                fill_par(body_jacobians, link_j, 0.0, lane, LANES);
                parent_to_world = Pose::default();
            }
        }

        workgroup_memory_barrier_with_group_sync();

        if k < num_links {
            let link_infos = stat_slice.at(k as usize);
            let link_j_part = link_j.columns(link_infos.assembly_id, link_infos.ndofs);
            joint_jacobian_accumulate_par(
                link_infos,
                parent_to_world.rotation * link_infos.data.local_frame_a.rotation,
                body_jacobians,
                link_j_part,
                lane,
                LANES,
            );
        }

        workgroup_memory_barrier_with_group_sync();

        if k < num_links {
            let link = ws_slice.at(k as usize);
            let (link_j_v, link_j_w) = link_j.rows_range_pair(0, DIM, DIM, ANG_DIM);
            gemm_skew_tr_lhs_par(
                body_jacobians,
                link_j_v,
                1.0,
                link.shift23,
                link_j_w,
                1.0,
                lane,
                LANES,
            );
        }

        workgroup_memory_barrier_with_group_sync();
    }

    // ============== Phase 3: Velocity Propagation ==============
    if lane == 0 {
        for k in 0..num_links {
            let k_usize = k as usize;
            let stat = stat_slice.read(k_usize);

            let (jv_local_lin, jv_local_ang) =
                jacobian_mul_coordinates(stat.data.locked_axes, stat.assembly_id, &vel_slice);

            let (joint_velocity, rb_vels) = if k == 0 {
                let jv = Velocity::new(jv_local_lin, jv_local_ang);
                (jv, jv)
            } else {
                let parent_id = stat.parent_link_id as usize;
                let parent_ws = ws_slice.at(parent_id);
                let parent_to_world_rot = parent_ws.local_to_world.rotation;
                let parent_world_com_pose = parent_ws.local_to_world;
                let parent_rb_lin = parent_ws.rb_vels.linear;
                let parent_rb_ang = parent_ws.rb_vels.angular;

                let parent_lmp = local_mprops_slice.read(parent_id);
                let transform_rot = parent_to_world_rot * stat.data.local_frame_a.rotation;

                #[cfg(feature = "dim3")]
                let joint_velocity = Velocity::new(
                    transform_rot * jv_local_lin,
                    transform_rot * jv_local_ang,
                );
                #[cfg(feature = "dim2")]
                let joint_velocity = Velocity::new(transform_rot * jv_local_lin, jv_local_ang);

                let (self_local_to_world, self_shift23) = {
                    let ws_ref = ws_slice.at(k_usize);
                    (ws_ref.local_to_world, ws_ref.shift23)
                };

                let lmp = local_mprops_slice.read(k_usize);
                let world_com = self_local_to_world * lmp.com;
                let parent_world_com = parent_world_com_pose * parent_lmp.com;
                let shift = world_com - parent_world_com;

                let mut new_lin = parent_rb_lin + joint_velocity.linear;
                let new_ang = parent_rb_ang + joint_velocity.angular;
                new_lin += gcross_av(parent_rb_ang, shift);
                new_lin += gcross_av(joint_velocity.angular, self_shift23);

                (joint_velocity, Velocity::new(new_lin, new_ang))
            };

            let link_mut = ws_slice.at_mut(k_usize);
            link_mut.joint_velocity = joint_velocity;
            link_mut.rb_vels = rb_vels;
        }
    }
    workgroup_memory_barrier_with_group_sync();

    // ============== Phase 4: CRBA + Coriolis Mass Matrix ==============
    let acc_augmented_mass = MatSlice::dense(mb_mm_base, ndofs, ndofs);
    fill_par(mass_matrices, acc_augmented_mass, 0.0, lane, LANES);

    let i_coriolis_dt_view = MatSlice::dense(mb_icdt_base, SPATIAL_DIM as u32, ndofs);
    let i_coriolis_dt_v = i_coriolis_dt_view.fixed_rows(0, DIM);
    let i_coriolis_dt_w = i_coriolis_dt_view.fixed_rows(DIM, ANG_DIM);

    workgroup_memory_barrier_with_group_sync();

    // NOTE: fixed number of iterations for uniform control flow.
    // TODO(PERF): on non-web platforms we could just use `mb.num_links` as the upper bound.
    for k in 0..MAX_MB_DOFS as u32 {
        let loop_is_active = k < num_links;
        let mut inv_mass_x = 0.0;
        let mut mass = 0.0;

        if loop_is_active {
            let lmp = local_mprops_slice.read(k as usize);

            inv_mass_x = lmp.inv_mass.x;

            if inv_mass_x == 0.0 {
                let coriolis_block = MatSlice::dense(
                    mb_cor_base + (k as usize) * (DIM as usize) * (ndofs as usize),
                    DIM,
                    ndofs,
                );
                fill_par(coriolis_packed, coriolis_block, 0.0, lane, LANES);
                fill_par(
                    coriolis_packed,
                    MatSlice::dense(
                        mb_cor_w_base + (k as usize) * (DIM as usize) * (ndofs as usize),
                        DIM,
                        ndofs,
                    ),
                    0.0,
                    lane,
                    LANES,
                );
            }
        }
        // Uniform barrier so subsequent parent-coriolis reads see consistent
        // state — WebGPU forbids a barrier inside divergent control flow.
        workgroup_memory_barrier_with_group_sync();

        let loop_is_active = k < num_links && inv_mass_x != 0.0;
        let coriolis_v_i = MatSlice::dense(
            mb_cor_base + (k as usize) * (DIM as usize) * (ndofs as usize),
            DIM,
            ndofs,
        );
        let coriolis_w_i = MatSlice::dense(
            mb_cor_w_base + (k as usize) * (DIM as usize) * (ndofs as usize),
            ANG_DIM,
            ndofs,
        );
        let body_jacobian = MatSlice::dense(
            mb_jac_base + (k as usize) * SPATIAL_DIM * (ndofs as usize),
            SPATIAL_DIM as u32,
            ndofs,
        );
        let rb_j_w = body_jacobian.fixed_rows(DIM, ANG_DIM);
        let mut rb_inertia = Default::default();

        if loop_is_active {
            let ws = ws_slice.at(k as usize);
            let lmp = local_mprops_slice.read(k as usize);
            mass = 1.0 / inv_mass_x;
            rb_inertia = link_world_inertia(ws, &lmp);

            #[cfg(feature = "dim3")]
            let augmented_inertia = {
                let angvel = ws.rb_vels.angular;
                let w_skew = crate::utils::linalg::skew(angvel);
                let i_omega = rb_inertia * angvel;
                let i_omega_skew = crate::utils::linalg::skew(i_omega);
                let gyro_mat = w_skew * rb_inertia - i_omega_skew;
                rb_inertia + gyro_mat * dt
            };
            #[cfg(feature = "dim2")]
            let augmented_inertia = rb_inertia;

            quadform_spatial_par(
                mass_matrices,
                acc_augmented_mass,
                1.0,
                mass,
                augmented_inertia,
                body_jacobians,
                body_jacobian,
                1.0,
                lane,
                LANES,
            );

            if k != 0 {
                let stat = stat_slice.read(k as usize);
                let parent_id = stat.parent_link_id;
                let parent_link = ws_slice.at(parent_id as usize);
                let parent_j = MatSlice::dense(
                    mb_jac_base + (parent_id as usize) * SPATIAL_DIM * (ndofs as usize),
                    SPATIAL_DIM as u32,
                    ndofs,
                );
                let parent_j_w = parent_j.fixed_rows(DIM, ANG_DIM);
                let parent_coriolis_v = MatSlice::dense(
                    mb_cor_base + (parent_id as usize) * (DIM as usize) * (ndofs as usize),
                    DIM,
                    ndofs,
                );
                let parent_coriolis_w = MatSlice::dense(
                    mb_cor_w_base + (parent_id as usize) * (DIM as usize) * (ndofs as usize),
                    ANG_DIM,
                    ndofs,
                );
                let parent_w = parent_link.rb_vels.angular;

                copy_from_par(coriolis_packed, coriolis_v_i, parent_coriolis_v, lane, LANES);
                copy_from_par(coriolis_packed, coriolis_w_i, parent_coriolis_w, lane, LANES);

                gemm_skew_tr_lhs_par(
                    coriolis_packed,
                    coriolis_v_i,
                    1.0,
                    ws.shift02,
                    parent_coriolis_w,
                    1.0,
                    lane,
                    LANES,
                );

                let dvel = crate::gcross_av(ws.rb_vels.angular, ws.shift02)
                    + ws.joint_velocity.linear * 2.0;
                gemm_skew_tr_lhs_cross_buf_par(
                    coriolis_packed,
                    coriolis_v_i,
                    1.0,
                    dvel,
                    body_jacobians,
                    parent_j_w,
                    1.0,
                    lane,
                    LANES,
                );

                gemm_skew_tr_lhs_cross_buf_par(
                    coriolis_packed,
                    coriolis_v_i,
                    1.0,
                    ws.joint_velocity.linear,
                    body_jacobians,
                    parent_j_w,
                    1.0,
                    lane,
                    LANES,
                );

                gemm_omega_skew_tr_cross_buf_par(
                    coriolis_packed,
                    coriolis_v_i,
                    1.0,
                    parent_w,
                    ws.shift02,
                    body_jacobians,
                    parent_j_w,
                    1.0,
                    lane,
                    LANES,
                );

                #[cfg(feature = "dim3")]
                {
                    gemm_skew_lhs_cross_buf_par(
                        coriolis_packed,
                        coriolis_w_i,
                        -1.0,
                        ws.joint_velocity.angular,
                        body_jacobians,
                        parent_j_w,
                        1.0,
                        lane,
                        LANES,
                    );
                }
            }
        }

        workgroup_memory_barrier_with_group_sync();

        if loop_is_active {
            if k != 0 {
                let stat = stat_slice.read(k as usize);
                let parent_id = stat.parent_link_id;
                let parent_link = ws_slice.at(parent_id as usize);

                if stat.kinematic == 0 {
                    let transform_rot =
                        parent_link.local_to_world.rotation * stat.data.local_frame_a.rotation;
                    let coriolis_v_part = coriolis_v_i.columns(stat.assembly_id, stat.ndofs);
                    let coriolis_w_part = coriolis_w_i.columns(stat.assembly_id, stat.ndofs);

                    #[cfg(feature = "dim3")]
                    {
                        let parent_w_skew = crate::utils::linalg::skew(parent_link.rb_vels.angular);
                        let c = lane;
                        if c < stat.ndofs {
                            let (jv, jw) = joint_jacobian_column(&stat, transform_rot, c);
                            let pv = parent_w_skew * jv;
                            let pw = parent_w_skew * jw;
                            let iv0 = coriolis_v_part.idx(0, c);
                            let iv1 = coriolis_v_part.idx(1, c);
                            let iv2 = coriolis_v_part.idx(2, c);
                            coriolis_packed.write(iv0, coriolis_packed.read(iv0) + 2.0 * pv.x);
                            coriolis_packed.write(iv1, coriolis_packed.read(iv1) + 2.0 * pv.y);
                            coriolis_packed.write(iv2, coriolis_packed.read(iv2) + 2.0 * pv.z);
                            let iw0 = coriolis_w_part.idx(0, c);
                            let iw1 = coriolis_w_part.idx(1, c);
                            let iw2 = coriolis_w_part.idx(2, c);
                            coriolis_packed.write(iw0, coriolis_packed.read(iw0) + pw.x);
                            coriolis_packed.write(iw1, coriolis_packed.read(iw1) + pw.y);
                            coriolis_packed.write(iw2, coriolis_packed.read(iw2) + pw.z);
                        }
                    }
                    #[cfg(feature = "dim2")]
                    {
                        let c = lane;
                        if c < stat.ndofs {
                            let (jv, _) = joint_jacobian_column(&stat, transform_rot, c);
                            let iv0 = coriolis_v_part.idx(0, c);
                            let iv1 = coriolis_v_part.idx(1, c);
                            coriolis_packed.write(iv0, coriolis_packed.read(iv0) + 2.0 * (-parent_w * jv.y));
                            coriolis_packed.write(iv1, coriolis_packed.read(iv1) + 2.0 * (parent_w * jv.x));
                        }
                        let _ = coriolis_w_part;
                    }
                }
            } else {
                fill_par(coriolis_packed, coriolis_v_i, 0.0, lane, LANES);
                fill_par(coriolis_packed, coriolis_w_i, 0.0, lane, LANES);
            }
        }

        workgroup_memory_barrier_with_group_sync();

        if loop_is_active {
            let ws = ws_slice.at(k as usize);
            gemm_skew_tr_lhs_par(
                coriolis_packed,
                coriolis_v_i,
                1.0,
                ws.shift23,
                coriolis_w_i,
                1.0,
                lane,
                LANES,
            );

            let dvel_23 = crate::gcross_av(ws.rb_vels.angular, ws.shift23);
            gemm_skew_tr_lhs_cross_buf_par(
                coriolis_packed,
                coriolis_v_i,
                1.0,
                dvel_23,
                body_jacobians,
                rb_j_w,
                1.0,
                lane,
                LANES,
            );

            gemm_omega_skew_tr_cross_buf_par(
                coriolis_packed,
                coriolis_v_i,
                1.0,
                ws.rb_vels.angular,
                ws.shift23,
                body_jacobians,
                rb_j_w,
                1.0,
                lane,
                LANES,
            );
        }

        workgroup_memory_barrier_with_group_sync();

        if loop_is_active {
            // i_coriolis_dt assembly: dt · (mass·coriolis_v, I·coriolis_w).
            {
                let scale = mass * dt;
                let c = lane;
                if c < ndofs {
                    for r in 0..DIM {
                        let v = coriolis_packed.read(coriolis_v_i.idx(r, c));
                        coriolis_packed.write(i_coriolis_dt_v.idx(r, c), scale * v);
                    }
                }
            }
            gemm_inertia_lhs_par(
                coriolis_packed,
                i_coriolis_dt_w,
                dt,
                rb_inertia,
                coriolis_w_i,
                0.0,
                lane,
                LANES,
            );
        }

        workgroup_memory_barrier_with_group_sync();

        if loop_is_active {
            gemm_tr_par(
                mass_matrices,
                acc_augmented_mass,
                1.0,
                body_jacobians,
                body_jacobian,
                coriolis_packed,
                i_coriolis_dt_view,
                1.0,
                lane,
                LANES,
            );
        }

        workgroup_memory_barrier_with_group_sync();
    }

    // Damping diagonal: M[i, i] += damping[i] * dt — parallel.
    let d = lane;
    if d < ndofs {
        let diag_idx = acc_augmented_mass.idx(d, d);
        let cur = mass_matrices.read(diag_idx);
        mass_matrices.write(diag_idx, cur + damping_slice.read(d as usize) * dt);
    }
}


/// Fused FK + body-jacobians + velocity propagation + CRBA-with-Coriolis.
#[spirv_bindgen]
#[spirv(compute(threads(32, 1, 1)))]
pub fn gpu_mb_compute_dynamics_without_coriolis_pre(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    links_workspace: &mut [MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] links_local_mprops: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] poses: &mut [Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] body_jacobians: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] mass_matrices: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] dof_state: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 9)] dt_uniform: &f32,
    #[spirv(uniform, descriptor_set = 0, binding = 10)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 11)] links_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 12)] jacobians_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 13)] mass_matrix_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 14)] dof_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 15)] colliders_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 16)] dof_damping_section_offset: &u32,
    // Dummy workgroup cell forces the khal CPU dispatch to use the coroutine
    // path (for parity with the original kernels that needed it). Cheap on
    // GPU — unused.
    #[spirv(workgroup)] _cpu_marker: &mut u32,
) {
    let batch_id = wg_id.y as usize;
    let mb_idx = wg_id.x;
    let lane = lid.x;
    // No early-return on out-of-range `mb_idx` — see `gpu_mb_lu_decompose`
    // for the WGSL uniformity rationale. Dummy multibody slots have zero
    // links / DOFs, so all per-link loops below iterate zero times.
    let _ = num_multibodies;

    let dt = *dt_uniform;

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let links_start = batch_id * *links_batch_capacity as usize;
    let jac_start = batch_id * *jacobians_batch_capacity as usize;
    let mm_start = batch_id * *mass_matrix_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;
    let coll_start = batch_id * *colliders_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let num_links = mb.num_links;
    let ndofs = mb.ndofs;
    let first_link_global = links_start + mb.first_link as usize;
    let mb_jac_base = jac_start + mb.jacobian_offset as usize;
    let mb_mm_base = mm_start + mb.mass_matrix_offset as usize;
    let damping_base = dof_start + mb.first_dof as usize;
    let gen_base = damping_base;

    let damp_off = *dof_damping_section_offset as usize;

    let stat_slice = Slice(links_static, first_link_global);
    let mut ws_slice = SliceMut(links_workspace, first_link_global);
    let local_mprops_slice = Slice(links_local_mprops, first_link_global);
    let mut poses_slice = SliceMut(poses, coll_start);
    let damping_slice = Slice(dof_state, damp_off + damping_base);
    let vel_slice = Slice(dof_state, gen_base);

    // ============== Phase 1: Forward Kinematics ==============
    // Sequential parent-before-child link walk; lane 0 only, others idle at
    // the trailing barrier.
    if lane == 0 {
        // Root pose.
        let stat0 = stat_slice.read(0);
        let root_pose = if mb.root_is_dynamic == 0 {
            poses_slice.read(stat0.rb_id as usize)
        } else {
            let ws_ref = ws_slice.at(0);
            let pose = body_to_parent(&stat0, ws_ref.joint_rot, &ws_ref.coords);
            poses_slice.write(stat0.rb_id as usize, pose);
            pose
        };
        let link0 = ws_slice.at_mut(0);
        link0.local_to_parent = root_pose;
        link0.local_to_world = root_pose;

        for k in 1..num_links {
            let k_usize = k as usize;
            let stat = stat_slice.at(k_usize);
            let local_to_parent;
            let parent_to_world;
            {
                let ws_ref = ws_slice.at(k_usize);
                let parent_ref = ws_slice.at(stat.parent_link_id as usize);
                parent_to_world = parent_ref.local_to_world;
                local_to_parent = body_to_parent(&stat, ws_ref.joint_rot, &ws_ref.coords);
            }
            let local_to_world = parent_to_world * local_to_parent;

            let parent_lmp = local_mprops_slice.read(stat.parent_link_id as usize);
            let lmp = local_mprops_slice.read(k_usize);
            let world_com = local_to_world * lmp.com;
            let parent_com_world = parent_to_world * parent_lmp.com;
            let child_anchor_world = local_to_world * stat.data.local_frame_b.translation;
            let shift02 = child_anchor_world - parent_com_world;
            let shift23 = world_com - child_anchor_world;

            let link_mut = ws_slice.at_mut(k_usize);
            link_mut.local_to_parent = local_to_parent;
            link_mut.local_to_world = local_to_world;
            link_mut.shift02 = shift02;
            link_mut.shift23 = shift23;
            poses_slice.write(stat.rb_id as usize, local_to_world);
        }
    }
    workgroup_memory_barrier_with_group_sync();

    // ============== Phase 2: Body Jacobians ==============
    for k in 0..num_links {
        let link_infos = stat_slice.at(k as usize);
        let link = ws_slice.at(k as usize);

        let link_j = MatSlice::dense(
            mb_jac_base + (k as usize) * SPATIAL_DIM * (ndofs as usize),
            SPATIAL_DIM as u32,
            ndofs,
        );

        let parent_to_world;
        if k != 0 {
            let parent_j = MatSlice::dense(
                mb_jac_base
                    + (link_infos.parent_link_id as usize)
                    * SPATIAL_DIM
                    * (ndofs as usize),
                SPATIAL_DIM as u32,
                ndofs,
            );
            let parent_link = ws_slice.at(link_infos.parent_link_id as usize);
            parent_to_world = parent_link.local_to_world;

            copy_from_par(body_jacobians, link_j, parent_j, lane, LANES);
            let link_j_v = link_j.fixed_rows(0, DIM);
            let parent_j_w = parent_j.fixed_rows(DIM, ANG_DIM);
            gemm_skew_tr_lhs_par(
                body_jacobians,
                link_j_v,
                1.0,
                link.shift02,
                parent_j_w,
                1.0,
                lane,
                LANES,
            );
        } else {
            fill_par(body_jacobians, link_j, 0.0, lane, LANES);
            parent_to_world = Pose::default();
        }

        workgroup_memory_barrier_with_group_sync();

        let link_j_part = link_j.columns(link_infos.assembly_id, link_infos.ndofs);
        joint_jacobian_accumulate_par(
            link_infos,
            parent_to_world.rotation * link_infos.data.local_frame_a.rotation,
            body_jacobians,
            link_j_part,
            lane,
            LANES,
        );

        workgroup_memory_barrier_with_group_sync();
        let (link_j_v, link_j_w) = link_j.rows_range_pair(0, DIM, DIM, ANG_DIM);
        gemm_skew_tr_lhs_par(
            body_jacobians,
            link_j_v,
            1.0,
            link.shift23,
            link_j_w,
            1.0,
            lane,
            LANES,
        );
        workgroup_memory_barrier_with_group_sync();
    }

    // ============== Phase 3: Velocity Propagation ==============
    if lane == 0 {
        for k in 0..num_links {
            let k_usize = k as usize;
            let stat = stat_slice.read(k_usize);

            let (jv_local_lin, jv_local_ang) =
                jacobian_mul_coordinates(stat.data.locked_axes, stat.assembly_id, &vel_slice);

            let (joint_velocity, rb_vels) = if k == 0 {
                let jv = Velocity::new(jv_local_lin, jv_local_ang);
                (jv, jv)
            } else {
                let parent_id = stat.parent_link_id as usize;
                let parent_ws = ws_slice.at(parent_id);
                let parent_to_world_rot = parent_ws.local_to_world.rotation;
                let parent_world_com_pose = parent_ws.local_to_world;
                let parent_rb_lin = parent_ws.rb_vels.linear;
                let parent_rb_ang = parent_ws.rb_vels.angular;

                let parent_lmp = local_mprops_slice.read(parent_id);
                let transform_rot = parent_to_world_rot * stat.data.local_frame_a.rotation;

                #[cfg(feature = "dim3")]
                let joint_velocity = Velocity::new(
                    transform_rot * jv_local_lin,
                    transform_rot * jv_local_ang,
                );
                #[cfg(feature = "dim2")]
                let joint_velocity = Velocity::new(transform_rot * jv_local_lin, jv_local_ang);

                let (self_local_to_world, self_shift23) = {
                    let ws_ref = ws_slice.at(k_usize);
                    (ws_ref.local_to_world, ws_ref.shift23)
                };

                let lmp = local_mprops_slice.read(k_usize);
                let world_com = self_local_to_world * lmp.com;
                let parent_world_com = parent_world_com_pose * parent_lmp.com;
                let shift = world_com - parent_world_com;

                let mut new_lin = parent_rb_lin + joint_velocity.linear;
                let new_ang = parent_rb_ang + joint_velocity.angular;
                new_lin += gcross_av(parent_rb_ang, shift);
                new_lin += gcross_av(joint_velocity.angular, self_shift23);

                (joint_velocity, Velocity::new(new_lin, new_ang))
            };

            let link_mut = ws_slice.at_mut(k_usize);
            link_mut.joint_velocity = joint_velocity;
            link_mut.rb_vels = rb_vels;
        }
    }
    workgroup_memory_barrier_with_group_sync();

    // ============== Phase 4: CRBA ==============
    let acc_augmented_mass = MatSlice::dense(mb_mm_base, ndofs, ndofs);
    fill_par(mass_matrices, acc_augmented_mass, 0.0, lane, LANES);
    workgroup_memory_barrier_with_group_sync();

    for k in 0..num_links {
        let ws = ws_slice.at(k as usize);
        let lmp = local_mprops_slice.read(k as usize);

        let inv_mass_x = lmp.inv_mass.x;
        if inv_mass_x == 0.0 {
            continue;
        }
        let mass = 1.0 / inv_mass_x;
        let inertia = link_world_inertia(ws, &lmp);

        let body_jacobian = MatSlice::dense(
            mb_jac_base + (k as usize) * SPATIAL_DIM * (ndofs as usize),
            SPATIAL_DIM as u32,
            ndofs,
        );

        quadform_spatial_par(
            mass_matrices,
            acc_augmented_mass,
            1.0,
            mass,
            inertia,
            body_jacobians,
            body_jacobian,
            1.0,
            lane,
            LANES,
        );

        workgroup_memory_barrier_with_group_sync();
    }

    // Damping diagonal: M[i, i] += damping[i] * dt — parallel.
    let d = lane;
    if d < ndofs {
        let diag_idx = acc_augmented_mass.idx(d, d);
        let cur = mass_matrices.read(diag_idx);
        mass_matrices.write(diag_idx, cur + damping_slice.read(d as usize) * dt);
    }

    let _ = MAX_MB_DOFS;
}

/// Body-local velocity contributed by a joint, reading dof velocities directly
/// from the slice. Mirrors `velocity::jacobian_mul_coordinates`.
#[inline]
fn jacobian_mul_coordinates(
    locked_axes: u32,
    assembly_id: u32,
    vel_slice: &Slice<f32>,
) -> (Vector, AngVector) {
    let mut lin = Vector::ZERO;
    #[cfg(feature = "dim3")]
    let mut ang = AngVector::ZERO;
    #[cfg(feature = "dim2")]
    let mut ang: AngVector = 0.0;
    let mut curr = 0u32;

    for i in 0..DIM {
        if (locked_axes & (1 << i)) == 0 {
            let v = vel_slice.read((assembly_id + curr) as usize);
            lin += Vector::ith(i as usize, v);
            curr += 1;
        }
    }

    let ang_locked = (locked_axes >> DIM) & ((1 << ANG_DIM) - 1);
    let num_ang = ANG_DIM - ang_locked.count_ones();
    if num_ang == 1 {
        #[cfg(feature = "dim3")]
        {
            let dof_id = (!ang_locked & 0x7).trailing_zeros();
            let v = vel_slice.read((assembly_id + curr) as usize);
            ang += Vector::ith(dof_id as usize, v);
        }
        #[cfg(feature = "dim2")]
        {
            let v = vel_slice.read((assembly_id + curr) as usize);
            ang += v;
        }
    } else if num_ang == 3 {
        #[cfg(feature = "dim3")]
        {
            let vx = vel_slice.read((assembly_id + curr) as usize);
            let vy = vel_slice.read((assembly_id + curr + 1) as usize);
            let vz = vel_slice.read((assembly_id + curr + 2) as usize);
            ang += AngVector::new(vx, vy, vz);
        }
    }
    (lin, ang)
}
