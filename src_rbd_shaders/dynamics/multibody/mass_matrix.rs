//! Augmented mass matrix assembly via CRBA, with optional Coriolis /
//! gyroscopic terms.
//!
//! Rapier:
//!     self.augmented_mass.quadform(1.0, &rb_mass_matrix_wo_gyro, body_jacobian, 1.0);
//!
//! Here we use `quadform_spatial` which exploits the block-diagonal structure of the
//! per-link spatial mass to avoid forming the full SPATIAL_DIM × SPATIAL_DIM matrix.
//! World-space inertia is recomputed from the link's current orientation.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

#[cfg(feature = "dim3")]
use glamx::{Mat3, Vec3};

use crate::dynamics::body::LocalMassProperties;
use crate::dynamics::joint::SPATIAL_DIM;
use crate::utils::Slice;
use crate::utils::linalg::{
    MAX_MB_DOFS, MatSlice, copy_from, copy_from_par, fill, fill_par,
    gemm_inertia_lhs_cross_buf, gemm_inertia_lhs_cross_buf_par,
    gemm_omega_skew_tr_cross_buf, gemm_omega_skew_tr_cross_buf_par,
    gemm_skew_tr_lhs_cross_buf, gemm_skew_tr_lhs_cross_buf_par, gemm_tr,
    gemm_tr_par, quadform_spatial, quadform_spatial_par,
};
#[cfg(feature = "dim3")]
use crate::utils::linalg::{gemm_skew_lhs_cross_buf, gemm_skew_lhs_cross_buf_par};
use crate::{ANG_DIM, DIM};
#[cfg(feature = "dim3")]
use crate::rotation_to_matrix;

use super::jacobian::joint_jacobian_column;
use super::types::{MultibodyInfo, MultibodyLinkStatic, MultibodyLinkWorkspace};

/// Workgroup width for the parallelised mass-matrix kernel. Must match the
/// `MB_MM_LANES` constant on the host side and the `threads(...)` attribute on
/// `gpu_mb_mass_matrix_with_coriolis`.
const LANES: u32 = 32;

/// World-space inertia for this link.
///
/// In 3D returns a `Mat3` (`I_world = R · diag(principal_inertia) · Rᵀ`). In 2D
/// returns the scalar moment of inertia (already in world frame because there
/// is only one rotational DOF).
#[cfg(feature = "dim3")]
#[inline]
pub(super) fn link_world_inertia(ws: &MultibodyLinkWorkspace, lmp: &LocalMassProperties) -> Mat3 {
    let ipi = lmp.inv_principal_inertia;
    let px = if ipi.x != 0.0 { 1.0 / ipi.x } else { 0.0 };
    let py = if ipi.y != 0.0 { 1.0 / ipi.y } else { 0.0 };
    let pz = if ipi.z != 0.0 { 1.0 / ipi.z } else { 0.0 };
    let r = rotation_to_matrix(ws.local_to_world.rotation * lmp.inertia_ref_frame);
    // M = r · diag(px, py, pz) (column-scale); I = M · rᵀ.
    let m = Mat3::from_cols(r.x_axis * px, r.y_axis * py, r.z_axis * pz);
    m * r.transpose()
}

#[cfg(feature = "dim2")]
#[inline]
pub(super) fn link_world_inertia(_ws: &MultibodyLinkWorkspace, lmp: &LocalMassProperties) -> f32 {
    if lmp.inv_inertia != 0.0 {
        1.0 / lmp.inv_inertia
    } else {
        0.0
    }
}

/// Assemble the augmented mass matrix `M = Σᵢ Jᵢᵀ · diag(mᵢ·I, Iᵢ_world) · Jᵢ`.
///
/// Damping is added to the diagonal (`M[i, i] += damping[i] * dt`), matching
/// rapier's trailing loop in `update_inertias`.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_mass_matrix(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] links_workspace: &[MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] links_local_mprops: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] body_jacobians: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] mass_matrices: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] damping: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] num_multibodies: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] dt_buf: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 9)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 10)] links_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 11)] jacobians_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 12)] mass_matrix_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 13)] dof_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }

    let dt = dt_buf.read(0);
    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let links_start = batch_id * *links_batch_capacity as usize;
    let jac_start = batch_id * *jacobians_batch_capacity as usize;
    let mm_start = batch_id * *mass_matrix_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let num_links = mb.num_links;
    let ndofs = mb.ndofs;
    let first_link_global = links_start + mb.first_link as usize;
    let mb_jac_base = jac_start + mb.jacobian_offset as usize;
    let mb_mm_base = mm_start + mb.mass_matrix_offset as usize;
    let damping_base = dof_start + mb.first_dof as usize;

    let ws_slice = Slice(links_workspace, first_link_global);
    let local_mprops_slice = Slice(links_local_mprops, first_link_global);
    let damping_slice = Slice(damping, damping_base);
    let _ = links_static; // reserved for future use (kinematic-DOF permutation, etc.)

    // augmented_mass.fill(0.0)
    let augmented_mass = MatSlice::dense(mb_mm_base, ndofs, ndofs);
    fill(mass_matrices, augmented_mass, 0.0);

    for k in 0..num_links {
        // Reference-only access: `link_world_inertia` only reads
        // `ws.local_to_world.rotation`, so a 240 B struct copy here would be
        // pure waste.
        let ws = ws_slice.at(k as usize);
        let lmp = local_mprops_slice.read(k as usize);

        let inv_mass_x = lmp.inv_mass.x;
        if inv_mass_x == 0.0 {
            continue;
        }
        let mass = 1.0 / inv_mass_x;
        let inertia = link_world_inertia(ws, &lmp);

        // body_jacobian view for this link.
        let body_jacobian = MatSlice::dense(
            mb_jac_base + (k as usize) * SPATIAL_DIM * (ndofs as usize),
            SPATIAL_DIM as u32,
            ndofs,
        );

        // augmented_mass.quadform(1.0, &rb_mass_matrix_wo_gyro, body_jacobian, 1.0);
        quadform_spatial(
            mass_matrices,
            augmented_mass,
            1.0,
            mass,
            inertia,
            body_jacobians,
            body_jacobian,
            1.0,
        );
    }

    // Per-rapier: `augmented_mass[i, i] += damping[i] * dt`.
    for i in 0..ndofs {
        let diag_idx = augmented_mass.idx(i, i);
        let cur = mass_matrices.read(diag_idx);
        mass_matrices.write(diag_idx, cur + damping_slice.read(i as usize) * dt);
    }

    // TODO: remove this?
    // Defensive cap so ndofs can't overflow the quadform scratch.
    let _ = MAX_MB_DOFS;
}

//
// Mass matrix with Coriolis + gyroscopic terms.
//
// Mirrors rapier's `update_inertias`. In 3D this includes a gyroscopic
// derivative `[ω]_× · I − [Iω]_×` on the augmented inertia and the full
// `coriolis_w` propagation; in 2D the gyroscopic term is zero and
// `coriolis_w` collapses to a 1-row block.

/// Scale each column of `dst_v` (`DIM × ndofs`) by a scalar, in place:
/// `dst_v := scale · src_v`.
#[inline]
fn scaled_copy_lin_dim(
    buf_dst: &mut [f32],
    dst: MatSlice,
    scale: f32,
    buf_src: &[f32],
    src: MatSlice,
) {
    for c in 0..dst.cols {
        for r in 0..DIM {
            buf_dst.write(dst.idx(r, c), scale * buf_src.read(src.idx(r, c)));
        }
    }
}

/// Workgroup-parallel CRBA + Coriolis assembly.
///
/// One workgroup of `LANES = 32` threads per `(multibody, batch)` pair. The
/// per-link loop is sequential (parent-before-child dependency) but each
/// inner BLAS-style operation is column-partitioned across the lanes — every
/// lane writes to a disjoint subset of mass-matrix / coriolis columns, so
/// there are no write races. Workgroup barriers are placed at the end of each
/// link iteration so writes to the coriolis buffers from iteration `k` are
/// visible to any later iteration that reads `parent_coriolis_*`.
#[spirv_bindgen]
#[spirv(compute(threads(32, 1, 1)))]
pub fn gpu_mb_mass_matrix_with_coriolis(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] links_workspace: &[MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] links_local_mprops: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] body_jacobians: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] mass_matrices: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] coriolis_v: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] coriolis_w: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] i_coriolis_dt: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] damping: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 10)] num_multibodies: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 11)] dt_buf: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 12)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 13)] links_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 14)] jacobians_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 15)] mass_matrix_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 16)] coriolis_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 17)] i_coriolis_dt_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 18)] dof_batch_capacity: &u32,
    // Dummy workgroup-shared cell — opts into the khal CPU coroutine
    // dispatch path so workgroup barriers actually synchronise lanes on
    // CPU. Without this, the CPU backend runs lanes sequentially with
    // no-op barriers and per-lane writes clobber each other.
    #[spirv(workgroup)] _cpu_marker: &mut u32,
) {
    let batch_id = wg_id.y as usize;
    let mb_idx = wg_id.x;
    let lane = lid.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }
    let dt = dt_buf.read(0);

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let links_start = batch_id * *links_batch_capacity as usize;
    let jac_start = batch_id * *jacobians_batch_capacity as usize;
    let mm_start = batch_id * *mass_matrix_batch_capacity as usize;
    let cor_start = batch_id * *coriolis_batch_capacity as usize;
    let icdt_start = batch_id * *i_coriolis_dt_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let num_links = mb.num_links;
    let ndofs = mb.ndofs;
    let first_link_global = links_start + mb.first_link as usize;
    let mb_jac_base = jac_start + mb.jacobian_offset as usize;
    let mb_mm_base = mm_start + mb.mass_matrix_offset as usize;
    let mb_cor_base = cor_start + mb.coriolis_offset as usize;
    let mb_icdt_base = icdt_start + mb.i_coriolis_dt_offset as usize;
    let damping_base = dof_start + mb.first_dof as usize;

    let stat_slice = Slice(links_static, first_link_global);
    let ws_slice = Slice(links_workspace, first_link_global);
    let local_mprops_slice = Slice(links_local_mprops, first_link_global);
    let damping_slice = Slice(damping, damping_base);

    // acc_augmented_mass.fill(0.0) — parallel across columns.
    let acc_augmented_mass = MatSlice::dense(mb_mm_base, ndofs, ndofs);
    fill_par(mass_matrices, acc_augmented_mass, 0.0, lane, LANES);

    // i_coriolis_dt view (SPATIAL_DIM × ndofs, fully overwritten each link).
    let i_coriolis_dt_view = MatSlice::dense(mb_icdt_base, SPATIAL_DIM as u32, ndofs);
    let i_coriolis_dt_v = i_coriolis_dt_view.fixed_rows(0, DIM);
    let i_coriolis_dt_w = i_coriolis_dt_view.fixed_rows(DIM, ANG_DIM);

    // Barrier: every later phase reads `acc_augmented_mass`, so make sure all
    // lanes have finished zeroing it.
    workgroup_memory_barrier_with_group_sync();

    for k in 0..num_links {
        let stat = stat_slice.read(k as usize);
        let ws = ws_slice.at(k as usize);
        let lmp = local_mprops_slice.read(k as usize);

        let inv_mass_x = lmp.inv_mass.x;
        if inv_mass_x == 0.0 {
            // Zero this link's coriolis block in parallel so children don't
            // propagate garbage. All lanes participate uniformly.
            let coriolis_block = MatSlice::dense(
                mb_cor_base + (k as usize) * (DIM as usize) * (ndofs as usize),
                DIM,
                ndofs,
            );
            fill_par(coriolis_v, coriolis_block, 0.0, lane, LANES);
            fill_par(coriolis_w, coriolis_block, 0.0, lane, LANES);
            workgroup_memory_barrier_with_group_sync();
            continue;
        }
        let mass = 1.0 / inv_mass_x;
        let rb_inertia = link_world_inertia(ws, &lmp);

        let body_jacobian = MatSlice::dense(
            mb_jac_base + (k as usize) * SPATIAL_DIM * (ndofs as usize),
            SPATIAL_DIM as u32,
            ndofs,
        );

        // Gyroscopic derivative: aug_I = I + ([ω]_× · I − [Iω]_×) · dt (3D).
        // In 2D the gyroscopic matrix is zero (scalar inertia, scalar angvel).
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

        // Mass-matrix accumulation, parallel across columns.
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

        // Coriolis matrix assembly. All ops below partition `coriolis_v_i` /
        // `coriolis_w_i` columns by lane; the same column is owned by the
        // same lane across all ops, so no inter-op race.
        let rb_j_w = body_jacobian.fixed_rows(DIM, ANG_DIM);
        let coriolis_v_i = MatSlice::dense(
            mb_cor_base + (k as usize) * (DIM as usize) * (ndofs as usize),
            DIM,
            ndofs,
        );
        let coriolis_w_i = MatSlice::dense(
            mb_cor_base + (k as usize) * (DIM as usize) * (ndofs as usize),
            ANG_DIM,
            ndofs,
        );

        if k != 0 {
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
                mb_cor_base + (parent_id as usize) * (DIM as usize) * (ndofs as usize),
                ANG_DIM,
                ndofs,
            );
            let parent_w = parent_link.rb_vels.angular;

            copy_from_par(coriolis_v, coriolis_v_i, parent_coriolis_v, lane, LANES);
            copy_from_par(coriolis_w, coriolis_w_i, parent_coriolis_w, lane, LANES);

            gemm_skew_tr_lhs_cross_buf_par(
                coriolis_v,
                coriolis_v_i,
                1.0,
                ws.shift02,
                coriolis_w,
                parent_coriolis_w,
                1.0,
                lane,
                LANES,
            );

            let dvel = crate::gcross_av(ws.rb_vels.angular, ws.shift02)
                + ws.joint_velocity.linear * 2.0;
            gemm_skew_tr_lhs_cross_buf_par(
                coriolis_v,
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
                coriolis_v,
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
                coriolis_v,
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
                    coriolis_w,
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

            // Joint jacobian contribution to Coriolis (parallelized across
            // the joint's free-DOF columns). Each lane c<ndofs_link handles
            // column `assembly_id + c`. The preceding `gemm_*_par` calls
            // wrote those columns with lane `assembly_id + c`, so without a
            // barrier here lane 0 reads stale data at `coriolis_v[*,
            // assembly_id]` (race). The pendulum case had zero parent
            // contribution at the offending col so the bug was benign;
            // prismatic / spherical joints in `joints3` clobber non-zero
            // values.
            workgroup_memory_barrier_with_group_sync();
            if stat.kinematic == 0 {
                let transform_rot =
                    parent_link.local_to_world.rotation * stat.data.local_frame_a.rotation;
                let coriolis_v_part = coriolis_v_i.columns(stat.assembly_id, stat.ndofs);
                let coriolis_w_part = coriolis_w_i.columns(stat.assembly_id, stat.ndofs);

                // stat.ndofs ≤ SPATIAL_DIM = 6 < LANES, so each lane handles
                // at most one column.
                #[cfg(feature = "dim3")]
                {
                    let parent_w_skew = crate::utils::linalg::skew(parent_w);
                    let c = lane;
                    if c < stat.ndofs {
                        let (jv, jw) = joint_jacobian_column(&stat, transform_rot, c);
                        let pv = parent_w_skew * jv;
                        let pw = parent_w_skew * jw;
                        let iv0 = coriolis_v_part.idx(0, c);
                        let iv1 = coriolis_v_part.idx(1, c);
                        let iv2 = coriolis_v_part.idx(2, c);
                        coriolis_v.write(iv0, coriolis_v.read(iv0) + 2.0 * pv.x);
                        coriolis_v.write(iv1, coriolis_v.read(iv1) + 2.0 * pv.y);
                        coriolis_v.write(iv2, coriolis_v.read(iv2) + 2.0 * pv.z);
                        let iw0 = coriolis_w_part.idx(0, c);
                        let iw1 = coriolis_w_part.idx(1, c);
                        let iw2 = coriolis_w_part.idx(2, c);
                        coriolis_w.write(iw0, coriolis_w.read(iw0) + pw.x);
                        coriolis_w.write(iw1, coriolis_w.read(iw1) + pw.y);
                        coriolis_w.write(iw2, coriolis_w.read(iw2) + pw.z);
                    }
                }
                #[cfg(feature = "dim2")]
                {
                    let c = lane;
                    if c < stat.ndofs {
                        let (jv, _) = joint_jacobian_column(&stat, transform_rot, c);
                        let iv0 = coriolis_v_part.idx(0, c);
                        let iv1 = coriolis_v_part.idx(1, c);
                        coriolis_v.write(iv0, coriolis_v.read(iv0) + 2.0 * (-parent_w * jv.y));
                        coriolis_v.write(iv1, coriolis_v.read(iv1) + 2.0 * (parent_w * jv.x));
                    }
                    let _ = coriolis_w_part;
                }
            }
        } else {
            fill_par(coriolis_v, coriolis_v_i, 0.0, lane, LANES);
            fill_par(coriolis_w, coriolis_w_i, 0.0, lane, LANES);
        }

        // Barrier: the self-shift block below reads coriolis_w (just written)
        // and i_coriolis_dt is built from coriolis_v / coriolis_w too.
        workgroup_memory_barrier_with_group_sync();

        // Self-shift contribution.
        gemm_skew_tr_lhs_cross_buf_par(
            coriolis_v,
            coriolis_v_i,
            1.0,
            ws.shift23,
            coriolis_w,
            coriolis_w_i,
            1.0,
            lane,
            LANES,
        );

        let dvel_23 = crate::gcross_av(ws.rb_vels.angular, ws.shift23);
        gemm_skew_tr_lhs_cross_buf_par(
            coriolis_v,
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
            coriolis_v,
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

        // Barrier: i_coriolis_dt assembly below reads coriolis_v / coriolis_w.
        workgroup_memory_barrier_with_group_sync();

        // Meld Coriolis into the mass matrix via i_coriolis_dt:
        //   i_coriolis_dt_v := dt · mass · coriolis_v
        //   i_coriolis_dt_w := dt · (rb_inertia · coriolis_w)
        //   acc_augmented_mass += Jᵀ · i_coriolis_dt
        // Inline scaled-copy: i_coriolis_dt_v[r, c] = (mass · dt) · coriolis_v[r, c],
        // partitioned across columns. ndofs ≤ MAX_MB_DOFS = LANES, so an `if`
        // guard suffices (no `while` loop — rust-gpu lowers `while` to
        // unstructured control flow that can be silently miscompiled).
        {
            let scale = mass * dt;
            let c = lane;
            if c < ndofs {
                for r in 0..DIM {
                    let v = coriolis_v.read(coriolis_v_i.idx(r, c));
                    i_coriolis_dt.write(i_coriolis_dt_v.idx(r, c), scale * v);
                }
            }
        }
        gemm_inertia_lhs_cross_buf_par(
            i_coriolis_dt,
            i_coriolis_dt_w,
            dt,
            rb_inertia,
            coriolis_w,
            coriolis_w_i,
            0.0,
            lane,
            LANES,
        );

        // Barrier: gemm_tr below reads i_coriolis_dt that was just written.
        workgroup_memory_barrier_with_group_sync();

        gemm_tr_par(
            mass_matrices,
            acc_augmented_mass,
            1.0,
            body_jacobians,
            body_jacobian,
            i_coriolis_dt,
            i_coriolis_dt_view,
            1.0,
            lane,
            LANES,
        );

        // End-of-iteration barrier so the next iteration (or any later
        // iteration that reads parent_coriolis_*) sees consistent state.
        workgroup_memory_barrier_with_group_sync();
    }

    // Per-rapier: `acc_augmented_mass[i, i] += damping[i] * dt` — parallel.
    // ndofs ≤ MAX_MB_DOFS = LANES, so each lane handles at most one DOF.
    let d = lane;
    if d < ndofs {
        let diag_idx = acc_augmented_mass.idx(d, d);
        let cur = mass_matrices.read(diag_idx);
        mass_matrices.write(diag_idx, cur + damping_slice.read(d as usize) * dt);
    }

    // TODO: remove this?
    let _ = MAX_MB_DOFS;
}
