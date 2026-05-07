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

#[cfg(feature = "dim3")]
use glamx::{Mat3, Vec3};

use crate::dynamics::body::LocalMassProperties;
use crate::dynamics::joint::SPATIAL_DIM;
use crate::utils::Slice;
use crate::utils::linalg::{
    MAX_MB_DOFS, MatSlice, copy_from, fill, gemm_inertia_lhs_cross_buf,
    gemm_omega_skew_tr_cross_buf, gemm_skew_tr_lhs_cross_buf, gemm_tr, quadform_spatial,
};
#[cfg(feature = "dim3")]
use crate::utils::linalg::gemm_skew_lhs_cross_buf;
use crate::{ANG_DIM, DIM};
#[cfg(feature = "dim3")]
use crate::rotation_to_matrix;

use super::jacobian::joint_jacobian_column;
use super::types::{MultibodyInfo, MultibodyLinkStatic, MultibodyLinkWorkspace};

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

#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_mass_matrix_with_coriolis(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
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

    // acc_augmented_mass.fill(0.0)
    let acc_augmented_mass = MatSlice::dense(mb_mm_base, ndofs, ndofs);
    fill(mass_matrices, acc_augmented_mass, 0.0);

    // i_coriolis_dt view (SPATIAL_DIM × ndofs, fully overwritten each link).
    let i_coriolis_dt_view = MatSlice::dense(mb_icdt_base, SPATIAL_DIM as u32, ndofs);
    let i_coriolis_dt_v = i_coriolis_dt_view.fixed_rows(0, DIM);
    let i_coriolis_dt_w = i_coriolis_dt_view.fixed_rows(DIM, ANG_DIM);

    for k in 0..num_links {
        let stat = stat_slice.read(k as usize);
        // Many fields of `ws` are read below. Going through a reference lets
        // SPIR-V emit per-field OpLoads instead of materializing the full
        // 240 B struct in Function storage.
        let ws = ws_slice.at(k as usize);
        let lmp = local_mprops_slice.read(k as usize);

        let inv_mass_x = lmp.inv_mass.x;
        if inv_mass_x == 0.0 {
            // Still need to zero this link's coriolis block so children don't
            // propagate garbage. Shared layout: each slot reserves DIM rows in
            // both buffers (the angular block uses only the first ANG_DIM rows).
            let coriolis_block = MatSlice::dense(
                mb_cor_base + (k as usize) * (DIM as usize) * (ndofs as usize),
                DIM,
                ndofs,
            );
            fill(coriolis_v, coriolis_block, 0.0);
            fill(coriolis_w, coriolis_block, 0.0);
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

        // acc_augmented_mass.quadform(1.0, &concat_rb_mass_matrix(mass, augmented_inertia),
        //                             body_jacobian, 1.0);
        quadform_spatial(
            mass_matrices,
            acc_augmented_mass,
            1.0,
            mass,
            augmented_inertia,
            body_jacobians,
            body_jacobian,
            1.0,
        );

        // Coriolis matrix assembly.
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

            // coriolis_v.copy_from(parent_coriolis_v); coriolis_w.copy_from(parent_coriolis_w).
            copy_from(coriolis_v, coriolis_v_i, parent_coriolis_v);
            copy_from(coriolis_w, coriolis_w_i, parent_coriolis_w);

            // coriolis_v += [shift02]^T_× · parent_coriolis_w.
            gemm_skew_tr_lhs_cross_buf(
                coriolis_v,
                coriolis_v_i,
                1.0,
                ws.shift02,
                coriolis_w,
                parent_coriolis_w,
                1.0,
            );

            // coriolis_v += dvel_cross^T · parent_j_w  with
            //   dvel = rb.vels.angvel × shift02 + 2 · joint_velocity.linvel.
            let dvel = crate::gcross_av(ws.rb_vels.angular, ws.shift02)
                + ws.joint_velocity.linear * 2.0;
            gemm_skew_tr_lhs_cross_buf(
                coriolis_v,
                coriolis_v_i,
                1.0,
                dvel,
                body_jacobians,
                parent_j_w,
                1.0,
            );

            // coriolis_v += [joint_vel_lin]^T_× · parent_j_w.
            gemm_skew_tr_lhs_cross_buf(
                coriolis_v,
                coriolis_v_i,
                1.0,
                ws.joint_velocity.linear,
                body_jacobians,
                parent_j_w,
                1.0,
            );

            // coriolis_v += (parent_w · shift02_cross_tr) · parent_j_w.
            gemm_omega_skew_tr_cross_buf(
                coriolis_v,
                coriolis_v_i,
                1.0,
                parent_w,
                ws.shift02,
                body_jacobians,
                parent_j_w,
                1.0,
            );

            // coriolis_w += -[joint_vel_ang]_× · parent_j_w (only contributes in 3D).
            #[cfg(feature = "dim3")]
            {
                gemm_skew_lhs_cross_buf(
                    coriolis_w,
                    coriolis_w_i,
                    -1.0,
                    ws.joint_velocity.angular,
                    body_jacobians,
                    parent_j_w,
                    1.0,
                );
            }

            // Joint jacobian contribution to Coriolis (skipped for kinematic joints).
            // Computes the joint jacobian column-by-column in registers — no
            // `[f32; SPATIAL_DIM * SPATIAL_DIM]` Function-storage scratch.
            if stat.kinematic == 0 {
                let transform_rot =
                    parent_link.local_to_world.rotation * stat.data.local_frame_a.rotation;
                let coriolis_v_part = coriolis_v_i.columns(stat.assembly_id, stat.ndofs);
                let coriolis_w_part = coriolis_w_i.columns(stat.assembly_id, stat.ndofs);

                #[cfg(feature = "dim3")]
                {
                    let parent_w_skew = crate::utils::linalg::skew(parent_w);
                    for c in 0..stat.ndofs {
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
                    // 2D: rb_joint_j_v = (jv_x, jv_y), parent_w is a scalar ω.
                    // [ω]_× · v = (-ω·v.y, ω·v.x).
                    for c in 0..stat.ndofs {
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
            fill(coriolis_v, coriolis_v_i, 0.0);
            fill(coriolis_w, coriolis_w_i, 0.0);
        }

        // Self-shift contribution:
        //   coriolis_v += [shift23]^T_× · coriolis_w
        //   coriolis_v += [ω × shift23]^T_× · rb_j_w
        //   coriolis_v += (skew(ω) · [shift23]^T_×) · rb_j_w
        {
            gemm_skew_tr_lhs_cross_buf(
                coriolis_v,
                coriolis_v_i,
                1.0,
                ws.shift23,
                coriolis_w,
                coriolis_w_i,
                1.0,
            );

            let dvel_23 = crate::gcross_av(ws.rb_vels.angular, ws.shift23);
            gemm_skew_tr_lhs_cross_buf(
                coriolis_v,
                coriolis_v_i,
                1.0,
                dvel_23,
                body_jacobians,
                rb_j_w,
                1.0,
            );

            gemm_omega_skew_tr_cross_buf(
                coriolis_v,
                coriolis_v_i,
                1.0,
                ws.rb_vels.angular,
                ws.shift23,
                body_jacobians,
                rb_j_w,
                1.0,
            );
        }

        // Meld Coriolis into the mass matrix via i_coriolis_dt:
        //   i_coriolis_dt_v := dt · mass · coriolis_v
        //   i_coriolis_dt_w := dt · (rb_inertia · coriolis_w)
        //   acc_augmented_mass += Jᵀ · i_coriolis_dt.
        scaled_copy_lin_dim(
            i_coriolis_dt,
            i_coriolis_dt_v,
            mass * dt,
            coriolis_v,
            coriolis_v_i,
        );
        gemm_inertia_lhs_cross_buf(
            i_coriolis_dt,
            i_coriolis_dt_w,
            dt,
            rb_inertia,
            coriolis_w,
            coriolis_w_i,
            0.0,
        );
        gemm_tr(
            mass_matrices,
            acc_augmented_mass,
            1.0,
            body_jacobians,
            body_jacobian,
            i_coriolis_dt,
            i_coriolis_dt_view,
            1.0,
        );
    }

    // Per-rapier: `acc_augmented_mass[i, i] += damping[i] * dt`.
    for i in 0..ndofs {
        let diag_idx = acc_augmented_mass.idx(i, i);
        let cur = mass_matrices.read(diag_idx);
        mass_matrices.write(diag_idx, cur + damping_slice.read(i as usize) * dt);
    }

    // TODO: remove this?
    let _ = MAX_MB_DOFS;
}
