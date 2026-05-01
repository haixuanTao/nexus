//! Augmented mass matrix assembly via CRBA, with optional Coriolis /
//! gyroscopic terms.
//!
//! Rapier:
//!     self.augmented_mass.quadform(1.0, &rb_mass_matrix_wo_gyro, body_jacobian, 1.0);
//!
//! Here we use `quadform_spatial` which exploits the block-diagonal structure of the
//! per-link 6×6 spatial mass (`diag(m·I₃, I_world)`) to avoid forming the full 6×6.
//! World-space inertia is recomputed from the link's current orientation.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

use glamx::{Mat3, Vec3};

use crate::dynamics::body::LocalMassProperties;
use crate::rotation_to_matrix;
use crate::utils::Slice;
use crate::utils::linalg::{
    MAX_MB_DOFS, MatSlice, copy_from, fill, gemm_tr, quadform_spatial, skew, skew_tr,
};

use super::jacobian::joint_jacobian;
use super::types::{MultibodyInfo, MultibodyLinkStatic, MultibodyLinkWorkspace};

/// World-space 3×3 inertia for this link:
/// `I_world = R · diag(principal_inertia) · Rᵀ` with `R = world_rot · inertia_ref_frame`.
#[inline]
pub(super) fn link_world_inertia(ws: &MultibodyLinkWorkspace, lmp: &LocalMassProperties) -> Mat3 {
    let ipi = lmp.inv_principal_inertia;
    let px = if ipi.x != 0.0 { 1.0 / ipi.x } else { 0.0 };
    let py = if ipi.y != 0.0 { 1.0 / ipi.y } else { 0.0 };
    let pz = if ipi.z != 0.0 { 1.0 / ipi.z } else { 0.0 };
    let r = rotation_to_matrix(ws.local_to_world.rotation * lmp.inertia_ref_frame);
    // M = r · diag(px, py, pz) (column-scale); I = M · rᵀ.
    let m = Mat3::from_cols(
        r.x_axis * px, r.y_axis * py, r.z_axis * pz,
    );
    m * r.transpose()
}

/// Assemble the augmented mass matrix `M = Σᵢ Jᵢᵀ · diag(mᵢ·I₃, Iᵢ_world) · Jᵢ`.
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
        let ws = ws_slice.read(k as usize);
        let lmp = local_mprops_slice.read(k as usize);

        let inv_mass_x = lmp.inv_mass.x;
        if inv_mass_x == 0.0 {
            continue;
        }
        let mass = 1.0 / inv_mass_x;
        let inertia = link_world_inertia(&ws, &lmp);

        // body_jacobian view for this link.
        let body_jacobian = MatSlice::dense(
            mb_jac_base + (k as usize) * 6 * (ndofs as usize),
            6,
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
// Mirrors the full `Multibody::update_inertias` algorithm from rapier:
//   1. Per link, compute the gyroscopically-augmented inertia
//        I_aug = I + ([ω]_× · I − [Iω]_×) · dt
//      and accumulate `acc_augmented_mass += Jᵢᵀ · diag(mᵢ·I₃, I_aug) · Jᵢ`.
//   2. Build per-link `coriolis_v[i]` (3×ndofs) and `coriolis_w[i]` (3×ndofs)
//      recursively from the parent, using all of shift02, parent ω, joint velocity.
//   3. Add the self-shift contribution (shift23 + own ω).
//   4. Meld into `acc_augmented_mass` via `i_coriolis_dt` scratch:
//        i_coriolis_dt_v = dt · mass · coriolis_v
//        i_coriolis_dt_w = dt · I · coriolis_w
//        acc_augmented_mass += Jᵀ · i_coriolis_dt
//
// Requires `gpu_mb_update_velocities` to have been run first so that
// `ws.joint_velocity` and `ws.rb_vels` hold the current per-link world velocities.

/// Scale each column of `dst_v` (3 × ndofs) by a scalar, in place: `dst_v := scale · src_v`.
#[inline]
fn scaled_copy_3xn(
    buf_dst: &mut [f32],
    dst: MatSlice,
    scale: f32,
    buf_src: &[f32],
    src: MatSlice,
) {
    for c in 0..dst.cols {
        for r in 0u32..3 {
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

    // i_coriolis_dt view (6 × ndofs, fully overwritten each link).
    let i_coriolis_dt_view = MatSlice::dense(mb_icdt_base, 6, ndofs);
    let i_coriolis_dt_v = i_coriolis_dt_view.fixed_rows(0, 3);
    let i_coriolis_dt_w = i_coriolis_dt_view.fixed_rows(3, 3);

    for k in 0..num_links {
        let stat = stat_slice.read(k as usize);
        let ws = ws_slice.read(k as usize);
        let lmp = local_mprops_slice.read(k as usize);

        let inv_mass_x = lmp.inv_mass.x;
        if inv_mass_x == 0.0 {
            // Still need to zero this link's coriolis block so children don't
            // propagate garbage.
            let coriolis_v_i =
                MatSlice::dense(mb_cor_base + (k as usize) * 3 * (ndofs as usize), 3, ndofs);
            let coriolis_w_i = coriolis_v_i; // same shape + location in the other buffer
            fill(coriolis_v, coriolis_v_i, 0.0);
            fill(coriolis_w, coriolis_w_i, 0.0);
            continue;
        }
        let mass = 1.0 / inv_mass_x;
        let rb_inertia = link_world_inertia(&ws, &lmp);

        let body_jacobian =
            MatSlice::dense(mb_jac_base + (k as usize) * 6 * (ndofs as usize), 6, ndofs);

        // Gyroscopic derivative: aug_I = I + ([ω]_× · I − [Iω]_×) · dt.
        let angvel = ws.rb_vels.angular;
        let w_skew = skew(angvel);
        let i_omega = rb_inertia * angvel;
        let i_omega_skew = skew(i_omega);
        let w_skew_i = w_skew * rb_inertia;
        let gyro_mat = w_skew - i_omega_skew;
        let augmented_inertia = rb_inertia + gyro_mat * dt;

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
        let rb_j_w = body_jacobian.fixed_rows(3, 3);
        let coriolis_v_i =
            MatSlice::dense(mb_cor_base + (k as usize) * 3 * (ndofs as usize), 3, ndofs);
        let coriolis_w_i = coriolis_v_i; // views are structurally identical in the two buffers.

        if k != 0 {
            let parent_id = stat.parent_link_id;
            let parent_link = ws_slice.read(parent_id as usize);
            let parent_j = MatSlice::dense(
                mb_jac_base + (parent_id as usize) * 6 * (ndofs as usize),
                6,
                ndofs,
            );
            let parent_j_w = parent_j.fixed_rows(3, 3);
            let parent_coriolis_v = MatSlice::dense(
                mb_cor_base + (parent_id as usize) * 3 * (ndofs as usize),
                3,
                ndofs,
            );
            let parent_coriolis_w = parent_coriolis_v;
            let parent_w = skew(parent_link.rb_vels.angular);

            // coriolis_v.copy_from(parent_coriolis_v);
            // coriolis_w.copy_from(parent_coriolis_w);
            copy_from(coriolis_v, coriolis_v_i, parent_coriolis_v);
            copy_from(coriolis_w, coriolis_w_i, parent_coriolis_w);

            // coriolis_v += [shift02]^T_× · parent_coriolis_w.
            // (parent_coriolis_w lives in `coriolis_w`, not in `coriolis_v`, hence the
            // cross-buffer variant.)
            let shift_cross_tr_02 = skew_tr(ws.shift02);
            gemm_mat3_cross_buf(
                coriolis_v,
                coriolis_v_i,
                1.0,
                shift_cross_tr_02,
                coriolis_w,
                parent_coriolis_w,
                1.0,
            );

            // coriolis_v += dvel_cross^T · parent_j_w, with
            //   dvel = rb.vels.angvel × shift02 + 2 · joint_velocity.linvel.
            let dvel = ws.rb_vels.angular.cross(ws.shift02)
                + ws.joint_velocity.linear * 2.0;
            let dvel_cross_tr = skew_tr(dvel);
            gemm_mat3_cross_buf(
                coriolis_v,
                coriolis_v_i,
                1.0,
                dvel_cross_tr,
                body_jacobians,
                parent_j_w,
                1.0,
            );

            // coriolis_v += [joint_vel_lin]^T_× · parent_j_w.
            let jv_lin_cross_tr = skew_tr(ws.joint_velocity.linear);
            gemm_mat3_cross_buf(
                coriolis_v,
                coriolis_v_i,
                1.0,
                jv_lin_cross_tr,
                body_jacobians,
                parent_j_w,
                1.0,
            );

            // coriolis_v += (parent_w · shift02_cross_tr) · parent_j_w.
            let combined = parent_w * shift_cross_tr_02;
            gemm_mat3_cross_buf(
                coriolis_v,
                coriolis_v_i,
                1.0,
                combined,
                body_jacobians,
                parent_j_w,
                1.0,
            );

            // coriolis_w += -[joint_vel_ang]_× · parent_j_w.
            let jv_ang_skew = skew(ws.joint_velocity.angular);
            gemm_mat3_cross_buf(
                coriolis_w,
                coriolis_w_i,
                -1.0,
                jv_ang_skew,
                body_jacobians,
                parent_j_w,
                1.0,
            );

            // Joint jacobian contribution to Coriolis (skipped for kinematic joints).
            if stat.kinematic == 0 {
                let mut tmp = [0.0f32; 36];
                let tmp_view = MatSlice::dense(0, 6, 6);
                let joint_j = tmp_view.columns(0, stat.ndofs);
                joint_jacobian(
                    &stat,
                    parent_link.local_to_world.rotation * stat.data.local_frame_a.rotation,
                    &mut tmp,
                    joint_j,
                );
                // coriolis_v_part += 2 · parent_w · rb_joint_j_v.
                // coriolis_w_part += parent_w · rb_joint_j_w.
                // Both operands are stack slices of `tmp`, so we inline a column-major
                // `gemm_mat3_lhs` variant that reads `src` by index.
                let coriolis_v_part = coriolis_v_i.columns(stat.assembly_id, stat.ndofs);
                let coriolis_w_part = coriolis_w_i.columns(stat.assembly_id, stat.ndofs);
                for c in 0..stat.ndofs {
                    let jv = Vec3::new(
                        tmp[tmp_view.idx(0, c)],
                        tmp[tmp_view.idx(1, c)],
                        tmp[tmp_view.idx(2, c)],
                    );
                    let jw = Vec3::new(
                        tmp[tmp_view.idx(3, c)],
                        tmp[tmp_view.idx(4, c)],
                        tmp[tmp_view.idx(5, c)],
                    );
                    let pv = parent_w * jv;
                    let pw = parent_w * jw;
                    // coriolis_v_part[:, c] += 2.0 * pv
                    let iv0 = coriolis_v_part.idx(0, c);
                    let iv1 = coriolis_v_part.idx(1, c);
                    let iv2 = coriolis_v_part.idx(2, c);
                    coriolis_v.write(iv0, coriolis_v.read(iv0) + 2.0 * pv.x);
                    coriolis_v.write(iv1, coriolis_v.read(iv1) + 2.0 * pv.y);
                    coriolis_v.write(iv2, coriolis_v.read(iv2) + 2.0 * pv.z);
                    // coriolis_w_part[:, c] += pw
                    let iw0 = coriolis_w_part.idx(0, c);
                    let iw1 = coriolis_w_part.idx(1, c);
                    let iw2 = coriolis_w_part.idx(2, c);
                    coriolis_w.write(iw0, coriolis_w.read(iw0) + pw.x);
                    coriolis_w.write(iw1, coriolis_w.read(iw1) + pw.y);
                    coriolis_w.write(iw2, coriolis_w.read(iw2) + pw.z);
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
            let shift_cross_tr_23 = skew_tr(ws.shift23);
            gemm_mat3_cross_buf(
                coriolis_v,
                coriolis_v_i,
                1.0,
                shift_cross_tr_23,
                coriolis_w,
                coriolis_w_i,
                1.0,
            );

            let dvel_cross_tr_23 = skew_tr(ws.rb_vels.angular.cross(ws.shift23));
            gemm_mat3_cross_buf(
                coriolis_v,
                coriolis_v_i,
                1.0,
                dvel_cross_tr_23,
                body_jacobians,
                rb_j_w,
                1.0,
            );

            let combined_self = skew(ws.rb_vels.angular) * shift_cross_tr_23;
            gemm_mat3_cross_buf(
                coriolis_v,
                coriolis_v_i,
                1.0,
                combined_self,
                body_jacobians,
                rb_j_w,
                1.0,
            );
        }

        // Meld Coriolis into the mass matrix via i_coriolis_dt:
        //   i_coriolis_dt_v := dt · mass · coriolis_v
        //   i_coriolis_dt_w := dt · (rb_inertia · coriolis_w)
        //   acc_augmented_mass += Jᵀ · i_coriolis_dt.
        scaled_copy_3xn(
            i_coriolis_dt,
            i_coriolis_dt_v,
            mass * dt,
            coriolis_v,
            coriolis_v_i,
        );
        gemm_mat3_cross_buf(
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

/// `c := beta * c + alpha * A_mat3 * b` where `A` is an inline 3×3 and `b`, `c`
/// live in *different* flat buffers. Same as `gemm_mat3_lhs` but with a second
/// buffer for the right-hand-side view.
#[inline]
fn gemm_mat3_cross_buf(
    buf_c: &mut [f32],
    c: MatSlice,
    alpha: f32,
    a: Mat3,
    buf_b: &[f32],
    b: MatSlice,
    beta: f32,
) {
    for j in 0..c.cols {
        let i0 = c.idx(0, j);
        let i1 = c.idx(1, j);
        let i2 = c.idx(2, j);
        let b = Vec3::new(
           buf_b.read(b.idx(0, j)),
           buf_b.read(b.idx(1, j)),
           buf_b.read(b.idx(2, j)),
        );
        let c = Vec3::new(
            buf_c.read(i0),
            buf_c.read(i1),
            buf_c.read(i2),
        );
        let abc = beta * c + alpha * (a * b);
        buf_c.write(i0, abc.x);
        buf_c.write(i1, abc.y);
        buf_c.write(i2, abc.z);
    }
}
