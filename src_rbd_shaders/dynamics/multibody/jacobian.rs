//! Body jacobian assembly.
//!
//! For each link i, the 6×ndofs body jacobian J_i maps generalized velocities
//! to the link's world-frame spatial velocity at its COM. It is built recursively:
//!
//!   1. J_i := J_parent
//!   2. J_i.linear_rows += [shift02]×ᵀ · J_parent.ang_rows
//!   3. J_i columns for this joint's DOFs += joint jacobian (in world frame)
//!   4. J_i.linear_rows += [shift23]×ᵀ · J_i.ang_rows
//!
//! Storage: column-major. `J[row, col] = jacobians[jac_base + col * 6 + row]`
//! with `jac_base = mb.jacobian_offset + (k - first_link) * 6 * ndofs`.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

use glamx::Quat;

use crate::Pose;
use crate::utils::Slice;
use crate::utils::linalg::{MatSlice, copy_from, fill, gemm_mat3_lhs, skew_tr};

use super::types::{MultibodyInfo, MultibodyLinkStatic, MultibodyLinkWorkspace};
use super::utils::basis_vec3;

/// Writes this joint's jacobian (world-frame) into the first `ndofs` columns of
/// an inline 6×6 scratch `out`, mirroring rapier's `MultibodyJoint::jacobian`.
///
/// `transform_rot` maps body-local axes (of the parent's `local_frame_a`) to world.
#[inline]
pub(super) fn joint_jacobian(
    stat: &MultibodyLinkStatic,
    transform_rot: Quat,
    out: &mut [f32; 36],
    view: MatSlice,
) {
    let locked = stat.data.locked_axes;
    let mut curr_free_dof = 0u32;

    // Linear DOFs (axis order).
    for i in 0u32..3 {
        if (locked & (1 << i)) == 0 {
            let axis = transform_rot * basis_vec3(i);
            out[view.idx(0, curr_free_dof)] = axis.x;
            out[view.idx(1, curr_free_dof)] = axis.y;
            out[view.idx(2, curr_free_dof)] = axis.z;
            out[view.idx(3, curr_free_dof)] = 0.0;
            out[view.idx(4, curr_free_dof)] = 0.0;
            out[view.idx(5, curr_free_dof)] = 0.0;
            curr_free_dof += 1;
        }
    }

    // Angular DOFs.
    let ang_locked = (locked >> 3) & 0x7;
    let num_ang = 3 - ang_locked.count_ones();
    if num_ang == 1 {
        let dof_id = (!ang_locked & 0x7).trailing_zeros();
        let axis = transform_rot * basis_vec3(dof_id);
        out[view.idx(0, curr_free_dof)] = 0.0;
        out[view.idx(1, curr_free_dof)] = 0.0;
        out[view.idx(2, curr_free_dof)] = 0.0;
        out[view.idx(3, curr_free_dof)] = axis.x;
        out[view.idx(4, curr_free_dof)] = axis.y;
        out[view.idx(5, curr_free_dof)] = axis.z;
    } else if num_ang == 3 {
        for k in 0u32..3 {
            let axis = transform_rot * basis_vec3(k);
            out[view.idx(0, curr_free_dof + k)] = 0.0;
            out[view.idx(1, curr_free_dof + k)] = 0.0;
            out[view.idx(2, curr_free_dof + k)] = 0.0;
            out[view.idx(3, curr_free_dof + k)] = axis.x;
            out[view.idx(4, curr_free_dof + k)] = axis.y;
            out[view.idx(5, curr_free_dof + k)] = axis.z;
        }
    } // TODO: num_ang == 2
}

/// Build per-link body jacobians. Mirrors rapier's `Multibody::update_body_jacobians`
/// nearly line-for-line, using `MatSlice` views + BLAS-style primitives in place of
/// nalgebra's matrix API.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_body_jacobians(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] links_workspace: &[MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] body_jacobians: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] links_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 7)] jacobians_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let links_start = batch_id * *links_batch_capacity as usize;
    let jac_start = batch_id * *jacobians_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let num_links = mb.num_links;
    let ndofs = mb.ndofs;
    let first_link_global = links_start + mb.first_link as usize;
    let mb_jac_base = jac_start + mb.jacobian_offset as usize;

    let stat_slice = Slice(links_static, first_link_global);
    let ws_slice = Slice(links_workspace, first_link_global);

    // Zero all per-link jacobian blocks of this multibody.
    let mb_block = MatSlice::dense(mb_jac_base, 6 * num_links, ndofs);

    for k in 0..num_links {
        let link_infos = stat_slice.at(k as usize);
        let link = ws_slice.at(k as usize);

        // View for this link's body jacobian (6 × ndofs, dense).
        let link_j = MatSlice::dense(mb_jac_base + (k as usize) * 6 * (ndofs as usize), 6, ndofs);

        let parent_to_world;
        if k != 0 {
            let parent_j = MatSlice::dense(
                mb_jac_base + (link_infos.parent_link_id as usize) * 6 * (ndofs as usize),
                6,
                ndofs,
            );
            let parent_link = ws_slice.at(link_infos.parent_link_id as usize);
            parent_to_world = parent_link.local_to_world;

            // link_j := parent_j
            copy_from(body_jacobians, link_j, parent_j);

            // link_j_v += [shift02]^T_× · parent_j_w
            let link_j_v = link_j.fixed_rows(0, 3);
            let parent_j_w = parent_j.fixed_rows(3, 3);
            let shift_tr = skew_tr(link.shift02);
            gemm_mat3_lhs(body_jacobians, link_j_v, 1.0, shift_tr, parent_j_w, 1.0);
        } else {
            fill(body_jacobians, link_j, 0.0);
            parent_to_world = Pose::default();
        }

        // Fill the joint jacobian into a 6×6 stack scratch, then splat its first
        // `ndofs_link` columns into link_j's `[assembly_id .. assembly_id + ndofs_link]`.
        // TODO(PERF): double-check the generated shader to verify this array
        //             doesn’t get copied over and over at each read/mutation.
        let mut tmp = [0.0f32; 36];
        let tmp_view = MatSlice::dense(0, 6, 6);
        let joint_j = tmp_view.columns(0, link_infos.ndofs);
        joint_jacobian(
            link_infos,
            parent_to_world.rotation * link_infos.data.local_frame_a.rotation,
            &mut tmp,
            joint_j,
        );
        // link_j_part += joint_j  (axpy with a stack-allocated RHS; rust-gpu can't
        // coerce `&[f32; 36]` to `&[f32]`, so this is expanded inline here).
        let link_j_part = link_j.columns(link_infos.assembly_id, link_infos.ndofs);
        for c in 0..link_infos.ndofs {
            for r in 0u32..6 {
                let idx = link_j_part.idx(r, c);
                let cur = body_jacobians.read(idx);
                body_jacobians.write(idx, cur + tmp[joint_j.idx(r, c)]);
            }
        }

        // link_j_v += [shift23]^T_× · link_j_w  (self-shift).
        let (link_j_v, link_j_w) = link_j.rows_range_pair(0, 3, 3, 3);
        let shift_tr = skew_tr(link.shift23);
        gemm_mat3_lhs(body_jacobians, link_j_v, 1.0, shift_tr, link_j_w, 1.0);
    }
}
