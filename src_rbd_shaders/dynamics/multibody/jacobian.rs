//! Body jacobian assembly.
//!
//! For each link i, the `SPATIAL_DIM × ndofs` body jacobian J_i maps generalized
//! velocities to the link's world-frame spatial velocity at its COM. It is built
//! recursively (rapier's `update_body_jacobians`):
//!
//!   1. J_i := J_parent
//!   2. J_i.linear_rows += [shift02]×ᵀ · J_parent.ang_rows
//!   3. J_i columns for this joint's DOFs += joint jacobian (in world frame)
//!   4. J_i.linear_rows += [shift23]×ᵀ · J_i.ang_rows
//!
//! Storage: column-major. `J[row, col] = jacobians[jac_base + col * SPATIAL_DIM + row]`
//! with `jac_base = mb.jacobian_offset + (k - first_link) * SPATIAL_DIM * ndofs`.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

use parry::math::VectorExt;
use crate::dynamics::joint::SPATIAL_DIM;
use crate::utils::Slice;
use crate::utils::linalg::{MatSlice, copy_from, fill, gemm_skew_tr_lhs};
use crate::{ANG_DIM, DIM, Pose, Rotation, Vector};
#[cfg(feature = "dim3")]
use crate::rotation_to_matrix;

use super::types::{MultibodyInfo, MultibodyLinkStatic, MultibodyLinkWorkspace};

/// Adds this joint's jacobian (world-frame) to the first `ndofs` columns of
/// `view` inside `out`, mirroring rapier's `MultibodyJoint::jacobian`. The
/// non-written entries (zeros in the formal joint jacobian) are skipped, so the
/// effect is a `+=` of the joint jacobian on `view`'s columns.
///
/// Used by `gpu_mb_body_jacobians` to write straight into the body-jacobian
/// buffer — no Function-storage scratch required.
#[inline]
pub(super) fn joint_jacobian_accumulate(
    stat: &MultibodyLinkStatic,
    transform_rot: Rotation,
    out: &mut [f32],
    view: MatSlice,
) {
    let locked = stat.data.locked_axes;
    let mut curr_free_dof = 0u32;

    // Linear DOFs (axis order). Only the linear rows are nonzero.
    for i in 0..DIM {
        if (locked & (1 << i)) == 0 {
            let axis = transform_rot * Vector::ith(i as usize, 1.0);
            #[cfg(feature = "dim3")]
            {
                let i0 = view.idx(0, curr_free_dof);
                let i1 = view.idx(1, curr_free_dof);
                let i2 = view.idx(2, curr_free_dof);
                out.write(i0, out.read(i0) + axis.x);
                out.write(i1, out.read(i1) + axis.y);
                out.write(i2, out.read(i2) + axis.z);
            }
            #[cfg(feature = "dim2")]
            {
                let i0 = view.idx(0, curr_free_dof);
                let i1 = view.idx(1, curr_free_dof);
                out.write(i0, out.read(i0) + axis.x);
                out.write(i1, out.read(i1) + axis.y);
            }
            curr_free_dof += 1;
        }
    }

    // Angular DOFs.
    let ang_locked = (locked >> DIM) & ((1 << ANG_DIM) - 1);
    let num_ang = ANG_DIM - ang_locked.count_ones();
    if num_ang == 1 {
        #[cfg(feature = "dim3")]
        {
            let dof_id = (!ang_locked & 0x7).trailing_zeros();
            let axis = transform_rot * Vector::ith(dof_id as usize, 1.0);
            let i3 = view.idx(3, curr_free_dof);
            let i4 = view.idx(4, curr_free_dof);
            let i5 = view.idx(5, curr_free_dof);
            out.write(i3, out.read(i3) + axis.x);
            out.write(i4, out.read(i4) + axis.y);
            out.write(i5, out.read(i5) + axis.z);
        }
        #[cfg(feature = "dim2")]
        {
            let i2 = view.idx(2, curr_free_dof);
            out.write(i2, out.read(i2) + 1.0);
        }
    } else if num_ang == 3 {
        #[cfg(feature = "dim3")]
        {
            let rotmat = rotation_to_matrix(transform_rot);
            for k in 0..3u32 {
                let axis = if k == 0 {
                    rotmat.x_axis
                } else if k == 1 {
                    rotmat.y_axis
                } else {
                    rotmat.z_axis
                };
                let i3 = view.idx(3, curr_free_dof + k);
                let i4 = view.idx(4, curr_free_dof + k);
                let i5 = view.idx(5, curr_free_dof + k);
                out.write(i3, out.read(i3) + axis.x);
                out.write(i4, out.read(i4) + axis.y);
                out.write(i5, out.read(i5) + axis.z);
            }
        }
        #[cfg(feature = "dim2")]
        {
            let _ = curr_free_dof;
        }
    } // TODO: num_ang == 2
}

/// Writes this joint's jacobian (world-frame) into the first `ndofs` columns of
/// an inline `SPATIAL_DIM × SPATIAL_DIM` scratch `out`, mirroring rapier's
/// `MultibodyJoint::jacobian`. Used by the Coriolis assembly which needs to
/// re-read the joint jacobian after building it.
///
/// `transform_rot` maps body-local axes (of the parent's `local_frame_a`) to world.
#[inline]
pub(super) fn joint_jacobian(
    stat: &MultibodyLinkStatic,
    transform_rot: Rotation,
    out: &mut [f32; SPATIAL_DIM * SPATIAL_DIM],
    view: MatSlice,
) {
    let locked = stat.data.locked_axes;
    let mut curr_free_dof = 0u32;

    // Linear DOFs (axis order).
    for i in 0..DIM {
        if (locked & (1 << i)) == 0 {
            let axis = transform_rot * Vector::ith(i as usize, 1.0);
            // Linear rows.
            #[cfg(feature = "dim3")]
            {
                out[view.idx(0, curr_free_dof)] = axis.x;
                out[view.idx(1, curr_free_dof)] = axis.y;
                out[view.idx(2, curr_free_dof)] = axis.z;
                out[view.idx(3, curr_free_dof)] = 0.0;
                out[view.idx(4, curr_free_dof)] = 0.0;
                out[view.idx(5, curr_free_dof)] = 0.0;
            }
            #[cfg(feature = "dim2")]
            {
                out[view.idx(0, curr_free_dof)] = axis.x;
                out[view.idx(1, curr_free_dof)] = axis.y;
                out[view.idx(2, curr_free_dof)] = 0.0;
            }
            curr_free_dof += 1;
        }
    }

    // Angular DOFs.
    let ang_locked = (locked >> DIM) & ((1 << ANG_DIM) - 1);
    let num_ang = ANG_DIM - ang_locked.count_ones();
    if num_ang == 1 {
        // Linear rows are zero; angular rows hold the rotated axis.
        #[cfg(feature = "dim3")]
        {
            let dof_id = (!ang_locked & 0x7).trailing_zeros();
            let axis = transform_rot * Vector::ith(dof_id as usize, 1.0);
            out[view.idx(0, curr_free_dof)] = 0.0;
            out[view.idx(1, curr_free_dof)] = 0.0;
            out[view.idx(2, curr_free_dof)] = 0.0;
            out[view.idx(3, curr_free_dof)] = axis.x;
            out[view.idx(4, curr_free_dof)] = axis.y;
            out[view.idx(5, curr_free_dof)] = axis.z;
        }
        #[cfg(feature = "dim2")]
        {
            // The 2D angular axis is "Z"; the joint jacobian's angular row
            // is just `1`. Linear rows are zero.
            out[view.idx(0, curr_free_dof)] = 0.0;
            out[view.idx(1, curr_free_dof)] = 0.0;
            out[view.idx(2, curr_free_dof)] = 1.0;
        }
    } else if num_ang == 3 {
        #[cfg(feature = "dim3")]
        {
            // Three free angular DOFs: angular rows = world rotation matrix.
            let rotmat = rotation_to_matrix(transform_rot);
            for k in 0..3u32 {
                let axis = if k == 0 {
                    rotmat.x_axis
                } else if k == 1 {
                    rotmat.y_axis
                } else {
                    rotmat.z_axis
                };
                out[view.idx(0, curr_free_dof + k)] = 0.0;
                out[view.idx(1, curr_free_dof + k)] = 0.0;
                out[view.idx(2, curr_free_dof + k)] = 0.0;
                out[view.idx(3, curr_free_dof + k)] = axis.x;
                out[view.idx(4, curr_free_dof + k)] = axis.y;
                out[view.idx(5, curr_free_dof + k)] = axis.z;
            }
        }
        #[cfg(feature = "dim2")]
        {
            // Unreachable in 2D (ANG_DIM = 1).
            let _ = curr_free_dof;
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

    for k in 0..num_links {
        let link_infos = stat_slice.at(k as usize);
        let link = ws_slice.at(k as usize);

        // View for this link's body jacobian (SPATIAL_DIM × ndofs, dense).
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

            // link_j := parent_j
            copy_from(body_jacobians, link_j, parent_j);

            // link_j_v += [shift02]^T_× · parent_j_w
            let link_j_v = link_j.fixed_rows(0, DIM);
            let parent_j_w = parent_j.fixed_rows(DIM, ANG_DIM);
            gemm_skew_tr_lhs(body_jacobians, link_j_v, 1.0, link.shift02, parent_j_w, 1.0);
        } else {
            fill(body_jacobians, link_j, 0.0);
            parent_to_world = Pose::default();
        }

        // Add the joint jacobian directly into link_j's columns
        // `[assembly_id .. assembly_id + ndofs_link]`. The accumulating variant
        // skips the formal-zero rows (linear-only or angular-only block), so we
        // avoid both the stack scratch and the redundant zero-add traffic.
        let link_j_part = link_j.columns(link_infos.assembly_id, link_infos.ndofs);
        joint_jacobian_accumulate(
            link_infos,
            parent_to_world.rotation * link_infos.data.local_frame_a.rotation,
            body_jacobians,
            link_j_part,
        );

        // link_j_v += [shift23]^T_× · link_j_w  (self-shift).
        let (link_j_v, link_j_w) = link_j.rows_range_pair(0, DIM, DIM, ANG_DIM);
        gemm_skew_tr_lhs(body_jacobians, link_j_v, 1.0, link.shift23, link_j_w, 1.0);
    }
}
