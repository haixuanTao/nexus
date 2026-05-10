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
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

use parry::math::VectorExt;
use crate::dynamics::joint::SPATIAL_DIM;
use crate::utils::Slice;
use crate::utils::linalg::{MatSlice, copy_from_par, fill_par, gemm_skew_tr_lhs_par, MAX_MB_DOFS};
use crate::{ANG_DIM, DIM, Pose, Rotation, Vector};
#[cfg(feature = "dim3")]
use crate::rotation_to_matrix;

use super::types::{MultibodyInfo, MultibodyLinkStatic, MultibodyLinkWorkspace};

/// Workgroup width for the parallelised body-jacobian kernel.
const LANES: u32 = 32;

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

/// Workgroup-parallel variant of `joint_jacobian_accumulate`.
///
/// Mirrors the sequential algorithm exactly: every lane walks the same
/// iteration counter `curr_free_dof`, but the inner write only fires for the
/// lane whose ID matches `curr_free_dof`. Each free DOF column is written by
/// exactly one lane; lanes outside `[0, total_free)` simply pass through with
/// no writes. This is closer to sequential semantics than a "find the c-th
/// unlocked axis" indirection and avoids subtle rust-gpu lowering bugs.
#[inline]
pub(super) fn joint_jacobian_accumulate_par(
    stat: &MultibodyLinkStatic,
    transform_rot: Rotation,
    out: &mut [f32],
    view: MatSlice,
    lane: u32,
    _lanes: u32,
) {
    let locked = stat.data.locked_axes;
    let mut curr_free_dof = 0u32;

    // Linear DOFs (axis order). Only the linear rows are nonzero.
    for i in 0..DIM {
        if (locked & (1 << i)) == 0 {
            if lane == curr_free_dof {
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
            }
            curr_free_dof += 1;
        }
    }

    // Angular DOFs.
    let ang_locked = (locked >> DIM) & ((1 << ANG_DIM) - 1);
    let num_ang = ANG_DIM - ang_locked.count_ones();
    if num_ang == 1 {
        if lane == curr_free_dof {
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
        }
    } else if num_ang == 3 {
        #[cfg(feature = "dim3")]
        {
            let rotmat = rotation_to_matrix(transform_rot);
            for k in 0..3u32 {
                if lane == curr_free_dof + k {
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
        }
    } // TODO: num_ang == 2
}

/// Computes column `c` (`0 ≤ c < stat.ndofs`) of this joint's body jacobian as
/// the pair `(linear_part, angular_part)`, in registers. Avoids the `[f32; 36]`
/// Function-storage scratch the Coriolis assembly used to need.
///
/// In 3D, returns `(Vec3, Vec3)`. In 2D, returns `(Vec2, f32)`.
///
/// Mirrors the sequential `joint_jacobian` walk: every call walks the same
/// iteration counter `curr_free_dof` and only assigns the result locals when
/// `curr_free_dof == c`. Single-exit form (no early `return` inside the
/// loop) — rust-gpu's structured control flow lowering can produce silently
/// incorrect results for `return` statements in the middle of a `for` loop.
#[cfg(feature = "dim3")]
#[inline]
pub(super) fn joint_jacobian_column(
    stat: &MultibodyLinkStatic,
    transform_rot: Rotation,
    c: u32,
) -> (Vector, glamx::Vec3) {
    let locked = stat.data.locked_axes;
    let mut curr_free_dof = 0u32;
    let mut result_lin = Vector::ZERO;
    let mut result_ang = glamx::Vec3::ZERO;

    // Linear DOFs in axis order.
    for i in 0..DIM {
        if (locked & (1 << i)) == 0 {
            if curr_free_dof == c {
                result_lin = transform_rot * Vector::ith(i as usize, 1.0);
            }
            curr_free_dof += 1;
        }
    }

    // Angular DOFs.
    let ang_locked = (locked >> DIM) & ((1 << ANG_DIM) - 1);
    let num_ang = ANG_DIM - ang_locked.count_ones();
    if num_ang == 1 {
        if curr_free_dof == c {
            let dof_id = (!ang_locked & 0x7).trailing_zeros();
            result_ang = transform_rot * Vector::ith(dof_id as usize, 1.0);
        }
    } else if num_ang == 3 {
        let rotmat = rotation_to_matrix(transform_rot);
        for k in 0..3u32 {
            if curr_free_dof + k == c {
                result_ang = if k == 0 {
                    rotmat.x_axis
                } else if k == 1 {
                    rotmat.y_axis
                } else {
                    rotmat.z_axis
                };
            }
        }
    }

    (result_lin, result_ang)
}

#[cfg(feature = "dim2")]
#[inline]
pub(super) fn joint_jacobian_column(
    stat: &MultibodyLinkStatic,
    transform_rot: Rotation,
    c: u32,
) -> (Vector, f32) {
    let locked = stat.data.locked_axes;
    let mut curr_free_dof = 0u32;
    let mut result_lin = Vector::ZERO;
    let mut result_ang = 0.0f32;

    for i in 0..DIM {
        if (locked & (1 << i)) == 0 {
            if curr_free_dof == c {
                result_lin = transform_rot * Vector::ith(i as usize, 1.0);
            }
            curr_free_dof += 1;
        }
    }
    let ang_locked = (locked >> DIM) & ((1 << ANG_DIM) - 1);
    let num_ang = ANG_DIM - ang_locked.count_ones();
    if num_ang == 1 && curr_free_dof == c {
        result_ang = 1.0;
    }
    (result_lin, result_ang)
}

/// Writes this joint's jacobian (world-frame) into the first `ndofs` columns of
/// an inline `SPATIAL_DIM × SPATIAL_DIM` scratch `out`, mirroring rapier's
/// `MultibodyJoint::jacobian`. Kept for callers that need the full table; the
/// Coriolis path now uses `joint_jacobian_column` instead.
///
/// `transform_rot` maps body-local axes (of the parent's `local_frame_a`) to world.
#[allow(dead_code)]
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

/// Build per-link body jacobians.
///
/// One workgroup of `LANES = 32` threads per `(multibody, batch)` pair. The
/// link loop is sequential (parent's jacobian must be written before its
/// child reads it), but each link's per-column work is partitioned across
/// lanes. Workgroup barriers at the end of each iteration ensure the parent's
/// jacobian writes are visible before children read them.
///
/// The `_cpu_marker` is a dummy `#[spirv(workgroup)]` parameter that makes
/// the khal CPU bindgen treat this as a shared-memory kernel and dispatch
/// each workgroup's lanes through the corosensei coroutine pool — only that
/// path implements proper barrier semantics on CPU. Without it the CPU
/// backend runs lanes sequentially (full kernel per lane, barriers as
/// no-ops), and lane 1's `copy_from_par` overwrites the joint contribution
/// that lane 0 already wrote.
#[spirv_bindgen]
#[spirv(compute(threads(32, 1, 1)))]
pub fn gpu_mb_body_jacobians(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] links_workspace: &[MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] body_jacobians: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] links_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 7)] jacobians_batch_capacity: &u32,
    #[spirv(workgroup)] _cpu_marker: &mut u32,
) {
    let batch_id = wg_id.y as usize;
    let mb_idx = wg_id.x;
    let lane = lid.x;
    // Padding multibody slots have `num_links == 0` so the per-link loop
    // below iterates zero times. We deliberately DO NOT early-return on
    // out-of-range `mb_idx`: WGSL's naga frontend can't prove that a
    // storage-loaded comparison is uniform across the workgroup, and any
    // subsequent `workgroupBarrier()` would then be flagged as "called from
    // non-uniform control flow". Letting all workgroups run keeps barriers
    // in uniform top-level control flow.
    let _ = num_multibodies;

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

    // NOTE: fixed number of iterations for uniform control flow.
    // TODO(PERF): on non-web platforms we could just use `mb.num_links` as the upper bound.
    for k in 0..MAX_MB_DOFS as u32 {
        let mut parent_to_world= Pose::default();
        // View for this link's body jacobian (SPATIAL_DIM × ndofs, dense).
        let link_j = MatSlice::dense(
            mb_jac_base + (k as usize) * SPATIAL_DIM * (ndofs as usize),
            SPATIAL_DIM as u32,
            ndofs,
        );

        if k < mb.num_links {
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

                // link_j := parent_j (parallel column-wise).
                copy_from_par(body_jacobians, link_j, parent_j, lane, LANES);
                // link_j_v += [shift02]^T_× · parent_j_w (parallel column-wise).
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
            }
        }

        // Barrier: the joint-jacobian accumulation below has lane c=0 (and
        // c=1, c=2, …) read+write `link_j[*, assembly_id+c]`, but those
        // columns were written by lane `assembly_id+c` in `copy_from_par` /
        // `gemm_skew_tr_lhs_par`. Without this barrier, lane 0 races against
        // lane `assembly_id` and may read stale data.
        workgroup_memory_barrier_with_group_sync();

        if k < mb.num_links {
            // Add the joint jacobian directly into link_j's columns
            // `[assembly_id .. assembly_id + ndofs_link]`. Each lane handles a
            // subset of `0..ndofs_link`, so writes are non-overlapping.
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

        // link_j_v += [shift23]^T_× · link_j_w  (self-shift, parallel
        // column-wise). Reads link_j_w which was just populated above —
        // barrier needed first.
        workgroup_memory_barrier_with_group_sync();

        if k < mb.num_links {
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

        // End-of-iteration barrier so children see the completed link_j.
        workgroup_memory_barrier_with_group_sync();
    }

}
