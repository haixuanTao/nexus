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
//! Storage: column-major. In 3D the layout is CHAIN-SPARSE: link `k` stores a
//! `SPATIAL_DIM × popcount(jac_chain_mask)` block at
//! `mb.jacobian_offset + stat.jac_offset`, whose columns are the set bits of
//! `stat.jac_chain_mask` in ascending DOF order (every other formal column is
//! exactly zero — branch-induced sparsity; the parent's block is a strict
//! prefix of the child's). 2D keeps the dense
//! `jac_base = mb.jacobian_offset + (k - first_link) * SPATIAL_DIM * ndofs`.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

use crate::dynamics::joint::SPATIAL_DIM;
#[cfg(feature = "dim3")]
use crate::rotation_to_matrix;
use crate::utils::Slice;
use crate::utils::linalg::{MAX_MB_DOFS, MatSlice, copy_from_par, fill_par, gemm_skew_tr_lhs_par};
use crate::{ANG_DIM, DIM, Pose, Rotation, Vector};
use parry::math::VectorExt;

use super::types::{MultibodyInfo, MultibodyLinkStatic, MultibodyLinkWorkspace};

/// Workgroup width for the parallelised body-jacobian kernel.
const LANES: u32 = 32;

/// Adds this joint's jacobian (world-frame) to the first `ndofs` columns of
/// `view` inside `out`, mirroring rapier's `MultibodyJoint::jacobian`. The
/// non-written entries (zeros in the formal joint jacobian) are skipped, so the
/// effect is a `+=` of the joint jacobian on `view`'s columns.
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
