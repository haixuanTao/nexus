//! LU decomposition + solve.
//!
//! Split into two kernels so the factorization can be reused across multiple
//! right-hand sides within a frame (e.g. gravity τ, contact impulses, …) —
//! mirrors nalgebra's `LU` / `LU::solve_mut` API.
//!
//! The augmented mass matrix from CRBA is symmetric positive definite, so
//! pivoting is not strictly needed, but partial pivoting is still performed
//! for robustness and parity with rapier.
//!
//! For simplicity the present implementation assumes no kinematic DOFs; all
//! DOFs participate in the solve. Rapier excludes kinematic DOFs via a
//! permutation — a follow-up can layer that on top of these primitives.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

use crate::utils::linalg::{MatSlice, lu_decompose, lu_solve_in_place};

use super::types::MultibodyInfo;

/// Factor `M` in-place into `P·L·U` and record the row pivots.
///
/// Input/output: `mass_matrices` holds the per-multibody mass matrix block on
/// entry and the packed LU factors on exit. `lu_pivots` receives one pivot index
/// per row per multibody. One workgroup per multibody.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_lu_decompose(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mass_matrices: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] lu_pivots: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 5)] mass_matrix_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] dof_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let mm_start = batch_id * *mass_matrix_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    if mb.ndofs == 0 {
        return;
    }
    let m = MatSlice::dense(mm_start + mb.mass_matrix_offset as usize, mb.ndofs, mb.ndofs);
    let piv_offset = dof_start + mb.first_dof as usize;

    lu_decompose(mass_matrices, m, lu_pivots, piv_offset);
}

/// Solve `M · x = rhs` in-place using the packed LU produced by
/// `gpu_mb_lu_decompose`. `rhs` is overwritten with `x`.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_lu_solve(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mass_matrices: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] lu_pivots: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] rhs: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] mass_matrix_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 7)] dof_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let mm_start = batch_id * *mass_matrix_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    if mb.ndofs == 0 {
        return;
    }
    let m = MatSlice::dense(mm_start + mb.mass_matrix_offset as usize, mb.ndofs, mb.ndofs);
    let piv_offset = dof_start + mb.first_dof as usize;
    let rhs_offset = piv_offset;

    lu_solve_in_place(mass_matrices, m, lu_pivots, piv_offset, rhs, rhs_offset);
}
