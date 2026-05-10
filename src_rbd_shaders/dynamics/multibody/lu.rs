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
//! Workgroup-parallel: one workgroup of `LU_LANES = 32` threads cooperates
//! per `(multibody, batch)` pair, holding the matrix in shared memory and
//! partitioning each pivot step's row-swap / column-scale / trailing-update
//! across lanes (Genesis-style tiled LU).

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

use crate::utils::linalg::{MAX_MB_DOFS, MatSlice};

use super::types::MultibodyInfo;

/// Workgroup width for the parallelised LU kernels. Must match the
/// `threads(N, 1, 1)` attribute and `MB_LU_LANES` on the host side.
pub(super) const LANES: u32 = 32;

/// Side length of the workgroup-shared matrix tile. Must equal both the lane
/// count and the maximum supported `ndofs`.
const MAX_DOFS_U32: u32 = MAX_MB_DOFS as u32;

/// Index helper for the shared `mat[col * MAX_MB_DOFS + row]` tile.
#[inline]
pub(super) fn sm_idx(r: u32, c: u32) -> usize {
    (c * MAX_DOFS_U32 + r) as usize
}

/// Workgroup-parallel LU factorization on the shared `mat` tile in place.
///
/// Each pivot step `k`:
///   - lane 0 finds the pivot row (sequential argmax over rows `k..n`),
///   - lanes 0..n participate in the row swap (each owns one column),
///   - lane 0 broadcasts `1/akk` via shared memory,
///   - each lane below the pivot scales its row entry,
///   - each lane handling a trailing column updates that whole column.
#[inline]
pub(super) fn lu_factor_in_shared(
    n: u32,
    lane: u32,
    mat: &mut [f32; (MAX_MB_DOFS * MAX_MB_DOFS) as usize],
    pivots_dst: &mut [u32],
    pivots_offset: usize,
    pivot_row_shared: &mut u32,
    inv_akk_shared: &mut f32,
) {
    // NOTE: fixed number of iterations for uniform control flow.
    // TODO(PERF): on non-web platforms we could just use `n` as the upper bound.
    for k in 0..MAX_DOFS_U32 {
        let active = k < n;
        if active && lane == 0 {
            let mut pivot_row = k;
            let mut pivot_val = {
                let v = mat.read(sm_idx(k, k));
                if v >= 0.0 { v } else { -v }
            };
            for i in (k + 1)..n {
                let v = mat.read(sm_idx(i, k));
                let av = if v >= 0.0 { v } else { -v };
                if av > pivot_val {
                    pivot_val = av;
                    pivot_row = i;
                }
            }
            *pivot_row_shared = pivot_row;
            pivots_dst.write(pivots_offset + k as usize, pivot_row);
        }
        workgroup_memory_barrier_with_group_sync();
        let pivot_row = *pivot_row_shared;

        if active && pivot_row != k && lane < n {
            let c = lane;
            let a = mat.read(sm_idx(k, c));
            let b = mat.read(sm_idx(pivot_row, c));
            mat.write(sm_idx(k, c), b);
            mat.write(sm_idx(pivot_row, c), a);
        }
        workgroup_memory_barrier_with_group_sync();

        if active && lane == 0 {
            let akk = mat.read(sm_idx(k, k));
            *inv_akk_shared = if akk != 0.0 { 1.0 / akk } else { 0.0 };
        }
        workgroup_memory_barrier_with_group_sync();
        let inv_akk = *inv_akk_shared;

        if active {
            let r = k + 1 + lane;
            if r < n {
                let v = mat.read(sm_idx(r, k)) * inv_akk;
                mat.write(sm_idx(r, k), v);
            }
        }
        workgroup_memory_barrier_with_group_sync();

        if active {
            let j = k + 1 + lane;
            if j < n {
                let akj = mat.read(sm_idx(k, j));
                for r in (k + 1)..n {
                    let lik = mat.read(sm_idx(r, k));
                    let v = mat.read(sm_idx(r, j)) - lik * akj;
                    mat.write(sm_idx(r, j), v);
                }
            }
        }
        workgroup_memory_barrier_with_group_sync();
    }
}

/// Factor `M` in-place into `P·L·U` and record the row pivots.
///
/// One workgroup of `LANES` threads per `(multibody, batch)` pair. Lane `t`
/// owns column `t` of the working matrix in shared memory.
#[spirv_bindgen]
#[spirv(compute(threads(32, 1, 1)))]
pub fn gpu_mb_lu_decompose(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mass_matrices: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] lu_pivots: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 5)] mass_matrix_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] dof_batch_capacity: &u32,
    // 32 × 32 column-major scratch (4 KiB).
    #[spirv(workgroup)] mat: &mut [f32; (MAX_MB_DOFS * MAX_MB_DOFS) as usize],
    // Lane-0 → all-lanes broadcast slots for the per-step pivot row + 1/akk.
    #[spirv(workgroup)] pivot_row_shared: &mut u32,
    #[spirv(workgroup)] inv_akk_shared: &mut f32,
) {
    let batch_id = wg_id.y as usize;
    let mb_idx = wg_id.x;
    let lane = lid.x;
    // Padding multibody slots have `ndofs == 0`, so the per-link loops below
    // naturally iterate zero times. We deliberately DO NOT early-return for
    // out-of-range `mb_idx` or zero-DOF multibodies: WGSL's naga frontend
    // can't prove that a storage-loaded comparison is uniform across the
    // workgroup, and any subsequent `workgroupBarrier()` would then be
    // flagged as "called from non-uniform control flow". Keeping the kernel
    // body branch-free at the top level means barriers stay uniform; dummy
    // workgroups do a few empty loops and exit cleanly.
    let _ = num_multibodies;

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let mm_start = batch_id * *mass_matrix_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let n = mb.ndofs;
    let m = MatSlice::dense(mm_start + mb.mass_matrix_offset as usize, n, n);
    let piv_offset = dof_start + mb.first_dof as usize;

    // Load the matrix from global → shared memory. Each lane owns one column.
    if lane < n {
        for r in 0..n {
            mat.write(sm_idx(r, lane), mass_matrices.read(m.idx(r, lane)));
        }
    }
    workgroup_memory_barrier_with_group_sync();

    lu_factor_in_shared(
        n,
        lane,
        mat,
        lu_pivots,
        piv_offset,
        pivot_row_shared,
        inv_akk_shared,
    );

    // Write the LU factors back to global memory.
    if lane < n {
        for r in 0..n {
            mass_matrices.write(m.idx(r, lane), mat.read(sm_idx(r, lane)));
        }
    }
}

/// Inner of the workgroup-parallel triangular solves used by
/// [`gpu_mb_lu_solve`] and [`gpu_mb_lu_factor_and_solve`]. Operates entirely on
/// shared memory: `mat` already holds the LU factors and `x` already holds the
/// permuted rhs. Each row's `Σ_{j} M[i, j] · x[j]` is parallelised across
/// lanes via a tree reduction in shared memory — the inherently sequential
/// `i` dependency keeps this O(n · log lanes) but every lane stays busy.
#[inline]
pub(super) fn lu_triangular_solve_in_place(
    n: u32,
    lane: u32,
    mat: &[f32; (MAX_MB_DOFS * MAX_MB_DOFS) as usize],
    x: &mut [f32; MAX_MB_DOFS],
    partial: &mut [f32; LANES as usize],
) {
    // NOTE: fixed number of iterations for uniform control flow.
    // TODO(PERF): on non-web platforms we could just use `n` as the upper bound.
    for i in 0..MAX_DOFS_U32 {
        let active = i < n;
        let s = if active && lane < i {
            mat.read(sm_idx(i, lane)) * x.read(lane as usize)
        } else {
            0.0f32
        };
        partial.write(lane as usize, s);
        workgroup_memory_barrier_with_group_sync();
        for step in 0..5u32 {
            let stride = 1u32 << (4 - step);
            if lane < stride {
                let v = partial.read(lane as usize) + partial.read((lane + stride) as usize);
                partial.write(lane as usize, v);
            }
            workgroup_memory_barrier_with_group_sync();
        }
        if active && lane == 0 {
            let cur = x.read(i as usize);
            x.write(i as usize, cur - partial.read(0));
        }
        workgroup_memory_barrier_with_group_sync();
    }

    // NOTE: fixed number of iterations for uniform control flow.
    // TODO(PERF): on non-web platforms we could just use `n` as the upper bound.
    for step in 0..MAX_DOFS_U32 {
        let active = step < n;
        // For dummy iterations (step >= n), `i` is not meaningful — guard
        // every use of it behind `active`.
        let i = if active { n - 1 - step } else { 0 };
        let s = if active && lane > i && lane < n {
            mat.read(sm_idx(i, lane)) * x.read(lane as usize)
        } else {
            0.0f32
        };
        partial.write(lane as usize, s);
        workgroup_memory_barrier_with_group_sync();
        for r in 0..5u32 {
            let stride = 1u32 << (4 - r);
            if lane < stride {
                let v = partial.read(lane as usize) + partial.read((lane + stride) as usize);
                partial.write(lane as usize, v);
            }
            workgroup_memory_barrier_with_group_sync();
        }
        if active && lane == 0 {
            let u = mat.read(sm_idx(i, i));
            let cur = x.read(i as usize) - partial.read(0);
            x.write(i as usize, if u != 0.0 { cur / u } else { 0.0 });
        }
        workgroup_memory_barrier_with_group_sync();
    }
}

/// Apply the recorded pivots (sequential — lane 0 only). `n` is small so the
/// extra parallelism wouldn't pay for the barrier.
#[inline]
pub(super) fn lu_apply_pivots(
    n: u32,
    lane: u32,
    buf_pivots: &[u32],
    pivots_offset: usize,
    x: &mut [f32; MAX_MB_DOFS],
) {
    if lane == 0 {
        for k in 0..n {
            let p = buf_pivots.read(pivots_offset + k as usize);
            if p != k {
                let a = x.read(k as usize);
                let b = x.read(p as usize);
                x.write(k as usize, b);
                x.write(p as usize, a);
            }
        }
    }
    workgroup_memory_barrier_with_group_sync();
}

/// Solve `M · x = rhs` in-place using the packed LU produced by
/// `gpu_mb_lu_decompose`. `rhs` is overwritten with `x`.
///
/// Workgroup-parallel: matrix in shared memory, RHS in a length-`MAX_MB_DOFS`
/// shared vector. Permutation is sequential (lane 0). Forward + backward
/// substitution have an inherently sequential `i` dependency, but the inner
/// dot product `Σ U[i,j] · x[j]` is parallelised across lanes via a tree
/// reduction in shared memory (small `n` makes a straight 32→1 reduction
/// cheap).
#[spirv_bindgen]
#[spirv(compute(threads(32, 1, 1)))]
pub fn gpu_mb_lu_solve(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mass_matrices: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] lu_pivots: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] rhs: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] mass_matrix_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 7)] dof_batch_capacity: &u32,
    // Working matrix tile in shared memory.
    #[spirv(workgroup)] mat: &mut [f32; (MAX_MB_DOFS * MAX_MB_DOFS) as usize],
    // RHS / solution vector in shared memory (length `MAX_MB_DOFS`).
    #[spirv(workgroup)] x: &mut [f32; MAX_MB_DOFS],
    // Per-lane partial-sum scratch for the tree reduction inside the
    // triangular solves.
    #[spirv(workgroup)] partial: &mut [f32; LANES as usize],
) {
    let batch_id = wg_id.y as usize;
    let mb_idx = wg_id.x;
    let lane = lid.x;
    // Padding multibody slots have `ndofs == 0` so the per-link loops below
    // iterate zero times — no early-return, see `gpu_mb_lu_decompose` for
    // the WGSL uniformity rationale.
    let _ = num_multibodies;

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let mm_start = batch_id * *mass_matrix_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let n = mb.ndofs;
    let m = MatSlice::dense(mm_start + mb.mass_matrix_offset as usize, n, n);
    let piv_offset = dof_start + mb.first_dof as usize;
    let rhs_offset = piv_offset;

    // Load matrix + rhs into shared memory.
    if lane < n {
        for r in 0..n {
            mat.write(sm_idx(r, lane), mass_matrices.read(m.idx(r, lane)));
        }
        x.write(lane as usize, rhs.read(rhs_offset + lane as usize));
    }
    workgroup_memory_barrier_with_group_sync();

    lu_apply_pivots(n, lane, lu_pivots, piv_offset, x);
    lu_triangular_solve_in_place(n, lane, mat, x, partial);

    if lane < n {
        rhs.write(rhs_offset + lane as usize, x.read(lane as usize));
    }
}

/// Fused factor + solve: factor `M = P·L·U` and immediately solve for `rhs`
/// in the same workgroup. `rhs` is overwritten with `x = M⁻¹·rhs` and the LU
/// factors / pivots are written back to global memory so subsequent solves
/// (joint-constraint init, contact-constraint finalize) can reuse them.
///
/// Saves one dispatch per `compute_dynamics` call versus running
/// `gpu_mb_lu_decompose` followed by `gpu_mb_lu_solve` — the matrix tile
/// already lives in shared memory after factoring, so the solve phase doesn't
/// need to reload it from global memory.
#[spirv_bindgen]
#[spirv(compute(threads(32, 1, 1)))]
pub fn gpu_mb_lu_factor_and_solve(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mass_matrices: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] lu_pivots: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] rhs: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] mass_matrix_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 7)] dof_batch_capacity: &u32,
    #[spirv(workgroup)] mat: &mut [f32; (MAX_MB_DOFS * MAX_MB_DOFS) as usize],
    #[spirv(workgroup)] x: &mut [f32; MAX_MB_DOFS],
    #[spirv(workgroup)] partial: &mut [f32; LANES as usize],
    #[spirv(workgroup)] pivot_row_shared: &mut u32,
    #[spirv(workgroup)] inv_akk_shared: &mut f32,
) {
    let batch_id = wg_id.y as usize;
    let mb_idx = wg_id.x;
    let lane = lid.x;
    // Padding multibody slots have `ndofs == 0` so the per-link loops below
    // iterate zero times — no early-return, see `gpu_mb_lu_decompose` for
    // the WGSL uniformity rationale.
    let _ = num_multibodies;

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let mm_start = batch_id * *mass_matrix_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let n = mb.ndofs;
    let m = MatSlice::dense(mm_start + mb.mass_matrix_offset as usize, n, n);
    let piv_offset = dof_start + mb.first_dof as usize;
    let rhs_offset = piv_offset;

    if lane < n {
        for r in 0..n {
            mat.write(sm_idx(r, lane), mass_matrices.read(m.idx(r, lane)));
        }
        x.write(lane as usize, rhs.read(rhs_offset + lane as usize));
    }
    workgroup_memory_barrier_with_group_sync();

    lu_factor_in_shared(
        n,
        lane,
        mat,
        lu_pivots,
        piv_offset,
        pivot_row_shared,
        inv_akk_shared,
    );

    // Persist LU factors to global memory so subsequent solves (joint /
    // contact constraint init) can reuse them. Each lane owns one column.
    if lane < n {
        for r in 0..n {
            mass_matrices.write(m.idx(r, lane), mat.read(sm_idx(r, lane)));
        }
    }

    // Apply the recorded pivots to the in-flight rhs and run the triangular
    // solves on the shared matrix (no reload needed).
    lu_apply_pivots(n, lane, lu_pivots, piv_offset, x);
    lu_triangular_solve_in_place(n, lane, mat, x, partial);

    if lane < n {
        rhs.write(rhs_offset + lane as usize, x.read(lane as usize));
    }
}
