//! **`NEXUS_SOA_CONTACTS=1`: batch-interleaved (SoA) contact-constraint
//! pipeline.**
//!
//! The env-major constraint kernels are layout-bound: each thread walks its
//! own env's DOFs at stride ~n, so a warp touches 32 cache lines per load.
//! The layout microbench (`layout_bench.rs`) measured the identical
//! back-solve at 291 → 75 µs (3.9×) by interleaving envs
//! (`elem·NB + batch`): adjacent lanes = adjacent batches = one line.
//!
//! Universal SoA rule used here: an element whose env-major address is
//! `section_start(batch) + off` lives at `off·NB + batch` in SoA. Applied to
//! the three STREAMED buffers only — jac rows (written SoA directly by
//! `gpu_mb_init_contact_constraints_soa`), solve columns, and a factors
//! mirror (`mass_matrices_soa`, filled once per step by the tiled transpose
//! below). Constraint structs, counts and `dof_state` keep their layouts
//! (small / heavily reused, cache-friendly).
//!
//! Kernels:
//! - [`gpu_mb_transpose_factors_soa`] — 32×32 smem-tiled transpose of the
//!   mass-matrix slab `[NB, mm_cap] → [mm_cap, NB]`, both sides coalesced.
//! - [`gpu_mb_finalize_contact_constraints_soa`] — one THREAD per
//!   (batch, mb, slot), warps batch-consecutive: Jᵀ copy + tree-sparse LᵀDL
//!   back-solve + `inv_lhs` dot, every access coalesced.
//! - [`gpu_mb_solve_contact_constraints_soa`] /
//!   [`gpu_mb_remove_solve_contact_no_bias_soa`] — one THREAD per
//!   (batch, mb): serial Gauss-Seidel (bit-order preserved per env), jac /
//!   column streams coalesced; `dof_state` stays env-major (35 floats/env,
//!   L1-resident across the 2·count passes).

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

use crate::dynamics::body::Velocity;
use crate::utils::BatchIndices;
use crate::{gdot};

use super::lu::NO_PARENT;
use super::types::{
    MAX_MB_CONTACT_CONSTRAINTS_PER_MB, MB_CONTACT_KIND_TANGENT, MultibodyContactConstraint,
    MultibodyInfo,
};

const MAX_N: u32 = 64;
const TILE: u32 = 32;

/// 32×32 smem-tiled transpose: `mass_matrices` (env-major slabs,
/// `batch·mm_cap + e`) → `mass_matrices_soa` (`e·NB + batch`). Loads are
/// coalesced along `e`, stores along `batch`. Grid:
/// `[ceil(NB/32)·32, ceil(mm_cap/32), 1]` threads (threads(32)).
#[spirv_bindgen]
#[spirv(compute(threads(32, 1, 1)))]
pub fn gpu_mb_transpose_factors_soa(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] mass_matrices: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mass_matrices_soa: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] num_batches_u: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
    // +1 padding column to dodge smem bank conflicts on the transposed reads.
    #[spirv(workgroup)] tile: &mut [f32; (TILE * (TILE + 1)) as usize],
) {
    let nb = *num_batches_u;
    let cap = batch_ids.mass_matrix_batch_capacity;
    let lane = lid.x;
    let b0 = wg_id.x * TILE; // first batch of this tile
    let e0 = wg_id.y * TILE; // first element of this tile

    // Load: row r covers batch b0+r, lanes sweep e0+lane (coalesced in e).
    for r in 0..TILE {
        let b = b0 + r;
        let e = e0 + lane;
        let v = if b < nb && e < cap {
            mass_matrices.read((b * cap + e) as usize)
        } else {
            0.0
        };
        tile.write((r * (TILE + 1) + lane) as usize, v);
    }
    workgroup_memory_barrier_with_group_sync();
    // Store: row r covers element e0+r, lanes sweep b0+lane (coalesced in b).
    for r in 0..TILE {
        let e = e0 + r;
        let b = b0 + lane;
        if b < nb && e < cap {
            let v = tile.read((lane * (TILE + 1) + r) as usize);
            mass_matrices_soa.write((e * nb + b) as usize, v);
        }
    }
}

/// Tree-sparse LᵀDL solve of `x` in place through strided index maps.
/// Factor element `(r, c)` (col-major flat `c·n + r` within the mb block) is
/// at `mat_base + (c·n + r)·mat_stride`; `x` element `i` at
/// `x_base + i·x_stride`. Same operation order as `ltdl_solve_in_shared`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn ltdl_solve_strided(
    mat: &[f32],
    x: &mut [f32],
    parents: &[u32],
    parents_offset: usize,
    n: u32,
    mat_base: usize,
    mat_stride: usize,
    x_base: usize,
    x_stride: usize,
) {
    // Factors: strict lower = L (element (k, i) with i ancestor of k at
    // col-major flat i·n + k), diagonal = D.
    // Solve Lᵀ·z = b (scatter descending).
    for step in 0..n {
        let i = n - 1 - step;
        let xi = x.read(x_base + (i as usize) * x_stride);
        let mut j = parents.read(parents_offset + i as usize);
        for _ in 0..MAX_N {
            if j == NO_PARENT {
                break;
            }
            let l = mat.read(mat_base + ((j * n + i) as usize) * mat_stride);
            let v = x.read(x_base + (j as usize) * x_stride) - l * xi;
            x.write(x_base + (j as usize) * x_stride, v);
            j = parents.read(parents_offset + j as usize);
        }
    }
    // z = D⁻¹·z.
    for i in 0..n {
        let d = mat.read(mat_base + ((i * n + i) as usize) * mat_stride);
        let v = x.read(x_base + (i as usize) * x_stride);
        x.write(
            x_base + (i as usize) * x_stride,
            if d != 0.0 { v / d } else { 0.0 },
        );
    }
    // Solve L·x = z (gather ascending).
    for i in 0..n {
        let mut s = x.read(x_base + (i as usize) * x_stride);
        let mut j = parents.read(parents_offset + i as usize);
        for _ in 0..MAX_N {
            if j == NO_PARENT {
                break;
            }
            let l = mat.read(mat_base + ((j * n + i) as usize) * mat_stride);
            s -= l * x.read(x_base + (j as usize) * x_stride);
            j = parents.read(parents_offset + j as usize);
        }
        x.write(x_base + (i as usize) * x_stride, s);
    }
}

/// SoA finalize: one thread per (batch, mb, slot); `t = elem·NB + batch` so a
/// warp is 32 consecutive batches on the same (mb, slot). Reads jacs +
/// factors SoA, writes columns SoA, sets `inv_lhs` in the (AoS) constraint.
/// Dispatch: `[NB · mb_cap · MAX_MB_CONTACT_CONSTRAINTS_PER_MB, 1, 1]`.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn gpu_mb_finalize_contact_constraints_soa(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mass_matrices_soa: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] lu_pivots: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] contact_constraint_jacs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)]
    contact_constraint_columns: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] contact_constraint_count: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 8)] num_batches_u: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 9)] batch_ids: &BatchIndices,
) {
    let nb = *num_batches_u;
    let mb_cap = batch_ids.multibodies_batch_capacity;
    let maxc = MAX_MB_CONTACT_CONSTRAINTS_PER_MB;
    let t = wg_id.x * 64 + lid.x;
    if t >= nb * mb_cap * maxc {
        return;
    }
    let batch_id = t % nb;
    let rest = t / nb;
    let mb_idx = rest / maxc;
    let s = rest % maxc;
    if mb_idx >= num_multibodies.read(batch_id as usize) {
        return;
    }
    let mb_start = batch_ids.mb_start(batch_id);
    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let ndofs = mb.ndofs;
    if ndofs == 0 || s >= contact_constraint_count.read(mb_start + mb_idx as usize) {
        return;
    }
    let nbz = nb as usize;
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let piv_offset = batch_ids.dof_start(batch_id) + mb.first_dof as usize;
    // SoA addresses: env-major offset `off` → `off·NB + batch`.
    let mat_base = (mb.mass_matrix_offset as usize) * nbz + batch_id as usize;
    let vec_off = ((mb_idx * maxc + s) as usize) * dofs_stride;
    let vec_base = vec_off * nbz + batch_id as usize;
    let cons_start = batch_ids.mb_contact_constraints_start(batch_id);
    let cons_base = cons_start + (mb_idx as usize) * (maxc as usize);

    // 1) Copy Jᵀ into the column (both SoA, fully coalesced).
    for i in 0..ndofs {
        let v = contact_constraint_jacs.read(vec_base + (i as usize) * nbz);
        contact_constraint_columns.write(vec_base + (i as usize) * nbz, v);
    }
    // 2) Solve M · column = Jᵀ in place.
    ltdl_solve_strided(
        mass_matrices_soa,
        contact_constraint_columns,
        lu_pivots,
        piv_offset,
        ndofs,
        mat_base,
        nbz,
        vec_base,
        nbz,
    );
    // 3) inv_lhs = 1 / (J·column + free-body term).
    let mut inv_r_mb = 0.0f32;
    for i in 0..ndofs {
        let j = contact_constraint_jacs.read(vec_base + (i as usize) * nbz);
        let c = contact_constraint_columns.read(vec_base + (i as usize) * nbz);
        inv_r_mb += j * c;
    }
    let mut cons = contact_constraints.read(cons_base + s as usize);
    let is_self = cons.free_body_id == u32::MAX;
    let inv_r_free = if is_self {
        0.0
    } else {
        cons.free_body_im + gdot(cons.ang_jac, cons.ii_ang_jac)
    };
    let total = inv_r_mb + inv_r_free;
    cons.inv_lhs = if total > 0.0 { 1.0 / total } else { 0.0 };
    contact_constraints.write(cons_base + s as usize, cons);
}

/// Scalar Gauss-Seidel sweep for one (batch, mb) with SoA jac / column
/// streams. Same constraint order as the cooperative sweep; the serial `J·v`
/// summation replaces its tree reduction (f32 rounding class only).
#[inline]
#[allow(clippy::too_many_arguments)]
fn contact_sweep_soa(
    multibody_info: &[MultibodyInfo],
    contact_constraints: &mut [MultibodyContactConstraint],
    contact_constraint_jacs: &[f32],
    contact_constraint_columns: &[f32],
    contact_constraint_count: &[u32],
    dof_state: &mut [f32],
    solver_vels: &mut [Velocity],
    batch_id: u32,
    mb_idx: u32,
    nb: usize,
    batch_ids: &BatchIndices,
    strip_bias: bool,
) {
    let mb_start = batch_ids.mb_start(batch_id);
    let cons_start = batch_ids.mb_contact_constraints_start(batch_id);
    let colliders_start = batch_ids.coll_start(batch_id);
    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let ndofs = mb.ndofs;
    if ndofs == 0 {
        return;
    }
    let maxc = MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize;
    let v_base = batch_ids.dof_start(batch_id) + mb.first_dof as usize;
    let cons_base = cons_start + (mb_idx as usize) * maxc;
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let count = contact_constraint_count.read(mb_start + mb_idx as usize);

    if strip_bias {
        for s in 0..count {
            let mut cons = contact_constraints.read(cons_base + s as usize);
            if cons.kind != 0 {
                cons.rhs = cons.rhs_wo_bias;
                contact_constraints.write(cons_base + s as usize, cons);
            }
        }
    }

    for s in 0..count {
        let vec_base = ((mb_idx as usize * maxc + s as usize) * dofs_stride) * nb
            + batch_id as usize;

        let mut j_dot_v = 0.0f32;
        for i in 0..ndofs {
            j_dot_v += contact_constraint_jacs.read(vec_base + (i as usize) * nb)
                * dof_state.read(v_base + i as usize);
        }

        let mut cons = contact_constraints.read(cons_base + s as usize);
        let is_self = cons.free_body_id == u32::MAX;
        let free = if is_self {
            Velocity::default()
        } else {
            solver_vels.read(colliders_start + cons.free_body_id as usize)
        };
        if !is_self {
            j_dot_v += cons.lin_jac.dot(free.linear) + gdot(cons.ang_jac, free.angular);
        }
        let rhs_total = j_dot_v + cons.rhs;
        let raw_imp = (cons.impulse - cons.inv_lhs * (rhs_total + cons.cfm_gain * cons.impulse))
            / (1.0 + cons.cfm_coeff);
        let new_imp = if cons.kind == MB_CONTACT_KIND_TANGENT {
            let normal =
                contact_constraints.read(cons_base + cons.normal_constraint_slot as usize);
            let limit = cons.friction_coeff * normal.impulse;
            if raw_imp > limit {
                limit
            } else if raw_imp < -limit {
                -limit
            } else {
                raw_imp
            }
        } else if raw_imp < 0.0 {
            0.0
        } else {
            raw_imp
        };
        let delta = new_imp - cons.impulse;
        cons.impulse = new_imp;
        contact_constraints.write(cons_base + s as usize, cons);
        if delta != 0.0 {
            if !is_self {
                let mut new_free = free;
                new_free.linear = new_free.linear + cons.lin_jac * (cons.free_body_im * delta);
                new_free.angular = new_free.angular + cons.ii_ang_jac * delta;
                solver_vels.write(colliders_start + cons.free_body_id as usize, new_free);
            }
            for i in 0..ndofs {
                let v_idx = v_base + i as usize;
                let cur = dof_state.read(v_idx);
                let col = contact_constraint_columns.read(vec_base + (i as usize) * nb);
                dof_state.write(v_idx, cur + delta * col);
            }
        }
    }
}

/// SoA PGS sweep: one thread per (batch, mb), `t = m·NB + batch`.
/// Dispatch: `[NB · mb_cap, 1, 1]` threads.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn gpu_mb_solve_contact_constraints_soa(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contact_constraint_jacs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] contact_constraint_columns: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] contact_constraint_count: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] dof_state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] solver_vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 8)] num_batches_u: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 9)] batch_ids: &BatchIndices,
) {
    let nb = *num_batches_u;
    let mb_cap = batch_ids.multibodies_batch_capacity;
    let t = wg_id.x * 64 + lid.x;
    if t >= nb * mb_cap {
        return;
    }
    let batch_id = t % nb;
    let mb_idx = t / nb;
    if mb_idx >= num_multibodies.read(batch_id as usize) {
        return;
    }
    contact_sweep_soa(
        multibody_info,
        contact_constraints,
        contact_constraint_jacs,
        contact_constraint_columns,
        contact_constraint_count,
        dof_state,
        solver_vels,
        batch_id,
        mb_idx,
        nb as usize,
        batch_ids,
        false,
    );
}

/// SoA fused stabilization sweep (strip bias + no-bias sweep), one thread per
/// (batch, mb). Dispatch: `[NB · mb_cap, 1, 1]` threads.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn gpu_mb_remove_solve_contact_no_bias_soa(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contact_constraint_jacs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] contact_constraint_columns: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] contact_constraint_count: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] dof_state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] solver_vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 8)] num_batches_u: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 9)] batch_ids: &BatchIndices,
) {
    let nb = *num_batches_u;
    let mb_cap = batch_ids.multibodies_batch_capacity;
    let t = wg_id.x * 64 + lid.x;
    if t >= nb * mb_cap {
        return;
    }
    let batch_id = t % nb;
    let mb_idx = t / nb;
    if mb_idx >= num_multibodies.read(batch_id as usize) {
        return;
    }
    contact_sweep_soa(
        multibody_info,
        contact_constraints,
        contact_constraint_jacs,
        contact_constraint_columns,
        contact_constraint_count,
        dof_state,
        solver_vels,
        batch_id,
        mb_idx,
        nb as usize,
        batch_ids,
        true,
    );
}
