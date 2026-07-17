//! **Layout microbench for the contact-constraint back-solves.**
//!
//! Question under test: is the finalize back-solve slow because of the
//! env-major memory LAYOUT (each thread walks its own env's DOFs at stride
//! ~n, so a warp touches 32 cache lines per load), and does an env-fastest
//! SoA layout (`buf[(slot)*num_envs + env]`, adjacent lanes = adjacent envs
//! = one cache line) reach the memory floor?
//!
//! Two kernels do the IDENTICAL finalize work per (env, constraint) —
//! copy `Jᵀ` into the column, tree-sparse LᵀDL back-solve in place, then
//! `inv_lhs = 1/(J·column)`:
//!
//! - [`gpu_bench_finalize_env_major`] mimics the production kernel: one
//!   workgroup per env, 32 lanes striding over constraints, env-major
//!   buffers (`[env*C*n + s*n + i]`).
//! - [`gpu_bench_finalize_soa`] is the hypothesis: one THREAD per
//!   (env, constraint), warps grouped env-consecutive on the same
//!   constraint, SoA buffers (`[(s*n + i)*E + env]`).
//!
//! All envs share one `parents` array (same robot topology), so control
//! flow is warp-uniform in both kernels — only the memory pattern differs.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

use super::lu::NO_PARENT;

const MAX_N: u32 = 64;

/// Tree-sparse LᵀDL solve of `x` in place, generic over the flat index maps.
/// `mat_idx(i, j)` addresses the factor matrix element `(row i, col j)`;
/// `x_idx(i)` addresses the solution/rhs vector element `i`.
#[inline]
fn ltdl_solve_mapped(
    mat: &[f32],
    x: &mut [f32],
    parents: &[u32],
    n: u32,
    mat_base: usize,
    mat_stride: usize,
    x_base: usize,
    x_stride: usize,
) {
    // mat element (i, j) -> mat_base + (i*n + j)*mat_stride
    // x element i        -> x_base + i*x_stride
    // Solve Lᵀ·z = b (scatter descending).
    for step in 0..n {
        let i = n - 1 - step;
        let xi = x.read(x_base + (i as usize) * x_stride);
        let mut j = parents.read(i as usize);
        for _ in 0..MAX_N {
            if j == NO_PARENT {
                break;
            }
            let l = mat.read(mat_base + ((i * n + j) as usize) * mat_stride);
            let v = x.read(x_base + (j as usize) * x_stride) - l * xi;
            x.write(x_base + (j as usize) * x_stride, v);
            j = parents.read(j as usize);
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
        let mut j = parents.read(i as usize);
        for _ in 0..MAX_N {
            if j == NO_PARENT {
                break;
            }
            let l = mat.read(mat_base + ((i * n + j) as usize) * mat_stride);
            s -= l * x.read(x_base + (j as usize) * x_stride);
            j = parents.read(j as usize);
        }
        x.write(x_base + (i as usize) * x_stride, s);
    }
}

/// Finalize-equivalent work for one (env, constraint): copy Jᵀ into the
/// column, back-solve, dot.
#[inline]
#[allow(clippy::too_many_arguments)]
fn finalize_one(
    mat: &[f32],
    jacs: &[f32],
    cols: &mut [f32],
    inv_lhs: &mut [f32],
    parents: &[u32],
    n: u32,
    mat_base: usize,
    mat_stride: usize,
    vec_base: usize,
    vec_stride: usize,
    out_idx: usize,
) {
    for i in 0..n {
        let v = jacs.read(vec_base + (i as usize) * vec_stride);
        cols.write(vec_base + (i as usize) * vec_stride, v);
    }
    ltdl_solve_mapped(mat, cols, parents, n, mat_base, mat_stride, vec_base, vec_stride);
    let mut acc = 0.0f32;
    for i in 0..n {
        acc += jacs.read(vec_base + (i as usize) * vec_stride)
            * cols.read(vec_base + (i as usize) * vec_stride);
    }
    inv_lhs.write(out_idx, if acc > 0.0 { 1.0 / acc } else { 0.0 });
}

/// Production-mimic: one workgroup per env (grid `[32, E, 1]` threads),
/// lanes stride over the `num_cons` constraints, env-major layout.
#[spirv_bindgen]
#[spirv(compute(threads(32, 1, 1)))]
pub fn gpu_bench_finalize_env_major(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] mat: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] jacs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] cols: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] inv_lhs: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] parents: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] n_u: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] num_cons_u: &u32,
) {
    let env = wg_id.y;
    let n = *n_u;
    let num_cons = *num_cons_u;
    // env-major: mat[env*n*n + i*n + j], vecs[env*C*n + s*n + i].
    let mat_base = (env * n * n) as usize;
    let mut s = lid.x;
    while s < num_cons {
        let vec_base = ((env * num_cons + s) * n) as usize;
        finalize_one(
            mat,
            jacs,
            cols,
            inv_lhs,
            parents,
            n,
            mat_base,
            1,
            vec_base,
            1,
            (env * num_cons + s) as usize,
        );
        s += 32;
    }
}

/// SoA hypothesis: one THREAD per (env, constraint), `t = s·E + env` so a
/// warp is 32 consecutive envs on the same constraint; env-fastest layout
/// (`mat[(i*n + j)*E + env]`, `vecs[(s*n + i)*E + env]`) makes every load
/// in the solve one coalesced line per warp. Grid `[E*C, 1, 1]` threads.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn gpu_bench_finalize_soa(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] mat: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] jacs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] cols: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] inv_lhs: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] parents: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] n_u: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] num_envs_u: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 7)] num_cons_u: &u32,
) {
    let n = *n_u;
    let num_envs = *num_envs_u;
    let num_cons = *num_cons_u;
    let t = wg_id.x * 64 + lid.x;
    if t >= num_envs * num_cons {
        return;
    }
    let s = t / num_envs;
    let env = t % num_envs;
    let e = num_envs as usize;
    // SoA: mat[(i*n + j)*E + env], vecs[(s*n + i)*E + env].
    finalize_one(
        mat,
        jacs,
        cols,
        inv_lhs,
        parents,
        n,
        env as usize,
        e,
        ((s * n) as usize) * e + env as usize,
        e,
        (s * num_envs + env) as usize,
    );
}
