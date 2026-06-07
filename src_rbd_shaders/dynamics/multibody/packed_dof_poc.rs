//! **PoC kernels for the DOF-major layout rewrite.**
//!
//! Three kernels:
//! 1. `gpu_mb_transpose_dof_state_to_packed` — copy the velocity section of
//!    `dof_state` (env-major: `[env*dpb + d]`) into `dof_state_packed`
//!    (DOF-major: `[d * num_batches + env]`).
//! 2. `gpu_mb_transpose_dof_state_from_packed` — copy back the other way,
//!    so kernels that haven't been ported to DOF-major keep working.
//! 3. `gpu_mb_integrate_velocities_packed` — does the same `v += a · dt`
//!    update as `gpu_mb_integrate_velocities` but reads / writes through the
//!    packed (DOF-major) buffers. With 32-lane warps, each lane handles one
//!    env so a load of `gen_forces_packed[d * N + env]` across the warp
//!    fetches 32 consecutive floats = ONE cache line (vs. the original
//!    env-major kernel where 32 envs' DOFs sat in 32 different cache lines).
//!
//! Once the layout pays off across more than one kernel, the transpose
//! kernels go away and every multibody kernel reads / writes the packed
//! layout directly. This file lets us measure the per-kernel win before
//! committing to that rewrite.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

use crate::utils::BatchIndices;

use super::types::MultibodyInfo;

/// 32 envs per workgroup. Each lane handles one env's (env, dof) pair, so
/// 32 consecutive (env, dof_d) writes hit one cache line in the packed
/// output. Dispatch grid is `[ceil(num_batches * dpb / 32), 1, 1]`.
const TRANSPOSE_LANES: u32 = 32;

/// Lay out the velocity half of `dof_state` (env-major,
/// `dpb · num_batches` floats) into `dof_state_packed` (DOF-major,
/// same total size). One thread per (env, dof) pair.
///
/// Dispatch: `[ceil(num_batches * dpb / 32), 1, 1]` with
/// `threads(32, 1, 1)`. Each thread:
///
/// - `global_id = wg_id.x * 32 + lid.x`
/// - `env = global_id / dpb`, `dof = global_id % dpb`
/// - `packed[dof * num_batches + env] = state[env * dpb + dof]`
///
/// Out-of-range threads early-out via `global_id >= num_batches * dpb`.
#[spirv_bindgen]
#[spirv(compute(threads(32, 1, 1)))]
pub fn gpu_mb_transpose_dof_state_to_packed(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] dof_state: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dof_state_packed: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] num_batches_u: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
) {
    let dpb = batch_ids.dof_batch_capacity;
    let num_batches = *num_batches_u;
    let total = num_batches * dpb;
    let global_id = wg_id.x * TRANSPOSE_LANES + lid.x;
    if global_id >= total {
        return;
    }
    let env = global_id / dpb;
    let dof = global_id % dpb;
    let src_idx = (env * dpb + dof) as usize;
    let dst_idx = (dof * num_batches + env) as usize;
    let v = dof_state.read(src_idx);
    dof_state_packed.write(dst_idx, v);
}

/// Reverse of [`gpu_mb_transpose_dof_state_to_packed`] — copies the DOF-major
/// `dof_state_packed` back into the velocity section of `dof_state` so the
/// downstream env-major kernels (joint constraints, contact constraints,
/// integrate-positions) still see fresh data.
#[spirv_bindgen]
#[spirv(compute(threads(32, 1, 1)))]
pub fn gpu_mb_transpose_dof_state_from_packed(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] dof_state_packed: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dof_state: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] num_batches_u: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
) {
    let dpb = batch_ids.dof_batch_capacity;
    let num_batches = *num_batches_u;
    let total = num_batches * dpb;
    let global_id = wg_id.x * TRANSPOSE_LANES + lid.x;
    if global_id >= total {
        return;
    }
    let env = global_id / dpb;
    let dof = global_id % dpb;
    let src_idx = (dof * num_batches + env) as usize;
    let dst_idx = (env * dpb + dof) as usize;
    let v = dof_state_packed.read(src_idx);
    dof_state.write(dst_idx, v);
}

/// Same as `gpu_mb_transpose_dof_state_to_packed` but for `gen_forces`
/// (which has no damping section, so the whole buffer is just velocities-
/// shaped generalized forces).
#[spirv_bindgen]
#[spirv(compute(threads(32, 1, 1)))]
pub fn gpu_mb_transpose_gen_forces_to_packed(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] gen_forces: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] gen_forces_packed: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] num_batches_u: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
) {
    let dpb = batch_ids.dof_batch_capacity;
    let num_batches = *num_batches_u;
    let total = num_batches * dpb;
    let global_id = wg_id.x * TRANSPOSE_LANES + lid.x;
    if global_id >= total {
        return;
    }
    let env = global_id / dpb;
    let dof = global_id % dpb;
    let src_idx = (env * dpb + dof) as usize;
    let dst_idx = (dof * num_batches + env) as usize;
    let v = gen_forces.read(src_idx);
    gen_forces_packed.write(dst_idx, v);
}

/// `v += a · dt` over the DOF-major packed buffers. 32 envs per workgroup,
/// one env per lane → coalesced loads (each warp loads 32 consecutive
/// floats from `gen_forces_packed[d * N + warp_base..warp_base+32]` per
/// DOF iteration).
///
/// Per-lane work: an `ndofs`-iter loop, one fused-multiply-add per iter,
/// reading `gen_forces_packed` and reading + writing `dof_state_packed`.
/// Critically: across the warp, the 32 lanes' reads land in ONE cache line
/// per iter (the failure mode of the prior env-major packed kernel was
/// 32 cache lines per iter).
#[spirv_bindgen]
#[spirv(compute(threads(32, 1, 1)))]
pub fn gpu_mb_integrate_velocities_packed_dof(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dof_state_packed: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] gen_forces_packed: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] dt_uniform: &f32,
    #[spirv(uniform, descriptor_set = 0, binding = 5)] num_batches_u: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] batch_ids: &BatchIndices,
) {
    let num_batches = *num_batches_u;
    let env = wg_id.x * 32 + lid.x;
    // OOB lanes early-exit via robust buffer access returning 0 from
    // `num_multibodies`; no explicit env-count check needed (see the prior
    // packed-kernel PoC's docs for the same pattern).
    let num_mb = num_multibodies.read(env as usize);
    if num_mb == 0 {
        return;
    }
    let dt = *dt_uniform;

    // Per-env one-multibody PoC: read the single multibody record and use its
    // ndofs + first_dof to bound the loop and offset into the DOF axis.
    let mb = batch_ids.mb_batch(env, multibody_info).read(0);
    let ndofs = mb.ndofs;
    if ndofs == 0 {
        return;
    }
    let first_dof = mb.first_dof;

    // Stride across the dof axis is num_batches (DOF-major layout). The
    // warp's 32 envs all read `[d * N + warp_base + lane]` — consecutive in
    // memory across lanes, ONE cache line per iter.
    for d in 0..ndofs {
        let idx = ((first_dof + d) * num_batches + env) as usize;
        let a = gen_forces_packed.read(idx);
        let v_old = dof_state_packed.read(idx);
        dof_state_packed.write(idx, v_old + a * dt);
    }
}
