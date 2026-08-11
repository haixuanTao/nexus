//! GPU motor-target scatter: write per-(env, actuated-joint) target positions
//! directly into `links_static` on the GPU, replacing the host-side
//! `stage_motor_position` + `flush_links_static` (a per-step H2D copy of the
//! whole mirror). Enables applying RL policy actions without a host
//! round-trip — a prerequisite for capturing the rollout into a CUDA graph
//! (no per-step host writes).
//!
//! `links_static` is batch-interleaved: link `l` of env `e` lives at
//! `l · num_envs + e`. Targets are row-major `[num_actuated × num_envs]`,
//! element `(j, env)` at `j · num_envs + env` (matches the policy action
//! buffer layout).

use khal_std::glamx::{UVec3, UVec4};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

use super::types::MultibodyLinkStatic;

/// One thread per (actuated-joint `x`, env `y`). Writes `target_pos` into the
/// matching motor and sets the `motor_axes` bit, same as
/// `stage_motor_position`.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_scatter_motor_targets(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] motor_targets: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    links_static: &mut [MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] actuated_link_ids: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] num_actuated: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 4)] num_envs: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 5)] axis_id: &u32,
) {
    let j = invocation_id.x;
    let env = invocation_id.y;
    if j >= *num_actuated || env >= *num_envs {
        return;
    }
    let link_id = actuated_link_ids[j as usize];
    // Batch-interleaved links layout.
    let global_idx = (link_id * *num_envs + env) as usize;
    let target = motor_targets[(j * *num_envs + env) as usize];

    // NOTE: the silly single-iteration loop matches
    // `gpu_lbvh_reset_collision_pairs`: rust-gpu occasionally prunes the
    // SPIR-V for kernels it deems too trivial; the loop shell keeps the
    // entry point emitted.
    for _ in 0..1 {
        let link = &mut links_static[global_idx];
        link.data.motors[*axis_id as usize].target_pos = target;
        link.data.motor_axes |= 1u32 << *axis_id;
    }
}

/// Per-step actuator-delay state refresh, on device: `tick ← 0`,
/// `k ← k_eff[env]`, and the actuated links' `prev_target` lanes copied from
/// the PREVIOUS step's motor-target tensor (row-major `[num_actuated × n]` —
/// the same buffer the target scatter consumed last step, read BEFORE this
/// step's scatter overwrites it). Replaces a full `stride × n` host rebuild +
/// upload per step with one `[n]` upload (`k_eff`) + this dispatch.
/// Non-actuated `prev` lanes stay at their zero-initialized value, exactly
/// like the host path's zero fill.
///
/// Dispatch `[num_actuated, num_envs, 1]` threads; lane `j == 0` also writes
/// the two scalar lanes.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_delay_state_update(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] prev_targets: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] k_eff: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] actuated_link_ids: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] delay_state: &mut [f32],
    // x = num_actuated, y = num_envs, z = stride (2 + links_per_batch).
    #[spirv(uniform, descriptor_set = 0, binding = 4)] params: &UVec4,
) {
    let j = invocation_id.x;
    let env = invocation_id.y;
    if j >= params.x || env >= params.y {
        return;
    }
    let base = (env * params.z) as usize;
    if j == 0 {
        delay_state.write(base, 0.0);
        delay_state.write(base + 1, k_eff.read(env as usize));
    }
    let link = actuated_link_ids.read(j as usize);
    delay_state.write(
        base + 2 + link as usize,
        prev_targets.read((j * params.y + env) as usize),
    );
}
