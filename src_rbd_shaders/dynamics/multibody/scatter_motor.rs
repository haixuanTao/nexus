//! GPU motor-target scatter: write per-(env, actuated-joint) target positions
//! directly into `links_static` on the GPU, replacing the host-side
//! `stage_motor_position` + `flush_links_static` (a per-step H2D copy). Enables
//! applying RL policy actions without a host round-trip — a prerequisite for
//! capturing the rollout into a CUDA graph (no per-step host writes).
//!
//! Layout mirrors `stage_motor_position` exactly: `links_static` is flat
//! `[num_envs * links_per_batch]`, global index = `env * links_per_batch +
//! link_id`. Targets are row-major `[num_actuated x num_envs]`, element
//! `(j, env)` at `j * num_envs + env` (matches the policy action buffer).

use khal_std::glamx::UVec3;
use khal_std::macros::{spirv, spirv_bindgen};

use super::types::MultibodyLinkStatic;

/// One thread per (actuated-joint `x`, env `y`). Writes `target_pos` into the
/// matching motor and sets the `motor_axes` bit, same as `stage_motor_position`.
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
    #[spirv(uniform, descriptor_set = 0, binding = 5)] links_per_batch: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] axis_id: &u32,
) {
    let j = invocation_id.x;
    let env = invocation_id.y;
    if j >= *num_actuated || env >= *num_envs {
        return;
    }
    let link_id = actuated_link_ids[j as usize];
    let global_idx = (env * *links_per_batch + link_id) as usize;
    let target = motor_targets[(j * *num_envs + env) as usize];

    let link = &mut links_static[global_idx];
    link.data.motors[*axis_id as usize].target_pos = target;
    link.data.motor_axes |= 1u32 << *axis_id;
}
