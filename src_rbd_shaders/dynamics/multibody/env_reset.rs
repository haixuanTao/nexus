//! Per-environment reset scatter for the RL teleport/reset primitives.
//!
//! Copies one environment's carry-over multibody state — SoA link workspace,
//! static link descriptors, generalized coordinates and velocities — from a
//! compact contiguous staging blob into the batch-interleaved live buffers.
//! One staging upload + one dispatch per reset, instead of the hundreds of
//! strided `write_buffer`s the interleaved layout would otherwise force (an
//! env's data is strided across the whole buffer: element `intra` of batch
//! `b` lives at `intra·num_batches + b`, and workspace quads at
//! `(link·WS_QUADS + q)·num_batches + b`). The staging blob is exactly the
//! `num_batches = 1` interleaving, so source indices are the flat `0..len`.

use glamx::{UVec4, Vec4};
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

use super::super::body::Velocity;
use super::types::MultibodyLinkStatic;
use super::ws_soa::WS_QUADS;
use crate::Pose;

/// Scatters `count` staged env states into the interleaved buffers in ONE
/// dispatch. Dispatch `[count · links_per_batch · WS_QUADS, 1, 1]` threads:
/// each env gets a contiguous span of `links_per_batch · WS_QUADS` threads
/// (the largest of the three per-element loops, since
/// `links_per_batch · WS_QUADS ≥ links_per_batch` and
/// `dofs_per_batch ≤ links_per_batch · WS_QUADS` for any real multibody).
///
/// BATCHING IS THE POINT. An RL rollout terminates thousands of envs per step,
/// and the previous one-env-per-dispatch form paid a staging upload, a params
/// allocation, an encoder build and a kernel launch EACH — tens of thousands of
/// launches per training iteration to move a few kilobytes each time. Here the
/// caller concatenates every terminating env's staging blob and passes the
/// destination env ids in `dst_envs`, so the whole step costs one upload set and
/// one launch.
///
/// Source layout is env-major and tightly packed: env `k` reads
/// `staging_ws[k · lpb · WS_QUADS ..]`, `staging_links[k · lpb ..]` and
/// `staging_dofs[k · 2 · dpb ..]`. Destination is the batch-interleaved live
/// buffer, so intra-env element `j` of env `dst_envs[k]` lands at
/// `j · num_batches + dst_envs[k]`.
///
/// Each env's `staging_dofs` block holds `dofs_per_batch` generalized
/// coordinates followed by `dofs_per_batch` generalized velocities. Only the
/// velocity section of `dof_state` is written — the damping section that
/// follows it is static configuration, not per-episode state.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_env_reset(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] staging_ws: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    staging_links: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] staging_dofs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dst_envs: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] links_workspace: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)]
    links_static: &mut [MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] dof_values: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] dof_state: &mut [f32],
    // x = count, y = num_batches, z = links_per_batch, w = dofs_per_batch.
    #[spirv(uniform, descriptor_set = 0, binding = 8)] params: &UVec4,
) {
    let i = invocation_id.x;
    let count = params.x;
    let nb = params.y;
    let lpb = params.z;
    let dpb = params.w;

    let span = lpb * WS_QUADS;
    let k = i / span; // which reset in this batch
    let j = i - k * span; // intra-env element

    // The dispatch rounds up to whole workgroups, so the tail threads (and any
    // slack past `count` envs) fall out here.
    if k < count {
        let env = dst_envs.read(k as usize);

        links_workspace.write((j * nb + env) as usize, staging_ws.read((k * span + j) as usize));
        if j < lpb {
            links_static.write((j * nb + env) as usize, staging_links.read((k * lpb + j) as usize));
        }
        if j < dpb {
            let base = k * 2 * dpb;
            dof_values.write((j * nb + env) as usize, staging_dofs.read((base + j) as usize));
            dof_state.write((j * nb + env) as usize, staging_dofs.read((base + dpb + j) as usize));
        }
    }
}

/// Companion to [`gpu_mb_env_reset`] for the rigid-body side of a reset: the
/// per-body world poses and velocities.
///
/// Unlike the multibody buffers, these are env-MAJOR — env `e` owns the
/// contiguous run `[e · per_batch, (e+1) · per_batch)`. A single env is
/// therefore one contiguous `write_buffer`, which is why this was left as a
/// host copy per reset originally; but at thousands of resets per rollout step
/// those two small H2D copies per env dominate, so batching them into one
/// staged upload plus one dispatch is the same win as the multibody scatter.
///
/// Dispatch `[count · max(bodies_per_batch, vels_per_batch), 1, 1]` threads.
/// Staging is env-major and tightly packed: env `k` reads
/// `staging_poses[k · bps ..]` and `staging_vels[k · vs ..]`.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_rbd_env_reset_bodies(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] staging_poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] staging_vels: &[Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] dst_envs: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] body_poses: &mut [Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] vels: &mut [Velocity],
    // x = count, y = bodies_per_batch, z = vels_per_batch, w = thread span.
    #[spirv(uniform, descriptor_set = 0, binding = 5)] params: &UVec4,
) {
    let i = invocation_id.x;
    let count = params.x;
    let bps = params.y;
    let vs = params.z;
    let span = params.w;

    let k = i / span; // which reset in this batch
    let j = i - k * span; // intra-env element

    if k < count {
        let env = dst_envs.read(k as usize);
        if j < bps {
            body_poses.write((env * bps + j) as usize, staging_poses.read((k * bps + j) as usize));
        }
        if j < vs {
            vels.write((env * vs + j) as usize, staging_vels.read((k * vs + j) as usize));
        }
    }
}
