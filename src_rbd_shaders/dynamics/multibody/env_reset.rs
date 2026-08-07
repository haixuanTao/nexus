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
#[allow(unused_imports)]
use super::ws_soa::{WS_COORDS, WS_LTP, WS_LTW};
use crate::Pose;
use crate::Vector;

/// Per-link translate flags, precomputed host-side from the (topology-constant)
/// template. Bit 0: this link belongs to a FREE-rooted multibody, so its
/// `local_to_world` moves with a spawn teleport. Bit 1: it is additionally that
/// multibody's root, whose `local_to_parent` and free linear coords also move.
/// Bit 1 implies bit 0.
pub const RESET_TRANSLATE_LTW: u32 = 1;
pub const RESET_TRANSLATE_ROOT: u32 = 2;

/// Spawn offset as a world-frame vector, from the packed per-reset quad.
#[cfg(feature = "dim3")]
#[inline]
fn off_vec(o: Vec4) -> Vector {
    Vector::new(o.x, o.y, o.z)
}

#[cfg(feature = "dim2")]
#[inline]
fn off_vec(o: Vec4) -> Vector {
    Vector::new(o.x, o.y)
}

/// Apply a spawn offset to workspace quad `q` of a link carrying `flags`.
///
/// This is the in-kernel form of the host's snapshot `translated()`: rotations,
/// velocities and non-root joint coordinates are translation-invariant, so only
/// the local-to-world translation (every link of a floating-base multibody) and
/// the root's local-to-parent translation + free linear coords move. Doing it
/// here is what lets the caller stage the UNtranslated template directly
/// instead of cloning and translating a snapshot per reset.
#[cfg(feature = "dim3")]
#[inline]
fn ws_apply_offset(v: Vec4, q: u32, flags: u32, o: Vec4) -> Vec4 {
    let shift = (flags & RESET_TRANSLATE_LTW != 0 && q == WS_LTW + 1)
        || (flags & RESET_TRANSLATE_ROOT != 0 && (q == WS_LTP + 1 || q == WS_COORDS));
    if shift {
        Vec4::new(v.x + o.x, v.y + o.y, v.z + o.z, v.w)
    } else {
        v
    }
}

/// dim2 mirror: poses are a single quad `(rot.re, rot.im, trans.x, trans.y)`,
/// and the free root's linear coords are `c0, c1`.
#[cfg(feature = "dim2")]
#[inline]
fn ws_apply_offset(v: Vec4, q: u32, flags: u32, o: Vec4) -> Vec4 {
    let pose_shift = (flags & RESET_TRANSLATE_LTW != 0 && q == WS_LTW)
        || (flags & RESET_TRANSLATE_ROOT != 0 && q == WS_LTP);
    if pose_shift {
        Vec4::new(v.x, v.y, v.z + o.x, v.w + o.y)
    } else if flags & RESET_TRANSLATE_ROOT != 0 && q == WS_COORDS {
        Vec4::new(v.x + o.x, v.y + o.y, v.z, v.w)
    } else {
        v
    }
}

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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] offsets: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] link_flags: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] links_workspace: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)]
    links_static: &mut [MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] dof_values: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] dof_state: &mut [f32],
    // x = count, y = num_batches, z = links_per_batch, w = dofs_per_batch.
    #[spirv(uniform, descriptor_set = 0, binding = 10)] params: &UVec4,
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
        let off = offsets.read(k as usize);

        // Spawn teleport is applied HERE rather than by cloning + translating a
        // snapshot per reset on the host.
        let link = j / WS_QUADS;
        let q = j - link * WS_QUADS;
        let ws = ws_apply_offset(
            staging_ws.read((k * span + j) as usize),
            q,
            link_flags.read(link as usize),
            off,
        );
        links_workspace.write((j * nb + env) as usize, ws);
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] offsets: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] body_flags: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] body_poses: &mut [Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] vels: &mut [Velocity],
    // x = count, y = bodies_per_batch, z = vels_per_batch, w = thread span.
    #[spirv(uniform, descriptor_set = 0, binding = 7)] params: &UVec4,
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
        let off = offsets.read(k as usize);
        if j < bps {
            // Only bodies backing a floating-base multibody move; ground and
            // terrain keep their template poses.
            let mut p = staging_poses.read((k * bps + j) as usize);
            if body_flags.read(j as usize) != 0 {
                p.translation += off_vec(off);
            }
            body_poses.write((env * bps + j) as usize, p);
        }
        if j < vs {
            vels.write((env * vs + j) as usize, staging_vels.read((k * vs + j) as usize));
        }
    }
}
