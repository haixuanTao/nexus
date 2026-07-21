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

use super::types::MultibodyLinkStatic;
use super::ws_soa::WS_QUADS;

/// Scatters the staged env state into the interleaved buffers. Dispatch
/// `[links_per_batch · WS_QUADS, 1, 1]` threads (the largest of the three
/// per-element loops; `links_per_batch · WS_QUADS ≥ links_per_batch`, and
/// `dofs_per_batch ≤ links_per_batch · WS_QUADS` for any real multibody).
///
/// `staging_dofs` holds `dofs_per_batch` generalized coordinates followed by
/// `dofs_per_batch` generalized velocities. Only the velocity section of
/// `dof_state` is written — the damping section that follows it is static
/// configuration, not per-episode state.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_env_reset(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] staging_ws: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    staging_links: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] staging_dofs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] links_workspace: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)]
    links_static: &mut [MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] dof_values: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] dof_state: &mut [f32],
    // x = dst_env, y = num_batches, z = links_per_batch, w = dofs_per_batch.
    #[spirv(uniform, descriptor_set = 0, binding = 7)] params: &UVec4,
) {
    let i = invocation_id.x;
    let env = params.x;
    let nb = params.y;
    let lpb = params.z;
    let dpb = params.w;

    if i < lpb * WS_QUADS {
        links_workspace.write((i * nb + env) as usize, staging_ws.read(i as usize));
    }
    if i < lpb {
        links_static.write((i * nb + env) as usize, staging_links.read(i as usize));
    }
    if i < dpb {
        dof_values.write((i * nb + env) as usize, staging_dofs.read(i as usize));
        dof_state.write((i * nb + env) as usize, staging_dofs.read((dpb + i) as usize));
    }
}
