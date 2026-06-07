//! Integrate kernel.
//!
//! Semi-implicit Euler:
//!   v += a * dt                           (a = generalized acceleration from solve)
//!   coords, joint_rot updated per-link using `v`
//!
//! The angular-DOF update mirrors rapier's `MultibodyJoint::integrate`:
//!   - 1 free angular DOF:  coords[DIM + dof_id] += v * dt; joint_rot from
//!     axis-angle (3D) / scalar angle (2D).
//!   - 3 free angular DOFs: joint_rot = exp(v * dt) * joint_rot;
//!     coords[3..6] += v * dt. (3D only.)
//!   - 0 free angular DOFs: no-op.
//!
//! After this pass, `dof_velocities` and each link's `coords` / `joint_rot` are updated.
//! Callers are expected to re-run forward kinematics to refresh link poses.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

#[cfg(feature = "dim2")]
use crate::rotation_from_angle;
use crate::utils::{BatchIndices, Slice, SliceMut};
use crate::{ANG_DIM, DIM};
#[cfg(feature = "dim3")]
use crate::{Vector, rotation_from_scaled_axis, rotation_renormalize_fast};
use parry::math::VectorExt;

use super::types::{MultibodyInfo, MultibodyLinkStatic, MultibodyLinkWorkspace};

/// Update generalized velocities: `v += a · dt`.
///
/// Split out from the position-update half so that joint-limit / motor
/// constraints can run in between (rapier's order: velocity update → constraint
/// solver → position update).
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_integrate_velocities(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dof_state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] gen_accelerations: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] dt_uniform: &f32,
    #[spirv(uniform, descriptor_set = 0, binding = 5)] batch_ids: &BatchIndices,
) {
    let batch_id = invocation_id.y;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id as usize);
    if mb_idx >= num_mb {
        return;
    }
    let dt = *dt_uniform;

    let mb = batch_ids
        .mb_batch(batch_id, multibody_info)
        .read(mb_idx as usize);
    let gen_base = batch_ids.dof_start(batch_id) + mb.first_dof as usize;

    let mut dof_vel = SliceMut(dof_state, gen_base);
    let acc = Slice(gen_accelerations, gen_base);

    for d in 0..mb.ndofs {
        let di = d as usize;
        dof_vel[di] = dof_vel[di] + acc[di] * dt;
    }
}

/// **PoC — DISABLED.** `threads(32, 1, 1)` packing variant of
/// `gpu_mb_integrate_velocities` — one lane per env instead of one workgroup
/// per env. Layout intent: ~100% SIMD lane occupancy instead of ~3%.
///
/// **Measured empirically to REGRESS throughput by 18–24% across N on both
/// champagne (RTX 5090 + Vulkan) and a M-series mac (WebGPU).** Per-step
/// GPU compute went up ~30% even though the kernel does the same scalar
/// work as the serial version. Suspected causes (not confirmed without
/// Nsight Compute / Metal Frame Capture profiling):
///
/// - **Memory access pattern**: each lane reads from a stride-18-floats
///   region (`dof_start(env_idx) + d`) — across 32 lanes that's 32 different
///   cache lines per dword. The original kernel was no better in absolute
///   bytes touched, but each warp serialised across one env's contiguous
///   region. Packing changed which lane reads which line within a warp
///   without changing total bytes — but apparently the GPU's L1/L2 prefetch
///   prefers the original layout. An env-interleaved storage layout (DOF[0]
///   of env 0,1,2,...,31, then DOF[1] of env 0,1,2,...) would give coalesced
///   reads and is likely necessary for this technique to actually win.
/// - **Shader codegen**: adding the new kernel may have changed how
///   rust-gpu / naga optimised the surrounding multibody kernels.
///
/// Kept in the source as a starting point + reference for the next attempt
/// (which needs the layout change first). The host-side dispatcher field
/// was removed since unused fields still cost kernel-load time at startup.
/// See `multibody.rs:1660` for the (now reverted) integration point.
#[spirv_bindgen]
#[spirv(compute(threads(32, 1, 1)))]
pub fn gpu_mb_integrate_velocities_packed(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dof_state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] gen_accelerations: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] dt_uniform: &f32,
    #[spirv(uniform, descriptor_set = 0, binding = 5)] batch_ids: &BatchIndices,
) {
    // Map (workgroup, lane) → env. Each workgroup covers 32 consecutive envs.
    let env_idx = wg_id.x * 32 + lid.x;
    // Bounds check. For envs past the real count, `num_multibodies.read` is
    // either reading padding-zeros (when the buffer was over-allocated) or an
    // OOB slot — WGPU's robust buffer access maps OOB reads to 0, so either
    // way we get `num_mb == 0` and early-exit. No explicit env-count uniform
    // needed.
    let num_mb = num_multibodies.read(env_idx as usize);
    if num_mb == 0 {
        return;
    }
    let dt = *dt_uniform;

    // batch_id = env_idx; mb_idx = 0 (PoC assumes 1 multibody per env).
    let mb = batch_ids.mb_batch(env_idx, multibody_info).read(0);
    let gen_base = batch_ids.dof_start(env_idx) + mb.first_dof as usize;

    let mut dof_vel = SliceMut(dof_state, gen_base);
    let acc = Slice(gen_accelerations, gen_base);

    for d in 0..mb.ndofs {
        let di = d as usize;
        dof_vel[di] = dof_vel[di] + acc[di] * dt;
    }
}

#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_integrate(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    links_workspace: &mut [MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dof_values: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] dof_state: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 6)] dt_uniform: &f32,
    #[spirv(uniform, descriptor_set = 0, binding = 7)] batch_ids: &BatchIndices,
) {
    let batch_id = invocation_id.y;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id as usize);
    if mb_idx >= num_mb {
        return;
    }
    let dt = *dt_uniform;

    let mb = batch_ids
        .mb_batch(batch_id, multibody_info)
        .read(mb_idx as usize);
    let num_links = mb.num_links;
    let gen_base = batch_ids.dof_start(batch_id) + mb.first_dof as usize;

    let stat_slice = batch_ids
        .mb_links_batch(batch_id, links_static)
        .offset(mb.first_link as usize);
    let mut ws_slice = batch_ids
        .mb_links_batch_mut(batch_id, links_workspace)
        .offset(mb.first_link as usize);
    let dof_val = SliceMut(dof_values, gen_base);
    let dof_vel = Slice(dof_state, gen_base);

    // Per-link coord / joint_rot update (uses the already-corrected `dof_velocities`).
    //
    // Only `coords` (≤ 24 B) and `joint_rot` (16 B) are modified. We mutate them
    // in place through `&mut ws_slice[k]` so SPIR-V emits field-targeted stores
    // instead of a whole `MultibodyLinkWorkspace` round-trip (~240 B in 3D).
    for k in 0..num_links {
        let k_usize = k as usize;
        let stat = stat_slice[k_usize];
        let locked = stat.data.locked_axes;
        let aid = stat.assembly_id as usize;
        let ws = &mut ws_slice[k_usize];

        // Free linear DOFs first, in axis order.
        let mut curr_free = 0u32;
        for i in 0..DIM {
            if (locked & (1 << i)) == 0 {
                let v = dof_vel[aid + curr_free as usize];
                *ws.coords.at_mut(i as usize) += v * dt;
                curr_free += 1;
            }
        }

        // Free angular DOFs.
        let ang_locked = (locked >> DIM) & ((1 << ANG_DIM) - 1);
        let num_ang = ANG_DIM - ang_locked.count_ones();
        if num_ang == 1 {
            #[cfg(feature = "dim3")]
            {
                let dof_id = (!ang_locked & 0x7).trailing_zeros();
                let v = dof_vel[aid + curr_free as usize];
                let idx = 3 + dof_id;
                let new = ws.coords.read(idx as usize) + v * dt;
                ws.coords.write(idx as usize, new);
                ws.joint_rot = rotation_from_scaled_axis(Vector::ith(dof_id as usize, new));
            }
            #[cfg(feature = "dim2")]
            {
                let v = dof_vel[aid + curr_free as usize];
                let new = ws.coords.read(DIM as usize) + v * dt;
                ws.coords.write(DIM as usize, new);
                ws.joint_rot = rotation_from_angle(new);
            }
        } else if num_ang == 3 {
            #[cfg(feature = "dim3")]
            {
                let vx = dof_vel[aid + curr_free as usize];
                let vy = dof_vel[aid + (curr_free + 1) as usize];
                let vz = dof_vel[aid + (curr_free + 2) as usize];
                let ang = Vector::new(vx, vy, vz);
                let disp = rotation_from_scaled_axis(ang * dt);
                ws.joint_rot = rotation_renormalize_fast(disp * ws.joint_rot);
                *ws.coords.at_mut(3) += vx * dt;
                *ws.coords.at_mut(4) += vy * dt;
                *ws.coords.at_mut(5) += vz * dt;
            }
        }
        // num_ang == 0: no-op.
    }

    // Silence dof_val unused warning — it will be used once we also support
    // setting coords directly (e.g. user-controlled kinematic DOFs).
    let _ = dof_val.0;
}
