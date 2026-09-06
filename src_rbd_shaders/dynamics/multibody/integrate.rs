//! Integrate kernel (semi-implicit Euler).
//!
//! Advances generalized velocities, then each link's `coords` / `joint_rot`.
//! After this pass, callers are expected to re-run forward kinematics to
//! refresh link poses.

use khal_std::glamx::UVec3;
use glamx::Vec4;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

#[cfg(feature = "dim2")]
use crate::rotation_from_angle;
use crate::utils::BatchIndices;
use crate::{ANG_DIM, DIM};
#[cfg(feature = "dim3")]
use crate::{Vector, rotation_from_scaled_axis, rotation_renormalize_fast};
#[cfg(feature = "dim3")]
use parry::math::VectorExt;

use super::types::{MultibodyInfo, MultibodyLinkStatic};
use super::ws_soa::{WS_JOINT_ROT, WsAddr, ws_coord, ws_rot, ws_set_coord, ws_set_rot};

/// Update generalized velocities: `v += a · dt`.
///
/// Split out from the position-update half so that joint-limit / motor
/// constraints can run in between (rapier's order: velocity update → constraint
/// solver → position update).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_integrate_velocities(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dof_state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] gen_accelerations: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] dt_uniform: &f32,
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batch_ids: &BatchIndices,
) {
    // Flattened (multibody, batch) grid — see `BatchIndices::num_batches`.
    let num_mb = batch_ids.multibodies_len;
    if invocation_id.x >= num_mb * batch_ids.num_batches {
        return;
    }
    let (batch_id, mb_idx) = crate::div_rem_nz(invocation_id.x, num_mb);
    let dt = *dt_uniform;

    let mb = batch_ids
        .ib(batch_id, multibody_info)
        .read(mb_idx as usize);

    let mut dof_vel = batch_ids
        .ib_mut(batch_id, dof_state)
        .offset(mb.first_dof as usize);
    let acc = batch_ids
        .ib(batch_id, gen_accelerations)
        .offset(mb.first_dof as usize);

    for d in 0..mb.ndofs {
        let di = d as usize;
        dof_vel[di] += acc[di] * dt;
    }
}

/// Per-DOF joint velocity limit (dof_state section 4, 1e30 = none): clamp the generalized velocities
/// the previous step left behind, before anything of the new step reads them (Coriolis terms, the
/// force-based motor PD, the velocity integration). Dispatched once at the start of each step, which is
/// the after-step clamp of a host-side PD loop. A generalized-velocity clamp has no reaction on the
/// parent link, so it is deliberately NOT applied inside the step (see `gpu_mb_integrate`, section 5).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_clamp_dof_velocities(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dof_state: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] batch_ids: &BatchIndices,
) {
    let num_mb = batch_ids.multibodies_len;
    if invocation_id.x >= num_mb * batch_ids.num_batches {
        return;
    }
    let (batch_id, mb_idx) = crate::div_rem_nz(invocation_id.x, num_mb);
    let mb = batch_ids
        .ib(batch_id, multibody_info)
        .read(mb_idx as usize);
    let mut dof_vel = batch_ids
        .ib_mut(batch_id, dof_state)
        .offset(mb.first_dof as usize);
    let lim_off = 4 * batch_ids.dof_batch_capacity as usize;
    for d in 0..mb.ndofs {
        let di = d as usize;
        let l = dof_vel[lim_off + di];
        let v = dof_vel[di];
        dof_vel[di] = v.max(-l).min(l);
    }
}

/// One link's semi-implicit position update — the body of
/// [`gpu_mb_integrate`]'s loop, shared with the fused
/// integrate+dynamics-pre entries (`compute_dynamics_pre.rs`), where the
/// links of each multibody are strided across its packed lanes instead of
/// walked serially. Order-independent across links.
#[inline(always)]
pub(super) fn integrate_link(
    stat_slice: &crate::utils::ISlice<'_, MultibodyLinkStatic>,
    links_workspace: &mut [Vec4],
    wa: WsAddr,
    k: u32,
    dof_vel: &crate::utils::ISlice<'_, f32>,
    dt: f32,
) {
    let stat = stat_slice.read(k as usize);
    let locked = stat.data.locked_axes;
    let aid = stat.assembly_id as usize;

    // Free linear DOFs first, in axis order.
    let mut curr_free = 0u32;
    for i in 0..DIM {
        if (locked & (1 << i)) == 0 {
            let v = dof_vel[aid + curr_free as usize];
            let cur = ws_coord(links_workspace, wa, k, i);
            ws_set_coord(links_workspace, wa, k, i, cur + v * dt);
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
            let new = ws_coord(links_workspace, wa, k, idx) + v * dt;
            ws_set_coord(links_workspace, wa, k, idx, new);
            ws_set_rot(
                links_workspace,
                wa,
                k,
                WS_JOINT_ROT,
                rotation_from_scaled_axis(Vector::ith(dof_id as usize, new)),
            );
        }
        #[cfg(feature = "dim2")]
        {
            let v = dof_vel[aid + curr_free as usize];
            let new = ws_coord(links_workspace, wa, k, DIM) + v * dt;
            ws_set_coord(links_workspace, wa, k, DIM, new);
            ws_set_rot(links_workspace, wa, k, WS_JOINT_ROT, rotation_from_angle(new));
        }
    } else if num_ang == 3 {
        #[cfg(feature = "dim3")]
        {
            let vx = dof_vel[aid + curr_free as usize];
            let vy = dof_vel[aid + (curr_free + 1) as usize];
            let vz = dof_vel[aid + (curr_free + 2) as usize];
            let ang = Vector::new(vx, vy, vz);
            let disp = rotation_from_scaled_axis(ang * dt);
            let jr = ws_rot(links_workspace, wa, k, WS_JOINT_ROT);
            ws_set_rot(
                links_workspace,
                wa,
                k,
                WS_JOINT_ROT,
                rotation_renormalize_fast(disp * jr),
            );
            let c3 = ws_coord(links_workspace, wa, k, 3);
            ws_set_coord(links_workspace, wa, k, 3, c3 + vx * dt);
            let c4 = ws_coord(links_workspace, wa, k, 4);
            ws_set_coord(links_workspace, wa, k, 4, c4 + vy * dt);
            let c5 = ws_coord(links_workspace, wa, k, 5);
            ws_set_coord(links_workspace, wa, k, 5, c5 + vz * dt);
        }
    }
    // num_ang == 0: no-op.
}

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_integrate(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] links_workspace: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dof_values: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] dof_state: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] dt_uniform: &f32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] batch_ids: &BatchIndices,
) {
    // Flattened (multibody, batch) grid — see `BatchIndices::num_batches`.
    let num_mb = batch_ids.multibodies_len;
    if invocation_id.x >= num_mb * batch_ids.num_batches {
        return;
    }
    let (batch_id, mb_idx) = crate::div_rem_nz(invocation_id.x, num_mb);
    let dt = *dt_uniform;

    let mb = batch_ids
        .ib(batch_id, multibody_info)
        .read(mb_idx as usize);
    let num_links = mb.num_links;

    let stat_slice = batch_ids
        .ib(batch_id, links_static)
        .offset(mb.first_link as usize);
    let wa = WsAddr::new(mb.first_link as usize, batch_ids.num_batches, batch_id);
    let dof_val = batch_ids
        .ib_mut(batch_id, dof_values)
        .offset(mb.first_dof as usize);
    // Optional second joint velocity clamp (dof_state section 5, 1e30 = none, the default) on the
    // solver-corrected velocities right before they integrate the positions. Off by default: like the
    // section-4 clamp it is a generalized-velocity clamp without a reaction on the parent link, and on a
    // floating base it pumps angular momentum into the base at every substep (measured: 4x the invalid-state
    // terminations under random actions on the G1). The section-4 clamp alone matches the after-step clamp of
    // the host-side PD (the velocity carried into the next substep is capped; the motion is not).
    {
        let lim_off = 5 * batch_ids.dof_batch_capacity as usize;
        let mut dv = batch_ids
            .ib_mut(batch_id, dof_state)
            .offset(mb.first_dof as usize);
        for d in 0..mb.ndofs {
            let di = d as usize;
            let l = dv[lim_off + di];
            let v = dv[di];
            dv[di] = v.max(-l).min(l);
        }
    }
    let dof_vel = batch_ids
        .ib(batch_id, &*dof_state)
        .offset(mb.first_dof as usize);

    // Per-link coord / joint_rot update (uses the already-corrected `dof_velocities`).
    //
    // Only `coords` (≤ 24 B) and `joint_rot` (16 B) are modified. We mutate them
    // in place through `&mut ws_slice[k]` so SPIR-V emits field-targeted stores
    // instead of a whole `MultibodyLinkWorkspace` round-trip (~240 B in 3D).
    for k in 0..num_links {
        integrate_link(&stat_slice, links_workspace, wa, k, &dof_vel, dt);
    }

    // Silence dof_val unused warning — it will be used once we also support
    // setting coords directly (e.g. user-controlled kinematic DOFs).
    let _ = dof_val.buf;
}
