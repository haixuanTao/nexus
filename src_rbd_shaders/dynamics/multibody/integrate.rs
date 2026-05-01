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

use parry::math::VectorExt;
use crate::utils::{Slice, SliceMut};
use crate::{ANG_DIM, DIM};
#[cfg(feature = "dim2")]
use crate::rotation_from_angle;
#[cfg(feature = "dim3")]
use crate::{Vector, rotation_from_scaled_axis, rotation_renormalize_fast};

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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dof_velocities: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] gen_accelerations: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] num_multibodies: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] dt_buf: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] dof_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }
    let dt = dt_buf.read(0);

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;
    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let gen_base = dof_start + mb.first_dof as usize;

    let mut dof_vel = SliceMut(dof_velocities, gen_base);
    let acc = Slice(gen_accelerations, gen_base);

    for d in 0..mb.ndofs {
        let di = d as usize;
        let cur = dof_vel.read(di);
        dof_vel.write(di, cur + acc.read(di) * dt);
    }
}

#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_integrate(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    links_workspace: &mut [MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dof_values: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] dof_velocities: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] num_multibodies: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] dt_buf: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 7)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 8)] links_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 9)] dof_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }
    let dt = dt_buf.read(0);

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let links_start = batch_id * *links_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let num_links = mb.num_links;
    let first_link_global = links_start + mb.first_link as usize;
    let gen_base = dof_start + mb.first_dof as usize;

    let stat_slice = Slice(links_static, first_link_global);
    let mut ws_slice = SliceMut(links_workspace, first_link_global);
    let dof_val = SliceMut(dof_values, gen_base);
    let dof_vel = Slice(dof_velocities, gen_base);

    // Per-link coord / joint_rot update (uses the already-corrected `dof_velocities`).
    for k in 0..num_links {
        let k_usize = k as usize;
        let stat = stat_slice.read(k_usize);
        let mut ws = ws_slice.read(k_usize);
        let locked = stat.data.locked_axes;
        let aid = stat.assembly_id as usize;

        // Free linear DOFs first, in axis order.
        let mut curr_free = 0u32;
        for i in 0..DIM {
            if (locked & (1 << i)) == 0 {
                let v = dof_vel.read(aid + curr_free as usize);
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
                let v = dof_vel.read(aid + curr_free as usize);
                let idx = 3 + dof_id;
                let new = ws.coords.read(idx as usize) + v * dt;
                ws.coords.write(idx as usize, new);
                ws.joint_rot = rotation_from_scaled_axis(Vector::ith(dof_id as usize, new));
            }
            #[cfg(feature = "dim2")]
            {
                let v = dof_vel.read(aid + curr_free as usize);
                let new = ws.coords.read(DIM as usize) + v * dt;
                ws.coords.write(DIM as usize, new);
                ws.joint_rot = rotation_from_angle(new);
            }
        } else if num_ang == 3 {
            #[cfg(feature = "dim3")]
            {
                let vx = dof_vel.read(aid + curr_free as usize);
                let vy = dof_vel.read(aid + (curr_free + 1) as usize);
                let vz = dof_vel.read(aid + (curr_free + 2) as usize);
                let ang = Vector::new(vx, vy, vz);
                let disp = rotation_from_scaled_axis(ang * dt);
                ws.joint_rot = rotation_renormalize_fast(disp * ws.joint_rot);
                *ws.coords.at_mut(3) += vx * dt;
                *ws.coords.at_mut(4) += vy * dt;
                *ws.coords.at_mut(5) += vz * dt;
            }
        }
        // num_ang == 0: no-op.

        ws_slice.write(k_usize, ws);
    }

    // Silence dof_val unused warning — it will be used once we also support
    // setting coords directly (e.g. user-controlled kinematic DOFs).
    let _ = dof_val.0;
}
