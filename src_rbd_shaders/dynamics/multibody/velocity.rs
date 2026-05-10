//! Velocity propagation (rapier's `update_dynamics` velocity phase).
//!
//! Computes per-link world-space `joint_velocity` and total `rb_vels` by walking
//! links parent-before-child, so that the Coriolis assembly can read them.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

use parry::math::VectorExt;
use crate::dynamics::body::{LocalMassProperties, Velocity};
use crate::utils::{Slice, SliceMut};
use crate::{ANG_DIM, AngVector, DIM, Vector, gcross_av};

use super::types::{MultibodyInfo, MultibodyLinkStatic, MultibodyLinkWorkspace};

/// Body-local velocity contributed by this joint, reading the joint's free-DOF
/// velocities directly from `vel_slice[assembly_id..]` rather than via a stack
/// array. The stack version forces a `[f32; MAX_JOINT_DOFS]` Function-storage
/// variable which spills to private memory.
#[inline]
fn jacobian_mul_coordinates(
    locked_axes: u32,
    assembly_id: u32,
    vel_slice: &Slice<f32>,
) -> (Vector, AngVector) {
    let mut lin = Vector::ZERO;
    #[cfg(feature = "dim3")]
    let mut ang = AngVector::ZERO;
    #[cfg(feature = "dim2")]
    let mut ang: AngVector = 0.0;
    let mut curr = 0u32;

    for i in 0..DIM {
        if (locked_axes & (1 << i)) == 0 {
            let v = vel_slice.read((assembly_id + curr) as usize);
            lin += Vector::ith(i as usize, v);
            curr += 1;
        }
    }

    let ang_locked = (locked_axes >> DIM) & ((1 << ANG_DIM) - 1);
    let num_ang = ANG_DIM - ang_locked.count_ones();
    if num_ang == 1 {
        #[cfg(feature = "dim3")]
        {
            let dof_id = (!ang_locked & 0x7).trailing_zeros();
            let v = vel_slice.read((assembly_id + curr) as usize);
            ang += Vector::ith(dof_id as usize, v);
        }
        #[cfg(feature = "dim2")]
        {
            let v = vel_slice.read((assembly_id + curr) as usize);
            ang += v;
        }
    } else if num_ang == 3 {
        #[cfg(feature = "dim3")]
        {
            let vx = vel_slice.read((assembly_id + curr) as usize);
            let vy = vel_slice.read((assembly_id + curr + 1) as usize);
            let vz = vel_slice.read((assembly_id + curr + 2) as usize);
            ang += AngVector::new(vx, vy, vz);
        }
    }
    (lin, ang)
}

/// Propagate link velocities parent-before-child. Mirrors rapier's:
///
/// ```text
///   let joint_velocity = link.joint.jacobian_mul_coordinates(&velocities[link.assembly_id..]);
///   link.joint_velocity = joint_velocity.transformed(
///       &(parent_link.local_to_world.rotation * link.joint.data.local_frame1.rotation));
///   let mut new_rb_vels = parent_rb.vels + link.joint_velocity;
///   new_rb_vels.linvel += parent_rb.vels.angvel.gcross(shift);
///   new_rb_vels.linvel += link.joint_velocity.angvel.gcross(link.shift23);
///   rb.vels = new_rb_vels;
/// ```
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_update_velocities(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    links_workspace: &mut [MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] links_local_mprops: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] dof_state: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 6)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 7)] links_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 8)] dof_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let links_start = batch_id * *links_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let num_links = mb.num_links;
    let first_link_global = links_start + mb.first_link as usize;
    let gen_base = dof_start + mb.first_dof as usize;

    let stat_slice = Slice(links_static, first_link_global);
    let mut ws_slice = SliceMut(links_workspace, first_link_global);
    let local_mprops_slice = Slice(links_local_mprops, first_link_global);
    let vel_slice = Slice(dof_state, gen_base);

    for k in 0..num_links {
        let k_usize = k as usize;
        let stat = stat_slice.read(k_usize);

        // Joint-local body velocity contribution. Reads `dof_velocities` directly
        // — no `[f32; MAX_JOINT_DOFS]` Function-storage scratch.
        let (jv_local_lin, jv_local_ang) = jacobian_mul_coordinates(
            stat.data.locked_axes,
            stat.assembly_id,
            &vel_slice,
        );

        let (joint_velocity, rb_vels) = if k == 0 {
            // Root: joint velocity already in world frame.
            let jv = Velocity::new(jv_local_lin, jv_local_ang);
            (jv, jv)
        } else {
            let parent_id = stat.parent_link_id as usize;
            // Read the parent's needed fields by-reference; avoid a full
            // workspace copy.
            let parent_ws = ws_slice.at(parent_id);
            let parent_to_world_rot = parent_ws.local_to_world.rotation;
            let parent_world_com_pose = parent_ws.local_to_world;
            let parent_rb_lin = parent_ws.rb_vels.linear;
            let parent_rb_ang = parent_ws.rb_vels.angular;

            let parent_lmp = local_mprops_slice.read(parent_id);
            let transform_rot = parent_to_world_rot * stat.data.local_frame_a.rotation;

            #[cfg(feature = "dim3")]
            let joint_velocity = Velocity::new(
                transform_rot * jv_local_lin,
                transform_rot * jv_local_ang,
            );
            #[cfg(feature = "dim2")]
            let joint_velocity = Velocity::new(transform_rot * jv_local_lin, jv_local_ang);

            // Read self fields needed for the shift correction without
            // materialising the whole struct.
            let (self_local_to_world, self_shift23) = {
                let ws_ref = ws_slice.at(k_usize);
                (ws_ref.local_to_world, ws_ref.shift23)
            };

            let lmp = local_mprops_slice.read(k_usize);
            let world_com = self_local_to_world * lmp.com;
            let parent_world_com = parent_world_com_pose * parent_lmp.com;
            let shift = world_com - parent_world_com;

            let mut new_lin = parent_rb_lin + joint_velocity.linear;
            let new_ang = parent_rb_ang + joint_velocity.angular;
            new_lin += gcross_av(parent_rb_ang, shift);
            new_lin += gcross_av(joint_velocity.angular, self_shift23);

            (joint_velocity, Velocity::new(new_lin, new_ang))
        };

        // Field-targeted writes: avoid the round-trip of the full ~240 B struct.
        let link_mut = ws_slice.at_mut(k_usize);
        link_mut.joint_velocity = joint_velocity;
        link_mut.rb_vels = rb_vels;
    }
}
