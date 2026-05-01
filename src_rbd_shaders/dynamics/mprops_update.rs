//! Mass properties update compute shader kernels
//!
//! This module contains the actual GPU compute shader entry points for mass properties update.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::iter::StepRng;
use khal_std::macros::{spirv, spirv_bindgen};

use crate::Pose;

use super::body::{LocalMassProperties, WorldMassProperties};
use crate::utils::{Slice, SliceMut};

const WORKGROUP_SIZE: u32 = 64;

/// Updates world-space mass properties for all rigid bodies.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_update_mprops(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] mprops: &mut [WorldMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    local_mprops: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] num_colliders: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] colliders_batch_capacity: &u32,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y as usize;
    let colliders_start = batch_id * *colliders_batch_capacity as usize;

    let num_colliders = num_colliders.read(batch_id);
    let mut mprops = SliceMut(mprops, colliders_start);
    let local_mprops = Slice(local_mprops, colliders_start);
    let poses = Slice(poses, colliders_start);

    for i in StepRng::new(invocation_id.x..num_colliders, num_threads) {
        let idx = i as usize;
        let new_mprops = local_mprops.at(idx).to_world(poses.at(idx));
        mprops.write(idx, new_mprops);
    }
}

/// Recomputes the world pose of every collider from the body world pose and the
/// collider's body-local offset. Mirrors rapier's `RigidBody → Collider` pose
/// propagation: `collider.position = body.position * collider.position_wrt_parent`.
///
/// Run after each integration substep (and after multibody forward-kinematics)
/// so the broad-phase, narrow-phase and contact-to-constraint pipeline see
/// up-to-date collider world poses without needing to recompute the composition
/// at every read site.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_sync_collider_poses(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] body_poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    collider_local_poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] collider_world_poses: &mut [Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] num_colliders: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] colliders_batch_capacity: &u32,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y as usize;
    let colliders_start = batch_id * *colliders_batch_capacity as usize;
    let num_colliders = num_colliders.read(batch_id);

    let body_poses = Slice(body_poses, colliders_start);
    let collider_local_poses = Slice(collider_local_poses, colliders_start);
    let mut collider_world_poses = SliceMut(collider_world_poses, colliders_start);

    for i in StepRng::new(invocation_id.x..num_colliders, num_threads) {
        let idx = i as usize;
        collider_world_poses.write(
            idx,
            body_poses.read(idx) * collider_local_poses.read(idx),
        );
    }
}
