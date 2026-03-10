//! Mass properties update compute shader kernels
//!
//! This module contains the actual GPU compute shader entry points for mass properties update.

use khal_derive::spirv_bindgen;
use spirv_std::glam::UVec3;
use spirv_std::spirv;

use vortx_shaders::utils::step::StepRng;

use crate::Pose;

use super::body::{update_mprops, LocalMassProperties, WorldMassProperties};
use crate::MaybeIndexUnchecked;
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
        let new_mprops = update_mprops(poses.read(idx), local_mprops.at(idx));
        mprops.write(idx, new_mprops);
    }
}
