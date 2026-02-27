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

const WORKGROUP_SIZE: u32 = 64;

/// Updates world-space mass properties for a rigid body.
///
/// For each body:
/// 1. Transform center of mass from local to world space
/// 2. Transform inertia tensor to world orientation (3D only)
///
/// In 2D: Only COM needs transformation (inertia is scalar)
/// In 3D: Both COM and inertia tensor need transformation
pub fn update_body_mprops(
    body_id: usize,
    poses: &[Pose],
    local_mprops: &[LocalMassProperties],
    mprops: &mut [WorldMassProperties],
) {
    // Transform mass properties from local to world space
    // - Transforms COM position by pose
    // - Rotates inertia tensor to world orientation (3D)
    let new_mprops = update_mprops(poses.read(body_id), local_mprops.at(body_id));

    // Write updated world-space mass properties
    mprops.write(body_id, new_mprops);
}

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
) {
    // Total number of threads across all workgroups
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    let num_bodies = poses.len() as u32;

    // Grid-stride loop: each thread processes multiple bodies if necessary
    for i in StepRng::new(invocation_id.x..num_bodies, num_threads) {
        let idx = i as usize;
        // Transform mass properties from local to world space
        // - Transforms COM position by pose
        // - Rotates inertia tensor to world orientation (3D)
        let new_mprops = update_mprops(poses.read(idx), local_mprops.at(idx));

        // Write updated world-space mass properties
        mprops.write(idx, new_mprops);
    }
}
