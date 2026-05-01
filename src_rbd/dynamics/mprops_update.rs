//! Rigid-bodies world-space mass properties calculation.

use crate::math::Pose;
use crate::shaders::dynamics::{GpuSyncColliderPoses, GpuUpdateMprops};
use crate::shaders::dynamics::{LocalMassProperties, WorldMassProperties};
use khal::Shader;
use khal::backend::{GpuBackendError, GpuPass};
use vortx::tensor::Tensor;

/// GPU shader for updating the world-space mass properties of rigid-bodies.
#[derive(Shader)]
pub struct GpuMpropsUpdate {
    /// Compute pipeline for the world mass properties update kernel.
    pub update_mprops_kernel: GpuUpdateMprops,
}

impl GpuMpropsUpdate {
    /// Dispatches the mass properties update kernel.
    pub fn dispatch(
        &self,
        pass: &mut GpuPass,
        mprops: &mut Tensor<WorldMassProperties>,
        local_mprops: &Tensor<LocalMassProperties>,
        body_poses: &Tensor<Pose>,
        num_shapes: &Tensor<u32>,
        colliders_batch_capacity: &Tensor<u32>,
        num_bodies: u32,
        num_batches: u32,
    ) -> Result<(), GpuBackendError> {
        self.update_mprops_kernel.call(
            pass,
            [num_bodies, num_batches, 1],
            mprops,
            local_mprops,
            body_poses,
            num_shapes,
            colliders_batch_capacity,
        )
    }
}

/// GPU shader that recomputes the world pose of every collider from the body
/// world pose and the collider's body-local offset.
#[derive(Shader)]
pub struct GpuSyncColliderPosesShader {
    /// Compute pipeline for the collider-pose sync kernel.
    pub sync_kernel: GpuSyncColliderPoses,
}

impl GpuSyncColliderPosesShader {
    /// Dispatches the collider-pose sync kernel. Should be called after every
    /// step that mutates `body_poses` (integration, multibody forward
    /// kinematics) and before the broad-phase / narrow-phase / contact
    /// constraint conversion read collider world poses.
    pub fn dispatch(
        &self,
        pass: &mut GpuPass,
        body_poses: &Tensor<Pose>,
        collider_local_poses: &Tensor<Pose>,
        collider_world_poses: &mut Tensor<Pose>,
        num_shapes: &Tensor<u32>,
        colliders_batch_capacity: &Tensor<u32>,
        num_bodies: u32,
        num_batches: u32,
    ) -> Result<(), GpuBackendError> {
        self.sync_kernel.call(
            pass,
            [num_bodies, num_batches, 1],
            body_poses,
            collider_local_poses,
            collider_world_poses,
            num_shapes,
            colliders_batch_capacity,
        )
    }
}
