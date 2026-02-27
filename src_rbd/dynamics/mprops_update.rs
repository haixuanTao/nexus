//! Rigid-bodies world-space mass properties calculation.

use crate::math::Pose;
use crate::shaders::dynamics::GpuUpdateMprops;
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
        poses: &Tensor<Pose>,
        num_bodies: u32,
    ) -> Result<(), GpuBackendError> {
        self.update_mprops_kernel
            .call(pass, num_bodies, mprops, local_mprops, poses)
    }
}
