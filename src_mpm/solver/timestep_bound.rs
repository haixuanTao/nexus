//! Timestep bound estimation kernels.
//!
//! Computes a CFL-based upper bound on the simulation timestep to prevent
//! numerical instability. Uses material sound speed and particle velocities
//! to determine the maximum safe timestep.

use crate::cast_tensor;
use crate::grid::grid::GpuGrid;
use crate::mpm_shaders::models::default::GpuParticleModel;
use crate::mpm_shaders::solver::timestep_bound::{GpuEstimateTimestepBound, GpuResetTimestepBound, GpuTimestepBounds};
use crate::solver::{GpuParticleModelData, GpuParticles};
use khal::backend::{Backend, Encoder, GpuBackend, GpuBackendError, GpuPass};
use khal::{BufferUsages, Shader};
use vortx::tensor::Tensor;

/// GPU kernel responsible for computing an estimate on the maximum timestep that can
/// be taken by the simulation without blowing up.
///
/// Note that this is a best-effort estimate. It does not completely eliminate risks of divergence.
#[derive(Shader)]
pub struct WgTimestepBounds {
    reset_timestep_bound: GpuResetTimestepBound,
    estimate_timestep_bound: GpuEstimateTimestepBound,
}

impl WgTimestepBounds {
    /// Launches the timestep bounds estimation, and returns the estimated maximum timestep length.
    ///
    /// This method:
    /// 1. Resets the atomic minimum to the maximum representable value
    /// 2. Each particle thread computes its local timestep bound and atomically reduces
    /// 3. Reads back the result to CPU
    pub async fn compute_bounds<GpuModel: GpuParticleModelData>(
        &self,
        backend: &GpuBackend,
        grid: &GpuGrid,
        particles: &GpuParticles<GpuModel>,
        bounds: &mut Tensor<GpuTimestepBounds>,
        bounds_staging: &mut Tensor<GpuTimestepBounds>,
    ) -> Result<f32, GpuBackendError> {
        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass();
        self.launch(&mut pass, grid, particles, bounds)?;
        drop(pass);
        bounds_staging.copy_from_view(&mut encoder, &*bounds)?;
        backend.submit(encoder)?;

        let mut result = [GpuTimestepBounds::default()];
        backend
            .read_buffer(bounds_staging.buffer(), &mut result)
            .await?;
        Ok(result[0].computed_max_dt_as_uint as f32 / GpuTimestepBounds::FLOAT_TO_INT)
    }

    fn launch<GpuModel: GpuParticleModelData>(
        &self,
        pass: &mut GpuPass,
        grid: &GpuGrid,
        particles: &GpuParticles<GpuModel>,
        bounds: &mut Tensor<GpuTimestepBounds>,
    ) -> Result<(), GpuBackendError> {
        self.reset_timestep_bound.call(
            pass,
            1u32,
            bounds,
        )?;

        let len = particles.len() as u32;
        self.estimate_timestep_bound.call(
            pass,
            [len, 1, 1],
            &grid.meta,
            cast_tensor::<GpuModel, GpuParticleModel>(&particles.models),
            &particles.dynamics,
            &particles.gpu_len,
            bounds,
        )
    }
}
