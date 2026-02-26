//! GPU readback preparation kernel and associated data structures.
//!
//! Computes per-particle render data on the GPU, reducing the amount of data
//! transferred back to the CPU compared to reading raw positions and dynamics.

use crate::grid::grid::GpuGrid;
use crate::mpm_shaders::solver::prep_readback::{GpuPrepReadback, GpuPrepReadbackRigid};
pub use crate::mpm_shaders::solver::prep_readback::{ReadbackData, RenderConfig};
use crate::solver::particle_model::GpuParticleModelData;
use crate::solver::{GpuParticles, GpuRigidParticles, GpuSimulationParams};
use glamx::Vec4;
use khal::backend::{Encoder, GpuBackend, GpuBackendError, GpuEncoder, GpuTimestamps};
use khal::{BufferUsages, Shader};
use vortx::tensor::Tensor;

/// GPU compute kernel for preparing per-particle readback data.
///
/// Runs a compute shader that transforms particle positions and dynamics
/// into compact `ReadbackData` suitable for rendering, then copies the
/// result to a staging buffer for CPU readback.
#[derive(Shader)]
pub struct WgPrepReadback {
    prep_readback: GpuPrepReadback,
    prep_readback_rigid: GpuPrepReadbackRigid,
}

/// GPU-resident buffers for the readback preparation pipeline.
///
/// Contains the render configuration, base colors, and output buffers
/// for the readback shader.
pub struct GpuReadbackData {
    /// Render mode configuration (uniform, written by CPU).
    pub mode: Tensor<RenderConfig>,
    /// Per-particle base colors (uploaded once, read by shader).
    pub base_colors: Tensor<Vec4>,
    /// Shader output buffer (written by GPU, source for staging copy).
    pub instances: Tensor<ReadbackData>,
    /// Staging buffer for CPU readback (MAP_READ).
    pub instances_staging: Tensor<ReadbackData>,
    /// Per-rigid-particle base colors.
    pub rigid_base_colors: Tensor<Vec4>,
    /// Rigid particle shader output buffer.
    pub rigid_instances: Tensor<ReadbackData>,
    /// Staging buffer for rigid particle CPU readback (MAP_READ).
    pub rigid_instances_staging: Tensor<ReadbackData>,
    /// Rigid particle count uniform for the shader.
    pub rigid_len: Tensor<u32>,
}

impl GpuReadbackData {
    /// Creates new readback data buffers for the given number of particles.
    pub fn new(
        backend: &GpuBackend,
        num_particles: usize,
        num_rigid_particles: usize,
        mode: u32,
    ) -> Result<Self, GpuBackendError> {
        let palette = [
            Vec4::new(124.0 / 255.0, 144.0 / 255.0, 1.0, 1.0),
            Vec4::new(8.0 / 255.0, 144.0 / 255.0, 1.0, 1.0),
            Vec4::new(124.0 / 255.0, 7.0 / 255.0, 1.0, 1.0),
            Vec4::new(124.0 / 255.0, 144.0 / 255.0, 7.0 / 255.0, 1.0),
            Vec4::new(200.0 / 255.0, 37.0 / 255.0, 1.0, 1.0),
            Vec4::new(124.0 / 255.0, 230.0 / 255.0, 25.0 / 255.0, 1.0),
        ];
        let base_colors: Vec<Vec4> = (0..num_particles)
            .map(|i| palette[i % palette.len()])
            .collect();
        let rigid_base_colors: Vec<Vec4> = (0..num_rigid_particles)
            .map(|i| palette[i % palette.len()])
            .collect();

        // Use at least 1 element for GPU buffers to avoid zero-sized allocations.
        let rigid_buf_len = num_rigid_particles.max(1) as u32;

        Ok(Self {
            mode: Tensor::scalar(
                backend,
                RenderConfig { mode },
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            )?,
            base_colors: Tensor::vector(backend, base_colors, BufferUsages::STORAGE)?,
            instances: Tensor::vector_uninit(
                backend,
                num_particles as u32,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            )?,
            instances_staging: Tensor::vector_uninit(
                backend,
                num_particles as u32,
                BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            )?,
            rigid_base_colors: Tensor::vector(backend, rigid_base_colors, BufferUsages::STORAGE)?,
            rigid_instances: Tensor::vector_uninit(
                backend,
                rigid_buf_len,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            )?,
            rigid_instances_staging: Tensor::vector_uninit(
                backend,
                rigid_buf_len,
                BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            )?,
            rigid_len: Tensor::scalar(
                backend,
                num_rigid_particles as u32,
                BufferUsages::STORAGE | BufferUsages::UNIFORM,
            )?,
        })
    }

    /// Recreates all buffers for a new particle count.
    pub fn resize(
        &mut self,
        backend: &GpuBackend,
        num_particles: usize,
        num_rigid_particles: usize,
        mode: u32,
    ) -> Result<(), GpuBackendError> {
        *self = Self::new(backend, num_particles, num_rigid_particles, mode)?;
        Ok(())
    }
}

impl WgPrepReadback {
    /// Launches the readback preparation shader and copies results to staging.
    ///
    /// This runs a compute pass that writes `ReadbackData` into `instances`,
    /// then copies `instances` → `instances_staging` for CPU readback.
    /// Also dispatches the rigid particle readback shader if there are rigid particles.
    pub fn launch<GpuModel: GpuParticleModelData>(
        &self,
        encoder: &mut GpuEncoder,
        timestamps: Option<&mut GpuTimestamps>,
        readback: &mut GpuReadbackData,
        sim_params: &GpuSimulationParams,
        grid: &GpuGrid,
        particles: &GpuParticles<GpuModel>,
        rigid_particles: &GpuRigidParticles,
    ) -> Result<(), GpuBackendError> {
        let len = particles.len() as u32;
        let rigid_len = rigid_particles.len() as u32;
        {
            let mut pass = encoder.begin_pass("prep-readback", timestamps);
            self.prep_readback.call(
                &mut pass,
                [len, 1, 1],
                &mut readback.instances,
                &particles.positions,
                &particles.kinematics,
                &particles.def_grad,
                &particles.properties,
                &grid.meta,
                &readback.base_colors,
                &sim_params.params,
                &readback.mode,
                &particles.gpu_len,
            )?;

            if rigid_len > 0 {
                self.prep_readback_rigid.call(
                    &mut pass,
                    [rigid_len, 1, 1],
                    &mut readback.rigid_instances,
                    &rigid_particles.sample_points,
                    &readback.rigid_base_colors,
                    &grid.meta,
                    &readback.rigid_len,
                )?;
            }
        }
        readback
            .instances_staging
            .copy_from_view(encoder, &readback.instances)?;
        if rigid_len > 0 {
            readback
                .rigid_instances_staging
                .copy_from_view(encoder, &readback.rigid_instances)?;
        }
        Ok(())
    }
}
