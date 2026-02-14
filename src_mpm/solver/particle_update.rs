//! Particle state update kernel.
//!
//! Updates particle positions, deformation gradients, and material state after
//! grid velocities have been transferred back to particles.

use crate::cast_tensor_mut;
use crate::grid::grid::GpuGrid;
use crate::mpm_shaders::models::default::GpuParticleModel;
use crate::mpm_shaders::solver::particle_update::GpuParticleUpdate;
use crate::solver::particle_model::GpuParticleModelData;
use crate::solver::{GpuParticles, GpuSimulationParams};
use khal::Shader;
use khal::backend::{GpuBackendError, GpuPass};

/// GPU compute kernel for updating particle state.
///
/// Integrates particle positions using updated velocities, updates deformation
/// gradients, and applies constitutive models (elasticity, plasticity).
#[derive(Shader)]
pub struct WgParticleUpdate {
    /// Compiled particle update compute shader.
    particle_update: GpuParticleUpdate,
}

impl WgParticleUpdate {
    /// Launches the particle update kernel.
    ///
    /// # Arguments
    ///
    /// * `pass` - Compute pass to record commands into
    /// * `sim_params` - Simulation parameters (timestep, gravity)
    /// * `grid` - Grid metadata for boundary conditions
    /// * `particles` - Particles to update (positions, deformations, material state)
    pub fn launch<GpuModel: GpuParticleModelData>(
        &self,
        pass: &mut GpuPass,
        sim_params: &GpuSimulationParams,
        grid: &GpuGrid,
        particles: &mut GpuParticles<GpuModel>,
    ) -> Result<(), GpuBackendError> {
        let len = particles.len() as u32;
        self.particle_update.call(
            pass,
            [len, 1, 1],
            &sim_params.params,
            &grid.meta,
            cast_tensor_mut::<GpuModel, GpuParticleModel>(&mut particles.models),
            &mut particles.positions,
            &mut particles.kinematics,
            &particles.cdf,
            &mut particles.material_state,
            &particles.gpu_len,
        )
    }
}
