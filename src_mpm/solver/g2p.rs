//! Grid-to-Particle (G2P) transfer kernel.
//!
//! Interpolates grid velocities back to particles and updates particle velocity
//! gradients. This happens after grid forces have been applied.

use crate::grid::grid::{indirect_dispatch_tensor, GpuGrid};
use crate::mpm_shaders::solver::g2p::GpuG2p;
use crate::solver::{GpuMaterials, GpuParticleModelData, GpuParticles, GpuSimulationParams};
use nexus::dynamics::GpuBodySet;
use khal::backend::{GpuBackendError, GpuPass};
use khal::Shader;

/// GPU compute kernel for Grid-to-Particle (G2P) velocity interpolation.
///
/// Samples grid velocities at particle positions using quadratic B-spline weights
/// and updates particle velocity gradients for deformation tracking (APIC method).
#[derive(Shader)]
pub struct WgG2P {
    /// Compiled G2P compute shader.
    g2p: GpuG2p,
}

impl WgG2P {
    /// Launches the G2P kernel to update particle velocities from grid.
    ///
    /// # Arguments
    ///
    /// * `pass` - Compute pass to record commands into
    /// * `sim_params` - Simulation parameters (timestep, gravity)
    /// * `grid` - Source grid to interpolate from
    /// * `particles` - Target particles to update
    /// * `bodies` - Rigid bodies for velocity blending near contacts
    /// * `body_materials` - Boundary conditions per rigid body
    pub fn launch<GpuModel: GpuParticleModelData>(
        &self,
        pass: &mut GpuPass,
        sim_params: &GpuSimulationParams,
        grid: &GpuGrid,
        particles: &mut GpuParticles<GpuModel>,
        bodies: &GpuBodySet,
        body_materials: &GpuMaterials,
    ) -> Result<(), GpuBackendError> {
        self.g2p.call(
            pass,
            indirect_dispatch_tensor(&grid.indirect_n_g2p_p2g_groups),
            &sim_params.params,
            &grid.meta,
            &grid.hmap_entries,
            &grid.active_blocks,
            &grid.nodes,
            &particles.sorted_ids,
            &mut particles.positions,
            &mut particles.dynamics,
            &bodies.vels,
            &bodies.mprops,
            &body_materials.materials,
        )
    }
}
