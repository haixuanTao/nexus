//! Particle-to-Grid (P2G) transfer kernel.
//!
//! Transfers particle mass, momentum, and forces to nearby grid nodes using
//! interpolation weights. This is the first major step of each MPM timestep.

use crate::cast_tensor_mut;
use crate::grid::grid::{GpuGrid, indirect_dispatch_tensor};
use crate::mpm_shaders::solver::p2g::{GpuP2g, IntegerImpulseAtomic};
use crate::solver::{GpuImpulses, GpuMaterials, GpuParticleModelData, GpuParticles};
use khal::Shader;
use khal::backend::{GpuBackendError, GpuPass};
use nexus::dynamics::GpuBodySet;

/// GPU compute kernel for Particle-to-Grid (P2G) momentum transfer.
///
/// Rasterizes particle mass and momentum onto the background grid using quadratic
/// B-spline interpolation. Also handles impulse accumulation for rigid body coupling.
#[derive(Shader)]
pub struct WgP2G {
    /// Compiled P2G compute shader.
    p2g: GpuP2g,
}

impl WgP2G {
    /// Launches the P2G kernel to transfer particle data to grid nodes.
    ///
    /// # Arguments
    ///
    /// * `pass` - Compute pass to record commands into
    /// * `grid` - Target grid to write momentum into
    /// * `particles` - Source particles to read from
    /// * `impulses` - Impulse buffers for rigid body coupling
    /// * `bodies` - Rigid bodies for coupling
    /// * `body_materials` - Boundary conditions per rigid body
    pub fn launch<GpuModel: GpuParticleModelData>(
        &self,
        pass: &mut GpuPass,
        grid: &mut GpuGrid,
        particles: &GpuParticles<GpuModel>,
        impulses: &mut GpuImpulses,
        bodies: &GpuBodySet,
        body_materials: &GpuMaterials,
    ) -> Result<(), GpuBackendError> {
        self.p2g.call(
            pass,
            indirect_dispatch_tensor(&grid.indirect_n_g2p_p2g_groups),
            &grid.meta,
            &grid.hmap_entries,
            &grid.active_blocks,
            &grid.nodes_linked_lists,
            particles.node_linked_lists(),
            particles.positions(),
            particles.kinematics(),
            particles.cdf(),
            &mut grid.nodes,
            &bodies.vels,
            cast_tensor_mut::<_, IntegerImpulseAtomic>(&mut impulses.incremental_impulses),
            &body_materials.materials,
        )
    }
}
