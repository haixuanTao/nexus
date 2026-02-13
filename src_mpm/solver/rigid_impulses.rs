//! Impulse accumulation and application for MPM-rigid body coupling.

use crate::grid::grid::GpuGrid;
use crate::mpm_shaders::solver::rigid_impulses::{
    GpuRigidImpulsesUpdate, GpuUpdateWorldMassProperties, IntegerImpulse,
};
use crate::solver::GpuSimulationParams;
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};
use khal::{BufferUsages, Shader};
use nexus::dynamics::GpuBodySet;
use vortx::tensor::Tensor;

/// GPU kernels for computing and applying impulses to rigid bodies from MPM.
///
/// Accumulates forces from MPM particles and applies them as impulses to
/// coupled rigid bodies for two-way interaction.
#[derive(Shader)]
pub struct WgRigidImpulses {
    /// Kernel for computing and applying impulses.
    update: GpuRigidImpulsesUpdate,
    /// Kernel for updating world-space mass properties.
    update_world_mass_properties: GpuUpdateWorldMassProperties,
}

/// GPU buffers for storing impulses from MPM to rigid bodies.
pub struct GpuImpulses {
    /// Per-timestep incremental impulses (uses atomic integer operations).
    pub incremental_impulses: Tensor<IntegerImpulse>,
}

impl GpuImpulses {
    /// Creates impulse buffers for rigid bodies.
    ///
    /// Allocates space for up to 16 bodies (CPIC limitation).
    pub fn new(backend: &GpuBackend) -> Result<Self, GpuBackendError> {
        const MAX_BODY_COUNT: usize = 16; // CPIC doesn't support more.
        let impulses = [IntegerImpulse::default(); MAX_BODY_COUNT];
        Ok(Self {
            incremental_impulses: Tensor::vector(backend, &impulses, BufferUsages::STORAGE)?,
        })
    }
}

impl WgRigidImpulses {
    /// Computes and applies impulses to rigid bodies from MPM grid.
    ///
    /// # Arguments
    ///
    /// * `pass` - Compute pass
    /// * `grid` - Grid containing accumulated momentum
    /// * `sim_params` - Simulation parameters
    /// * `impulses` - Impulse buffers to write
    /// * `bodies` - Target rigid bodies
    pub fn launch(
        &self,
        pass: &mut GpuPass,
        grid: &GpuGrid,
        sim_params: &GpuSimulationParams,
        impulses: &mut GpuImpulses,
        bodies: &mut GpuBodySet,
    ) -> Result<(), GpuBackendError> {
        if bodies.is_empty() {
            return Ok(());
        }

        self.update.call(
            pass,
            1u32,
            &sim_params.params,
            &grid.meta,
            &bodies.local_mprops,
            &mut bodies.poses,
            &mut bodies.vels,
            &mut bodies.mprops,
            &mut impulses.incremental_impulses,
        )
    }

    /// Updates world-space mass properties for rigid bodies.
    ///
    /// Transforms local inertia tensors to world coordinates based on current poses.
    ///
    /// # Arguments
    ///
    /// * `pass` - Compute pass
    /// * `impulses` - Impulse buffers (updated with center of mass)
    /// * `bodies` - Bodies to update
    pub fn launch_update_world_mass_properties(
        &self,
        pass: &mut GpuPass,
        impulses: &mut GpuImpulses,
        bodies: &mut GpuBodySet,
    ) -> Result<(), GpuBackendError> {
        if bodies.is_empty() {
            return Ok(());
        }

        let len = bodies.len();
        self.update_world_mass_properties.call(
            pass,
            [len, 1, 1],
            &bodies.poses,
            &bodies.local_mprops,
            &mut bodies.mprops,
            &mut impulses.incremental_impulses,
        )
    }
}
