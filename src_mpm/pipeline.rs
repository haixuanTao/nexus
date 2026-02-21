//! High-level MPM simulation pipeline orchestration.
//!
//! This module provides the main entry point for running MPM simulations. The pipeline
//! coordinates the execution of all MPM algorithm stages on the GPU.

use crate::grid::grid::{GpuGrid, WgGrid};
use nexus::utils::{PrefixSumWorkspace, GpuPrefixSum};
use crate::grid::sort::WgSort;
use crate::solver::{
    BoundaryCondition, BoundaryConditionExt, GpuImpulses, GpuMaterials, GpuParticleModelData,
    GpuParticles, GpuRigidParticles, GpuSimulationParams, GpuTimestepBounds, Particle,
    SimulationParams, WgG2P, WgG2PCdf, WgGridUpdate, WgGridUpdateCdf, WgIntegrateBodies, WgP2G,
    WgP2GCdf, WgParticleUpdate, WgRigidParticleUpdate, WgTimestepBounds,
};
use khal::backend::{Backend, Encoder, GpuBackend, GpuBackendError, GpuEncoder, GpuTimestamps};
use khal::{BufferUsages, Shader};
use nexus::dynamics::GpuBodySet;
use nexus::math::{Pose, Vector};
use std::any::Any;
use std::marker::PhantomData;
use vortx::tensor::Tensor;

#[cfg(feature = "from_rapier")]
use nexus::dynamics::body::{BodyCoupling, RapierBodyCouplingEntry};

/// GPU compute pipeline for Material Point Method simulation.
///
/// This struct holds all the compiled compute shaders needed to execute a complete
/// MPM simulation step. It orchestrates the following stages:
/// 1. Update rigid body particles from coupled colliders
/// 2. Sort particles into grid cells
/// 3. Transfer data from particles to grid (P2G)
/// 4. Update grid velocities with forces and boundary conditions
/// 5. Transfer data from grid back to particles (G2P)
/// 6. Update particle positions and deformation gradients
/// 7. Apply impulses to coupled rigid bodies
///
/// # Type Parameters
///
/// * `GpuModel` - Particle material model data layout (must match shader expectations)
pub struct MpmPipeline<GpuModel: GpuParticleModelData> {
    grid: WgGrid,
    prefix_sum: GpuPrefixSum,
    sort: WgSort,
    p2g: WgP2G,
    p2g_cdf: WgP2GCdf,
    grid_update_cdf: WgGridUpdateCdf,
    grid_update: WgGridUpdate,
    particles_update: WgParticleUpdate,
    g2p: WgG2P,
    g2p_cdf: WgG2PCdf,
    rigid_particles_update: WgRigidParticleUpdate,
    /// Maximum timestep bound calculation.
    pub timestep_bounds: WgTimestepBounds,
    /// Rigid body impulse computation kernel (publicly accessible for external use).
    pub integrate_bodies: WgIntegrateBodies,
    _phantom: PhantomData<GpuModel>,
}

/// Callbacks for adding custom steps to the MPM pipeline.
pub trait MpmPipelineHooks<GpuModel: GpuParticleModelData> {
    fn max_substep_dt(
        &mut self,
        _backend: &GpuBackend,
        _timestamps: Option<&mut GpuTimestamps>,
        _data: &mut MpmData<GpuModel>,
        _state: &mut dyn Any,
    ) -> Option<f32> {
        None
    }

    /// Custom operation run after particles are sorted and attached to the grid.
    fn after_particle_sort(
        &mut self,
        _backend: &GpuBackend,
        _encoder: &mut GpuEncoder,
        _timestamps: Option<&mut GpuTimestamps>,
        _data: &mut MpmData<GpuModel>,
        _state: &mut dyn Any,
    ) -> Result<(), GpuBackendError> {
        Ok(())
    }

    /// Custom operation run after the main Particle-To-Grid transfer.
    fn after_p2g(
        &mut self,
        _backend: &GpuBackend,
        _encoder: &mut GpuEncoder,
        _timestamps: Option<&mut GpuTimestamps>,
        _data: &mut MpmData<GpuModel>,
        _state: &mut dyn Any,
    ) -> Result<(), GpuBackendError> {
        Ok(())
    }

    /// Custom operation run after updating the grid.
    fn after_grid_update(
        &mut self,
        _backend: &GpuBackend,
        _encoder: &mut GpuEncoder,
        _timestamps: Option<&mut GpuTimestamps>,
        _data: &mut MpmData<GpuModel>,
        _state: &mut dyn Any,
    ) -> Result<(), GpuBackendError> {
        Ok(())
    }

    /// Custom operation run after the Grid-To-Particle transfer.
    fn after_g2p(
        &mut self,
        _backend: &GpuBackend,
        _encoder: &mut GpuEncoder,
        _timestamps: Option<&mut GpuTimestamps>,
        _data: &mut MpmData<GpuModel>,
        _state: &mut dyn Any,
    ) -> Result<(), GpuBackendError> {
        Ok(())
    }

    fn particle_update_enabled(&self) -> bool {
        true
    }

    fn g2p_enabled(&self) -> bool {
        true
    }

    fn p2g_enabled(&self) -> bool {
        true
    }

    /// Custom operation run after updating particles.
    fn after_particles_update(
        &mut self,
        _backend: &GpuBackend,
        _encoder: &mut GpuEncoder,
        _timestamps: Option<&mut GpuTimestamps>,
        _data: &mut MpmData<GpuModel>,
        _state: &mut dyn Any,
    ) -> Result<(), GpuBackendError> {
        Ok(())
    }
}

impl<GpuModel: GpuParticleModelData> MpmPipelineHooks<GpuModel> for () {}

/// GPU-resident simulation state for MPM.
///
/// Contains all the data needed to execute an MPM simulation step, including
/// particles, grid, rigid body coupling information, and simulation parameters.
/// All data lives in GPU memory for efficient computation.
pub struct MpmData<GpuModel: GpuParticleModelData> {
    /// The simulation timestep.
    pub base_dt: f32,
    pub gravity: Vector,
    pub use_cpic: bool,
    /// Global simulation parameters (gravity, timestep).
    pub sim_params: GpuSimulationParams,
    /// Spatial grid for momentum transfer.
    pub grid: GpuGrid,
    /// MPM particles (positions, velocities, masses, material properties).
    pub particles: GpuParticles<GpuModel>,
    /// Particles sampled from rigid body collider surfaces for two-way coupling.
    pub rigid_particles: GpuRigidParticles,
    /// Rigid bodies coupled with the MPM simulation.
    pub bodies: GpuBodySet,
    /// MPM materials associated to each rigid-body.
    pub body_materials: GpuMaterials,
    /// Accumulated impulses to apply to rigid bodies from MPM interactions.
    pub impulses: GpuImpulses,
    /// Staging buffer for reading rigid body poses back to CPU.
    pub poses_staging: Tensor<Pose>,
    /// The timestep estimate computed from particles and their models.
    pub timestep_bounds: Tensor<GpuTimestepBounds>,
    /// Staging buffer for reading the timestep bound estimate.
    pub timestep_bounds_staging: Tensor<GpuTimestepBounds>,
    prefix_sum: PrefixSumWorkspace,
    #[cfg(feature = "from_rapier")]
    coupling: Vec<RapierBodyCouplingEntry>,
}

#[cfg(feature = "from_rapier")]
impl<GpuModel: GpuParticleModelData> MpmData<GpuModel> {
    /// Creates new MPM simulation data with default two-way coupling for all colliders.
    pub fn new(
        backend: &GpuBackend,
        params: SimulationParams,
        particles: &[Particle<GpuModel::Model>],
        bodies: &rapier::dynamics::RigidBodySet,
        colliders: &rapier::geometry::ColliderSet,
        materials: &[(rapier::geometry::ColliderHandle, BoundaryCondition)],
        cell_width: f32,
        grid_capacity: u32,
    ) -> Result<Self, GpuBackendError> {
        let coupling: Vec<_> = colliders
            .iter()
            .filter_map(|(co_handle, co)| {
                let rb_handle = co.parent()?;
                Some(RapierBodyCouplingEntry {
                    body: rb_handle,
                    collider: co_handle,
                    mode: BodyCoupling::OneWay,
                })
            })
            .collect();
        let materials: Vec<_> = coupling
            .iter()
            .map(|c| {
                materials
                    .iter()
                    .find(|e| e.0 == c.collider)
                    .map(|e| e.1)
                    .unwrap_or(BoundaryConditionExt::separate(1.0))
            })
            .collect();
        Self::with_select_coupling(
            backend,
            params,
            particles,
            bodies,
            colliders,
            coupling,
            &materials,
            cell_width,
            grid_capacity,
        )
    }

    /// Creates new MPM simulation data with custom rigid body coupling configuration.
    pub fn with_select_coupling(
        backend: &GpuBackend,
        params: SimulationParams,
        particles: &[Particle<GpuModel::Model>],
        bodies: &rapier::dynamics::RigidBodySet,
        colliders: &rapier::geometry::ColliderSet,
        coupling: Vec<RapierBodyCouplingEntry>,
        materials: &[BoundaryCondition],
        cell_width: f32,
        grid_capacity: u32,
    ) -> Result<Self, GpuBackendError> {
        assert_eq!(coupling.len(), materials.len());

        let sampling_step = cell_width;
        let bodies = GpuBodySet::from_rapier(backend, bodies, colliders, &coupling);
        let body_materials = GpuMaterials::new(backend, materials)?;
        let sim_params = GpuSimulationParams::new(backend, params)?;
        let particles = GpuParticles::from_particles(backend, particles)?;
        let rigid_particles =
            GpuRigidParticles::from_rapier(backend, colliders, &bodies, &coupling, sampling_step)?;
        let grid = GpuGrid::with_capacity(backend, grid_capacity, cell_width)?;
        let prefix_sum = PrefixSumWorkspace::with_capacity(backend, grid_capacity);
        let impulses = GpuImpulses::new(backend)?;
        let poses_staging = Tensor::vector_uninit(
            backend,
            bodies.len() as u32,
            BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        )?;
        let bounds = GpuTimestepBounds::default();
        let timestep_bounds = Tensor::scalar(
            backend,
            bounds,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )?;
        let timestep_bounds_staging = Tensor::scalar(
            backend,
            bounds,
            BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        )?;

        Ok(Self {
            sim_params,
            particles,
            gravity: params.gravity,
            use_cpic: true,
            rigid_particles,
            bodies,
            body_materials,
            impulses,
            grid,
            prefix_sum,
            poses_staging,
            coupling,
            timestep_bounds,
            timestep_bounds_staging,
            base_dt: params.dt,
        })
    }

    /// Returns the list of rigid body coupling entries.
    pub fn coupling(&self) -> &[RapierBodyCouplingEntry] {
        &self.coupling
    }
}

impl<GpuModel: GpuParticleModelData> MpmPipeline<GpuModel> {
    /// Creates a new MPM compute pipeline by compiling all necessary shaders.
    pub fn new(backend: &GpuBackend) -> Result<Self, GpuBackendError> {
        Ok(Self {
            grid: WgGrid::from_backend(backend)?,
            prefix_sum: GpuPrefixSum::from_backend(backend)?,
            sort: WgSort::from_backend(backend)?,
            p2g: WgP2G::from_backend(backend)?,
            p2g_cdf: WgP2GCdf::from_backend(backend)?,
            grid_update: WgGridUpdate::from_backend(backend)?,
            grid_update_cdf: WgGridUpdateCdf::from_backend(backend)?,
            particles_update: WgParticleUpdate::from_backend(backend)?,
            rigid_particles_update: WgRigidParticleUpdate::from_backend(backend)?,
            g2p: WgG2P::from_backend(backend)?,
            g2p_cdf: WgG2PCdf::from_backend(backend)?,
            integrate_bodies: WgIntegrateBodies::from_backend(backend)?,
            timestep_bounds: WgTimestepBounds::from_backend(backend)?,
            _phantom: PhantomData,
        })
    }

    /// Executes one complete MPM simulation timestep.
    pub async fn launch_step(
        &self,
        backend: &GpuBackend,
        encoder: &mut GpuEncoder,
        data: &mut MpmData<GpuModel>,
        mut timestamps: Option<&mut GpuTimestamps>,
        hooks: &mut dyn MpmPipelineHooks<GpuModel>,
        hooks_state: &mut dyn Any,
    ) -> Result<(), GpuBackendError> {
        {
            let mut pass = encoder.begin_pass("Rigid update", timestamps.as_deref_mut());
            self.integrate_bodies.launch_update_world_mass_properties(
                &mut pass,
                &mut data.impulses,
                &mut data.bodies,
            )?;
            self.rigid_particles_update.launch(
                &mut pass,
                &mut data.bodies,
                &mut data.rigid_particles,
            )?;
        }

        {
            let mut pass = encoder.begin_pass("Grid sort", timestamps.as_deref_mut());
            data.grid.swap_buffers();
            self.grid.launch_sort(
                backend,
                &mut pass,
                &mut data.particles,
                data.use_cpic.then_some(&mut data.rigid_particles),
                &mut data.grid,
                &mut data.prefix_sum,
                &self.sort,
                &self.prefix_sum,
            )?;

            if data.use_cpic {
                self.sort.launch_sort_rigid_particles(
                    &mut pass,
                    &mut data.rigid_particles,
                    &mut data.grid,
                )?;
            }
        }

        hooks.after_particle_sort(
            backend,
            encoder,
            timestamps.as_deref_mut(),
            data,
            hooks_state,
        )?;

        if data.use_cpic {
            {
                let mut pass = encoder.begin_pass("CDF grid update", timestamps.as_deref_mut());
                self.grid_update_cdf
                    .launch(&mut pass, &mut data.grid, &data.bodies)?;
            }

            {
                let mut pass = encoder.begin_pass("CDF P2G", timestamps.as_deref_mut());
                self.p2g_cdf.launch(
                    &mut pass,
                    &mut data.grid,
                    &data.rigid_particles,
                    &data.bodies,
                )?;
            }

            {
                let mut pass = encoder.begin_pass("CDF G2P", timestamps.as_deref_mut());
                self.g2p_cdf.launch(
                    &mut pass,
                    &data.sim_params,
                    &data.grid,
                    &mut data.particles,
                )?;
            }
        }

        if hooks.p2g_enabled() {
            let mut pass = encoder.begin_pass("P2G", timestamps.as_deref_mut());
            self.p2g.launch(
                &mut pass,
                data.use_cpic,
                &mut data.grid,
                &data.particles,
                &mut data.impulses,
                &data.bodies,
                &data.body_materials,
            )?;
        }

        hooks.after_p2g(
            backend,
            encoder,
            timestamps.as_deref_mut(),
            data,
            hooks_state,
        )?;

        {
            let mut pass = encoder.begin_pass("Grid update", timestamps.as_deref_mut());
            self.grid_update.launch(
                &mut pass,
                data.use_cpic,
                &data.sim_params,
                &mut data.grid,
                &data.bodies,
                &data.body_materials,
            )?;
        }

        hooks.after_grid_update(
            backend,
            encoder,
            timestamps.as_deref_mut(),
            data,
            hooks_state,
        )?;

        if hooks.g2p_enabled() {
            let mut pass = encoder.begin_pass("G2P", timestamps.as_deref_mut());
            self.g2p.launch(
                &mut pass,
                data.use_cpic,
                &data.sim_params,
                &data.grid,
                &mut data.particles,
                &data.bodies,
                &data.body_materials,
            )?;
        }

        hooks.after_g2p(
            backend,
            encoder,
            timestamps.as_deref_mut(),
            data,
            hooks_state,
        )?;

        if hooks.particle_update_enabled() {
            let mut pass = encoder.begin_pass("Particle update", timestamps.as_deref_mut());
            self.particles_update.launch(
                &mut pass,
                &data.sim_params,
                &data.grid,
                &mut data.particles,
            )?;
        }

        hooks.after_particles_update(
            backend,
            encoder,
            timestamps.as_deref_mut(),
            data,
            hooks_state,
        )?;

        {
            let mut pass = encoder.begin_pass("Integrate bodies", timestamps.as_deref_mut());
            self.integrate_bodies.launch(
                &mut pass,
                &data.grid,
                &data.sim_params,
                &mut data.impulses,
                &mut data.bodies,
            )?;
        }

        Ok(())
    }
}
