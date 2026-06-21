use crate::fem::mesh::FemMesh;
use crate::fem::pipeline::{FemPipeline, FemState};
use crate::fem::solver::{FemConfig, FemMaterial};
use crate::mpm::pipeline::{MpmPipeline, MpmState};
use crate::mpm::solver::{BoundaryCondition, BoundaryConditionExt, Particle, SimulationParams};
use crate::rapier::data::{Arena, Coarena, Index};
use crate::rapier::prelude::{
    Collider, ColliderHandle, GenericJoint, ImpulseJointHandle, MultibodyJointHandle, PhysicsWorld,
    RigidBody, RigidBodyHandle,
};
use crate::rbd::dynamics::{
    RbdSimParams,
    body::{BodyCoupling, RapierBodyCouplingEntry},
};
use crate::rbd::pipeline::{RbdPipeline, RbdState, RunStats};
use khal::backend::{Backend, GpuBackend, GpuBackendError, GpuTimestamps};

/// Handle referencing a rigid-body managed by a [`NexusState`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct NexusRbdHandle(Index);

/// Handle referencing a *chunk* of MPM particles managed by a [`NexusState`].
///
/// Particles are addressed by chunk rather than individually (a per-particle
/// handle map would be prohibitive at MPM scale). A chunk is mutable: particles
/// can be appended to it ([`NexusState::extend_chunk`]) or removed from it
/// ([`NexusState::remove_particles_from_chunk`] / [`NexusState::remove_chunk`]).
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct NexusParticleChunk(Index);

bitflags::bitflags! {
    /// A bit mask describing how a rigid-body interacts with the MPM continuum.
    #[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    pub struct RbdCoupling: u8 {
        const NONE = 0;
        const MPM_ONE_WAY_COUPLING = 1 << 0;
        const MPM_TWO_WAY_COUPLING = 1 << 1;
    }
}

impl RbdCoupling {
    /// Returns the [`BodyCoupling`] mode this flag set maps to, or `None` if the
    /// body isn’t coupled with the MPM simulation at all.
    fn body_coupling(self) -> Option<BodyCoupling> {
        if self.contains(RbdCoupling::MPM_TWO_WAY_COUPLING) {
            Some(BodyCoupling::TwoWays)
        } else if self.contains(RbdCoupling::MPM_ONE_WAY_COUPLING) {
            Some(BodyCoupling::OneWay)
        } else {
            None
        }
    }
}

/// Initial capacities used when allocating the GPU-resident physics states.
#[derive(Copy, Clone, Debug)]
pub struct NexusCapacities {
    /// Number of independent rigid-body simulation batches (environments).
    pub rbd_body_batches: usize,
    /// Maximum number of rigid-bodies per batch.
    pub rbd_body_capacity: usize,
    /// Number of grid cells reserved for the MPM background grid.
    pub mpm_grid_size: usize,
    /// Maximum number of MPM particles.
    pub mpm_particles_capacity: usize,
}

impl Default for NexusCapacities {
    fn default() -> Self {
        NexusCapacities {
            rbd_body_batches: 1,
            rbd_body_capacity: 65536,
            mpm_grid_size: 32768,
            mpm_particles_capacity: 65536,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct GpuRigidBodyRef {
    pub coupling: RbdCoupling,
    pub gpu_id: u32,
}

impl Default for GpuRigidBodyRef {
    fn default() -> Self {
        Self {
            coupling: RbdCoupling::NONE,
            gpu_id: u32::MAX,
        }
    }
}

/// Entity counts for the current scene, surfaced in the viewer UI. Rigid-body
/// counts are summed across all environments (batches).
#[derive(Clone, Copy, Default, Debug)]
pub struct NexusCounts {
    pub num_environments: usize,
    pub rigid_bodies: usize,
    pub colliders: usize,
    pub impulse_joints: usize,
    pub multibodies: usize,
    pub multibody_dofs: usize,
    pub particles: usize,
    pub fem_vertices: usize,
    pub fem_elements: usize,
}

/// High-level, GPU-resident state of a multiphysics simulation.
///
/// Each sub-state (`rbd`/`mpm`/`fem`) is lazily allocated the first time content
/// of the corresponding kind is added. The `*2gpu` maps translate the stable
/// public handles into the (unstable) GPU buffer slots, which shift around as
/// bodies/particles are inserted and removed.
pub struct NexusState {
    /// Rigid-body sub-state, allocated on the first [`Self::add_rigid_bodies`].
    pub rbd: Option<RbdState>,
    /// MPM sub-state, allocated on the first [`Self::add_particles`] (or the
    /// first coupled rigid-body insertion).
    pub mpm: Option<MpmState>,
    /// FEM sub-state.
    pub fem: Option<FemState>,

    pub run_stats: RunStats,

    /// Handle → GPU-slot map, one [`Coarena`] per simulation environment
    /// (batch). `rbd2gpu[env].get(body_handle.0).gpu_id` indexes into the
    /// flattened pose buffer (`env * num_colliders_per_batch + local_idx`).
    pub rbd2gpu: Vec<Coarena<GpuRigidBodyRef>>,
    multibody2gpu: Coarena<u32>,
    mpm_body2gpu: Arena<u32>,

    /// Live particle count per MPM chunk (the arena key is the public
    /// [`NexusParticleChunk`] handle).
    mpm_chunks: Arena<usize>,
    /// Owning chunk for each GPU particle slot, kept in sync under the
    /// swap-removal performed by [`Self::remove_chunk`] /
    /// [`Self::remove_particles_from_chunk`].
    slot2chunk: Vec<Index>,
    /// MPM simulation params / grid cell width requested before the MPM
    /// sub-state is lazily created. Applied in [`Self::mpm_or_insert`].
    mpm_params: Option<SimulationParams>,
    mpm_cell_width: f32,
    /// Number of MPM substeps run per [`Self::simulate`] call.
    pub mpm_substeps: u32,
    /// Desired CPIC rigid-coupling flag, kept here so it survives until the MPM
    /// sub-state is lazily created (and is what [`Self::mpm_use_cpic`] reports
    /// meanwhile). Applied in [`Self::mpm_or_insert`].
    mpm_use_cpic: bool,
    /// Set when particles or MPM-coupled bodies change; consumed by
    /// [`Self::finalize`] to rebuild the MPM↔rapier coupling.
    mpm_dirty: bool,

    // Initial capacities used to allocate the states lazily.
    capacities: NexusCapacities,

    /// One rapier world per simulation environment (batch). Environment 0
    /// always exists; the non-`*_in` insert helpers target it. Batched demos
    /// add more via [`Self::add_environment`].
    rbd_envs: Vec<PhysicsWorld>,
    /// Per-environment simulation parameters (same length as `rbd_envs`).
    rbd_sim_params: Vec<RbdSimParams>,
    /// Set whenever the rapier worlds change; consumed by [`Self::finalize`] to
    /// decide whether the GPU [`RbdState`] needs rebuilding.
    rbd_dirty: bool,
    /// Number of rigid-body solver steps advanced per [`Self::simulate`] call.
    pub rbd_steps_per_frame: u32,
    /// Per-environment GPU collider-slot reservation. When > 0, the GPU
    /// [`RbdState`] is built with this many slots (rather than exactly the
    /// current body count), leaving room for [`Self::add_rigid_body`] to append
    /// bodies in place — without rebuilding the whole scene. Set via
    /// [`Self::reserve_rigid_bodies`].
    rbd_reserve_per_env: usize,
    // TODO: keep track of whether there is any non-fixed rigid-body (if there isn’t, we can
    //       skip the rbd pipeline entirely).
}

impl Default for NexusState {
    fn default() -> Self {
        Self::new(NexusCapacities::default())
    }
}

impl NexusState {
    /// Creates an empty state. The GPU sub-states are allocated lazily (sized
    /// from `capacities`) the first time matching content is added.
    pub fn new(capacities: NexusCapacities) -> Self {
        Self {
            rbd: None,
            mpm: None,
            fem: None,
            run_stats: RunStats::default(),
            rbd_envs: vec![PhysicsWorld::default()],
            rbd_sim_params: vec![RbdSimParams::tgs_soft()],
            rbd_dirty: false,
            rbd_steps_per_frame: 1,
            rbd_reserve_per_env: 0,
            rbd2gpu: vec![Coarena::new()],
            mpm_body2gpu: Arena::new(),
            multibody2gpu: Coarena::new(),
            mpm_chunks: Arena::new(),
            slot2chunk: Vec::new(),
            mpm_params: None,
            mpm_cell_width: 1.0,
            mpm_substeps: 20,
            mpm_use_cpic: true,
            mpm_dirty: false,
            capacities,
        }
    }

    /// Reserves additional capacity in the handle maps to avoid reallocations
    /// when a known number of bodies/particles is about to be inserted.
    pub fn reserve(&mut self, additional: NexusCapacities) {
        for env in &mut self.rbd2gpu {
            env.reserve(additional.rbd_body_capacity);
        }
        // self.mpm_bodies2gpu
        //     .reserve(additional.rbd_body_batches * additional.rbd_body_capacity);
        // self.particle2gpu.reserve(additional.mpm_particles_capacity);
        // TODO: resize the GPU buffers too.
    }

    /// Sets the MPM simulation parameters (gravity, timestep) and grid cell
    /// width. Call before the first [`Self::add_particles`]; the values are
    /// applied when the MPM sub-state is created. If MPM already exists they are
    /// applied immediately (the grid is reset, so prefer calling this first).
    pub fn set_mpm_params(
        &mut self,
        backend: &GpuBackend,
        params: SimulationParams,
        cell_width: f32,
    ) -> Result<(), GpuBackendError> {
        self.mpm_params = Some(params);
        self.mpm_cell_width = cell_width;
        if let Some(mpm) = self.mpm.as_mut() {
            mpm.set_cell_width(backend, cell_width, self.capacities.mpm_grid_size as u32)?;
            mpm.set_simulation_params(backend, params)?;
        }
        Ok(())
    }

    /// Sets the number of MPM substeps run per [`Self::simulate`] call (default
    /// 20). More substeps → smaller timestep → more stable but slower.
    pub fn set_mpm_substeps(&mut self, substeps: u32) {
        self.mpm_substeps = substeps.max(1);
    }

    /// Number of MPM substeps run per [`Self::simulate`] call.
    pub fn mpm_substeps(&self) -> u32 {
        self.mpm_substeps
    }

    /// Enables/disables CPIC (compatible particle-in-cell) rigid coupling. The
    /// preference is stored so it survives until MPM is lazily allocated. Not
    /// overwritten by [`Self::finalize`] unless the coupling set changes.
    pub fn set_mpm_use_cpic(&mut self, enabled: bool) {
        self.mpm_use_cpic = enabled;
        if let Some(mpm) = self.mpm.as_mut() {
            mpm.use_cpic = enabled;
        }
    }

    /// Whether CPIC rigid coupling is enabled. Falls back to the stored
    /// preference before MPM is lazily allocated.
    pub fn mpm_use_cpic(&self) -> bool {
        self.mpm
            .as_ref()
            .map(|m| m.use_cpic)
            .unwrap_or(self.mpm_use_cpic)
    }

    /// Whether this state uses the MPM solver. True once MPM has been configured
    /// via [`Self::set_mpm_params`], even before the sub-state is lazily
    /// allocated on the first [`Self::add_particles`] — so a particle emitter
    /// that starts empty still reports its MPM usage (e.g. to the viewer UI).
    pub fn has_mpm(&self) -> bool {
        self.mpm.is_some() || self.mpm_params.is_some()
    }

    /// Sets the MPM gravity vector. Applied on the next [`Self::simulate`] (the
    /// per-substep params are re-uploaded each frame), so this is cheap.
    pub fn set_mpm_gravity(&mut self, gravity: crate::rbd::math::Vector) {
        // Keep the stored params authoritative so the gravity survives until MPM
        // is lazily allocated (and is what `mpm_gravity` reports meanwhile).
        if let Some(params) = self.mpm_params.as_mut() {
            params.gravity = gravity;
        }
        if let Some(mpm) = self.mpm.as_mut() {
            mpm.gravity = gravity;
        }
    }

    /// Current MPM gravity vector. Falls back to the gravity configured via
    /// [`Self::set_mpm_params`] before MPM is lazily allocated, and only to zero
    /// if no params were ever set.
    pub fn mpm_gravity(&self) -> crate::rbd::math::Vector {
        self.mpm
            .as_ref()
            .map(|m| m.gravity)
            .or_else(|| self.mpm_params.map(|p| p.gravity))
            .unwrap_or(crate::rbd::math::Vector::ZERO)
    }

    // ── FEM runtime settings ────────────────────────────────────────────

    /// Sets the number of FEM substeps run per [`Self::simulate`] call.
    pub fn set_fem_substeps(&mut self, substeps: u32) {
        if let Some(fem) = self.fem.as_mut() {
            fem.num_substeps = substeps.max(1);
        }
    }

    /// Number of FEM substeps per [`Self::simulate`] call (1 if FEM is absent).
    pub fn fem_substeps(&self) -> u32 {
        self.fem.as_ref().map(|f| f.num_substeps).unwrap_or(1)
    }

    /// Sets the FEM gravity vector (no-op if FEM isn't allocated).
    pub fn set_fem_gravity(
        &mut self,
        backend: &GpuBackend,
        gravity: crate::rbd::math::Vector,
    ) -> Result<(), GpuBackendError> {
        if let Some(fem) = self.fem.as_mut() {
            fem.set_gravity(backend, gravity)?;
        }
        Ok(())
    }

    /// Current FEM gravity vector (zero if FEM isn't allocated).
    pub fn fem_gravity(&self) -> crate::rbd::math::Vector {
        self.fem
            .as_ref()
            .map(|f| f.gravity())
            .unwrap_or(crate::rbd::math::Vector::ZERO)
    }

    /// Sets the FEM (mass-proportional) damping coefficient.
    pub fn set_fem_damping(
        &mut self,
        backend: &GpuBackend,
        damping: f32,
    ) -> Result<(), GpuBackendError> {
        if let Some(fem) = self.fem.as_mut() {
            fem.set_damping(backend, damping)?;
        }
        Ok(())
    }

    /// Current FEM damping coefficient (0 if FEM isn't allocated).
    pub fn fem_damping(&self) -> f32 {
        self.fem.as_ref().map(|f| f.damping()).unwrap_or(0.0)
    }

    // ── Rigid-body runtime settings ─────────────────────────────────────

    /// Sets the number of rigid-body solver steps advanced per
    /// [`Self::simulate`] call (default 1). Acts as a simulation-speed control.
    pub fn set_rbd_steps_per_frame(&mut self, steps: u32) {
        self.rbd_steps_per_frame = steps.max(1);
    }

    /// Number of rigid-body solver steps per [`Self::simulate`] call.
    pub fn rbd_steps_per_frame(&self) -> u32 {
        self.rbd_steps_per_frame
    }

    /// Current entity counts (rigid bodies, colliders, joints, multibody DOFs,
    /// particles, FEM vertices/elements) for display in the UI. Rigid-body
    /// counts are summed across all environments.
    pub fn counts(&self) -> NexusCounts {
        let mut c = NexusCounts {
            num_environments: self.rbd_envs.len(),
            ..Default::default()
        };
        for world in &self.rbd_envs {
            c.rigid_bodies += world.bodies.len();
            c.colliders += world.colliders.len();
            c.impulse_joints += world.impulse_joints.len();
            for mb in world.multibody_joints.multibodies() {
                c.multibodies += 1;
                c.multibody_dofs += mb.ndofs();
            }
        }
        if let Some(mpm) = self.mpm.as_ref() {
            c.particles = mpm.particles.len();
        }
        if let Some(fem) = self.fem.as_ref() {
            c.fem_vertices = fem.num_vertices as usize;
            c.fem_elements = fem.num_elements as usize;
        }
        c
    }

    /// Returns a mutable reference to the MPM sub-state, allocating an empty one
    /// (sized from the stored capacities, configured from [`Self::set_mpm_params`])
    /// if it doesn’t exist yet.
    fn mpm_or_insert(&mut self, backend: &GpuBackend) -> Result<&mut MpmState, GpuBackendError> {
        if self.mpm.is_none() {
            let grid_capacity = self.capacities.mpm_grid_size as u32;
            let mut mpm = MpmState::empty(backend, grid_capacity)?;
            mpm.set_cell_width(backend, self.mpm_cell_width, grid_capacity)?;
            if let Some(params) = self.mpm_params {
                mpm.set_simulation_params(backend, params)?;
            }
            mpm.use_cpic = self.mpm_use_cpic;
            self.mpm = Some(mpm);
        }
        Ok(self.mpm.as_mut().unwrap())
    }

    /// Returns a mutable reference to the rbd sub-state, allocating an empty one
    /// (sized from the stored capacities) if it doesn’t exist yet.
    fn rbd_or_insert(&mut self, backend: &GpuBackend) -> &mut RbdState {
        if self.rbd.is_none() {
            self.rbd = Some(RbdState::empty(
                backend,
                self.capacities.rbd_body_batches as u32,
                self.capacities.rbd_body_capacity as u32,
            ));
        }
        self.rbd.as_mut().unwrap()
    }

    /// Adds a new (empty) simulation environment (batch) and returns its index.
    ///
    /// Environment 0 always exists; batched demos call this once per extra
    /// environment, then insert into it with the `*_in` helpers. Every
    /// environment is solved independently on the GPU and rendered at its own
    /// poses.
    pub fn add_environment(&mut self) -> usize {
        self.rbd_envs.push(PhysicsWorld::default());
        self.rbd_sim_params.push(RbdSimParams::tgs_soft());
        self.rbd2gpu.push(Coarena::new());
        self.rbd_dirty = true;
        self.rbd_envs.len() - 1
    }

    /// Number of simulation environments (batches).
    pub fn num_environments(&self) -> usize {
        self.rbd_envs.len()
    }

    /// Mutable access to environment `env`'s rapier world, e.g. for loaders
    /// (URDF) that insert directly into the rapier sets. Marks the rbd state
    /// dirty so [`Self::finalize`] rebuilds the GPU buffers.
    pub fn rbd_world_mut(&mut self, env: usize) -> &mut PhysicsWorld {
        self.rbd_dirty = true;
        &mut self.rbd_envs[env]
    }

    pub fn insert_rigid_body(
        &mut self,
        body: RigidBody,
        collider: Collider,
        coupling: RbdCoupling,
    ) -> RigidBodyHandle {
        self.insert_rigid_body_in(0, body, collider, coupling)
    }

    /// Inserts a body + collider into environment `env`.
    pub fn insert_rigid_body_in(
        &mut self,
        env: usize,
        body: RigidBody,
        collider: Collider,
        coupling: RbdCoupling,
    ) -> RigidBodyHandle {
        let (handle, _) = self.rbd_envs[env].insert(body, collider);
        self.rbd2gpu[env].insert(
            handle.0,
            GpuRigidBodyRef {
                coupling,
                gpu_id: u32::MAX,
            },
        );
        self.rbd_dirty = true;
        // MPM-coupled boundary colliders live only in environment 0 and feed the
        // MPM coupling rebuild in `finalize`.
        if env == 0 && coupling.body_coupling().is_some() {
            self.mpm_dirty = true;
        }
        handle
    }

    /// Reserves `per_env` GPU collider slots per environment so that bodies can
    /// later be added with [`Self::add_rigid_body`] *in place* — appended to the
    /// existing GPU buffers instead of rebuilding the whole scene.
    ///
    /// Call this before the first [`Self::finalize`]/[`Self::simulate`]. Intended
    /// for single-environment scenes (the appended body data is shared across
    /// batches). `per_env` is a hard cap: once it's full, `add_rigid_body` falls
    /// back to a full rebuild.
    pub fn reserve_rigid_bodies(&mut self, per_env: usize) {
        self.rbd_reserve_per_env = per_env;
    }

    /// Adds a body + collider to environment 0, appending it directly to the GPU
    /// [`RbdState`] **without rebuilding the scene** — provided the state already
    /// exists and has spare capacity (see [`Self::reserve_rigid_bodies`]). If
    /// there is no GPU state yet, or the reservation is full, it falls back to a
    /// normal insert (a full rebuild on the next `finalize`).
    ///
    /// Only primitive (vertex-less) colliders are supported on the fast path.
    pub fn add_rigid_body(
        &mut self,
        backend: &GpuBackend,
        body: RigidBody,
        collider: Collider,
        coupling: RbdCoupling,
    ) -> Result<RigidBodyHandle, GpuBackendError> {
        let handles = self.add_rigid_bodies(backend, [(body, collider, coupling)])?;
        Ok(handles[0])
    }

    /// Adds several body + collider pairs to environment 0 in a single in-place
    /// GPU append — the batched form of [`Self::add_rigid_body`]. One
    /// `append_bodies` call (one buffer upload + one `rebuild_batch_indices`)
    /// covers the whole batch, so it's much cheaper than calling `add_rigid_body`
    /// in a loop. Returns the handles in input order.
    ///
    /// Like the single-body version it appends without rebuilding the scene when
    /// the GPU state exists and has room for the *entire* batch; otherwise it
    /// falls back to a full rebuild on the next `finalize`. Only primitive
    /// (vertex-less) colliders are supported on the fast path.
    pub fn add_rigid_bodies(
        &mut self,
        backend: &GpuBackend,
        bodies: impl IntoIterator<Item = (RigidBody, Collider, RbdCoupling)>,
    ) -> Result<Vec<RigidBodyHandle>, GpuBackendError> {
        // Keep copies for the GPU append before the rapier world consumes them.
        let mut gpu_pairs: Vec<(RigidBody, Collider)> = Vec::new();
        let mut handles: Vec<RigidBodyHandle> = Vec::new();
        let mut couplings: Vec<RbdCoupling> = Vec::new();
        for (body, collider, coupling) in bodies {
            gpu_pairs.push((body.clone(), collider.clone()));
            let (handle, _) = self.rbd_envs[0].insert(body, collider);
            handles.push(handle);
            couplings.push(coupling);
        }
        if handles.is_empty() {
            return Ok(handles);
        }

        let appended = match self.rbd.as_mut() {
            Some(rbd)
                if (rbd.num_active_colliders() as usize) + gpu_pairs.len()
                    <= rbd.num_colliders_per_batch() as usize =>
            {
                let range = rbd.append_bodies(backend, &gpu_pairs)?;
                // Single environment: the per-batch local slot is the gpu_id.
                for (i, (&handle, &coupling)) in handles.iter().zip(&couplings).enumerate() {
                    self.rbd2gpu[0].insert(
                        handle.0,
                        GpuRigidBodyRef {
                            coupling,
                            gpu_id: range.start + i as u32,
                        },
                    );
                }
                true
            }
            _ => false,
        };

        if !appended {
            // No GPU state yet, or not enough room for the whole batch: fall back
            // to a full rebuild on the next `finalize`.
            for (&handle, &coupling) in handles.iter().zip(&couplings) {
                self.rbd2gpu[0].insert(
                    handle.0,
                    GpuRigidBodyRef {
                        coupling,
                        gpu_id: u32::MAX,
                    },
                );
            }
            self.rbd_dirty = true;
        }
        if couplings.iter().any(|c| c.body_coupling().is_some()) {
            self.mpm_dirty = true;
        }
        Ok(handles)
    }

    /// Inserts a rigid-body without any attached collider (e.g. a joint anchor).
    pub fn insert_body(&mut self, body: RigidBody, coupling: RbdCoupling) -> RigidBodyHandle {
        self.insert_body_in(0, body, coupling)
    }

    /// Inserts a collider-less rigid-body into environment `env`.
    pub fn insert_body_in(
        &mut self,
        env: usize,
        body: RigidBody,
        coupling: RbdCoupling,
    ) -> RigidBodyHandle {
        let handle = self.rbd_envs[env].insert_body(body);
        self.rbd2gpu[env].insert(
            handle.0,
            GpuRigidBodyRef {
                coupling,
                gpu_id: u32::MAX,
            },
        );
        self.rbd_dirty = true;
        handle
    }

    /// Attaches a collider to an existing body (or inserts a parent-less one) in
    /// environment `env`.
    pub fn insert_collider_in(
        &mut self,
        env: usize,
        collider: Collider,
        parent: Option<RigidBodyHandle>,
    ) -> ColliderHandle {
        self.rbd_dirty = true;
        self.rbd_envs[env].insert_collider(collider, parent)
    }

    /// Inserts an impulse joint into environment 0.
    pub fn insert_impulse_joint(
        &mut self,
        body1: RigidBodyHandle,
        body2: RigidBodyHandle,
        joint: impl Into<GenericJoint>,
    ) -> ImpulseJointHandle {
        self.insert_impulse_joint_in(0, body1, body2, joint)
    }

    /// Inserts an impulse joint between two bodies of environment `env`.
    pub fn insert_impulse_joint_in(
        &mut self,
        env: usize,
        body1: RigidBodyHandle,
        body2: RigidBodyHandle,
        joint: impl Into<GenericJoint>,
    ) -> ImpulseJointHandle {
        self.rbd_dirty = true;
        self.rbd_envs[env].insert_impulse_joint(body1, body2, joint)
    }

    /// Inserts a multibody joint into environment 0.
    ///
    /// Returns `None` if the joint would create an invalid kinematic chain
    /// (e.g. a cycle).
    pub fn insert_multibody_joint(
        &mut self,
        body1: RigidBodyHandle,
        body2: RigidBodyHandle,
        joint: impl Into<GenericJoint>,
    ) -> Option<MultibodyJointHandle> {
        self.insert_multibody_joint_in(0, body1, body2, joint)
    }

    /// Inserts a multibody joint between two bodies of environment `env`.
    pub fn insert_multibody_joint_in(
        &mut self,
        env: usize,
        body1: RigidBodyHandle,
        body2: RigidBodyHandle,
        joint: impl Into<GenericJoint>,
    ) -> Option<MultibodyJointHandle> {
        self.rbd_dirty = true;
        self.rbd_envs[env].insert_multibody_joint(body1, body2, joint)
    }

    /// Number of GPU batches (== number of environments) once finalized.
    pub fn rbd_num_batches(&self) -> u32 {
        self.rbd.as_ref().map(|r| r.num_batches()).unwrap_or(0)
    }

    /// Sets a multibody joint motor's target velocity on the GPU state (used by
    /// the URDF demo for per-frame actuation). No-op until the rbd state exists.
    #[cfg(feature = "dim3")]
    pub fn set_multibody_motor_velocity(
        &mut self,
        backend: &GpuBackend,
        batch: u32,
        link_id: u32,
        axis: crate::rapier::dynamics::JointAxis,
        target_vel: f32,
    ) -> Result<(), GpuBackendError> {
        if let Some(rbd) = self.rbd.as_mut() {
            rbd.multibodies_mut()
                .set_motor_velocity(backend, batch, link_id, axis, target_vel)?;
        }
        Ok(())
    }

    /// Appends a new chunk of MPM particles (`O(added)`) and returns its handle.
    pub fn add_particles(
        &mut self,
        backend: &GpuBackend,
        particles: Vec<Particle>,
    ) -> Result<NexusParticleChunk, GpuBackendError> {
        let n = particles.len();
        let chunk = self.mpm_chunks.insert(n);
        {
            let mpm = self.mpm_or_insert(backend)?;
            mpm.particles.append(backend, &particles)?;
        }
        self.slot2chunk.extend(std::iter::repeat(chunk).take(n));
        self.mpm_dirty = true;
        Ok(NexusParticleChunk(chunk))
    }

    /// Appends more particles to an existing chunk (`O(added)`).
    pub fn extend_chunk(
        &mut self,
        backend: &GpuBackend,
        chunk: NexusParticleChunk,
        particles: Vec<Particle>,
    ) -> Result<(), GpuBackendError> {
        let n = particles.len();
        {
            let mpm = self.mpm_or_insert(backend)?;
            mpm.particles.append(backend, &particles)?;
        }
        self.slot2chunk.extend(std::iter::repeat(chunk.0).take(n));
        if let Some(c) = self.mpm_chunks.get_mut(chunk.0) {
            *c += n;
        }
        self.mpm_dirty = true;
        Ok(())
    }

    /// Inserts a FEM soft body (one or more `(mesh, material)` pairs solved
    /// together) into the simulation. There is at most one FEM sub-state; a
    /// second call replaces it.
    pub fn insert_fem(
        &mut self,
        backend: &GpuBackend,
        meshes: &[(FemMesh, FemMaterial)],
        config: &FemConfig,
    ) -> Result<(), GpuBackendError> {
        self.fem = Some(FemState::new(backend, meshes, config)?);
        Ok(())
    }

    /// MPM background-grid cell width (used by the viewer to size rendered
    /// particles).
    pub fn mpm_cell_width(&self) -> f32 {
        self.mpm_cell_width
    }

    /// Removes every particle of a chunk (`O(removed)`) and drops the handle.
    pub fn remove_chunk(
        &mut self,
        backend: &GpuBackend,
        chunk: NexusParticleChunk,
    ) -> Result<(), GpuBackendError> {
        let slots: Vec<u32> = self
            .slot2chunk
            .iter()
            .enumerate()
            .filter(|(_, c)| **c == chunk.0)
            .map(|(i, _)| i as u32)
            .collect();
        self.swap_remove_particle_slots(backend, &slots)?;
        self.mpm_chunks.remove(chunk.0);
        Ok(())
    }

    /// Removes up to `count` particles from a chunk (`O(removed)`), returning the
    /// number actually removed. The chunk itself is kept (even if emptied).
    pub fn remove_particles_from_chunk(
        &mut self,
        backend: &GpuBackend,
        chunk: NexusParticleChunk,
        count: usize,
    ) -> Result<usize, GpuBackendError> {
        let mut slots: Vec<u32> = self
            .slot2chunk
            .iter()
            .enumerate()
            .filter(|(_, c)| **c == chunk.0)
            .map(|(i, _)| i as u32)
            .collect();
        // Remove the highest GPU slots first — keeps the swap-removal cheap.
        slots.sort_unstable_by(|a, b| b.cmp(a));
        slots.truncate(count);
        let removed = slots.len();
        self.swap_remove_particle_slots(backend, &slots)?;
        if let Some(c) = self.mpm_chunks.get_mut(chunk.0) {
            *c = c.saturating_sub(removed);
        }
        Ok(removed)
    }

    /// Swap-removes the given GPU particle slots and patches `slot2chunk` to
    /// follow the relocations the GPU performed.
    fn swap_remove_particle_slots(
        &mut self,
        backend: &GpuBackend,
        slots: &[u32],
    ) -> Result<(), GpuBackendError> {
        if slots.is_empty() {
            return Ok(());
        }
        let remaps = {
            let Some(mpm) = self.mpm.as_mut() else {
                return Ok(());
            };
            mpm.particles.swap_remove(backend, slots)?
        };
        // Each `(from, to)`: the tail particle at `from` was moved down to the
        // freed slot `to`, so its chunk ownership moves with it.
        for (from, to) in remaps {
            self.slot2chunk[to as usize] = self.slot2chunk[from as usize];
        }
        let new_len = self.mpm.as_ref().unwrap().particles.len();
        self.slot2chunk.truncate(new_len);
        Ok(())
    }

    pub async fn finalize(&mut self, backend: &GpuBackend) -> Result<(), GpuBackendError> {
        let rbd_was_dirty = self.rbd_dirty;
        if self.rbd_dirty {
            // Full (re)build of the GPU rbd state from the rapier worlds. With a
            // reservation (`reserve_rigid_bodies`) the buffers are sized for
            // spare slots so later `add_rigid_body` calls can append in place;
            // otherwise the state is sized exactly to the current body count.
            let rbd_state = if self.rbd_reserve_per_env > 0 {
                let num_envs = self.rbd_envs.len() as u32;
                let max_count = self
                    .rbd_envs
                    .iter()
                    .map(|w| w.colliders.len())
                    .max()
                    .unwrap_or(0);
                let capacity = self.rbd_reserve_per_env.max(max_count) as u32;
                let mut st = RbdState::empty(backend, num_envs, capacity);
                // Append environment 0's bodies in collider-iteration order, so
                // the per-batch slot index matches the `from_rapier` layout.
                let world = &self.rbd_envs[0];
                let mut bodies = Vec::new();
                for (_, collider) in world.colliders.iter() {
                    if let Some(bh) = collider.parent() {
                        bodies.push((world.bodies[bh].clone(), collider.clone()));
                    }
                }
                if !bodies.is_empty() {
                    st.append_bodies(backend, &bodies)?;
                }
                st
            } else {
                let environments: Vec<_> = self
                    .rbd_envs
                    .iter()
                    .zip(self.rbd_sim_params.iter())
                    .map(|(w, sp)| {
                        (
                            &w.bodies,
                            &w.colliders,
                            &w.impulse_joints,
                            &w.multibody_joints,
                            sp,
                        )
                    })
                    .collect();
                RbdState::from_rapier(backend, &environments)
            };

            // Rebuild the per-environment handle → GPU-slot maps to match the
            // flattened pose layout `from_rapier` produced: colliders are laid
            // out env-major with stride `num_colliders_per_batch`, in rapier's
            // own collider iteration order (one collider per body). Per-body
            // coupling recorded at insertion time is preserved.
            let stride = rbd_state.num_colliders_per_batch();
            for (env_idx, world) in self.rbd_envs.iter().enumerate() {
                for (local_idx, (_, collider)) in world.colliders.iter().enumerate() {
                    let Some(body_handle) = collider.parent() else {
                        continue;
                    };
                    let coupling = self.rbd2gpu[env_idx]
                        .get(body_handle.0)
                        .map(|r| r.coupling)
                        .unwrap_or(RbdCoupling::NONE);
                    self.rbd2gpu[env_idx].insert(
                        body_handle.0,
                        GpuRigidBodyRef {
                            coupling,
                            gpu_id: env_idx as u32 * stride + local_idx as u32,
                        },
                    );
                }
            }
            self.rbd = Some(rbd_state);
            self.rbd_dirty = false;
        }

        // MPM ↔ rapier coupling. Boundary/coupled colliders are inserted into
        // environment 0 as rigid bodies tagged `RbdCoupling::MPM_*`; rebuild the
        // coupling (sampled rigid particles + uploaded body set) whenever those
        // bodies — or the particle set — changed.
        if (rbd_was_dirty || self.mpm_dirty) && self.mpm.is_some() {
            let world = &self.rbd_envs[0];
            let mut coupling = Vec::new();
            let mut materials = Vec::new();
            for (collider_handle, collider) in world.colliders.iter() {
                let Some(body_handle) = collider.parent() else {
                    continue;
                };
                let Some(gpu_ref) = self.rbd2gpu[0].get(body_handle.0) else {
                    continue;
                };
                let Some(mode) = gpu_ref.coupling.body_coupling() else {
                    continue;
                };
                coupling.push(RapierBodyCouplingEntry {
                    body: body_handle,
                    collider: collider_handle,
                    mode,
                });
                // Default boundary behavior; a richer per-collider material API
                // can be layered on later.
                materials.push(BoundaryCondition::stick());
            }
            if !coupling.is_empty() {
                let cell_width = self.mpm_cell_width;
                let mpm = self.mpm.as_mut().unwrap();
                mpm.set_coupling(
                    backend,
                    &world.bodies,
                    &world.colliders,
                    coupling,
                    &materials,
                    cell_width,
                )?;
            }
            self.mpm_dirty = false;
        }
        Ok(())
    }

    /// Removes the given rigid-bodies from the simulation.
    pub fn remove_rigid_bodies(
        &mut self,
        backend: &GpuBackend,
        bodies: &[NexusRbdHandle],
    ) -> Result<(), GpuBackendError> {
        // // 1. Resolve the rbd GPU slots of the bodies to remove.
        // let gpu_ids: Vec<u32> = bodies
        //     .iter()
        //     .filter_map(|h| self.rbd2gpu.get(h.0).copied())
        //     .collect();
        //
        // // 2. Swap-remove them from the rbd GPU buffers. `remove_bodies` returns
        // //    the slot relocations it performed (`(from, to)`) so we can patch the
        // //    handle map.
        // let remaps = match self.rbd.as_mut() {
        //     Some(rbd) => rbd.remove_bodies(backend, &gpu_ids)?,
        //     None => Vec::new(),
        // };
        //
        // // 3. Drop the removed handles, then patch the relocated slots.
        // for h in bodies {
        //     self.rbd2gpu.remove(h.0);
        // }
        // for (from, to) in remaps {
        //     for (_, slot) in self.rbd2gpu.iter_mut() {
        //         if *slot == from {
        //             *slot = to;
        //         }
        //     }
        // }
        //
        // // 4. Mirror the removal on the MPM coupling side. The MPM body buffers
        // //    are densely packed, so a shift-remove relocates every higher slot;
        // //    we rebuild the (handle -> MPM slot) map accordingly.
        // //
        // //    NOTE: as on the insertion side, the per-collider materials, impulse
        // //    accumulators and sampled rigid particles are not resized here yet.
        // let mut mpm_remove: Vec<u32> = bodies
        //     .iter()
        //     .filter_map(|h| self.mpm_bodies2gpu.get(h.0).copied())
        //     .collect();
        // if !mpm_remove.is_empty() {
        //     mpm_remove.sort_unstable_by(|a, b| b.cmp(a));
        //     mpm_remove.dedup();
        //
        //     let removed: std::collections::HashSet<Index> = bodies.iter().map(|h| h.0).collect();
        //     let survivors: Vec<(Index, u32)> = self
        //         .mpm_bodies2gpu
        //         .iter()
        //         .filter(|(idx, _)| !removed.contains(idx))
        //         .map(|(idx, &slot)| (idx, slot))
        //         .collect();
        //
        //     if let Some(mpm) = self.mpm.as_mut() {
        //         for slot in &mpm_remove {
        //             mpm.bodies
        //                 .shift_remove(backend, *slot as usize..*slot as usize + 1)?;
        //         }
        //     }
        //
        //     let mut new_map = Coarena::new();
        //     for (idx, slot) in survivors {
        //         let shift = mpm_remove.iter().filter(|&&r| r < slot).count() as u32;
        //         new_map.insert(idx, slot - shift);
        //     }
        //     self.mpm_bodies2gpu = new_map;
        // }

        Ok(())
    }
}
