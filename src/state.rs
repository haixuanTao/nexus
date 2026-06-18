use crate::fem::pipeline::{FemPipeline, FemState};
use crate::mpm::pipeline::{MpmPipeline, MpmState};
use crate::mpm::solver::Particle;
use crate::rapier::data::{Arena, Coarena, Index};
use crate::rapier::prelude::{Collider, ColliderHandle, GenericJoint, ImpulseJointHandle, MultibodyJointHandle, PhysicsWorld, RigidBody, RigidBodyHandle};
use crate::rbd::dynamics::{body::BodyCoupling, RbdSimParams};
use crate::rbd::pipeline::{RbdPipeline, RbdState, RbdStats};
use khal::backend::{Backend, GpuBackend, GpuBackendError, GpuTimestamps};
use std::ops::RangeBounds;

/// Handle referencing a rigid-body managed by a [`NexusState`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct NexusRbdHandle(Index);

/// Handle referencing an MPM particle managed by a [`NexusState`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct NexusParticleHandle(Index);

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
            gpu_id: u32::MAX
        }
    }
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

    pub rbd_stats: RbdStats,

    /// Handle → GPU-slot map, one [`Coarena`] per simulation environment
    /// (batch). `rbd2gpu[env].get(body_handle.0).gpu_id` indexes into the
    /// flattened pose buffer (`env * num_colliders_per_batch + local_idx`).
    pub rbd2gpu: Vec<Coarena<GpuRigidBodyRef>>,
    multibody2gpu: Coarena<u32>,
    mpm_body2gpu: Arena<u32>,

    // Initial capacities used to allocate the states lazily.
    capacities: NexusCapacities,

    // GPU compute pipelines, lazily created the first time the matching sub-state
    // is stepped in [`Self::simulate`].
    rbd_pipeline: Option<RbdPipeline>,
    mpm_pipeline: Option<MpmPipeline>,
    fem_pipeline: Option<FemPipeline>,

    /// One rapier world per simulation environment (batch). Environment 0
    /// always exists; the non-`*_in` insert helpers target it. Batched demos
    /// add more via [`Self::add_environment`].
    rbd_envs: Vec<PhysicsWorld>,
    /// Per-environment simulation parameters (same length as `rbd_envs`).
    rbd_sim_params: Vec<RbdSimParams>,
    /// Set whenever the rapier worlds change; consumed by [`Self::finalize`] to
    /// decide whether the GPU [`RbdState`] needs rebuilding.
    rbd_dirty: bool,

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
            rbd_stats: RbdStats::default(),
            rbd_envs: vec![PhysicsWorld::default()],
            rbd_sim_params: vec![RbdSimParams::tgs_soft()],
            rbd_dirty: false,
            rbd2gpu: vec![Coarena::new()],
            mpm_body2gpu: Arena::new(),
            multibody2gpu: Coarena::new(),
            capacities,
            rbd_pipeline: None,
            mpm_pipeline: None,
            fem_pipeline: None,
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

    /// Returns a mutable reference to the MPM sub-state, allocating an empty one
    /// (sized from the stored capacities) if it doesn’t exist yet.
    fn mpm_or_insert(&mut self, backend: &GpuBackend) -> Result<&mut MpmState, GpuBackendError> {
        if self.mpm.is_none() {
            self.mpm = Some(MpmState::empty(
                backend,
                self.capacities.mpm_grid_size as u32,
            )?);
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

    pub fn insert_rigid_body(&mut self, body: RigidBody, collider: Collider, coupling: RbdCoupling) -> RigidBodyHandle {
        self.insert_rigid_body_in(0, body, collider, coupling)
    }

    /// Inserts a body + collider into environment `env`.
    pub fn insert_rigid_body_in(&mut self, env: usize, body: RigidBody, collider: Collider, coupling: RbdCoupling) -> RigidBodyHandle {
        let (handle, _) = self.rbd_envs[env].insert(body, collider);
        self.rbd2gpu[env].insert(handle.0, GpuRigidBodyRef {
            coupling,
            gpu_id: u32::MAX
        });
        self.rbd_dirty = true;
        handle
    }

    /// Inserts a rigid-body without any attached collider (e.g. a joint anchor).
    pub fn insert_body(&mut self, body: RigidBody, coupling: RbdCoupling) -> RigidBodyHandle {
        self.insert_body_in(0, body, coupling)
    }

    /// Inserts a collider-less rigid-body into environment `env`.
    pub fn insert_body_in(&mut self, env: usize, body: RigidBody, coupling: RbdCoupling) -> RigidBodyHandle {
        let handle = self.rbd_envs[env].insert_body(body);
        self.rbd2gpu[env].insert(handle.0, GpuRigidBodyRef {
            coupling,
            gpu_id: u32::MAX
        });
        self.rbd_dirty = true;
        handle
    }

    /// Attaches a collider to an existing body (or inserts a parent-less one) in
    /// environment `env`.
    pub fn insert_collider_in(&mut self, env: usize, collider: Collider, parent: Option<RigidBodyHandle>) -> ColliderHandle {
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

    /// Appends MPM particles to the simulation and returns their handles.
    pub fn add_particles(
        &mut self,
        backend: &GpuBackend,
        particles: Vec<Particle>,
    ) -> Result<Vec<NexusParticleHandle>, GpuBackendError> {
        // let mpm = self.mpm_or_insert(backend)?;
        // // 1. Append to the particle buffer.
        // let base = mpm.particles.len() as u32;
        // mpm.particles.append(backend, &particles)?;
        //
        // // 2. Update particle2gpu and return the handles.
        // let handles = (0..particles.len() as u32)
        //     .map(|i| NexusParticleHandle(self.particle2gpu.insert(base + i)))
        //     .collect();
        // Ok(handles)
        todo!()
    }

    pub async fn finalize(&mut self, backend: &GpuBackend) -> Result<(), GpuBackendError> {
        if self.rbd_dirty {
            // NOTE: this rebuilds the whole GPU rbd state from the rapier worlds
            //       (one per environment / batch). It is correct to call more
            //       than once but always does a full rebuild — incremental
            //       insertion isn't wired yet.
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
            let rbd_state = RbdState::from_rapier(backend, &environments);

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
        Ok(())
    }

    /// Advances the physics simulation by one GPU timestep.
    ///
    /// The compute pipelines are compiled lazily the first time their
    /// sub-state is stepped, so the initial call is more expensive (shader
    /// compilation) than the subsequent ones.
    ///
    /// In addition, resources are loaded lazily on the GPU, so the first step
    /// after inserting/removing entities can be slower too. Call `Self::finalize`
    /// to pay that cost upfront.
    pub async fn simulate(&mut self, backend: &GpuBackend, mut timestamps: Option<&mut GpuTimestamps>) -> Result<(), GpuBackendError> {
        self.finalize(backend).await?;

        if let Some(timestamps) = &mut timestamps {
            timestamps.reset();
        }

        let t0 = web_time::Instant::now();

        // Rigid-bodies. `auto_resize_buffers` grows the collision-pair / coloring
        // buffers when the previous step overflowed them.
        if let Some(rbd) = self.rbd.as_mut() {
            let pipeline = self
                .rbd_pipeline
                .get_or_insert_with(|| RbdPipeline::from_backend(backend));
            self.rbd_stats = pipeline.step(backend, rbd, timestamps.as_deref_mut())?;
            let _ = backend.synchronize();
            self.rbd_stats.total_simulation_time_without_readback = t0.elapsed();
            pipeline.auto_resize_buffers(backend, rbd).await;
        }

        // MPM continuum.
        if let Some(mpm) = self.mpm.as_mut() {
            if self.mpm_pipeline.is_none() {
                if let Ok(pipeline) = MpmPipeline::new(backend) {
                    self.mpm_pipeline = Some(pipeline);
                }
            }
            if let Some(pipeline) = self.mpm_pipeline.as_ref() {
                let _ = pipeline.step(backend, mpm, timestamps.as_deref_mut());
                let _ = backend.synchronize();
            }
        }

        // FEM soft-bodies.
        if let Some(fem) = self.fem.as_mut() {
            if self.fem_pipeline.is_none() {
                if let Ok(pipeline) = FemPipeline::new(backend) {
                    self.fem_pipeline = Some(pipeline);
                }
            }
            if let Some(pipeline) = self.fem_pipeline.as_ref() {
                let _ = pipeline.step(backend, fem, timestamps.as_deref_mut());
                let _ = backend.synchronize();
            }
        }

        // Read timestamp results.
        // TODO: the caller should probably be responsible for that.
        if let Some(timestamps) = timestamps {
            if let Ok(results) = timestamps.read(backend).await {
                let mut aggregated: Vec<(String, f64)> = Vec::new();
                for r in &results {
                    if let Some(existing) = aggregated.iter_mut().find(|(label, _)| label == &r.label) {
                        existing.1 += r.duration_ms;
                    } else {
                        aggregated.push((r.label.clone(), r.duration_ms));
                    }
                }
                self.rbd_stats.gpu_total_time = aggregated.iter().map(|e| e.1).sum();
                self.rbd_stats.gpu_pass_times = aggregated;
            }
        }
        self.rbd_stats.total_simulation_time_with_readback = t0.elapsed();

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

    /// Removes the given particles from the simulation.
    pub fn remove_particles(
        &mut self,
        backend: &GpuBackend,
        particles: &[NexusParticleHandle],
    ) -> Result<(), GpuBackendError> {
        // let Some(mpm) = self.mpm.as_mut() else {
        //     return Ok(());
        // };
        //
        // // 1. Resolve the GPU slots, sorted descending so that each removal
        // //    doesn’t shift the slots of the not-yet-removed particles.
        // let mut gpu_ids: Vec<u32> = particles
        //     .iter()
        //     .filter_map(|h| self.particle2gpu.get(h.0).copied())
        //     .collect();
        // gpu_ids.sort_unstable_by(|a, b| b.cmp(a));
        // gpu_ids.dedup();
        //
        // // Drop the handles from the map up-front.
        // for h in particles {
        //     self.particle2gpu.remove(h.0);
        // }
        //
        // // 2. Shift-remove from the GPU buffers and patch the remaining slots.
        // for gpu_id in gpu_ids {
        //     mpm.particles
        //         .shift_remove(backend, gpu_id as usize..gpu_id as usize + 1)?;
        //     // 3. Every particle above the removed slot shifted down by one.
        //     for (_, slot) in self.particle2gpu.iter_mut() {
        //         if *slot > gpu_id {
        //             *slot -= 1;
        //         }
        //     }
        // }

        Ok(())
    }

    /// Removes a contiguous range of particle slots from the simulation.
    ///
    /// This is the cheaper bulk variant of [`Self::remove_particles`] when the
    /// GPU slot range is known directly (e.g. a whole emitted chunk).
    pub fn remove_particles_range(
        &mut self,
        backend: &GpuBackend,
        range: impl RangeBounds<usize> + Clone,
    ) -> Result<(), GpuBackendError> {
        // let Some(mpm) = self.mpm.as_mut() else {
        //     return Ok(());
        // };
        //
        // let start = match range.start_bound() {
        //     std::ops::Bound::Included(i) => *i,
        //     std::ops::Bound::Excluded(i) => *i + 1,
        //     std::ops::Bound::Unbounded => 0,
        // };
        // let removed = mpm.particles.shift_remove(backend, range)?;
        //
        // // Update particle2gpu: drop the handles that fell inside the removed
        // // range, and shift down every handle that pointed past it.
        // self.particle2gpu.retain(|_, slot| {
        //     let s = *slot as usize;
        //     if s < start {
        //         true
        //     } else if s < start + removed {
        //         false
        //     } else {
        //         *slot -= removed as u32;
        //         true
        //     }
        // });

        Ok(())
    }
}
