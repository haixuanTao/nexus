use crate::fem::pipeline::FemState;
use crate::mpm::{pipeline::MpmState, solver::Particle};
use crate::rapier::data::{Arena, Coarena, Index};
use crate::rapier::prelude::{Collider, RigidBody};
use crate::rbd::dynamics::body::BodyCoupling;
use crate::rbd::pipeline::RbdState;
use khal::backend::{GpuBackend, GpuBackendError};
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
// TODO: implement sensible defaults.
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
    // Maps each nexus rigid-body handle to its global slot in the rbd GPU buffers.
    rbd2gpu: Arena<u32>,
    // Maps a coupled rigid-body (keyed by its `rbd2gpu` handle) to its slot in
    // the MPM body buffers.
    mpm_bodies2gpu: Coarena<u32>,
    // Maps each nexus particle handle to its slot in the MPM particle buffers.
    particle2gpu: Arena<u32>,
    // TODO: keep track of whether there is any non-fixed rigid-body (if there isn’t, we can
    //       skip the rbd pipeline entirely).

    // Initial capacities used to allocate the states lazily.
    capacities: NexusCapacities,
}

impl NexusState {
    /// Creates an empty state. The GPU sub-states are allocated lazily (sized
    /// from `capacities`) the first time matching content is added.
    pub fn new(capacities: NexusCapacities) -> Self {
        Self {
            rbd: None,
            mpm: None,
            fem: None,
            rbd2gpu: Arena::new(),
            mpm_bodies2gpu: Coarena::new(),
            particle2gpu: Arena::new(),
            capacities,
        }
    }

    /// Reserves additional capacity in the handle maps to avoid reallocations
    /// when a known number of bodies/particles is about to be inserted.
    pub fn reserve(&mut self, additional: NexusCapacities) {
        self.rbd2gpu
            .reserve(additional.rbd_body_batches * additional.rbd_body_capacity);
        self.mpm_bodies2gpu
            .reserve(additional.rbd_body_batches * additional.rbd_body_capacity);
        self.particle2gpu.reserve(additional.mpm_particles_capacity);
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

    // TODO (later): any value in being able to init multiple batches simultaneously?
    //       Maybe we should consider reducing the amount of stuffs we allow
    //       to change across batches.
    /// Inserts a set of rigid-bodies into the given simulation `batch_id` and
    /// returns their handles.
    pub fn add_rigid_bodies(
        &mut self,
        backend: &GpuBackend,
        bodies: Vec<(RigidBody, Collider, RbdCoupling)>,
        batch_id: usize,
    ) -> Result<Vec<NexusRbdHandle>, GpuBackendError> {
        // 1. Insert the rigid-bodies into nexus-rbd.
        let rbd_pairs: Vec<(RigidBody, Collider)> = bodies
            .iter()
            .map(|(rb, co, _)| (rb.clone(), co.clone()))
            .collect();
        let rbd = self.rbd_or_insert(backend);
        let gpu_ids = rbd.append_bodies(backend, &rbd_pairs, batch_id)?;
        debug_assert_eq!(gpu_ids.len(), bodies.len());

        // 2. Register the handles, mapping each to its rbd GPU slot.
        let handles: Vec<NexusRbdHandle> = gpu_ids
            .iter()
            .map(|&gpu_id| NexusRbdHandle(self.rbd2gpu.insert(gpu_id)))
            .collect();

        // 3. For coupled bodies, also insert them into the MPM body buffers.
        //
        // NOTE: only the body set itself is updated here. Two-way (CPIC)
        //       coupling additionally needs the per-collider boundary-condition
        //       material, the rigid-body impulse accumulators, and the sampled
        //       rigid particles to be grown in lockstep — none of which have an
        //       incremental API yet (and CPIC is capped at 16 colliders). That
        //       resampling is left as a follow-up; one-way coupling (body acts
        //       as a moving obstacle for the continuum) already works with just
        //       the body set populated.
        let coupled: Vec<(usize, (RigidBody, Collider, BodyCoupling))> = bodies
            .into_iter()
            .enumerate()
            .filter_map(|(i, (rb, co, coupling))| {
                coupling.body_coupling().map(|c| (i, (rb, co, c)))
            })
            .collect();

        if !coupled.is_empty() {
            let coupled_descs: Vec<(RigidBody, Collider, BodyCoupling)> =
                coupled.iter().map(|(_, d)| d.clone()).collect();
            let mpm = self.mpm_or_insert(backend)?;
            let mpm_ids = mpm.bodies.append_rapier(backend, &coupled_descs)?;

            // Map each coupled body's handle (found via its original index) to
            // its freshly-allocated MPM body slot.
            for ((orig_idx, _), mpm_id) in coupled.iter().zip(mpm_ids) {
                self.mpm_bodies2gpu.insert(handles[*orig_idx].0, mpm_id);
            }
        }

        Ok(handles)
    }

    /// Appends MPM particles to the simulation and returns their handles.
    pub fn add_particles(
        &mut self,
        backend: &GpuBackend,
        particles: Vec<Particle>,
    ) -> Result<Vec<NexusParticleHandle>, GpuBackendError> {
        let mpm = self.mpm_or_insert(backend)?;
        // 1. Append to the particle buffer.
        let base = mpm.particles.len() as u32;
        mpm.particles.append(backend, &particles)?;

        // 2. Update particle2gpu and return the handles.
        let handles = (0..particles.len() as u32)
            .map(|i| NexusParticleHandle(self.particle2gpu.insert(base + i)))
            .collect();
        Ok(handles)
    }

    /// Removes the given rigid-bodies from the simulation.
    pub fn remove_rigid_bodies(
        &mut self,
        backend: &GpuBackend,
        bodies: &[NexusRbdHandle],
    ) -> Result<(), GpuBackendError> {
        // 1. Resolve the rbd GPU slots of the bodies to remove.
        let gpu_ids: Vec<u32> = bodies
            .iter()
            .filter_map(|h| self.rbd2gpu.get(h.0).copied())
            .collect();

        // 2. Swap-remove them from the rbd GPU buffers. `remove_bodies` returns
        //    the slot relocations it performed (`(from, to)`) so we can patch the
        //    handle map.
        let remaps = match self.rbd.as_mut() {
            Some(rbd) => rbd.remove_bodies(backend, &gpu_ids)?,
            None => Vec::new(),
        };

        // 3. Drop the removed handles, then patch the relocated slots.
        for h in bodies {
            self.rbd2gpu.remove(h.0);
        }
        for (from, to) in remaps {
            for (_, slot) in self.rbd2gpu.iter_mut() {
                if *slot == from {
                    *slot = to;
                }
            }
        }

        // 4. Mirror the removal on the MPM coupling side. The MPM body buffers
        //    are densely packed, so a shift-remove relocates every higher slot;
        //    we rebuild the (handle -> MPM slot) map accordingly.
        //
        //    NOTE: as on the insertion side, the per-collider materials, impulse
        //    accumulators and sampled rigid particles are not resized here yet.
        let mut mpm_remove: Vec<u32> = bodies
            .iter()
            .filter_map(|h| self.mpm_bodies2gpu.get(h.0).copied())
            .collect();
        if !mpm_remove.is_empty() {
            mpm_remove.sort_unstable_by(|a, b| b.cmp(a));
            mpm_remove.dedup();

            let removed: std::collections::HashSet<Index> = bodies.iter().map(|h| h.0).collect();
            let survivors: Vec<(Index, u32)> = self
                .mpm_bodies2gpu
                .iter()
                .filter(|(idx, _)| !removed.contains(idx))
                .map(|(idx, &slot)| (idx, slot))
                .collect();

            if let Some(mpm) = self.mpm.as_mut() {
                for slot in &mpm_remove {
                    mpm.bodies
                        .shift_remove(backend, *slot as usize..*slot as usize + 1)?;
                }
            }

            let mut new_map = Coarena::new();
            for (idx, slot) in survivors {
                let shift = mpm_remove.iter().filter(|&&r| r < slot).count() as u32;
                new_map.insert(idx, slot - shift);
            }
            self.mpm_bodies2gpu = new_map;
        }

        Ok(())
    }

    /// Removes the given particles from the simulation.
    pub fn remove_particles(
        &mut self,
        backend: &GpuBackend,
        particles: &[NexusParticleHandle],
    ) -> Result<(), GpuBackendError> {
        let Some(mpm) = self.mpm.as_mut() else {
            return Ok(());
        };

        // 1. Resolve the GPU slots, sorted descending so that each removal
        //    doesn’t shift the slots of the not-yet-removed particles.
        let mut gpu_ids: Vec<u32> = particles
            .iter()
            .filter_map(|h| self.particle2gpu.get(h.0).copied())
            .collect();
        gpu_ids.sort_unstable_by(|a, b| b.cmp(a));
        gpu_ids.dedup();

        // Drop the handles from the map up-front.
        for h in particles {
            self.particle2gpu.remove(h.0);
        }

        // 2. Shift-remove from the GPU buffers and patch the remaining slots.
        for gpu_id in gpu_ids {
            mpm.particles
                .shift_remove(backend, gpu_id as usize..gpu_id as usize + 1)?;
            // 3. Every particle above the removed slot shifted down by one.
            for (_, slot) in self.particle2gpu.iter_mut() {
                if *slot > gpu_id {
                    *slot -= 1;
                }
            }
        }

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
        let Some(mpm) = self.mpm.as_mut() else {
            return Ok(());
        };

        let start = match range.start_bound() {
            std::ops::Bound::Included(i) => *i,
            std::ops::Bound::Excluded(i) => *i + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let removed = mpm.particles.shift_remove(backend, range)?;

        // Update particle2gpu: drop the handles that fell inside the removed
        // range, and shift down every handle that pointed past it.
        self.particle2gpu.retain(|_, slot| {
            let s = *slot as usize;
            if s < start {
                true
            } else if s < start + removed {
                false
            } else {
                *slot -= removed as u32;
                true
            }
        });

        Ok(())
    }
}
