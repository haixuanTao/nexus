//! Physics simulation pipeline orchestrating broad-phase, narrow-phase, and constraint solving.
//!
//! This module provides the high-level physics pipeline that coordinates all stages of a physics
//! simulation step on the GPU. The pipeline manages collision detection, contact generation,
//! constraint solving, and integration.

use crate::broad_phase::{GpuNarrowPhase, Lbvh, LbvhState};
use crate::dynamics::{
    ColoringArgs, GpuColoring, GpuImpulseJointSet, GpuJointSolver, GpuMpropsUpdate, GpuSolver,
    GpuWarmstart, JointSolverArgs, SolverArgs, warmstart::WarmstartArgs,
};
#[cfg(feature = "dim3")]
use crate::dynamics::{GpuMultibodySet, GpuMultibodySolver};
use crate::math::{Pose, Vector};
use crate::queries::GpuIndexedContact;
use crate::shaders::PaddedVector;
use crate::shaders::broad_phase::{LbvhNode, NarrowPhasePfmPair};
use crate::shaders::dynamics::{
    LocalMassProperties as GpuLocalMassProperties, SimParams as GpuSimParams, TwoBodyConstraint,
    TwoBodyConstraintBuilder, Velocity as GpuVelocity,
    WorldMassProperties as GpuWorldMassProperties,
};
use crate::shaders::shapes::Shape;
use crate::utils::{GpuPrefixSum, PrefixSumWorkspace};
use khal::Shader;

use khal::BufferUsages;
use khal::backend::{Backend, Encoder, GpuBackend, GpuTimestamps};
use std::time::Duration;
use vortx::tensor::Tensor;

#[cfg(feature = "from_rapier")]
use {
    crate::math::Point,
    crate::rapier::dynamics::{ImpulseJointSet, MultibodyJointSet, RigidBodySet},
    crate::rapier::geometry::ColliderSet,
    crate::shapes::ShapeBuffers,
    crate::shapes::shape_from_parry,
    std::collections::HashMap,
};

/// Performance statistics collected during a physics simulation step.
///
/// This structure tracks timing and iteration counts for various stages of the physics pipeline,
/// useful for profiling and optimization.
#[derive(Default, Clone, Debug)]
pub struct RunStats {
    /// Number of colors used in the graph coloring algorithm for parallel constraint solving.
    pub num_colors: u32,
    /// Duration from the start of the step until collision pair count is read back from GPU.
    pub start_to_pairs_count_time: Duration,
    /// Time spent on the graph coloring algorithm.
    pub coloring_time: Duration,
    /// Number of iterations the coloring algorithm took to converge.
    pub coloring_iterations: u32,
    /// Time spent on the fallback coloring method (if the primary method failed).
    pub coloring_fallback_time: Duration,
    /// Total simulation time including GPU-to-CPU readbacks.
    pub total_simulation_time_with_readback: Duration,
    /// Per-pass GPU timestamp durations (label, milliseconds).
    pub gpu_pass_times: Vec<(String, f64)>,
    /// Total GPU time across all measured passes, in milliseconds.
    pub gpu_total_time: f64,
}

impl RunStats {
    /// Returns the total simulation time in milliseconds.
    pub fn total_simulation_time_ms(&self) -> f32 {
        self.total_simulation_time_with_readback.as_secs_f32() * 1000.0
    }
}

/// GPU-resident physics simulation state containing all rigid bodies, shapes, and solver data.
///
/// This structure holds all the buffers needed for a complete physics simulation on the GPU:
/// - Rigid body poses, velocities, and mass properties
/// - Collision shapes and contact data
/// - Constraints and solver state
/// - Auxiliary data structures (LBVH, prefix sum workspace, etc.)
///
/// The state can be initialized from CPU-side Rapier data structures and then updated
/// entirely on the GPU each frame.
pub struct GpuPhysicsState {
    num_batches: u32,
    num_colliders_per_batch: u32,
    num_solver_iterations: u32,
    sim_params: Tensor<GpuSimParams>,
    poses: Tensor<Pose>,
    local_mprops: Tensor<GpuLocalMassProperties>,
    mprops: Tensor<GpuWorldMassProperties>,
    vels: Tensor<GpuVelocity>,
    solver_vels: Tensor<GpuVelocity>,
    solver_vels_out: Tensor<GpuVelocity>,
    solver_vels_inc: Tensor<GpuVelocity>,
    vertex_buffers: Tensor<PaddedVector>,
    index_buffers: Tensor<u32>,
    shapes: Tensor<Shape>,
    num_shapes: Tensor<u32>,
    collision_pairs: Tensor<[u32; 2]>,
    collision_pairs_len: Tensor<u32>,
    #[allow(dead_code)]
    collision_pairs_len_staging: Tensor<u32>,
    collision_pairs_indirect: Tensor<[u32; 3]>,
    collision_pairs_batch_capacity: Tensor<u32>,
    contacts_batch_capacity: Tensor<u32>,
    colliders_batch_capacity: Tensor<u32>,
    pfm_pairs: Tensor<NarrowPhasePfmPair>,
    pfm_pairs_len: Tensor<u32>,
    pfm_pairs_indirect: Tensor<[u32; 3]>,
    contacts: Tensor<GpuIndexedContact>,
    contacts_len: Tensor<u32>,
    contacts_indirect: Tensor<[u32; 3]>,
    new_constraints: Tensor<TwoBodyConstraint>,
    new_constraint_builders: Tensor<TwoBodyConstraintBuilder>,
    new_constraints_counts: Tensor<u32>,
    new_body_constraint_ids: Tensor<u32>,
    old_constraints: Tensor<TwoBodyConstraint>,
    old_constraint_builders: Tensor<TwoBodyConstraintBuilder>,
    old_constraints_counts: Tensor<u32>,
    old_body_constraint_ids: Tensor<u32>,
    constraints_colors: Tensor<u32>,
    colored: Tensor<u32>,
    constraints_rands: Tensor<u32>,
    curr_color: Tensor<u32>,
    uncolored: Tensor<u32>,
    uncolored_staging: Tensor<u32>,
    lbvh: LbvhState,
    joints: GpuImpulseJointSet,
    #[cfg(feature = "dim3")]
    multibodies: GpuMultibodySet,
    prefix_sum_workspace: PrefixSumWorkspace,
}

#[cfg(feature = "from_rapier")]
impl GpuPhysicsState {
    /// Creates a new GPU physics state from per-environment Rapier data structures.
    ///
    /// Environments with fewer colliders/joints are padded with dummy fixed bodies.
    /// Panics if any rigid body has more than one collider attached.
    pub fn from_rapier(
        backend: &GpuBackend,
        environments: &[(
            &RigidBodySet,
            &ColliderSet,
            &ImpulseJointSet,
            &MultibodyJointSet,
            &GpuSimParams,
        )],
    ) -> Self {
        let num_batches = environments.len() as u32;
        let max_colliders = environments
            .iter()
            .map(|(_, c, _, _, _)| c.len())
            .max()
            .unwrap_or(0);

        let mut all_poses = Vec::new();
        let mut all_local_mprops = Vec::new();
        let mut all_mprops = Vec::new();
        let mut all_shapes = Vec::new();
        let mut all_num_shapes = Vec::new();
        let mut shape_buffers = ShapeBuffers::default();
        let mut joint_envs: Vec<(
            &ImpulseJointSet,
            HashMap<crate::rapier::dynamics::RigidBodyHandle, u32>,
        )> = Vec::new();

        // Collect per-batch sim params, adjusting dt for substeps.
        let num_solver_iterations = environments
            .iter()
            .map(|(_, _, _, _, sp)| sp.num_solver_iterations)
            .max()
            .unwrap_or(4);
        let all_sim_params: Vec<GpuSimParams> = environments
            .iter()
            .map(|(_, _, _, _, sp)| {
                let mut sp = **sp;
                sp.dt /= sp.num_solver_iterations as f32;
                sp
            })
            .collect();
        // Pick representative dt (outer dt, not the per-substep one) from any batch.
        let multibody_dt = environments
            .first()
            .map(|(_, _, _, _, sp)| sp.dt)
            .unwrap_or(1.0 / 60.0);

        // Dummy data for padding shorter environments.
        let dummy_pose = Pose::default();
        let dummy_local_mprops = GpuLocalMassProperties::default();
        let dummy_mprops = GpuWorldMassProperties::default();

        #[cfg(feature = "dim3")]
        let mut multibody_envs: Vec<(
            &MultibodyJointSet,
            HashMap<crate::rapier::dynamics::RigidBodyHandle, u32>,
            &RigidBodySet,
        )> = Vec::new();

        for (bodies, colliders, impulse_joints, multibody_joints, _sim_params) in environments {
            let env_collider_count = colliders.len();
            all_num_shapes.push(env_collider_count as u32);
            let mut body_ids = HashMap::new();
            let mut env_collider_idx = 0u32;

            for (_, co) in colliders.iter() {
                let parent = co.parent().map(|h| &bodies[h]);

                if let Some(parent) = parent {
                    assert_eq!(
                        parent.colliders().len(),
                        1,
                        "Only bodies with exactly one collider are supported."
                    );
                }

                let mut local_mprops = GpuLocalMassProperties::default();
                let mut mprops = GpuWorldMassProperties {
                    com: parent
                        .map(|body| body.translation())
                        .unwrap_or(Vector::ZERO),
                    ..Default::default()
                };
                if parent.map(|b| !b.is_dynamic()).unwrap_or(true) {
                    local_mprops.inv_mass = Vector::ZERO;
                    #[cfg(feature = "dim3")]
                    {
                        local_mprops.inv_principal_inertia = glamx::Vec3::ZERO;
                    }
                    #[cfg(feature = "dim2")]
                    {
                        local_mprops.inv_inertia = 0.0;
                    }
                    mprops.inv_mass = Vector::ZERO;
                    #[cfg(feature = "dim3")]
                    {
                        mprops.inv_inertia = glamx::Mat4::ZERO;
                    }
                    #[cfg(feature = "dim2")]
                    {
                        mprops.inv_inertia = 0.0;
                    }
                }

                if let Some(h) = co.parent() {
                    body_ids.insert(h, env_collider_idx);
                }

                env_collider_idx += 1;
                all_local_mprops.push(local_mprops);
                all_mprops.push(mprops);
                all_shapes.push(
                    shape_from_parry(co.shape(), &mut shape_buffers).expect("Unsupported shape"),
                );
                all_poses.push(*co.position());
            }

            // Pad to max_colliders with dummy fixed bodies.
            let dummy_shape = all_shapes.last().copied().unwrap_or_default();
            for _ in env_collider_count..max_colliders {
                all_poses.push(dummy_pose);
                all_local_mprops.push(dummy_local_mprops);
                all_mprops.push(dummy_mprops);
                all_shapes.push(dummy_shape);
            }

            #[cfg(feature = "dim3")]
            multibody_envs.push((multibody_joints, body_ids.clone(), bodies));
            joint_envs.push((impulse_joints, body_ids));
        }

        // NOTE: GPU doesn't like empty storage buffer bindings so add dummy data
        //       instead of leaving them empty (which is fine considering they are
        //       not referenced by any collider).
        if shape_buffers.vertices.is_empty() {
            shape_buffers.vertices.push(Point::ZERO.into());
        }
        if shape_buffers.indices.is_empty() {
            shape_buffers.indices.extend_from_slice(&[0; 3]);
        }

        let vertex_buffers =
            Tensor::vector(backend, &shape_buffers.vertices, BufferUsages::STORAGE).unwrap();
        let index_buffers =
            Tensor::vector(backend, &shape_buffers.indices, BufferUsages::STORAGE).unwrap();

        let joint_env_refs: Vec<(
            &ImpulseJointSet,
            &HashMap<crate::rapier::dynamics::RigidBodyHandle, u32>,
        )> = joint_envs
            .iter()
            .map(|(joints, body_ids)| (*joints, body_ids))
            .collect();
        let joints = GpuImpulseJointSet::from_rapier(backend, &joint_env_refs);

        // Convert multibodies (3D only).
        #[cfg(feature = "dim3")]
        let multibodies = {
            let mb_refs: Vec<(
                &MultibodyJointSet,
                &HashMap<crate::rapier::dynamics::RigidBodyHandle, u32>,
                &RigidBodySet,
            )> = multibody_envs
                .iter()
                .map(|(mb, ids, bodies)| (*mb, ids, *bodies))
                .collect();
            let mut mb = GpuMultibodySet::from_rapier(
                backend,
                &mb_refs,
                [0.0, -9.81, 0.0],
            );
            mb.set_dt(backend, multibody_dt);
            mb
        };

        // Mark multibody-controlled bodies as kinematic (inv_mass = 0) in the shared
        // body buffers so the rigid-body pipeline leaves them alone. The multibody
        // solver owns their masses internally.
        #[cfg(feature = "dim3")]
        {
            for (batch_idx, (mb_set, body_ids, _)) in multibody_envs.iter().enumerate() {
                let batch_offset = batch_idx * max_colliders;
                for mb in mb_set.multibodies() {
                    for link in mb.links() {
                        if let Some(&rb_local_id) = body_ids.get(&link.rigid_body_handle()) {
                            let global = batch_offset + rb_local_id as usize;
                            all_local_mprops[global].inv_mass = Vector::ZERO;
                            all_local_mprops[global].inv_principal_inertia = glamx::Vec3::ZERO;
                            all_mprops[global].inv_mass = Vector::ZERO;
                            all_mprops[global].inv_inertia = glamx::Mat4::ZERO;
                        }
                    }
                }
            }
        }

        let num_colliders_per_batch = max_colliders;
        let num_bodies_total = num_colliders_per_batch * num_batches as usize;

        let all_vels = vec![GpuVelocity::default(); num_bodies_total];
        let storage: BufferUsages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let shapes = Tensor::vector(backend, &all_shapes, storage).unwrap();

        let num_shapes = Tensor::vector(
            backend,
            &all_num_shapes,
            BufferUsages::STORAGE | BufferUsages::UNIFORM,
        )
        .unwrap();

        let colliders_batch_capacity = Tensor::scalar(
            backend,
            num_colliders_per_batch as u32,
            BufferUsages::STORAGE | BufferUsages::UNIFORM,
        )
        .unwrap();

        const DEFAULT_CONTACT_COUNTS: u32 = 32; // 1024;
        let collision_pairs =
            Tensor::vector_uninit(backend, DEFAULT_CONTACT_COUNTS * num_batches, storage).unwrap();
        let collision_pairs_len = Tensor::vector_uninit(
            backend,
            num_batches,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();
        let collision_pairs_len_staging =
            Tensor::scalar_uninit(backend, BufferUsages::MAP_READ | BufferUsages::COPY_DST)
                .unwrap();
        let collision_pairs_indirect =
            Tensor::scalar_uninit(backend, BufferUsages::STORAGE | BufferUsages::INDIRECT).unwrap();
        let collision_pairs_batch_capacity = Tensor::scalar(
            backend,
            DEFAULT_CONTACT_COUNTS,
            BufferUsages::STORAGE | BufferUsages::UNIFORM,
        )
        .unwrap();
        let contacts_batch_capacity = Tensor::scalar(
            backend,
            DEFAULT_CONTACT_COUNTS,
            BufferUsages::STORAGE | BufferUsages::UNIFORM,
        )
        .unwrap();

        let contacts =
            Tensor::vector_uninit(backend, DEFAULT_CONTACT_COUNTS * num_batches, storage).unwrap();
        let contacts_len = Tensor::vector_uninit(
            backend,
            num_batches,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();
        let contacts_indirect =
            Tensor::scalar_uninit(backend, BufferUsages::STORAGE | BufferUsages::INDIRECT).unwrap();
        let pfm_pairs_indirect =
            Tensor::scalar_uninit(backend, BufferUsages::STORAGE | BufferUsages::INDIRECT).unwrap();
        let pfm_pairs =
            Tensor::vector_uninit(backend, DEFAULT_CONTACT_COUNTS * num_batches, storage).unwrap();
        let pfm_pairs_len = Tensor::vector_uninit(
            backend,
            num_batches,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();
        let old_constraints =
            Tensor::vector_uninit(backend, DEFAULT_CONTACT_COUNTS * num_batches, storage).unwrap();
        let old_constraint_builders =
            Tensor::vector_uninit(backend, DEFAULT_CONTACT_COUNTS * num_batches, storage).unwrap();
        let new_constraints =
            Tensor::vector_uninit(backend, DEFAULT_CONTACT_COUNTS * num_batches, storage).unwrap();
        let new_constraint_builders =
            Tensor::vector_uninit(backend, DEFAULT_CONTACT_COUNTS * num_batches, storage).unwrap();
        let constraints_colors =
            Tensor::vector_uninit(backend, DEFAULT_CONTACT_COUNTS * num_batches, storage).unwrap();
        let colored =
            Tensor::vector_uninit(backend, DEFAULT_CONTACT_COUNTS * num_batches, storage).unwrap();
        let constraints_rands =
            Tensor::vector_uninit(backend, DEFAULT_CONTACT_COUNTS * num_batches, storage).unwrap();
        let old_constraints_counts = Tensor::vector_uninit(
            backend,
            num_colliders_per_batch as u32 * num_batches,
            storage,
        )
        .unwrap();
        let new_constraints_counts = Tensor::vector_uninit(
            backend,
            num_colliders_per_batch as u32 * num_batches,
            storage,
        )
        .unwrap();
        let old_body_constraint_ids =
            Tensor::vector_uninit(backend, DEFAULT_CONTACT_COUNTS * 2 * num_batches, storage)
                .unwrap();
        let new_body_constraint_ids =
            Tensor::vector_uninit(backend, DEFAULT_CONTACT_COUNTS * 2 * num_batches, storage)
                .unwrap();

        let lbvh_usages = if crate::VALIDATE_LBVH_TOPOLOGY {
            BufferUsages::STORAGE | BufferUsages::COPY_SRC
        } else {
            BufferUsages::STORAGE
        };

        Self {
            num_batches,
            num_colliders_per_batch: num_colliders_per_batch as u32,
            num_solver_iterations,
            sim_params: Tensor::vector(backend, &all_sim_params, BufferUsages::STORAGE).unwrap(),
            vels: Tensor::vector(backend, &all_vels, storage).unwrap(),
            solver_vels: Tensor::vector(backend, &all_vels, storage).unwrap(),
            solver_vels_out: Tensor::vector(backend, &all_vels, storage).unwrap(),
            solver_vels_inc: Tensor::vector(backend, &all_vels, storage).unwrap(),
            joints,
            #[cfg(feature = "dim3")]
            multibodies,
            local_mprops: Tensor::vector(backend, &all_local_mprops, storage).unwrap(),
            mprops: Tensor::vector(backend, &all_mprops, storage).unwrap(),
            poses: Tensor::vector(
                backend,
                &all_poses,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            )
            .unwrap(),
            vertex_buffers,
            index_buffers,
            shapes,
            num_shapes,
            collision_pairs,
            collision_pairs_len,
            collision_pairs_len_staging,
            collision_pairs_indirect,
            collision_pairs_batch_capacity,
            contacts_batch_capacity,
            colliders_batch_capacity,
            contacts,
            contacts_len,
            contacts_indirect,
            pfm_pairs,
            pfm_pairs_len,
            pfm_pairs_indirect,
            old_constraints,
            old_constraint_builders,
            old_constraints_counts,
            new_constraints,
            new_constraint_builders,
            new_constraints_counts,
            constraints_colors,
            colored,
            constraints_rands,
            curr_color: Tensor::scalar(
                backend,
                0u32,
                BufferUsages::STORAGE
                    | BufferUsages::UNIFORM
                    | BufferUsages::COPY_DST
                    | BufferUsages::COPY_SRC,
            )
            .unwrap(),
            uncolored: Tensor::scalar(
                backend,
                0,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            )
            .unwrap(),
            uncolored_staging: Tensor::scalar(
                backend,
                0,
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            )
            .unwrap(),
            old_body_constraint_ids,
            new_body_constraint_ids,
            prefix_sum_workspace: PrefixSumWorkspace::default(),
            lbvh: LbvhState::with_usages(backend, lbvh_usages),
        }
    }
}

impl GpuPhysicsState {
    /// Returns a reference to the GPU buffer containing rigid body poses.
    ///
    /// The poses are represented as similarity transformations (position + rotation + scale)
    /// in world space.
    pub fn poses(&self) -> &Tensor<Pose> {
        &self.poses
    }

    /// The set of joints part of the simulation.
    pub fn joints(&self) -> &GpuImpulseJointSet {
        &self.joints
    }

    /// Returns a reference to the GPU buffer containing collision shapes.
    ///
    /// Each shape corresponds to one rigid body in the simulation.
    pub fn shapes(&self) -> &Tensor<Shape> {
        &self.shapes
    }

    /// The number of colliders per batch.
    pub fn num_colliders_per_batch(&self) -> u32 {
        self.num_colliders_per_batch
    }

    /// The number of batches.
    pub fn num_batches(&self) -> u32 {
        self.num_batches
    }

    /// The number of solver iterations (max across all environments).
    pub fn num_solver_iterations(&self) -> u32 {
        self.num_solver_iterations
    }
}

/// The main GPU physics pipeline coordinating all simulation stages.
pub struct GpuPhysicsPipeline {
    mprops_update: GpuMpropsUpdate,
    narrow_phase: GpuNarrowPhase,
    solver: GpuSolver,
    joint_solver: GpuJointSolver,
    #[cfg(feature = "dim3")]
    multibody_solver: GpuMultibodySolver,
    prefix_sum: GpuPrefixSum,
    lbvh: Lbvh,
    coloring: GpuColoring,
    warmstart: GpuWarmstart,
}

impl GpuPhysicsPipeline {
    /// Creates a new physics pipeline from a GPU backend.
    ///
    /// This method loads all the compute shaders needed for the physics simulation.
    pub fn from_backend(backend: &GpuBackend) -> Self {
        Self {
            mprops_update: GpuMpropsUpdate::from_backend(backend).unwrap(),
            narrow_phase: GpuNarrowPhase::from_backend(backend).unwrap(),
            solver: GpuSolver::from_backend(backend).unwrap(),
            joint_solver: GpuJointSolver::from_backend(backend).unwrap(),
            #[cfg(feature = "dim3")]
            multibody_solver: GpuMultibodySolver::from_backend(backend).unwrap(),
            prefix_sum: GpuPrefixSum::from_backend(backend).unwrap(),
            lbvh: Lbvh::from_backend(backend),
            coloring: GpuColoring::from_backend(backend).unwrap(),
            warmstart: GpuWarmstart::from_backend(backend).unwrap(),
        }
    }

    /// Executes one physics simulation timestep on the GPU.
    ///
    /// Automatically resizes buffers (next power of two) if collision pair count exceeds capacity.
    pub async fn step(
        &self,
        backend: &GpuBackend,
        state: &mut GpuPhysicsState,
        mut timestamps: Option<&mut GpuTimestamps>,
    ) -> RunStats {
        let mut stats = RunStats::default();
        let t_phase1 = web_time::Instant::now();

        // Phase 0: Multibody step (3D only).
        //
        // Updates each articulated multibody's coords from the previous step's
        // generalized acceleration, then refreshes link poses in the shared pose
        // buffer so the rigid-body pipeline sees the articulated bodies in their
        // new configuration. Contacts and constraints involving multibodies are
        // not currently supported.
        #[cfg(feature = "dim3")]
        {
            if !state.multibodies.is_empty() {
                let mut encoder = backend.begin_encoding();
                let mut pass = encoder.begin_pass("multibody-step", timestamps.as_deref_mut());
                let args = crate::dynamics::MultibodySolverArgs {
                    poses: &mut state.poses,
                    colliders_batch_capacity: &state.colliders_batch_capacity,
                };
                self.multibody_solver
                    .step(&mut pass, &mut state.multibodies, args)
                    .unwrap();
                drop(pass);
                backend.submit(encoder).unwrap();
            }
        }

        // Phase 1: Update mass properties, build LBVH, and find collision pairs
        {
            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass("update-mprops", timestamps.as_deref_mut());

            // Update mass properties
            self.mprops_update
                .dispatch(
                    &mut pass,
                    &mut state.mprops,
                    &state.local_mprops,
                    &state.poses,
                    &state.num_shapes,
                    &state.colliders_batch_capacity,
                    state.num_colliders_per_batch,
                    state.num_batches,
                )
                .unwrap();

            drop(pass);

            // Build LBVH and find collision pairs
            self.lbvh
                .update_tree(
                    backend,
                    &mut encoder,
                    &mut state.lbvh,
                    state.poses.len() as u32,
                    state.num_batches,
                    &state.poses,
                    &state.vertex_buffers,
                    &state.shapes,
                    &state.num_shapes,
                    &state.colliders_batch_capacity,
                    timestamps.as_deref_mut(),
                )
                .unwrap();

            // Debug: validate LBVH topology after tree construction
            if crate::VALIDATE_LBVH_TOPOLOGY {
                backend.submit(encoder).unwrap();

                let num_colliders = state.poses.len() as u32;
                let tree: Vec<LbvhNode> = backend
                    .slow_read_vec(state.lbvh.tree().buffer())
                    .await
                    .unwrap();
                let sorted_colliders: Vec<u32> = backend
                    .slow_read_vec(state.lbvh.sorted_colliders().buffer())
                    .await
                    .unwrap();
                validate_lbvh_topology(&tree, &sorted_colliders, num_colliders);

                encoder = backend.begin_encoding();
                let _pass = encoder.begin_pass("broad-phase-find-pairs", timestamps.as_deref_mut());
            }

            let mut pass = encoder.begin_pass("lbvh-find-pairs", timestamps.as_deref_mut());
            self.lbvh
                .find_pairs(
                    &mut pass,
                    &mut state.lbvh,
                    state.poses.len() as u32,
                    state.num_batches,
                    &state.num_shapes,
                    &state.colliders_batch_capacity,
                    &state.collision_pairs_batch_capacity,
                    &mut state.collision_pairs,
                    &mut state.collision_pairs_len,
                    &mut state.collision_pairs_indirect,
                )
                .unwrap();

            drop(pass);
            backend.submit(encoder).unwrap();
        }

        // Read back collision pair counts (requires CPU-GPU sync)
        let collision_pair_counts: Vec<u32> = backend
            .slow_read_vec(state.collision_pairs_len.buffer())
            .await
            .unwrap();
        let num_collision_pairs = collision_pair_counts.iter().copied().max().unwrap_or(0);
        stats.start_to_pairs_count_time = t_phase1.elapsed();

        // Per-batch capacity for collision pairs and contacts.
        let per_batch_capacity = state.collision_pairs.len() as u32 / state.num_batches;

        // Resize buffers if needed
        if num_collision_pairs >= per_batch_capacity {
            let storage: BufferUsages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
            let desired_len = num_collision_pairs.next_power_of_two();
            let nb = state.num_batches;

            state.collision_pairs =
                Tensor::vector_uninit(backend, desired_len * nb, storage).unwrap();
            state.collision_pairs_batch_capacity = Tensor::scalar(
                backend,
                desired_len,
                BufferUsages::STORAGE | BufferUsages::UNIFORM,
            )
            .unwrap();
            state.contacts_batch_capacity = Tensor::scalar(
                backend,
                desired_len,
                BufferUsages::STORAGE | BufferUsages::UNIFORM,
            )
            .unwrap();

            state.contacts = Tensor::vector_uninit(backend, desired_len * nb, storage).unwrap();
            state.pfm_pairs = Tensor::vector_uninit(backend, desired_len * nb, storage).unwrap();
            state.old_constraints =
                Tensor::vector_uninit(backend, desired_len * nb, storage).unwrap();
            state.old_constraint_builders =
                Tensor::vector_uninit(backend, desired_len * nb, storage).unwrap();
            state.old_body_constraint_ids =
                Tensor::vector_uninit(backend, desired_len * 2 * nb, storage).unwrap();
            state.new_constraints =
                Tensor::vector_uninit(backend, desired_len * nb, storage).unwrap();
            state.new_constraint_builders =
                Tensor::vector_uninit(backend, desired_len * nb, storage).unwrap();
            state.new_body_constraint_ids =
                Tensor::vector_uninit(backend, desired_len * 2 * nb, storage).unwrap();
            state.constraints_colors =
                Tensor::vector_uninit(backend, desired_len * nb, storage).unwrap();
            state.colored = Tensor::vector_uninit(backend, desired_len * nb, storage).unwrap();
            state.constraints_rands =
                Tensor::vector_uninit(backend, desired_len * nb, storage).unwrap();

            // Re-run find_pairs with resized buffers
            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass(
                "broad-phase-find-pairs (after resize)",
                timestamps.as_deref_mut(),
            );
            self.lbvh
                .find_pairs(
                    &mut pass,
                    &mut state.lbvh,
                    state.poses.len() as u32,
                    state.num_batches,
                    &state.num_shapes,
                    &state.colliders_batch_capacity,
                    &state.collision_pairs_batch_capacity,
                    &mut state.collision_pairs,
                    &mut state.collision_pairs_len,
                    &mut state.collision_pairs_indirect,
                )
                .unwrap();
            drop(pass);
            backend.submit(encoder).unwrap();
        }

        // Phase 2: Narrow phase and solver preparation
        {
            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass("narrow-phase", timestamps.as_deref_mut());

            // Narrow phase
            self.narrow_phase
                .dispatch(
                    &mut pass,
                    state.poses.len() as u32,
                    &state.poses,
                    &state.shapes,
                    &state.vertex_buffers,
                    &state.index_buffers,
                    &state.collision_pairs,
                    &state.collision_pairs_len,
                    &state.collision_pairs_indirect,
                    &mut state.contacts,
                    &mut state.contacts_len,
                    &mut state.contacts_indirect,
                    &mut state.pfm_pairs,
                    &mut state.pfm_pairs_len,
                    &mut state.pfm_pairs_indirect,
                    &state.contacts_batch_capacity,
                    &state.colliders_batch_capacity,
                )
                .unwrap();

            drop(pass);
            let mut pass = encoder.begin_pass("solver-prep", timestamps.as_deref_mut());

            // Solver preparation - create args here to avoid borrow conflicts
            let prepare_args = SolverArgs {
                contacts: &state.contacts,
                contacts_len: &state.contacts_len,
                contacts_len_indirect: &state.contacts_indirect,
                constraints: &mut state.new_constraints,
                constraint_builders: &mut state.new_constraint_builders,
                sim_params: &state.sim_params,
                colliders_len: &state.num_shapes,
                poses: &mut state.poses,
                vels: &mut state.vels,
                solver_vels: &mut state.solver_vels,
                solver_vels_out: &state.solver_vels_out,
                solver_vels_inc: &mut state.solver_vels_inc,
                mprops: &state.mprops,
                local_mprops: &state.local_mprops,
                body_constraint_counts: &mut state.new_constraints_counts,
                body_constraint_ids: &mut state.new_body_constraint_ids,
                constraints_colors: &state.constraints_colors,
                curr_color: &mut state.curr_color,
                prefix_sum: &self.prefix_sum,
                num_colors: 0,
                contacts_batch_capacity: &state.contacts_batch_capacity,
                colliders_batch_capacity: &state.colliders_batch_capacity,
                num_batches: state.num_batches,
                num_colliders: state.num_colliders_per_batch,
                num_solver_iterations: state.num_solver_iterations,
            };
            self.solver
                .prepare(
                    backend,
                    &mut pass,
                    prepare_args,
                    &mut state.prefix_sum_workspace,
                )
                .unwrap();

            // Warmstart
            let warmstart_args = WarmstartArgs {
                contacts_len: &state.contacts_len,
                old_body_constraint_counts: &state.old_constraints_counts,
                old_constraint_builders: &state.old_constraint_builders,
                old_body_constraint_ids: &state.old_body_constraint_ids,
                old_constraints: &state.old_constraints,
                new_constraints: &mut state.new_constraints,
                new_constraint_builders: &state.new_constraint_builders,
                contacts_len_indirect: &state.contacts_indirect,
                contacts_batch_capacity: &state.contacts_batch_capacity,
                colliders_batch_capacity: &state.colliders_batch_capacity,
            };

            self.warmstart
                .transfer_warmstart_impulses(&mut pass, warmstart_args)
                .unwrap();

            drop(pass);
            backend.submit(encoder).unwrap();
        }

        // Graph coloring
        let coloring_args = ColoringArgs {
            contacts_len_indirect: &state.contacts_indirect,
            body_constraint_counts: &state.new_constraints_counts,
            body_constraint_ids: &state.new_body_constraint_ids,
            constraints: &state.new_constraints,
            constraints_colors: &mut state.constraints_colors,
            constraints_rands: &mut state.constraints_rands,
            curr_color: &mut state.curr_color,
            uncolored: &mut state.uncolored,
            uncolored_staging: &state.uncolored_staging,
            contacts_len: &state.contacts_len,
            colored: &mut state.colored,
            contacts_batch_capacity: &state.contacts_batch_capacity,
            colliders_batch_capacity: &state.colliders_batch_capacity,
        };

        let num_colors = if let Some(colors) = self
            .coloring
            .dispatch_topo_gc(
                backend,
                coloring_args,
                &mut stats,
                timestamps.as_deref_mut(),
            )
            .await
        {
            colors
        } else {
            // Rebuild coloring_args for luby fallback
            let coloring_args = ColoringArgs {
                contacts_len_indirect: &state.contacts_indirect,
                body_constraint_counts: &state.new_constraints_counts,
                body_constraint_ids: &state.new_body_constraint_ids,
                constraints: &state.new_constraints,
                constraints_colors: &mut state.constraints_colors,
                constraints_rands: &mut state.constraints_rands,
                curr_color: &mut state.curr_color,
                uncolored: &mut state.uncolored,
                uncolored_staging: &state.uncolored_staging,
                contacts_len: &state.contacts_len,
                colored: &mut state.colored,
                contacts_batch_capacity: &state.contacts_batch_capacity,
                colliders_batch_capacity: &state.colliders_batch_capacity,
            };
            self.coloring
                .dispatch_luby(backend, coloring_args, &mut stats)
                .await
        };

        stats.num_colors = num_colors;

        // Create solver_args for solve phase (after coloring is complete)
        let solver_args = SolverArgs {
            contacts: &state.contacts,
            contacts_len: &state.contacts_len,
            contacts_len_indirect: &state.contacts_indirect,
            constraints: &mut state.new_constraints,
            constraint_builders: &mut state.new_constraint_builders,
            sim_params: &state.sim_params,
            colliders_len: &state.num_shapes,
            poses: &mut state.poses,
            vels: &mut state.vels,
            solver_vels: &mut state.solver_vels,
            solver_vels_out: &state.solver_vels_out,
            solver_vels_inc: &mut state.solver_vels_inc,
            mprops: &state.mprops,
            local_mprops: &state.local_mprops,
            body_constraint_counts: &mut state.new_constraints_counts,
            body_constraint_ids: &mut state.new_body_constraint_ids,
            constraints_colors: &state.constraints_colors,
            curr_color: &mut state.curr_color,
            prefix_sum: &self.prefix_sum,
            num_colors,
            contacts_batch_capacity: &state.contacts_batch_capacity,
            colliders_batch_capacity: &state.colliders_batch_capacity,
            num_batches: state.num_batches,
            num_colliders: state.num_colliders_per_batch,
            num_solver_iterations: state.num_solver_iterations,
        };

        // Phase 3: Solve constraints
        let joint_solver_args = JointSolverArgs {
            num_batches: state.num_batches,
            sim_params: &state.sim_params,
            mprops: &state.mprops,
            local_mprops: &state.local_mprops,
            joints: &mut state.joints,
            colliders_batch_capacity: &state.colliders_batch_capacity,
        };

        {
            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass("solver", timestamps.as_deref_mut());
            self.solver
                .solve_tgs(
                    &mut pass,
                    &self.joint_solver,
                    solver_args,
                    joint_solver_args,
                )
                .unwrap();
            drop(pass);

            // Resolve all accumulated timestamps before the final submit.
            if let Some(ts) = &timestamps {
                ts.resolve(&mut encoder);
            }

            backend.submit(encoder).unwrap();
        }

        // Swap buffers for warmstarting next frame
        std::mem::swap(&mut state.old_constraints, &mut state.new_constraints);
        std::mem::swap(
            &mut state.old_constraint_builders,
            &mut state.new_constraint_builders,
        );
        std::mem::swap(
            &mut state.old_body_constraint_ids,
            &mut state.new_body_constraint_ids,
        );
        std::mem::swap(
            &mut state.old_constraints_counts,
            &mut state.new_constraints_counts,
        );

        stats
    }
}

fn validate_lbvh_topology(tree: &[LbvhNode], sorted_colliders: &[u32], num_colliders: u32) {
    let n = num_colliders as usize;
    if n < 2 {
        println!("[LBVH] Skipping validation: num_colliders={}", n);
        return;
    }

    let num_internal = n - 1;
    let first_leaf = num_internal;
    let total_nodes = 2 * n - 1;
    let mut errors = 0u32;

    println!(
        "[LBVH] Validating topology: {} colliders, {} nodes",
        n, total_nodes
    );

    // 1. Check internal node topology (nodes 0..num_internal)
    for i in 0..num_internal {
        let node = &tree[i];
        let left = node.left as usize;
        let right = node.right as usize;

        if left >= total_nodes {
            eprintln!(
                "  ERROR: internal node {} has left={} (out of bounds, max={})",
                i,
                left,
                total_nodes - 1
            );
            errors += 1;
        }
        if right >= total_nodes {
            eprintln!(
                "  ERROR: internal node {} has right={} (out of bounds, max={})",
                i,
                right,
                total_nodes - 1
            );
            errors += 1;
        }

        // Children should point back to this node as parent
        if left < total_nodes && tree[left].parent as usize != i {
            eprintln!(
                "  ERROR: internal node {} left child {} has parent={} (expected {})",
                i, left, tree[left].parent, i
            );
            errors += 1;
        }
        if right < total_nodes && tree[right].parent as usize != i {
            eprintln!(
                "  ERROR: internal node {} right child {} has parent={} (expected {})",
                i, right, tree[right].parent, i
            );
            errors += 1;
        }

        // left and right should be different
        if left == right {
            eprintln!("  ERROR: internal node {} has left == right == {}", i, left);
            errors += 1;
        }
    }

    // 2. Check leaf nodes (nodes first_leaf..total_nodes)
    let mut collider_seen = vec![false; n];
    for (leaf_offset, node) in tree[first_leaf..total_nodes].iter().enumerate() {
        let leaf_index = first_leaf + leaf_offset;
        let collider_id = node.left as usize;

        if collider_id >= n {
            eprintln!(
                "  ERROR: leaf {} has collider_id={} (out of bounds, max={})",
                leaf_index,
                collider_id,
                n - 1
            );
            errors += 1;
        } else if collider_seen[collider_id] {
            eprintln!(
                "  ERROR: leaf {} has duplicate collider_id={}",
                leaf_index, collider_id
            );
            errors += 1;
        } else {
            collider_seen[collider_id] = true;
        }
    }

    let missing: Vec<usize> = collider_seen
        .iter()
        .enumerate()
        .filter(|(_, seen)| !**seen)
        .map(|(id, _)| id)
        .collect();
    if !missing.is_empty() {
        eprintln!(
            "  ERROR: {} colliders missing from leaves: {:?}",
            missing.len(),
            &missing[..missing.len().min(20)]
        );
        errors += 1;
    }

    // 3. Check sorted_colliders matches leaf assignment
    for i in 0..n {
        let expected_collider = sorted_colliders[i];
        let leaf_collider = tree[first_leaf + i].left;
        if expected_collider != leaf_collider {
            eprintln!(
                "  ERROR: sorted_colliders[{}]={} but tree leaf {}.left={}",
                i,
                expected_collider,
                first_leaf + i,
                leaf_collider
            );
            errors += 1;
            if errors > 50 {
                break;
            }
        }
    }

    // 4. Check AABBs: parent AABB should contain both children
    let mut aabb_errors = 0u32;
    for i in 0..num_internal {
        let node = &tree[i];
        let left = node.left as usize;
        let right = node.right as usize;
        if left >= total_nodes || right >= total_nodes {
            continue;
        }

        let parent_aabb = &node.aabb;
        let left_aabb = &tree[left].aabb;
        let right_aabb = &tree[right].aabb;

        let eps = 1.0e-5;
        let parent_valid = parent_aabb.mins.x <= parent_aabb.maxs.x;
        let left_valid = left_aabb.mins.x <= left_aabb.maxs.x;
        let right_valid = right_aabb.mins.x <= right_aabb.maxs.x;

        if !parent_valid {
            if aabb_errors < 10 {
                eprintln!(
                    "  ERROR: internal node {} has invalid AABB (mins > maxs): mins={:?} maxs={:?}",
                    i, parent_aabb.mins, parent_aabb.maxs
                );
            }
            aabb_errors += 1;
            continue;
        }

        if left_valid
            && (parent_aabb.mins.x > left_aabb.mins.x + eps
                || parent_aabb.mins.y > left_aabb.mins.y + eps
                || parent_aabb.maxs.x < left_aabb.maxs.x - eps
                || parent_aabb.maxs.y < left_aabb.maxs.y - eps)
        {
            if aabb_errors < 10 {
                eprintln!(
                    "  ERROR: node {} AABB does not contain left child {} AABB",
                    i, left
                );
                eprintln!(
                    "    parent: mins={:?} maxs={:?}",
                    parent_aabb.mins, parent_aabb.maxs
                );
                eprintln!(
                    "    left:   mins={:?} maxs={:?}",
                    left_aabb.mins, left_aabb.maxs
                );
            }
            aabb_errors += 1;
        }

        if right_valid
            && (parent_aabb.mins.x > right_aabb.mins.x + eps
                || parent_aabb.mins.y > right_aabb.mins.y + eps
                || parent_aabb.maxs.x < right_aabb.maxs.x - eps
                || parent_aabb.maxs.y < right_aabb.maxs.y - eps)
        {
            if aabb_errors < 10 {
                eprintln!(
                    "  ERROR: node {} AABB does not contain right child {} AABB",
                    i, right
                );
                eprintln!(
                    "    parent: mins={:?} maxs={:?}",
                    parent_aabb.mins, parent_aabb.maxs
                );
                eprintln!(
                    "    right:  mins={:?} maxs={:?}",
                    right_aabb.mins, right_aabb.maxs
                );
            }
            aabb_errors += 1;
        }
    }

    if aabb_errors > 0 {
        eprintln!("  AABB errors total: {} (showing first 10)", aabb_errors);
        errors += aabb_errors;
    }

    // 5. Check reachability from root via BFS
    let mut visited = vec![false; total_nodes];
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(0usize);
    visited[0] = true;
    let mut visited_count = 0usize;

    while let Some(id) = queue.pop_front() {
        visited_count += 1;
        if id < num_internal {
            let left = tree[id].left as usize;
            let right = tree[id].right as usize;
            if left < total_nodes && !visited[left] {
                visited[left] = true;
                queue.push_back(left);
            }
            if right < total_nodes && !visited[right] {
                visited[right] = true;
                queue.push_back(right);
            }
        }
    }

    if visited_count != total_nodes {
        let unreachable: Vec<usize> = visited
            .iter()
            .enumerate()
            .filter(|(_, v)| !**v)
            .map(|(id, _)| id)
            .collect();
        eprintln!(
            "  ERROR: {} nodes unreachable from root. First few: {:?}",
            unreachable.len(),
            &unreachable[..unreachable.len().min(20)]
        );
        errors += 1;
    }

    if errors == 0 {
        println!(
            "[LBVH] Topology OK: all {} nodes valid, all {} colliders present, all AABBs consistent",
            total_nodes, n
        );
    } else {
        eprintln!("[LBVH] VALIDATION FAILED: {} errors found", errors);
    }
}
