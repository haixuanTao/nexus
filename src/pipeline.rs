//! Physics simulation pipeline orchestrating broad-phase, narrow-phase, and constraint solving.
//!
//! This module provides the high-level physics pipeline that coordinates all stages of a physics
//! simulation step on the GPU. The pipeline manages collision detection, contact generation,
//! constraint solving, and integration.

use crate::broad_phase::{GpuNarrowPhase, Lbvh, LbvhState};
use crate::dynamics::{
    ColoringArgs, GpuColoring, GpuImpulseJointSet, GpuJointSolver, GpuMpropsUpdate, GpuSolver,
    GpuWarmstart, JointSolverArgs, SolverArgs,
    prefix_sum::{GpuPrefixSum, PrefixSumWorkspace},
    warmstart::WarmstartArgs,
};
use crate::math::{Pose, Vector};
use crate::queries::GpuIndexedContact;
use crate::shaders::VectorWithPadding;
use crate::shaders::broad_phase::{LbvhNode, NarrowPhasePfmPair};
use crate::shaders::dynamics::{
    LocalMassProperties as GpuLocalMassProperties, SimParams as GpuSimParams, TwoBodyConstraint,
    TwoBodyConstraintBuilder, Velocity as GpuVelocity,
    WorldMassProperties as GpuWorldMassProperties,
};
use crate::shaders::shapes::Shape;
use khal::Shader;

use khal::BufferUsages;
use khal::backend::{Backend, Encoder, GpuBackend};
use std::time::Duration;
use vortx::tensor::Tensor;

#[cfg(feature = "from_rapier")]
use {
    crate::math::Point,
    crate::rapier::dynamics::{ImpulseJointSet, RigidBodySet},
    crate::rapier::geometry::ColliderSet,
    crate::shapes::ShapeBuffers,
    crate::shapes::shape_from_parry,
    std::collections::HashMap,
};

/// Performance statistics collected during a physics simulation step.
///
/// This structure tracks timing and iteration counts for various stages of the physics pipeline,
/// useful for profiling and optimization.
#[derive(Default, Copy, Clone, Debug)]
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
    /// GPU timestamp for updating the mass properties.
    pub timestamp_update_mass_props: f64,
    /// GPU timestamp for the broad-phase collision detection.
    pub timestamp_broad_phase: f64,
    /// GPU timestamp for the narrow-phase contact generation.
    pub timestamp_narrow_phase: f64,
    /// GPU timestamp for constraint solver preparation.
    pub timestamp_solver_prep: f64,
    /// GPU timestamp for the constraint solver.
    pub timestamp_solver_solve: f64,
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
    sim_params: Tensor<GpuSimParams>,
    poses: Tensor<Pose>,
    local_mprops: Tensor<GpuLocalMassProperties>,
    mprops: Tensor<GpuWorldMassProperties>,
    vels: Tensor<GpuVelocity>,
    solver_vels: Tensor<GpuVelocity>,
    solver_vels_out: Tensor<GpuVelocity>,
    solver_vels_inc: Tensor<GpuVelocity>,
    vertex_buffers: Tensor<VectorWithPadding>,
    index_buffers: Tensor<u32>,
    shapes: Tensor<Shape>,
    num_shapes: Tensor<u32>,
    num_shapes_indirect: Tensor<[u32; 3]>,
    collision_pairs: Tensor<[u32; 2]>,
    collision_pairs_len: Tensor<u32>,
    #[allow(dead_code)]
    collision_pairs_len_staging: Tensor<u32>,
    collision_pairs_indirect: Tensor<[u32; 3]>,
    max_collision_pairs: Tensor<u32>,
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

    prefix_sum_workspace: PrefixSumWorkspace,

    #[allow(dead_code)]
    debug_aabb_mins: Tensor<Vector>,
    #[allow(dead_code)]
    debug_aabb_maxs: Tensor<Vector>,
}

#[cfg(feature = "from_rapier")]
impl GpuPhysicsState {
    /// Creates a new GPU physics state from CPU-side Rapier data structures.
    ///
    /// This method extracts rigid body and collider data from Rapier's CPU representations
    /// and uploads them to GPU buffers. Each collider is treated as a separate rigid body
    /// in the GPU simulation.
    ///
    /// # Parameters
    ///
    /// - `backend`: The GPU backend used to allocate GPU buffers.
    /// - `bodies`: The set of rigid bodies from Rapier.
    /// - `colliders`: The set of colliders from Rapier.
    ///
    /// # Panics
    ///
    /// Panics if any rigid body has more than one collider attached, as this is not currently supported.
    pub fn from_rapier(
        backend: &GpuBackend,
        bodies: &RigidBodySet,
        colliders: &ColliderSet,
        impulse_joints: &ImpulseJointSet,
    ) -> Self {
        let mut rb_poses = Vec::new();
        let mut rb_local_mprops = Vec::new();
        let mut rb_mprops = Vec::new();
        let mut shapes = Vec::new();
        let mut shape_buffers = ShapeBuffers::default();
        let mut body_ids = HashMap::new();

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
                let id = rb_poses.len();
                body_ids.insert(h, id as u32);
            }

            rb_local_mprops.push(local_mprops);
            rb_mprops.push(mprops);
            shapes
                .push(shape_from_parry(co.shape(), &mut shape_buffers).expect("Unsupported shape"));

            rb_poses.push(*co.position());
        }

        // NOTE: GPU doesn't like empty storage buffer bindings.
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

        let joints = GpuImpulseJointSet::from_rapier(backend, impulse_joints, &body_ids);

        let num_bodies = rb_poses.len();
        let rb_vels = vec![GpuVelocity::default(); num_bodies];
        let storage: BufferUsages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let shapes = Tensor::vector(backend, &shapes, storage).unwrap();
        let num_shapes = Tensor::scalar_encased(
            backend,
            num_bodies as u32,
            BufferUsages::STORAGE | BufferUsages::UNIFORM,
        )
        .unwrap();
        let num_shapes_indirect = Tensor::scalar(
            backend,
            [num_bodies.div_ceil(64) as u32, 1, 1],
            BufferUsages::STORAGE | BufferUsages::INDIRECT,
        )
        .unwrap();

        const DEFAULT_CONTACT_COUNTS: u32 = 1024;
        let collision_pairs =
            Tensor::vector_uninit(backend, DEFAULT_CONTACT_COUNTS, storage).unwrap();
        let collision_pairs_len =
            Tensor::scalar_uninit(backend, BufferUsages::STORAGE | BufferUsages::COPY_SRC).unwrap();
        let collision_pairs_len_staging =
            Tensor::scalar_uninit(backend, BufferUsages::MAP_READ | BufferUsages::COPY_DST)
                .unwrap();
        let collision_pairs_indirect =
            Tensor::scalar_uninit(backend, BufferUsages::STORAGE | BufferUsages::INDIRECT).unwrap();
        let max_collision_pairs = Tensor::scalar_encased(
            backend,
            DEFAULT_CONTACT_COUNTS,
            BufferUsages::STORAGE | BufferUsages::UNIFORM,
        )
        .unwrap();

        let contacts = Tensor::vector_uninit(backend, DEFAULT_CONTACT_COUNTS, storage).unwrap();
        let contacts_len =
            Tensor::scalar_uninit_encased(backend, BufferUsages::STORAGE | BufferUsages::UNIFORM)
                .unwrap();
        let contacts_indirect =
            Tensor::scalar_uninit(backend, BufferUsages::STORAGE | BufferUsages::INDIRECT).unwrap();
        let pfm_pairs_indirect =
            Tensor::scalar_uninit(backend, BufferUsages::STORAGE | BufferUsages::INDIRECT).unwrap();
        let pfm_pairs = Tensor::vector_uninit(backend, DEFAULT_CONTACT_COUNTS, storage).unwrap();
        let pfm_pairs_len =
            Tensor::scalar_uninit_encased(backend, BufferUsages::STORAGE | BufferUsages::UNIFORM)
                .unwrap();
        let old_constraints =
            Tensor::vector_uninit(backend, DEFAULT_CONTACT_COUNTS, storage).unwrap();
        let old_constraint_builders =
            Tensor::vector_uninit(backend, DEFAULT_CONTACT_COUNTS, storage).unwrap();
        let new_constraints =
            Tensor::vector_uninit(backend, DEFAULT_CONTACT_COUNTS, storage).unwrap();
        let new_constraint_builders =
            Tensor::vector_uninit(backend, DEFAULT_CONTACT_COUNTS, storage).unwrap();
        let constraints_colors =
            Tensor::vector_uninit(backend, DEFAULT_CONTACT_COUNTS, storage).unwrap();
        let colored = Tensor::vector_uninit(backend, DEFAULT_CONTACT_COUNTS, storage).unwrap();
        let constraints_rands =
            Tensor::vector_uninit(backend, DEFAULT_CONTACT_COUNTS, storage).unwrap();
        let old_constraints_counts =
            Tensor::vector_uninit(backend, num_bodies as u32, storage).unwrap();
        let new_constraints_counts =
            Tensor::vector_uninit(backend, num_bodies as u32, storage).unwrap();
        let old_body_constraint_ids =
            Tensor::vector_uninit(backend, DEFAULT_CONTACT_COUNTS * 2, storage).unwrap();
        let new_body_constraint_ids =
            Tensor::vector_uninit(backend, DEFAULT_CONTACT_COUNTS * 2, storage).unwrap();

        let mut sim_params = GpuSimParams::tgs_soft();
        sim_params.dt /= sim_params.num_solver_iterations as f32;

        let lbvh_usages = if crate::VALIDATE_LBVH_TOPOLOGY {
            BufferUsages::STORAGE | BufferUsages::COPY_SRC
        } else {
            BufferUsages::STORAGE
        };

        Self {
            sim_params: Tensor::scalar_encased(
                backend,
                sim_params,
                BufferUsages::STORAGE | BufferUsages::UNIFORM,
            )
            .unwrap(),
            vels: Tensor::vector(backend, &rb_vels, storage).unwrap(),
            solver_vels: Tensor::vector(backend, &rb_vels, storage).unwrap(),
            solver_vels_out: Tensor::vector(backend, &rb_vels, storage).unwrap(),
            solver_vels_inc: Tensor::vector(backend, &rb_vels, storage).unwrap(),
            joints,
            local_mprops: Tensor::vector(backend, &rb_local_mprops, storage).unwrap(),
            mprops: Tensor::vector(backend, &rb_mprops, storage).unwrap(),
            poses: Tensor::vector(
                backend,
                &rb_poses,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            )
            .unwrap(),
            vertex_buffers,
            index_buffers,
            shapes,
            num_shapes,
            num_shapes_indirect,
            collision_pairs,
            collision_pairs_len,
            collision_pairs_len_staging,
            collision_pairs_indirect,
            max_collision_pairs,
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
            curr_color: Tensor::scalar_encased(
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
            debug_aabb_mins: Tensor::vector_uninit_encased(backend, num_bodies as u32, storage)
                .unwrap(),
            debug_aabb_maxs: Tensor::vector_uninit_encased(backend, num_bodies as u32, storage)
                .unwrap(),
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
}

/// The main GPU physics pipeline coordinating all simulation stages.
///
/// This structure contains all the compute shaders needed to run a complete physics simulation
/// on the GPU. It orchestrates the following stages in each simulation step:
///
/// 1. **Gravity application**: Updates velocities with gravitational forces.
/// 2. **Broad-phase**: Uses LBVH to find potentially colliding pairs.
/// 3. **Narrow-phase**: Generates detailed contact information for collision pairs.
/// 4. **Constraint preparation**: Converts contacts into solver constraints.
/// 5. **Graph coloring**: Colors constraints to enable parallel solving.
/// 6. **Constraint solving**: Iteratively solves constraints using TGS or PGS.
/// 7. **Integration**: Updates poses based on solved velocities.
pub struct GpuPhysicsPipeline {
    mprops_update: GpuMpropsUpdate,
    narrow_phase: GpuNarrowPhase,
    solver: GpuSolver,
    joint_solver: GpuJointSolver,
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
            prefix_sum: GpuPrefixSum::from_backend(backend).unwrap(),
            lbvh: Lbvh::from_backend(backend),
            coloring: GpuColoring::from_backend(backend).unwrap(),
            warmstart: GpuWarmstart::from_backend(backend).unwrap(),
        }
    }

    /// Executes one physics simulation timestep on the GPU.
    ///
    /// This method runs the complete physics pipeline:
    /// 1. Update world-space mass-properties.
    /// 2. Builds LBVH and finds collision pairs (broad-phase).
    /// 3. Generates contact manifolds (narrow-phase).
    /// 4. Prepares solver constraints from contacts.
    /// 5. Colors constraints for parallel solving.
    /// 6. Solves constraints iteratively using TGS.
    /// 7. Integrates velocities to update poses.
    ///
    /// # Buffer Resizing
    ///
    /// If the number of collision pairs exceeds buffer capacity, this method automatically
    /// allocates larger buffers (next power of two) and re-runs the broad-phase.
    pub async fn step(&self, backend: &GpuBackend, state: &mut GpuPhysicsState) -> RunStats {
        let mut stats = RunStats::default();
        let t_phase1 = web_time::Instant::now();

        // Phase 1: Update mass properties, build LBVH, and find collision pairs
        {
            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass("update-mprops", None);

            // Update mass properties
            self.mprops_update
                .dispatch(
                    &mut pass,
                    &mut state.mprops,
                    &state.local_mprops,
                    &state.poses,
                    state.poses.len() as u32,
                )
                .unwrap();

            // Build LBVH and find collision pairs
            self.lbvh
                .update_tree(
                    backend,
                    &mut pass,
                    &mut state.lbvh,
                    state.poses.len() as u32,
                    &state.poses,
                    &state.vertex_buffers,
                    &state.shapes,
                    &state.num_shapes,
                )
                .unwrap();

            // Debug: validate LBVH topology after tree construction
            if crate::VALIDATE_LBVH_TOPOLOGY {
                drop(pass);
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
                pass = encoder.begin_pass("lbvh-find-pairs", None);
            }

            self.lbvh
                .find_pairs(
                    &mut pass,
                    &mut state.lbvh,
                    state.poses.len() as u32,
                    &state.num_shapes,
                    &state.max_collision_pairs,
                    &mut state.collision_pairs,
                    &mut state.collision_pairs_len,
                    &mut state.collision_pairs_indirect,
                )
                .unwrap();

            drop(pass);
            backend.submit(encoder).unwrap();
        }

        // Read back collision pair count (requires CPU-GPU sync)
        let num_collision_pairs = backend
            .slow_read_vec(state.collision_pairs_len.buffer())
            .await
            .unwrap()[0];
        stats.start_to_pairs_count_time = t_phase1.elapsed();

        // Resize buffers if needed
        if num_collision_pairs >= state.collision_pairs.len() as u32 {
            let storage: BufferUsages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
            let desired_len = num_collision_pairs.next_power_of_two();

            state.collision_pairs = Tensor::vector_uninit(backend, desired_len, storage).unwrap();
            state.max_collision_pairs = Tensor::scalar_encased(
                backend,
                desired_len,
                BufferUsages::STORAGE | BufferUsages::UNIFORM,
            )
            .unwrap();
            state.contacts = Tensor::vector_uninit(backend, desired_len, storage).unwrap();
            state.pfm_pairs = Tensor::vector_uninit(backend, desired_len, storage).unwrap();
            state.old_constraints = Tensor::vector_uninit(backend, desired_len, storage).unwrap();
            state.old_constraint_builders =
                Tensor::vector_uninit(backend, desired_len, storage).unwrap();
            state.old_body_constraint_ids =
                Tensor::vector_uninit(backend, desired_len * 2, storage).unwrap();
            state.new_constraints = Tensor::vector_uninit(backend, desired_len, storage).unwrap();
            state.new_constraint_builders =
                Tensor::vector_uninit(backend, desired_len, storage).unwrap();
            state.new_body_constraint_ids =
                Tensor::vector_uninit(backend, desired_len * 2, storage).unwrap();
            state.constraints_colors =
                Tensor::vector_uninit(backend, desired_len, storage).unwrap();
            state.colored = Tensor::vector_uninit(backend, desired_len, storage).unwrap();
            state.constraints_rands = Tensor::vector_uninit(backend, desired_len, storage).unwrap();

            // Re-run find_pairs with resized buffers
            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass("lbvh-find-pairs", None);
            self.lbvh
                .find_pairs(
                    &mut pass,
                    &mut state.lbvh,
                    state.poses.len() as u32,
                    &state.num_shapes,
                    &state.max_collision_pairs,
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
            let mut pass = encoder.begin_pass("narrow-phase", None);

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
                )
                .unwrap();

            // Solver preparation - create args here to avoid borrow conflicts
            let prepare_args = SolverArgs {
                num_colliders: state.poses.len() as u32,
                contacts: &state.contacts,
                contacts_len: &state.contacts_len,
                contacts_len_indirect: &state.contacts_indirect,
                constraints: &mut state.new_constraints,
                constraint_builders: &mut state.new_constraint_builders,
                sim_params: &state.sim_params,
                colliders_len: &state.num_shapes,
                colliders_len_indirect: &state.num_shapes_indirect,
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
        };

        let num_colors = if let Some(colors) = self
            .coloring
            .dispatch_topo_gc(backend, coloring_args, &mut stats)
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
            };
            self.coloring
                .dispatch_luby(backend, coloring_args, &mut stats)
                .await
        };

        stats.num_colors = num_colors;

        // Create solver_args for solve phase (after coloring is complete)
        let solver_args = SolverArgs {
            num_colliders: state.poses.len() as u32,
            contacts: &state.contacts,
            contacts_len: &state.contacts_len,
            contacts_len_indirect: &state.contacts_indirect,
            constraints: &mut state.new_constraints,
            constraint_builders: &mut state.new_constraint_builders,
            sim_params: &state.sim_params,
            colliders_len: &state.num_shapes,
            colliders_len_indirect: &state.num_shapes_indirect,
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
        };

        // Phase 3: Solve constraints
        let joint_solver_args = JointSolverArgs {
            sim_params: &state.sim_params,
            mprops: &state.mprops,
            local_mprops: &state.local_mprops,
            joints: &mut state.joints,
        };

        {
            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass("solve-tgs", None);
            self.solver
                .solve_tgs(
                    &mut pass,
                    &self.joint_solver,
                    solver_args,
                    joint_solver_args,
                )
                .unwrap();
            drop(pass);
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
