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
use crate::dynamics::{GpuMultibodySet, GpuMultibodySnapshot, GpuMultibodySolver};
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
use crate::shaders::utils::BatchIndices;
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
    /// Total simulation time not including GPU-to-CPU readbacks.
    pub total_simulation_time_without_readback: Duration,
    /// Total simulation time including GPU-to-CPU readbacks.
    pub total_simulation_time_with_readback: Duration,
    /// Per-pass GPU timestamp durations (label, milliseconds).
    pub gpu_pass_times: Vec<(String, f64)>,
    /// Total GPU time across all measured passes, in milliseconds.
    pub gpu_total_time: f64,
}

impl RunStats {
    /// Returns the total simulation time in milliseconds.
    pub fn total_simulation_time_with_readback_ms(&self) -> f32 {
        self.total_simulation_time_with_readback.as_secs_f32() * 1000.0
    }

    /// Returns the total simulation time in milliseconds.
    pub fn total_simulation_time_without_readback_ms(&self) -> f32 {
        self.total_simulation_time_without_readback.as_secs_f32() * 1000.0
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
    /// Per-body world-origin pose (matches rapier's `RigidBody::position`). This
    /// is the canonical pose stored between steps and the input to per-step
    /// mass-properties update and multibody FK. The substep loop does NOT
    /// touch this — see [`Self::solver_body_poses`].
    body_poses: Tensor<Pose>,
    /// Per-body COM-centered pose (rapier's `SolverPose`). Equals
    /// `body_poses[i].prepend_translation(local_mprops[i].com)`. Seeded from
    /// `body_poses` at step start, mutated by the solver substep loop, and
    /// converted back to `body_poses` by `finalize` at step end.
    solver_body_poses: Tensor<Pose>,
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
    /// Per-collider local pose, expressed in the parent rigid-body's frame. The
    /// world pose of the collider's shape is `poses[i] * collider_local_poses[i]`,
    /// matching rapier's `Collider::position()` semantics. Set to identity for
    /// colliders without a parent.
    collider_local_poses: Tensor<Pose>,
    /// World-pose of colliders, used by collision detection.
    collider_world_poses: Tensor<Pose>,
    /// Per-collider [`crate::rapier::geometry::InteractionGroups`]. Used by the
    /// broad-phase to skip pairs whose groups don't authorize an interaction.
    /// Padded slots use empty memberships AND empty filter so they never match.
    collision_groups: Tensor<crate::rapier::geometry::InteractionGroups>,
    collision_pairs: Tensor<[u32; 2]>,
    collision_pairs_len: Tensor<u32>,
    collision_pairs_len_staging: Tensor<u32>,
    collision_pairs_indirect: Tensor<[u32; 3]>,
    collision_pairs_batch_capacity: Tensor<u32>,
    contacts_batch_capacity: Tensor<u32>,
    colliders_batch_capacity: Tensor<u32>,
    /// CPU-side mirrors of the dynamic batch capacities above. Kept in sync
    /// with the `*_batch_capacity` tensors so [`Self::batch_indices`] can be
    /// rebuilt whenever any of them grows.
    contacts_per_batch_cpu: u32,
    collision_pairs_per_batch_cpu: u32,
    /// Single uniform aggregating every per-batch capacity and packed-buffer
    /// section offset consumed by the compute kernels (multibody and RBD
    /// sides). Rebuilt by [`Self::rebuild_batch_indices`] whenever any of its
    /// constituent caps changes (e.g. when the contacts buffer grows).
    batch_indices: Tensor<BatchIndices>,
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
    /// Constant per-color uniform buffers holding `[1, 2, …, N]`. Used by the
    /// solver's per-color sweeps INSTEAD of bumping `curr_color` via the
    /// `reset_color`/`inc_color` kernels (which were ~28% of per-step dispatches
    /// doing zero math). Each loop iteration binds `color_uniforms[c]`.
    color_uniforms: Vec<Tensor<u32>>,
    uncolored: Tensor<u32>,
    uncolored_staging: Tensor<u32>,
    lbvh: LbvhState,
    joints: GpuImpulseJointSet,
    #[cfg(feature = "dim3")]
    multibodies: GpuMultibodySet,
    /// Per-body "graph group" id, used by graph coloring to treat all bodies of
    /// the same multibody as a single node. For free bodies, `body_group[i] = i`.
    /// For bodies belonging to a multibody, all share the same group id (chosen
    /// as the body id of the multibody's root link). Two contacts touching
    /// different bodies of the same multibody therefore share the same group
    /// and cannot be assigned the same color.
    ///
    /// The buffer is laid out flat across batches with stride
    /// `colliders_batch_capacity`, matching the shared body-keyed buffers
    /// (`new_constraints_counts` / `new_body_constraint_ids` / etc.).
    #[allow(dead_code)]
    body_group: Tensor<u32>,
    prefix_sum_workspace: PrefixSumWorkspace,
    /// Maximum number of constraint colors the solver will iterate.
    max_colors: u32,
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
        // Pick the dispatch-grid strategy from the backend: fixed-grid on CUDA
        // (the INDIRECT host round-trip is a ~1100 ms/rollout GPU drain there),
        // indirect on WebGPU (native/cheap). `BIPED_FIXED_GRID` overrides.
        crate::set_fixed_dispatch_grid_default(backend.is_cuda());

        let num_batches = environments.len() as u32;
        let max_colliders = environments
            .iter()
            .map(|(_, c, _, _, _)| c.len())
            .max()
            .unwrap_or(0);

        let mut all_poses = Vec::new();
        let mut all_vels = Vec::new();
        let mut all_local_mprops = Vec::new();
        let mut all_mprops = Vec::new();
        let mut all_shapes = Vec::new();
        let mut all_num_shapes = Vec::new();
        let mut all_collision_groups: Vec<crate::rapier::geometry::InteractionGroups> = Vec::new();
        let mut all_collider_local_poses: Vec<Pose> = Vec::new();
        let mut shape_buffers = ShapeBuffers::default();
        // TriMesh dedupe: batched envs often share one terrain mesh (the same
        // parry `SharedShape` Arc cloned per env). Emitting its flat BVH +
        // pseudo-normals once and reusing the `Shape` descriptor (whose ranges
        // point into the shared `shape_buffers`) keeps memory O(unique meshes)
        // instead of O(envs). Keyed by the parry shape data pointer, TriMesh
        // ONLY — scenes without trimeshes produce byte-identical buffers.
        let mut trimesh_cache: HashMap<usize, crate::shaders::shapes::Shape> = HashMap::new();
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

                // Mass properties — taken from the rapier rigid-body so the GPU
                // solver sees the same effective mass / COM / inertia as rapier
                // would. This includes the body-local COM offset, which the
                // integrator and joint solver assume is in the body's frame.
                // Parentless colliders are modeled as colliders attached to an
                // implicit fixed body anchored at the world origin: the body pose
                // is IDENTITY and the collider's world transform is folded into
                // its body-local offset. This keeps every code path that consumes
                // `body_poses` agnostic of the parent/no-parent distinction.
                let (body_pose, collider_local_pose) = match parent {
                    Some(b) => (
                        *b.position(),
                        co.position_wrt_parent().copied().unwrap_or(Pose::IDENTITY),
                    ),
                    None => (Pose::IDENTITY, *co.position()),
                };
                let is_dynamic = parent.map(|b| b.is_dynamic()).unwrap_or(false);
                let (local_mprops, mprops) = if let (Some(parent), true) = (parent, is_dynamic) {
                    let m = parent.mass_properties();
                    let local = local_mprops_from_rapier(&m.local_mprops);
                    let world = world_mprops_from_local(&body_pose, &local);
                    (local, world)
                } else {
                    let mut local = GpuLocalMassProperties::default();
                    let mut world = GpuWorldMassProperties {
                        com: body_pose.translation,
                        ..Default::default()
                    };
                    local.inv_mass = Vector::ZERO;
                    world.inv_mass = Vector::ZERO;
                    #[cfg(feature = "dim3")]
                    {
                        local.inv_principal_inertia = glamx::Vec3::ZERO;
                        world.inv_inertia = glamx::Mat4::ZERO;
                    }
                    #[cfg(feature = "dim2")]
                    {
                        local.inv_inertia = 0.0;
                        world.inv_inertia = 0.0;
                    }
                    (local, world)
                };

                if let Some(h) = co.parent() {
                    body_ids.insert(h, env_collider_idx);
                }

                env_collider_idx += 1;
                all_local_mprops.push(local_mprops);
                all_mprops.push(mprops);
                let gpu_shape = match co.shape().as_typed_shape() {
                    crate::parry::shape::TypedShape::TriMesh(tm) => {
                        let key = tm as *const _ as *const u8 as usize;
                        match trimesh_cache.get(&key) {
                            Some(&s) => s,
                            None => {
                                let s = shape_from_parry(co.shape(), &mut shape_buffers)
                                    .expect("Unsupported shape");
                                trimesh_cache.insert(key, s);
                                s
                            }
                        }
                    }
                    _ => shape_from_parry(co.shape(), &mut shape_buffers)
                        .expect("Unsupported shape"),
                };
                all_shapes.push(gpu_shape);
                // Mirror rapier: bodies hold a world-origin pose, colliders hold
                // their local offset and a world pose `body * local`. Joint /
                // solver / integrator / mprops_update / multibody-FK consume
                // `body_poses`. Broad-phase, narrow-phase and contact-to-constraint
                // consume `collider_world_poses`, which the
                // `gpu_sync_collider_poses` kernel keeps in sync each step.
                all_poses.push(body_pose);
                all_vels.push(match parent {
                    Some(b) => GpuVelocity::new(b.linvel(), b.angvel()),
                    None => GpuVelocity::default(),
                });
                all_collider_local_poses.push(collider_local_pose);
                all_collision_groups.push(co.collision_groups());
            }

            // Pad to max_colliders with dummy fixed bodies.
            let dummy_shape = all_shapes.last().copied().unwrap_or_default();
            for _ in env_collider_count..max_colliders {
                all_poses.push(dummy_pose);
                all_vels.push(GpuVelocity::default());
                all_collider_local_poses.push(Pose::IDENTITY);
                all_local_mprops.push(dummy_local_mprops);
                all_mprops.push(dummy_mprops);
                all_shapes.push(dummy_shape);
                // Padding bodies: zero membership AND zero filter so the broad
                // phase never matches them with anything.
                all_collision_groups.push(crate::rapier::geometry::InteractionGroups::new(
                    crate::rapier::geometry::Group::NONE,
                    crate::rapier::geometry::Group::NONE,
                    crate::rapier::geometry::InteractionTestMode::And,
                ));
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

        // Per-environment "graph group" table: each body local id maps to either
        // its own id (free body) or a shared multibody-group id. Bodies in the
        // same multibody collapse to a single node for impulse-joint coloring,
        // matching rapier's rule.
        //
        // We also build `is_mb_body[env][body_local]` — true iff the body is
        // part of some multibody. The regular `GpuImpulseJointSet` skips any
        // joint touching such a body; those joints are routed to
        // `GpuMultibodySet::set_impulse_joints` instead so they go through
        // the generic `M⁻¹·Jᵀ` solver path.
        #[cfg(feature = "dim3")]
        let (multibody_groups, is_mb_body): (Vec<Vec<u32>>, Vec<Vec<bool>>) = {
            let mut all_groups: Vec<Vec<u32>> = Vec::with_capacity(num_batches as usize);
            let mut all_is_mb: Vec<Vec<bool>> = Vec::with_capacity(num_batches as usize);
            for (env_idx, (mb_set, body_ids, _)) in multibody_envs.iter().enumerate() {
                let _ = env_idx;
                let max_id = body_ids.values().copied().max().unwrap_or_default();
                let mut group: Vec<u32> = (0..=max_id).collect();
                let mut is_mb: Vec<bool> = vec![false; (max_id + 1) as usize];
                let mut next_group = max_id + 1;
                for mb in mb_set.multibodies() {
                    let g = next_group;
                    next_group += 1;
                    for link in mb.links() {
                        if let Some(&id) = body_ids.get(&link.rigid_body_handle()) {
                            group[id as usize] = g;
                            is_mb[id as usize] = true;
                        }
                    }
                }
                all_groups.push(group);
                all_is_mb.push(is_mb);
            }
            (all_groups, all_is_mb)
        };
        #[cfg(not(feature = "dim3"))]
        let (multibody_groups, is_mb_body): (Vec<Vec<u32>>, Vec<Vec<bool>>) =
            (Vec::new(), Vec::new());

        let joints = GpuImpulseJointSet::from_rapier_filtered(
            backend,
            &joint_env_refs,
            &multibody_groups,
            &is_mb_body,
        );

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
            // Per-env contact friction μ, read from the rapier collider
            // material so per-env friction DR actually reaches the GPU contact
            // solver (the builder previously hardcoded 0.5). Representative
            // value: the friction of the first collider on a dynamic body (the
            // robot's own colliders — set to the env's DR friction); falls back
            // to 0.5 when none found. Aligned with `mb_refs` because
            // `multibody_envs` is pushed one-per-env, parallel to `environments`.
            let mb_frictions: Vec<f32> = environments
                .iter()
                .map(|(bodies, colliders, _, _, _)| {
                    // The foot↔ground contact μ. Both the foot and the ground
                    // collider are set to the env's DR friction, while the other
                    // link colliders keep rapier's 0.5 default — so read the
                    // GROUND (the fixed-body collider): it's the unambiguous,
                    // single carrier of the contact friction for this env.
                    colliders
                        .iter()
                        .find_map(|(_, c)| {
                            let b = bodies.get(c.parent()?)?;
                            b.is_fixed().then(|| c.friction())
                        })
                        .unwrap_or(0.5)
                })
                .collect();
            if std::env::var_os("NEXUS_DEBUG_FRICTION").is_some() {
                let n = mb_frictions.len().min(8);
                eprintln!("[nexus] per-env contact μ (first {n}) = {:?}", &mb_frictions[..n]);
            }
            let mut mb = GpuMultibodySet::from_rapier(
                backend,
                &mb_refs,
                [0.0, -9.81, 0.0],
                max_colliders as u32,
                &mb_frictions,
            );
            mb.set_visible_dt(backend, multibody_dt);

            // Route MB-touching impulse joints (those skipped by the
            // regular `GpuImpulseJointSet`) to the multibody generic
            // constraint path.
            let imp_refs: Vec<(
                &crate::rapier::dynamics::ImpulseJointSet,
                &MultibodyJointSet,
                &HashMap<crate::rapier::dynamics::RigidBodyHandle, u32>,
                &RigidBodySet,
            )> = joint_envs
                .iter()
                .zip(multibody_envs.iter())
                .map(|((imp, body_ids), (mb_set, _, bodies))| (*imp, *mb_set, body_ids, *bodies))
                .collect();
            mb.set_impulse_joints(backend, &imp_refs);
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
                            // Multibody links must have a zero local COM in the
                            // shared body buffer: the multibody substep writes
                            // body-origin poses to `solver_body_poses`, and we
                            // need `init_solver_bodies` / `finalize` (which
                            // shift by ±local_com) to be no-ops here so the
                            // multibody apply_substep can treat solver poses
                            // as body-origin poses.
                            all_local_mprops[global].com = Vector::ZERO;
                            all_mprops[global].inv_mass = Vector::ZERO;
                            all_mprops[global].inv_inertia = glamx::Mat4::ZERO;
                        }
                    }
                }
            }
        }

        // Build the per-body "graph group" lookup. Free bodies map to themselves
        // (one body = one graph node). Bodies belonging to a multibody all map
        // to a single shared group id (= the body id of the multibody's root
        // link). Coloring kernels read `body_group[body_id]` instead of `body_id`
        // when computing constraint adjacency, so contacts touching different
        // bodies of the same multibody correctly conflict and never share a
        // color.
        // `body_group` stores PER-BATCH local indices (so a kernel can use the
        // same `Slice(buf, colliders_start)` pattern as for `body_constraint_*`
        // and just index by `group_local`).
        let mut all_body_group: Vec<u32> = Vec::with_capacity(max_colliders * num_batches as usize);
        for _batch_idx in 0..num_batches as usize {
            for b in 0..max_colliders {
                all_body_group.push(b as u32);
            }
        }
        #[cfg(feature = "dim3")]
        for (batch_idx, (mb_set, body_ids, _)) in multibody_envs.iter().enumerate() {
            let base = batch_idx * max_colliders;
            for mb in mb_set.multibodies() {
                let group_local = mb
                    .links()
                    .next()
                    .and_then(|root| body_ids.get(&root.rigid_body_handle()).copied());
                let Some(group_local) = group_local else {
                    continue;
                };
                for link in mb.links() {
                    if let Some(&local) = body_ids.get(&link.rigid_body_handle()) {
                        all_body_group[base + local as usize] = group_local;
                    }
                }
            }
        }
        let body_group = Tensor::vector(backend, &all_body_group, BufferUsages::STORAGE).unwrap();

        let num_colliders_per_batch = max_colliders;
        let num_bodies_total = num_colliders_per_batch * num_batches as usize;

        // Initial body velocities were accumulated in body-slot order alongside
        // `all_poses`; zero-filling here would silently drop each body's initial
        // linvel/angvel (gravity used to mask the linear part, but e.g. an
        // initial spin was lost entirely).
        debug_assert_eq!(all_vels.len(), num_bodies_total);
        let storage: BufferUsages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let shapes = Tensor::vector(backend, &all_shapes, storage).unwrap();
        let collider_local_poses =
            Tensor::vector(backend, &all_collider_local_poses, storage).unwrap();
        let collision_groups = Tensor::vector(backend, &all_collision_groups, storage).unwrap();

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

        let contacts_per_batch_cpu = DEFAULT_CONTACT_COUNTS;
        let collision_pairs_per_batch_cpu = DEFAULT_CONTACT_COUNTS;
        let mut bi = BatchIndices::default();
        bi.colliders_batch_capacity = num_colliders_per_batch as u32;
        bi.collision_pairs_batch_capacity = collision_pairs_per_batch_cpu;
        bi.contacts_batch_capacity = contacts_per_batch_cpu;
        bi.impulse_joints_batch_capacity = joints.joints_per_batch();
        bi.color_groups_batch_capacity = joints.num_colors();
        #[cfg(feature = "dim3")]
        multibodies.fill_batch_indices(&mut bi);
        let batch_indices = Tensor::scalar(
            backend,
            bi,
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        )
        .unwrap();

        Self {
            num_batches,
            num_colliders_per_batch: num_colliders_per_batch as u32,
            num_solver_iterations,
            sim_params: Tensor::vector(backend, &all_sim_params, BufferUsages::STORAGE).unwrap(),
            // COPY_DST so `reset_env_from` can overwrite a single env's velocities.
            vels: Tensor::vector(backend, &all_vels, storage | BufferUsages::COPY_DST).unwrap(),
            solver_vels: Tensor::vector(backend, &all_vels, storage).unwrap(),
            solver_vels_out: Tensor::vector(backend, &all_vels, storage).unwrap(),
            solver_vels_inc: Tensor::vector(backend, &all_vels, storage).unwrap(),
            joints,
            #[cfg(feature = "dim3")]
            multibodies,
            body_group,
            local_mprops: Tensor::vector(backend, &all_local_mprops, storage).unwrap(),
            mprops: Tensor::vector(backend, &all_mprops, storage).unwrap(),
            body_poses: Tensor::vector(
                backend,
                &all_poses,
                // COPY_DST so `reset_env_from` can overwrite a single env's poses.
                BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            )
            .unwrap(),
            // Sized like `body_poses`. Will be (re-)seeded each step before
            // the solver runs; the initial contents don't matter, but using
            // the body poses here keeps the buffer in a sensible state.
            solver_body_poses: Tensor::vector(
                backend,
                &all_poses,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            )
            .unwrap(),
            // Sized like `body_poses`. Refreshed by `gpu_sync_collider_poses`
            // once per step before broad-phase / narrow-phase /
            // contact-to-constraint init.
            collider_world_poses: Tensor::vector(
                backend,
                &all_poses,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            )
            .unwrap(),
            vertex_buffers,
            index_buffers,
            shapes,
            num_shapes,
            collider_local_poses,
            collision_groups,
            collision_pairs,
            collision_pairs_len,
            collision_pairs_len_staging,
            collision_pairs_indirect,
            collision_pairs_batch_capacity,
            contacts_batch_capacity,
            colliders_batch_capacity,
            contacts_per_batch_cpu,
            collision_pairs_per_batch_cpu,
            batch_indices,
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
            // Constant color values [1..=256] for the solver's per-color sweeps.
            color_uniforms: (1u32..=256)
                .map(|c| {
                    Tensor::scalar(backend, c, BufferUsages::STORAGE | BufferUsages::UNIFORM)
                        .unwrap()
                })
                .collect(),
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
            max_colors: 8,
        }
    }
}

impl GpuPhysicsState {
    /// Re-upload the shared `BatchIndices` uniform after any of its
    /// constituent per-batch capacities has changed (e.g. after the contacts
    /// buffer grows in [`Self::auto_resize_buffers`], or after multibody
    /// impulse-joint capacities are updated via
    /// [`GpuMultibodySet::set_impulse_joints`]). Call whenever a cap edit
    /// happens that any kernel reads via its `batch_ids` uniform.
    fn rebuild_batch_indices(&mut self, backend: &GpuBackend) {
        let mut bi = BatchIndices::default();
        bi.colliders_batch_capacity = self.num_colliders_per_batch;
        bi.collision_pairs_batch_capacity = self.collision_pairs_per_batch_cpu;
        bi.contacts_batch_capacity = self.contacts_per_batch_cpu;
        bi.impulse_joints_batch_capacity = self.joints.joints_per_batch();
        bi.color_groups_batch_capacity = self.joints.num_colors();
        #[cfg(feature = "dim3")]
        self.multibodies.fill_batch_indices(&mut bi);
        self.batch_indices = Tensor::scalar(
            backend,
            bi,
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        )
        .unwrap();
    }

    /// Shared per-batch index uniform — see [`Self::rebuild_batch_indices`].
    pub fn batch_indices(&self) -> &Tensor<BatchIndices> {
        &self.batch_indices
    }

    /// Sets the maximum number of constraint colors used by the per-step
    /// graph coloring + Gauss-Seidel solver loop. Lower values cap solver
    /// time at the cost of dropping over-budget constraints.
    pub fn set_max_colors(&mut self, max_colors: u32) {
        self.max_colors = max_colors.max(1);
    }

    /// Returns the configured max color count.
    pub fn max_colors(&self) -> u32 {
        self.max_colors
    }
}

impl GpuPhysicsState {
    /// Per-collider world pose (= `body_poses[i] * collider_local_poses[i]`).
    /// This is what rendering / debug tooling typically wants — the actual
    /// pose of each collider's shape in world space.
    ///
    /// Refreshed once per step before broad-phase / narrow-phase / contact
    /// constraint init; not mutated during the substep loop.
    pub fn poses(&self) -> &Tensor<Pose> {
        &self.collider_world_poses
    }

    /// DEBUG accessors for the contact/pair buffers (native-CUDA diagnosis).
    pub fn dbg_collision_pairs(&self) -> &Tensor<[u32; 2]> {
        &self.collision_pairs
    }
    pub fn dbg_collision_pairs_len(&self) -> &Tensor<u32> {
        &self.collision_pairs_len
    }
    pub fn dbg_contacts_len(&self) -> &Tensor<u32> {
        &self.contacts_len
    }
    pub fn dbg_contacts(&self) -> &Tensor<GpuIndexedContact> {
        &self.contacts
    }
    pub fn dbg_num_batches(&self) -> u32 {
        self.num_batches
    }
    pub fn dbg_contacts_capacity(&self) -> u32 {
        self.collision_pairs.len() as u32 / self.num_batches
    }

    /// Per-body world-origin pose (matches rapier's `RigidBody::position`).
    pub fn body_poses(&self) -> &Tensor<Pose> {
        &self.body_poses
    }

    /// Per-collider world pose. Same as [`Self::poses`].
    pub fn collider_world_poses(&self) -> &Tensor<Pose> {
        &self.collider_world_poses
    }

    /// The set of joints part of the simulation.
    pub fn joints(&self) -> &GpuImpulseJointSet {
        &self.joints
    }

    /// Mutable access to the multibody set, useful for runtime mutations like
    /// per-step motor changes.
    #[cfg(feature = "dim3")]
    pub fn multibodies_mut(&mut self) -> &mut crate::dynamics::GpuMultibodySet {
        &mut self.multibodies
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

    /// Reset a single environment in-place to the state of a freshly built
    /// single-env state `src` (which must describe one batch with the same scene
    /// layout as this one). Copies the per-env carry-over state — body poses,
    /// velocities, and the multibody joint-space state — so env `dst_env` starts
    /// fresh while the other environments keep running. The per-step scratch
    /// (contacts, constraints, colors, broad-phase) is recomputed each step and
    /// needs no copy. This is the per-env "reset" RL environments need.
    #[cfg(feature = "dim3")]
    pub async fn reset_env_from(&mut self, backend: &GpuBackend, dst_env: u32, src: &GpuPhysicsState) {
        let nb = self.num_batches as u64;
        // body_poses + velocities (per-batch slabs; stride = len / num_batches)
        let mut bp = bytemuck::zeroed_vec(src.body_poses.len() as usize);
        backend.slow_read_buffer(src.body_poses.buffer(), &mut bp).await.unwrap();
        let bps = (self.body_poses.len() / nb) as usize;
        backend
            .write_buffer(self.body_poses.buffer_mut(), dst_env as u64 * bps as u64, &bp[..bps])
            .unwrap();

        let mut vv = bytemuck::zeroed_vec(src.vels.len() as usize);
        backend.slow_read_buffer(src.vels.buffer(), &mut vv).await.unwrap();
        let vs = (self.vels.len() / nb) as usize;
        backend
            .write_buffer(self.vels.buffer_mut(), dst_env as u64 * vs as u64, &vv[..vs])
            .unwrap();

        self.multibodies.reset_env_from(backend, dst_env, &src.multibodies).await;
    }

    /// Read this (template) physics state off the GPU into a CPU snapshot. Call
    /// once per template at setup; pass the result to
    /// [`Self::reset_env_from_snapshot`] for readback-free per-env resets.
    #[cfg(feature = "dim3")]
    pub async fn snapshot(&self, backend: &GpuBackend) -> GpuPhysicsSnapshot {
        let mut body_poses = bytemuck::zeroed_vec(self.body_poses.len() as usize);
        backend.slow_read_buffer(self.body_poses.buffer(), &mut body_poses).await.unwrap();
        let mut vels = bytemuck::zeroed_vec(self.vels.len() as usize);
        backend.slow_read_buffer(self.vels.buffer(), &mut vels).await.unwrap();
        let mb = self.multibodies.snapshot(backend).await;
        GpuPhysicsSnapshot { body_poses, vels, mb }
    }

    /// Reset env `dst_env` from a CPU snapshot using `write_buffer` only — no
    /// GPU→CPU readback. Equivalent to [`Self::reset_env_from`] against a template
    /// matching `snap`, but eliminates the ~6 per-reset sync stalls that dominate
    /// reset cost on the WebGPU backend.
    #[cfg(feature = "dim3")]
    pub fn reset_env_from_snapshot(
        &mut self,
        backend: &GpuBackend,
        dst_env: u32,
        snap: &GpuPhysicsSnapshot,
    ) {
        let nb = self.num_batches as u64;
        let bps = (self.body_poses.len() / nb) as usize;
        backend
            .write_buffer(self.body_poses.buffer_mut(), dst_env as u64 * bps as u64, &snap.body_poses[..bps])
            .unwrap();
        let vs = (self.vels.len() / nb) as usize;
        backend
            .write_buffer(self.vels.buffer_mut(), dst_env as u64 * vs as u64, &snap.vels[..vs])
            .unwrap();
        self.multibodies.reset_env_from_snapshot(backend, dst_env, &snap.mb);
    }

    /// [`Self::reset_env_from_snapshot`] with the robot rigidly translated by
    /// `offset` (world frame) — the teleport primitive for terrain-curriculum
    /// style spawn placement. Only floating-base multibody links move; fixed
    /// bodies (ground, terrain) keep their snapshot poses. Costs one small
    /// snapshot clone per call (single-env sized).
    #[cfg(feature = "dim3")]
    pub fn reset_env_from_snapshot_offset(
        &mut self,
        backend: &GpuBackend,
        dst_env: u32,
        snap: &GpuPhysicsSnapshot,
        offset: Vector,
    ) {
        let moved = snap.translated(offset);
        self.reset_env_from_snapshot(backend, dst_env, &moved);
    }
}

/// CPU-side snapshot of one (single-batch) physics template — body poses,
/// velocities, and the multibody joint-space state — read off the GPU once for
/// readback-free resets. See [`GpuPhysicsState::snapshot`].
#[cfg(feature = "dim3")]
#[derive(Clone)]
pub struct GpuPhysicsSnapshot {
    body_poses: Vec<Pose>,
    vels: Vec<GpuVelocity>,
    mb: GpuMultibodySnapshot,
}

#[cfg(feature = "dim3")]
impl GpuPhysicsSnapshot {
    /// A copy with every floating-base multibody translated by `offset`:
    /// the affected links' `body_poses` plus the multibody workspace (root
    /// free-joint coords, local_to_parent, per-link local_to_world). Fixed
    /// bodies (ground/terrain) and velocities are untouched.
    pub fn translated(&self, offset: Vector) -> GpuPhysicsSnapshot {
        let mut out = self.clone();
        out.mb = self.mb.translated(offset);
        self.mb.for_each_link_rb_id(|rb_id| {
            if let Some(p) = out.body_poses.get_mut(rb_id as usize) {
                p.translation += offset;
            }
        });
        out
    }
}

/// The main GPU physics pipeline coordinating all simulation stages.
pub struct GpuPhysicsPipeline {
    mprops_update: GpuMpropsUpdate,
    sync_collider_poses: crate::dynamics::GpuSyncColliderPosesShader,
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
            sync_collider_poses: crate::dynamics::GpuSyncColliderPosesShader::from_backend(backend)
                .unwrap(),
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

        // Phase 0: Multibody once-per-visible-step setup (3D only for now).
        #[cfg(feature = "dim3")]
        {
            if !state.multibodies.is_empty() {
                let mut encoder = backend.begin_encoding();
                let mut pass = encoder.begin_pass("multibody-init-step", timestamps.as_deref_mut());
                let mut args = crate::dynamics::MultibodySolverArgs {
                    poses: &mut state.body_poses,
                    collider_world_poses: &state.collider_world_poses,
                    mprops: &state.mprops,
                    contacts: &state.contacts,
                    contacts_len: &state.contacts_len,
                    solver_vels: &mut state.solver_vels,
                    batch_indices: &state.batch_indices,
                    sim_params: &state.sim_params,
                };
                self.multibody_solver
                    .init_step(&mut pass, &mut state.multibodies, &mut args)
                    .unwrap();
                drop(pass);
                drop(args);
                backend.submit(encoder).unwrap();
            }
        }

        // Phase 1: Update mass properties, build LBVH, and find collision pairs.
        {
            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass("update-mprops", timestamps.as_deref_mut());

            // Update mass properties — uses body world poses to compute the
            // world COM and inertia tensor.
            self.mprops_update
                .dispatch(
                    &mut pass,
                    &mut state.mprops,
                    &state.local_mprops,
                    &state.body_poses,
                    &state.num_shapes,
                    &state.batch_indices,
                    state.num_colliders_per_batch,
                    state.num_batches,
                )
                .unwrap();

            // Update collider world-space poses from their parent rigid-body poses.
            self.sync_collider_poses
                .dispatch(
                    &mut pass,
                    &state.body_poses,
                    &state.collider_local_poses,
                    &mut state.collider_world_poses,
                    &state.num_shapes,
                    &state.batch_indices,
                    state.num_colliders_per_batch,
                    state.num_batches,
                )
                .unwrap();

            drop(pass);

            // Build LBVH and find collision pairs.
            self.lbvh
                .update_tree(
                    backend,
                    &mut encoder,
                    &mut state.lbvh,
                    state.collider_local_poses.len() as u32,
                    state.num_batches,
                    &state.collider_world_poses,
                    &state.vertex_buffers,
                    &state.shapes,
                    &state.num_shapes,
                    &state.batch_indices,
                    timestamps.as_deref_mut(),
                )
                .unwrap();

            // Debug: validate LBVH topology after tree construction
            if crate::VALIDATE_LBVH_TOPOLOGY {
                backend.submit(encoder).unwrap();

                let num_colliders = state.collider_world_poses.len() as u32;
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
                    state.body_poses.len() as u32,
                    state.num_batches,
                    &state.num_shapes,
                    &state.batch_indices,
                    &mut state.collision_pairs,
                    &mut state.collision_pairs_len,
                    &mut state.collision_pairs_indirect,
                    &state.collision_groups,
                )
                .unwrap();

            drop(pass);
            backend.submit(encoder).unwrap();
        }

        stats.start_to_pairs_count_time = t_phase1.elapsed();

        // Phase 2a: Narrow phase. Split out from solver-prep + coloring
        // so its CPU encoding overlaps with Phase 1's GPU work and its
        // own GPU work overlaps with Phase 2b's CPU encoding.
        {
            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass("narrow-phase", timestamps.as_deref_mut());

            self.narrow_phase
                .dispatch(
                    &mut pass,
                    state.body_poses.len() as u32,
                    &state.collider_world_poses,
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
                    &state.batch_indices,
                )
                .unwrap();

            drop(pass);
            backend.submit(encoder).unwrap();
        }

        // Phase 2b: solver-prep + warmstart + bounded coloring. Separate
        // submit from narrow-phase to enable CPU/GPU overlap with the
        // upcoming Phase 3 solver substep loop.
        {
            let mut encoder = backend.begin_encoding();
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
                body_poses: &mut state.body_poses,
                solver_body_poses: &mut state.solver_body_poses,
                collider_local_poses: &state.collider_local_poses,
                collider_world_poses: &state.collider_world_poses,
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
                color_uniforms: &state.color_uniforms,
                prefix_sum: &self.prefix_sum,
                num_colors: 0,
                num_batches: state.num_batches,
                num_colliders: state.num_colliders_per_batch,
                num_solver_iterations: state.num_solver_iterations,
                body_group: &state.body_group,
                batch_indices: &state.batch_indices,
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
                batch_indices: &state.batch_indices,
            };

            self.warmstart
                .transfer_warmstart_impulses(&mut pass, warmstart_args)
                .unwrap();

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
                batch_indices: &state.batch_indices,
                body_group: &state.body_group,
            };
            self.coloring
                .dispatch_topo_gc_bounded(&mut pass, coloring_args, state.max_colors)
                .unwrap();

            // `+1` because solver iterates 1..=max_colors (color 0 is unassigned).
            let num_colors = state.max_colors + 1;
            stats.num_colors = num_colors;

            drop(pass);
            backend.submit(encoder).unwrap();
        }

        let num_colors = stats.num_colors;

        // Create solver_args for solve phase (after coloring is complete)
        let solver_args = SolverArgs {
            contacts: &state.contacts,
            contacts_len: &state.contacts_len,
            contacts_len_indirect: &state.contacts_indirect,
            constraints: &mut state.new_constraints,
            constraint_builders: &mut state.new_constraint_builders,
            sim_params: &state.sim_params,
            colliders_len: &state.num_shapes,
            body_poses: &mut state.body_poses,
            solver_body_poses: &mut state.solver_body_poses,
            collider_local_poses: &state.collider_local_poses,
            collider_world_poses: &state.collider_world_poses,
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
            color_uniforms: &state.color_uniforms,
            prefix_sum: &self.prefix_sum,
            num_colors,
            num_batches: state.num_batches,
            num_colliders: state.num_colliders_per_batch,
            num_solver_iterations: state.num_solver_iterations,
            body_group: &state.body_group,
            batch_indices: &state.batch_indices,
        };

        // Phase 3: Solve constraints
        let joint_solver_args = JointSolverArgs {
            num_batches: state.num_batches,
            sim_params: &state.sim_params,
            mprops: &state.mprops,
            local_mprops: &state.local_mprops,
            joints: &mut state.joints,
            batch_indices: &state.batch_indices,
            color_uniforms: &state.color_uniforms,
        };

        {
            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass("solver", timestamps.as_deref_mut());
            #[cfg(feature = "dim3")]
            let mb = if state.multibodies.is_empty() {
                None
            } else {
                Some((&self.multibody_solver, &mut state.multibodies))
            };
            self.solver
                .solve_tgs(
                    &mut pass,
                    &self.joint_solver,
                    solver_args,
                    joint_solver_args,
                    #[cfg(feature = "dim3")]
                    mb,
                )
                .unwrap();
            drop(pass);

            // Resolve all accumulated timestamps before the final submit.
            if let Some(ts) = &timestamps {
                ts.resolve(&mut encoder);
            }
            backend.submit(encoder).unwrap();
        }

        // Swap buffers for warm-starting next frame
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

    pub async fn auto_resize_buffers(&self, backend: &GpuBackend, state: &mut GpuPhysicsState) {
        let mut encoder = backend.begin_encoding();
        encoder
            .copy_buffer_to_buffer(
                state.collision_pairs_len.buffer(),
                0,
                state.collision_pairs_len_staging.buffer_mut(),
                0,
                1,
            )
            .unwrap();
        encoder
            .copy_buffer_to_buffer(
                state.uncolored.buffer(),
                0,
                state.uncolored_staging.buffer_mut(),
                0,
                1,
            )
            .unwrap();
        backend.submit(encoder).unwrap();

        let mut collision_pairs_len = [0u32];
        let mut coloring_converged = [0u32];
        backend
            .read_buffer(
                state.collision_pairs_len_staging.buffer(),
                &mut collision_pairs_len,
            )
            .await
            .unwrap();
        backend
            .read_buffer(state.uncolored_staging.buffer(), &mut coloring_converged)
            .await
            .unwrap();

        if coloring_converged[0] == 0 {
            state.max_colors += 5;
        }

        // Lazy resize: grow collision-pair / contact / constraint buffers
        // based on the *previous* frame's max pair count.
        // This can create a one-frame delay where a bunch of contacts are ignored for a frame
        // if their count exceed the allocated buffer’s size for contacts. But this delay allows
        // us to avoid a gpu-cpu sync in the middle of the physics pipeline.
        {
            let per_batch_capacity = state.collision_pairs.len() as u32 / state.num_batches;
            // Add a 25% slack so we resize once for a band of nearby
            // overflows instead of bouncing on each new pair.
            let needed = collision_pairs_len[0].saturating_add(collision_pairs_len[0] / 4);
            if needed >= per_batch_capacity {
                let storage: BufferUsages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
                let desired_len = needed.next_power_of_two().max(per_batch_capacity);
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
                state.pfm_pairs =
                    Tensor::vector_uninit(backend, desired_len * nb, storage).unwrap();
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

                state.collision_pairs_per_batch_cpu = desired_len;
                state.contacts_per_batch_cpu = desired_len;
                state.rebuild_batch_indices(backend);
            }
        }
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

/// Builds a GPU-side [`LocalMassProperties`] from a parry/rapier
/// [`crate::rapier::prelude::MassProperties`]. The body-local COM, principal-axis
/// frame, inverse mass and inverse principal inertia are copied verbatim.
#[cfg(feature = "from_rapier")]
fn local_mprops_from_rapier(
    mprops: &crate::rapier::prelude::MassProperties,
) -> GpuLocalMassProperties {
    #[cfg(feature = "dim2")]
    {
        GpuLocalMassProperties {
            inv_mass: glamx::Vec2::splat(mprops.inv_mass),
            com: mprops.local_com,
            padding2: 0,
            inv_inertia: mprops.inv_principal_inertia,
        }
    }
    #[cfg(feature = "dim3")]
    {
        GpuLocalMassProperties {
            inertia_ref_frame: mprops.principal_inertia_local_frame,
            inv_principal_inertia: mprops.inv_principal_inertia,
            padding0: 0,
            inv_mass: glamx::Vec3::splat(mprops.inv_mass),
            padding1: 0,
            com: mprops.local_com,
            padding2: 0,
        }
    }
}

/// Computes the world-space mass properties of a body from its body-origin world
/// pose and body-local mass properties. Mirrors the GPU `update_mprops` shader
/// so the buffer is consistent the moment the simulation starts.
#[cfg(feature = "from_rapier")]
fn world_mprops_from_local(pose: &Pose, local: &GpuLocalMassProperties) -> GpuWorldMassProperties {
    #[cfg(feature = "dim2")]
    {
        GpuWorldMassProperties {
            inv_inertia: local.inv_inertia,
            inv_mass: local.inv_mass,
            padding1: 0,
            com: *pose * local.com,
        }
    }
    #[cfg(feature = "dim3")]
    {
        // Build the world-space inverse inertia tensor: R * diag * R^T, with R
        // the rotation taking body space to the world principal-inertia frame.
        // Mirrors the GPU `update_mprops` shader so the buffer is consistent
        // before the first `update_mprops` dispatch.
        let world_principal_frame = pose.rotation * local.inertia_ref_frame;
        let rot_mat = glamx::Mat3::from_quat(world_principal_frame);
        let scaled = glamx::Mat3::from_cols(
            rot_mat.x_axis * local.inv_principal_inertia.x,
            rot_mat.y_axis * local.inv_principal_inertia.y,
            rot_mat.z_axis * local.inv_principal_inertia.z,
        );
        let inv_inertia_3 = scaled * rot_mat.transpose();
        let inv_inertia = glamx::Mat4::from_mat3(inv_inertia_3);
        GpuWorldMassProperties {
            inv_inertia,
            inv_mass: local.inv_mass,
            padding0: 0,
            com: *pose * local.com,
            padding1: 0,
        }
    }
}
