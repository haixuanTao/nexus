//! Host-side GPU multibody: buffer packing and kernel dispatch.
//!
//! A single `GpuMultibodySet` packs N rapier `Multibody`s across multiple simulation
//! batches into flat GPU tensors (see the shader module for the memory layout).
//!
//! The `GpuMultibodySolver::step` runs the following passes per workgroup-per-multibody:
//!
//!   1. `integrate` — advance coords/velocities using the previous step's acceleration.
//!   2. `forward_kinematics` — link poses / shifts (also writes into the shared body poses).
//!   3. `body_jacobians` — per-link `6 × ndofs` jacobians.
//!   4. `update_velocities` — world-frame per-link `rb_vels` / `joint_velocity`.
//!   5. Mass matrix: `mass_matrix_with_coriolis` if `implicit_coriolis` is on,
//!      otherwise the plain CRBA `mass_matrix`. Both paths add `damping·dt` to the diagonal.
//!   6. `apply_gravity_with_coriolis` — `τ = Σ Jᵀ · (m·g − m·a_kin, −gyro − I·α_kin) − damping ⊙ ẋ`.
//!   7. `lu_decompose` + `lu_solve` — solve for the generalized acceleration in place.
//!
//! Per-joint damping is seeded from rapier's `default_damping` (0.1 on every free
//! angular DOF).
//!
//! Contacts and user-defined joint constraints are intentionally not handled.

#![cfg(feature = "dim3")]

use crate::shaders::utils::linalg::MAX_MB_DOFS;
use crate::math::Pose;
use crate::queries::GpuIndexedContact;
use crate::shaders::dynamics::{
    GpuMbApplyGravityWithCoriolis, GpuMbBodyJacobians, GpuMbFinalizeContactConstraints,
    GpuMbForwardKinematics, GpuMbInitContactConstraints, GpuMbInitJointConstraints, GpuMbIntegrate,
    GpuMbComputeDynamicsPre, GpuMbGravityAndLu, GpuMbIntegrateVelocities, GpuMbLuDecompose,
    GpuMbComputeDynamicsWithoutCoriolisPre,
    GpuMbLuFactorAndSolve, GpuMbLuSolve, GpuMbMassMatrix,
    GpuMbMassMatrixWithCoriolis, GpuMbRemoveContactConstraintBias,
    GpuMbRemoveImpulseJointConstraintBias, GpuMbRemoveJointConstraintBias,
    GpuMbSolveContactConstraints, GpuMbSolveImpulseJointConstraints, GpuMbSolveJointConstraints,
    GpuMbUpdateImpulseJointConstraints, GpuMbUpdateVelocities, LocalMassProperties,
    MAX_AXIS_CONSTRAINTS, MAX_MB_CONTACT_CONSTRAINTS_PER_MB, MbImpulseJointBuilder,
    MbImpulseJointConstraint,
    MultibodyContactConstraint, MultibodyInfo, MultibodyJointConstraint, MultibodyLinkStatic,
    MultibodyLinkWorkspace, SIDE_KIND_BODY, SIDE_KIND_FIXED, SIDE_KIND_MB, Velocity,
    WorldMassProperties,
};
use khal::backend::{Backend, GpuBackend, GpuBackendError, GpuPass};
use khal::{BufferUsages, Shader};
use rapier3d::prelude::JointAxis;
use vortx::tensor::Tensor;

/// Workgroup width (in lanes) for the parallelised mass-matrix kernel. Must
/// match the `threads(N, 1, 1)` attribute on
/// `gpu_mb_mass_matrix_with_coriolis`.
const MB_MM_LANES: u32 = 32;

/// Workgroup width for the parallelised body-jacobian kernel. Must match the
/// `threads(N, 1, 1)` attribute on `gpu_mb_body_jacobians`.
const MB_BJ_LANES: u32 = 32;

/// Workgroup width for the parallelised gravity / Coriolis-force kernel. Must
/// match the `threads(N, 1, 1)` attribute on
/// `gpu_mb_apply_gravity_with_coriolis`.
const MB_GRAV_LANES: u32 = 32;

/// Workgroup width for the parallelised LU decompose / solve kernels. Must
/// match the `threads(N, 1, 1)` attribute on `gpu_mb_lu_decompose` and
/// `gpu_mb_lu_solve`.
const MB_LU_LANES: u32 = 32;

#[cfg(feature = "from_rapier")]
use {
    crate::rapier::dynamics::{
        ImpulseJointSet, MultibodyJointSet, RigidBodyHandle, RigidBodySet,
    },
    crate::shaders::dynamics::{GenericJoint, JointLimits, JointMotor},
    std::collections::HashMap,
};

/// GPU-resident articulated multibody set, packed across simulation batches.
///
/// Every buffer is a flat tensor with per-batch capacity (`*_batch_capacity`) and
/// a per-batch length (`num_multibodies`, `num_links`).
pub struct GpuMultibodySet {
    num_batches: u32,
    multibodies_per_batch: u32,
    links_per_batch: u32,
    #[allow(dead_code)]
    dofs_per_batch: u32,
    #[allow(dead_code)]
    jacobian_entries_per_batch: u32,
    #[allow(dead_code)]
    mass_matrix_entries_per_batch: u32,
    #[allow(dead_code)]
    coriolis_entries_per_batch: u32,
    #[allow(dead_code)]
    i_coriolis_dt_entries_per_batch: u32,
    /// When `true`, the Coriolis / gyroscopic terms are folded into the mass
    /// matrix (implicit integration). When `false`, the mass matrix stays the
    /// plain CRBA form but the Coriolis and gyroscopic forces are still applied
    /// explicitly as part of the RHS (via `apply_gravity_with_coriolis`).
    implicit_coriolis: bool,
    /// Sum across batches/multibodies of `MultibodyInfo::max_constraints`. When
    /// zero (e.g. no joint limits / motors anywhere), the entire joint
    /// constraint init / solve / remove-bias kernel chain can be skipped on the
    /// host side — saves O(substeps × #kernels) WebGPU dispatches per frame.
    has_joint_constraints: bool,

    /// Per-batch number of multibodies.
    num_multibodies: Tensor<u32>,
    /// Per-batch multibody descriptors.
    multibody_info: Tensor<MultibodyInfo>,
    /// Per-batch static link data.
    links_static: Tensor<MultibodyLinkStatic>,
    /// CPU-side mirror of [`Self::links_static`] used to support runtime
    /// mutations like motor changes without round-tripping through a GPU read.
    links_static_mirror: Vec<MultibodyLinkStatic>,
    /// Per-batch per-step link workspace.
    links_workspace: Tensor<MultibodyLinkWorkspace>,
    /// Per-link mass properties (owned by the multibody — the shared body mprops
    /// are zeroed for multibody-controlled bodies so the RBD pipeline skips them).
    links_mprops: Tensor<LocalMassProperties>,
    /// Generalized coordinates (flat).
    dof_values: Tensor<f32>,
    /// Generalized velocities (flat).
    dof_velocities: Tensor<f32>,
    /// Per-DOF damping (same layout as `dof_velocities`). Seeded from each joint's
    /// `default_damping` — 0.1 on every free angular DOF, 0 elsewhere.
    damping: Tensor<f32>,
    /// Generalized forces / after solve, generalized accelerations.
    gen_forces: Tensor<f32>,
    /// Per-link `6 × ndofs` column-major jacobians.
    body_jacobians: Tensor<f32>,
    /// Per-multibody `ndofs × ndofs` mass matrices (also used as LU work buffer).
    mass_matrices: Tensor<f32>,
    /// Per-DOF pivot buffer used by LU.
    lu_pivots: Tensor<u32>,

    /// Per-link `3 × ndofs` Coriolis-linear-rows buffer (rapier's `coriolis_v`).
    coriolis_v: Tensor<f32>,
    /// Per-link `3 × ndofs` Coriolis-angular-rows buffer (rapier's `coriolis_w`).
    coriolis_w: Tensor<f32>,
    /// Per-multibody `6 × ndofs` scratch (rapier's `i_coriolis_dt`).
    i_coriolis_dt: Tensor<f32>,

    /// Per-multibody flat bank of unit (1-DOF) limit / motor constraints.
    joint_constraints: Tensor<MultibodyJointConstraint>,
    /// Per-constraint columns of `M⁻¹` (length `ndofs` each, contiguous per multibody).
    joint_constraint_columns: Tensor<f32>,

    /// Per-body lookup `[multibody_idx, link_idx]` (`u32::MAX` sentinel for
    /// free / non-multibody bodies). Indexed by the per-batch local body id;
    /// matches the layout of the shared body buffers (stride =
    /// `colliders_batch_capacity`). Used by the contact-constraint
    /// generation kernel to find which multibody / link a contact touches.
    body_to_link: Tensor<[u32; 2]>,

    /// Per-multibody bank of contact constraints (1 normal + 2 friction per
    /// touched contact point). Each constraint's M⁻¹ column lives at the
    /// matching slot in `contact_constraint_columns`, and its `Jᵀ` row at the
    /// matching slot in `contact_constraint_jacs`.
    contact_constraints: Tensor<MultibodyContactConstraint>,
    /// Per-constraint `Jᵀ` row (length `ndofs`) — the multibody side's
    /// contribution to the constraint Jacobian, written by the init kernel.
    contact_constraint_jacs: Tensor<f32>,
    /// Per-constraint M⁻¹·Jᵀ column (length `ndofs`) — written by the
    /// finalize kernel via LU back-substitution.
    contact_constraint_columns: Tensor<f32>,
    /// Per-multibody count of currently-active contact constraints. Filled
    /// by the init kernel; read by the solve / finalize kernels.
    contact_constraint_count: Tensor<u32>,

    /// Per-batch number of multibody-touching impulse joints. Counts
    /// joints whose body1 OR body2 is part of any multibody — these go
    /// through the `MbImpulseJointConstraint` path because the regular
    /// impulse-joint solver can't propagate impulses through `M⁻¹·Jᵀ`.
    mb_imp_joint_count: Tensor<u32>,
    /// Per-batch slab of impulse-joint builder descriptors. One slot per
    /// joint touching the multibody side; padded to
    /// `mb_imp_joints_per_batch` with all-zero entries.
    mb_imp_joint_builders: Tensor<MbImpulseJointBuilder>,
    /// Per-batch slab of axis constraints — `MAX_AXIS_CONSTRAINTS` slots
    /// per builder. Filled (and inactive-marked) by
    /// `gpu_mb_update_impulse_joint_constraints`.
    mb_imp_joint_constraints: Tensor<MbImpulseJointConstraint>,
    /// Per-batch flat jacobians buffer — stores `J / W·J` for both sides
    /// of every axis constraint of every joint. See
    /// `MbImpulseJointConstraint` for the per-axis layout.
    mb_imp_joint_jacobians: Tensor<f32>,

    /// Capacities (per-batch strides) for the impulse-joint slabs above.
    mb_imp_joints_batch_capacity: Tensor<u32>,
    mb_imp_joint_constraints_batch_capacity: Tensor<u32>,
    mb_imp_joint_jacobians_batch_capacity: Tensor<u32>,
    mb_imp_joints_per_batch: u32,

    /// Number of solver iterations to run on `joint_constraints` per `step()`.
    num_solver_iterations: u32,

    multibodies_batch_capacity: Tensor<u32>,
    links_batch_capacity: Tensor<u32>,
    dof_batch_capacity: Tensor<u32>,
    jacobians_batch_capacity: Tensor<u32>,
    mass_matrix_batch_capacity: Tensor<u32>,
    coriolis_batch_capacity: Tensor<u32>,
    i_coriolis_dt_batch_capacity: Tensor<u32>,
    joint_constraints_batch_capacity: Tensor<u32>,
    joint_constraint_columns_batch_capacity: Tensor<u32>,
    contact_constraints_batch_capacity: Tensor<u32>,
    contact_constraint_columns_batch_capacity: Tensor<u32>,
    /// Stride (per-batch capacity) for `body_to_link` — same as colliders.
    contacts_batch_capacity_for_mb: Tensor<u32>,

    /// Gravity vector (uploaded as `[f32; 3]`).
    gravity: Tensor<[f32; 3]>,
    /// Current integration timestep (1-element buffer so kernels can read it as f32).
    dt: Tensor<f32>,
}

impl GpuMultibodySet {
    /// Number of simulation batches.
    pub fn num_batches(&self) -> u32 {
        self.num_batches
    }

    /// Capacity (max multibodies) per batch.
    pub fn multibodies_per_batch(&self) -> u32 {
        self.multibodies_per_batch
    }

    /// True if the set contains no multibodies in any batch.
    pub fn is_empty(&self) -> bool {
        self.multibodies_per_batch == 0 || self.links_per_batch == 0
    }

    /// GPU buffer for generalized velocities (flat, one slot per DOF).
    pub fn dof_velocities(&self) -> &Tensor<f32> {
        &self.dof_velocities
    }

    /// GPU buffer for generalized coordinates.
    pub fn dof_values(&self) -> &Tensor<f32> {
        &self.dof_values
    }

    /// GPU buffer for the last-computed generalized accelerations (populated by
    /// `GpuMultibodySolver::solve_gravity`).
    pub fn gen_accelerations(&self) -> &Tensor<f32> {
        &self.gen_forces
    }

    /// Enable implicit integration of the Coriolis / gyroscopic terms.
    ///
    /// In both modes the Coriolis / gyroscopic forces are computed and applied
    /// explicitly to the generalized-force vector `τ` (`apply_gravity_with_coriolis`).
    /// The only difference is the mass matrix:
    ///
    /// - `true`: the mass matrix is `M + M_gyro·dt + C·dt` (matches rapier's
    ///   `acc_augmented_mass`). This implicit treatment stabilizes the integrator
    ///   at large time-steps.
    /// - `false`: the mass matrix is the plain CRBA form `Σ Jᵀ · diag(m·I, I) · J`.
    ///   Simpler and slightly cheaper; can become unstable for fast rotations.
    pub fn set_implicit_coriolis(&mut self, enabled: bool) {
        self.implicit_coriolis = enabled;
    }

    /// Whether the Coriolis / gyroscopic terms are folded into the mass matrix
    /// (implicit integration) in the next `step()`.
    pub fn implicit_coriolis(&self) -> bool {
        self.implicit_coriolis
    }

    /// Number of TGS-soft substeps per visible step. Each substep runs the full
    /// pipeline (FK, mass matrix, gravity, LU, integrate, constraint solve,
    /// stabilization) with `dt' = visible_dt / num_solver_iterations` — matches
    /// rapier's `num_solver_iterations`.
    pub fn num_solver_iterations(&self) -> u32 {
        self.num_solver_iterations
    }

    /// Set the number of TGS-soft substeps (default `4`). Note: this does not
    /// re-upload `dt`; call [`set_visible_dt`](Self::set_visible_dt) afterwards
    /// to refresh the GPU substep-dt buffer.
    pub fn set_num_solver_iterations(&mut self, n: u32) {
        self.num_solver_iterations = n;
    }

    /// Upload the visible-frame `dt`. Internally divides by `num_solver_iterations`
    /// and stores the *substep* dt (which is what the GPU kernels read).
    pub fn set_visible_dt(&mut self, backend: &GpuBackend, visible_dt: f32) {
        let n = self.num_solver_iterations.max(1) as f32;
        self.dt = Tensor::scalar(
            backend,
            visible_dt / n,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        )
        .unwrap();
    }

    /// Sets a motor's target velocity on a multibody joint and uploads the
    /// updated link to the GPU. `link_id` is the global link id within the
    /// batch (matches the body / collider index that was given to
    /// [`from_rapier`](Self::from_rapier)). `axis` is the joint axis index
    /// (0..=2 for linear, 3..=5 for angular).
    ///
    /// The motor is also auto-enabled (its bit is set in `motor_axes`) so the
    /// solver actually drives the joint at the requested velocity.
    pub fn set_motor_velocity(
        &mut self,
        backend: &GpuBackend,
        batch: u32,
        link_id: u32,
        axis: JointAxis,
        target_vel: f32,
    ) -> Result<(), GpuBackendError> {
        let stride = self.links_per_batch;
        let global_idx = (batch * stride + link_id) as usize;
        let axis_id = axis as usize;
        let entry = match self.links_static_mirror.get_mut(global_idx) {
            Some(e) => e,
            None => return Ok(()),
        };
        entry.data.motors[axis_id].target_vel = target_vel;
        entry.data.motor_axes |= 1u32 << axis_id;
        let snapshot = *entry;
        backend.write_buffer(
            self.links_static.buffer_mut(),
            global_idx as u64,
            std::slice::from_ref(&snapshot),
        )
    }

    /// Upload a new gravity vector.
    pub fn set_gravity(&mut self, backend: &GpuBackend, g: [f32; 3]) {
        self.gravity = Tensor::scalar(
            backend,
            g,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        )
        .unwrap();
    }

    /// Convert a slice of per-batch `(MultibodyJointSet, body_ids_map)` pairs into
    /// packed GPU buffers. `body_ids` maps each rapier `RigidBodyHandle` to the
    /// corresponding collider/body index used elsewhere (poses, mprops buffers).
    ///
    /// Root links must be the first link in their multibody (rapier guarantees
    /// this via assembly ids being assigned in traversal order).
    #[cfg(feature = "from_rapier")]
    pub fn from_rapier(
        backend: &GpuBackend,
        environments: &[(
            &MultibodyJointSet,
            &HashMap<RigidBodyHandle, u32>,
            &RigidBodySet,
        )],
        gravity: [f32; 3],
        colliders_per_batch: u32,
    ) -> Self {
        let num_batches = environments.len() as u32;

        // Stage 1: per-batch counts.
        let mut per_env_infos: Vec<Vec<MultibodyInfo>> = Vec::with_capacity(num_batches as usize);
        let mut per_env_links_static: Vec<Vec<MultibodyLinkStatic>> =
            Vec::with_capacity(num_batches as usize);
        let mut per_env_links_workspace: Vec<Vec<MultibodyLinkWorkspace>> =
            Vec::with_capacity(num_batches as usize);
        let mut per_env_links_mprops: Vec<Vec<LocalMassProperties>> =
            Vec::with_capacity(num_batches as usize);
        let mut per_env_dof_values: Vec<Vec<f32>> = Vec::with_capacity(num_batches as usize);
        let mut per_env_dof_vels: Vec<Vec<f32>> = Vec::with_capacity(num_batches as usize);
        let mut per_env_dof_damping: Vec<Vec<f32>> = Vec::with_capacity(num_batches as usize);

        let mut global_max_mb = 0u32;
        let mut global_max_links = 0u32;
        let mut global_max_dofs = 0u32;
        let mut global_max_jac = 0u32;
        let mut global_max_mm = 0u32;
        let mut global_max_cor = 0u32;
        let mut global_max_icdt = 0u32;
        let mut global_max_cons = 0u32;

        for (set, body_ids, bodies) in environments {
            let mut infos = Vec::new();
            let mut statics = Vec::new();
            let mut workspaces = Vec::new();
            let mut mprops = Vec::new();
            let mut dof_vals = Vec::new();
            let mut dof_vels = Vec::new();
            let mut dof_damping = Vec::new();

            let mut first_link = 0u32;
            let mut first_dof = 0u32;
            let mut jac_off = 0u32;
            let mut mm_off = 0u32;
            let mut cor_off = 0u32;
            let mut icdt_off = 0u32;
            let mut cons_off = 0u32;

            for (mb_idx, mb) in set.multibodies().enumerate() {
                if mb.ndofs() > MAX_MB_DOFS {
                    panic!("Multibody {} dofs {} exceed the maximum supported {}.", mb_idx, mb.ndofs(), MAX_MB_DOFS);
                }

                // rapier always creates the root with a free 6-DOF joint and only
                // converts it to a fixed joint later during its own step. Since we
                // don't run rapier's step here, detect a fixed root body and lock
                // all 6 DOFs ourselves.
                let root_is_dynamic = mb
                    .link(0)
                    .and_then(|r| bodies.get(r.rigid_body_handle()))
                    .map(|rb| rb.is_dynamic())
                    .unwrap_or(false);

                let root_ndof_adjust = if !root_is_dynamic && mb.link(0).is_some() {
                    mb.link(0).unwrap().joint().ndofs() as u32
                } else {
                    0
                };
                let ndofs = mb.ndofs() as u32 - root_ndof_adjust;
                let num_links = mb.num_links() as u32;

                // Count maximum constraint slots this multibody could need: for
                // each non-root non-kinematic joint, every free axis with a limit
                // OR a motor enabled produces one constraint slot, plus an
                // additional one if BOTH limit and motor are enabled on the same
                // axis (rapier emits them as separate constraints).
                let max_constraints = mb
                    .links()
                    .enumerate()
                    .map(|(li, link)| {
                        if link.joint().kinematic {
                            return 0u32;
                        }
                        if li == 0 && !root_is_dynamic {
                            return 0u32;
                        }
                        let j = link.joint().data;
                        let locked = j.locked_axes.bits() as u32;
                        let limit_axes = j.limit_axes.bits() as u32 & !locked;
                        let motor_axes = j.motor_axes.bits() as u32 & !locked;
                        // 1 per active limit + 1 per active motor (axis-wise).
                        let mut n = 0u32;
                        for ax in 0u32..6 {
                            if (limit_axes >> ax) & 1 != 0 {
                                n += 1;
                            }
                            if (motor_axes >> ax) & 1 != 0 {
                                n += 1;
                            }
                        }
                        n
                    })
                    .sum::<u32>();

                infos.push(MultibodyInfo {
                    first_link,
                    num_links,
                    first_dof,
                    ndofs,
                    jacobian_offset: jac_off,
                    mass_matrix_offset: mm_off,
                    root_is_dynamic: if root_is_dynamic { 1 } else { 0 },
                    coriolis_offset: cor_off,
                    i_coriolis_dt_offset: icdt_off,
                    first_constraint: cons_off,
                    max_constraints,
                });

                // `assembly_id` is not exposed publicly on `MultibodyLink`, so we
                // reconstruct it ourselves — rapier assigns ids in the same traversal
                // order as `links()`.
                let mut assembly_counter = 0u32;
                for (link_idx, link) in mb.links().enumerate() {
                    let rb_id = body_ids.get(&link.rigid_body_handle()).copied().unwrap_or(0);
                    let parent_id = match link.parent_id() {
                        Some(p) => p as u32,
                        None => u32::MAX,
                    };

                    // Lock all 6 DOFs on the root if its body is fixed.
                    let mut data = convert_generic_joint(link.joint().data);
                    let link_ndofs = if link_idx == 0 && !root_is_dynamic {
                        data.locked_axes = 0x3f;
                        0u32
                    } else {
                        link.joint().ndofs() as u32
                    };

                    let stat = MultibodyLinkStatic {
                        rb_id,
                        parent_link_id: parent_id,
                        multibody_id: mb_idx as u32,
                        assembly_id: assembly_counter,
                        ndofs: link_ndofs,
                        kinematic: if link.joint().kinematic { 1 } else { 0 },
                        _pad0: [0; 2],
                        data,
                    };
                    statics.push(stat);
                    assembly_counter += link_ndofs;

                    let mut ws = make_workspace_init();
                    ws.coords = link.joint.coords().into();
                    ws.joint_rot = link.joint.joint_rot();

                    // For free joints at the root, copy the rigid-body pose directly.
                    if link.joint.data.locked_axes.is_empty() {
                        if let Some(rb) = bodies.get(link.rigid_body_handle()) {
                            let pos = rb.position();
                            ws.coords[0] = pos.translation.x;
                            ws.coords[1] = pos.translation.y;
                            ws.coords[2] = pos.translation.z;
                            ws.joint_rot = pos.rotation;
                        }
                    }

                    workspaces.push(ws);

                    // Per-link mass properties (real masses stored here so the
                    // multibody solver sees correct values even when the shared
                    // body mprops are zeroed out).
                    let mp = bodies
                        .get(link.rigid_body_handle())
                        .map(|rb| convert_link_mprops(&rb.mass_properties().local_mprops))
                        .unwrap_or_default();
                    // For fixed-root links, mass/inertia are zeroed so they don't
                    // contribute to the CRBA mass matrix (rapier skips them too).
                    let mp = if link_idx == 0 && !root_is_dynamic {
                        let mut z = mp;
                        z.inv_mass = glamx::Vec3::ZERO;
                        z.inv_principal_inertia = glamx::Vec3::ZERO;
                        z
                    } else {
                        mp
                    };
                    mprops.push(mp);

                    // Seed per-DOF damping slots for this link (matches rapier's
                    // `MultibodyJoint::default_damping`: 0.1 on every free angular
                    // DOF, 0 on linear ones).
                    let link_damping = joint_default_damping(data.locked_axes);
                    for d in 0..link_ndofs as usize {
                        dof_vals.push(0.0);
                        dof_vels.push(0.0);
                        dof_damping.push(link_damping[d]);
                    }
                }

                first_link += num_links;
                first_dof += ndofs;
                jac_off += num_links * 6 * ndofs;
                mm_off += ndofs * ndofs;
                cor_off += num_links * 3 * ndofs;
                icdt_off += 6 * ndofs;
                cons_off += max_constraints;
            }

            global_max_mb = global_max_mb.max(infos.len() as u32);
            global_max_links = global_max_links.max(statics.len() as u32);
            global_max_dofs = global_max_dofs.max(dof_vals.len() as u32);
            global_max_jac = global_max_jac.max(jac_off);
            global_max_mm = global_max_mm.max(mm_off);
            global_max_cor = global_max_cor.max(cor_off);
            global_max_icdt = global_max_icdt.max(icdt_off);
            global_max_cons = global_max_cons.max(cons_off);

            per_env_infos.push(infos);
            per_env_links_static.push(statics);
            per_env_links_workspace.push(workspaces);
            per_env_links_mprops.push(mprops);
            per_env_dof_values.push(dof_vals);
            per_env_dof_vels.push(dof_vels);
            per_env_dof_damping.push(dof_damping);
        }

        // Pad capacities (avoid empty buffers — GPU dislikes size-zero storage bindings).
        let mb_cap = global_max_mb.max(1);
        let links_cap = global_max_links.max(1);
        let dofs_cap = global_max_dofs.max(1);
        let jac_cap = global_max_jac.max(1);
        let mm_cap = global_max_mm.max(1);
        let cor_cap = global_max_cor.max(1);
        let icdt_cap = global_max_icdt.max(1);
        let cons_cap = global_max_cons.max(1);
        // One length-`dofs_cap` column of `M⁻¹` per constraint slot.
        let cons_col_cap = cons_cap.saturating_mul(dofs_cap).max(1);

        // Per-multibody contact-constraint banks: every multibody owns a
        // fixed-size slab of `MAX_MB_CONTACT_CONSTRAINTS_PER_MB` slots —
        // each contact point produces 1 normal + (DIM-1) friction tangent
        // constraint slots. The init kernel marks unused slots as `kind = 0`.
        let contact_cons_cap = mb_cap
            .saturating_mul(MAX_MB_CONTACT_CONSTRAINTS_PER_MB)
            .max(1);
        let contact_cons_col_cap = contact_cons_cap.saturating_mul(dofs_cap).max(1);
        let body_to_link_cap = colliders_per_batch.max(1);

        // Build the per-body multibody/link lookup. Free / non-multibody bodies
        // get the sentinel `[u32::MAX, u32::MAX]`. The kernel reads
        // `body_to_link[batch_offset + body_local_id]` and skips the
        // sentinel.
        let mut all_body_to_link: Vec<[u32; 2]> =
            vec![[u32::MAX, u32::MAX]; (body_to_link_cap * num_batches) as usize];
        for (batch_idx, (set, body_ids, _)) in environments.iter().enumerate() {
            let base = batch_idx * body_to_link_cap as usize;
            for (mb_idx, mb) in set.multibodies().enumerate() {
                for (link_idx, link) in mb.links().enumerate() {
                    if let Some(&local) = body_ids.get(&link.rigid_body_handle()) {
                        if (local as u32) < body_to_link_cap {
                            all_body_to_link[base + local as usize] =
                                [mb_idx as u32, link_idx as u32];
                        }
                    }
                }
            }
        }

        // Flatten, padding each batch to `*_cap`.
        let mut all_infos: Vec<MultibodyInfo> =
            Vec::with_capacity((mb_cap * num_batches) as usize);
        let mut all_statics: Vec<MultibodyLinkStatic> =
            Vec::with_capacity((links_cap * num_batches) as usize);
        let mut all_ws: Vec<MultibodyLinkWorkspace> =
            Vec::with_capacity((links_cap * num_batches) as usize);
        let mut all_mprops: Vec<LocalMassProperties> =
            Vec::with_capacity((links_cap * num_batches) as usize);
        let mut all_dof_vals: Vec<f32> = Vec::with_capacity((dofs_cap * num_batches) as usize);
        let mut all_dof_vels: Vec<f32> = Vec::with_capacity((dofs_cap * num_batches) as usize);
        let mut all_dof_damping: Vec<f32> = Vec::with_capacity((dofs_cap * num_batches) as usize);
        let mut all_num_mb: Vec<u32> = Vec::with_capacity(num_batches as usize);

        let dummy_info = MultibodyInfo::default();
        let dummy_stat: MultibodyLinkStatic = bytemuck::Zeroable::zeroed();
        let dummy_ws = make_workspace_init();

        for i in 0..num_batches as usize {
            all_num_mb.push(per_env_infos[i].len() as u32);

            all_infos.extend_from_slice(&per_env_infos[i]);
            for _ in per_env_infos[i].len()..mb_cap as usize {
                all_infos.push(dummy_info);
            }

            all_statics.extend_from_slice(&per_env_links_static[i]);
            for _ in per_env_links_static[i].len()..links_cap as usize {
                all_statics.push(dummy_stat);
            }

            all_ws.extend_from_slice(&per_env_links_workspace[i]);
            for _ in per_env_links_workspace[i].len()..links_cap as usize {
                all_ws.push(dummy_ws);
            }

            all_mprops.extend_from_slice(&per_env_links_mprops[i]);
            for _ in per_env_links_mprops[i].len()..links_cap as usize {
                all_mprops.push(LocalMassProperties::default());
            }

            all_dof_vals.extend_from_slice(&per_env_dof_values[i]);
            for _ in per_env_dof_values[i].len()..dofs_cap as usize {
                all_dof_vals.push(0.0);
            }
            all_dof_vels.extend_from_slice(&per_env_dof_vels[i]);
            for _ in per_env_dof_vels[i].len()..dofs_cap as usize {
                all_dof_vels.push(0.0);
            }
            all_dof_damping.extend_from_slice(&per_env_dof_damping[i]);
            for _ in per_env_dof_damping[i].len()..dofs_cap as usize {
                all_dof_damping.push(0.0);
            }
        }

        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let usage_u = storage | BufferUsages::UNIFORM;

        Self {
            num_batches,
            multibodies_per_batch: mb_cap,
            links_per_batch: links_cap,
            dofs_per_batch: dofs_cap,
            jacobian_entries_per_batch: jac_cap,
            mass_matrix_entries_per_batch: mm_cap,
            coriolis_entries_per_batch: cor_cap,
            i_coriolis_dt_entries_per_batch: icdt_cap,
            implicit_coriolis: false,
            has_joint_constraints: all_infos.iter().any(|info| info.max_constraints > 0),

            num_multibodies: Tensor::vector(backend, &all_num_mb, usage_u).unwrap(),
            multibody_info: Tensor::vector(backend, &all_infos, storage).unwrap(),
            links_static: Tensor::vector(
                backend,
                &all_statics,
                storage | BufferUsages::COPY_DST,
            )
            .unwrap(),
            links_static_mirror: all_statics.clone(),
            links_workspace: Tensor::vector(backend, &all_ws, storage).unwrap(),
            links_mprops: Tensor::vector(backend, &all_mprops, storage).unwrap(),
            dof_values: Tensor::vector(backend, &all_dof_vals, storage).unwrap(),
            dof_velocities: Tensor::vector(backend, &all_dof_vels, storage).unwrap(),
            damping: Tensor::vector(backend, &all_dof_damping, storage).unwrap(),
            gen_forces: Tensor::vector(
                backend,
                &vec![0.0f32; (dofs_cap * num_batches) as usize],
                storage | BufferUsages::COPY_SRC,
            )
            .unwrap(),
            body_jacobians: Tensor::vector(
                backend,
                &vec![0.0f32; (jac_cap * num_batches) as usize],
                storage,
            )
            .unwrap(),
            mass_matrices: Tensor::vector(
                backend,
                &vec![0.0f32; (mm_cap * num_batches) as usize],
                storage,
            )
            .unwrap(),
            lu_pivots: Tensor::vector(
                backend,
                &vec![0u32; (dofs_cap * num_batches) as usize],
                storage,
            )
            .unwrap(),
            coriolis_v: Tensor::vector(
                backend,
                &vec![0.0f32; (cor_cap * num_batches) as usize],
                storage,
            )
            .unwrap(),
            coriolis_w: Tensor::vector(
                backend,
                &vec![0.0f32; (cor_cap * num_batches) as usize],
                storage,
            )
            .unwrap(),
            i_coriolis_dt: Tensor::vector(
                backend,
                &vec![0.0f32; (icdt_cap * num_batches) as usize],
                storage,
            )
            .unwrap(),
            joint_constraints: Tensor::vector(
                backend,
                &vec![MultibodyJointConstraint::default(); (cons_cap * num_batches) as usize],
                storage,
            )
            .unwrap(),
            joint_constraint_columns: Tensor::vector(
                backend,
                &vec![0.0f32; (cons_col_cap * num_batches) as usize],
                storage,
            )
            .unwrap(),
            body_to_link: Tensor::vector(backend, &all_body_to_link, storage).unwrap(),
            contact_constraints: Tensor::vector(
                backend,
                &vec![
                    MultibodyContactConstraint::default();
                    (contact_cons_cap * num_batches) as usize
                ],
                storage,
            )
            .unwrap(),
            contact_constraint_jacs: Tensor::vector(
                backend,
                &vec![0.0f32; (contact_cons_col_cap * num_batches) as usize],
                storage,
            )
            .unwrap(),
            contact_constraint_columns: Tensor::vector(
                backend,
                &vec![0.0f32; (contact_cons_col_cap * num_batches) as usize],
                storage,
            )
            .unwrap(),
            contact_constraint_count: Tensor::vector(
                backend,
                &vec![0u32; (mb_cap * num_batches) as usize],
                storage,
            )
            .unwrap(),

            // Impulse-joint buffers are sized for "no MB-touching joints" by
            // default — `set_impulse_joints` resizes them at pipeline build
            // time when the host has actually counted the joints.
            mb_imp_joint_count: Tensor::vector(
                backend,
                &vec![0u32; num_batches as usize],
                storage | BufferUsages::UNIFORM,
            )
            .unwrap(),
            mb_imp_joint_builders: Tensor::vector(
                backend,
                &vec![<MbImpulseJointBuilder as bytemuck::Zeroable>::zeroed(); num_batches as usize],
                storage,
            )
            .unwrap(),
            mb_imp_joint_constraints: Tensor::vector(
                backend,
                &vec![
                    MbImpulseJointConstraint::default();
                    (MAX_AXIS_CONSTRAINTS as usize) * (num_batches as usize)
                ],
                storage,
            )
            .unwrap(),
            mb_imp_joint_jacobians: Tensor::vector(
                backend,
                &vec![0.0f32; num_batches as usize],
                storage,
            )
            .unwrap(),
            mb_imp_joints_batch_capacity: Tensor::scalar(backend, 1u32, usage_u).unwrap(),
            mb_imp_joint_constraints_batch_capacity: Tensor::scalar(
                backend,
                MAX_AXIS_CONSTRAINTS,
                usage_u,
            )
            .unwrap(),
            mb_imp_joint_jacobians_batch_capacity: Tensor::scalar(backend, 1u32, usage_u).unwrap(),
            mb_imp_joints_per_batch: 0,

            num_solver_iterations: 4,

            multibodies_batch_capacity: Tensor::scalar(backend, mb_cap, usage_u).unwrap(),
            links_batch_capacity: Tensor::scalar(backend, links_cap, usage_u).unwrap(),
            dof_batch_capacity: Tensor::scalar(backend, dofs_cap, usage_u).unwrap(),
            jacobians_batch_capacity: Tensor::scalar(backend, jac_cap, usage_u).unwrap(),
            mass_matrix_batch_capacity: Tensor::scalar(backend, mm_cap, usage_u).unwrap(),
            coriolis_batch_capacity: Tensor::scalar(backend, cor_cap, usage_u).unwrap(),
            i_coriolis_dt_batch_capacity: Tensor::scalar(backend, icdt_cap, usage_u).unwrap(),
            joint_constraints_batch_capacity: Tensor::scalar(backend, cons_cap, usage_u).unwrap(),
            joint_constraint_columns_batch_capacity: Tensor::scalar(backend, cons_col_cap, usage_u).unwrap(),
            contact_constraints_batch_capacity: Tensor::scalar(backend, contact_cons_cap, usage_u)
                .unwrap(),
            contact_constraint_columns_batch_capacity: Tensor::scalar(
                backend,
                contact_cons_col_cap,
                usage_u,
            )
            .unwrap(),
            contacts_batch_capacity_for_mb: Tensor::scalar(backend, body_to_link_cap, usage_u)
                .unwrap(),

            // FIXME: should be read from the simulation settings.
            gravity: Tensor::scalar(
                backend,
                gravity,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            )
            .unwrap(),
            dt: Tensor::scalar(
                backend,
                1.0f32 / 60.0,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            )
            .unwrap(),
        }
    }

    /// Number of multibody-touching impulse joints in any batch.
    pub fn mb_impulse_joints_per_batch(&self) -> u32 {
        self.mb_imp_joints_per_batch
    }

    /// Upload the per-batch impulse joints whose body1 OR body2 is part
    /// of a multibody. These joints are routed through the
    /// `MbImpulseJointConstraint` solver path (rapier's
    /// `JointGenericExternalConstraintBuilder`); free-only impulse joints
    /// stay in the regular `GpuImpulseJointSet` path because they don't
    /// need `M⁻¹·Jᵀ` propagation.
    ///
    /// `environments` matches the layout used elsewhere in the pipeline:
    /// one entry per batch, in the same order as the multibody envs that
    /// were passed to `from_rapier`. Free-only joints are silently
    /// skipped here.
    #[cfg(feature = "from_rapier")]
    pub fn set_impulse_joints(
        &mut self,
        backend: &GpuBackend,
        environments: &[(
            &ImpulseJointSet,
            &MultibodyJointSet,
            &HashMap<RigidBodyHandle, u32>,
            &RigidBodySet,
        )],
    ) {
        assert_eq!(environments.len() as u32, self.num_batches);

        // Stage 1 — per-batch list of touched joints + their side metadata.
        let mut per_env_builders: Vec<Vec<MbImpulseJointBuilder>> =
            Vec::with_capacity(self.num_batches as usize);
        let mut max_joints = 0u32;
        let mut max_jac_floats = 0u32;

        for (batch_idx, (impulse_joints, mb_set, body_ids, bodies)) in
            environments.iter().enumerate()
        {
            let _ = batch_idx;
            // body local id → (mb_index_in_batch, link_index_within_mb).
            let mut body_to_mb_link: HashMap<u32, (u32, u32)> = HashMap::new();
            for (mb_idx, mb) in mb_set.multibodies().enumerate() {
                for (link_idx, link) in mb.links().enumerate() {
                    if let Some(&local) = body_ids.get(&link.rigid_body_handle()) {
                        body_to_mb_link.insert(local, (mb_idx as u32, link_idx as u32));
                    }
                }
            }

            let mut builders = Vec::new();
            let mut jac_offset = 0u32;
            let mut constraint_id = 0u32;

            for (_handle, joint) in impulse_joints.iter() {
                let body1 = joint.body1();
                let body2 = joint.body2();
                let local1 = match body_ids.get(&body1) {
                    Some(&id) => id,
                    None => continue,
                };
                let local2 = match body_ids.get(&body2) {
                    Some(&id) => id,
                    None => continue,
                };

                let mb1 = body_to_mb_link.get(&local1).copied();
                let mb2 = body_to_mb_link.get(&local2).copied();
                if mb1.is_none() && mb2.is_none() {
                    continue; // Free-only joint; existing path handles it.
                }

                let rb1 = bodies.get(body1);
                let rb2 = bodies.get(body2);

                // Mirror rapier's `LinkOrBody` resolution + `transform_to_solver_body_space`.
                // Side A:
                let (side_a_kind, side_a_id, side_a_link, ndofs_a) = match (mb1, rb1) {
                    (Some((mb_idx, link_idx)), _) => {
                        let mb = mb_set.multibodies().nth(mb_idx as usize).unwrap();
                        (
                            SIDE_KIND_MB,
                            mb_idx,
                            link_idx,
                            mb.ndofs() as u32,
                        )
                    }
                    (None, Some(rb)) if rb.is_dynamic() => {
                        (SIDE_KIND_BODY, local1, 0, 6)
                    }
                    _ => (SIDE_KIND_FIXED, u32::MAX, 0, 0),
                };

                let (side_b_kind, side_b_id, side_b_link, ndofs_b) = match (mb2, rb2) {
                    (Some((mb_idx, link_idx)), _) => {
                        let mb = mb_set.multibodies().nth(mb_idx as usize).unwrap();
                        (
                            SIDE_KIND_MB,
                            mb_idx,
                            link_idx,
                            mb.ndofs() as u32,
                        )
                    }
                    (None, Some(rb)) if rb.is_dynamic() => {
                        (SIDE_KIND_BODY, local2, 0, 6)
                    }
                    _ => (SIDE_KIND_FIXED, u32::MAX, 0, 0),
                };

                if ndofs_a + ndofs_b == 0 {
                    continue; // Both sides static — no constraint to solve.
                }

                // Mirror rapier `GenericJoint::transform_to_solver_body_space`:
                // shift each anchor frame's translation into COM space (and,
                // if the side is fixed, fold the body's pose into the local
                // frame). For now we only handle the dynamic / multibody
                // cases — fixed side support is a TODO that mirrors
                // rapier's `is_fixed` branch.
                let mut joint_data = convert_generic_joint(joint.data);
                if side_a_kind != SIDE_KIND_FIXED {
                    if let Some(rb) = rb1 {
                        let com = rb.mass_properties().local_mprops.local_com;
                        joint_data.local_frame_a.translation -= com;
                    }
                }
                if side_b_kind != SIDE_KIND_FIXED {
                    if let Some(rb) = rb2 {
                        let com = rb.mass_properties().local_mprops.local_com;
                        joint_data.local_frame_b.translation -= com;
                    }
                }

                // Per-axis stride = 2 * (ndofs_a + ndofs_b); reserve
                // MAX_AXIS_CONSTRAINTS slots up front so the kernel can
                // walk them sequentially without rechecking.
                let stride = 2 * (ndofs_a + ndofs_b);
                let cap_floats = stride * MAX_AXIS_CONSTRAINTS;
                let builder = MbImpulseJointBuilder {
                    joint: joint_data,
                    side_a_kind,
                    side_a_id,
                    side_a_link,
                    joint_id: builders.len() as u32,
                    side_b_kind,
                    side_b_id,
                    side_b_link,
                    constraint_id: constraint_id,
                    jacobian_offset: jac_offset,
                    jacobian_capacity: cap_floats,
                    #[cfg(feature = "dim3")]
                    _pad0: [0; 2],
                };
                builders.push(builder);
                constraint_id += MAX_AXIS_CONSTRAINTS;
                jac_offset += cap_floats;
            }

            max_joints = max_joints.max(builders.len() as u32);
            max_jac_floats = max_jac_floats.max(jac_offset);
            per_env_builders.push(builders);
        }

        // Stage 2 — flatten with per-batch padding to `max_joints`.
        let joints_cap = max_joints.max(1);
        let cons_cap = (joints_cap * MAX_AXIS_CONSTRAINTS).max(1);
        let jac_cap = max_jac_floats.max(1);

        let mut all_builders: Vec<MbImpulseJointBuilder> =
            Vec::with_capacity((joints_cap * self.num_batches) as usize);
        let mut all_counts: Vec<u32> = Vec::with_capacity(self.num_batches as usize);
        let dummy: MbImpulseJointBuilder = bytemuck::Zeroable::zeroed();
        for env in &per_env_builders {
            all_counts.push(env.len() as u32);
            all_builders.extend_from_slice(env);
            for _ in env.len()..joints_cap as usize {
                all_builders.push(dummy);
            }
        }

        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let usage_u = storage | BufferUsages::UNIFORM;
        self.mb_imp_joint_count = Tensor::vector(backend, &all_counts, usage_u).unwrap();
        self.mb_imp_joint_builders = Tensor::vector(backend, &all_builders, storage).unwrap();
        self.mb_imp_joint_constraints = Tensor::vector(
            backend,
            &vec![MbImpulseJointConstraint::default(); (cons_cap * self.num_batches) as usize],
            storage,
        )
        .unwrap();
        self.mb_imp_joint_jacobians = Tensor::vector(
            backend,
            &vec![0.0f32; (jac_cap * self.num_batches) as usize],
            storage,
        )
        .unwrap();
        self.mb_imp_joints_batch_capacity = Tensor::scalar(backend, joints_cap, usage_u).unwrap();
        self.mb_imp_joint_constraints_batch_capacity =
            Tensor::scalar(backend, cons_cap, usage_u).unwrap();
        self.mb_imp_joint_jacobians_batch_capacity =
            Tensor::scalar(backend, jac_cap, usage_u).unwrap();
        self.mb_imp_joints_per_batch = joints_cap;
    }

    /// Upload a new integration timestep.
    pub fn set_dt(&mut self, backend: &GpuBackend, dt: f32) {
        self.dt = Tensor::scalar(
            backend,
            dt,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        )
        .unwrap();
    }
}

#[cfg(feature = "from_rapier")]
fn convert_link_mprops(m: &crate::rapier::prelude::MassProperties) -> LocalMassProperties {
    LocalMassProperties {
        inertia_ref_frame: m.principal_inertia_local_frame,
        inv_principal_inertia: m.inv_principal_inertia,
        padding0: 0,
        inv_mass: glamx::Vec3::splat(m.inv_mass),
        padding1: 0,
        com: m.local_com,
        padding2: 0,
    }
}

/// Per-DOF defaults matching rapier's `MultibodyJoint::default_damping`: 0.1 on
/// every free angular DOF, 0 elsewhere. The returned array is packed in
/// generalized-velocity order — free linear DOFs first (in axis order), then
/// free angular DOFs.
#[cfg(feature = "from_rapier")]
fn joint_default_damping(locked_axes: u32) -> [f32; 6] {
    let mut out = [0.0f32; 6];
    // Index of the first free angular DOF in the joint's generalized-velocity slice.
    let num_free_lin = 3 - (locked_axes & 0x7).count_ones();
    let mut curr = num_free_lin as usize;
    for i in 3u32..6 {
        if locked_axes & (1 << i) == 0 {
            out[curr] = 0.1;
            curr += 1;
        }
    }
    out
}

#[cfg(feature = "from_rapier")]
fn convert_generic_joint(j: crate::rapier::dynamics::GenericJoint) -> GenericJoint {
    GenericJoint {
        local_frame_a: j.local_frame1,
        local_frame_b: j.local_frame2,
        locked_axes: j.locked_axes.bits() as u32,
        limit_axes: j.limit_axes.bits() as u32,
        motor_axes: j.motor_axes.bits() as u32,
        coupled_axes: j.coupled_axes.bits() as u32,
        limits: j.limits.map(|l| JointLimits {
            min: l.min,
            max: l.max,
            impulse: l.impulse,
        }),
        motors: j.motors.map(|m| JointMotor {
            target_vel: m.target_vel,
            target_pos: m.target_pos,
            stiffness: m.stiffness,
            damping: m.damping,
            max_force: m.max_force,
            impulse: m.impulse,
            model: match m.model {
                crate::rapier::prelude::MotorModel::AccelerationBased => 0,
                crate::rapier::prelude::MotorModel::ForceBased => 1,
            },
        }),
    }
}

//
// Zero-initialised workspaces would leave `joint_rot` as the all-zero quaternion, which
// is not a valid rotation — seed it with the identity instead.
//
#[cfg(feature = "from_rapier")]
fn make_workspace_init() -> MultibodyLinkWorkspace {
    let mut w: MultibodyLinkWorkspace = bytemuck::Zeroable::zeroed();
    w.joint_rot = glamx::Quat::IDENTITY;
    w.local_to_parent = Pose::default();
    w.local_to_world = Pose::default();
    w
}

/// GPU shader bundle for multibody dynamics.
#[derive(Shader)]
pub struct GpuMultibodySolver {
    forward_kinematics: GpuMbForwardKinematics,
    body_jacobians: GpuMbBodyJacobians,
    update_velocities: GpuMbUpdateVelocities,
    mass_matrix: GpuMbMassMatrix,
    mass_matrix_with_coriolis: GpuMbMassMatrixWithCoriolis,
    apply_gravity_with_coriolis: GpuMbApplyGravityWithCoriolis,
    lu_decompose: GpuMbLuDecompose,
    lu_solve: GpuMbLuSolve,
    lu_factor_and_solve: GpuMbLuFactorAndSolve,
    gravity_and_lu: GpuMbGravityAndLu,
    compute_dynamics_pre: GpuMbComputeDynamicsPre,
    compute_dynamics_without_coriolis_pre: GpuMbComputeDynamicsWithoutCoriolisPre,
    init_joint_constraints: GpuMbInitJointConstraints,
    solve_joint_constraints: GpuMbSolveJointConstraints,
    remove_joint_constraint_bias: GpuMbRemoveJointConstraintBias,
    init_contact_constraints: GpuMbInitContactConstraints,
    finalize_contact_constraints: GpuMbFinalizeContactConstraints,
    solve_contact_constraints: GpuMbSolveContactConstraints,
    remove_contact_constraint_bias: GpuMbRemoveContactConstraintBias,
    update_impulse_joint_constraints: GpuMbUpdateImpulseJointConstraints,
    solve_impulse_joint_constraints: GpuMbSolveImpulseJointConstraints,
    remove_impulse_joint_constraint_bias: GpuMbRemoveImpulseJointConstraintBias,
    integrate_velocities: GpuMbIntegrateVelocities,
    integrate: GpuMbIntegrate,
}

/// Arguments for one multibody dispatch. The poses buffer is shared with the rest
/// of the rigid-body pipeline (FK writes link poses there); mass properties are
/// now owned by the multibody itself.
pub struct MultibodySolverArgs<'a> {
    /// Body poses (written by FK; consumed by every per-body computation —
    /// gravity, jacobians, mass matrix, integration). Inside the substep loop
    /// this points to `solver_body_poses` (rapier's COM-centered solver
    /// pose); during phase-0 init this points to `body_poses`. Multibody
    /// links carry zero local-COM in the shared mprops buffer so the two are
    /// equivalent for their slots.
    pub poses: &'a mut Tensor<Pose>,
    /// Per-collider world poses, used by `init_contact_constraints` to
    /// recover world-space contact normals and points from manifold features
    /// expressed in collider-local space.
    pub collider_world_poses: &'a Tensor<Pose>,
    /// Colliders-per-batch capacity (stride in the pose tensor).
    pub colliders_batch_capacity: &'a Tensor<u32>,
    /// Free-body world mass properties (read by `init_contact_constraints`).
    pub mprops: &'a Tensor<WorldMassProperties>,
    /// Per-batch contact manifold list (filled by narrow-phase).
    pub contacts: &'a Tensor<GpuIndexedContact>,
    /// Per-batch contact count (parallel to `contacts`).
    pub contacts_len: &'a Tensor<u32>,
    /// Per-batch contact-buffer stride.
    pub contacts_batch_capacity: &'a Tensor<u32>,
    /// Free-body solver velocities (updated in place by `solve_contact_constraints`).
    pub solver_vels: &'a mut Tensor<Velocity>,
}

impl GpuMultibodySolver {
    /// Runs FK → jacobians → mass matrix → gravity → LU solve in sequence on one pass.
    ///
    /// After completion, `mb.gen_accelerations()` holds `ẍ = M⁻¹ τ_g` (one per DOF).
    pub fn solve_gravity(
        &self,
        pass: &mut GpuPass,
        mb: &mut GpuMultibodySet,
        args: MultibodySolverArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        if mb.is_empty() {
            return Ok(());
        }
        let dispatch = [mb.multibodies_per_batch, mb.num_batches, 1];

        // Fused FK + body-jacobians + velocity propagation + CRBA-with-Coriolis
        // mass-matrix assembly (4 dispatches → 1) — see
        // `gpu_mb_compute_dynamics_pre`. Only the implicit-Coriolis path is
        // wired through the fused kernel; the explicit-Coriolis fallback keeps
        // the legacy split path.
        let pre_dispatch = [mb.multibodies_per_batch * MB_LU_LANES, mb.num_batches, 1];
        if mb.implicit_coriolis {
            self.compute_dynamics_pre.call(
                pass,
                pre_dispatch,
                &mb.multibody_info,
                &mb.links_static,
                &mut mb.links_workspace,
                &mb.links_mprops,
                args.poses,
                &mut mb.body_jacobians,
                &mut mb.mass_matrices,
                &mut mb.coriolis_v,
                &mut mb.coriolis_w,
                &mut mb.i_coriolis_dt,
                &mb.dof_velocities,
                &mb.damping,
                &mb.num_multibodies,
                &mb.dt,
                &mb.multibodies_batch_capacity,
                &mb.links_batch_capacity,
                &mb.jacobians_batch_capacity,
                &mb.mass_matrix_batch_capacity,
                &mb.coriolis_batch_capacity,
                &mb.i_coriolis_dt_batch_capacity,
                &mb.dof_batch_capacity,
                args.colliders_batch_capacity,
            )?;
        } else {
            self.compute_dynamics_without_coriolis_pre.call(
                pass,
                pre_dispatch,
                &mb.multibody_info,
                &mb.links_static,
                &mut mb.links_workspace,
                &mb.links_mprops,
                args.poses,
                &mut mb.body_jacobians,
                &mut mb.mass_matrices,
                &mb.dof_velocities,
                &mb.damping,
                &mb.num_multibodies,
                &mb.dt,
                &mb.multibodies_batch_capacity,
                &mb.links_batch_capacity,
                &mb.jacobians_batch_capacity,
                &mb.mass_matrix_batch_capacity,
                &mb.dof_batch_capacity,
                args.colliders_batch_capacity,
            )?;
        }

        // Fused: gravity / Coriolis force assembly + LU factor + LU solve in
        // a single dispatch. Replaces the previous 2-dispatch chain
        // (apply_gravity_with_coriolis → lu_factor_and_solve) — drops one
        // WebGPU dispatch per `compute_dynamics` call.
        let grav_lu_dispatch = [mb.multibodies_per_batch * MB_LU_LANES, mb.num_batches, 1];
        self.gravity_and_lu.call(
            pass,
            grav_lu_dispatch,
            &mb.multibody_info,
            &mb.links_static,
            &mut mb.links_workspace,
            &mb.links_mprops,
            &mb.body_jacobians,
            &mut mb.gen_forces,
            &mut mb.mass_matrices,
            &mut mb.lu_pivots,
            &mb.dof_velocities,
            &mb.damping,
            &mb.num_multibodies,
            &mb.gravity,
            &mb.multibodies_batch_capacity,
            &mb.links_batch_capacity,
            &mb.jacobians_batch_capacity,
            &mb.mass_matrix_batch_capacity,
            &mb.dof_batch_capacity,
        )?;

        Ok(())
    }

    /// Advance the multibody state by one timestep, mirroring rapier's order:
    ///
    ///   FK → body_jacobians → update_velocities → mass_matrix → apply_gravity_with_coriolis
    ///   → lu_decompose → lu_solve  (=> generalized acceleration `a`)
    ///   → integrate_velocities  (v += a · dt)
    ///   → init_joint_constraints  (build M⁻¹ columns and biases)
    ///   → N × solve_joint_constraints  (PGS sweeps over limits / motors)
    ///   → integrate  (coords / joint_rot += v · dt with the corrected `v`)
    /// Once-per-visible-step setup: FK → body jacobians → velocity propagation →
    /// mass matrix (with damping diagonal) → generalized gravity (Coriolis-aware) →
    /// LU decompose → LU solve. After this call, `gen_forces` holds the
    /// generalized acceleration `a = M⁻¹ τ` and `mass_matrices` holds the LU
    /// factors. The caller then runs `apply_substep` once per substep, with the
    /// last call carrying `is_last_substep = true`.
    ///
    /// Mirrors rapier's `init_solver_velocities_and_solver_bodies` →
    /// `multibody.update_dynamics + update_acceleration` block.
    pub fn init_step(
        &self,
        pass: &mut GpuPass,
        mb: &mut GpuMultibodySet,
        args: &mut MultibodySolverArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        if mb.is_empty() {
            return Ok(());
        }
        self.compute_dynamics(pass, mb, args)
    }

    /// Per-substep work — interleaved with the rigid-body substep by the pipeline.
    ///
    /// Mirrors one iteration of rapier's `velocity_solver::solve_constraints`
    /// inner loop for the multibody side:
    ///
    ///   1. `dof_velocities += a · dt'`          (apply velocity increment)
    ///   2. `init_joint_constraints`             (rebuild biases + M⁻¹ columns)
    ///   3. `solve_joint_constraints` (with bias)
    ///   4. `integrate`                          (coords / joint_rot += v · dt')
    ///   5. **if not last substep**: rebuild dynamics (FK → jacobians → vel →
    ///      M → gravity → LU → solve) so the next substep has a fresh `a`.
    ///   6. `remove_joint_constraint_bias`       (rapier's `remove_bias_from_rhs`)
    ///   7. `solve_joint_constraints` (without bias)  — stabilization
    pub fn apply_substep(
        &self,
        pass: &mut GpuPass,
        mb: &mut GpuMultibodySet,
        args: &mut MultibodySolverArgs<'_>,
        is_last_substep: bool,
    ) -> Result<(), GpuBackendError> {
        if mb.is_empty() {
            return Ok(());
        }
        let dispatch = [mb.multibodies_per_batch, mb.num_batches, 1];

        // 1. v += a · dt_substep.
        self.integrate_velocities.call(
            pass,
            dispatch,
            &mb.multibody_info,
            &mut mb.dof_velocities,
            &mb.gen_forces,
            &mb.num_multibodies,
            &mb.dt,
            &mb.multibodies_batch_capacity,
            &mb.dof_batch_capacity,
        )?;

        // 2. Build limit / motor constraints (uses the cached LU + current coords).
        // 3. PGS sweep WITH bias (positional correction).
        // Both are no-ops when no multibody declared limit/motor axes — skip
        // the dispatches entirely (they count against WebGPU dispatch overhead).
        if mb.has_joint_constraints {
            self.init_joint_constraints.call(
                pass,
                dispatch,
                &mb.multibody_info,
                &mb.links_static,
                &mb.links_workspace,
                &mb.mass_matrices,
                &mb.lu_pivots,
                &mut mb.joint_constraints,
                &mut mb.joint_constraint_columns,
                &mb.num_multibodies,
                &mb.dt,
                &mb.multibodies_batch_capacity,
                &mb.links_batch_capacity,
                &mb.mass_matrix_batch_capacity,
                &mb.dof_batch_capacity,
                &mb.joint_constraints_batch_capacity,
                &mb.joint_constraint_columns_batch_capacity,
            )?;

            self.solve_joint_constraints.call(
                pass,
                dispatch,
                &mb.multibody_info,
                &mut mb.joint_constraints,
                &mb.joint_constraint_columns,
                &mut mb.dof_velocities,
                &mb.num_multibodies,
                &mb.multibodies_batch_capacity,
                &mb.dof_batch_capacity,
                &mb.joint_constraints_batch_capacity,
                &mb.joint_constraint_columns_batch_capacity,
            )?;
        }

        // 3b. Build + finalize + solve contact constraints (normal-only, free
        //     body × multibody pairs only). Mirrors rapier's interleaved
        //     "generic constraint" sweep order.
        self.init_contact_constraints.call(
            pass,
            dispatch,
            // descriptor_set 0 (sorted by binding index).
            &mb.multibody_info,
            &mb.links_static,
            &mb.links_workspace,
            &mb.body_jacobians,
            &mb.body_to_link,
            &mut mb.contact_constraints,
            &mut mb.contact_constraint_jacs,
            &mut mb.contact_constraint_count,
            &mb.num_multibodies,
            &mb.dt,
            &mb.multibodies_batch_capacity,
            &mb.links_batch_capacity,
            &mb.jacobians_batch_capacity,
            &mb.dof_batch_capacity,
            &mb.contact_constraints_batch_capacity,
            &mb.contact_constraint_columns_batch_capacity,
            // descriptor_set 1.
            args.mprops,
            args.collider_world_poses,
            args.contacts,
            args.contacts_len,
            args.contacts_batch_capacity,
            args.colliders_batch_capacity,
            &mb.contacts_batch_capacity_for_mb,
        )?;

        self.finalize_contact_constraints.call(
            pass,
            dispatch,
            &mb.multibody_info,
            &mb.mass_matrices,
            &mb.lu_pivots,
            &mut mb.contact_constraints,
            &mb.contact_constraint_jacs,
            &mut mb.contact_constraint_columns,
            &mb.contact_constraint_count,
            &mb.num_multibodies,
            &mb.multibodies_batch_capacity,
            &mb.mass_matrix_batch_capacity,
            &mb.dof_batch_capacity,
            &mb.contact_constraints_batch_capacity,
            &mb.contact_constraint_columns_batch_capacity,
        )?;

        self.solve_contact_constraints.call(
            pass,
            dispatch,
            &mb.multibody_info,
            &mut mb.contact_constraints,
            &mb.contact_constraint_jacs,
            &mb.contact_constraint_columns,
            &mb.contact_constraint_count,
            &mut mb.dof_velocities,
            args.solver_vels,
            &mb.num_multibodies,
            &mb.multibodies_batch_capacity,
            &mb.dof_batch_capacity,
            &mb.contact_constraints_batch_capacity,
            &mb.contact_constraint_columns_batch_capacity,
            args.colliders_batch_capacity,
        )?;

        // 3c. Multibody-touching impulse joints — generic (rb-mb / mb-mb)
        //     constraints. Mirrors rapier's `JointGenericExternalConstraintBuilder::update`
        //     plus a PGS sweep WITH bias.
        if mb.mb_imp_joints_per_batch > 0 {
            let imp_dispatch = [mb.mb_imp_joints_per_batch, mb.num_batches, 1];
            self.update_impulse_joint_constraints.call(
                pass,
                imp_dispatch,
                // set 0 storage (binding 0..=4)
                &mb.mb_imp_joint_builders,
                &mut mb.mb_imp_joint_constraints,
                &mut mb.mb_imp_joint_jacobians,
                &mb.mb_imp_joint_count,
                &mb.dt,
                // set 0 uniforms (binding 5..=7)
                &mb.mb_imp_joints_batch_capacity,
                &mb.mb_imp_joint_constraints_batch_capacity,
                &mb.mb_imp_joint_jacobians_batch_capacity,
                // set 1 storage (binding 0..=6)
                &mb.multibody_info,
                &mb.links_workspace,
                &mb.body_jacobians,
                &mb.mass_matrices,
                &mb.lu_pivots,
                args.poses,
                args.mprops,
                // set 1 uniforms (binding 7..=12)
                &mb.multibodies_batch_capacity,
                &mb.links_batch_capacity,
                &mb.jacobians_batch_capacity,
                &mb.mass_matrix_batch_capacity,
                &mb.dof_batch_capacity,
                args.colliders_batch_capacity,
            )?;
            // Solve sweep is single-workgroup, threads(1) — see kernel doc.
            self.solve_impulse_joint_constraints.call(
                pass,
                [1u32, mb.num_batches, 1],
                // set 0 storage (binding 0..=3)
                &mb.mb_imp_joint_builders,
                &mut mb.mb_imp_joint_constraints,
                &mb.mb_imp_joint_jacobians,
                &mb.mb_imp_joint_count,
                // set 0 uniforms (binding 4..=5)
                &mb.mb_imp_joints_batch_capacity,
                &mb.mb_imp_joint_constraints_batch_capacity,
                // set 1 storage (binding 0..=2)
                &mb.multibody_info,
                &mut mb.dof_velocities,
                args.solver_vels,
                // set 1 uniforms (binding 3..=5)
                &mb.multibodies_batch_capacity,
                &mb.dof_batch_capacity,
                args.colliders_batch_capacity,
            )?;
        }

        // 4. Integrate positions with the corrected `v`.
        self.integrate.call(
            pass,
            dispatch,
            &mb.multibody_info,
            &mb.links_static,
            &mut mb.links_workspace,
            &mut mb.dof_values,
            &mb.dof_velocities,
            &mb.num_multibodies,
            &mb.dt,
            &mb.multibodies_batch_capacity,
            &mb.links_batch_capacity,
            &mb.dof_batch_capacity,
        )?;

        // 5. Recompute `a` for the next substep — orientations / positions just
        //    changed so M and τ are stale. Skipped on the last substep (rapier
        //    skips it too: `if !is_last_substep`).
        if !is_last_substep {
            self.compute_dynamics(pass, mb, args)?;
        }

        // 6. Stabilization: strip positional bias from each constraint's `rhs`.
        if mb.has_joint_constraints {
            self.remove_joint_constraint_bias.call(
                pass,
                dispatch,
                &mb.multibody_info,
                &mut mb.joint_constraints,
                &mb.num_multibodies,
                &mb.multibodies_batch_capacity,
                &mb.joint_constraints_batch_capacity,
            )?;
        }
        self.remove_contact_constraint_bias.call(
            pass,
            dispatch,
            &mut mb.contact_constraints,
            &mb.contact_constraint_count,
            &mb.num_multibodies,
            &mb.multibodies_batch_capacity,
            &mb.contact_constraints_batch_capacity,
        )?;
        if mb.mb_imp_joints_per_batch > 0 {
            let imp_dispatch = [mb.mb_imp_joints_per_batch, mb.num_batches, 1];
            self.remove_impulse_joint_constraint_bias.call(
                pass,
                imp_dispatch,
                &mb.mb_imp_joint_builders,
                &mut mb.mb_imp_joint_constraints,
                &mb.mb_imp_joint_count,
                &mb.mb_imp_joints_batch_capacity,
                &mb.mb_imp_joint_constraints_batch_capacity,
            )?;
        }

        // 7. Final PGS sweep WITHOUT bias — settles velocity to pure-zero
        //    along constrained DOFs, eliminating the rebound that drives jitter.
        if mb.has_joint_constraints {
            self.solve_joint_constraints.call(
                pass,
                dispatch,
                &mb.multibody_info,
                &mut mb.joint_constraints,
                &mb.joint_constraint_columns,
                &mut mb.dof_velocities,
                &mb.num_multibodies,
                &mb.multibodies_batch_capacity,
                &mb.dof_batch_capacity,
                &mb.joint_constraints_batch_capacity,
                &mb.joint_constraint_columns_batch_capacity,
            )?;
        }
        self.solve_contact_constraints.call(
            pass,
            dispatch,
            &mb.multibody_info,
            &mut mb.contact_constraints,
            &mb.contact_constraint_jacs,
            &mb.contact_constraint_columns,
            &mb.contact_constraint_count,
            &mut mb.dof_velocities,
            args.solver_vels,
            &mb.num_multibodies,
            &mb.multibodies_batch_capacity,
            &mb.dof_batch_capacity,
            &mb.contact_constraints_batch_capacity,
            &mb.contact_constraint_columns_batch_capacity,
            args.colliders_batch_capacity,
        )?;
        if mb.mb_imp_joints_per_batch > 0 {
            // Final stabilization sweep WITHOUT bias.
            self.solve_impulse_joint_constraints.call(
                pass,
                [1u32, mb.num_batches, 1],
                // set 0 storage
                &mb.mb_imp_joint_builders,
                &mut mb.mb_imp_joint_constraints,
                &mb.mb_imp_joint_jacobians,
                &mb.mb_imp_joint_count,
                // set 0 uniforms
                &mb.mb_imp_joints_batch_capacity,
                &mb.mb_imp_joint_constraints_batch_capacity,
                // set 1 storage
                &mb.multibody_info,
                &mut mb.dof_velocities,
                args.solver_vels,
                // set 1 uniforms
                &mb.multibodies_batch_capacity,
                &mb.dof_batch_capacity,
                args.colliders_batch_capacity,
            )?;
        }

        Ok(())
    }

    /// FK → jacobians → vel propagation → mass matrix → gravity force → LU decompose
    /// → LU solve. Called once per visible step (via `init_step`) and again at the
    /// end of every substep except the last. After this call, `gen_forces` holds
    /// the generalized acceleration `a` for the *next* substep's velocity update.
    fn compute_dynamics(
        &self,
        pass: &mut GpuPass,
        mb: &mut GpuMultibodySet,
        args: &mut MultibodySolverArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        let dispatch = [mb.multibodies_per_batch, mb.num_batches, 1];

        // Fused FK + body-jacobians + velocity propagation + CRBA-with-Coriolis
        // mass-matrix assembly (4 dispatches → 1).
        let pre_dispatch = [mb.multibodies_per_batch * MB_LU_LANES, mb.num_batches, 1];
        if mb.implicit_coriolis {
            self.compute_dynamics_pre.call(
                pass,
                pre_dispatch,
                &mb.multibody_info,
                &mb.links_static,
                &mut mb.links_workspace,
                &mb.links_mprops,
                args.poses,
                &mut mb.body_jacobians,
                &mut mb.mass_matrices,
                &mut mb.coriolis_v,
                &mut mb.coriolis_w,
                &mut mb.i_coriolis_dt,
                &mb.dof_velocities,
                &mb.damping,
                &mb.num_multibodies,
                &mb.dt,
                &mb.multibodies_batch_capacity,
                &mb.links_batch_capacity,
                &mb.jacobians_batch_capacity,
                &mb.mass_matrix_batch_capacity,
                &mb.coriolis_batch_capacity,
                &mb.i_coriolis_dt_batch_capacity,
                &mb.dof_batch_capacity,
                args.colliders_batch_capacity,
            )?;
        } else {
            self.compute_dynamics_without_coriolis_pre.call(
                pass,
                pre_dispatch,
                &mb.multibody_info,
                &mb.links_static,
                &mut mb.links_workspace,
                &mb.links_mprops,
                args.poses,
                &mut mb.body_jacobians,
                &mut mb.mass_matrices,
                &mb.dof_velocities,
                &mb.damping,
                &mb.num_multibodies,
                &mb.dt,
                &mb.multibodies_batch_capacity,
                &mb.links_batch_capacity,
                &mb.jacobians_batch_capacity,
                &mb.mass_matrix_batch_capacity,
                &mb.dof_batch_capacity,
                args.colliders_batch_capacity,
            )?;
        }

        // Fused gravity + LU factor + LU solve.
        let grav_lu_dispatch = [mb.multibodies_per_batch * MB_LU_LANES, mb.num_batches, 1];
        self.gravity_and_lu.call(
            pass,
            grav_lu_dispatch,
            &mb.multibody_info,
            &mb.links_static,
            &mut mb.links_workspace,
            &mb.links_mprops,
            &mb.body_jacobians,
            &mut mb.gen_forces,
            &mut mb.mass_matrices,
            &mut mb.lu_pivots,
            &mb.dof_velocities,
            &mb.damping,
            &mb.num_multibodies,
            &mb.gravity,
            &mb.multibodies_batch_capacity,
            &mb.links_batch_capacity,
            &mb.jacobians_batch_capacity,
            &mb.mass_matrix_batch_capacity,
            &mb.dof_batch_capacity,
        )?;

        Ok(())
    }
}
