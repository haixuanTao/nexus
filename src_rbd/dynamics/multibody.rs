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

use crate::math::Pose;
use crate::shaders::dynamics::{
    GpuMbApplyGravityWithCoriolis, GpuMbBodyJacobians, GpuMbForwardKinematics,
    GpuMbInitJointConstraints, GpuMbIntegrate, GpuMbIntegrateVelocities, GpuMbLuDecompose,
    GpuMbLuSolve, GpuMbMassMatrix, GpuMbMassMatrixWithCoriolis, GpuMbRemoveJointConstraintBias,
    GpuMbSolveJointConstraints, GpuMbUpdateVelocities, LocalMassProperties, MultibodyInfo,
    MultibodyJointConstraint, MultibodyLinkStatic, MultibodyLinkWorkspace,
};
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};
use khal::{BufferUsages, Shader};
use vortx::tensor::Tensor;

#[cfg(feature = "from_rapier")]
use {
    crate::rapier::dynamics::{MultibodyJointSet, RigidBodyHandle, RigidBodySet},
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

    /// Per-batch number of multibodies.
    num_multibodies: Tensor<u32>,
    /// Per-batch multibody descriptors.
    multibody_info: Tensor<MultibodyInfo>,
    /// Per-batch static link data.
    links_static: Tensor<MultibodyLinkStatic>,
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

                    let ws = make_workspace_init();
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
            implicit_coriolis: true,

            num_multibodies: Tensor::vector(backend, &all_num_mb, usage_u).unwrap(),
            multibody_info: Tensor::vector(backend, &all_infos, storage).unwrap(),
            links_static: Tensor::vector(backend, &all_statics, storage).unwrap(),
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
    init_joint_constraints: GpuMbInitJointConstraints,
    solve_joint_constraints: GpuMbSolveJointConstraints,
    remove_joint_constraint_bias: GpuMbRemoveJointConstraintBias,
    integrate_velocities: GpuMbIntegrateVelocities,
    integrate: GpuMbIntegrate,
}

/// Arguments for one multibody dispatch. The poses buffer is shared with the rest
/// of the rigid-body pipeline (FK writes link poses there); mass properties are
/// now owned by the multibody itself.
pub struct MultibodySolverArgs<'a> {
    /// Body world-space poses (written by FK).
    pub poses: &'a mut Tensor<Pose>,
    /// Colliders-per-batch capacity (stride in the pose tensor).
    pub colliders_batch_capacity: &'a Tensor<u32>,
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

        self.forward_kinematics.call(
            pass,
            dispatch,
            &mb.multibody_info,
            &mut mb.links_static,
            &mut mb.links_workspace,
            args.poses,
            &mb.links_mprops,
            &mb.num_multibodies,
            &mb.multibodies_batch_capacity,
            &mb.links_batch_capacity,
            args.colliders_batch_capacity,
        )?;

        self.body_jacobians.call(
            pass,
            dispatch,
            &mb.multibody_info,
            &mb.links_static,
            &mb.links_workspace,
            &mut mb.body_jacobians,
            &mb.num_multibodies,
            &mb.multibodies_batch_capacity,
            &mb.links_batch_capacity,
            &mb.jacobians_batch_capacity,
        )?;

        // Velocity propagation is needed in both modes — the explicit Coriolis
        // force assembly reads `rb_vels` and `joint_velocity` from the workspace.
        self.update_velocities.call(
            pass,
            dispatch,
            &mb.multibody_info,
            &mb.links_static,
            &mut mb.links_workspace,
            &mb.links_mprops,
            &mb.dof_velocities,
            &mb.num_multibodies,
            &mb.multibodies_batch_capacity,
            &mb.links_batch_capacity,
            &mb.dof_batch_capacity,
        )?;

        // Mass-matrix assembly: only the mass-matrix path depends on the toggle.
        // Implicit mode folds `M_gyro·dt + C·dt` into the matrix; explicit mode
        // keeps the plain CRBA mass matrix and relies on the force side below.
        if mb.implicit_coriolis {
            self.mass_matrix_with_coriolis.call(
                pass,
                dispatch,
                &mb.multibody_info,
                &mb.links_static,
                &mb.links_workspace,
                &mb.links_mprops,
                &mb.body_jacobians,
                &mut mb.mass_matrices,
                &mut mb.coriolis_v,
                &mut mb.coriolis_w,
                &mut mb.i_coriolis_dt,
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
            )?;
        } else {
            self.mass_matrix.call(
                pass,
                dispatch,
                &mb.multibody_info,
                &mb.links_static,
                &mb.links_workspace,
                &mb.links_mprops,
                &mb.body_jacobians,
                &mut mb.mass_matrices,
                &mb.damping,
                &mb.num_multibodies,
                &mb.dt,
                &mb.multibodies_batch_capacity,
                &mb.links_batch_capacity,
                &mb.jacobians_batch_capacity,
                &mb.mass_matrix_batch_capacity,
                &mb.dof_batch_capacity,
            )?;
        }

        // Explicit force assembly: τ = Jᵀ · (m·g - m·a_kin, -gyro - I·α_kin) + damping.
        // Always runs — the explicit Coriolis/gyroscopic terms are present regardless
        // of whether the mass matrix also folds them in.
        self.apply_gravity_with_coriolis.call(
            pass,
            dispatch,
            &mb.multibody_info,
            &mb.links_static,
            &mut mb.links_workspace,
            &mb.links_mprops,
            &mb.body_jacobians,
            &mut mb.gen_forces,
            &mb.dof_velocities,
            &mb.damping,
            &mb.num_multibodies,
            &mb.gravity,
            &mb.multibodies_batch_capacity,
            &mb.links_batch_capacity,
            &mb.jacobians_batch_capacity,
            &mb.dof_batch_capacity,
        )?;

        // Factor M = P·L·U once, then solve for the gravity RHS. The factorization
        // (in `mass_matrices` + `lu_pivots`) can be reused for additional RHSes.
        self.lu_decompose.call(
            pass,
            dispatch,
            &mb.multibody_info,
            &mut mb.mass_matrices,
            &mut mb.lu_pivots,
            &mb.num_multibodies,
            &mb.multibodies_batch_capacity,
            &mb.mass_matrix_batch_capacity,
            &mb.dof_batch_capacity,
        )?;

        self.lu_solve.call(
            pass,
            dispatch,
            &mb.multibody_info,
            &mb.mass_matrices,
            &mb.lu_pivots,
            &mut mb.gen_forces,
            &mb.num_multibodies,
            &mb.multibodies_batch_capacity,
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

        // 3. PGS sweep WITH bias (positional correction).
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
        self.remove_joint_constraint_bias.call(
            pass,
            dispatch,
            &mb.multibody_info,
            &mut mb.joint_constraints,
            &mb.num_multibodies,
            &mb.multibodies_batch_capacity,
            &mb.joint_constraints_batch_capacity,
        )?;

        // 7. Final PGS sweep WITHOUT bias — settles velocity to pure-zero
        //    along constrained DOFs, eliminating the rebound that drives jitter.
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

        self.forward_kinematics.call(
            pass,
            dispatch,
            &mb.multibody_info,
            &mut mb.links_static,
            &mut mb.links_workspace,
            args.poses,
            &mb.links_mprops,
            &mb.num_multibodies,
            &mb.multibodies_batch_capacity,
            &mb.links_batch_capacity,
            args.colliders_batch_capacity,
        )?;
        self.body_jacobians.call(
            pass,
            dispatch,
            &mb.multibody_info,
            &mb.links_static,
            &mb.links_workspace,
            &mut mb.body_jacobians,
            &mb.num_multibodies,
            &mb.multibodies_batch_capacity,
            &mb.links_batch_capacity,
            &mb.jacobians_batch_capacity,
        )?;
        self.update_velocities.call(
            pass,
            dispatch,
            &mb.multibody_info,
            &mb.links_static,
            &mut mb.links_workspace,
            &mb.links_mprops,
            &mb.dof_velocities,
            &mb.num_multibodies,
            &mb.multibodies_batch_capacity,
            &mb.links_batch_capacity,
            &mb.dof_batch_capacity,
        )?;

        if mb.implicit_coriolis {
            self.mass_matrix_with_coriolis.call(
                pass,
                dispatch,
                &mb.multibody_info,
                &mb.links_static,
                &mb.links_workspace,
                &mb.links_mprops,
                &mb.body_jacobians,
                &mut mb.mass_matrices,
                &mut mb.coriolis_v,
                &mut mb.coriolis_w,
                &mut mb.i_coriolis_dt,
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
            )?;
        } else {
            self.mass_matrix.call(
                pass,
                dispatch,
                &mb.multibody_info,
                &mb.links_static,
                &mb.links_workspace,
                &mb.links_mprops,
                &mb.body_jacobians,
                &mut mb.mass_matrices,
                &mb.damping,
                &mb.num_multibodies,
                &mb.dt,
                &mb.multibodies_batch_capacity,
                &mb.links_batch_capacity,
                &mb.jacobians_batch_capacity,
                &mb.mass_matrix_batch_capacity,
                &mb.dof_batch_capacity,
            )?;
        }

        self.apply_gravity_with_coriolis.call(
            pass,
            dispatch,
            &mb.multibody_info,
            &mb.links_static,
            &mut mb.links_workspace,
            &mb.links_mprops,
            &mb.body_jacobians,
            &mut mb.gen_forces,
            &mb.dof_velocities,
            &mb.damping,
            &mb.num_multibodies,
            &mb.gravity,
            &mb.multibodies_batch_capacity,
            &mb.links_batch_capacity,
            &mb.jacobians_batch_capacity,
            &mb.dof_batch_capacity,
        )?;

        self.lu_decompose.call(
            pass,
            dispatch,
            &mb.multibody_info,
            &mut mb.mass_matrices,
            &mut mb.lu_pivots,
            &mb.num_multibodies,
            &mb.multibodies_batch_capacity,
            &mb.mass_matrix_batch_capacity,
            &mb.dof_batch_capacity,
        )?;
        self.lu_solve.call(
            pass,
            dispatch,
            &mb.multibody_info,
            &mb.mass_matrices,
            &mb.lu_pivots,
            &mut mb.gen_forces,
            &mb.num_multibodies,
            &mb.multibodies_batch_capacity,
            &mb.mass_matrix_batch_capacity,
            &mb.dof_batch_capacity,
        )?;

        Ok(())
    }
}
