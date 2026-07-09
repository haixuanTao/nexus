//! The [`GpuMultibodySet`] buffers: struct definition, accessors and
//! runtime-mutation entry points (motors, gravity, dt, softness).

use crate::math::Pose;
use crate::shaders::dynamics::{
    ConstraintSoftness, LocalMassProperties, MbImpulseJointBuilder, MbImpulseJointConstraint,
    MultibodyContactConstraint, MultibodyInfo, MultibodyJointConstraint, MultibodyLinkStatic,
    MultibodyLinkWorkspace, RbdSimParams,
};
use crate::shaders::utils::BatchIndices;
use glamx::Vec4;
use khal::BufferUsages;
use khal::backend::{Backend, GpuBackend, GpuBackendError};
use rapier3d::prelude::JointAxis;
use vortx::tensor::Tensor;

/// Workgroup width for the parallelised LU decompose / solve kernels. Must
/// match the `threads(N, 1, 1)` attribute on `gpu_mb_lu_decompose` and
/// `gpu_mb_lu_solve`.
pub(super) const MB_LU_LANES: u32 = 64;

/// Maximum TOTAL multibody count (capacity × batches) for which the
/// constraint-space (Delassus) contact solve is enabled — each multibody's
/// Delassus block costs `MAX_MB_CONTACT_CONSTRAINTS_PER_MB²` floats (~147 KB
/// in 3D), so huge batched scenes keep the dof-space sweep (~19 MB at this
/// cap).
pub(super) const MAX_DELASSUS_MULTIBODIES: u32 = 128;

use crate::shaders::dynamics::{GenericJoint, JointLimits, JointMotor};

/// GPU-resident articulated multibody set, packed across simulation batches.
///
/// Every buffer is a flat tensor with per-batch capacity (`*_batch_capacity`) and
/// a per-batch length. The multibody/link counts are identical across batches
/// (equal-topology invariant) and read from the `BatchIndices` uniform.
pub struct GpuMultibodySet {
    pub(super) num_batches: u32,
    pub(super) multibodies_per_batch: u32,
    /// Number of *active* multibodies per batch. Identical across batches by
    /// the equal-topology invariant; differs from `multibodies_per_batch` when
    /// the latter is padded to ≥1 to avoid size-zero buffers.
    pub(super) num_active_multibodies: u32,
    pub(super) links_per_batch: u32,
    pub(super) dofs_per_batch: u32,
    pub(super) jacobian_entries_per_batch: u32,
    pub(super) mass_matrix_entries_per_batch: u32,
    pub(super) coriolis_entries_per_batch: u32,
    pub(super) i_coriolis_dt_entries_per_batch: u32,
    /// When `true`, the Coriolis / gyroscopic terms are folded into the mass
    /// matrix (implicit integration). When `false`, they are applied explicitly
    /// as part of the RHS.
    pub(super) implicit_coriolis: bool,
    /// When `false` (no joint limits / motors anywhere), the joint constraint
    /// kernel chain is skipped on the host side.
    pub(super) has_joint_constraints: bool,

    /// Per-batch multibody descriptors.
    pub(super) multibody_info: Tensor<MultibodyInfo>,
    /// Per-batch static link data.
    pub(super) links_static: Tensor<MultibodyLinkStatic>,
    /// CPU-side mirror of [`Self::links_static`] used to support runtime
    /// mutations like motor changes without round-tripping through a GPU read.
    pub(super) links_static_mirror: Vec<MultibodyLinkStatic>,
    /// Lazily-created shader bundle + staging buffers for the per-env RL
    /// reset scatter (`gpu_mb_env_reset`).
    pub(super) env_reset: Option<EnvResetBundle>,
    /// Actuator-delay state for the force-based PD, per-batch stride
    /// `2 + links_per_batch`: `[tick, k, prev_target × links]`. Zeroed =
    /// delay off. See `apply_force_based_pd` in the gravity kernels.
    pub(super) motor_delay_state: Tensor<f32>,
    /// Per-batch per-step link workspace, SoA quad layout — see
    /// `crate::shaders::dynamics::ws_soa`.
    pub(super) links_workspace: Tensor<glamx::Vec4>,
    /// Generalized coordinates (flat).
    pub(super) dof_values: Tensor<f32>,
    /// Packed buffer holding generalized velocities (offset 0) and per-DOF
    /// damping coefficients (offset `damping_section_offset`). Callers reading
    /// velocities should use only the velocity section.
    pub(super) dof_state: Tensor<f32>,
    /// Generalized forces / after solve, generalized accelerations.
    pub(super) gen_forces: Tensor<f32>,
    /// Per-link `6 × ndofs` column-major jacobians.
    pub(super) body_jacobians: Tensor<f32>,
    /// Per-multibody `ndofs × ndofs` mass matrices (also used as LU work buffer).
    pub(super) mass_matrices: Tensor<f32>,
    /// Per-DOF pivot buffer used by LU.
    pub(super) lu_pivots: Tensor<u32>,

    /// Packed buffer holding the three Coriolis scratch sections back-to-back.
    pub(super) coriolis_packed: Tensor<f32>,

    /// Per-multibody flat bank of unit (1-DOF) limit / motor constraints.
    pub(super) joint_constraints: Tensor<MultibodyJointConstraint>,
    /// Per-constraint columns of `M⁻¹` (length `ndofs` each, contiguous per multibody).
    pub(super) joint_constraint_columns: Tensor<f32>,

    /// Per-body lookup `[multibody_idx, link_idx]` (`u32::MAX` sentinel for
    /// free / non-multibody bodies). Indexed by the per-batch local body id.
    pub(super) body_to_link: Tensor<[u32; 2]>,

    /// Per-multibody bank of contact constraints (1 normal + 2 friction per
    /// touched contact point).
    pub(super) contact_constraints: Tensor<MultibodyContactConstraint>,
    /// Per-constraint `Jᵀ` row (length `ndofs`) — the multibody side's
    /// contribution to the constraint Jacobian.
    pub(super) contact_constraint_jacs: Tensor<f32>,
    /// Per-constraint M⁻¹·Jᵀ column (length `ndofs`).
    pub(super) contact_constraint_columns: Tensor<f32>,
    /// Per-multibody Delassus blocks (`MAX_MB_CONTACT_CONSTRAINTS_PER_MB²`
    /// floats each) for the constraint-space contact sweep. Only allocated
    /// when the total multibody count is at most
    /// [`MAX_DELASSUS_MULTIBODIES`] (the blocks are ~147 KB each in 3D);
    /// `None` selects the dof-space solve path.
    pub(super) contact_delassus: Option<Tensor<f32>>,

    /// Per-batch number of multibody-touching impulse joints (body1 OR body2
    /// part of any multibody).
    pub(super) mb_imp_joint_count: Tensor<u32>,
    /// Per-batch slab of impulse-joint builder descriptors.
    pub(super) mb_imp_joint_builders: Tensor<MbImpulseJointBuilder>,
    /// Per-batch slab of axis constraints.
    pub(super) mb_imp_joint_constraints: Tensor<MbImpulseJointConstraint>,
    /// Per-batch flat jacobians buffer — stores `J / W·J` for both sides
    /// of every axis constraint of every joint.
    pub(super) mb_imp_joint_jacobians: Tensor<f32>,

    /// Capacities (per-batch strides) for the impulse-joint slabs above.
    /// Mirrored into `BatchIndices` via [`Self::fill_batch_indices`].
    pub(super) mb_imp_joints_per_batch: u32,
    pub(super) mb_imp_joint_constraints_per_batch: u32,
    pub(super) mb_imp_joint_jacobians_per_batch: u32,

    /// Per-batch prefix-sum over the color-sorted `mb_imp_joint_builders`.
    /// Built at init time by `set_impulse_joints` (greedy graph coloring).
    pub(super) mb_imp_joint_color_groups: Tensor<u32>,
    /// Number of colors (per-batch stride of `mb_imp_joint_color_groups`,
    /// and the host color-loop trip count). CPU mirror.
    pub(crate) mb_imp_joint_num_colors: u32,
    /// Max `ndofs` across every multibody in every batch (CPU mirror of
    /// `BatchIndices::mb_max_ndofs`).
    pub(super) max_ndofs: u32,
    /// Max link count across every multibody in every batch (CPU mirror of
    /// `BatchIndices::mb_max_links`).
    pub(super) max_links: u32,
    /// Largest color group across batches — the per-color dispatch width.
    pub(super) mb_imp_joint_max_color_group_len: u32,
    /// Per-batch capacities of the joint / contact constraint slabs (CPU-side
    /// mirror). Stored so `RbdState` can rebuild its `BatchIndices` when caps change.
    pub(super) joint_constraints_per_batch: u32,
    pub(super) joint_constraint_columns_per_batch: u32,
    pub(super) contact_constraints_per_batch: u32,
    pub(super) contact_constraint_columns_per_batch: u32,

    /// Number of solver iterations to run on `joint_constraints` per `step()`.
    pub(super) num_solver_iterations: u32,

    /// Gravity vector (only the first 3 components are read by the shaders).
    pub(super) gravity: Tensor<Vec4>,
    /// Current integration timestep.
    pub(super) dt: Tensor<f32>,
    /// Precomputed soft-constraint coefficients (contact + joint, rapier
    /// TGS-soft).
    pub(super) constraint_softness: Tensor<ConstraintSoftness>,
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

    /// Thread-count grid for the per-multibody kernels, with `(multibody,
    /// batch)` flattened into X. A 2D `[per_batch, num_batches]` grid gives
    /// every batch its own workgroup — mostly idle lanes when each
    /// environment holds a single robot. The kernels decode
    /// `batch_id = x / multibodies_len`, `mb_idx = x % multibodies_len`.
    pub(crate) fn flat_mb_dispatch(&self) -> [u32; 3] {
        [self.num_active_multibodies * self.num_batches, 1, 1]
    }

    /// Lanes per multibody for the packed per-multibody dynamics kernels
    /// (`compute_dynamics_pre`, `gravity_and_lu`) — mirrored into
    /// `BatchIndices::mb_pack_lanes`.
    ///
    /// `1` selects the SERIAL tier (Genesis-style): one thread runs its
    /// multibody's whole FK/CRBA/LU chain with no barriers at all, 64
    /// multibodies per workgroup with every lane busy. For small robots this
    /// beats lane-parallelism — whose ~60-barrier dependency chain caps how
    /// fast one multibody can finish — but ONLY once there are enough
    /// multibodies for the thread count to hide the long serial chain's
    /// latency (measured crossover ≈1024 on Apple M-series with the SoA
    /// workspace; below that, spreading each robot across 8 lanes wins
    /// despite the barriers).
    pub(crate) fn pack_lanes(&self) -> u32 {
        let total_mb = self.num_active_multibodies * self.num_batches;
        // The serial tier's dynamics numerically diverge from the lane tier
        // (~1e-4 relative after ONE step on a contact-free chain, ~3% after
        // 25 — beyond FP-reordering noise; likely the in-place unpivoted LU).
        // Upstream's total_mb >= 1024 auto-switch therefore changes the
        // physics an RL policy sees depending on how many envs it trains
        // with. On this branch physics consistency wins by default:
        //   unset / `NEXUS_SERIAL_MB=0` → lane-parallel always
        //   `NEXUS_SERIAL_MB=1`        → force serial (small robots)
        //   `NEXUS_SERIAL_MB=auto`     → upstream's measured-crossover
        let mode = std::env::var("NEXUS_SERIAL_MB");
        let serial = match mode.as_deref() {
            Ok("1") => self.max_ndofs <= 8,
            Ok("auto") => self.max_ndofs <= 8 && total_mb >= 1024,
            _ => false,
        };
        if serial {
            1
        } else {
            self.max_ndofs.next_power_of_two().clamp(8, MB_LU_LANES)
        }
    }

    /// Thread-count grid for the packed per-multibody WORKGROUP kernels
    /// (`compute_dynamics_pre`, `gravity_and_lu`): `64 / pack_lanes`
    /// multibodies per 64-lane workgroup, flattened `(multibody, batch)`.
    pub(crate) fn packed_wg_dispatch(&self) -> [u32; 3] {
        let slots = MB_LU_LANES / self.pack_lanes();
        let total = self.num_active_multibodies * self.num_batches;
        [total.div_ceil(slots) * MB_LU_LANES, 1, 1]
    }

    /// True if the set contains no multibodies in any batch.
    ///
    /// Uses the *active* count: the per-batch capacity is padded to >= 1 to
    /// avoid zero-sized buffers, so testing it would run the whole multibody
    /// kernel chain every step for scenes without any multibody.
    pub fn is_empty(&self) -> bool {
        self.num_active_multibodies == 0 || self.links_per_batch == 0
    }

    /// Number of colors used by the colored multibody impulse-joint sweeps.
    pub fn mb_imp_joint_num_colors(&self) -> u32 {
        self.mb_imp_joint_num_colors
    }

    /// GPU buffer holding generalized velocities followed by per-DOF damping.
    /// The velocity section is `[0, dof_batch_capacity * num_batches)`; the
    /// damping section follows. Callers reading velocities should use only the
    /// first half.
    pub fn dof_state(&self) -> &Tensor<f32> {
        &self.dof_state
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

    /// Enable implicit integration of the Coriolis / gyroscopic terms. Implicit
    /// treatment stabilizes the integrator at large time-steps; the explicit
    /// form is slightly cheaper but can become unstable for fast rotations.
    pub fn set_implicit_coriolis(&mut self, enabled: bool) {
        self.implicit_coriolis = enabled;
    }

    /// Whether the Coriolis / gyroscopic terms are folded into the mass matrix
    /// (implicit integration) in the next `step()`.
    pub fn implicit_coriolis(&self) -> bool {
        self.implicit_coriolis
    }

    /// Number of TGS-soft substeps per visible step (matches rapier's
    /// `num_solver_iterations`).
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
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        )
        .unwrap();
    }

    /// Upload the soft contact-constraint coefficients, computed from the
    /// (substep) sim params. Must be called whenever the contact softness /
    /// timestep changes.
    pub fn set_constraint_softness(&mut self, backend: &GpuBackend, params: &RbdSimParams) {
        self.constraint_softness = Tensor::scalar(
            backend,
            ConstraintSoftness::from_params(params),
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
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
        // Batch-interleaved links layout: element `link_id` of batch `batch`
        // lives at `link_id · num_batches + batch` (mirror included).
        let global_idx = (link_id * self.num_batches + batch) as usize;
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

    /// Sets a motor's target *position* on a multibody joint and uploads the
    /// updated link to the GPU. The companion of
    /// [`set_motor_velocity`](Self::set_motor_velocity) for PD position
    /// control: the solver drives the joint toward `target_pos` using the
    /// motor's `stiffness`/`damping`/`max_force`, configured once at build
    /// time on the rapier joint and carried through `from_rapier`. This
    /// setter is the per-step hot path: it writes only the target.
    ///
    /// `link_id` is the link index within the batch; `axis` is the joint axis
    /// (0..=2 linear, 3..=5 angular). The motor axis bit is auto-enabled.
    /// `target_vel` is left untouched, so the velocity term acts as damping.
    pub fn set_motor_position(
        &mut self,
        backend: &GpuBackend,
        batch: u32,
        link_id: u32,
        axis: JointAxis,
        target_pos: f32,
    ) -> Result<(), GpuBackendError> {
        // Batch-interleaved links layout (mirror included).
        let global_idx = (link_id * self.num_batches + batch) as usize;
        let axis_id = axis as usize;
        let entry = match self.links_static_mirror.get_mut(global_idx) {
            Some(e) => e,
            None => return Ok(()),
        };
        entry.data.motors[axis_id].target_pos = target_pos;
        entry.data.motor_axes |= 1u32 << axis_id;
        let snapshot = *entry;
        backend.write_buffer(
            self.links_static.buffer_mut(),
            global_idx as u64,
            std::slice::from_ref(&snapshot),
        )
    }

    /// Bulk version of [`set_motor_position`](Self::set_motor_position):
    /// stage updates into the host mirror without touching the GPU, then call
    /// [`flush_links_static`](Self::flush_links_static) once to push the
    /// whole mirror in a single `write_buffer`. Avoids the
    /// `N_envs · N_joints` per-step write_buffer overhead.
    pub fn stage_motor_position(
        &mut self,
        batch: u32,
        link_id: u32,
        axis: JointAxis,
        target_pos: f32,
    ) {
        let global_idx = (link_id * self.num_batches + batch) as usize;
        let axis_id = axis as usize;
        let Some(entry) = self.links_static_mirror.get_mut(global_idx) else {
            return;
        };
        entry.data.motors[axis_id].target_pos = target_pos;
        entry.data.motor_axes |= 1u32 << axis_id;
    }

    /// Push the entire host-side `links_static_mirror` to the GPU in a single
    /// `write_buffer` call. Pairs with
    /// [`stage_motor_position`](Self::stage_motor_position) for batched
    /// per-step motor target updates.
    pub fn flush_links_static(&mut self, backend: &GpuBackend) -> Result<(), GpuBackendError> {
        backend.write_buffer(self.links_static.buffer_mut(), 0, &self.links_static_mirror)
    }

    /// Per-batch stride of the actuator-delay state buffer:
    /// `[tick, k, prev_target × links_per_batch]`.
    pub fn motor_delay_stride(&self) -> u32 {
        2 + self.links_per_batch
    }

    /// Upload the actuator-delay state for all batches (see
    /// [`Self::motor_delay_stride`] for the layout; `data.len()` must be
    /// `stride × num_batches`). Zeroed state = delay off. Call BEFORE the
    /// step's kernels are queued: an H2D copy issued between queued substeps
    /// stalls the stream — exactly what this GPU-side delay removes.
    pub fn write_motor_delay_state(
        &mut self,
        backend: &GpuBackend,
        data: &[f32],
    ) -> Result<(), GpuBackendError> {
        backend.write_buffer(self.motor_delay_state.buffer_mut(), 0, data)
    }

    /// Scatter per-(actuated-joint, env) motor target positions into
    /// `links_static` on the GPU — the on-device equivalent of
    /// [`stage_motor_position`](Self::stage_motor_position) +
    /// [`flush_links_static`](Self::flush_links_static). `targets` is
    /// row-major `[num_actuated × num_batches]` (element `(j, env)` at
    /// `j·num_batches+env`); `actuated_link_ids[j]` is the link index of
    /// actuated joint `j`. Writes `motors[axis].target_pos` and sets the
    /// `motor_axes` bit. Prerequisite for applying RL actions without a host
    /// round-trip (CUDA-graph-capturable rollout).
    ///
    /// NOTE: bypasses `links_static_mirror` (targets live only on the GPU
    /// afterwards) — don't interleave with the mirror-based setters for the
    /// same axis.
    pub fn scatter_motor_targets(
        &mut self,
        backend: &GpuBackend,
        targets: &[f32],
        actuated_link_ids: &[u32],
        axis: u32,
    ) -> Result<(), GpuBackendError> {
        use crate::shaders::dynamics::GpuScatterMotorTargets;
        use khal::backend::Encoder;

        /// `#[derive(Shader)]` supplies `from_backend` for the embedded entry.
        #[derive(Shader)]
        struct MotorScatterBundle {
            scatter: GpuScatterMotorTargets,
        }

        let num_actuated = actuated_link_ids.len() as u32;
        let uu = BufferUsages::STORAGE | BufferUsages::UNIFORM;
        let t_targets =
            Tensor::vector(backend, targets, BufferUsages::STORAGE | BufferUsages::COPY_DST)?;
        let t_links = Tensor::vector(backend, actuated_link_ids, BufferUsages::STORAGE)?;
        let u_na = Tensor::scalar(backend, num_actuated, uu)?;
        let u_ne = Tensor::scalar(backend, self.num_batches, uu)?;
        let u_ax = Tensor::scalar(backend, axis, uu)?;
        let bundle = MotorScatterBundle::from_backend(backend)?;
        let mut enc = backend.begin_encoding();
        {
            let mut pass = enc.begin_pass("scatter_motor_targets", None);
            bundle.scatter.call(
                &mut pass,
                [num_actuated, self.num_batches, 1],
                &t_targets,
                &mut self.links_static,
                &t_links,
                &u_na,
                &u_ne,
                &u_ax,
            )?;
        }
        backend.submit(enc)?;
        Ok(())
    }

    /// Diagnostic readback accessor for the static link descriptors
    /// (motor targets live here). Not a stable API.
    pub fn dbg_links_static(&self) -> &Tensor<MultibodyLinkStatic> {
        &self.links_static
    }


    /// Number of link slots per environment (the per-link-buffer stride
    /// factor; buffers are batch-interleaved, element `link` of env `e` at
    /// `link · num_batches + e`).
    pub fn links_per_batch(&self) -> u32 {
        self.links_per_batch
    }

    /// Refreshes every link's joint parameters (motor targets/gains, limits) of
    /// environment `env` from a rapier multibody set laid out identically to the
    /// one this GPU set was built from (same multibody/link traversal order as
    /// [`from_rapier`](Self::from_rapier)), then uploads the `links_static`
    /// buffer in one write.
    ///
    /// This is the per-step control path for actuated robots: mutate the motors
    /// on the CPU rapier joints (e.g. via `rapier3d-mjcf`'s
    /// `apply_controls_multibody`, which implements the MJCF actuator
    /// semantics), then call this to push the new motor state to the GPU. Only
    /// joint data is refreshed — coordinates, velocities and mass properties are
    /// untouched, so this cannot be used to teleport links.
    pub fn sync_joint_data_from_rapier(
        &mut self,
        backend: &GpuBackend,
        env: u32,
        set: &crate::rapier::dynamics::MultibodyJointSet,
        bodies: &crate::rapier::dynamics::RigidBodySet,
    ) -> Result<(), GpuBackendError> {
        let nb = self.num_batches as usize;
        let mut offset = 0usize;
        for mb in set.multibodies() {
            // Mirror `from_rapier`'s fixed-root handling: a non-dynamic root has
            // all 6 DOFs locked on the GPU even though rapier models it as free.
            let root_is_dynamic = mb
                .link(0)
                .and_then(|r| bodies.get(r.rigid_body_handle()))
                .map(|rb| rb.is_dynamic())
                .unwrap_or(false);
            for (link_idx, link) in mb.links().enumerate() {
                // Batch-interleaved links layout.
                let global_idx = offset * nb + env as usize;
                let Some(entry) = self.links_static_mirror.get_mut(global_idx) else {
                    return Ok(());
                };
                let mut data = convert_generic_joint(link.joint().data);
                if link_idx == 0 && !root_is_dynamic {
                    data.locked_axes = 0x3f;
                }
                entry.data = data;
                offset += 1;
            }
        }
        backend.write_buffer(
            self.links_static.buffer_mut(),
            0,
            &self.links_static_mirror,
        )
    }
    /// Upload a new gravity vector.
    pub fn set_gravity(&mut self, backend: &GpuBackend, g: [f32; 3]) {
        self.gravity = Tensor::scalar(
            backend,
            Vec4::new(g[0], g[1], g[2], 0.0),
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        )
        .unwrap();
    }

    /// Number of multibody-touching impulse joints in any batch.
    pub fn mb_impulse_joints_per_batch(&self) -> u32 {
        self.mb_imp_joints_per_batch
    }

    /// Populate the multibody-owned fields of `BatchIndices`. Leaves the
    /// RBD-side fields (`colliders_batch_capacity`, `contacts_batch_capacity`,
    /// `collision_pairs_batch_capacity`, `impulse_joints_batch_capacity`,
    /// `color_groups_batch_capacity`) untouched — the caller fills those.
    pub(crate) fn fill_batch_indices(&self, dst: &mut BatchIndices) {
        dst.multibodies_batch_capacity = self.multibodies_per_batch;
        dst.multibodies_len = self.num_active_multibodies;
        dst.links_batch_capacity = self.links_per_batch;
        dst.jacobians_batch_capacity = self.jacobian_entries_per_batch;
        dst.mass_matrix_batch_capacity = self.mass_matrix_entries_per_batch;
        dst.coriolis_batch_capacity = self.coriolis_entries_per_batch;
        dst.i_coriolis_dt_batch_capacity = self.i_coriolis_dt_entries_per_batch;
        dst.dof_batch_capacity = self.dofs_per_batch;
        dst.mb_joint_constraints_batch_capacity = self.joint_constraints_per_batch;
        dst.mb_joint_constraint_columns_batch_capacity = self.joint_constraint_columns_per_batch;
        dst.mb_contact_constraints_batch_capacity = self.contact_constraints_per_batch;
        dst.mb_contact_constraint_columns_batch_capacity =
            self.contact_constraint_columns_per_batch;
        dst.mb_imp_joints_batch_capacity = self.mb_imp_joints_per_batch.max(1);
        dst.mb_imp_joint_constraints_batch_capacity = self.mb_imp_joint_constraints_per_batch;
        dst.mb_imp_joint_jacobians_batch_capacity = self.mb_imp_joint_jacobians_per_batch;
        dst.mb_imp_joint_color_groups_batch_capacity = self.mb_imp_joint_num_colors.max(1);
        dst.mb_max_ndofs = self.max_ndofs;
        dst.mb_max_links = self.max_links;
        dst.mb_pack_lanes = self.pack_lanes();
        dst.coriolis_w_section_offset = self.coriolis_entries_per_batch * self.num_batches;
        dst.i_coriolis_dt_section_offset = 2 * self.coriolis_entries_per_batch * self.num_batches;
        dst.dof_damping_section_offset = self.dofs_per_batch * self.num_batches;
    }

    /// Upload a new integration timestep.
    pub fn set_dt(&mut self, backend: &GpuBackend, dt: f32) {
        self.dt = Tensor::scalar(
            backend,
            dt,
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        )
        .unwrap();
    }
}

pub(super) fn convert_link_mprops(
    m: &crate::rapier::prelude::MassProperties,
) -> LocalMassProperties {
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

pub(super) fn convert_generic_joint(j: crate::rapier::dynamics::GenericJoint) -> GenericJoint {
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
pub(super) fn make_workspace_init() -> MultibodyLinkWorkspace {
    let mut w: MultibodyLinkWorkspace = bytemuck::Zeroable::zeroed();
    w.joint_rot = glamx::Quat::IDENTITY;
    w.local_to_parent = Pose::default();
    w.local_to_world = Pose::default();
    w
}

/*
 * RL primitives: per-env snapshot / teleport-reset.
 */

use crate::shaders::dynamics::{
    GpuMbEnvReset, MULTIBODY_ROOT, WS_JOINT_ROT, WS_JOINT_VEL, WS_KIN_ACC, WS_LTP, WS_LTW,
    WS_QUADS, WS_RB_VELS, WS_SHIFT02, WS_SHIFT23, WsAddr, ws_coords, ws_pose, ws_rot,
    ws_soa_from_structs, ws_vec, ws_vel,
};
use glamx::UVec4;
use khal::Shader;
use khal::backend::Encoder;

/// CPU snapshot of one environment's multibody carry-over state (the state a
/// per-env RL reset must restore): AoS link workspace, static link
/// descriptors (motor targets/impulses live here), generalized coordinates
/// and generalized velocities. Captured from batch 0 of a (typically
/// single-env template) [`GpuMultibodySet`]; applied to any env of a live set
/// with [`GpuMultibodySet::reset_env_from_snapshot`] — write/dispatch only,
/// no per-reset GPU→CPU readback.
#[derive(Clone)]
pub struct GpuMultibodySnapshot {
    /// AoS per-link workspace of batch 0 (`links_per_batch` entries,
    /// including padding slots). Converted to the SoA quad layout on upload.
    links_workspace: Vec<MultibodyLinkWorkspace>,
    links_static: Vec<MultibodyLinkStatic>,
    /// Generalized coordinates of batch 0 (`dofs_per_batch`).
    dof_values: Vec<f32>,
    /// Generalized velocities of batch 0 (`dofs_per_batch`; the velocity
    /// section of `dof_state` — the damping section is static config).
    dof_vels: Vec<f32>,
}

impl GpuMultibodySnapshot {
    /// True for entries that describe a real link (the buffers are padded to
    /// `links_per_batch` with zeroed slots: rb_id 0, parent 0, ndofs 0 — a
    /// combination no real link can have, since a chain's body-0 link is its
    /// root and roots carry `parent_link_id == MULTIBODY_ROOT`).
    fn link_is_valid(ls: &MultibodyLinkStatic) -> bool {
        ls.parent_link_id == MULTIBODY_ROOT || ls.ndofs > 0 || ls.rb_id != 0
    }

    /// True if this link's multibody has a FREE root joint (floating base).
    /// Only such multibodies can be rigidly translated; a fixed-base chain is
    /// welded to the world and must not move.
    fn mb_root_is_free(&self, multibody_id: u32) -> bool {
        self.links_static.iter().any(|ls| {
            Self::link_is_valid(ls)
                && ls.multibody_id == multibody_id
                && ls.parent_link_id == MULTIBODY_ROOT
                && ls.data.locked_axes == 0
        })
    }

    /// Calls `f(rb_id)` for every rigid body backing a link of a FREE-rooted
    /// (floating-base) multibody — the set of bodies an offset-reset moves.
    pub(crate) fn for_each_link_rb_id(&self, mut f: impl FnMut(u32)) {
        for ls in &self.links_static {
            if Self::link_is_valid(ls) && self.mb_root_is_free(ls.multibody_id) {
                f(ls.rb_id);
            }
        }
    }

    /// A copy of this snapshot with every FLOATING-BASE multibody translated
    /// by `offset` (world frame). Rotations, joint coordinates past the free
    /// linear DOFs, velocities and `dof_values` are translation-invariant;
    /// the free root's world position lives in `coords[0..3]` +
    /// `local_to_parent` (the root's parent frame IS the world), and every
    /// link's `local_to_world` carries its body pose. Fixed-base multibodies
    /// are left untouched. `body_poses` (owned by `RbdSnapshot`) must be
    /// translated by the caller for the same rb_ids.
    pub(crate) fn translated(&self, offset: glamx::Vec3) -> GpuMultibodySnapshot {
        let mut out = self.clone();
        for (ws, ls) in out.links_workspace.iter_mut().zip(&self.links_static) {
            if !Self::link_is_valid(ls) || !self.mb_root_is_free(ls.multibody_id) {
                continue;
            }
            ws.local_to_world.translation += offset;
            if ls.parent_link_id == MULTIBODY_ROOT {
                ws.local_to_parent.translation += offset;
                ws.coords[0] += offset.x;
                ws.coords[1] += offset.y;
                ws.coords[2] += offset.z;
            }
        }
        out
    }
}

/// Standalone shader bundle + persistent staging buffers for the per-env
/// reset scatter. Lazily created on first reset (allocations stay outside
/// any captured region).
pub(super) struct EnvResetBundle {
    shader: EnvResetShader,
    staging_ws: Tensor<Vec4>,
    staging_links: Tensor<MultibodyLinkStatic>,
    staging_dofs: Tensor<f32>,
    params: Tensor<UVec4>,
}

/// `#[derive(Shader)]` supplies `from_backend`, which loads the embedded
/// `gpu_mb_env_reset` entry point.
#[derive(Shader)]
struct EnvResetShader {
    kernel: GpuMbEnvReset,
}

impl EnvResetBundle {
    fn new(backend: &GpuBackend, lpb: u32, dpb: u32) -> Self {
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        Self {
            shader: EnvResetShader::from_backend(backend).unwrap(),
            staging_ws: Tensor::vector_uninit(backend, lpb * WS_QUADS, storage).unwrap(),
            staging_links: Tensor::vector_uninit(backend, lpb, storage).unwrap(),
            staging_dofs: Tensor::vector_uninit(backend, (dpb * 2).max(1), storage).unwrap(),
            params: Tensor::scalar(
                backend,
                UVec4::ZERO,
                BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            )
            .unwrap(),
        }
    }
}

impl GpuMultibodySet {
    /// Read this (typically single-env template) set's batch-0 state off the
    /// GPU into a CPU snapshot. Call once per template at setup; pass the
    /// result to [`Self::reset_env_from_snapshot`] for readback-free per-env
    /// resets.
    pub async fn snapshot(&self, backend: &GpuBackend) -> GpuMultibodySnapshot {
        let nb = self.num_batches;
        let lpb = self.links_per_batch as usize;
        let dpb = self.dofs_per_batch as usize;

        let mut ws_soa: Vec<Vec4> = bytemuck::zeroed_vec(self.links_workspace.len() as usize);
        backend
            .slow_read_buffer(self.links_workspace.buffer(), &mut ws_soa)
            .await
            .unwrap();
        let mut ls_all: Vec<MultibodyLinkStatic> =
            bytemuck::zeroed_vec(self.links_static.len() as usize);
        backend
            .slow_read_buffer(self.links_static.buffer(), &mut ls_all)
            .await
            .unwrap();
        let mut dv_all: Vec<f32> = bytemuck::zeroed_vec(self.dof_values.len() as usize);
        backend
            .slow_read_buffer(self.dof_values.buffer(), &mut dv_all)
            .await
            .unwrap();
        let mut ds_all: Vec<f32> = bytemuck::zeroed_vec(self.dof_state.len() as usize);
        backend
            .slow_read_buffer(self.dof_state.buffer(), &mut ds_all)
            .await
            .unwrap();

        // Gather batch 0 out of the interleave; de-SoA the workspace through
        // the shared layout accessors (one source of truth with the kernels).
        let a = WsAddr::new(0, nb, 0);
        let mut links_workspace = Vec::with_capacity(lpb);
        for k in 0..lpb as u32 {
            let mut w: MultibodyLinkWorkspace = bytemuck::Zeroable::zeroed();
            w.joint_rot = ws_rot(&ws_soa, a, k, WS_JOINT_ROT);
            w.coords = ws_coords(&ws_soa, a, k);
            w.local_to_parent = ws_pose(&ws_soa, a, k, WS_LTP);
            w.local_to_world = ws_pose(&ws_soa, a, k, WS_LTW);
            w.shift02 = ws_vec(&ws_soa, a, k, WS_SHIFT02);
            w.shift23 = ws_vec(&ws_soa, a, k, WS_SHIFT23);
            w.joint_velocity = ws_vel(&ws_soa, a, k, WS_JOINT_VEL);
            w.rb_vels = ws_vel(&ws_soa, a, k, WS_RB_VELS);
            w.kinematic_acc = ws_vel(&ws_soa, a, k, WS_KIN_ACC);
            links_workspace.push(w);
        }
        GpuMultibodySnapshot {
            links_workspace,
            links_static: (0..lpb).map(|k| ls_all[k * nb as usize]).collect(),
            dof_values: (0..dpb).map(|d| dv_all[d * nb as usize]).collect(),
            dof_vels: (0..dpb).map(|d| ds_all[d * nb as usize]).collect(),
        }
    }

    /// Reset env `dst_env` in-place from the single-env template `src`
    /// (readback + scatter). Prefer snapshotting the template once and using
    /// [`Self::reset_env_from_snapshot`] in reset loops.
    pub async fn reset_env_from(
        &mut self,
        backend: &GpuBackend,
        dst_env: u32,
        src: &GpuMultibodySet,
    ) {
        if self.is_empty() {
            return;
        }
        let snap = src.snapshot(backend).await;
        self.reset_env_from_snapshot(backend, dst_env, &snap);
    }

    /// Reset env `dst_env` from a CPU snapshot: one staging upload + one
    /// scatter dispatch — no GPU→CPU readback, no per-element strided writes.
    pub fn reset_env_from_snapshot(
        &mut self,
        backend: &GpuBackend,
        dst_env: u32,
        snap: &GpuMultibodySnapshot,
    ) {
        if self.is_empty() {
            return;
        }
        let nb = self.num_batches;
        let lpb = self.links_per_batch;
        let dpb = self.dofs_per_batch;
        debug_assert_eq!(snap.links_static.len(), lpb as usize);
        debug_assert_eq!(snap.dof_values.len(), dpb as usize);

        // Keep the host mirror in lockstep (the motor setters read-modify-
        // write it).
        for k in 0..lpb as usize {
            self.links_static_mirror[k * nb as usize + dst_env as usize] = snap.links_static[k];
        }

        // Take the bundle out to sidestep the simultaneous &mut borrows of
        // the live buffers below.
        let mut bundle = match self.env_reset.take() {
            Some(b) => b,
            None => EnvResetBundle::new(backend, lpb, dpb),
        };

        let ws = ws_soa_from_structs(&snap.links_workspace, lpb, 1);
        backend
            .write_buffer(bundle.staging_ws.buffer_mut(), 0, &ws)
            .unwrap();
        backend
            .write_buffer(bundle.staging_links.buffer_mut(), 0, &snap.links_static)
            .unwrap();
        let mut dofs = snap.dof_values.clone();
        dofs.extend_from_slice(&snap.dof_vels);
        if !dofs.is_empty() {
            backend
                .write_buffer(bundle.staging_dofs.buffer_mut(), 0, &dofs)
                .unwrap();
        }
        bundle.params = Tensor::scalar(
            backend,
            UVec4::new(dst_env, nb, lpb, dpb),
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        )
        .unwrap();

        let mut encoder = backend.begin_encoding();
        {
            let mut pass = encoder.begin_pass("[RBD] mb-env-reset", None);
            let pass = &mut pass;
            bundle
                .shader
                .kernel
                .call(
                    pass,
                    lpb * WS_QUADS,
                    &bundle.staging_ws,
                    &bundle.staging_links,
                    &bundle.staging_dofs,
                    &mut self.links_workspace,
                    &mut self.links_static,
                    &mut self.dof_values,
                    &mut self.dof_state,
                    &bundle.params,
                )
                .unwrap();
        }
        backend.submit(encoder).unwrap();
        self.env_reset = Some(bundle);
    }
}
