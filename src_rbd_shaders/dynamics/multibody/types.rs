//! Per-link / per-multibody data structures shared across all multibody
//! kernels.

use glamx::{Quat, Vec3};

use crate::Pose;
use crate::dynamics::body::Velocity;
use crate::dynamics::joint::GenericJoint;

/// Max degrees of freedom any single joint can expose (6 = free root).
pub const MAX_JOINT_DOFS: usize = 6;

/// Maximum number of simultaneously-active multibody contact constraints per
/// multibody. Sized for typical use (a single multibody touching the
/// environment with up to ~32 contact points × 2 manifold sides). Per-multibody
/// banks of this size are pre-allocated; surplus slots are left inactive.
pub const MAX_MB_CONTACTS_PER_MB: u32 = 64;

/// Sentinel marking a link with no parent (the root).
pub const MULTIBODY_ROOT: u32 = u32::MAX;

/// Per-link static configuration: backing body, parent, joint definition.
///
/// Written once at init time.
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct MultibodyLinkStatic { // TODO: change the name to `MultibodyLink` ?
    /// Index of the rigid body backing this link in the shared body buffers.
    pub rb_id: u32,
    /// Parent link index within the owning multibody. `MULTIBODY_ROOT` for the root.
    pub parent_link_id: u32,
    /// Index of the owning multibody in the `multibody_info` tensor.
    pub multibody_id: u32,
    /// Starting column (in the jacobian / mass-matrix / gen-force tensors) for this
    /// link's DOFs. Assembly ids are contiguous and parent-before-child.
    pub assembly_id: u32,
    /// Number of DOFs this joint contributes.
    pub ndofs: u32,
    /// 1 if this joint's generalized velocities are user-controlled (ignored by the
    /// LU solve). 0 otherwise.
    pub kinematic: u32,
    /// Pad to 16-byte alignment before `data`.
    pub _pad0: [u32; 2],
    /// Joint configuration — reused directly from the impulse-joint infrastructure.
    pub data: GenericJoint,
}

/// Per-link workspace updated every step.
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct MultibodyLinkWorkspace {
    /// Accumulated joint rotation (fed to `body_to_parent`).
    pub joint_rot: Quat,
    /// Generalized coordinates for this joint. Only the first `ndofs` entries are
    /// meaningful. Free linear DOFs come first (in axis order), then free angular DOFs.
    pub coords: [f32; MAX_JOINT_DOFS],
    /// Pad: `joint_rot` (16) + `coords` (24) = 40; need 8 more before Pose (align 16).
    pub _pad0: [u32; 2],
    /// Local-to-parent transform.
    pub local_to_parent: Pose,
    /// Local-to-world transform (the link's body pose).
    pub local_to_world: Pose,
    /// Vector (world frame) from the parent COM to the joint frame on the parent side.
    pub shift02: Vec3,
    pub _pad1: u32,
    /// Vector (world frame) from the joint frame on the child side to this link's COM.
    pub shift23: Vec3,
    pub _pad2: u32,
    /// World-space spatial velocity added by this joint (rapier's `link.joint_velocity`).
    pub joint_velocity: Velocity,
    /// World-space total rigid-body velocity (rapier's `rb.vels`). Used by the
    /// Coriolis / gyroscopic assembly. Computed by `gpu_mb_update_velocities`.
    pub rb_vels: Velocity,
    /// Per-link kinematic acceleration (rapier's `workspace.accs[i]`).
    /// Populated by the Coriolis  variant of `apply_gravity`.
    pub kinematic_acc: Velocity,
}

/// One unit (1-DOF) constraint generated from a multibody joint's limit or
/// motor, exactly mirroring rapier's `unit_joint_*_constraint` output.
///
/// Each constraint targets a single generalized DOF. The "second jacobian row"
/// — the column of `M⁻¹` corresponding to that DOF — lives in a separate flat
/// buffer (`joint_constraint_columns`) so that the solver can update all DOFs of
/// the multibody when applying an impulse.
///
/// `kind` values: 0 = inactive, 1 = limit, 2 = motor.
#[derive(Clone, Copy, Default)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct MultibodyJointConstraint { // TODO: rename to MultibodyUnitJointConstraint?
    /// Index of the constrained DOF, relative to the multibody's `first_dof`.
    pub dof_id: u32,
    /// 0 = inactive (skipped by the solver), 1 = limit, 2 = motor.
    pub kind: u32,
    /// Constraint kind extras (for future extensions). Currently, always 0.
    pub _kind_extra: u32,
    pub _pad0: u32,

    /// `J·v` reference + bias velocity (rapier's `rhs`, includes positional bias).
    pub rhs: f32,
    /// Same as `rhs` minus the positional bias (rapier's `rhs_wo_bias`); used by
    /// the post-substep "remove bias" pass.
    pub rhs_wo_bias: f32,
    /// `1 / (Jᵀ·M⁻¹·J) = 1 / M⁻¹[d, d]`.
    pub inv_lhs: f32,
    /// Accumulated impulse (warmstart-able across substeps).
    pub impulse: f32,

    /// Lower / upper bounds for the impulse clamping.
    pub impulse_lo: f32,
    pub impulse_hi: f32,
    /// Constraint-force-mixing coefficients: `cfm_coeff` is `1 / (1 + cfm_coeff)`
    /// as a multiplier on Δimpulse; `cfm_gain` is subtracted from the rhs.
    /// Matches rapier's `cfm_coeff` / `cfm_gain` fields.
    pub cfm_coeff: f32,
    pub cfm_gain: f32,
}

/// One normal-direction contact constraint between a free rigid body and a
/// link of a multibody.
///
/// Mirrors rapier's pattern of "generic" two-body constraints — one side is a
/// regular rigid body (impulse applied via inv_mass / inv_inertia), the other
/// is a multibody whose impulse is propagated through `M⁻¹ · Jᵀ` (stored as a
/// per-constraint column in `contact_constraint_columns`).
///
/// TODO: handle friction and contact between two multibodies
#[derive(Clone, Copy, Default)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct MultibodyContactConstraint {
    /// Multibody index within the batch.
    pub multibody_id: u32,
    /// Link index within `multibody_id`.
    pub link_id: u32,
    /// 0 = inactive (skipped), 1 = active normal constraint.
    pub kind: u32,
    /// Local body id (in the shared body buffers) of the free-body side.
    pub free_body_id: u32,

    /// Free body's effective inverse mass (scalar — assumes isotropic mass).
    /// Zero for static bodies.
    pub free_body_im: f32,
    pub _pad0: [u32; 3],

    /// Free-body linear jacobian: `+normal` on body B's side or `-normal`
    /// on body A's side, depending on which side of the contact pair is
    /// the multibody.
    pub lin_jac: Vec3,
    pub _pad1: u32,
    /// Free-body angular jacobian (`r_free × jac_dir`).
    pub ang_jac: Vec3,
    pub _pad2: u32,
    /// Same as `ang_jac` but pre-multiplied by the free body's
    /// `effective_world_inv_inertia`. Used to update `solver_vels.angular`
    /// without re-multiplying every PGS sweep.
    pub ii_ang_jac: Vec3,
    pub _pad3: u32,

    /// `1 / (J · M⁻¹ · Jᵀ)`.
    pub inv_lhs: f32,
    /// `J·v_target + bias` — bias from penetration (`erp_inv_dt · depth`).
    pub rhs: f32,
    /// `rhs` without the positional bias (used by the stabilization sweep).
    pub rhs_wo_bias: f32,
    /// Accumulated impulse (warmstart-able).
    pub impulse: f32,

    /// CFM coefficients (matches rapier's `cfm_coeff` / `cfm_gain`).
    pub cfm_coeff: f32,
    pub cfm_gain: f32,
    pub _pad4: [u32; 2],
}

/// Descriptor for one multibody: where its links live, how many DOFs it has, and
/// the offsets into the dense jacobian/mass-matrix/gen-force tensors.
#[derive(Clone, Copy, Default)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct MultibodyInfo {
    /// First link index (relative to this batch's link slice).
    pub first_link: u32,
    /// Number of links in the multibody.
    pub num_links: u32,
    /// First DOF offset (relative to this batch's DOF slice).
    pub first_dof: u32,
    /// Total DOFs (sum of each link's `ndofs`).
    pub ndofs: u32,
    /// Offset (in f32 entries) into the `body_jacobians` tensor; each link has
    /// `6 * ndofs` contiguous entries, stacked link-by-link in assembly order.
    pub jacobian_offset: u32,
    /// Offset (in f32 entries) into the `mass_matrices` tensor. Block size: `ndofs * ndofs`.
    pub mass_matrix_offset: u32,
    /// 0 if the root joint is fixed, 1 if it's a free 6-DOF joint.
    pub root_is_dynamic: u32,
    /// Offset (in f32 entries) into `coriolis_v` / `coriolis_w`. Each link has
    /// `3 * ndofs` contiguous entries, stacked link-by-link in assembly order.
    pub coriolis_offset: u32,
    /// Offset (in f32 entries) into `i_coriolis_dt`. One 6×ndofs scratch slot
    /// per multibody (transient — overwritten per link during assembly).
    pub i_coriolis_dt_offset: u32,
    /// First constraint index for this multibody in the `joint_constraints`
    /// buffer. Each multibody owns `max_constraints` contiguous slots; the
    /// init kernel marks unused slots with `kind = 0`.
    pub first_constraint: u32,
    /// Maximum constraints this multibody can hold (sum over its joints of
    /// `2 * num_free_axes`). Slots beyond this are not touched.
    pub max_constraints: u32,
}
