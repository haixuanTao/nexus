//! Reduced-coordinates articulated multibody (3D).
//!
//! GPU port of rapier's `Multibody`, restricted to:
//! - Forward kinematics (link world transforms from generalized coordinates).
//! - Body jacobians (one `6 × ndofs` per link).
//! - Augmented mass matrix via CRBA: `M = Σ Jᵢᵀ Mᵢ Jᵢ`.
//! - Gravity generalized force: `τ = Σ Jᵢᵀ (mᵢ g, 0)`.
//! - In-place LU solve of `M ẍ = τ`.
//!
//! No constraints, contacts, or Coriolis terms (not in scope).
//!
//! ### Memory layout
//!
//! Links and multibodies are stored flat across all simulation batches. Each batch
//! has a capacity; unused slots are padded out and skipped via per-batch length
//! counts (mirrors the impulse-joint infrastructure).
//!
//! - `links_static: Tensor<MultibodyLinkStatic>`: constant per-link config.
//! - `links_workspace: Tensor<MultibodyLinkWorkspace>`: per-step scratch (pose, shifts).
//! - `multibody_info: Tensor<MultibodyInfo>`: offsets/sizes per multibody.
//! - `dof_values: Tensor<f32>`: generalized coordinates (flat, ndofs per multibody).
//! - `dof_velocities: Tensor<f32>`: generalized velocities.
//! - `gen_forces: Tensor<f32>`: generalized forces (receives gravity).
//! - `body_jacobians: Tensor<f32>`: per-link `6 × ndofs` column-major.
//! - `mass_matrices: Tensor<f32>`: per-multibody `ndofs × ndofs` column-major.
//!
//! ### Kernel topology
//!
//! Forward kinematics, jacobian assembly, mass-matrix assembly, and LU are
//! inherently sequential within a single multibody (parent before child, or
//! i-th elimination step before (i+1)-th). They run as `threads(1)` with one
//! workgroup per multibody. Links are independent across multibodies so the
//! batch × multibody grid parallelises fine.

#![cfg(feature = "dim3")]

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

use glamx::{Mat3, Quat, Vec3};

use crate::dynamics::body::{LocalMassProperties, Velocity, WorldMassProperties};
use crate::dynamics::joint::{ANG_AXES_MASK, GenericJoint, LIN_AXES_MASK};
use crate::queries::IndexedManifold;
use crate::rotation_to_matrix;
use crate::utils::linalg::{
    MAX_MB_DOFS, MatSlice, copy_from, fill, gemm_mat3_lhs, gemm_tr, gemv_tr_spatial, lu_decompose,
    lu_solve_in_place, quadform_spatial, skew, skew_tr,
};
use crate::utils::{Slice, SliceMut};
use crate::{Pose, rotation_from_scaled_axis, rotation_renormalize_fast};

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
/// Written once at init time. Kept in its own struct (separate from the workspace)
/// so the layout is clean — `GenericJoint` has 16-byte alignment and lumping
/// everything into one struct forced awkward padding on top of `Pose`.
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct MultibodyLinkStatic {
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
    /// Per-link kinematic acceleration (rapier's `workspace.accs[i]`). Equations
    /// 42–45 of Featherstone-style multibody dynamics. Populated by the Coriolis
    /// variant of `apply_gravity`.
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
pub struct MultibodyJointConstraint {
    /// Index of the constrained DOF, relative to the multibody's `first_dof`.
    pub dof_id: u32,
    /// 0 = inactive (skipped by the solver), 1 = limit, 2 = motor.
    pub kind: u32,
    /// Constraint kind extras (for future extensions). Currently always 0.
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
/// Friction is **not** modelled in this revision (normal force only); two-
/// multibody contacts are also not handled (one side is always a free body).
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

//
// Math helpers.
//

/// i-th cartesian basis vector. Branches on value to avoid SPIR-V pointer phis.
#[inline]
pub fn basis_vec3(i: u32) -> Vec3 {
    if i == 0 {
        Vec3::X
    } else if i == 1 {
        Vec3::Y
    } else {
        Vec3::Z
    }
}

/// Read index `i` (0..=5) of a `[f32; MAX_JOINT_DOFS]` by value.
#[inline]
pub fn coord_get(arr: &[f32; MAX_JOINT_DOFS], i: u32) -> f32 {
    if i == 0 {
        arr[0]
    } else if i == 1 {
        arr[1]
    } else if i == 2 {
        arr[2]
    } else if i == 3 {
        arr[3]
    } else if i == 4 {
        arr[4]
    } else {
        arr[5]
    }
}

/// Write index `i` (0..=5) of a `[f32; MAX_JOINT_DOFS]`.
#[inline]
pub fn coord_set(arr: &mut [f32; MAX_JOINT_DOFS], i: u32, v: f32) {
    if i == 0 {
        arr[0] = v;
    } else if i == 1 {
        arr[1] = v;
    } else if i == 2 {
        arr[2] = v;
    } else if i == 3 {
        arr[3] = v;
    } else if i == 4 {
        arr[4] = v;
    } else {
        arr[5] = v;
    }
}

/// Number of free DOFs implied by a `locked_axes` bitmask.
#[inline]
pub fn count_free_dofs(locked: u32) -> u32 {
    6 - (locked & 0x3f).count_ones()
}

/// Number of free linear DOFs (bits 0..3).
#[inline]
pub fn count_free_lin_dofs(locked: u32) -> u32 {
    3 - (locked & LIN_AXES_MASK).count_ones()
}

/// Number of free angular DOFs (bits 3..6).
#[inline]
pub fn count_free_ang_dofs(locked: u32) -> u32 {
    3 - ((locked & ANG_AXES_MASK) >> 3).count_ones()
}

/// Compute the link's `local_to_parent` pose given its current joint coords/rotation.
///
/// Mirrors rapier's `MultibodyJoint::body_to_parent`: starts from `joint_rot * local_frame_b⁻¹`,
/// prepends a translation for each free linear DOF, and finally composes with `local_frame_a`.
pub fn body_to_parent(stat: &MultibodyLinkStatic, ws: &MultibodyLinkWorkspace) -> Pose {
    let locked = stat.data.locked_axes;
    let mut transform = Pose::from_parts(Vec3::ZERO, ws.joint_rot) * stat.data.local_frame_b.inverse();

    for i in 0u32..3 {
        if (locked & (1 << i)) == 0 {
            let t = basis_vec3(i) * coord_get(&ws.coords, i);
            transform = Pose::from_parts(t, Quat::IDENTITY) * transform;
        }
    }

    stat.data.local_frame_a * transform
}

//
// Kernels.
//

/// Forward kinematics: one workgroup per multibody, links walked sequentially.
///
/// Writes `local_to_parent`, `local_to_world`, `shift02`, `shift23` into the workspace,
/// and publishes the link's world pose to the shared `poses` buffer for downstream
/// consumption (e.g. mprops update, collision).
#[spirv_bindgen]
// TODO(PERF): if we restricted all batches to have the same multibody topologies,
//             we could have multiple threads per workgroup working on these multibodies?
//             compute(threads(1, 64, 1)) ?
#[spirv(compute(threads(1)))]
pub fn gpu_mb_forward_kinematics(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    links_workspace: &mut [MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] poses: &mut [Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] links_mprops: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 6)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 7)] links_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 8)] colliders_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx_in_batch = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx_in_batch >= num_mb {
        return;
    }

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let links_start = batch_id * *links_batch_capacity as usize;
    let coll_start = batch_id * *colliders_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx_in_batch as usize);
    let num_links = mb.num_links;
    let first_link_global = links_start + mb.first_link as usize;

    let stat_slice = Slice(links_static, first_link_global);
    let mut ws_slice = SliceMut(links_workspace, first_link_global);
    let mprops_slice = Slice(links_mprops, first_link_global);
    let mut poses_slice = SliceMut(poses, coll_start);

    // Special case for the root, which has no parent.
    {
        let root_pose = poses_slice.read(stat_slice.at(0).rb_id as usize);
        let ws = ws_slice.at_mut(0);
        ws.local_to_parent = root_pose;
        ws.local_to_world = root_pose;
    }

    for k in 1..num_links {
        let k_usize = k as usize;
        let stat = stat_slice.read(k_usize);
        let mut ws = ws_slice.read(k_usize);

        let parent_to_world = ws_slice.read(stat.parent_link_id as usize).local_to_world;
        let local_to_parent = body_to_parent(&stat, &ws);
        let local_to_world = parent_to_world * local_to_parent;
        ws.local_to_parent = local_to_parent;
        ws.local_to_world = local_to_world;

        let parent_ws = ws_slice.read(stat.parent_link_id as usize);
        let parent_lmp = mprops_slice.read(stat.parent_link_id as usize);
        let lmp = mprops_slice.read(k_usize);
        let world_com = local_to_world * lmp.com;
        let parent_com_world = parent_ws.local_to_world * parent_lmp.com;
        let child_anchor_world = local_to_world * stat.data.local_frame_b.translation;
        ws.shift02 = child_anchor_world - parent_com_world;
        ws.shift23 = world_com - child_anchor_world;

        ws_slice.write(k_usize, ws);
        poses_slice.write(stat.rb_id as usize, local_to_world);
    }
}

//
// Body jacobian kernel.
//
// For each link i, the 6×ndofs body jacobian J_i maps generalized velocities
// to the link's world-frame spatial velocity at its COM. It is built recursively:
//
//   1. J_i := J_parent
//   2. J_i.linear_rows += [shift02]×ᵀ · J_parent.ang_rows
//   3. J_i columns for this joint's DOFs += joint jacobian (in world frame)
//   4. J_i.linear_rows += [shift23]×ᵀ · J_i.ang_rows
//
// Storage: column-major. `J[row, col] = jacobians[jac_base + col * 6 + row]`
// with `jac_base = mb.jacobian_offset + (k - first_link) * 6 * ndofs`.

/// Writes this joint's jacobian (world-frame) into the first `ndofs` columns of
/// an inline 6×6 scratch `out`, mirroring rapier's `MultibodyJoint::jacobian`.
///
/// `transform_rot` maps body-local axes (of the parent's `local_frame_a`) to world.
#[inline]
fn joint_jacobian(
    stat: &MultibodyLinkStatic,
    transform_rot: Quat,
    out: &mut [f32; 36],
    view: MatSlice,
) {
    let locked = stat.data.locked_axes;
    let mut curr_free_dof = 0u32;

    // Linear DOFs (axis order).
    for i in 0u32..3 {
        if (locked & (1 << i)) == 0 {
            let axis = transform_rot * basis_vec3(i);
            out[view.idx(0, curr_free_dof)] = axis.x;
            out[view.idx(1, curr_free_dof)] = axis.y;
            out[view.idx(2, curr_free_dof)] = axis.z;
            out[view.idx(3, curr_free_dof)] = 0.0;
            out[view.idx(4, curr_free_dof)] = 0.0;
            out[view.idx(5, curr_free_dof)] = 0.0;
            curr_free_dof += 1;
        }
    }

    // Angular DOFs.
    let ang_locked = (locked >> 3) & 0x7;
    let num_ang = 3 - ang_locked.count_ones();
    if num_ang == 1 {
        let dof_id = (!ang_locked & 0x7).trailing_zeros();
        let axis = transform_rot * basis_vec3(dof_id);
        out[view.idx(0, curr_free_dof)] = 0.0;
        out[view.idx(1, curr_free_dof)] = 0.0;
        out[view.idx(2, curr_free_dof)] = 0.0;
        out[view.idx(3, curr_free_dof)] = axis.x;
        out[view.idx(4, curr_free_dof)] = axis.y;
        out[view.idx(5, curr_free_dof)] = axis.z;
    } else if num_ang == 3 {
        for k in 0u32..3 {
            let axis = transform_rot * basis_vec3(k);
            out[view.idx(0, curr_free_dof + k)] = 0.0;
            out[view.idx(1, curr_free_dof + k)] = 0.0;
            out[view.idx(2, curr_free_dof + k)] = 0.0;
            out[view.idx(3, curr_free_dof + k)] = axis.x;
            out[view.idx(4, curr_free_dof + k)] = axis.y;
            out[view.idx(5, curr_free_dof + k)] = axis.z;
        }
    }
}

/// Build per-link body jacobians. Mirrors rapier's `Multibody::update_body_jacobians`
/// nearly line-for-line, using `MatSlice` views + BLAS-style primitives in place of
/// nalgebra's matrix API.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_body_jacobians(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] links_workspace: &[MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] body_jacobians: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] links_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 7)] jacobians_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let links_start = batch_id * *links_batch_capacity as usize;
    let jac_start = batch_id * *jacobians_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let num_links = mb.num_links;
    let ndofs = mb.ndofs;
    let first_link_global = links_start + mb.first_link as usize;
    let mb_jac_base = jac_start + mb.jacobian_offset as usize;

    let stat_slice = Slice(links_static, first_link_global);
    let ws_slice = Slice(links_workspace, first_link_global);

    // Zero all per-link jacobian blocks of this multibody.
    let mb_block = MatSlice::dense(mb_jac_base, 6 * num_links, ndofs);
    fill(body_jacobians, mb_block, 0.0);

    for k in 0..num_links {
        let stat = stat_slice.read(k as usize);
        let ws = ws_slice.read(k as usize);

        // View for this link's body jacobian (6 × ndofs, dense).
        let link_j = MatSlice::dense(mb_jac_base + (k as usize) * 6 * (ndofs as usize), 6, ndofs);

        let parent_to_world;
        if k != 0 {
            let parent_j = MatSlice::dense(
                mb_jac_base + (stat.parent_link_id as usize) * 6 * (ndofs as usize),
                6,
                ndofs,
            );
            let parent_ws = ws_slice.read(stat.parent_link_id as usize);
            parent_to_world = parent_ws.local_to_world;

            // link_j := parent_j
            copy_from(body_jacobians, link_j, parent_j);

            // link_j_v += [shift02]^T_× · parent_j_w
            let link_j_v = link_j.fixed_rows(0, 3);
            let parent_j_w = parent_j.fixed_rows(3, 3);
            let shift_tr = skew_tr(ws.shift02);
            gemm_mat3_lhs(body_jacobians, link_j_v, 1.0, shift_tr, parent_j_w, 1.0);
        } else {
            parent_to_world = Pose::default();
        }

        // Fill the joint jacobian into a 6×6 stack scratch, then splat its first
        // `ndofs_link` columns into link_j's `[assembly_id .. assembly_id + ndofs_link]`.
        let mut tmp = [0.0f32; 36];
        let tmp_view = MatSlice::dense(0, 6, 6);
        let joint_j = tmp_view.columns(0, stat.ndofs);
        joint_jacobian(
            &stat,
            parent_to_world.rotation * stat.data.local_frame_a.rotation,
            &mut tmp,
            joint_j,
        );
        // link_j_part += joint_j  (axpy with a stack-allocated RHS; rust-gpu can't
        // coerce `&[f32; 36]` to `&[f32]`, so this is expanded inline here).
        let link_j_part = link_j.columns(stat.assembly_id, stat.ndofs);
        for c in 0..stat.ndofs {
            for r in 0u32..6 {
                let idx = link_j_part.idx(r, c);
                let cur = body_jacobians.read(idx);
                body_jacobians.write(idx, cur + tmp[joint_j.idx(r, c)]);
            }
        }

        // link_j_v += [shift23]^T_× · link_j_w  (self-shift).
        let (link_j_v, link_j_w) = link_j.rows_range_pair(0, 3, 3, 3);
        let shift_tr = skew_tr(ws.shift23);
        gemm_mat3_lhs(body_jacobians, link_j_v, 1.0, shift_tr, link_j_w, 1.0);
    }
}

//
// Mass matrix (CRBA).
//
// Rapier:
//     self.augmented_mass.quadform(1.0, &rb_mass_matrix_wo_gyro, body_jacobian, 1.0);
//
// Here we use `quadform_spatial` which exploits the block-diagonal structure of the
// per-link 6×6 spatial mass (`diag(m·I₃, I_world)`) to avoid forming the full 6×6.
// World-space inertia is recomputed from the link's current orientation.
//
// Coriolis/gyroscopic terms are not computed (out of scope for this pass).

/// World-space 3×3 inertia for this link:
/// `I_world = R · diag(principal_inertia) · Rᵀ` with `R = world_rot · inertia_ref_frame`.
#[inline]
fn link_world_inertia(ws: &MultibodyLinkWorkspace, lmp: &LocalMassProperties) -> Mat3 {
    let ipi = lmp.inv_principal_inertia;
    let px = if ipi.x != 0.0 { 1.0 / ipi.x } else { 0.0 };
    let py = if ipi.y != 0.0 { 1.0 / ipi.y } else { 0.0 };
    let pz = if ipi.z != 0.0 { 1.0 / ipi.z } else { 0.0 };
    let r = rotation_to_matrix(ws.local_to_world.rotation * lmp.inertia_ref_frame);
    // M = r · diag(px, py, pz) (column-scale); I = M · rᵀ.
    let m0 = r.x_axis * px;
    let m1 = r.y_axis * py;
    let m2 = r.z_axis * pz;
    Mat3::from_cols(
        Vec3::new(
            m0.x * r.x_axis.x + m1.x * r.y_axis.x + m2.x * r.z_axis.x,
            m0.y * r.x_axis.x + m1.y * r.y_axis.x + m2.y * r.z_axis.x,
            m0.z * r.x_axis.x + m1.z * r.y_axis.x + m2.z * r.z_axis.x,
        ),
        Vec3::new(
            m0.x * r.x_axis.y + m1.x * r.y_axis.y + m2.x * r.z_axis.y,
            m0.y * r.x_axis.y + m1.y * r.y_axis.y + m2.y * r.z_axis.y,
            m0.z * r.x_axis.y + m1.z * r.y_axis.y + m2.z * r.z_axis.y,
        ),
        Vec3::new(
            m0.x * r.x_axis.z + m1.x * r.y_axis.z + m2.x * r.z_axis.z,
            m0.y * r.x_axis.z + m1.y * r.y_axis.z + m2.y * r.z_axis.z,
            m0.z * r.x_axis.z + m1.z * r.y_axis.z + m2.z * r.z_axis.z,
        ),
    )
}

/// Assemble the augmented mass matrix `M = Σᵢ Jᵢᵀ · diag(mᵢ·I₃, Iᵢ_world) · Jᵢ`.
///
/// Damping is added to the diagonal (`M[i, i] += damping[i] * dt`), matching
/// rapier's trailing loop in `update_inertias`.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_mass_matrix(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] links_workspace: &[MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] links_mprops: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] body_jacobians: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] mass_matrices: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] damping: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] num_multibodies: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] dt_buf: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 9)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 10)] links_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 11)] jacobians_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 12)] mass_matrix_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 13)] dof_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }

    let dt = dt_buf.read(0);
    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let links_start = batch_id * *links_batch_capacity as usize;
    let jac_start = batch_id * *jacobians_batch_capacity as usize;
    let mm_start = batch_id * *mass_matrix_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let num_links = mb.num_links;
    let ndofs = mb.ndofs;
    let first_link_global = links_start + mb.first_link as usize;
    let mb_jac_base = jac_start + mb.jacobian_offset as usize;
    let mb_mm_base = mm_start + mb.mass_matrix_offset as usize;
    let damping_base = dof_start + mb.first_dof as usize;

    let ws_slice = Slice(links_workspace, first_link_global);
    let mprops_slice = Slice(links_mprops, first_link_global);
    let damping_slice = Slice(damping, damping_base);
    let _ = links_static; // reserved for future use (kinematic-DOF permutation, etc.)

    // augmented_mass.fill(0.0)
    let augmented_mass = MatSlice::dense(mb_mm_base, ndofs, ndofs);
    fill(mass_matrices, augmented_mass, 0.0);

    for k in 0..num_links {
        let ws = ws_slice.read(k as usize);
        let lmp = mprops_slice.read(k as usize);

        let inv_mass_x = lmp.inv_mass.x;
        if inv_mass_x == 0.0 {
            continue;
        }
        let mass = 1.0 / inv_mass_x;
        let inertia = link_world_inertia(&ws, &lmp);

        // body_jacobian view for this link.
        let body_jacobian = MatSlice::dense(
            mb_jac_base + (k as usize) * 6 * (ndofs as usize),
            6,
            ndofs,
        );

        // augmented_mass.quadform(1.0, &rb_spatial_mass, body_jacobian, 1.0);
        quadform_spatial(
            mass_matrices,
            augmented_mass,
            1.0,
            mass,
            inertia,
            body_jacobians,
            body_jacobian,
            1.0,
        );
    }

    // Per-rapier: `augmented_mass[i, i] += damping[i] * dt`.
    for i in 0..ndofs {
        let diag_idx = augmented_mass.idx(i, i);
        let cur = mass_matrices.read(diag_idx);
        mass_matrices.write(diag_idx, cur + damping_slice.read(i as usize) * dt);
    }

    // Defensive cap so ndofs can't overflow the quadform scratch.
    let _ = MAX_MB_DOFS;
}

//
// Mass matrix with Coriolis + gyroscopic terms.
//
// Mirrors the full `Multibody::update_inertias` algorithm from rapier:
//   1. Per link, compute the gyroscopically-augmented inertia
//        I_aug = I + ([ω]_× · I − [Iω]_×) · dt
//      and accumulate `acc_augmented_mass += Jᵢᵀ · diag(mᵢ·I₃, I_aug) · Jᵢ`.
//   2. Build per-link `coriolis_v[i]` (3×ndofs) and `coriolis_w[i]` (3×ndofs)
//      recursively from the parent, using all of shift02, parent ω, joint velocity.
//   3. Add the self-shift contribution (shift23 + own ω).
//   4. Meld into `acc_augmented_mass` via `i_coriolis_dt` scratch:
//        i_coriolis_dt_v = dt · mass · coriolis_v
//        i_coriolis_dt_w = dt · I · coriolis_w
//        acc_augmented_mass += Jᵀ · i_coriolis_dt
//
// Requires `gpu_mb_update_velocities` to have been run first so that
// `ws.joint_velocity` and `ws.rb_vels` hold the current per-link world velocities.

/// Scale each column of `dst_v` (3 × ndofs) by a scalar, in place: `dst_v := scale · src_v`.
#[inline]
fn scaled_copy_3xn(
    buf_dst: &mut [f32],
    dst: MatSlice,
    scale: f32,
    buf_src: &[f32],
    src: MatSlice,
) {
    for c in 0..dst.cols {
        for r in 0u32..3 {
            buf_dst.write(dst.idx(r, c), scale * buf_src.read(src.idx(r, c)));
        }
    }
}

#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_mass_matrix_with_coriolis(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] links_workspace: &[MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] links_mprops: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] body_jacobians: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] mass_matrices: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] coriolis_v: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] coriolis_w: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] i_coriolis_dt: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] damping: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 10)] num_multibodies: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 11)] dt_buf: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 12)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 13)] links_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 14)] jacobians_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 15)] mass_matrix_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 16)] coriolis_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 17)] i_coriolis_dt_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 18)] dof_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }
    let dt = dt_buf.read(0);

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let links_start = batch_id * *links_batch_capacity as usize;
    let jac_start = batch_id * *jacobians_batch_capacity as usize;
    let mm_start = batch_id * *mass_matrix_batch_capacity as usize;
    let cor_start = batch_id * *coriolis_batch_capacity as usize;
    let icdt_start = batch_id * *i_coriolis_dt_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let num_links = mb.num_links;
    let ndofs = mb.ndofs;
    let first_link_global = links_start + mb.first_link as usize;
    let mb_jac_base = jac_start + mb.jacobian_offset as usize;
    let mb_mm_base = mm_start + mb.mass_matrix_offset as usize;
    let mb_cor_base = cor_start + mb.coriolis_offset as usize;
    let mb_icdt_base = icdt_start + mb.i_coriolis_dt_offset as usize;
    let damping_base = dof_start + mb.first_dof as usize;

    let stat_slice = Slice(links_static, first_link_global);
    let ws_slice = Slice(links_workspace, first_link_global);
    let mprops_slice = Slice(links_mprops, first_link_global);
    let damping_slice = Slice(damping, damping_base);

    // acc_augmented_mass.fill(0.0)
    let acc_augmented_mass = MatSlice::dense(mb_mm_base, ndofs, ndofs);
    fill(mass_matrices, acc_augmented_mass, 0.0);

    // i_coriolis_dt view (6 × ndofs, fully overwritten each link).
    let i_coriolis_dt_view = MatSlice::dense(mb_icdt_base, 6, ndofs);
    let i_coriolis_dt_v = i_coriolis_dt_view.fixed_rows(0, 3);
    let i_coriolis_dt_w = i_coriolis_dt_view.fixed_rows(3, 3);

    for k in 0..num_links {
        let stat = stat_slice.read(k as usize);
        let ws = ws_slice.read(k as usize);
        let lmp = mprops_slice.read(k as usize);

        let inv_mass_x = lmp.inv_mass.x;
        if inv_mass_x == 0.0 {
            // Still need to zero this link's coriolis block so children don't
            // propagate garbage.
            let coriolis_v_i =
                MatSlice::dense(mb_cor_base + (k as usize) * 3 * (ndofs as usize), 3, ndofs);
            let coriolis_w_i = coriolis_v_i; // same shape + location in the other buffer
            fill(coriolis_v, coriolis_v_i, 0.0);
            fill(coriolis_w, coriolis_w_i, 0.0);
            continue;
        }
        let mass = 1.0 / inv_mass_x;
        let rb_inertia = link_world_inertia(&ws, &lmp);

        let body_jacobian =
            MatSlice::dense(mb_jac_base + (k as usize) * 6 * (ndofs as usize), 6, ndofs);

        // Gyroscopic derivative: aug_I = I + ([ω]_× · I − [Iω]_×) · dt.
        let angvel = ws.rb_vels.angular;
        let w_skew = skew(angvel);
        let i_omega = rb_inertia.x_axis * angvel.x
            + rb_inertia.y_axis * angvel.y
            + rb_inertia.z_axis * angvel.z;
        let i_omega_skew = skew(i_omega);
        let w_skew_i = mat3_mul(w_skew, rb_inertia);
        let gyro_mat = Mat3::from_cols(
            w_skew_i.x_axis - i_omega_skew.x_axis,
            w_skew_i.y_axis - i_omega_skew.y_axis,
            w_skew_i.z_axis - i_omega_skew.z_axis,
        );
        let augmented_inertia = Mat3::from_cols(
            rb_inertia.x_axis + gyro_mat.x_axis * dt,
            rb_inertia.y_axis + gyro_mat.y_axis * dt,
            rb_inertia.z_axis + gyro_mat.z_axis * dt,
        );

        // acc_augmented_mass.quadform(1.0, &concat_rb_mass_matrix(mass, augmented_inertia),
        //                             body_jacobian, 1.0);
        quadform_spatial(
            mass_matrices,
            acc_augmented_mass,
            1.0,
            mass,
            augmented_inertia,
            body_jacobians,
            body_jacobian,
            1.0,
        );

        // Coriolis matrix assembly.
        let rb_j_w = body_jacobian.fixed_rows(3, 3);
        let coriolis_v_i =
            MatSlice::dense(mb_cor_base + (k as usize) * 3 * (ndofs as usize), 3, ndofs);
        let coriolis_w_i = coriolis_v_i; // views are structurally identical in the two buffers.

        if k != 0 {
            let parent_id = stat.parent_link_id;
            let parent_ws = ws_slice.read(parent_id as usize);
            let parent_j = MatSlice::dense(
                mb_jac_base + (parent_id as usize) * 6 * (ndofs as usize),
                6,
                ndofs,
            );
            let parent_j_w = parent_j.fixed_rows(3, 3);
            let parent_coriolis_v = MatSlice::dense(
                mb_cor_base + (parent_id as usize) * 3 * (ndofs as usize),
                3,
                ndofs,
            );
            let parent_coriolis_w = parent_coriolis_v;
            let parent_w = skew(parent_ws.rb_vels.angular);

            // coriolis_v.copy_from(parent_coriolis_v);
            // coriolis_w.copy_from(parent_coriolis_w);
            copy_from(coriolis_v, coriolis_v_i, parent_coriolis_v);
            copy_from(coriolis_w, coriolis_w_i, parent_coriolis_w);

            // coriolis_v += [shift02]^T_× · parent_coriolis_w.
            // (parent_coriolis_w lives in `coriolis_w`, not in `coriolis_v`, hence the
            // cross-buffer variant.)
            let shift_cross_tr_02 = skew_tr(ws.shift02);
            gemm_mat3_cross_buf(
                coriolis_v,
                coriolis_v_i,
                1.0,
                shift_cross_tr_02,
                coriolis_w,
                parent_coriolis_w,
                1.0,
            );

            // coriolis_v += dvel_cross^T · parent_j_w, with
            //   dvel = rb.vels.angvel × shift02 + 2 · joint_velocity.linvel.
            let dvel = ws.rb_vels.angular.cross(ws.shift02)
                + ws.joint_velocity.linear * 2.0;
            let dvel_cross_tr = skew_tr(dvel);
            gemm_mat3_cross_buf(
                coriolis_v,
                coriolis_v_i,
                1.0,
                dvel_cross_tr,
                body_jacobians,
                parent_j_w,
                1.0,
            );

            // coriolis_v += [joint_vel_lin]^T_× · parent_j_w.
            let jv_lin_cross_tr = skew_tr(ws.joint_velocity.linear);
            gemm_mat3_cross_buf(
                coriolis_v,
                coriolis_v_i,
                1.0,
                jv_lin_cross_tr,
                body_jacobians,
                parent_j_w,
                1.0,
            );

            // coriolis_v += (parent_w · shift02_cross_tr) · parent_j_w.
            let combined = mat3_mul(parent_w, shift_cross_tr_02);
            gemm_mat3_cross_buf(
                coriolis_v,
                coriolis_v_i,
                1.0,
                combined,
                body_jacobians,
                parent_j_w,
                1.0,
            );

            // coriolis_w += -[joint_vel_ang]_× · parent_j_w.
            let jv_ang_skew = skew(ws.joint_velocity.angular);
            gemm_mat3_cross_buf(
                coriolis_w,
                coriolis_w_i,
                -1.0,
                jv_ang_skew,
                body_jacobians,
                parent_j_w,
                1.0,
            );

            // Joint jacobian contribution to Coriolis (skipped for kinematic joints).
            if stat.kinematic == 0 {
                let mut tmp = [0.0f32; 36];
                let tmp_view = MatSlice::dense(0, 6, 6);
                let joint_j = tmp_view.columns(0, stat.ndofs);
                joint_jacobian(
                    &stat,
                    parent_ws.local_to_world.rotation * stat.data.local_frame_a.rotation,
                    &mut tmp,
                    joint_j,
                );
                // coriolis_v_part += 2 · parent_w · rb_joint_j_v.
                // coriolis_w_part += parent_w · rb_joint_j_w.
                // Both operands are stack slices of `tmp`, so we inline a column-major
                // `gemm_mat3_lhs` variant that reads `src` by index.
                let coriolis_v_part = coriolis_v_i.columns(stat.assembly_id, stat.ndofs);
                let coriolis_w_part = coriolis_w_i.columns(stat.assembly_id, stat.ndofs);
                for c in 0..stat.ndofs {
                    let jv = Vec3::new(
                        tmp[tmp_view.idx(0, c)],
                        tmp[tmp_view.idx(1, c)],
                        tmp[tmp_view.idx(2, c)],
                    );
                    let jw = Vec3::new(
                        tmp[tmp_view.idx(3, c)],
                        tmp[tmp_view.idx(4, c)],
                        tmp[tmp_view.idx(5, c)],
                    );
                    let pv = parent_w.x_axis * jv.x + parent_w.y_axis * jv.y + parent_w.z_axis * jv.z;
                    let pw = parent_w.x_axis * jw.x + parent_w.y_axis * jw.y + parent_w.z_axis * jw.z;
                    // coriolis_v_part[:, c] += 2.0 * pv
                    let iv0 = coriolis_v_part.idx(0, c);
                    let iv1 = coriolis_v_part.idx(1, c);
                    let iv2 = coriolis_v_part.idx(2, c);
                    coriolis_v.write(iv0, coriolis_v.read(iv0) + 2.0 * pv.x);
                    coriolis_v.write(iv1, coriolis_v.read(iv1) + 2.0 * pv.y);
                    coriolis_v.write(iv2, coriolis_v.read(iv2) + 2.0 * pv.z);
                    // coriolis_w_part[:, c] += pw
                    let iw0 = coriolis_w_part.idx(0, c);
                    let iw1 = coriolis_w_part.idx(1, c);
                    let iw2 = coriolis_w_part.idx(2, c);
                    coriolis_w.write(iw0, coriolis_w.read(iw0) + pw.x);
                    coriolis_w.write(iw1, coriolis_w.read(iw1) + pw.y);
                    coriolis_w.write(iw2, coriolis_w.read(iw2) + pw.z);
                }
            }
        } else {
            fill(coriolis_v, coriolis_v_i, 0.0);
            fill(coriolis_w, coriolis_w_i, 0.0);
        }

        // Self-shift contribution:
        //   coriolis_v += [shift23]^T_× · coriolis_w
        //   coriolis_v += [ω × shift23]^T_× · rb_j_w
        //   coriolis_v += (skew(ω) · [shift23]^T_×) · rb_j_w
        {
            let shift_cross_tr_23 = skew_tr(ws.shift23);
            gemm_mat3_cross_buf(
                coriolis_v,
                coriolis_v_i,
                1.0,
                shift_cross_tr_23,
                coriolis_w,
                coriolis_w_i,
                1.0,
            );

            let dvel_cross_tr_23 = skew_tr(ws.rb_vels.angular.cross(ws.shift23));
            gemm_mat3_cross_buf(
                coriolis_v,
                coriolis_v_i,
                1.0,
                dvel_cross_tr_23,
                body_jacobians,
                rb_j_w,
                1.0,
            );

            let combined_self = mat3_mul(skew(ws.rb_vels.angular), shift_cross_tr_23);
            gemm_mat3_cross_buf(
                coriolis_v,
                coriolis_v_i,
                1.0,
                combined_self,
                body_jacobians,
                rb_j_w,
                1.0,
            );
        }

        // Meld Coriolis into the mass matrix via i_coriolis_dt:
        //   i_coriolis_dt_v := dt · mass · coriolis_v
        //   i_coriolis_dt_w := dt · (rb_inertia · coriolis_w)
        //   acc_augmented_mass += Jᵀ · i_coriolis_dt.
        scaled_copy_3xn(
            i_coriolis_dt,
            i_coriolis_dt_v,
            mass * dt,
            coriolis_v,
            coriolis_v_i,
        );
        gemm_mat3_cross_buf(
            i_coriolis_dt,
            i_coriolis_dt_w,
            dt,
            rb_inertia,
            coriolis_w,
            coriolis_w_i,
            0.0,
        );
        gemm_tr(
            mass_matrices,
            acc_augmented_mass,
            1.0,
            body_jacobians,
            body_jacobian,
            i_coriolis_dt,
            i_coriolis_dt_view,
            1.0,
        );
    }

    // Per-rapier: `acc_augmented_mass[i, i] += damping[i] * dt`.
    for i in 0..ndofs {
        let diag_idx = acc_augmented_mass.idx(i, i);
        let cur = mass_matrices.read(diag_idx);
        mass_matrices.write(diag_idx, cur + damping_slice.read(i as usize) * dt);
    }

    let _ = MAX_MB_DOFS;
}

/// `c := beta * c + alpha * A_mat3 * b` where `A` is an inline 3×3 and `b`, `c`
/// live in *different* flat buffers. Same as `gemm_mat3_lhs` but with a second
/// buffer for the right-hand-side view.
#[inline]
fn gemm_mat3_cross_buf(
    buf_c: &mut [f32],
    c: MatSlice,
    alpha: f32,
    a: Mat3,
    buf_b: &[f32],
    b: MatSlice,
    beta: f32,
) {
    for j in 0..c.cols {
        let bx = buf_b.read(b.idx(0, j));
        let by = buf_b.read(b.idx(1, j));
        let bz = buf_b.read(b.idx(2, j));
        let p = a.x_axis * bx + a.y_axis * by + a.z_axis * bz;
        let i0 = c.idx(0, j);
        let i1 = c.idx(1, j);
        let i2 = c.idx(2, j);
        buf_c.write(i0, beta * buf_c.read(i0) + alpha * p.x);
        buf_c.write(i1, beta * buf_c.read(i1) + alpha * p.y);
        buf_c.write(i2, beta * buf_c.read(i2) + alpha * p.z);
    }
}

//
// Velocity propagation (rapier's `update_dynamics` velocity phase).
//
// Computes per-link world-space `joint_velocity` and total `rb_vels` by walking
// links parent-before-child, so that the Coriolis assembly can read them.

/// Body-local velocity contributed by this joint, given the joint's free-DOF
/// velocities `vels` (rapier's `MultibodyJoint::jacobian_mul_coordinates`).
#[inline]
fn jacobian_mul_coordinates(
    locked_axes: u32,
    vels: [f32; MAX_JOINT_DOFS],
) -> (Vec3, Vec3) {
    let mut lin = Vec3::ZERO;
    let mut ang = Vec3::ZERO;
    let mut curr = 0u32;

    for i in 0u32..3 {
        if (locked_axes & (1 << i)) == 0 {
            lin += basis_vec3(i) * vels[curr as usize];
            curr += 1;
        }
    }

    let ang_locked = (locked_axes >> 3) & 0x7;
    let num_ang = 3 - ang_locked.count_ones();
    if num_ang == 1 {
        let dof_id = (!ang_locked & 0x7).trailing_zeros();
        ang += basis_vec3(dof_id) * vels[curr as usize];
    } else if num_ang == 3 {
        ang += Vec3::new(
            vels[curr as usize],
            vels[(curr + 1) as usize],
            vels[(curr + 2) as usize],
        );
    }
    (lin, ang)
}

/// Mat3 × Mat3 (column-major, `a.x_axis` is first column).
#[inline]
fn mat3_mul(a: Mat3, b: Mat3) -> Mat3 {
    Mat3::from_cols(
        a.x_axis * b.x_axis.x + a.y_axis * b.x_axis.y + a.z_axis * b.x_axis.z,
        a.x_axis * b.y_axis.x + a.y_axis * b.y_axis.y + a.z_axis * b.y_axis.z,
        a.x_axis * b.z_axis.x + a.y_axis * b.z_axis.y + a.z_axis * b.z_axis.z,
    )
}

/// Propagate link velocities parent-before-child. Mirrors rapier's:
///
/// ```text
///   let joint_velocity = link.joint.jacobian_mul_coordinates(&velocities[link.assembly_id..]);
///   link.joint_velocity = joint_velocity.transformed(
///       &(parent_link.local_to_world.rotation * link.joint.data.local_frame1.rotation));
///   let mut new_rb_vels = parent_rb.vels + link.joint_velocity;
///   new_rb_vels.linvel += parent_rb.vels.angvel.gcross(shift);
///   new_rb_vels.linvel += link.joint_velocity.angvel.gcross(link.shift23);
///   rb.vels = new_rb_vels;
/// ```
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_update_velocities(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    links_workspace: &mut [MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] links_mprops: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] dof_velocities: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 6)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 7)] links_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 8)] dof_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let links_start = batch_id * *links_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let num_links = mb.num_links;
    let first_link_global = links_start + mb.first_link as usize;
    let gen_base = dof_start + mb.first_dof as usize;

    let stat_slice = Slice(links_static, first_link_global);
    let mut ws_slice = SliceMut(links_workspace, first_link_global);
    let mprops_slice = Slice(links_mprops, first_link_global);
    let vel_slice = Slice(dof_velocities, gen_base);

    for k in 0..num_links {
        let k_usize = k as usize;
        let stat = stat_slice.read(k_usize);
        let mut ws = ws_slice.read(k_usize);

        // Gather this joint's free-DOF velocities from the flat tensor.
        let mut vels = [0.0f32; MAX_JOINT_DOFS];
        for d in 0..stat.ndofs {
            vels[d as usize] = vel_slice.read((stat.assembly_id + d) as usize);
        }
        let (jv_local_lin, jv_local_ang) =
            jacobian_mul_coordinates(stat.data.locked_axes, vels);

        if k == 0 {
            // Root: joint velocity already in world frame.
            ws.joint_velocity = Velocity::new(jv_local_lin, jv_local_ang);
            ws.rb_vels = ws.joint_velocity;
        } else {
            let parent_id = stat.parent_link_id as usize;
            let parent_ws = ws_slice.read(parent_id);
            let parent_lmp = mprops_slice.read(parent_id);
            let transform_rot =
                parent_ws.local_to_world.rotation * stat.data.local_frame_a.rotation;

            ws.joint_velocity.linear = transform_rot * jv_local_lin;
            ws.joint_velocity.angular = transform_rot * jv_local_ang;

            // new_rb_vels = parent_rb.vels + joint_velocity, then shift corrections.
            let mut new_lin = parent_ws.rb_vels.linear + ws.joint_velocity.linear;
            let new_ang = parent_ws.rb_vels.angular + ws.joint_velocity.angular;

            let lmp = mprops_slice.read(k_usize);
            let world_com = ws.local_to_world * lmp.com;
            let parent_world_com = parent_ws.local_to_world * parent_lmp.com;
            let shift = world_com - parent_world_com;

            new_lin += parent_ws.rb_vels.angular.cross(shift);
            new_lin += ws.joint_velocity.angular.cross(ws.shift23);

            ws.rb_vels = Velocity::new(new_lin, new_ang);
        }

        ws_slice.write(k_usize, ws);
    }
}

//
// Coriolis-aware generalized force assembly.
//
// Mirrors rapier's `update_acceleration` pre-solve logic (equations 42–45):
// per link we build a kinematic acceleration `acc` recursively from the parent's,
// then form the "external force" with the inertial / gyroscopic corrections:
//
//   acc[i] = acc[parent] + 2·parent_ω × joint_vel.linvel + parent_ω × joint_vel.angvel
//          + parent_ω × (parent_ω × shift02) + parent_α × shift02
//   acc[i].linvel += rb.ω × (rb.ω × shift23) + acc[i].angvel × shift23
//   gyroscopic     = rb.ω × (I · rb.ω)
//   f_ext_lin  = rb.F_lin - m · acc.linvel        (here rb.F_lin = m·g)
//   f_ext_ang  = rb.τ     - gyroscopic - I · acc.angvel
//   τ         += J_iᵀ · (f_ext_lin, f_ext_ang)
//
// Finally, `τ -= damping ⊙ velocities`, matching
//   `self.accelerations.cmpy(-1.0, &self.damping, &self.velocities, 1.0)`.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_apply_gravity_with_coriolis(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] links_workspace: &mut [MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] links_mprops: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] body_jacobians: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] gen_forces: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] dof_velocities: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] damping: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] num_multibodies: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] gravity: &[f32; 3],
    #[spirv(uniform, descriptor_set = 0, binding = 10)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 11)] links_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 12)] jacobians_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 13)] dof_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let links_start = batch_id * *links_batch_capacity as usize;
    let jac_start = batch_id * *jacobians_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let num_links = mb.num_links;
    let ndofs = mb.ndofs;
    let first_link_global = links_start + mb.first_link as usize;
    let mb_jac_base = jac_start + mb.jacobian_offset as usize;
    let gen_base = dof_start + mb.first_dof as usize;

    let stat_slice = Slice(links_static, first_link_global);
    let mut ws_slice = SliceMut(links_workspace, first_link_global);
    let mprops_slice = Slice(links_mprops, first_link_global);
    let vel_slice = Slice(dof_velocities, gen_base);
    let damping_slice = Slice(damping, gen_base);

    // accelerations.fill(0.0) for this multibody.
    let accelerations = MatSlice::dense(gen_base, ndofs, 1);
    fill(gen_forces, accelerations, 0.0);

    let gx = gravity[0];
    let gy = gravity[1];
    let gz = gravity[2];

    for k in 0..num_links {
        let stat = stat_slice.read(k as usize);
        let mut ws = ws_slice.read(k as usize);

        // Build kinematic acceleration `acc` (eqs 42–45).
        let mut acc_lin = Vec3::ZERO;
        let mut acc_ang = Vec3::ZERO;
        if k != 0 {
            let parent_ws = ws_slice.read(stat.parent_link_id as usize);
            let parent_acc = parent_ws.kinematic_acc;
            let parent_ang = parent_ws.rb_vels.angular;

            acc_lin = parent_acc.linear;
            acc_ang = parent_acc.angular;

            // 2 · parent_ω × joint_vel.linvel
            acc_lin += parent_ang.cross(ws.joint_velocity.linear) * 2.0;
            // parent_ω × joint_vel.angvel
            acc_ang += parent_ang.cross(ws.joint_velocity.angular);
            // parent_ω × (parent_ω × shift02)
            acc_lin += parent_ang.cross(parent_ang.cross(ws.shift02));
            // parent_α × shift02
            acc_lin += parent_acc.angular.cross(ws.shift02);
        }
        // Self-shift: rb.ω × (rb.ω × shift23), acc.ω × shift23.
        let rb_ang = ws.rb_vels.angular;
        acc_lin += rb_ang.cross(rb_ang.cross(ws.shift23));
        acc_lin += acc_ang.cross(ws.shift23);

        ws.kinematic_acc = Velocity::new(acc_lin, acc_ang);
        ws_slice.write(k as usize, ws);

        let lmp = mprops_slice.read(k as usize);
        let inv_mass_x = lmp.inv_mass.x;
        if inv_mass_x == 0.0 {
            continue;
        }
        let mass = 1.0 / inv_mass_x;
        let rb_inertia = link_world_inertia(&ws, &lmp);

        // rb.forces = (m·g, 0). Build `external_forces` per rapier.
        let gyroscopic = {
            let i_omega = rb_inertia.x_axis * rb_ang.x
                + rb_inertia.y_axis * rb_ang.y
                + rb_inertia.z_axis * rb_ang.z;
            rb_ang.cross(i_omega)
        };
        let i_acc_ang = rb_inertia.x_axis * acc_ang.x
            + rb_inertia.y_axis * acc_ang.y
            + rb_inertia.z_axis * acc_ang.z;
        let f_lin = Vec3::new(mass * gx, mass * gy, mass * gz) - acc_lin * mass;
        let f_ang = -gyroscopic - i_acc_ang;
        let external_forces = [f_lin.x, f_lin.y, f_lin.z, f_ang.x, f_ang.y, f_ang.z];

        let body_jacobian = MatSlice::dense(
            mb_jac_base + (k as usize) * 6 * (ndofs as usize),
            6,
            ndofs,
        );

        gemv_tr_spatial(
            gen_forces,
            gen_base,
            1.0,
            body_jacobians,
            body_jacobian,
            external_forces,
            1.0,
        );
    }

    // `accelerations.cmpy(-1.0, &damping, &velocities, 1.0)`.
    for i in 0..ndofs {
        let idx = gen_base + i as usize;
        let cur = gen_forces.read(idx);
        gen_forces.write(
            idx,
            cur - damping_slice.read(i as usize) * vel_slice.read(i as usize),
        );
    }
}

//
// LU decomposition + solve.
//
// Split into two kernels so the factorization can be reused across multiple
// right-hand sides within a frame (e.g. gravity τ, contact impulses, …) —
// mirrors nalgebra's `LU` / `LU::solve_mut` API.
//
// The augmented mass matrix from CRBA is symmetric positive definite, so
// pivoting is not strictly needed, but partial pivoting is still performed
// for robustness and parity with rapier.
//
// For simplicity the present implementation assumes no kinematic DOFs; all
// DOFs participate in the solve. Rapier excludes kinematic DOFs via a
// permutation — a follow-up can layer that on top of these primitives.

/// Factor `M` in-place into `P·L·U` and record the row pivots.
///
/// Input/output: `mass_matrices` holds the per-multibody mass matrix block on
/// entry and the packed LU factors on exit. `lu_pivots` receives one pivot index
/// per row per multibody. One workgroup per multibody.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_lu_decompose(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mass_matrices: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] lu_pivots: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 5)] mass_matrix_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] dof_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let mm_start = batch_id * *mass_matrix_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    if mb.ndofs == 0 {
        return;
    }
    let m = MatSlice::dense(mm_start + mb.mass_matrix_offset as usize, mb.ndofs, mb.ndofs);
    let piv_offset = dof_start + mb.first_dof as usize;

    lu_decompose(mass_matrices, m, lu_pivots, piv_offset);
}

/// Solve `M · x = rhs` in-place using the packed LU produced by
/// `gpu_mb_lu_decompose`. `rhs` is overwritten with `x`.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_lu_solve(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mass_matrices: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] lu_pivots: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] rhs: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] mass_matrix_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 7)] dof_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let mm_start = batch_id * *mass_matrix_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    if mb.ndofs == 0 {
        return;
    }
    let m = MatSlice::dense(mm_start + mb.mass_matrix_offset as usize, mb.ndofs, mb.ndofs);
    let piv_offset = dof_start + mb.first_dof as usize;
    let rhs_offset = piv_offset;

    lu_solve_in_place(mass_matrices, m, lu_pivots, piv_offset, rhs, rhs_offset);
}

//
// Integrate kernel.
//
// Semi-implicit Euler:
//   v += a * dt                           (a = generalized acceleration from solve)
//   coords, joint_rot updated per-link using `v`
//
// The angular-DOF update mirrors rapier's `MultibodyJoint::integrate`:
//   - 1 free angular DOF:  coords[DIM + dof_id] += v * dt; joint_rot from axis-angle.
//   - 3 free angular DOFs: joint_rot = exp(v * dt) * joint_rot; coords[3..6] += v * dt.
//   - 0 free angular DOFs: no-op.
//
// After this pass, `dof_velocities` and each link's `coords` / `joint_rot` are updated.
// Callers are expected to re-run forward kinematics to refresh link poses.

//
// Multibody joint limit / motor constraints.
//
// Mirrors rapier's `unit_joint_limit_constraint` + `unit_joint_motor_constraint`
// + the PGS solver. Each constraint targets a single generalized DOF `d`:
//
//   * jacobian = e_d (1 in slot d, 0 elsewhere)
//   * inv_lhs  = 1 / (e_dᵀ · M⁻¹ · e_d)
//   * column   = M⁻¹ · e_d         (full ndofs vector — used to update v)
//
// The solver iterates PGS sweeps:
//
//   rhs_total = J · v + self.rhs                   (= v[d] + bias)
//   new_imp   = clamp(impulse + inv_lhs * (rhs_total - cfm_gain * impulse), bounds)
//   Δimp      = new_imp - impulse
//   v         -= Δimp · column                     (subtract: rapier's sign convention)
//
// Per-multibody, all constraint slots are scanned (`kind == 0` ones are skipped).

/// Compute joint motor parameters mirroring rapier's `JointMotor::motor_params`.
#[inline]
fn motor_params(motor: &crate::dynamics::joint::JointMotor, dt: f32) -> (f32, f32, f32, f32, f32) {
    // Returns (erp_inv_dt, cfm_coeff, cfm_gain, target_vel_clamp_inv_dt, max_impulse).
    let inv_dt = if dt != 0.0 { 1.0 / dt } else { 0.0 };
    let mp = crate::dynamics::joint::motor_params(motor, dt);
    (
        mp.erp_inv_dt,
        mp.cfm_coeff,
        mp.cfm_gain,
        inv_dt,
        mp.max_impulse,
    )
}

/// Solve `M · x = e_d` in place (writes `x` into `dst[0..n]`). Uses the same
/// LU factor + pivots produced by `gpu_mb_lu_decompose`.
#[inline]
fn lu_solve_unit(
    buf_m: &[f32],
    m: MatSlice,
    buf_pivots: &[u32],
    pivots_offset: usize,
    dst: &mut [f32],
    dst_offset: usize,
    dof_id: u32,
) {
    let n = m.rows;
    // dst[0..n] := e_{dof_id}  (then permuted by lu_solve_in_place).
    for i in 0..n {
        dst[dst_offset + i as usize] = if i == dof_id { 1.0 } else { 0.0 };
    }
    lu_solve_in_place(buf_m, m, buf_pivots, pivots_offset, dst, dst_offset);
}

/// Initialize the multibody's joint-limit / joint-motor unit constraints.
///
/// For each link, scans every free DOF that has either `limit_axes` or `motor_axes`
/// set, and emits one `MultibodyJointConstraint` per active limit and one per
/// active motor (rapier emits these separately even when both are on the same axis).
///
/// Must run after `gpu_mb_lu_decompose` — the LU factors of `M` are used to compute
/// the per-constraint M⁻¹ column and effective inverse mass.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_init_joint_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] links_workspace: &[MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] mass_matrices: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] lu_pivots: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] joint_constraints: &mut [MultibodyJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] joint_constraint_columns: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] num_multibodies: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] dt_buf: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 9)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 10)] links_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 11)] mass_matrix_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 12)] dof_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 13)] joint_constraints_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 14)] joint_constraint_columns_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }
    let dt = dt_buf.read(0);

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let links_start = batch_id * *links_batch_capacity as usize;
    let mm_start = batch_id * *mass_matrix_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;
    let cons_start = batch_id * *joint_constraints_batch_capacity as usize;
    let col_start = batch_id * *joint_constraint_columns_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let num_links = mb.num_links;
    let ndofs = mb.ndofs;
    if ndofs == 0 {
        return;
    }
    let first_link_global = links_start + mb.first_link as usize;
    let mb_mm_base = mm_start + mb.mass_matrix_offset as usize;
    let piv_offset = dof_start + mb.first_dof as usize;
    let cons_base = cons_start + mb.first_constraint as usize;
    // One column of M⁻¹ per constraint slot, ndofs floats each.
    let col_base = col_start + (mb.first_constraint as usize) * (ndofs as usize);

    let stat_slice = Slice(links_static, first_link_global);
    let ws_slice = Slice(links_workspace, first_link_global);
    let m = MatSlice::dense(mb_mm_base, ndofs, ndofs);

    // Mark all slots as inactive; the loop below activates the live ones.
    for s in 0..mb.max_constraints {
        let mut cz: MultibodyJointConstraint = joint_constraints.read(cons_base + s as usize);
        cz.kind = 0;
        cz.impulse = 0.0;
        joint_constraints.write(cons_base + s as usize, cz);
    }

    let inv_dt = if dt != 0.0 { 1.0 / dt } else { 0.0 };

    let mut slot = 0u32;
    for k in 0..num_links {
        let stat = stat_slice.read(k as usize);
        let ws = ws_slice.read(k as usize);
        let locked = stat.data.locked_axes;
        let limit_axes = stat.data.limit_axes & !locked;
        let motor_axes = stat.data.motor_axes & !locked;
        if limit_axes == 0 && motor_axes == 0 {
            continue;
        }
        if stat.kinematic != 0 {
            continue;
        }

        // Walk free axes in DOF order, mirroring `MultibodyJoint::velocity_constraints`.
        // `curr_free_dof` tracks the position within this joint's slice of the
        // multibody's generalized-velocity vector; the absolute index is
        // `stat.assembly_id + curr_free_dof`.
        let mut curr_free_dof = 0u32;

        // Linear DOFs first.
        for axis in 0u32..3 {
            if (locked & (1 << axis)) != 0 {
                continue;
            }
            let abs_dof = stat.assembly_id + curr_free_dof;
            let curr_pos = coord_get(&ws.coords, axis);

            if (motor_axes & (1 << axis)) != 0 {
                let has_limits = (limit_axes & (1 << axis)) != 0;
                let limit_min = stat.data.limits[axis as usize].min;
                let limit_max = stat.data.limits[axis as usize].max;
                emit_motor_constraint(
                    joint_constraints,
                    joint_constraint_columns,
                    cons_base,
                    col_base,
                    slot,
                    abs_dof,
                    ndofs,
                    curr_pos,
                    inv_dt,
                    dt,
                    &stat.data.motors[axis as usize],
                    has_limits,
                    limit_min,
                    limit_max,
                    mass_matrices,
                    m,
                    lu_pivots,
                    piv_offset,
                );
                slot += 1;
            }
            if (limit_axes & (1 << axis)) != 0 {
                emit_limit_constraint(
                    joint_constraints,
                    joint_constraint_columns,
                    cons_base,
                    col_base,
                    slot,
                    abs_dof,
                    ndofs,
                    curr_pos,
                    [stat.data.limits[axis as usize].min, stat.data.limits[axis as usize].max],
                    dt,
                    mass_matrices,
                    m,
                    lu_pivots,
                    piv_offset,
                );
                slot += 1;
            }
            curr_free_dof += 1;
        }

        // Angular DOFs.
        for axis in 3u32..6 {
            if (locked & (1 << axis)) != 0 {
                continue;
            }
            let abs_dof = stat.assembly_id + curr_free_dof;
            let curr_pos = coord_get(&ws.coords, axis);

            if (limit_axes & (1 << axis)) != 0 {
                emit_limit_constraint(
                    joint_constraints,
                    joint_constraint_columns,
                    cons_base,
                    col_base,
                    slot,
                    abs_dof,
                    ndofs,
                    curr_pos,
                    [stat.data.limits[axis as usize].min, stat.data.limits[axis as usize].max],
                    dt,
                    mass_matrices,
                    m,
                    lu_pivots,
                    piv_offset,
                );
                slot += 1;
            }
            if (motor_axes & (1 << axis)) != 0 {
                let has_limits = (limit_axes & (1 << axis)) != 0;
                let limit_min = stat.data.limits[axis as usize].min;
                let limit_max = stat.data.limits[axis as usize].max;
                emit_motor_constraint(
                    joint_constraints,
                    joint_constraint_columns,
                    cons_base,
                    col_base,
                    slot,
                    abs_dof,
                    ndofs,
                    curr_pos,
                    inv_dt,
                    dt,
                    &stat.data.motors[axis as usize],
                    has_limits,
                    limit_min,
                    limit_max,
                    mass_matrices,
                    m,
                    lu_pivots,
                    piv_offset,
                );
                slot += 1;
            }
            curr_free_dof += 1;
        }
    }
}

/// Solve `M · column = e_{dof_id}` and pack `inv_lhs = 1 / column[dof_id]`,
/// matching `inv_lhs = 1 / (Jᵀ M⁻¹ J)` for J = e_{dof_id}.
#[inline]
fn compute_constraint_column(
    joint_constraint_columns: &mut [f32],
    col_base: usize,
    slot: u32,
    ndofs: u32,
    dof_id: u32,
    mass_matrices: &[f32],
    m: MatSlice,
    lu_pivots: &[u32],
    piv_offset: usize,
) -> f32 {
    let col_offset = col_base + (slot as usize) * (ndofs as usize);
    lu_solve_unit(
        mass_matrices,
        m,
        lu_pivots,
        piv_offset,
        joint_constraint_columns,
        col_offset,
        dof_id,
    );
    let lhs = joint_constraint_columns.read(col_offset + dof_id as usize);
    if lhs != 0.0 { 1.0 / lhs } else { 0.0 }
}

/// Initialize a single limit constraint slot. Mirrors rapier's
/// `unit_joint_limit_constraint`.
#[inline]
fn emit_limit_constraint(
    joint_constraints: &mut [MultibodyJointConstraint],
    joint_constraint_columns: &mut [f32],
    cons_base: usize,
    col_base: usize,
    slot: u32,
    dof_id: u32,
    ndofs: u32,
    curr_pos: f32,
    limits: [f32; 2],
    dt: f32,
    mass_matrices: &[f32],
    m: MatSlice,
    lu_pivots: &[u32],
    piv_offset: usize,
) {
    // Fixed regularization values matching rapier's defaults for joint softness:
    // erp_inv_dt = 1 / dt, cfm_coeff = 0 — full positional bias, no compliance.
    let erp_inv_dt = if dt != 0.0 { 1.0 / dt } else { 0.0 };
    let cfm_coeff = 0.0f32;

    let min_enabled = curr_pos < limits[0];
    let max_enabled = limits[1] < curr_pos;
    let lo_excess = (limits[0] - curr_pos).max(0.0);
    let hi_excess = (curr_pos - limits[1]).max(0.0);
    let rhs_bias = (hi_excess - lo_excess) * erp_inv_dt;
    let rhs_wo_bias = 0.0f32;

    let inv_lhs = compute_constraint_column(
        joint_constraint_columns,
        col_base,
        slot,
        ndofs,
        dof_id,
        mass_matrices,
        m,
        lu_pivots,
        piv_offset,
    );

    let max_neg_impulse = if min_enabled { -1.0e30f32 } else { 0.0 };
    let max_pos_impulse = if max_enabled { 1.0e30f32 } else { 0.0 };

    let cons = MultibodyJointConstraint {
        dof_id,
        kind: 1,
        _kind_extra: 0,
        _pad0: 0,
        rhs: rhs_wo_bias + rhs_bias,
        rhs_wo_bias,
        inv_lhs,
        impulse: 0.0,
        impulse_lo: max_neg_impulse,
        impulse_hi: max_pos_impulse,
        cfm_coeff,
        cfm_gain: 0.0,
    };
    joint_constraints.write(cons_base + slot as usize, cons);
}

/// Initialize a single motor constraint slot. Mirrors rapier's
/// `unit_joint_motor_constraint`. `has_limits` + `(limit_min, limit_max)` flatten
/// rapier's `Option<[Real; 2]>` parameter (rust-gpu can't represent enums).
#[inline]
fn emit_motor_constraint(
    joint_constraints: &mut [MultibodyJointConstraint],
    joint_constraint_columns: &mut [f32],
    cons_base: usize,
    col_base: usize,
    slot: u32,
    dof_id: u32,
    ndofs: u32,
    curr_pos: f32,
    inv_dt: f32,
    dt: f32,
    motor: &crate::dynamics::joint::JointMotor,
    has_limits: bool,
    limit_min: f32,
    limit_max: f32,
    mass_matrices: &[f32],
    m: MatSlice,
    lu_pivots: &[u32],
    piv_offset: usize,
) {
    let (erp_inv_dt, cfm_coeff, cfm_gain, _, max_impulse) = motor_params(motor, dt);

    let mut rhs_wo_bias = 0.0f32;
    if erp_inv_dt != 0.0 {
        rhs_wo_bias += (curr_pos - motor.target_pos) * erp_inv_dt;
    }

    let mut target_vel = motor.target_vel;
    if has_limits {
        let lo = (limit_min - curr_pos) * inv_dt;
        let hi = (limit_max - curr_pos) * inv_dt;
        if target_vel < lo {
            target_vel = lo;
        }
        if target_vel > hi {
            target_vel = hi;
        }
    }
    rhs_wo_bias += -target_vel;

    let inv_lhs = compute_constraint_column(
        joint_constraint_columns,
        col_base,
        slot,
        ndofs,
        dof_id,
        mass_matrices,
        m,
        lu_pivots,
        piv_offset,
    );

    let cons = MultibodyJointConstraint {
        dof_id,
        kind: 2,
        _kind_extra: 0,
        _pad0: 0,
        rhs: rhs_wo_bias,
        rhs_wo_bias,
        inv_lhs,
        impulse: 0.0,
        impulse_lo: -max_impulse,
        impulse_hi: max_impulse,
        cfm_coeff,
        cfm_gain,
    };
    joint_constraints.write(cons_base + slot as usize, cons);
}

/// Replace each active constraint's `rhs` with `rhs_wo_bias`, mirroring rapier's
/// `GenericJointConstraint::remove_bias_from_rhs`.
///
/// Used by the TGS-soft substep loop: bias-driven PGS happens before position
/// integration, then `remove_bias` runs and a final PGS sweep settles velocity
/// along constrained DOFs to zero (no rebound from positional bias).
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_remove_joint_constraint_bias(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] joint_constraints: &mut [MultibodyJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 4)] joint_constraints_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let cons_start = batch_id * *joint_constraints_batch_capacity as usize;
    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let cons_base = cons_start + mb.first_constraint as usize;

    for s in 0..mb.max_constraints {
        let mut cons = joint_constraints.read(cons_base + s as usize);
        if cons.kind == 0 {
            continue;
        }
        cons.rhs = cons.rhs_wo_bias;
        joint_constraints.write(cons_base + s as usize, cons);
    }
}

/// One PGS sweep: iterates the multibody's active limit/motor constraints and
/// updates `dof_velocities` in place. Mirrors rapier's `JointConstraint::solve_generic`
/// for a 1-DOF jacobian.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_solve_joint_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] joint_constraints: &mut [MultibodyJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] joint_constraint_columns: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dof_velocities: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] dof_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 7)] joint_constraints_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 8)] joint_constraint_columns_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;
    let cons_start = batch_id * *joint_constraints_batch_capacity as usize;
    let col_start = batch_id * *joint_constraint_columns_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let ndofs = mb.ndofs;
    if ndofs == 0 || mb.max_constraints == 0 {
        return;
    }
    let v_base = dof_start + mb.first_dof as usize;
    let cons_base = cons_start + mb.first_constraint as usize;
    let col_base = col_start + (mb.first_constraint as usize) * (ndofs as usize);

    for s in 0..mb.max_constraints {
        let mut cons = joint_constraints.read(cons_base + s as usize);
        if cons.kind == 0 {
            continue;
        }

        // J · v for J = e_{dof_id} is just v[dof_id].
        let v_d = dof_velocities.read(v_base + cons.dof_id as usize);
        let rhs_total = v_d + cons.rhs;
        let raw_imp = cons.impulse + cons.inv_lhs * (rhs_total - cons.cfm_gain * cons.impulse);
        let mut new_imp = raw_imp;
        if new_imp < cons.impulse_lo {
            new_imp = cons.impulse_lo;
        }
        if new_imp > cons.impulse_hi {
            new_imp = cons.impulse_hi;
        }
        let delta = new_imp - cons.impulse;
        cons.impulse = new_imp;
        joint_constraints.write(cons_base + s as usize, cons);

        // v -= delta · column   (column = M⁻¹ · e_d).
        for i in 0..ndofs {
            let v_idx = v_base + i as usize;
            let cur = dof_velocities.read(v_idx);
            let col = joint_constraint_columns.read(col_base + (s as usize) * (ndofs as usize) + i as usize);
            dof_velocities.write(v_idx, cur - delta * col);
        }
    }
}

/// Update generalized velocities: `v += a · dt`.
///
/// Split out from the position-update half so that joint-limit / motor
/// constraints can run in between (rapier's order: velocity update → constraint
/// solver → position update).
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_integrate_velocities(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dof_velocities: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] gen_accelerations: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] num_multibodies: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] dt_buf: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] dof_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }
    let dt = dt_buf.read(0);

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;
    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let gen_base = dof_start + mb.first_dof as usize;

    let mut dof_vel = SliceMut(dof_velocities, gen_base);
    let acc = Slice(gen_accelerations, gen_base);

    for d in 0..mb.ndofs {
        let di = d as usize;
        let cur = dof_vel.read(di);
        dof_vel.write(di, cur + acc.read(di) * dt);
    }
}

#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_integrate(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    links_workspace: &mut [MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dof_values: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] dof_velocities: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] num_multibodies: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] dt_buf: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 7)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 8)] links_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 9)] dof_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }
    let dt = dt_buf.read(0);

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let links_start = batch_id * *links_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let num_links = mb.num_links;
    let first_link_global = links_start + mb.first_link as usize;
    let gen_base = dof_start + mb.first_dof as usize;

    let stat_slice = Slice(links_static, first_link_global);
    let mut ws_slice = SliceMut(links_workspace, first_link_global);
    let dof_val = SliceMut(dof_values, gen_base);
    let dof_vel = Slice(dof_velocities, gen_base);

    // Per-link coord / joint_rot update (uses the already-corrected `dof_velocities`).
    for k in 0..num_links {
        let k_usize = k as usize;
        let stat = stat_slice.read(k_usize);
        let mut ws = ws_slice.read(k_usize);
        let locked = stat.data.locked_axes;
        let aid = stat.assembly_id as usize;

        // Free linear DOFs first, in axis order.
        let mut curr_free = 0u32;
        for i in 0u32..3 {
            if (locked & (1 << i)) == 0 {
                let v = dof_vel.read(aid + curr_free as usize);
                let new = coord_get(&ws.coords, i) + v * dt;
                coord_set(&mut ws.coords, i, new);
                curr_free += 1;
            }
        }

        // Free angular DOFs.
        let ang_locked = (locked >> 3) & 0x7;
        let num_ang = 3 - ang_locked.count_ones();
        if num_ang == 1 {
            let dof_id = (!ang_locked & 0x7).trailing_zeros();
            let v = dof_vel.read(aid + curr_free as usize);
            let idx = 3 + dof_id;
            let new = coord_get(&ws.coords, idx) + v * dt;
            coord_set(&mut ws.coords, idx, new);
            ws.joint_rot = rotation_from_scaled_axis(basis_vec3(dof_id) * new);
        } else if num_ang == 3 {
            let vx = dof_vel.read(aid + curr_free as usize);
            let vy = dof_vel.read(aid + (curr_free + 1) as usize);
            let vz = dof_vel.read(aid + (curr_free + 2) as usize);
            let ang = Vec3::new(vx, vy, vz);
            let disp = rotation_from_scaled_axis(ang * dt);
            ws.joint_rot = rotation_renormalize_fast(disp * ws.joint_rot);
            let c3 = coord_get(&ws.coords, 3) + vx * dt;
            let c4 = coord_get(&ws.coords, 4) + vy * dt;
            let c5 = coord_get(&ws.coords, 5) + vz * dt;
            coord_set(&mut ws.coords, 3, c3);
            coord_set(&mut ws.coords, 4, c4);
            coord_set(&mut ws.coords, 5, c5);
        }
        // num_ang == 0: no-op.

        ws_slice.write(k_usize, ws);
    }

    // Silence dof_val unused warning — it will be used once we also support
    // setting coords directly (e.g. user-controlled kinematic DOFs).
    let _ = dof_val.0;
}

//
// Multibody contact constraints.
//
// Mirrors rapier's `RigidBodyMultibodyContactConstraint` flow but currently
// limited to the **normal** component (no friction) of contacts where exactly
// one side is a multibody (the other is a free rigid body).
//
// Pipeline, called once per substep from `apply_substep`:
//
//   1. `gpu_mb_init_contact_constraints` — scan the contacts buffer; for each
//      contact point touching a link of this multibody, emit a normal-direction
//      constraint and write the multibody-side `Jᵀ` row into
//      `contact_constraint_jacs`.
//   2. `gpu_mb_finalize_contact_constraints` — for each emitted constraint,
//      LU back-solve `M · column = Jᵀ` (writing the column into
//      `contact_constraint_columns`) and set `inv_lhs = 1 / (Jᵀ·column +
//      free_body_inv_r)`.
//   3. `gpu_mb_solve_contact_constraints` — one PGS sweep updating both
//      `dof_velocities` (multibody side) and `solver_vels` (free body side).
//   4. `gpu_mb_remove_contact_constraint_bias` — strip the positional bias
//      from `rhs` (mirrors `gpu_mb_remove_joint_constraint_bias`) for the
//      stabilization sweep.
//

/// Read the `link_id`-th column block of the multibody's body jacobian and
/// project it through the per-side `(unit_force, unit_torque)` pair,
/// **adding** the resulting `Jᵀ` row to `out_jacs[col_offset ..]` (so two
/// calls accumulate — used by self-collisions, which combine the two
/// touched links into a single net `Jᵀ` row).
///
/// Mirrors rapier's `Multibody::fill_jacobians` (the scalar inner kernel),
/// returning `j · invm_j` is deferred to the finalize kernel; here we just
/// pack the row.
#[inline]
fn fill_contact_jac_row(
    body_jacobians: &[f32],
    mb_jac_base: usize,
    ndofs: u32,
    link_id: u32,
    unit_force: Vec3,
    unit_torque: Vec3,
    out_jacs: &mut [f32],
    col_offset: usize,
    accumulate: bool,
) {
    // Per-link 6×ndofs jacobian (rows 0-2 = J_v, rows 3-5 = J_w).
    let link_jac_base = mb_jac_base + (link_id as usize) * 6 * (ndofs as usize);
    let link_j = MatSlice::dense(link_jac_base, 6, ndofs);
    let (link_j_v, link_j_w) = link_j.rows_range_pair(0, 3, 3, 3);
    for j in 0..ndofs {
        let jv0 = body_jacobians.read(link_j_v.idx(0, j));
        let jv1 = body_jacobians.read(link_j_v.idx(1, j));
        let jv2 = body_jacobians.read(link_j_v.idx(2, j));
        let jw0 = body_jacobians.read(link_j_w.idx(0, j));
        let jw1 = body_jacobians.read(link_j_w.idx(1, j));
        let jw2 = body_jacobians.read(link_j_w.idx(2, j));
        let dot = unit_force.x * jv0
            + unit_force.y * jv1
            + unit_force.z * jv2
            + unit_torque.x * jw0
            + unit_torque.y * jw1
            + unit_torque.z * jw2;
        let prev = if accumulate {
            out_jacs.read(col_offset + j as usize)
        } else {
            0.0
        };
        out_jacs.write(col_offset + j as usize, prev + dot);
    }
}

/// Pack the per-link world-space contact point into the constraint.
///
/// Pass 1: scans every contact in `contacts[batch]` and, for each contact
/// point touching a link of this multibody, emits a normal-direction
/// `MultibodyContactConstraint` and writes the multibody-side `Jᵀ` row
/// (`mb_normal · J_v + (mb_shift × mb_normal) · J_w`) into
/// `contact_constraint_jacs`. Friction tangents and multibody-multibody
/// contacts are not yet handled — such contacts are skipped.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_init_contact_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] links_workspace: &[MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] body_jacobians: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] body_to_link: &[[u32; 2]],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] contact_constraint_jacs: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] contact_constraint_count: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] num_multibodies: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] dt_buf: &[f32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] mprops: &[WorldMassProperties],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 2)] contacts: &[IndexedManifold],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 3)] contacts_len: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 10)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 11)] links_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 12)] jacobians_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 13)] dof_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 14)] contact_constraints_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 15)] contact_constraint_columns_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 1, binding = 4)] contacts_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 1, binding = 5)] colliders_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 1, binding = 6)] body_to_link_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }
    let dt = dt_buf.read(0);
    let inv_dt = if dt != 0.0 { 1.0 / dt } else { 0.0 };
    let erp_inv_dt = inv_dt;
    let allowed_lin_err = 0.001f32;
    let max_corr_velocity = 10.0f32;

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let links_start = batch_id * *links_batch_capacity as usize;
    let jac_start = batch_id * *jacobians_batch_capacity as usize;
    let cons_start = batch_id * *contact_constraints_batch_capacity as usize;
    let col_start = batch_id * *contact_constraint_columns_batch_capacity as usize;
    let contacts_start = batch_id * *contacts_batch_capacity as usize;
    let colliders_start = batch_id * *colliders_batch_capacity as usize;
    let b2l_start = batch_id * *body_to_link_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let ndofs = mb.ndofs;
    if ndofs == 0 {
        contact_constraint_count.write(mb_start + mb_idx as usize, 0);
        return;
    }
    let mb_jac_base = jac_start + mb.jacobian_offset as usize;
    let cons_base = cons_start + (mb_idx as usize) * (MAX_MB_CONTACTS_PER_MB as usize);
    // Each constraint slot reserves `dof_batch_capacity` floats in the
    // column buffer (matches the allocation in `from_rapier` and avoids any
    // overlap between multibodies of differing `ndofs`).
    let dofs_stride = *dof_batch_capacity as usize;
    let col_base =
        col_start + (mb_idx as usize) * (MAX_MB_CONTACTS_PER_MB as usize) * dofs_stride;

    let ws_slice = Slice(links_workspace, links_start + mb.first_link as usize);
    let _ = links_static;

    let n_contacts = contacts_len.read(batch_id);
    let mut count = 0u32;

    for ci in 0..n_contacts {
        if count >= MAX_MB_CONTACTS_PER_MB {
            break;
        }
        let im = contacts.read(contacts_start + ci as usize);
        let id1 = im.colliders.x;
        let id2 = im.colliders.y;

        let l1 = body_to_link.read(b2l_start + id1 as usize);
        let l2 = body_to_link.read(b2l_start + id2 as usize);
        let mb_on_1 = l1[0] == mb_idx;
        let mb_on_2 = l2[0] == mb_idx;

        if !mb_on_1 && !mb_on_2 {
            continue;
        }
        // Inter-multibody contacts (each side is a DIFFERENT multibody) are
        // not yet handled — skip them. Self-collisions (both sides on this
        // SAME multibody) are handled below.
        if l1[0] != u32::MAX && l2[0] != u32::MAX && l1[0] != l2[0] {
            continue;
        }

        let is_self = mb_on_1 && mb_on_2;
        let (mb_link_id_a, mb_link_id_b, free_body_id) = if is_self {
            (l1[1], l2[1], u32::MAX)
        } else if mb_on_1 {
            (l1[1], u32::MAX, id2)
        } else {
            (l2[1], u32::MAX, id1)
        };

        // Skip degenerate self-contacts on the same link (geometry shouldn't
        // produce these, but be defensive).
        if is_self && mb_link_id_a == mb_link_id_b {
            continue;
        }

        let pose1 = poses.read(colliders_start + id1 as usize);
        let world_normal = pose1.rotation * im.contact.normal_a;
        // Convention: `lin_jac` = impulse direction on the "B-side" body.
        //   - Free contact: B-side = free body. lin_jac = +world_normal_a if
        //     mb=1, -world_normal_a if mb=2.
        //   - Self-contact: B-side = link `mb_link_id_b` (= rapier's body 2).
        //     lin_jac = +world_normal_a (impulse on body 2 = -force_dir1).
        let lin_jac = if is_self || mb_on_1 { world_normal } else { -world_normal };
        let mb_normal = -lin_jac;

        // Free-body mass-properties (only valid for the free-contact path).
        let free_mp = if is_self {
            WorldMassProperties::default()
        } else {
            mprops.read(colliders_start + free_body_id as usize)
        };
        let free_im = if is_self { 0.0 } else { free_mp.inv_mass.x };

        let link_ws_a = ws_slice.read(mb_link_id_a as usize);
        let link_origin_a = link_ws_a.local_to_world.translation;
        // For self-contacts the second link's origin is read inside the
        // contact-point loop; for free contacts this is unused.
        let link_origin_b_default = link_origin_a;

        for k in 0..im.contact.len {
            if count >= MAX_MB_CONTACTS_PER_MB {
                break;
            }
            let pt_local = im.contact.points_a.read(k as usize).pt;
            let dist = im.contact.points_a.read(k as usize).dist;
            // World contact point — mid-point between the two surfaces, matching
            // rapier's `pose1 * (pt_a + normal_a * dist / 2)`.
            let pt_world = pose1 * (pt_local + im.contact.normal_a * (dist * 0.5));

            // A-side (link `mb_link_id_a`, rapier's body 1): impulse along
            // `force_dir1 = -world_normal_a = mb_normal`.
            let shift_a = pt_world - link_origin_a;
            let torque_a = shift_a.cross(mb_normal);

            // Penetration bias: rapier's clamped `erp_inv_dt · (dist + allowed_lin_err)`.
            let rhs_bias =
                (erp_inv_dt * (dist + allowed_lin_err)).clamp(-max_corr_velocity, 0.0);
            // Repulsion against any positive distance — clears float drift.
            let rhs_wo_bias = if dist > 0.0 { dist * inv_dt } else { 0.0 };

            let slot = count;
            let col_offset = col_base + (slot as usize) * dofs_stride;

            // Always start by writing A-side jacobian (overwriting any prior
            // slot content). For self-contacts we then accumulate the B-side.
            fill_contact_jac_row(
                body_jacobians,
                mb_jac_base,
                ndofs,
                mb_link_id_a,
                mb_normal,
                torque_a,
                contact_constraint_jacs,
                col_offset,
                false,
            );

            // For free contacts, the free body's J row is encoded via
            // `lin_jac` / `ang_jac` / `ii_ang_jac` on the constraint and
            // applied directly to `solver_vels` during solve. For
            // self-contacts, both sides go through the same multibody, so
            // the B-side jacobian must be added into the same `Jᵀ` row.
            let (ang_jac, ii_ang_jac, link_id_for_struct) = if is_self {
                let link_ws_b = ws_slice.read(mb_link_id_b as usize);
                let link_origin_b = link_ws_b.local_to_world.translation;
                let _ = link_origin_b_default;
                let shift_b = pt_world - link_origin_b;
                // B-side (link `mb_link_id_b`, rapier's body 2): impulse
                // along `+world_normal_a = lin_jac`. Torque arg matches
                // rapier's `torque_dir2 = dp2 × (-force_dir1) = shift_b ×
                // lin_jac`.
                let torque_b = shift_b.cross(lin_jac);
                fill_contact_jac_row(
                    body_jacobians,
                    mb_jac_base,
                    ndofs,
                    mb_link_id_b,
                    lin_jac,
                    torque_b,
                    contact_constraint_jacs,
                    col_offset,
                    true,
                );
                // ang_jac / ii_ang_jac aren't used by the solve path when
                // `free_body_id == u32::MAX`; keep them zero for clarity.
                (Vec3::ZERO, Vec3::ZERO, mb_link_id_a)
            } else {
                let _ = link_origin_b_default;
                let free_shift = pt_world - free_mp.com;
                let aj = free_shift.cross(lin_jac);
                let iiaj = free_mp.inv_inertia_mul(aj);
                (aj, iiaj, mb_link_id_a)
            };

            let cons = MultibodyContactConstraint {
                multibody_id: mb_idx,
                link_id: link_id_for_struct,
                kind: 1,
                free_body_id,
                free_body_im: free_im,
                _pad0: [0; 3],
                lin_jac,
                _pad1: 0,
                ang_jac,
                _pad2: 0,
                ii_ang_jac,
                _pad3: 0,
                inv_lhs: 0.0,
                rhs: rhs_wo_bias + rhs_bias,
                rhs_wo_bias,
                impulse: 0.0,
                cfm_coeff: 0.0,
                cfm_gain: 0.0,
                _pad4: [0; 2],
            };
            contact_constraints.write(cons_base + slot as usize, cons);
            count += 1;
        }
    }

    // Mark surplus slots as inactive so the solve sweep skips them.
    for s in count..MAX_MB_CONTACTS_PER_MB {
        let mut cz = contact_constraints.read(cons_base + s as usize);
        cz.kind = 0;
        cz.impulse = 0.0;
        contact_constraints.write(cons_base + s as usize, cz);
    }
    contact_constraint_count.write(mb_start + mb_idx as usize, count);
}

/// Pass 2: for each emitted constraint, LU back-solve `M · column = Jᵀ`
/// (the row produced by the init kernel) and set `inv_lhs = 1 / (Jᵀ ·
/// column + free_body_inv_r)`.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_finalize_contact_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mass_matrices: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] lu_pivots: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] contact_constraint_jacs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] contact_constraint_columns: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] contact_constraint_count: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 8)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 9)] mass_matrix_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 10)] dof_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 11)] contact_constraints_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 12)] contact_constraint_columns_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let mm_start = batch_id * *mass_matrix_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;
    let cons_start = batch_id * *contact_constraints_batch_capacity as usize;
    let col_start = batch_id * *contact_constraint_columns_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let ndofs = mb.ndofs;
    if ndofs == 0 {
        return;
    }
    let mb_mm_base = mm_start + mb.mass_matrix_offset as usize;
    let piv_offset = dof_start + mb.first_dof as usize;
    let cons_base = cons_start + (mb_idx as usize) * (MAX_MB_CONTACTS_PER_MB as usize);
    let dofs_stride = *dof_batch_capacity as usize;
    let col_base =
        col_start + (mb_idx as usize) * (MAX_MB_CONTACTS_PER_MB as usize) * dofs_stride;

    let m = MatSlice::dense(mb_mm_base, ndofs, ndofs);
    let count = contact_constraint_count.read(mb_start + mb_idx as usize);

    for s in 0..count {
        let col_offset = col_base + (s as usize) * dofs_stride;
        // 1) Copy J^T row into the column buffer (it'll be overwritten by the
        //    LU solve with the M⁻¹·Jᵀ result).
        for i in 0..ndofs {
            let v = contact_constraint_jacs.read(col_offset + i as usize);
            contact_constraint_columns.write(col_offset + i as usize, v);
        }
        // 2) Solve M · column = J^T  (in place).
        lu_solve_in_place(
            mass_matrices,
            m,
            lu_pivots,
            piv_offset,
            contact_constraint_columns,
            col_offset,
        );
        // 3) inv_r_mb = J · column.
        let mut inv_r_mb = 0.0f32;
        for i in 0..ndofs {
            let j = contact_constraint_jacs.read(col_offset + i as usize);
            let c = contact_constraint_columns.read(col_offset + i as usize);
            inv_r_mb += j * c;
        }
        // 4) Add free body's contribution: im (since lin_jac is unit) +
        //    ang_jac · ii_ang_jac. For self-contacts both sides are already
        //    folded into the multibody-side `Jᵀ`, so there's no free-body
        //    term — `inv_lhs` is just `1 / (Jᵀ·column)`.
        let mut cons = contact_constraints.read(cons_base + s as usize);
        let is_self = cons.free_body_id == u32::MAX;
        let inv_r_free = if is_self {
            0.0
        } else {
            cons.free_body_im + cons.ang_jac.dot(cons.ii_ang_jac)
        };
        let total = inv_r_mb + inv_r_free;
        cons.inv_lhs = if total > 0.0 { 1.0 / total } else { 0.0 };
        contact_constraints.write(cons_base + s as usize, cons);
    }
}

/// One PGS sweep over the multibody's active contact constraints. Updates
/// the multibody's `dof_velocities` and the free body's `solver_vels`.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_solve_contact_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contact_constraint_jacs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] contact_constraint_columns: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] contact_constraint_count: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] dof_velocities: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] solver_vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 8)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 9)] dof_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 10)] contact_constraints_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 11)] contact_constraint_columns_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 12)] colliders_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;
    let cons_start = batch_id * *contact_constraints_batch_capacity as usize;
    let col_start = batch_id * *contact_constraint_columns_batch_capacity as usize;
    let colliders_start = batch_id * *colliders_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let ndofs = mb.ndofs;
    if ndofs == 0 {
        return;
    }
    let v_base = dof_start + mb.first_dof as usize;
    let cons_base = cons_start + (mb_idx as usize) * (MAX_MB_CONTACTS_PER_MB as usize);
    let dofs_stride = *dof_batch_capacity as usize;
    let col_base =
        col_start + (mb_idx as usize) * (MAX_MB_CONTACTS_PER_MB as usize) * dofs_stride;

    let count = contact_constraint_count.read(mb_start + mb_idx as usize);
    for s in 0..count {
        let mut cons = contact_constraints.read(cons_base + s as usize);
        if cons.kind == 0 {
            continue;
        }
        let col_offset = col_base + (s as usize) * dofs_stride;

        // J · u = J_mb · v_mb_dofs + J_free · v_free.
        // For self-contacts (`free_body_id == u32::MAX`), the B-side jacobian
        // is folded into `J_mb` already, so there's no separate free-body
        // term to add.
        let is_self = cons.free_body_id == u32::MAX;
        let mut j_dot_v = 0.0f32;
        for i in 0..ndofs {
            let j = contact_constraint_jacs.read(col_offset + i as usize);
            let v = dof_velocities.read(v_base + i as usize);
            j_dot_v += j * v;
        }
        let free = if is_self {
            Velocity::default()
        } else {
            solver_vels.read(colliders_start + cons.free_body_id as usize)
        };
        if !is_self {
            j_dot_v += cons.lin_jac.dot(free.linear) + cons.ang_jac.dot(free.angular);
        }

        // rapier's contact PGS step: `dlambda = -r · (dvel + cfm·λ)` where
        // `dvel = J·u + rhs`. Note this is the OPPOSITE sign convention from
        // the joint-limit kernel, which uses `+r · (dvel - cfm·λ)`. The
        // difference: joint limits encode "target velocity = -rhs" with
        // positive rhs at excess; contacts encode "target separation = -rhs"
        // with negative rhs at penetration.
        let rhs_total = j_dot_v + cons.rhs;
        let raw_imp = cons.impulse
            - cons.inv_lhs * (rhs_total + cons.cfm_gain * cons.impulse);
        // Normal impulse must be ≥ 0 (no pulling apart).
        let new_imp = if raw_imp < 0.0 { 0.0 } else { raw_imp };
        let delta = new_imp - cons.impulse;
        cons.impulse = new_imp;
        contact_constraints.write(cons_base + s as usize, cons);

        if delta != 0.0 {
            // Multibody side: `Jᵀ` was packed with `mb_normal = -lin_jac` for
            // the A-side (and accumulated with `+lin_jac` for the B-side on
            // self-contacts). To push the multibody apart, add `delta·column`.
            for i in 0..ndofs {
                let v_idx = v_base + i as usize;
                let cur = dof_velocities.read(v_idx);
                let col = contact_constraint_columns.read(col_offset + i as usize);
                dof_velocities.write(v_idx, cur + delta * col);
            }
            if !is_self {
                // Free body side: solver_vels += delta · M_free⁻¹ · J_free^T.
                let mut new_free = free;
                new_free.linear =
                    new_free.linear + cons.lin_jac * (cons.free_body_im * delta);
                new_free.angular = new_free.angular + cons.ii_ang_jac * delta;
                solver_vels.write(colliders_start + cons.free_body_id as usize, new_free);
            }
        }
    }
}

/// Strip the positional bias from each active contact constraint's `rhs`,
/// matching `gpu_mb_remove_joint_constraint_bias`.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_remove_contact_constraint_bias(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] contact_constraint_count: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 4)] contact_constraints_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let cons_start = batch_id * *contact_constraints_batch_capacity as usize;
    let cons_base = cons_start + (mb_idx as usize) * (MAX_MB_CONTACTS_PER_MB as usize);
    let count = contact_constraint_count.read(mb_start + mb_idx as usize);

    for s in 0..count {
        let mut cons = contact_constraints.read(cons_base + s as usize);
        if cons.kind == 0 {
            continue;
        }
        cons.rhs = cons.rhs_wo_bias;
        contact_constraints.write(cons_base + s as usize, cons);
    }
}
