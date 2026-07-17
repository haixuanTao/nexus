//! Generic impulse-joint constraints for joints touching one or two multibodies.
//!
//! Mirrors rapier's `GenericJointConstraint` / `JointGenericExternalConstraintBuilder`
//! pipeline. Used for any impulse joint whose endpoints are not both free
//! rigid bodies — when at least one side is a multibody link the regular
//! `JointConstraint` solver path can't propagate impulses through `M⁻¹·Jᵀ`,
//! so this dedicated path is used instead.
//!
//! Pipeline (one MB joint = one builder slot, up to `MAX_AXIS_CONSTRAINTS`
//! axis constraint slots):
//!
//!   1. `gpu_mb_init_impulse_joint_constraints` — once per step, after FK / LU.
//!      Reads the per-builder joint description and current poses, builds the
//!      `JointConstraintHelper` (basis / cmat / lin_err / ang_err) and emits
//!      one [`MbImpulseJointConstraint`] per active locked / limited / motorised
//!      axis. The per-side `Jᵀ` row and `M⁻¹·Jᵀ` column are written into the
//!      flat `jacobians` buffer (matches rapier's `DVector` jacobians).
//!   2. `gpu_mb_update_impulse_joint_constraints` — once per substep,
//!      regenerates everything (rapier's `update` is also a full rebuild).
//!   3. `gpu_mb_solve_impulse_joint_constraints` — one PGS sweep, updates
//!      both sides' velocities (`dof_velocities` for multibody, `solver_vels`
//!      for free body).
//!   4. `gpu_mb_remove_impulse_joint_constraint_bias` — strips the positional
//!      bias from `rhs` before the stabilization sweep (rapier's
//!      `remove_bias_from_rhs`).

use crate::ColumnIndex;
use glamx::Vec2;
#[cfg(feature = "dim3")]
use glamx::{Mat3, Mat4, Quat, Vec3};

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

use crate::dynamics::body::{Velocity, WorldMassProperties};
use crate::dynamics::joint::{
    ANG_AXES_MASK, GenericJoint, JointMotor, LIN_AXES_MASK, MotorParameters, SPATIAL_DIM,
    motor_params,
};
use crate::utils::BatchIndices;
use crate::utils::linalg::{MatSlice, lu_solve_in_place};
use crate::{ANG_DIM, AngVector, DIM, Pose, Vector, gcross, gdot, rotation_to_matrix};

use super::lu::LANES;
use super::types::{MultibodyInfo, MultibodyLinkStatic, MultibodyLinkWorkspace};

/// Maximum unit-axis constraints any single impulse joint can produce.
///
/// `SPATIAL_DIM * 2` covers a free joint with both limits AND motors enabled
/// on every axis (rapier emits limits and motors as separate constraints,
/// and locks reuse those slots when no limit/motor is enabled). The
/// auxiliary motor / limit unit constraint count is bounded by `SPATIAL_DIM`
/// each so we reserve `2 * SPATIAL_DIM` slots per joint.
pub const MAX_AXIS_CONSTRAINTS: u32 = (SPATIAL_DIM as u32) * 2;

/// Sentinel "no body" — used when a side is `Fixed` (rapier `LinkOrBody::Fixed`).
pub const SIDE_FIXED: u32 = u32::MAX;

const MAX_F32: f32 = 1.0e20;

#[cfg(feature = "dim2")]
const DIM_USIZE: usize = 2;
#[cfg(feature = "dim3")]
const DIM_USIZE: usize = 3;

/// Tag distinguishing how each side of a generic impulse joint connects
/// to the solver state.
///
/// Mirrors rapier's `LinkOrBody`:
///   * `0` — Free rigid body. `body_id` indexes into the per-batch solver
///     velocity / mprops buffer; `ndofs` is `SPATIAL_DIM`.
///   * `1` — Multibody link. `mb_id` indexes the per-batch
///     `multibody_info`; `link_id` indexes the link within the multibody;
///     `ndofs` is `mb.ndofs`.
///   * `2` — Static fixed pose. No DOFs, no velocity update.
pub const SIDE_KIND_BODY: u32 = 0;
pub const SIDE_KIND_MB: u32 = 1;
pub const SIDE_KIND_FIXED: u32 = 2;

/// Per-impulse-joint static descriptor — the GPU mirror of rapier's
/// `JointGenericExternalConstraintBuilder`.
///
/// One slot per joint that touches at least one multibody. The init kernel
/// reads it to (re)build the joint's axis constraints in the per-batch
/// `constraints` slab.
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct MbImpulseJointBuilder {
    /// Joint description — frames already shifted into solver-body
    /// (COM-centered) space at host time, mirroring
    /// `GenericJoint::transform_to_solver_body_space`.
    pub joint: GenericJoint,

    /// `SIDE_KIND_BODY` / `SIDE_KIND_MB` / `SIDE_KIND_FIXED`.
    pub side_a_kind: u32,
    /// Free-body local id (when `SIDE_KIND_BODY`) or multibody index in
    /// the per-batch `multibody_info` (when `SIDE_KIND_MB`). `SIDE_FIXED`
    /// when `side_a_kind == SIDE_KIND_FIXED`.
    pub side_a_id: u32,
    /// Link index within the multibody (only meaningful for `SIDE_KIND_MB`).
    pub side_a_link: u32,
    /// Source impulse-joint id, for impulse writeback.
    pub joint_id: u32,

    pub side_b_kind: u32,
    pub side_b_id: u32,
    pub side_b_link: u32,
    /// First constraint slot (in the per-batch constraints slab) reserved
    /// for this joint's axis constraints.
    pub constraint_id: u32,

    /// First float index (in the per-batch jacobians buffer) reserved for
    /// this joint. The init kernel walks this offset axis-by-axis: each
    /// axis constraint takes `2 * (ndofs_a + ndofs_b)` floats — `J_a`,
    /// `M⁻¹·J_a`, `J_b`, `M⁻¹·J_b` packed in that order.
    pub jacobian_offset: u32,
    /// Total floats reserved for this joint's jacobian block (= per-axis
    /// stride × `MAX_AXIS_CONSTRAINTS`). Used by the init kernel to bound
    /// its writes; the per-axis stride is recomputed from `ndofs_a / b`.
    pub jacobian_capacity: u32,
    /// Pad to GenericJoint's alignment (16 bytes in 3D — see ImpulseJoint).
    /// `GenericJoint` is 320 bytes + 40 bytes of side metadata = 360 bytes;
    /// add 2 u32 to round up to 368 (multiple of 16).
    #[cfg(feature = "dim3")]
    pub _pad0: [u32; 2],
}

/// One unit-axis generic impulse-joint constraint — the GPU mirror of
/// rapier's `GenericJointConstraint`.
///
/// `kind` values: `0` = inactive / unused slot, `1` = active.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct MbImpulseJointConstraint {
    /// `0` = inactive, `1` = active.
    pub kind: u32,
    /// Joint id of the source impulse joint (for impulse writeback).
    pub joint_id: u32,
    /// Writeback type — mirrors rapier's `WritebackId`:
    ///   * `0` = `Dof(writeback_axis)` (lock)
    ///   * `1` = `Limit(writeback_axis)`
    ///   * `2` = `Motor(writeback_axis)`
    pub writeback_kind: u32,
    /// Axis index for the writeback (0..SPATIAL_DIM).
    pub writeback_axis: u32,

    pub side_a_kind: u32,
    pub side_a_id: u32,
    pub side_a_link: u32,
    pub ndofs_a: u32,

    pub side_b_kind: u32,
    pub side_b_id: u32,
    pub side_b_link: u32,
    pub ndofs_b: u32,

    /// First float of `J_a` in the per-batch jacobians buffer. Layout per
    /// axis constraint:
    ///
    ///   `[J_a (ndofs_a), W·J_a (ndofs_a), J_b (ndofs_b), W·J_b (ndofs_b)]`
    pub j_id_a: u32,
    /// First float of `J_b` (= `j_id_a + 2*ndofs_a`).
    pub j_id_b: u32,
    /// Padding so the 32-byte block ahead of the f32 fields keeps a 4-byte
    /// alignment — matches the constraint's natural f32 layout.
    pub _pad0: [u32; 2],

    pub impulse: f32,
    pub impulse_lo: f32,
    pub impulse_hi: f32,
    pub inv_lhs: f32,

    pub rhs: f32,
    pub rhs_wo_bias: f32,
    pub cfm_coeff: f32,
    pub cfm_gain: f32,
}

// Returns the multibody side jacobian's offset / size in the per-batch
// jacobians buffer. `wj_id` is the start of the corresponding `M⁻¹·J`
// block (= `j_id + ndofs`).
#[inline]
fn wj_id(j_id: u32, ndofs: u32) -> usize {
    (j_id + ndofs) as usize
}

/// `k`-th component of a free body's spatial velocity, in the same order
/// the jacobian rows are packed: `[lin (DIM), ang (ANG_DIM)]`. Returns a
/// value (not a reference) to avoid SPIR-V pointer-phi nodes.
#[inline]
fn spatial_component(v: Velocity, k: u32) -> f32 {
    #[cfg(feature = "dim3")]
    {
        if k == 0 {
            v.linear.x
        } else if k == 1 {
            v.linear.y
        } else if k == 2 {
            v.linear.z
        } else if k == 3 {
            v.angular.x
        } else if k == 4 {
            v.angular.y
        } else {
            v.angular.z
        }
    }
    #[cfg(feature = "dim2")]
    {
        if k == 0 {
            v.linear.x
        } else if k == 1 {
            v.linear.y
        } else {
            v.angular
        }
    }
}

/// Workgroup-cooperative `J · v` for a generic side. Every one of the
/// `LANES` lanes forms a single product term, which are tree-reduced
/// through `partial`; the scalar result is broadcast to all lanes.
///
/// The barrier sequence is **identical for every side kind and for
/// inactive constraints** (`active == false` just zeroes the term), so the
/// enclosing per-axis loop stays in workgroup-uniform control flow — no
/// lane ever skips a barrier another lane executes.
#[inline]
fn side_dot_vel_par(
    active: bool,
    kind: u32,
    j_id: u32,
    ndofs: u32,
    body_id: u32,
    jacobians: &[f32],
    dof_vels: &[f32],
    dof_base_for_mb: usize,
    solver_vels: &[Velocity],
    colliders_start: usize,
    lane: u32,
    partial: &mut impl MaybeIndexUnchecked<f32>,
) -> f32 {
    // Lane → term mapping:
    //   * SIDE_KIND_MB:   lane l (< ndofs)      → J[l] · v_dof[l]
    //   * SIDE_KIND_BODY: lane k (< SPATIAL_DIM) → J[k] · v_spatial[k]
    //   * FIXED / inactive / out-of-range        → 0
    let term = if !active || kind == SIDE_KIND_FIXED {
        0.0f32
    } else if kind == SIDE_KIND_BODY {
        if lane < SPATIAL_DIM as u32 {
            let v = solver_vels.read(colliders_start + body_id as usize);
            jacobians.read(j_id as usize + lane as usize) * spatial_component(v, lane)
        } else {
            0.0f32
        }
    } else {
        // SIDE_KIND_MB
        if lane < ndofs {
            jacobians.read(j_id as usize + lane as usize)
                * dof_vels.read(dof_base_for_mb + lane as usize)
        } else {
            0.0f32
        }
    };

    partial.write(lane as usize, term);
    workgroup_memory_barrier_with_group_sync();
    // Tree reduction over the 32 lanes (2^5 == LANES).
    for step in 0..5u32 {
        let stride = 1u32 << (4 - step);
        if lane < stride {
            let v = partial.read(lane as usize) + partial.read((lane + stride) as usize);
            partial.write(lane as usize, v);
        }
        workgroup_memory_barrier_with_group_sync();
    }
    let result = partial.read(0);
    // Trailing barrier: guarantees every lane has read `partial[0]` before
    // the next reduction (or caller) overwrites `partial`.
    workgroup_memory_barrier_with_group_sync();
    result
}

/// Workgroup-cooperative `±delta · W·J` apply. The multibody side is
/// lane-split over its DOFs (lane `l` owns DOF `l` → disjoint writes, no
/// barrier); the free-body side is a single shared velocity so lane 0 does
/// the read-modify-write alone. Contains no barriers — the caller issues
/// one unconditional barrier per axis after both apply calls so the
/// velocity writes are visible to the next axis's dot products.
#[inline]
fn side_apply_impulse_par(
    active: bool,
    kind: u32,
    j_id: u32,
    ndofs: u32,
    body_id: u32,
    sign: f32,
    delta: f32,
    jacobians: &[f32],
    dof_vels: &mut [f32],
    dof_base_for_mb: usize,
    solver_vels: &mut [Velocity],
    colliders_start: usize,
    lane: u32,
) {
    // All operands are workgroup-uniform, so this early-out is uniform.
    if !active || kind == SIDE_KIND_FIXED || delta == 0.0 {
        return;
    }
    let wj0 = wj_id(j_id, ndofs);
    let scaled = sign * delta;
    if kind == SIDE_KIND_BODY {
        if lane == 0 {
            let coll_idx = colliders_start + body_id as usize;
            let mut v = solver_vels.read(coll_idx);
            #[cfg(feature = "dim3")]
            {
                v.linear.x += scaled * jacobians.read(wj0);
                v.linear.y += scaled * jacobians.read(wj0 + 1);
                v.linear.z += scaled * jacobians.read(wj0 + 2);
                v.angular.x += scaled * jacobians.read(wj0 + 3);
                v.angular.y += scaled * jacobians.read(wj0 + 4);
                v.angular.z += scaled * jacobians.read(wj0 + 5);
            }
            #[cfg(feature = "dim2")]
            {
                v.linear.x += scaled * jacobians.read(wj0);
                v.linear.y += scaled * jacobians.read(wj0 + 1);
                v.angular += scaled * jacobians.read(wj0 + 2);
            }
            solver_vels.write(coll_idx, v);
        }
        return;
    }
    // SIDE_KIND_MB — lane l owns DOF l (disjoint → race-free).
    if lane < ndofs {
        let v_idx = dof_base_for_mb + lane as usize;
        let cur = dof_vels.read(v_idx);
        let w = jacobians.read(wj0 + lane as usize);
        dof_vels.write(v_idx, cur + scaled * w);
    }
}

/// Stride (in floats) reserved per axis-constraint in the jacobians buffer:
/// `J_a (ndofs_a) + W·J_a (ndofs_a) + J_b (ndofs_b) + W·J_b (ndofs_b)`.
#[inline]
fn axis_stride(ndofs_a: u32, ndofs_b: u32) -> u32 {
    2 * (ndofs_a + ndofs_b)
}

/// Write a free body's `(unit_force, unit_torque)` Jᵀ row + its `W·Jᵀ`
/// (= `(im⊙f, ii·t)`) into the jacobians buffer at `j_id`. Mirrors rapier's
/// `JointSolverBody::fill_jacobians`.
fn fill_body_jacobians(
    jacobians: &mut [f32],
    j_id: u32,
    body_id: u32,
    unit_force: Vector,
    unit_torque: AngVector,
    mprops: &[WorldMassProperties],
    colliders_start: usize,
) {
    let mp = mprops.read(colliders_start + body_id as usize);
    let im = mp.inv_mass;

    let base = j_id as usize;
    #[cfg(feature = "dim3")]
    {
        jacobians.write(base, unit_force.x);
        jacobians.write(base + 1, unit_force.y);
        jacobians.write(base + 2, unit_force.z);
        jacobians.write(base + 3, unit_torque.x);
        jacobians.write(base + 4, unit_torque.y);
        jacobians.write(base + 5, unit_torque.z);
    }
    #[cfg(feature = "dim2")]
    {
        jacobians.write(base, unit_force.x);
        jacobians.write(base + 1, unit_force.y);
        jacobians.write(base + 2, unit_torque);
    }

    // W·J: linear part = im ⊙ unit_force; angular part = ii · unit_torque.
    let wbase = base + SPATIAL_DIM;
    #[cfg(feature = "dim3")]
    {
        jacobians.write(wbase, im.x * unit_force.x);
        jacobians.write(wbase + 1, im.y * unit_force.y);
        jacobians.write(wbase + 2, im.z * unit_force.z);
        let inv_i = mp.inv_inertia;
        let it = (inv_i * unit_torque.extend(0.0)).truncate();
        jacobians.write(wbase + 3, it.x);
        jacobians.write(wbase + 4, it.y);
        jacobians.write(wbase + 5, it.z);
    }
    #[cfg(feature = "dim2")]
    {
        jacobians.write(wbase, im.x * unit_force.x);
        jacobians.write(wbase + 1, im.y * unit_force.y);
        jacobians.write(wbase + 2, mp.inv_inertia * unit_torque);
    }
}

/// Write a multibody link's projected `Jᵀ` row + the LU back-solved
/// `M⁻¹·Jᵀ` column into the jacobians buffer at `j_id`. Mirrors rapier's
/// `Multibody::fill_jacobians`:
///
///   1. `j[0..ndofs] = link_J^T · (unit_force, unit_torque)` — i.e. project the
///      link's body jacobian against the unit force/torque.
///   2. `wj[0..ndofs] = M⁻¹ · j[0..ndofs]` — LU back-solve in place using the
///      cached factor.
fn fill_mb_jacobians(
    jacobians: &mut [f32],
    j_id: u32,
    mb: &MultibodyInfo,
    link_id: u32,
    unit_force: Vector,
    unit_torque: AngVector,
    body_jacobians: &[f32],
    jac_start: usize,
    mass_matrices: &[f32],
    mm_start: usize,
    lu_pivots: &[u32],
    dof_start: usize,
) {
    let ndofs = mb.ndofs;
    let mb_jac_base = jac_start + mb.jacobian_offset as usize;
    let link_jac_base = mb_jac_base + (link_id as usize) * SPATIAL_DIM * (ndofs as usize);
    let link_j = MatSlice::dense(link_jac_base, SPATIAL_DIM as u32, ndofs);
    let (link_j_v, link_j_w) = link_j.rows_range_pair(0, DIM, DIM, ANG_DIM);

    // 1) j = link_J^T · (unit_force, unit_torque). Same kernel used by
    //    `fill_contact_jac_row` in `contact_constraints`.
    for k in 0..ndofs {
        let dot;
        #[cfg(feature = "dim3")]
        {
            let jv0 = body_jacobians.read(link_j_v.idx(0, k));
            let jv1 = body_jacobians.read(link_j_v.idx(1, k));
            let jv2 = body_jacobians.read(link_j_v.idx(2, k));
            let jw0 = body_jacobians.read(link_j_w.idx(0, k));
            let jw1 = body_jacobians.read(link_j_w.idx(1, k));
            let jw2 = body_jacobians.read(link_j_w.idx(2, k));
            dot = unit_force.x * jv0
                + unit_force.y * jv1
                + unit_force.z * jv2
                + unit_torque.x * jw0
                + unit_torque.y * jw1
                + unit_torque.z * jw2;
        }
        #[cfg(feature = "dim2")]
        {
            let jv0 = body_jacobians.read(link_j_v.idx(0, k));
            let jv1 = body_jacobians.read(link_j_v.idx(1, k));
            let jw0 = body_jacobians.read(link_j_w.idx(0, k));
            dot = unit_force.x * jv0 + unit_force.y * jv1 + unit_torque * jw0;
        }
        jacobians.write(j_id as usize + k as usize, dot);
    }

    // 2) wj = M⁻¹ · j (LU back-solve). Copy j into wj first so the in-place
    //    solve writes back into wj without clobbering j.
    let wj_base = wj_id(j_id, ndofs);
    for k in 0..ndofs {
        let v = jacobians.read(j_id as usize + k as usize);
        jacobians.write(wj_base + k as usize, v);
    }
    let mb_mm_base = mm_start + mb.mass_matrix_offset as usize;
    let m = MatSlice::dense(mb_mm_base, ndofs, ndofs);
    let piv_offset = dof_start + mb.first_dof as usize;
    lu_solve_in_place(mass_matrices, m, lu_pivots, piv_offset, jacobians, wj_base);
}

/// `JointConstraintHelper`-equivalent: precomputed per-joint quantities used
/// by `lock_*`, `limit_*`, `motor_*`. Mirrors the homonymous rapier struct.
struct JointConstraintHelper {
    #[cfg(feature = "dim2")]
    basis: glamx::Mat2,
    #[cfg(feature = "dim2")]
    cmat1_basis: [f32; 2],
    #[cfg(feature = "dim2")]
    cmat2_basis: [f32; 2],
    #[cfg(feature = "dim3")]
    basis: Mat3,
    #[cfg(feature = "dim3")]
    basis2: Mat3,
    #[cfg(feature = "dim3")]
    cmat1_basis: Mat3,
    #[cfg(feature = "dim3")]
    cmat2_basis: Mat3,
    #[cfg(feature = "dim3")]
    ang_basis: Mat3,
    lin_err: Vector,
    #[cfg(feature = "dim2")]
    ang_err: crate::Rotation,
    #[cfg(feature = "dim3")]
    ang_err: [f32; 3],
}

#[cfg(feature = "dim3")]
fn quat_dot(a: Quat, b: Quat) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
}

#[cfg(feature = "dim3")]
fn rotation_diff_conj1_2(q1: Quat, q2: Quat) -> Mat3 {
    let v1 = q1.xyz();
    let v2 = q2.xyz();
    let w1 = q1.w;
    let w2 = q2.w;

    let tensor_product = Mat3::from_cols(v1 * v2.x, v1 * v2.y, v1 * v2.z);
    let diag = Mat3::from_cols(
        Vec3::new(w1 * w2, 0.0, 0.0),
        Vec3::new(0.0, w1 * w2, 0.0),
        Vec3::new(0.0, 0.0, w1 * w2),
    );
    let cross_sum = gcross_matrix3(v1 * w2 + v2 * w1);
    let cross_v1 = gcross_matrix3(v1);
    let cross_v2 = gcross_matrix3(v2);

    (tensor_product + diag - cross_sum + cross_v1 * cross_v2) * 0.5
}

#[cfg(feature = "dim3")]
fn gcross_matrix3(r: Vec3) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(0.0, r.z, -r.y),
        Vec3::new(-r.z, 0.0, r.x),
        Vec3::new(r.y, -r.x, 0.0),
    )
}

#[cfg(feature = "dim2")]
fn gcross_matrix2(r: glamx::Vec2) -> glamx::Vec2 {
    glamx::Vec2::new(-r.y, r.x)
}

fn new_helper(
    frame1_in: Pose,
    frame2: Pose,
    world_com1: Vector,
    world_com2: Vector,
    locked_lin_axes: u32,
) -> JointConstraintHelper {
    let mut frame1 = frame1_in;
    let basis = rotation_to_matrix(frame1.rotation);
    let lin_err = frame2.translation - frame1.translation;

    // Snap the frame1 origin to frame2's center along free axes (rapier's
    // `JointConstraintHelper::new`), so the lock jacobians act at a single
    // point rather than at the now-relatively-displaced anchor1.
    let mut new_center1 = frame2.translation;
    for i in 0..DIM_USIZE {
        if (locked_lin_axes & (1u32 << i)) != 0 {
            let axis = basis.col_at(i);
            new_center1 -= axis * lin_err.dot(axis);
        }
    }
    frame1.translation = new_center1;

    let r1 = frame1.translation - world_com1;
    let r2 = frame2.translation - world_com2;

    #[cfg(feature = "dim2")]
    {
        let cmat1 = gcross_matrix2(r1);
        let cmat2 = gcross_matrix2(r2);
        let ang_err = frame1.rotation.inverse() * frame2.rotation;
        JointConstraintHelper {
            basis,
            cmat1_basis: [cmat1.dot(basis.col_at(0)), cmat1.dot(basis.col_at(1))],
            cmat2_basis: [cmat2.dot(basis.col_at(0)), cmat2.dot(basis.col_at(1))],
            lin_err,
            ang_err,
        }
    }
    #[cfg(feature = "dim3")]
    {
        let cmat1 = gcross_matrix3(r1);
        let cmat2 = gcross_matrix3(r2);
        let mut ang_basis = rotation_diff_conj1_2(frame1.rotation, frame2.rotation).transpose();
        let quat_err = frame1.rotation.inverse() * frame2.rotation;
        let sgn = if quat_dot(frame1.rotation, frame2.rotation) > 0.0 {
            1.0
        } else {
            -1.0
        };
        ang_basis *= sgn;
        let ang_err = [quat_err.x * sgn, quat_err.y * sgn, quat_err.z * sgn];
        JointConstraintHelper {
            basis,
            basis2: rotation_to_matrix(frame2.rotation),
            cmat1_basis: cmat1 * basis,
            cmat2_basis: cmat2 * basis,
            ang_basis,
            lin_err,
            ang_err,
        }
    }
}

#[inline]
fn pseudo_inv(x: f32) -> f32 {
    if x == 0.0 { 0.0 } else { 1.0 / x }
}

/// Bundle of per-side context the `lock_jacobians_generic` analog needs to
/// fill the jacobians for both sides of a generic constraint and stamp the
/// constraint header.
///
/// `mb` is read by value (not behind a reference / Option) so the SPIR-V
/// backend doesn't need pointers-to-arbitrary-storage. `side_kind` gates
/// whether `mb` carries meaningful data.
#[derive(Clone, Copy)]
struct SideCtx {
    side_kind: u32,
    side_id: u32,
    side_link: u32,
    ndofs: u32,
    mb: MultibodyInfo,
}

/// Mirrors rapier's `JointConstraintHelper::lock_jacobians_generic`: pack
/// per-side `Jᵀ` + `M⁻¹·Jᵀ` into the jacobians buffer and stamp a fresh
/// constraint with the side metadata. Caller fills the rhs / impulse-bound
/// fields afterwards.
fn lock_jacobians_generic(
    out: &mut MbImpulseJointConstraint,
    jacobians: &mut [f32],
    j_id_a: u32,
    j_id_b: u32,
    a: &SideCtx,
    b: &SideCtx,
    lin_jac: Vector,
    ang_jac1: AngVector,
    ang_jac2: AngVector,
    body_jacobians: &[f32],
    jac_start: usize,
    mass_matrices: &[f32],
    mm_start: usize,
    lu_pivots: &[u32],
    dof_start: usize,
    mprops: &[WorldMassProperties],
    colliders_start: usize,
) {
    out.side_a_kind = a.side_kind;
    out.side_a_id = a.side_id;
    out.side_a_link = a.side_link;
    out.ndofs_a = a.ndofs;
    out.j_id_a = j_id_a;
    if a.side_kind == SIDE_KIND_BODY {
        fill_body_jacobians(
            jacobians,
            j_id_a,
            a.side_id,
            lin_jac,
            ang_jac1,
            mprops,
            colliders_start,
        );
    } else if a.side_kind == SIDE_KIND_MB {
        fill_mb_jacobians(
            jacobians,
            j_id_a,
            &a.mb,
            a.side_link,
            lin_jac,
            ang_jac1,
            body_jacobians,
            jac_start,
            mass_matrices,
            mm_start,
            lu_pivots,
            dof_start,
        );
    }

    out.side_b_kind = b.side_kind;
    out.side_b_id = b.side_id;
    out.side_b_link = b.side_link;
    out.ndofs_b = b.ndofs;
    out.j_id_b = j_id_b;
    if b.side_kind == SIDE_KIND_BODY {
        fill_body_jacobians(
            jacobians,
            j_id_b,
            b.side_id,
            lin_jac,
            ang_jac2,
            mprops,
            colliders_start,
        );
    } else if b.side_kind == SIDE_KIND_MB {
        fill_mb_jacobians(
            jacobians,
            j_id_b,
            &b.mb,
            b.side_link,
            lin_jac,
            ang_jac2,
            body_jacobians,
            jac_start,
            mass_matrices,
            mm_start,
            lu_pivots,
            dof_start,
        );
    }

    out.kind = 1;
    out.impulse = 0.0;
    out.impulse_lo = -MAX_F32;
    out.impulse_hi = MAX_F32;
    out.inv_lhs = 0.0; // filled by `finalize_generic_constraints`.
    out.rhs = 0.0;
    out.rhs_wo_bias = 0.0;
    out.cfm_coeff = 0.0;
    out.cfm_gain = 0.0;
}

/// Compute `J · W·J` over both sides — matches the dot product rapier
/// performs in `finalize_generic_constraints` to set `inv_lhs`.
#[inline]
fn dot_j_wj(c: &MbImpulseJointConstraint, jacobians: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..c.ndofs_a {
        let j = jacobians.read(c.j_id_a as usize + i as usize);
        let wj = jacobians.read(wj_id(c.j_id_a, c.ndofs_a) + i as usize);
        acc += j * wj;
    }
    for i in 0..c.ndofs_b {
        let j = jacobians.read(c.j_id_b as usize + i as usize);
        let wj = jacobians.read(wj_id(c.j_id_b, c.ndofs_b) + i as usize);
        acc += j * wj;
    }
    acc
}

/// Populate `inv_lhs` / `cfm_gain` for each filled constraint. Mirrors
/// rapier's `finalize_generic_constraints` (orthogonalization branch is
/// disabled there too — `ORTHOGONALIZE = false`).
#[inline]
fn finalize_generic_constraint(c: &mut MbImpulseJointConstraint, jacobians: &[f32]) {
    let dot_jj = dot_j_wj(c, jacobians);
    let cfm_gain = dot_jj * c.cfm_coeff + c.cfm_gain;
    c.inv_lhs = pseudo_inv(dot_jj + cfm_gain);
    c.cfm_gain = cfm_gain;
}

#[inline]
#[cfg(feature = "dim2")]
fn ang_jac_for_axis(_helper: &JointConstraintHelper, _axis: usize) -> AngVector {
    1.0
}

#[inline]
#[cfg(feature = "dim3")]
fn ang_jac_for_axis(helper: &JointConstraintHelper, axis: usize) -> AngVector {
    helper.ang_basis.col_at(axis)
}

#[inline]
#[cfg(feature = "dim2")]
fn motor_ang_jac(_helper: &JointConstraintHelper, _axis: usize) -> AngVector {
    1.0
}

#[inline]
#[cfg(feature = "dim3")]
fn motor_ang_jac(helper: &JointConstraintHelper, axis: usize) -> AngVector {
    helper.basis.col_at(axis)
}

#[inline]
#[cfg(feature = "dim2")]
fn ang_err_axis(helper: &JointConstraintHelper, _axis: usize) -> f32 {
    crate::sin(crate::rotation_angle(helper.ang_err) * 0.5) * 2.0 * 0.5 // = sin(a/2)
}

#[inline]
#[cfg(feature = "dim3")]
fn ang_err_axis(helper: &JointConstraintHelper, axis: usize) -> f32 {
    helper.ang_err.read(axis)
}

#[inline]
fn erp_inv_dt(motor: &JointMotor, dt: f32) -> f32 {
    motor_params(motor, dt).erp_inv_dt
}

/// Lock one linear axis (`Dof(axis)` writeback). Mirrors rapier's
/// `JointConstraintHelper::lock_linear_generic`.
#[allow(clippy::too_many_arguments)]
fn lock_linear_generic(
    out: &mut MbImpulseJointConstraint,
    helper: &JointConstraintHelper,
    joint_id: u32,
    a: &SideCtx,
    b: &SideCtx,
    locked_axis: usize,
    erp_inv_dt_val: f32,
    cfm_coeff: f32,
    jacobians: &mut [f32],
    j_id_a: u32,
    j_id_b: u32,
    body_jacobians: &[f32],
    jac_start: usize,
    mass_matrices: &[f32],
    mm_start: usize,
    lu_pivots: &[u32],
    dof_start: usize,
    mprops: &[WorldMassProperties],
    colliders_start: usize,
) {
    let lin_jac = helper.basis.col_at(locked_axis);
    #[cfg(feature = "dim2")]
    let ang_jac1 = helper.cmat1_basis.read(locked_axis);
    #[cfg(feature = "dim2")]
    let ang_jac2 = helper.cmat2_basis.read(locked_axis);
    #[cfg(feature = "dim3")]
    let ang_jac1 = helper.cmat1_basis.col_at(locked_axis);
    #[cfg(feature = "dim3")]
    let ang_jac2 = helper.cmat2_basis.col_at(locked_axis);

    lock_jacobians_generic(
        out,
        jacobians,
        j_id_a,
        j_id_b,
        a,
        b,
        lin_jac,
        ang_jac1,
        ang_jac2,
        body_jacobians,
        jac_start,
        mass_matrices,
        mm_start,
        lu_pivots,
        dof_start,
        mprops,
        colliders_start,
    );
    out.joint_id = joint_id;
    out.writeback_kind = 0;
    out.writeback_axis = locked_axis as u32;
    out.cfm_coeff = cfm_coeff;
    let rhs_bias = lin_jac.dot(helper.lin_err) * erp_inv_dt_val;
    out.rhs_wo_bias = 0.0;
    out.rhs = rhs_bias;
}

/// Lock one angular axis. Mirrors `lock_angular_generic`.
#[allow(clippy::too_many_arguments)]
fn lock_angular_generic(
    out: &mut MbImpulseJointConstraint,
    helper: &JointConstraintHelper,
    joint_id: u32,
    a: &SideCtx,
    b: &SideCtx,
    locked_axis: usize,
    erp_inv_dt_val: f32,
    cfm_coeff: f32,
    jacobians: &mut [f32],
    j_id_a: u32,
    j_id_b: u32,
    body_jacobians: &[f32],
    jac_start: usize,
    mass_matrices: &[f32],
    mm_start: usize,
    lu_pivots: &[u32],
    dof_start: usize,
    mprops: &[WorldMassProperties],
    colliders_start: usize,
) {
    let ang_jac = ang_jac_for_axis(helper, locked_axis);
    lock_jacobians_generic(
        out,
        jacobians,
        j_id_a,
        j_id_b,
        a,
        b,
        Vector::ZERO,
        ang_jac,
        ang_jac,
        body_jacobians,
        jac_start,
        mass_matrices,
        mm_start,
        lu_pivots,
        dof_start,
        mprops,
        colliders_start,
    );
    out.joint_id = joint_id;
    out.writeback_kind = 0;
    out.writeback_axis = (DIM_USIZE + locked_axis) as u32;
    out.cfm_coeff = cfm_coeff;
    let rhs_bias = ang_err_axis(helper, locked_axis) * erp_inv_dt_val;
    out.rhs_wo_bias = 0.0;
    out.rhs = rhs_bias;
}

/// Limit one linear axis. Mirrors `limit_linear_generic`.
#[allow(clippy::too_many_arguments)]
fn limit_linear_generic(
    out: &mut MbImpulseJointConstraint,
    helper: &JointConstraintHelper,
    joint_id: u32,
    a: &SideCtx,
    b: &SideCtx,
    limited_axis: usize,
    limits: [f32; 2],
    erp_inv_dt_val: f32,
    cfm_coeff: f32,
    jacobians: &mut [f32],
    j_id_a: u32,
    j_id_b: u32,
    body_jacobians: &[f32],
    jac_start: usize,
    mass_matrices: &[f32],
    mm_start: usize,
    lu_pivots: &[u32],
    dof_start: usize,
    mprops: &[WorldMassProperties],
    colliders_start: usize,
) {
    let lin_jac = helper.basis.col_at(limited_axis);
    #[cfg(feature = "dim2")]
    let ang_jac1 = helper.cmat1_basis.read(limited_axis);
    #[cfg(feature = "dim2")]
    let ang_jac2 = helper.cmat2_basis.read(limited_axis);
    #[cfg(feature = "dim3")]
    let ang_jac1 = helper.cmat1_basis.col_at(limited_axis);
    #[cfg(feature = "dim3")]
    let ang_jac2 = helper.cmat2_basis.col_at(limited_axis);

    lock_jacobians_generic(
        out,
        jacobians,
        j_id_a,
        j_id_b,
        a,
        b,
        lin_jac,
        ang_jac1,
        ang_jac2,
        body_jacobians,
        jac_start,
        mass_matrices,
        mm_start,
        lu_pivots,
        dof_start,
        mprops,
        colliders_start,
    );
    out.joint_id = joint_id;
    out.writeback_kind = 1;
    out.writeback_axis = limited_axis as u32;
    out.cfm_coeff = cfm_coeff;

    let dist = helper.lin_err.dot(lin_jac);
    let min_enabled = dist <= limits[0];
    let max_enabled = limits[1] <= dist;
    let rhs_bias = ((dist - limits[1]).max(0.0) - (limits[0] - dist).max(0.0)) * erp_inv_dt_val;
    out.rhs_wo_bias = 0.0;
    out.rhs = rhs_bias;
    out.impulse_lo = if min_enabled { -MAX_F32 } else { 0.0 };
    out.impulse_hi = if max_enabled { MAX_F32 } else { 0.0 };
}

/// Limit one angular axis. Mirrors `limit_angular_generic`.
#[allow(clippy::too_many_arguments)]
fn limit_angular_generic(
    out: &mut MbImpulseJointConstraint,
    helper: &JointConstraintHelper,
    joint_id: u32,
    a: &SideCtx,
    b: &SideCtx,
    limited_axis: usize,
    limits: [f32; 2],
    erp_inv_dt_val: f32,
    cfm_coeff: f32,
    jacobians: &mut [f32],
    j_id_a: u32,
    j_id_b: u32,
    body_jacobians: &[f32],
    jac_start: usize,
    mass_matrices: &[f32],
    mm_start: usize,
    lu_pivots: &[u32],
    dof_start: usize,
    mprops: &[WorldMassProperties],
    colliders_start: usize,
) {
    let ang_jac = ang_jac_for_axis(helper, limited_axis);
    lock_jacobians_generic(
        out,
        jacobians,
        j_id_a,
        j_id_b,
        a,
        b,
        Vector::ZERO,
        ang_jac,
        ang_jac,
        body_jacobians,
        jac_start,
        mass_matrices,
        mm_start,
        lu_pivots,
        dof_start,
        mprops,
        colliders_start,
    );
    out.joint_id = joint_id;
    out.writeback_kind = 1;
    out.writeback_axis = (DIM_USIZE + limited_axis) as u32;
    out.cfm_coeff = cfm_coeff;

    let s_limits = [crate::sin(limits[0] * 0.5), crate::sin(limits[1] * 0.5)];
    let s_ang = ang_err_axis(helper, limited_axis);
    let min_enabled = s_ang <= s_limits[0];
    let max_enabled = s_limits[1] <= s_ang;
    let rhs_bias =
        ((s_ang - s_limits[1]).max(0.0) - (s_limits[0] - s_ang).max(0.0)) * erp_inv_dt_val;
    out.rhs_wo_bias = 0.0;
    out.rhs = rhs_bias;
    out.impulse_lo = if min_enabled { -MAX_F32 } else { 0.0 };
    out.impulse_hi = if max_enabled { MAX_F32 } else { 0.0 };
}

/// Linear motor. Mirrors `motor_linear_generic`.
#[allow(clippy::too_many_arguments)]
fn motor_linear_generic(
    out: &mut MbImpulseJointConstraint,
    helper: &JointConstraintHelper,
    joint_id: u32,
    a: &SideCtx,
    b: &SideCtx,
    motor_axis: usize,
    motor: &JointMotor,
    dt: f32,
    jacobians: &mut [f32],
    j_id_a: u32,
    j_id_b: u32,
    body_jacobians: &[f32],
    jac_start: usize,
    mass_matrices: &[f32],
    mm_start: usize,
    lu_pivots: &[u32],
    dof_start: usize,
    mprops: &[WorldMassProperties],
    colliders_start: usize,
) {
    let mp = motor_params(motor, dt);
    let lin_jac = helper.basis.col_at(motor_axis);
    #[cfg(feature = "dim2")]
    let ang_jac1 = helper.cmat1_basis.read(motor_axis);
    #[cfg(feature = "dim2")]
    let ang_jac2 = helper.cmat2_basis.read(motor_axis);
    #[cfg(feature = "dim3")]
    let ang_jac1 = helper.cmat1_basis.col_at(motor_axis);
    #[cfg(feature = "dim3")]
    let ang_jac2 = helper.cmat2_basis.col_at(motor_axis);

    lock_jacobians_generic(
        out,
        jacobians,
        j_id_a,
        j_id_b,
        a,
        b,
        lin_jac,
        ang_jac1,
        ang_jac2,
        body_jacobians,
        jac_start,
        mass_matrices,
        mm_start,
        lu_pivots,
        dof_start,
        mprops,
        colliders_start,
    );
    out.joint_id = joint_id;
    out.writeback_kind = 2;
    out.writeback_axis = motor_axis as u32;

    let mut rhs_wo_bias = 0.0;
    if mp.erp_inv_dt != 0.0 {
        let dist = helper.lin_err.dot(lin_jac);
        rhs_wo_bias += (dist - mp.target_pos) * mp.erp_inv_dt;
    }
    rhs_wo_bias += -mp.target_vel;

    out.cfm_coeff = mp.cfm_coeff;
    out.cfm_gain = mp.cfm_gain;
    out.impulse_lo = -mp.max_impulse;
    out.impulse_hi = mp.max_impulse;
    out.rhs = rhs_wo_bias;
    out.rhs_wo_bias = rhs_wo_bias;
}

/// Angular motor. Mirrors `motor_angular_generic`.
#[allow(clippy::too_many_arguments)]
fn motor_angular_generic(
    out: &mut MbImpulseJointConstraint,
    helper: &JointConstraintHelper,
    joint_id: u32,
    a: &SideCtx,
    b: &SideCtx,
    motor_axis: usize,
    motor: &JointMotor,
    dt: f32,
    jacobians: &mut [f32],
    j_id_a: u32,
    j_id_b: u32,
    body_jacobians: &[f32],
    jac_start: usize,
    mass_matrices: &[f32],
    mm_start: usize,
    lu_pivots: &[u32],
    dof_start: usize,
    mprops: &[WorldMassProperties],
    colliders_start: usize,
) {
    let mp = motor_params(motor, dt);
    let ang_jac = motor_ang_jac(helper, motor_axis);
    lock_jacobians_generic(
        out,
        jacobians,
        j_id_a,
        j_id_b,
        a,
        b,
        Vector::ZERO,
        ang_jac,
        ang_jac,
        body_jacobians,
        jac_start,
        mass_matrices,
        mm_start,
        lu_pivots,
        dof_start,
        mprops,
        colliders_start,
    );
    out.joint_id = joint_id;
    out.writeback_kind = 2;
    out.writeback_axis = (DIM_USIZE + motor_axis) as u32;

    let mut rhs_wo_bias = 0.0;
    if mp.erp_inv_dt != 0.0 {
        #[cfg(feature = "dim3")]
        let s_ang_dist = helper.ang_err.read(motor_axis);
        #[cfg(feature = "dim2")]
        let s_ang_dist = crate::sin(crate::rotation_angle(helper.ang_err) * 0.5);
        let s_target_ang = crate::sin(mp.target_pos * 0.5);
        // smallest_abs_diff_between_sin_angles — using the simpler form
        // (dist - target) since the two-pi wrap concerns the rotation part
        // and we operate on sin-half-angles already.
        rhs_wo_bias += (s_ang_dist - s_target_ang) * mp.erp_inv_dt;
    }
    rhs_wo_bias += -mp.target_vel;

    out.cfm_coeff = mp.cfm_coeff;
    out.cfm_gain = mp.cfm_gain;
    out.impulse_lo = -mp.max_impulse;
    out.impulse_hi = mp.max_impulse;
    out.rhs = rhs_wo_bias;
    out.rhs_wo_bias = rhs_wo_bias;
}

/// (Re)build all axis constraints of every multibody-touching impulse joint.
///
/// Run once per substep. Mirrors rapier's
/// `JointGenericExternalConstraintBuilder::update`: rebuilds all `J / W·J`
/// rows + biases from current poses / mass matrix.
///
/// One thread per joint slot — joints are independent at build time
/// (they only share read-only access to body / multibody state, and write
/// to disjoint constraint slots). The PGS sweep that follows is what needs
/// coloring for race-freedom; here every thread is safe to run in parallel.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_update_impulse_joint_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] builders: &[MbImpulseJointBuilder],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    constraints: &mut [MbImpulseJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] jacobians: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] dt_uniform: &f32,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)]
    links_workspace: &[MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 2)] body_jacobians: &[f32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 3)] mass_matrices: &[f32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 4)] lu_pivots: &[u32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 5)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 6)] mprops: &[WorldMassProperties],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * 64;
    let batch_id = invocation_id.y;
    let cap = batch_ids.mb_imp_joints_batch_capacity;
    if invocation_id.x >= cap {
        return;
    }
    let dt = *dt_uniform;

    let joints_start = batch_ids.mb_imp_joints_start(batch_id);
    let cons_start = batch_ids.mb_imp_joint_constraints_start(batch_id);
    let jac_buf_start = batch_ids.mb_imp_joint_jacobians_start(batch_id);
    let mb_start = batch_ids.mb_start(batch_id);
    let links_start = batch_ids.links_start(batch_id);
    let body_jac_start = batch_ids.jac_start(batch_id);
    let mm_start = batch_ids.mm_start(batch_id);
    let dof_start = batch_ids.dof_start(batch_id);
    let colliders_start = batch_ids.coll_start(batch_id);

    // Loop chunked across `num_threads` so a single workgroup row processes
    // all joints in a batch (matches `gpu_init_joint_constraints` style).
    // Iterating to `cap` instead of `len` lets us drop the per-batch
    // `num_joints` storage binding — the host pads unused builder slots
    // with `side_a_kind == SIDE_KIND_FIXED && side_b_kind == SIDE_KIND_FIXED`
    // which we use as the inactive-slot sentinel below.
    let mut i = invocation_id.x;
    while i < cap {
        let builder = builders.read(joints_start + i as usize);
        let is_dummy =
            builder.side_a_kind == SIDE_KIND_FIXED && builder.side_b_kind == SIDE_KIND_FIXED;
        if !is_dummy {
            update_one_joint(
                &builder,
                constraints,
                cons_start,
                jacobians,
                jac_buf_start,
                multibody_info,
                mb_start,
                links_workspace,
                links_start,
                body_jacobians,
                body_jac_start,
                mass_matrices,
                mm_start,
                lu_pivots,
                dof_start,
                poses,
                colliders_start,
                mprops,
                dt,
            );
        }
        i += num_threads;
    }
}

#[allow(clippy::too_many_arguments)]
fn update_one_joint(
    builder: &MbImpulseJointBuilder,
    constraints: &mut [MbImpulseJointConstraint],
    cons_start: usize,
    jacobians: &mut [f32],
    jac_buf_start: usize,
    multibody_info: &[MultibodyInfo],
    mb_start: usize,
    links_workspace: &[MultibodyLinkWorkspace],
    links_start: usize,
    body_jacobians: &[f32],
    body_jac_start: usize,
    mass_matrices: &[f32],
    mm_start: usize,
    lu_pivots: &[u32],
    dof_start: usize,
    poses: &[Pose],
    colliders_start: usize,
    mprops: &[WorldMassProperties],
    dt: f32,
) {
    let cons_base = cons_start + builder.constraint_id as usize;
    // Mark all axis-constraint slots inactive up-front; the active branches
    // below overwrite the live ones (rapier rebuilds the entire
    // `out[start..len]` slab each `update` call, so unfilled slots are
    // guaranteed inactive).
    for s in 0..MAX_AXIS_CONSTRAINTS {
        let mut cz = constraints.read(cons_base + s as usize);
        cz.kind = 0;
        cz.impulse = 0.0;
        constraints.write(cons_base + s as usize, cz);
    }

    // Resolve per-side multibody descriptors (read by value to avoid
    // SPIR-V's "pointer to arbitrary element" restriction). Free / fixed
    // sides ignore the read.
    let mb_a = if builder.side_a_kind == SIDE_KIND_MB {
        multibody_info.read(mb_start + builder.side_a_id as usize)
    } else {
        MultibodyInfo::default()
    };
    let mb_b = if builder.side_b_kind == SIDE_KIND_MB {
        multibody_info.read(mb_start + builder.side_b_id as usize)
    } else {
        MultibodyInfo::default()
    };

    let pose_a = side_world_pose(
        builder.side_a_kind,
        builder.side_a_id,
        builder.side_a_link,
        &mb_a,
        links_workspace,
        links_start,
        poses,
        colliders_start,
    );
    let pose_b = side_world_pose(
        builder.side_b_kind,
        builder.side_b_id,
        builder.side_b_link,
        &mb_b,
        links_workspace,
        links_start,
        poses,
        colliders_start,
    );

    let frame1 = pose_a * builder.joint.local_frame_a;
    let frame2 = pose_b * builder.joint.local_frame_b;
    let world_com1 = pose_a.translation;
    let world_com2 = pose_b.translation;

    let helper = new_helper(
        frame1,
        frame2,
        world_com1,
        world_com2,
        builder.joint.locked_axes,
    );

    let ndofs_a = if builder.side_a_kind == SIDE_KIND_BODY {
        SPATIAL_DIM as u32
    } else if builder.side_a_kind == SIDE_KIND_MB {
        mb_a.ndofs
    } else {
        0
    };
    let ndofs_b = if builder.side_b_kind == SIDE_KIND_BODY {
        SPATIAL_DIM as u32
    } else if builder.side_b_kind == SIDE_KIND_MB {
        mb_b.ndofs
    } else {
        0
    };
    let a_ctx = SideCtx {
        side_kind: builder.side_a_kind,
        side_id: builder.side_a_id,
        side_link: builder.side_a_link,
        ndofs: ndofs_a,
        mb: mb_a,
    };
    let b_ctx = SideCtx {
        side_kind: builder.side_b_kind,
        side_id: builder.side_b_id,
        side_link: builder.side_b_link,
        ndofs: ndofs_b,
        mb: mb_b,
    };
    let stride = axis_stride(ndofs_a, ndofs_b);
    let j_base = jac_buf_start + builder.jacobian_offset as usize;

    // SimParams is per-batch in the rest of the codebase, but this kernel
    // only needs `dt` for motor params. `erp_inv_dt` / `cfm_coeff` for
    // locks/limits use rapier's defaults — match the values produced by
    // `joint_erp_inv_dt(params)` / `joint_cfm_coeff(params)` from
    // `sim_params.rs` by computing them here too.
    //
    // Defaults when no per-joint softness is set: rapier picks
    // `softness.erp_inv_dt(dt) = 0.8 / dt` and `cfm_coeff = 0.0`.
    // (The host-side `SimParams` exposes the same; for simplicity we
    // bake the rapier default here — the existing free-body GPU joints
    // do the same via `joint_erp_inv_dt(params)`.)
    let inv_dt = if dt != 0.0 { 1.0 / dt } else { 0.0 };
    let lock_erp_inv_dt = 0.8 * inv_dt;
    let lock_cfm_coeff = 0.0f32;

    let locked_axes = builder.joint.locked_axes;
    let motor_axes = builder.joint.motor_axes & !locked_axes;
    let limit_axes = builder.joint.limit_axes & !locked_axes;

    let mut len = 0u32;
    let mut j_off = j_base as u32;

    // Order matches rapier's `lock_axes`: motors → locks → limits.
    // Within each kind: angular axes before linear axes.

    // Angular motors.
    for i in DIM..(SPATIAL_DIM as u32) {
        if (motor_axes & (1 << i)) != 0 {
            if len >= MAX_AXIS_CONSTRAINTS {
                break;
            }
            let mut c = constraints.read(cons_base + len as usize);
            let j_id_a = j_off;
            let j_id_b = j_off + 2 * ndofs_a;
            motor_angular_generic(
                &mut c,
                &helper,
                builder.joint_id,
                &a_ctx,
                &b_ctx,
                (i - DIM) as usize,
                builder.joint.motors.at(i as usize),
                dt,
                jacobians,
                j_id_a,
                j_id_b,
                body_jacobians,
                body_jac_start,
                mass_matrices,
                mm_start,
                lu_pivots,
                dof_start,
                mprops,
                colliders_start,
            );
            finalize_generic_constraint(&mut c, jacobians);
            constraints.write(cons_base + len as usize, c);
            len += 1;
            j_off += stride;
        }
    }

    // Linear motors.
    for i in 0..(DIM as u32) {
        if (motor_axes & (1 << i)) != 0 {
            if len >= MAX_AXIS_CONSTRAINTS {
                break;
            }
            let mut c = constraints.read(cons_base + len as usize);
            let j_id_a = j_off;
            let j_id_b = j_off + 2 * ndofs_a;
            motor_linear_generic(
                &mut c,
                &helper,
                builder.joint_id,
                &a_ctx,
                &b_ctx,
                i as usize,
                builder.joint.motors.at(i as usize),
                dt,
                jacobians,
                j_id_a,
                j_id_b,
                body_jacobians,
                body_jac_start,
                mass_matrices,
                mm_start,
                lu_pivots,
                dof_start,
                mprops,
                colliders_start,
            );
            finalize_generic_constraint(&mut c, jacobians);
            constraints.write(cons_base + len as usize, c);
            len += 1;
            j_off += stride;
        }
    }

    // Angular locks.
    for i in DIM..(SPATIAL_DIM as u32) {
        if (locked_axes & (1 << i)) != 0 {
            if len >= MAX_AXIS_CONSTRAINTS {
                break;
            }
            let mut c = constraints.read(cons_base + len as usize);
            let j_id_a = j_off;
            let j_id_b = j_off + 2 * ndofs_a;
            lock_angular_generic(
                &mut c,
                &helper,
                builder.joint_id,
                &a_ctx,
                &b_ctx,
                (i - DIM) as usize,
                lock_erp_inv_dt,
                lock_cfm_coeff,
                jacobians,
                j_id_a,
                j_id_b,
                body_jacobians,
                body_jac_start,
                mass_matrices,
                mm_start,
                lu_pivots,
                dof_start,
                mprops,
                colliders_start,
            );
            finalize_generic_constraint(&mut c, jacobians);
            constraints.write(cons_base + len as usize, c);
            len += 1;
            j_off += stride;
        }
    }

    // Linear locks.
    for i in 0..(DIM as u32) {
        if (locked_axes & (1 << i)) != 0 {
            if len >= MAX_AXIS_CONSTRAINTS {
                break;
            }
            let mut c = constraints.read(cons_base + len as usize);
            let j_id_a = j_off;
            let j_id_b = j_off + 2 * ndofs_a;
            lock_linear_generic(
                &mut c,
                &helper,
                builder.joint_id,
                &a_ctx,
                &b_ctx,
                i as usize,
                lock_erp_inv_dt,
                lock_cfm_coeff,
                jacobians,
                j_id_a,
                j_id_b,
                body_jacobians,
                body_jac_start,
                mass_matrices,
                mm_start,
                lu_pivots,
                dof_start,
                mprops,
                colliders_start,
            );
            finalize_generic_constraint(&mut c, jacobians);
            constraints.write(cons_base + len as usize, c);
            len += 1;
            j_off += stride;
        }
    }

    // Angular limits.
    for i in DIM..(SPATIAL_DIM as u32) {
        if (limit_axes & (1 << i)) != 0 {
            if len >= MAX_AXIS_CONSTRAINTS {
                break;
            }
            let mut c = constraints.read(cons_base + len as usize);
            let j_id_a = j_off;
            let j_id_b = j_off + 2 * ndofs_a;
            let lim = builder.joint.limits.at(i as usize);
            limit_angular_generic(
                &mut c,
                &helper,
                builder.joint_id,
                &a_ctx,
                &b_ctx,
                (i - DIM) as usize,
                [lim.min, lim.max],
                lock_erp_inv_dt,
                lock_cfm_coeff,
                jacobians,
                j_id_a,
                j_id_b,
                body_jacobians,
                body_jac_start,
                mass_matrices,
                mm_start,
                lu_pivots,
                dof_start,
                mprops,
                colliders_start,
            );
            finalize_generic_constraint(&mut c, jacobians);
            constraints.write(cons_base + len as usize, c);
            len += 1;
            j_off += stride;
        }
    }

    // Linear limits.
    for i in 0..(DIM as u32) {
        if (limit_axes & (1 << i)) != 0 {
            if len >= MAX_AXIS_CONSTRAINTS {
                break;
            }
            let mut c = constraints.read(cons_base + len as usize);
            let j_id_a = j_off;
            let j_id_b = j_off + 2 * ndofs_a;
            let lim = builder.joint.limits.at(i as usize);
            limit_linear_generic(
                &mut c,
                &helper,
                builder.joint_id,
                &a_ctx,
                &b_ctx,
                i as usize,
                [lim.min, lim.max],
                lock_erp_inv_dt,
                lock_cfm_coeff,
                jacobians,
                j_id_a,
                j_id_b,
                body_jacobians,
                body_jac_start,
                mass_matrices,
                mm_start,
                lu_pivots,
                dof_start,
                mprops,
                colliders_start,
            );
            finalize_generic_constraint(&mut c, jacobians);
            constraints.write(cons_base + len as usize, c);
            len += 1;
            j_off += stride;
        }
    }

    // Silence unused-variable warning on 2D where ANG_AXES_MASK isn't read.
    let _ = ANG_AXES_MASK;
    let _ = LIN_AXES_MASK;
}

/// Look up the world-space pose of a side. Free-body sides read from the
/// shared `poses` buffer (COM-centered solver pose); multibody sides take
/// their link's `local_to_world` from the multibody workspace (which also
/// stores body-origin = COM-centered, since multibody links have a zeroed
/// `local_com`, as set up by the host pipeline). The `mb` argument is read
/// by value to keep SPIR-V happy and is only meaningful when `side_kind ==
/// SIDE_KIND_MB`.
#[inline]
fn side_world_pose(
    side_kind: u32,
    side_id: u32,
    side_link: u32,
    mb: &MultibodyInfo,
    links_workspace: &[MultibodyLinkWorkspace],
    links_start: usize,
    poses: &[Pose],
    colliders_start: usize,
) -> Pose {
    if side_kind == SIDE_KIND_FIXED {
        return Pose::IDENTITY;
    }
    if side_kind == SIDE_KIND_BODY {
        return poses.read(colliders_start + side_id as usize);
    }
    let link_global = links_start + mb.first_link as usize + side_link as usize;
    links_workspace.read(link_global).local_to_world
}

/// One PGS sweep over the multibody-touching impulse-joint axis constraints
/// of a single color — **one workgroup per joint**, the 32 lanes cooperating
/// on that joint's per-axis `J·v` reductions and `W·J` applies.
///
/// Joints are graph-colored at init time (see `set_impulse_joints`): within
/// one color no two joints share a multibody or a free body, so every joint
/// of the current color touches disjoint mutable state and runs race-free in
/// parallel. The host dispatches one color per iteration, so a full sweep is
/// an exact sequential Gauss–Seidel sweep in color-sorted order.
///
/// Workgroup `wg_id.x` owns the joint at sorted-builder slot
/// `start + wg_id.x`; colors smaller than the dispatch grid leave trailing
/// workgroups idle via a workgroup-uniform early return. The per-axis loop
/// stays sequential (Gauss–Seidel within the joint — axis `s+1` reads the
/// velocities axis `s` just wrote), but the divergent per-joint work is gone:
/// lanes now split the dot products / applies for the *same* joint instead of
/// straddling different joints.
#[spirv_bindgen]
#[spirv(compute(threads(32, 1, 1)))]
pub fn gpu_mb_solve_impulse_joint_constraints(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] builders: &[MbImpulseJointBuilder],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    constraints: &mut [MbImpulseJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] jacobians: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] all_color_groups: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batch_ids: &BatchIndices,
    #[spirv(uniform, descriptor_set = 0, binding = 5)] curr_color: &u32,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] dof_state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 2)] solver_vels: &mut [Velocity],
    // Per-lane scratch for the J·v tree reductions.
    #[spirv(workgroup)] partial: &mut [f32; LANES as usize],
) {
    let batch_id = wg_id.y;
    let lane = lid.x;

    let joints_start = batch_ids.mb_imp_joints_start(batch_id);
    let cons_start = batch_ids.mb_imp_joint_constraints_start(batch_id);
    let mb_start = batch_ids.mb_start(batch_id);
    let dof_start = batch_ids.dof_start(batch_id);
    let colliders_start = batch_ids.coll_start(batch_id);

    // `color_groups` is a per-batch prefix-sum over the color-sorted
    // builders: color `c` owns the sorted-builder range
    // `[color_groups[c-1], color_groups[c])` (start `0` for color `0`).
    let color = *curr_color as usize;
    let color_groups = batch_ids.mb_imp_joint_color_groups_batch(batch_id, all_color_groups);
    let start = if color > 0 { color_groups[color - 1] } else { 0 };
    let end = color_groups[color];

    let mut j = start + wg_id.x;
    let workgroup_is_active = j < end;
    if !workgroup_is_active {
        // Technically, if we enter here, we should return. However, on the web, a return would
        // break uniform control flow. This could be avoided with a `workgroupUniformLoad` but
        // that’s not supported by rust-gpu.
        j = start; // Any valid index will do.
    }

    let builder = builders.at(joints_start + j as usize);
    let cons_base = cons_start + builder.constraint_id as usize;

    // Per-multibody dof base: same for every axis constraint of this joint.
    let dof_base_a = if builder.side_a_kind == SIDE_KIND_MB {
        let mb = multibody_info.at(mb_start + builder.side_a_id as usize);
        dof_start + mb.first_dof as usize
    } else {
        0
    };
    let dof_base_b = if builder.side_b_kind == SIDE_KIND_MB {
        let mb = multibody_info.at(mb_start + builder.side_b_id as usize);
        dof_start + mb.first_dof as usize
    } else {
        0
    };

    // TODO(PERF): load jacobians into shared memory and keep the velocity deltat on shared
    //             memory and only writeback after all the axis constraints are solved.
    for s in 0..MAX_AXIS_CONSTRAINTS {
        let c = constraints.at_mut(cons_base + s as usize);
        let active = workgroup_is_active && c.kind != 0;

        // dvel = J_b · v_b - J_a · v_a   (rapier's `vel2 - vel1`).
        let v1 = side_dot_vel_par(
            active,
            c.side_a_kind,
            c.j_id_a,
            c.ndofs_a,
            c.side_a_id,
            jacobians,
            dof_state,
            dof_base_a,
            solver_vels,
            colliders_start,
            lane,
            partial,
        );
        let v2 = side_dot_vel_par(
            active,
            c.side_b_kind,
            c.j_id_b,
            c.ndofs_b,
            c.side_b_id,
            jacobians,
            dof_state,
            dof_base_b,
            solver_vels,
            colliders_start,
            lane,
            partial,
        );

        let delta = if active {
            let dvel = c.rhs + (v2 - v1);
            let total = (c.impulse + c.inv_lhs * (dvel - c.cfm_gain * c.impulse))
                // NOTE: should be `clamp`, but `clamp` breaks uniform control flow for some reasons.
                .max(c.impulse_lo).min(c.impulse_hi);
            let d = total - c.impulse;
            if lane == 0 {
                c.impulse = total;
            }
            d
        } else {
            0.0f32
        };

        // Apply ±delta · W·J: sign +1 for side A, -1 for side B (matches
        // rapier `solver_vel1.axpy(delta_impulse, &wj1, 1.0)`,
        // `solver_vel2.axpy(-delta_impulse, &wj2, 1.0)`).
        side_apply_impulse_par(
            active,
            c.side_a_kind,
            c.j_id_a,
            c.ndofs_a,
            c.side_a_id,
            1.0,
            delta,
            jacobians,
            dof_state,
            dof_base_a,
            solver_vels,
            colliders_start,
            lane,
        );
        side_apply_impulse_par(
            active,
            c.side_b_kind,
            c.j_id_b,
            c.ndofs_b,
            c.side_b_id,
            -1.0,
            delta,
            jacobians,
            dof_state,
            dof_base_b,
            solver_vels,
            colliders_start,
            lane,
        );

        workgroup_memory_barrier_with_group_sync();
    }
}

/// Strip the positional bias from each active constraint's `rhs` for the
/// stabilization sweep — mirrors `gpu_mb_remove_joint_constraint_bias`.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_remove_impulse_joint_constraint_bias(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] builders: &[MbImpulseJointBuilder],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    constraints: &mut [MbImpulseJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] num_joints: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * 64;
    let batch_id = invocation_id.y;
    let len = num_joints.read(batch_id as usize);
    let joints_start = batch_ids.mb_imp_joints_start(batch_id);
    let cons_start = batch_ids.mb_imp_joint_constraints_start(batch_id);
    let mut i = invocation_id.x;
    while i < len {
        let builder = builders.at(joints_start + i as usize);
        let cons_base = cons_start + builder.constraint_id as usize;
        for s in 0..MAX_AXIS_CONSTRAINTS {
            let c = constraints.at_mut(cons_base + s as usize);
            if c.kind != 0 {
                c.rhs = c.rhs_wo_bias;
            }
        }
        i += num_threads;
    }
}
