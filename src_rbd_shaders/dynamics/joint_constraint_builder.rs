//! Joint constraint builder
//!
//! This module contains functions to build and update joint constraints from
//! joint definitions and body states.

#[cfg(feature = "dim2")]
use crate::rotation_angle;
#[cfg(feature = "dim2")]
use crate::{Rotation};
use crate::{gdot, rotation_to_matrix, AngVector, Pose, Vector};
use khal_std::index::MaybeIndexUnchecked;
use super::body::{Velocity, WorldMassProperties};
use super::joint::{
    motor_params, GenericJoint, MotorParameters, ANG_AXES_MASK, LIN_AXES_MASK, SPATIAL_DIM,
};
use super::joint_constraint::{JointConstraint, JointConstraintElement, JointSolverBody};
use super::sim_params::{inv_dt, joint_cfm_coeff, joint_erp_inv_dt, SimParams, TWO_PI};
use crate::utils::{Slice, SliceMut};

#[cfg(feature = "dim2")]
use glamx::{Mat2, Vec2};
#[cfg(feature = "dim3")]
use glamx::{Mat3, Quat, Vec2, Vec3};

/// Maximum value for unbounded impulses.
const MAX: f32 = 1.0e20;

#[cfg(feature = "dim2")]
const DIM: usize = 2;
#[cfg(feature = "dim3")]
const DIM: usize = 3;

/// Builder data for constructing joint constraints.
#[derive(Clone, Copy)]
#[cfg_attr(not(any(target_arch = "spirv", target_arch = "nvptx64")), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct JointConstraintBuilder {
    pub body1: u32,
    pub body2: u32,
    pub joint_id: u32,
    pub constraint_id: u32,
    pub joint: GenericJoint,
}

/// Helper structure for building joint constraints.
#[derive(Clone, Copy)]
pub struct JointConstraintHelper {
    #[cfg(feature = "dim2")]
    pub basis: Mat2,
    #[cfg(feature = "dim2")]
    pub cmat1_basis: [f32; 2],
    #[cfg(feature = "dim2")]
    pub cmat2_basis: [f32; 2],
    #[cfg(feature = "dim2")]
    pub lin_err: Vector,
    #[cfg(feature = "dim2")]
    pub ang_err: Rotation,

    #[cfg(feature = "dim3")]
    pub basis: Mat3,
    #[cfg(feature = "dim3")]
    pub basis2: Mat3, // TODO: used only for angular coupling. Can we avoid storing this?
    #[cfg(feature = "dim3")]
    pub cmat1_basis: Mat3,
    #[cfg(feature = "dim3")]
    pub cmat2_basis: Mat3,
    #[cfg(feature = "dim3")]
    pub ang_basis: Mat3,
    #[cfg(feature = "dim3")]
    pub lin_err: Vector,
    #[cfg(feature = "dim3")]
    pub ang_err: [f32; 3], // Imaginary part of the angular error quaternion.
}

/// Helper function for pseudo inverse.
fn pseudo_inv(x: f32) -> f32 {
    if x == 0.0 {
        0.0
    } else {
        1.0 / x
    }
}

#[cfg(feature = "dim2")]
fn gcross_matrix(r: Vec2) -> Vec2 {
    Vec2::new(-r.y, r.x)
}

#[cfg(feature = "dim3")]
fn gcross_matrix(r: Vec3) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(0.0, r.z, -r.y),
        Vec3::new(-r.z, 0.0, r.x),
        Vec3::new(r.y, -r.x, 0.0),
    )
}

/// Computes the smallest absolute difference between two angles.
fn smallest_abs_diff_between_angles(a: f32, b: f32) -> f32 {
    // Select the smallest path among the two angles to reach the target.
    let s_err = a - b;
    let sgn = if s_err < 0.0 { -1.0 } else { 1.0 };
    let s_err_complement = s_err - sgn * TWO_PI;
    let s_err_is_smallest = s_err.abs() < s_err_complement.abs();
    if s_err_is_smallest {
        s_err
    } else {
        s_err_complement
    }
}

#[cfg(feature = "dim2")]
/// Creates a new joint constraint helper (2D version).
pub fn new_helper(
    frame1_: Pose,
    frame2: Pose,
    world_com1: Vector,
    world_com2: Vector,
    locked_lin_axes: u32,
) -> JointConstraintHelper {
    let mut frame1 = frame1_;
    let basis = rotation_to_matrix(frame1.rotation);
    let lin_err = frame2.translation - frame1.translation;

    // Adjust the point of application of the force for the first body,
    // by snapping free axes to the second frame's center (to account for
    // the allowed relative movement).
    {
        let mut new_center1 = frame2.translation; // First, assume all dofs are free.

        // Then snap the locked ones.
        for i in 0..DIM {
            if (locked_lin_axes & (1u32 << i)) != 0 {
                let axis = basis.col(i);
                new_center1 -= axis * lin_err.dot(axis);
            }
        }
        frame1.translation = new_center1;
    }

    let r1 = frame1.translation - world_com1;
    let r2 = frame2.translation - world_com2;

    let cmat1 = gcross_matrix(r1);
    let cmat2 = gcross_matrix(r2);

    let ang_err = frame1.rotation.inverse() * frame2.rotation;

    JointConstraintHelper {
        basis,
        // In 2D, cmat is a Vec2 representing [-r.y, r.x], and we need the dot product
        // with each column of basis to get the angular Jacobian components.
        cmat1_basis: [cmat1.dot(basis.col(0)), cmat1.dot(basis.col(1))],
        cmat2_basis: [cmat2.dot(basis.col(0)), cmat2.dot(basis.col(1))],
        lin_err,
        ang_err,
    }
}

#[cfg(feature = "dim3")]
/// Creates a new joint constraint helper (3D version).
pub fn new_helper(
    frame1_: Pose,
    frame2: Pose,
    world_com1: Vector,
    world_com2: Vector,
    locked_lin_axes: u32,
) -> JointConstraintHelper {
    let mut frame1 = frame1_;
    let basis = rotation_to_matrix(frame1.rotation);
    let lin_err = frame2.translation - frame1.translation;

    // Adjust the point of application of the force for the first body,
    // by snapping free axes to the second frame's center (to account for
    // the allowed relative movement).
    {
        let mut new_center1 = frame2.translation; // First, assume all dofs are free.

        // Then snap the locked ones.
        for i in 0..DIM {
            if (locked_lin_axes & (1u32 << i)) != 0 {
                let axis = basis.col(i);
                new_center1 -= axis * lin_err.dot(axis);
            }
        }
        frame1.translation = new_center1;
    }

    let r1 = frame1.translation - world_com1;
    let r2 = frame2.translation - world_com2;

    let cmat1 = gcross_matrix(r1);
    let cmat2 = gcross_matrix(r2);

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

    // TODO: this can probably be optimized a lot by unrolling the ops.
    let tensor_product = Mat3::from_cols(v1 * v2.x, v1 * v2.y, v1 * v2.z);
    let diag = Mat3::from_cols(
        Vec3::new(w1 * w2, 0.0, 0.0),
        Vec3::new(0.0, w1 * w2, 0.0),
        Vec3::new(0.0, 0.0, w1 * w2),
    );
    let cross_sum = gcross_matrix(v1 * w2 + v2 * w1);
    let cross_v1 = gcross_matrix(v1);
    let cross_v2 = gcross_matrix(v2);

    (tensor_product + diag - cross_sum + cross_v1 * cross_v2) * 0.5
}

/// Creates a linear lock constraint element.
pub fn lock_linear(
    helper: &JointConstraintHelper,
    joint_id: u32,
    body1: &JointSolverBody,
    body2: &JointSolverBody,
    locked_axis: usize,
    params: &SimParams,
) -> JointConstraintElement {
    #[cfg(feature = "dim2")]
    let lin_jac = helper.basis.col(locked_axis);
    #[cfg(feature = "dim3")]
    let lin_jac = helper.basis.col(locked_axis);

    #[cfg(feature = "dim2")]
    let ang_jac1 = helper.cmat1_basis.read(locked_axis);
    #[cfg(feature = "dim2")]
    let ang_jac2 = helper.cmat2_basis.read(locked_axis);

    #[cfg(feature = "dim3")]
    let ang_jac1 = helper.cmat1_basis.col(locked_axis);
    #[cfg(feature = "dim3")]
    let ang_jac2 = helper.cmat2_basis.col(locked_axis);

    let rhs_wo_bias = 0.0;
    let erp_inv_dt = joint_erp_inv_dt(params);
    let cfm_coeff = joint_cfm_coeff(params);
    let rhs_bias = lin_jac.dot(helper.lin_err) * erp_inv_dt;

    let ii_ang_jac1 = body1.ii_mul(ang_jac1);
    let ii_ang_jac2 = body2.ii_mul(ang_jac2);

    JointConstraintElement {
        joint_id,
        impulse: 0.0,
        impulse_bounds: Vec2::new(-MAX, MAX),
        lin_jac,
        ang_jac_a: ang_jac1,
        ang_jac_b: ang_jac2,
        ii_ang_jac_a: ii_ang_jac1,
        ii_ang_jac_b: ii_ang_jac2,
        inv_lhs: 0.0, // Will be set during orthogonalization.
        rhs: rhs_wo_bias + rhs_bias,
        rhs_wo_bias,
        cfm_gain: 0.0,
        cfm_coeff,
        #[cfg(feature = "dim2")]
        padding: 0,
    }
}

/// Creates an angular lock constraint element.
pub fn lock_angular(
    helper: &JointConstraintHelper,
    joint_id: u32,
    body1: &JointSolverBody,
    body2: &JointSolverBody,
    _locked_axis: usize,
    params: &SimParams,
) -> JointConstraintElement {
    #[cfg(feature = "dim2")]
    let ang_jac = 1.0;
    #[cfg(feature = "dim3")]
    let ang_jac = helper.ang_basis.col(_locked_axis);

    let rhs_wo_bias = 0.0;
    let erp_inv_dt = joint_erp_inv_dt(params);
    let cfm_coeff = joint_cfm_coeff(params);

    #[cfg(feature = "dim2")]
    let rhs_bias = helper.ang_err.sin() * erp_inv_dt;
    #[cfg(feature = "dim3")]
    let rhs_bias = helper.ang_err.read(_locked_axis) * erp_inv_dt;

    let ii_ang_jac1 = body1.ii_mul(ang_jac);
    let ii_ang_jac2 = body2.ii_mul(ang_jac);

    JointConstraintElement {
        joint_id,
        impulse: 0.0,
        impulse_bounds: Vec2::new(-MAX, MAX),
        lin_jac: Vector::ZERO,
        ang_jac_a: ang_jac,
        ang_jac_b: ang_jac,
        ii_ang_jac_a: ii_ang_jac1,
        ii_ang_jac_b: ii_ang_jac2,
        inv_lhs: 0.0, // Will be set during orthogonalization.
        rhs: rhs_wo_bias + rhs_bias,
        rhs_wo_bias,
        cfm_gain: 0.0,
        cfm_coeff,
        #[cfg(feature = "dim2")]
        padding: 0,
    }
}

/// Solves a joint constraint.
pub fn solve_joint_constraint(constraint: &mut JointConstraint, solver_vels: &mut SliceMut<Velocity>) {
    let mut solver_vel1 = solver_vels.read(constraint.solver_vel_a as usize);
    let mut solver_vel2 = solver_vels.read(constraint.solver_vel_b as usize);

    for i in 0..(constraint.len as usize) {
        let element = constraint.elements.at_mut(i);
        let dlinvel = element.lin_jac.dot(solver_vel2.linear - solver_vel1.linear);
        let dangvel = gdot(element.ang_jac_b, solver_vel2.angular)
            - gdot(element.ang_jac_a, solver_vel1.angular);

        let rhs = dlinvel + dangvel + element.rhs;
        let total_impulse = (element.impulse
            + element.inv_lhs * (rhs - element.cfm_gain * element.impulse))
            .clamp(element.impulse_bounds.x, element.impulse_bounds.y);
        let delta_impulse = total_impulse - element.impulse;
        element.impulse = total_impulse;

        let lin_impulse = element.lin_jac * delta_impulse;

        solver_vel1.linear += lin_impulse * constraint.im_a;
        solver_vel1.angular += element.ii_ang_jac_a * delta_impulse;
        solver_vel2.linear -= lin_impulse * constraint.im_b;
        solver_vel2.angular -= element.ii_ang_jac_b * delta_impulse;
    }

    solver_vels.write(constraint.solver_vel_a as usize, solver_vel1);
    solver_vels.write(constraint.solver_vel_b as usize, solver_vel2);
}

/// Creates a linear limit constraint element.
pub fn limit_linear(
    helper: &JointConstraintHelper,
    joint_id: u32,
    body1: &JointSolverBody,
    body2: &JointSolverBody,
    limited_axis: usize,
    limits: Vec2,
    params: &SimParams,
) -> JointConstraintElement {
    let mut constraint = lock_linear(helper, joint_id, body1, body2, limited_axis, params);

    let dist = helper.lin_err.dot(constraint.lin_jac);
    let min_enabled = dist <= limits.x;
    let max_enabled = limits.y <= dist;

    let erp_inv_dt = joint_erp_inv_dt(params);
    let cfm_coeff = joint_cfm_coeff(params);
    let rhs_bias = ((dist - limits.y).max(0.0) - (limits.x - dist).max(0.0)) * erp_inv_dt;
    constraint.rhs = constraint.rhs_wo_bias + rhs_bias;
    constraint.cfm_coeff = cfm_coeff;
    constraint.impulse_bounds = Vec2::new(
        if min_enabled { -MAX } else { 0.0 },
        if max_enabled { MAX } else { 0.0 },
    );

    constraint
}

/// Creates a coupled linear limit constraint element.
pub fn limit_linear_coupled(
    helper: &JointConstraintHelper,
    joint_id: u32,
    body1: &JointSolverBody,
    body2: &JointSolverBody,
    coupled_axes: u32,
    limits: Vec2,
    params: &SimParams,
) -> JointConstraintElement {
    let mut lin_jac = Vector::ZERO;
    let mut ang_jac1 = AngVector::default();
    let mut ang_jac2 = AngVector::default();

    #[cfg(feature = "dim2")]
    for i in 0..DIM {
        if (coupled_axes & (1u32 << i)) != 0 {
            let coeff = helper.basis.col(i).dot(helper.lin_err);
            lin_jac += helper.basis.col(i) * coeff;
            ang_jac1 += helper.cmat1_basis.read(i) * coeff;
            ang_jac2 += helper.cmat2_basis.read(i) * coeff;
        }
    }

    #[cfg(feature = "dim3")]
    for i in 0..DIM {
        if (coupled_axes & (1u32 << i)) != 0 {
            let coeff = helper.basis.col(i).dot(helper.lin_err);
            lin_jac += helper.basis.col(i) * coeff;
            ang_jac1 += helper.cmat1_basis.col(i) * coeff;
            ang_jac2 += helper.cmat2_basis.col(i) * coeff;
        }
    }

    // FIXME: handle min limit too.
    let dist = lin_jac.length();
    let inv_dist = pseudo_inv(dist);
    lin_jac *= inv_dist;
    ang_jac1 *= inv_dist;
    ang_jac2 *= inv_dist;

    let rhs_wo_bias = (dist - limits.y).min(0.0) * inv_dt(params);

    let ii_ang_jac1 = body1.ii_mul(ang_jac1);
    let ii_ang_jac2 = body2.ii_mul(ang_jac2);

    let erp_inv_dt = joint_erp_inv_dt(params);
    let cfm_coeff = joint_cfm_coeff(params);
    let rhs_bias = (dist - limits.y).max(0.0) * erp_inv_dt;
    let rhs = rhs_wo_bias + rhs_bias;
    let impulse_bounds = Vec2::new(0.0, MAX);

    JointConstraintElement {
        joint_id,
        impulse: 0.0,
        impulse_bounds,
        lin_jac,
        ang_jac_a: ang_jac1,
        ang_jac_b: ang_jac2,
        ii_ang_jac_a: ii_ang_jac1,
        ii_ang_jac_b: ii_ang_jac2,
        inv_lhs: 0.0, // Will be set during orthogonalization.
        rhs,
        rhs_wo_bias,
        cfm_gain: 0.0,
        cfm_coeff,
        #[cfg(feature = "dim2")]
        padding: 0,
    }
}

/// Creates a linear motor constraint element.
pub fn motor_linear(
    helper: &JointConstraintHelper,
    joint_id: u32,
    body1: &JointSolverBody,
    body2: &JointSolverBody,
    motor_axis: usize,
    motor_params: &MotorParameters,
    limits: Vec2,
    params: &SimParams,
) -> JointConstraintElement {
    let dt_inv = inv_dt(params);
    let mut constraint = lock_linear(helper, joint_id, body1, body2, motor_axis, params);

    let mut rhs_wo_bias = 0.0;
    if motor_params.erp_inv_dt != 0.0 {
        let dist = helper.lin_err.dot(constraint.lin_jac);
        rhs_wo_bias += (dist - motor_params.target_pos) * motor_params.erp_inv_dt;
    }

    let mut target_vel = motor_params.target_vel;
    if limits != Vec2::new(-MAX, MAX) {
        let dist = helper.lin_err.dot(constraint.lin_jac);
        target_vel = target_vel.clamp((limits.x - dist) * dt_inv, (limits.y - dist) * dt_inv);
    }

    rhs_wo_bias += -target_vel;

    constraint.cfm_coeff = motor_params.cfm_coeff;
    constraint.cfm_gain = motor_params.cfm_gain;
    constraint.impulse_bounds = Vec2::new(-motor_params.max_impulse, motor_params.max_impulse);
    constraint.rhs = rhs_wo_bias;
    constraint.rhs_wo_bias = rhs_wo_bias;
    constraint
}

/// Creates a coupled linear motor constraint element.
pub fn motor_linear_coupled(
    helper: &JointConstraintHelper,
    joint_id: u32,
    body1: &JointSolverBody,
    body2: &JointSolverBody,
    coupled_axes: u32,
    motor_params: &MotorParameters,
    limits: Vec2,
    params: &SimParams,
) -> JointConstraintElement {
    let dt_inv = inv_dt(params);

    let mut lin_jac = Vector::ZERO;
    let mut ang_jac1 = AngVector::default();
    let mut ang_jac2 = AngVector::default();

    #[cfg(feature = "dim2")]
    for i in 0..DIM {
        if (coupled_axes & (1u32 << i)) != 0 {
            let coeff = helper.basis.col(i).dot(helper.lin_err);
            lin_jac += helper.basis.col(i) * coeff;
            ang_jac1 += helper.cmat1_basis.read(i) * coeff;
            ang_jac2 += helper.cmat2_basis.read(i) * coeff;
        }
    }

    #[cfg(feature = "dim3")]
    for i in 0..DIM {
        if (coupled_axes & (1u32 << i)) != 0 {
            let coeff = helper.basis.col(i).dot(helper.lin_err);
            lin_jac += helper.basis.col(i) * coeff;
            ang_jac1 += helper.cmat1_basis.col(i) * coeff;
            ang_jac2 += helper.cmat2_basis.col(i) * coeff;
        }
    }

    let dist = lin_jac.length();
    let inv_dist = pseudo_inv(dist);
    lin_jac *= inv_dist;
    ang_jac1 *= inv_dist;
    ang_jac2 *= inv_dist;

    let mut rhs_wo_bias = 0.0;
    if motor_params.erp_inv_dt != 0.0 {
        rhs_wo_bias += (dist - motor_params.target_pos) * motor_params.erp_inv_dt;
    }

    let mut target_vel = motor_params.target_vel;
    if limits != Vec2::new(-MAX, MAX) {
        target_vel = target_vel.clamp((limits.x - dist) * dt_inv, (limits.y - dist) * dt_inv);
    }

    rhs_wo_bias += -target_vel;

    let ii_ang_jac1 = body1.ii_mul(ang_jac1);
    let ii_ang_jac2 = body2.ii_mul(ang_jac2);

    JointConstraintElement {
        joint_id,
        impulse: 0.0,
        impulse_bounds: Vec2::new(-motor_params.max_impulse, motor_params.max_impulse),
        lin_jac,
        ang_jac_a: ang_jac1,
        ang_jac_b: ang_jac2,
        ii_ang_jac_a: ii_ang_jac1,
        ii_ang_jac_b: ii_ang_jac2,
        inv_lhs: 0.0, // Will be set during orthogonalization.
        rhs: rhs_wo_bias,
        rhs_wo_bias,
        cfm_gain: motor_params.cfm_gain,
        cfm_coeff: motor_params.cfm_coeff,
        #[cfg(feature = "dim2")]
        padding: 0,
    }
}

/// Creates an angular limit constraint element.
pub fn limit_angular(
    helper: &JointConstraintHelper,
    joint_id: u32,
    body1: &JointSolverBody,
    body2: &JointSolverBody,
    _limited_axis: usize,
    limits: Vec2,
    params: &SimParams,
) -> JointConstraintElement {
    let s_limits = Vec2::new(crate::sin(limits.x * 0.5), crate::sin(limits.y * 0.5));

    #[cfg(feature = "dim2")]
    let s_ang = crate::sin(rotation_angle(helper.ang_err) * 0.5);
    #[cfg(feature = "dim3")]
    let s_ang = helper.ang_err.read(_limited_axis);

    let min_enabled = s_ang <= s_limits.x;
    let max_enabled = s_limits.y <= s_ang;

    let impulse_bounds = Vec2::new(
        if min_enabled { -MAX } else { 0.0 },
        if max_enabled { MAX } else { 0.0 },
    );

    #[cfg(feature = "dim2")]
    let ang_jac = 1.0;
    #[cfg(feature = "dim3")]
    let ang_jac = helper.ang_basis.col(_limited_axis);

    let rhs_wo_bias = 0.0;
    let erp_inv_dt = joint_erp_inv_dt(params);
    let cfm_coeff = joint_cfm_coeff(params);
    let rhs_bias = ((s_ang - s_limits.y).max(0.0) - (s_limits.x - s_ang).max(0.0)) * erp_inv_dt;

    let ii_ang_jac1 = body1.ii_mul(ang_jac);
    let ii_ang_jac2 = body2.ii_mul(ang_jac);

    JointConstraintElement {
        joint_id,
        impulse: 0.0,
        impulse_bounds,
        lin_jac: Vector::ZERO,
        ang_jac_a: ang_jac,
        ang_jac_b: ang_jac,
        ii_ang_jac_a: ii_ang_jac1,
        ii_ang_jac_b: ii_ang_jac2,
        inv_lhs: 0.0, // Will be set during orthogonalization.
        rhs: rhs_wo_bias + rhs_bias,
        rhs_wo_bias,
        cfm_gain: 0.0,
        cfm_coeff,
        #[cfg(feature = "dim2")]
        padding: 0,
    }
}

/// Creates an angular motor constraint element.
pub fn motor_angular(
    helper: &JointConstraintHelper,
    joint_id: u32,
    body1: &JointSolverBody,
    body2: &JointSolverBody,
    _motor_axis: usize,
    motor_params: &MotorParameters,
) -> JointConstraintElement {
    #[cfg(feature = "dim2")]
    let ang_jac = 1.0;
    #[cfg(feature = "dim3")]
    let ang_jac = helper.basis.col(_motor_axis);

    let mut rhs_wo_bias = 0.0;
    if motor_params.erp_inv_dt != 0.0 {
        #[cfg(feature = "dim2")]
        let ang_dist = rotation_angle(helper.ang_err);
        #[cfg(feature = "dim3")]
        let ang_dist = {
            // Clamp the component from -1.0 to 1.0 to account for slight imprecision
            let clamped_err = helper.ang_err.read(_motor_axis).clamp(-1.0, 1.0);
            crate::asin(clamped_err) * 2.0
        };

        let target_ang = motor_params.target_pos;
        rhs_wo_bias +=
            smallest_abs_diff_between_angles(ang_dist, target_ang) * motor_params.erp_inv_dt;
    }

    rhs_wo_bias += -motor_params.target_vel;

    let ii_ang_jac1 = body1.ii_mul(ang_jac);
    let ii_ang_jac2 = body2.ii_mul(ang_jac);

    JointConstraintElement {
        joint_id,
        impulse: 0.0,
        impulse_bounds: Vec2::new(-motor_params.max_impulse, motor_params.max_impulse),
        lin_jac: Vector::ZERO,
        ang_jac_a: ang_jac,
        ang_jac_b: ang_jac,
        ii_ang_jac_a: ii_ang_jac1,
        ii_ang_jac_b: ii_ang_jac2,
        inv_lhs: 0.0, // Will be set during orthogonalization.
        rhs: rhs_wo_bias,
        rhs_wo_bias,
        cfm_gain: motor_params.cfm_gain,
        cfm_coeff: motor_params.cfm_coeff,
        #[cfg(feature = "dim2")]
        padding: 0,
    }
}

/// Updates a joint constraint for a new substep.
pub fn update_constraint(
    builder: &JointConstraintBuilder,
    constraint: &mut JointConstraint,
    poses: &Slice<Pose>,
    mprops: &Slice<WorldMassProperties>,
    params: &SimParams,
) {
    // NOTE: right now, the "update", is basically reconstructing all the
    //       constraints entirely. Could we make this more incremental?
    let joint = &builder.joint;
    let body1 = builder.body1;
    let body2 = builder.body2;
    let pose1 = poses.read(body1 as usize);
    let pose2 = poses.read(body2 as usize);
    let mprops1 = mprops.at(body1 as usize);
    let mprops2 = mprops.at(body2 as usize);

    let frame1 = pose1 * joint.local_frame_a;
    let frame2 = pose2 * joint.local_frame_b;

    // TODO: needs adjustment if the pose origin isn't the same as the
    //       center of mass.
    let world_com1 = pose1.translation;
    let world_com2 = pose2.translation;

    let joint_body1 = JointSolverBody {
        im: mprops1.inv_mass,
        ii: mprops1.inv_inertia,
        world_com: world_com1,
        solver_vel: body1,
    };
    let joint_body2 = JointSolverBody {
        im: mprops2.inv_mass,
        ii: mprops2.inv_inertia,
        world_com: world_com2,
        solver_vel: body2,
    };

    let mut len = 0usize;
    let locked_axes = joint.locked_axes;
    let motor_axes = joint.motor_axes & !locked_axes;
    let limit_axes = joint.limit_axes & !locked_axes;
    let coupled_axes = joint.coupled_axes;

    // The has_lin/ang_coupling test is needed to avoid shl overflow later.
    let has_lin_coupling = (coupled_axes & LIN_AXES_MASK) != 0;
    let first_coupled_lin_axis_id = (coupled_axes & LIN_AXES_MASK).trailing_zeros() as usize;

    #[cfg(feature = "dim3")]
    let _has_ang_coupling = (coupled_axes & ANG_AXES_MASK) != 0;
    #[cfg(feature = "dim3")]
    let _first_coupled_ang_axis_id = (coupled_axes & ANG_AXES_MASK).trailing_zeros() as usize;

    let helper = new_helper(frame1, frame2, mprops1.com, mprops2.com, locked_axes);

    let mut start = len;

    // Angular motors (uncoupled)
    for i in DIM..SPATIAL_DIM {
        if ((motor_axes & !coupled_axes) & (1u32 << i)) != 0 {
            let mp = motor_params(joint.motors.at(i), params.dt);
            constraint.elements.write(
                len,
                motor_angular(
                    &helper,
                    builder.constraint_id,
                    &joint_body1,
                    &joint_body2,
                    i - DIM,
                    &mp,
                ),
            );
            len += 1;
        }
    }

    // Linear motors (uncoupled)
    for i in 0..DIM {
        if ((motor_axes & !coupled_axes) & (1u32 << i)) != 0 {
            let mut limits = Vec2::new(-MAX, MAX);

            if (limit_axes & (1u32 << i)) != 0 {
                limits = Vec2::new(joint.limits.at(i).min, joint.limits.at(i).max);
            }

            let mp = motor_params(joint.motors.at(i), params.dt);

            constraint.elements.write(
                len,
                motor_linear(
                    &helper,
                    builder.constraint_id,
                    &joint_body1,
                    &joint_body2,
                    i,
                    &mp,
                    limits,
                    params,
                ),
            );
            len += 1;
        }
    }

    // Coupled angular motors
    if ((motor_axes & coupled_axes) & ANG_AXES_MASK) != 0 {
        // TODO: coupled angular motor constraint.
    }

    // Coupled linear motors
    if ((motor_axes & coupled_axes) & LIN_AXES_MASK) != 0 {
        let mut limits = Vec2::new(-MAX, MAX);
        if (limit_axes & (1u32 << first_coupled_lin_axis_id)) != 0 {
            limits = Vec2::new(
                joint.limits.at(first_coupled_lin_axis_id).min,
                joint.limits.at(first_coupled_lin_axis_id).max,
            );
        }

        let mp = motor_params(joint.motors.at(first_coupled_lin_axis_id), params.dt);

        constraint.elements.write(
            len,
            motor_linear_coupled(
                &helper,
                builder.constraint_id,
                &joint_body1,
                &joint_body2,
                coupled_axes,
                &mp,
                limits,
                params,
            ),
        );
        len += 1;
    }

    orthogonalize_constraints(constraint, start, len);

    start = len;

    // Angular locks
    for i in DIM..SPATIAL_DIM {
        if (locked_axes & (1u32 << i)) != 0 {
            constraint.elements.write(
                len,
                lock_angular(
                    &helper,
                    builder.constraint_id,
                    &joint_body1,
                    &joint_body2,
                    i - DIM,
                    params,
                ),
            );
            len += 1;
        }
    }

    // Linear locks
    for i in 0..DIM {
        if (locked_axes & (1u32 << i)) != 0 {
            constraint.elements.write(
                len,
                lock_linear(
                    &helper,
                    builder.constraint_id,
                    &joint_body1,
                    &joint_body2,
                    i,
                    params,
                ),
            );
            len += 1;
        }
    }

    // Angular limits (uncoupled)
    for i in DIM..SPATIAL_DIM {
        if ((limit_axes & !coupled_axes) & (1u32 << i)) != 0 {
            constraint.elements.write(
                len,
                limit_angular(
                    &helper,
                    builder.constraint_id,
                    &joint_body1,
                    &joint_body2,
                    i - DIM,
                    Vec2::new(joint.limits.at(i).min, joint.limits.at(i).max),
                    params,
                ),
            );
            len += 1;
        }
    }

    // Linear limits (uncoupled)
    for i in 0..DIM {
        if ((limit_axes & !coupled_axes) & (1u32 << i)) != 0 {
            constraint.elements.write(
                len,
                limit_linear(
                    &helper,
                    builder.constraint_id,
                    &joint_body1,
                    &joint_body2,
                    i,
                    Vec2::new(joint.limits.at(i).min, joint.limits.at(i).max),
                    params,
                ),
            );
            len += 1;
        }
    }

    // Coupled linear limits
    if has_lin_coupling && (limit_axes & (1u32 << first_coupled_lin_axis_id)) != 0 {
        constraint.elements.write(
            len,
            limit_linear_coupled(
                &helper,
                builder.constraint_id,
                &joint_body1,
                &joint_body2,
                coupled_axes,
                Vec2::new(
                    joint.limits.at(first_coupled_lin_axis_id).min,
                    joint.limits.at(first_coupled_lin_axis_id).max,
                ),
                params,
            ),
        );
        len += 1;
    }

    orthogonalize_constraints(constraint, start, len);
    constraint.len = len as u32;
}

/// Orthogonalizes constraints using modified Gram-Schmidt and sets their inv_lhs field.
pub fn orthogonalize_constraints(constraint: &mut JointConstraint, start: usize, end: usize) {
    let len = end - start;

    if len == 0 {
        return;
    }

    let imsum = constraint.im_a + constraint.im_b;

    // Use the modified Gram-Schmidt orthogonalization.
    for j in start..end {
        let dot_jj = constraint
            .elements
            .at(j)
            .lin_jac
            .dot(imsum * constraint.elements.at(j).lin_jac)
            + gdot(
                constraint.elements.at(j).ii_ang_jac_a,
                constraint.elements.at(j).ang_jac_a,
            )
            + gdot(
                constraint.elements.at(j).ii_ang_jac_b,
                constraint.elements.at(j).ang_jac_b,
            );
        let cfm_gain =
            dot_jj * constraint.elements.at(j).cfm_coeff + constraint.elements.at(j).cfm_gain;
        let inv_dot_jj = pseudo_inv(dot_jj);
        constraint.elements.at_mut(j).inv_lhs = pseudo_inv(dot_jj + cfm_gain);
        constraint.elements.at_mut(j).cfm_gain = cfm_gain;

        if constraint.elements.at(j).impulse_bounds != Vec2::new(-MAX, MAX) {
            // Don't remove constraints with limited forces from the others
            // because they may not deliver the necessary forces to fulfill
            // the removed parts of other constraints.
            continue;
        }

        for i in (j + 1)..end {
            // Read element j values first to avoid borrow conflicts
            let elem_j_lin_jac = constraint.elements.at(j).lin_jac;
            let elem_j_ang_jac_a = constraint.elements.at(j).ang_jac_a;
            let elem_j_ang_jac_b = constraint.elements.at(j).ang_jac_b;
            let elem_j_ii_ang_jac_a = constraint.elements.at(j).ii_ang_jac_a;
            let elem_j_ii_ang_jac_b = constraint.elements.at(j).ii_ang_jac_b;
            let elem_j_rhs_wo_bias = constraint.elements.at(j).rhs_wo_bias;
            let elem_j_rhs = constraint.elements.at(j).rhs;

            let dot_ij = constraint
                .elements
                .at(i)
                .lin_jac
                .dot(imsum * elem_j_lin_jac)
                + gdot(constraint.elements.at(i).ii_ang_jac_a, elem_j_ang_jac_a)
                + gdot(constraint.elements.at(i).ii_ang_jac_b, elem_j_ang_jac_b);
            let coeff = dot_ij * inv_dot_jj;

            constraint.elements.at_mut(i).lin_jac -= elem_j_lin_jac * coeff;
            constraint.elements.at_mut(i).ang_jac_a -= elem_j_ang_jac_a * coeff;
            constraint.elements.at_mut(i).ang_jac_b -= elem_j_ang_jac_b * coeff;
            constraint.elements.at_mut(i).ii_ang_jac_a -= elem_j_ii_ang_jac_a * coeff;
            constraint.elements.at_mut(i).ii_ang_jac_b -= elem_j_ii_ang_jac_b * coeff;
            constraint.elements.at_mut(i).rhs_wo_bias -= elem_j_rhs_wo_bias * coeff;
            constraint.elements.at_mut(i).rhs -= elem_j_rhs * coeff;
        }
    }
}
