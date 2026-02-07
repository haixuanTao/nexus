//! Joint constraint data structures
//!
//! This module defines the constraint structures used by the joint solver.

use crate::{AngVector, Vector};

use super::joint::SPATIAL_DIM;

#[cfg(feature = "dim2")]
use glamx::Vec2;
#[cfg(feature = "dim3")]
use glamx::{Mat4, Vec2};

use khal_derive::spirv_bindgen;
use spirv_std::glam::UVec3;
use spirv_std::spirv;

use vortx_shaders::utils::step::StepRng;

use crate::MaybeIndexUnchecked;
use crate::Pose;

use super::body::{LocalMassProperties, Velocity, WorldMassProperties};
use super::joint::ImpulseJoint;
use super::joint_constraint_builder::{
    solve_joint_constraint, update_constraint, JointConstraintBuilder,
};
use super::sim_params::SimParams;

const WORKGROUP_SIZE: u32 = 64;

/// Motor parameters for constraint solving.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct MotorParameters {
    pub erp_inv_dt: f32,
    pub cfm_coeff: f32,
    pub cfm_gain: f32,
    pub target_pos: f32,
    pub target_vel: f32,
    pub max_impulse: f32,
}

/// Solver body data for joint constraints.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct JointSolverBody {
    /// Inverse mass along each axis.
    pub im: Vector,

    #[cfg(feature = "dim2")]
    /// Inverse inertia (scalar in 2D).
    pub ii: f32,

    #[cfg(feature = "dim3")]
    /// Inverse inertia tensor (3x3 matrix in 3D).
    pub ii: Mat4,

    // TODO: is this still needed now that the solver body poses are expressed at the center of mass?
    /// World-space center of mass.
    pub world_com: Vector,

    /// Index in solver velocity array.
    pub solver_vel: u32,
}

impl JointSolverBody {
    pub fn ii_mul(&self, v: AngVector) -> AngVector {
        #[cfg(feature = "dim2")]
        return self.ii * v;
        #[cfg(feature = "dim3")]
        return (self.ii * v.extend(0.0)).truncate();
    }
}

/// A joint constraint between two rigid bodies.
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch = "spirv"), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
#[cfg(feature = "dim2")] // Same as 3D but with different field ordering to avoid padding.
pub struct JointConstraint {
    /// Index of body A in solver velocity array.
    pub solver_vel_a: u32,
    /// Index of body B in solver velocity array.
    pub solver_vel_b: u32,
    /// Inverse mass of body A.
    pub im_a: Vector,
    /// Inverse mass of body B.
    pub im_b: Vector,

    /// The constraints for a joint. Up to 6 in 3D, and up to 3 in 2D.
    pub elements: [JointConstraintElement; SPATIAL_DIM],
    /// The number of active `JointConstraint::elements`.
    pub len: u32,
    pub padding: [u32; 1],
}

/// A joint constraint between two rigid bodies.
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch = "spirv"), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
#[cfg(feature = "dim3")] // Same as 2D but with different field ordering to avoid padding.
pub struct JointConstraint {
    /// Inverse mass of body A.
    pub im_a: Vector,
    /// Index of body A in solver velocity array.
    pub solver_vel_a: u32,
    /// Inverse mass of body B.
    pub im_b: Vector,
    /// Index of body B in solver velocity array.
    pub solver_vel_b: u32,

    /// The constraints for a joint. Up to 6 in 3D, and up to 3 in 2D.
    pub elements: [JointConstraintElement; SPATIAL_DIM],
    /// The number of active `JointConstraint::elements`.
    pub len: u32,
    pub padding: [u32; 3],
}

impl Default for JointConstraint {
    fn default() -> Self {
        Self {
            solver_vel_a: 0,
            solver_vel_b: 0,
            im_a: Vector::ZERO,
            im_b: Vector::ZERO,
            elements: [JointConstraintElement::default(); SPATIAL_DIM],
            len: 0,
            padding: [0; _],
        }
    }
}

/// A single element (DOF) of a joint constraint.
// NOTE: field order has been selected meticulously to reduce padding in both 2D and 3D versions.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch = "spirv"), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct JointConstraintElement {
    /// Linear Jacobian direction.
    pub lin_jac: Vector,
    /// Joint ID for writeback.
    pub joint_id: u32,
    /// Angular Jacobian for body A.
    pub ang_jac_a: AngVector,
    /// Accumulated impulse.
    pub impulse: f32,
    /// Angular Jacobian for body B.
    pub ang_jac_b: AngVector,
    /// Inverse effective mass (1 / m_eff).
    pub inv_lhs: f32,
    /// Angular Jacobian for body A multiplied by inverse inertia.
    pub ii_ang_jac_a: AngVector,
    /// Right-hand side (target velocity).
    pub rhs: f32,
    /// Angular Jacobian for body B multiplied by inverse inertia.
    pub ii_ang_jac_b: AngVector,
    /// Right-hand side without bias.
    pub rhs_wo_bias: f32,
    /// CFM gain for soft constraints.
    pub cfm_gain: f32,
    /// CFM coefficient for soft constraints.
    pub cfm_coeff: f32,
    #[cfg(feature = "dim2")]
    pub padding: u32,
    /// Impulse bounds (min, max).
    pub impulse_bounds: Vec2,
}

/// Resets the joint color to 0.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_reset_joint_color(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] curr_color: &mut u32,
) {
    // NOTE: this `for` loop is silly. It doesn’t do anything
    //       more than a `*curr_color = 0` in a convoluted
    //       way because otherwise rustgpu apparently does not generate
    //       the spirv for this kernel (seems to happen if the kernel is
    //       too trivial.
    for k in 0..1 {
        // NOTE: for joints, our first colors start at 0.
        *curr_color = k;
    }
}

/// Increments the joint color.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_inc_joint_color(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] curr_color: &mut u32,
) {
    // NOTE: this `for` loop is silly. It doesn’t do anything
    //       more than a `*curr_color += 1` in a convoluted
    //       way because otherwise rustgpu apparently does not generate
    //       the spirv for this kernel (seems to happen if the kernel is
    //       too trivial.
    for k in 0..1 {
        *curr_color += 1 + k;
    }
}

/// Initializes joint constraint builders and constraints.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_init_joint_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] joints: &[ImpulseJoint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    builders: &mut [JointConstraintBuilder],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] constraints: &mut [JointConstraint],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)]
    local_mprops: &[LocalMassProperties],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] joints_len: &u32,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    let len = *joints_len;

    for i in StepRng::new(invocation_id.x..len, num_threads) {
        let idx = i as usize;
        let joint = joints.at(idx);

        builders.write(
            idx,
            JointConstraintBuilder {
                body1: joint.body_a,
                body2: joint.body_b,
                joint_id: i,
                joint: joint.data,
                constraint_id: i,
            },
        );

        constraints.at_mut(idx).solver_vel_a = joint.body_a;
        constraints.at_mut(idx).solver_vel_b = joint.body_b;
        constraints.at_mut(idx).im_a = local_mprops.at(joint.body_a as usize).inv_mass;
        constraints.at_mut(idx).im_b = local_mprops.at(joint.body_b as usize).inv_mass;
        constraints.at_mut(idx).len = 0; // Constraint elements will be filled later.
    }
}

/// Updates joint constraints for a new substep.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_update_joint_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] builders: &[JointConstraintBuilder],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] constraints: &mut [JointConstraint],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] mprops: &[WorldMassProperties],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] joints_len: &u32,
    #[spirv(uniform, descriptor_set = 1, binding = 2)] params: &SimParams,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    let len = *joints_len;

    for i in StepRng::new(invocation_id.x..len, num_threads) {
        let idx = i as usize;
        update_constraint(
            builders.at(idx),
            constraints.at_mut(idx),
            poses,
            mprops,
            params,
        );
    }
}

/// Removes bias from joint constraints for the final substep.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_remove_joint_bias(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] constraints: &mut [JointConstraint],
    #[spirv(uniform, descriptor_set = 0, binding = 1)] joints_len: &u32,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    let len = *joints_len;

    for i in StepRng::new(invocation_id.x..len, num_threads) {
        let idx = i as usize;
        for j in 0..(constraints.at(idx).len as usize) {
            constraints.at_mut(idx).elements.at_mut(j).rhs =
                constraints.at(idx).elements.at(j).rhs_wo_bias;
        }
    }
}

/// Solves joint constraints.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_solve_joint_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] constraints: &mut [JointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] solver_vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] color_groups: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] curr_color: &u32,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    let color = *curr_color as usize;

    let start = if color > 0 {
        color_groups.read(color - 1)
    } else {
        0
    };
    let end = color_groups.read(color);

    for i in StepRng::new(start + invocation_id.x..end, num_threads) {
        solve_joint_constraint(constraints.at_mut(i as usize), solver_vels);
    }
}
