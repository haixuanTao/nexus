//! Rigid body dynamics module.
//!
//! This module provides:
//! - Body state and mass properties
//! - Contact constraints
//! - Joint constraints
//! - Constraint solver (PGS/Sequential Impulse)
//! - Graph coloring for parallel solving

// Data structures and algorithms
mod body;
mod constraint;
mod joint;
mod joint_constraint;
mod joint_constraint_builder;
mod sim_params;
mod solver_utils;
mod warmstart;

// GPU compute shader kernels
mod coloring;
mod mprops_update;
mod prefix_sum;
mod solver;

pub use body::*;
pub use constraint::*;
// Re-export joint items explicitly; MotorParameters is also in joint_constraint
pub use joint::{
    motor_params, GenericJoint, ImpulseJoint, JointLimits, JointMotor, MotorParameters,
    ACCELERATION_BASED, ANG_AXES_MASK, FORCE_BASED, LIN_AXES_MASK, SPATIAL_DIM,
};
// Re-export joint_constraint items; MotorParameters comes from joint
pub use joint_constraint::*;
// Re-export joint_constraint_builder items; update_constraint is also in solver
pub use joint_constraint_builder::{
    limit_angular, limit_linear, limit_linear_coupled, lock_angular, lock_linear, motor_angular,
    motor_linear, motor_linear_coupled, new_helper, orthogonalize_constraints,
    solve_joint_constraint, JointConstraintBuilder, JointConstraintHelper,
};
pub use sim_params::*;
// Re-export solver items; update_constraint comes from joint_constraint_builder for joints
pub use coloring::*;
pub use mprops_update::*;
pub use prefix_sum::*;
pub use solver::*;
pub use solver_utils::{
    contact_to_constraint, integrate_pose, remove_cfm_and_bias, solve_constraint_gauss_seidel,
    warmstart_body, warmstart_constraint,
};
pub use warmstart::*;
