//! Impulse-based joint constraints for articulated bodies.
//!
//! This module provides GPU-accelerated joint constraint solving, allowing bodies
//! to be connected with various joint types (revolute, prismatic, fixed, etc.).

use crate::math::Pose;
use crate::shaders::dynamics::{
    GpuIncJointColor, GpuInitJointConstraints, GpuRemoveJointBias, GpuResetJointColor,
    GpuSolveJointConstraints, GpuUpdateJointConstraints, ImpulseJoint, JointConstraint,
    JointConstraintBuilder, LocalMassProperties, SimParams, Velocity, WorldMassProperties,
};
use khal::Shader;
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};
use vortx::tensor::Tensor;

#[cfg(feature = "from_rapier")]
use {
    crate::rapier::dynamics::{
        GenericJoint as RapierGenericJoint, ImpulseJoint as RapierImpulseJoint, ImpulseJointSet,
        JointLimits as RapierJointLimits, JointMotor as RapierJointMotor, RigidBodyHandle,
    },
    crate::rapier::prelude::MotorModel,
    crate::shaders::dynamics::{GenericJoint, JointLimits, JointMotor},
    khal::BufferUsages,
    std::collections::HashMap,
};

#[cfg(feature = "from_rapier")]
fn convert_joint_limits(limits: RapierJointLimits<f32>) -> JointLimits {
    JointLimits {
        min: limits.min,
        max: limits.max,
        impulse: limits.impulse,
    }
}

#[cfg(feature = "from_rapier")]
fn convert_joint_motor(motor: RapierJointMotor) -> JointMotor {
    JointMotor {
        target_vel: motor.target_vel,
        target_pos: motor.target_pos,
        stiffness: motor.stiffness,
        damping: motor.damping,
        max_force: motor.max_force,
        impulse: motor.impulse,
        model: match motor.model {
            MotorModel::AccelerationBased => 0,
            MotorModel::ForceBased => 1,
        },
    }
}

#[cfg(feature = "from_rapier")]
fn convert_generic_joint(joint: RapierGenericJoint) -> GenericJoint {
    GenericJoint {
        local_frame_a: joint.local_frame1,
        local_frame_b: joint.local_frame2,
        locked_axes: joint.locked_axes.bits() as u32,
        limit_axes: joint.limit_axes.bits() as u32,
        motor_axes: joint.motor_axes.bits() as u32,
        coupled_axes: joint.coupled_axes.bits() as u32,
        limits: joint.limits.map(convert_joint_limits),
        motors: joint.motors.map(convert_joint_motor),
    }
}

#[cfg(feature = "from_rapier")]
fn convert_impulse_joint(
    joint: &RapierImpulseJoint,
    body_ids: &HashMap<RigidBodyHandle, u32>,
) -> ImpulseJoint {
    ImpulseJoint {
        body_a: body_ids[&joint.body1],
        body_b: body_ids[&joint.body2],
        #[cfg(feature = "dim3")]
        padding: [0, 0],
        data: convert_generic_joint(joint.data),
    }
}

/// A set of impulse joints simulated on the GPU.
pub struct GpuImpulseJointSet {
    len: u32,
    num_colors: u32,
    max_color_group_len: u32,
    num_joints: Tensor<u32>,
    curr_color: Tensor<u32>,
    color_groups: Tensor<u32>,
    joints: Tensor<ImpulseJoint>,
    builders: Tensor<JointConstraintBuilder>,
    constraints: Tensor<JointConstraint>,
}

impl GpuImpulseJointSet {
    /// Converts a set of Rapier joints to a set of GPU joints.
    #[cfg(feature = "from_rapier")]
    pub fn from_rapier(
        backend: &GpuBackend,
        joints: &ImpulseJointSet,
        body_ids: &HashMap<RigidBodyHandle, u32>,
    ) -> Self {
        let usage = BufferUsages::STORAGE;
        let len = joints.len() as u32;
        let max_body_id = body_ids.values().copied().max().unwrap_or_default();

        // Convert joints.
        let mut unsorted_gpu_joints = vec![];
        for (_, joint) in joints.iter() {
            unsorted_gpu_joints.push(convert_impulse_joint(joint, body_ids));
        }

        /*
         * Run a simple static greedy graph coloring, and group the joints.
         */
        let mut colors = vec![];
        let mut body_masks = vec![0u128; max_body_id as usize + 1];

        // Find colors.
        for joint in &unsorted_gpu_joints {
            let a = joint.body_a as usize;
            let b = joint.body_b as usize;
            let mask = body_masks[a] | body_masks[b];
            let color = mask.trailing_ones();
            colors.push(color);
            body_masks[a] |= 1 << color;
            body_masks[b] |= 1 << color;
        }

        let num_colors = colors
            .iter()
            .copied()
            .max()
            .map(|n| n + 1)
            .unwrap_or_default();
        let mut color_groups = vec![0u32; num_colors as usize];

        // Count size of color groups.
        for color in &colors {
            color_groups[*color as usize] += 1;
        }

        let max_color_group_len = color_groups.iter().copied().max().unwrap_or_default();

        // Prefix sum.
        for i in 0..color_groups.len().saturating_sub(1) {
            color_groups[i + 1] += color_groups[i];
        }

        // Bucket sort.
        let mut target = color_groups.clone();
        target.insert(0, 0);
        let mut sorted_gpu_joints = unsorted_gpu_joints.clone();

        for (joint, color) in unsorted_gpu_joints.iter().zip(colors.iter()) {
            sorted_gpu_joints[target[*color as usize] as usize] = *joint;
            target[*color as usize] += 1;
        }

        Self {
            len,
            num_colors,
            max_color_group_len,
            num_joints: Tensor::scalar_encased(backend, len, usage | BufferUsages::UNIFORM)
                .unwrap(),
            curr_color: Tensor::scalar_encased(backend, 0u32, usage | BufferUsages::UNIFORM)
                .unwrap(),
            color_groups: Tensor::vector(backend, &color_groups, usage).unwrap(),
            joints: Tensor::vector(backend, &sorted_gpu_joints, usage).unwrap(),
            builders: Tensor::vector_uninit(backend, len, usage).unwrap(),
            constraints: Tensor::vector_uninit(backend, len, usage).unwrap(),
        }
    }

    /// Is this set empty?
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The number of joints in this set.
    pub fn len(&self) -> usize {
        self.len as usize
    }
}

/// GPU shader for joint constraint solving.
#[derive(Shader)]
pub struct GpuJointSolver {
    /// Initializes joint constraints.
    init_joint_constraints: GpuInitJointConstraints,
    /// Updates joint constraints each substep.
    update_joint_constraints: GpuUpdateJointConstraints,
    reset_joint_color: GpuResetJointColor,
    inc_joint_color: GpuIncJointColor,
    /// Solves joint constraints.
    solve_joint_constraints: GpuSolveJointConstraints,
    /// Removes bias from joint constraints.
    remove_joint_bias: GpuRemoveJointBias,
}

/// Arguments given to the joint solver.
pub struct JointSolverArgs<'a> {
    /// The simulation parameters.
    pub sim_params: &'a Tensor<SimParams>,
    /// The set of joints to solve.
    pub joints: &'a mut GpuImpulseJointSet,
    /// World-space mass properties.
    pub mprops: &'a Tensor<WorldMassProperties>,
    /// Local-space mass properties.
    pub local_mprops: &'a Tensor<LocalMassProperties>,
}

impl GpuJointSolver {
    /// Generate joint constraints for this set of joints.
    pub fn init(
        &self,
        pass: &mut GpuPass,
        args: &mut JointSolverArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        if args.joints.is_empty() {
            return Ok(());
        }

        self.init_joint_constraints.call(
            pass,
            args.joints.len,
            &args.joints.joints,
            &mut args.joints.builders,
            &mut args.joints.constraints,
            &args.joints.num_joints,
            args.local_mprops,
        )
    }

    /// Updates the non-linear terms of the joint constraints.
    pub fn update(
        &self,
        pass: &mut GpuPass,
        args: &mut JointSolverArgs<'_>,
        poses: &Tensor<Pose>,
    ) -> Result<(), GpuBackendError> {
        if args.joints.is_empty() {
            return Ok(());
        }

        self.update_joint_constraints.call(
            pass,
            args.joints.len,
            &args.joints.builders,
            &mut args.joints.constraints,
            &args.joints.num_joints,
            poses,
            args.mprops,
            args.sim_params,
        )
    }

    /// Apply a single Projected-Gauss-Seidel step for solving joints.
    pub fn solve(
        &self,
        pass: &mut GpuPass,
        args: &mut JointSolverArgs<'_>,
        solver_vels: &mut Tensor<Velocity>,
        use_bias: bool,
    ) -> Result<(), GpuBackendError> {
        if args.joints.is_empty() {
            return Ok(());
        }

        if !use_bias {
            self.remove_joint_bias.call(
                pass,
                args.joints.len,
                &mut args.joints.constraints,
                &args.joints.num_joints,
            )?;
        }

        self.reset_joint_color
            .call(pass, 1u32, &mut args.joints.curr_color)?;

        for _ in 0..args.joints.num_colors {
            // TODO PERF: figure out a way to dispatch a number of threads that fits
            //            more tightly the size of the current color.
            self.solve_joint_constraints.call(
                pass,
                args.joints.max_color_group_len,
                &mut args.joints.constraints,
                solver_vels,
                &args.joints.color_groups,
                &args.joints.curr_color,
            )?;
            self.inc_joint_color
                .call(pass, 1u32, &mut args.joints.curr_color)?
        }

        Ok(())
    }
}
