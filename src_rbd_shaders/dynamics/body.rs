//! Rigid-body dynamics data structures and integration
//!
//! This module defines the fundamental data structures and operations for rigid-body
//! dynamics, including mass properties, forces, velocities, and integration routines.
//!
//! Key Concepts:
//! - Mass Properties: Defines how a body resists linear and angular motion
//! - Velocity Integration: Semi-implicit Euler integration for pose updates
//! - Impulse Application: Computing velocity changes from constraint forces
//!
//! Dimension Support:
//! - 2D: Rotation represented as scalar angle, inertia as scalar
//! - 3D: Rotation represented as quaternion, inertia as 3x3 matrix

use crate::{AngVector, Pose, Vector};
#[cfg(feature = "dim3")]
use crate::{rotation_from_scaled_axis, rotation_renormalize_fast, rotation_to_matrix};

#[cfg(feature = "dim3")]
use glamx::{Mat3, Mat4, Quat, Vec3};
#[cfg(feature = "dim2")]
use glamx::{Rot2, Vec2};

/// The mass-properties of a rigid-body in local (body-space) coordinates.
///
/// Local mass properties are defined relative to the body's local coordinate frame
/// and remain constant unless the body's shape changes.
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable, Debug)
)]
#[repr(C)]
pub struct LocalMassProperties {
    #[cfg(feature = "dim3")]
    /// The reference frame for the principal inertia axes (3D only).
    pub inertia_ref_frame: Quat,

    #[cfg(feature = "dim3")]
    /// The inverse principal inertia components (3D only).
    pub inv_principal_inertia: Vec3,

    #[cfg(feature = "dim3")]
    pub padding0: u32,

    /// The rigid-body's inverse mass along each coordinate axis.
    pub inv_mass: Vector,

    #[cfg(feature = "dim3")]
    pub padding1: u32,

    /// The rigid-body's center of mass in local coordinates.
    pub com: Vector,

    pub padding2: u32,

    #[cfg(feature = "dim2")]
    /// The inverse inertia tensor (scalar in 2D).
    pub inv_inertia: f32,
}

impl Default for LocalMassProperties {
    fn default() -> Self {
        Self {
            #[cfg(feature = "dim3")]
            inertia_ref_frame: Quat::IDENTITY,
            #[cfg(feature = "dim3")]
            inv_principal_inertia: Vec3::ONE,
            #[cfg(feature = "dim3")]
            padding0: 0,
            inv_mass: Vector::ONE,
            #[cfg(feature = "dim3")]
            padding1: 0,
            com: Vector::ZERO,
            padding2: 0,
            #[cfg(feature = "dim2")]
            inv_inertia: 1.0,
        }
    }
}

/// The mass-properties of a rigid-body in world-space coordinates.
///
/// World mass properties are computed from local mass properties and the body's
/// current pose. They are updated each frame.
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(Debug, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct WorldMassProperties {
    #[cfg(feature = "dim3")]
    /// The inverse inertia tensor in world space (3x3 matrix in 3D).
    pub inv_inertia: Mat4, // NOTE using a Mat4 to avoid implicit padding.

    /// The rigid-body's inverse mass along each coordinate axis in world space.
    pub inv_mass: Vector,

    #[cfg(feature = "dim3")]
    pub padding0: u32,

    /// The rigid-body's center of mass in world-space coordinates.
    pub com: Vector,

    pub padding1: u32,

    #[cfg(feature = "dim2")]
    /// The inverse inertia tensor in world space (scalar in 2D).
    pub inv_inertia: f32,
}

impl WorldMassProperties {
    #[inline]
    pub fn inv_inertia_mul(&self, v: AngVector) -> AngVector {
        #[cfg(feature = "dim2")]
        return self.inv_inertia * v;
        // TODO PERF: this is ugly. Only needed to avoid internal implicit padding in Mat3.
        //            Is there a better option?
        //            Does this have a negative performance impact?
        #[cfg(feature = "dim3")]
        return (self.inv_inertia * v.extend(0.0)).truncate();
    }
}

impl Default for WorldMassProperties {
    fn default() -> Self {
        Self {
            #[cfg(feature = "dim3")]
            inv_inertia: Mat4::IDENTITY,
            inv_mass: Vector::ONE,
            #[cfg(feature = "dim3")]
            padding0: 0,
            com: Vector::ZERO,
            padding1: 0,
            #[cfg(feature = "dim2")]
            inv_inertia: 1.0,
        }
    }
}

/// An impulse (instantaneous change in momentum).
#[derive(Clone, Copy, Default)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct Impulse {
    /// Linear impulse component (change in linear momentum).
    pub linear: Vector,

    /// Angular impulse component (change in angular momentum / torque impulse).
    pub angular: AngVector,
}

impl Impulse {
    pub fn new(linear: Vector, angular: AngVector) -> Self {
        Self { linear, angular }
    }
}

/// A force and torque applied to a rigid body.
#[derive(Clone, Copy, Default)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct Force {
    /// Linear force component.
    pub linear: Vector,

    /// Angular force component (torque).
    pub angular: AngVector,
}

impl Force {
    pub fn new(linear: Vector, angular: AngVector) -> Self {
        Self { linear, angular }
    }
}

/// Linear and angular velocity of a rigid body.
#[derive(Clone, Copy, Default)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct Velocity {
    /// Linear (translational) velocity.
    pub linear: Vector,
    #[cfg(feature = "dim3")]
    pub padding0: u32,

    /// Angular (rotational) velocity.
    pub angular: AngVector,
    pub padding1: u32,
}

impl Velocity {
    pub fn new(linear: Vector, angular: AngVector) -> Self {
        Self {
            linear,
            angular,
            #[cfg(feature = "dim3")]
            padding0: 0,
            padding1: 0,
        }
    }
}

/// Complete state of a rigid body (pose and velocity).
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct RigidBodyState {
    /// The rigid-body's pose (position, orientation, and uniform scale).
    pub pose: Pose,

    /// The rigid-body's velocity (linear and angular).
    pub velocity: Velocity,
}

/// Applies an impulse to a rigid body, computing the resulting velocity change.
///
/// This function implements the fundamental impulse-velocity relationship:
/// Δv_linear = impulse_linear / mass
/// Δv_angular = I⁻¹ * impulse_angular (where I is the inertia tensor)
///
/// Parameters:
/// - mprops: The body's world-space mass properties
/// - velocity: The body's current velocity
/// - imp: The impulse to apply
///
/// Returns: The updated velocity after applying the impulse
pub fn apply_impulse(mprops: &WorldMassProperties, velocity: &Velocity, imp: &Impulse) -> Velocity {
    // Linear velocity change: Δv = impulse * inv_mass
    let acc_lin = mprops.inv_mass * imp.linear;

    // Angular velocity change: Δω = I⁻¹ * angular_impulse
    let acc_ang = mprops.inv_inertia_mul(imp.angular);

    Velocity::new(velocity.linear + acc_lin, velocity.angular + acc_ang)
}

/// Integrates forces over a timestep to compute velocity changes.
///
/// This implements explicit (forward) Euler integration for force application:
/// v_new = v_old + (F / m) * dt
/// ω_new = ω_old + (I⁻¹ * τ) * dt
///
/// Parameters:
/// - mprops: The body's world-space mass properties
/// - velocity: The body's current velocity
/// - force: The force to apply
/// - dt: The timestep duration (seconds)
///
/// Returns: The updated velocity after integrating the force
pub fn integrate_forces(
    mprops: &WorldMassProperties,
    velocity: &Velocity,
    force: &Force,
    dt: f32,
) -> Velocity {
    // Linear acceleration: a = F / m
    let acc_lin = mprops.inv_mass * force.linear;

    // Angular acceleration: α = I⁻¹ * τ
    let acc_ang = mprops.inv_inertia_mul(force.angular);

    // Explicit Euler: v_new = v_old + a * dt
    Velocity::new(
        velocity.linear + acc_lin * dt,
        velocity.angular + acc_ang * dt,
    )
}

#[cfg(feature = "dim2")]
/// Integrates velocity over a timestep to compute the new pose (2D version).
pub fn integrate_velocity(pose: Pose, vels: &Velocity, local_com: Vector, dt: f32) -> Pose {
    use glamx::Pose2;

    // Transform local COM to world space
    let init_com = pose * local_com;
    let init_tra = pose.translation;

    // Integrate angular velocity
    let delta_ang = Rot2::new(vels.angular * dt);

    // Integrate linear velocity
    let delta_lin = vels.linear * dt;

    // New translation: rotate the offset around COM, then add linear displacement
    let new_translation = init_com + delta_ang * (init_tra - init_com) + delta_lin;

    // Compose rotations
    let new_rotation = delta_ang * pose.rotation;

    Pose2::from_parts(new_translation, new_rotation)
}

#[cfg(feature = "dim3")]
/// Integrates velocity over a timestep to compute the new pose (3D version).
pub fn integrate_velocity(pose: Pose, vels: &Velocity, local_com: Vector, dt: f32) -> Pose {
    use glamx::Pose3;

    // Transform local COM to world space
    let init_com = pose * local_com;
    let init_tra = pose.translation;

    // Integrate angular velocity using exponential map
    let delta_ang = rotation_from_scaled_axis(vels.angular * dt);

    // Integrate linear velocity
    let delta_lin = vels.linear * dt;

    // New translation: rotate the offset around COM, then add linear displacement
    let new_translation = init_com + delta_ang * (init_tra - init_com) + delta_lin;

    // Compose quaternions and renormalize
    let new_rotation = rotation_renormalize_fast(delta_ang * pose.rotation);

    Pose3::from_parts(new_translation, new_rotation)
}

#[cfg(feature = "dim2")]
/// Updates world-space mass properties from local properties and current pose (2D version).
pub fn update_mprops(pose: Pose, local_mprops: &LocalMassProperties) -> WorldMassProperties {
    let world_com = pose * local_mprops.com;

    // In 2D, inertia is scalar and doesn't change with rotation
    WorldMassProperties {
        inv_inertia: local_mprops.inv_inertia,
        inv_mass: local_mprops.inv_mass,
        com: world_com,
        #[cfg(feature = "dim2")]
        padding1: 0,
    }
}

#[cfg(feature = "dim3")]
/// Updates world-space mass properties from local properties and current pose (3D version).
pub fn update_mprops(pose: Pose, local_mprops: &LocalMassProperties) -> WorldMassProperties {
    let world_com = pose * local_mprops.com;

    // Combine body rotation with principal axes frame rotation
    let rot_mat = rotation_to_matrix(pose.rotation * local_mprops.inertia_ref_frame);

    // Create diagonal matrix from principal inertia components
    let diag = Mat3::from_cols(
        Vec3::new(local_mprops.inv_principal_inertia.x, 0.0, 0.0),
        Vec3::new(0.0, local_mprops.inv_principal_inertia.y, 0.0),
        Vec3::new(0.0, 0.0, local_mprops.inv_principal_inertia.z),
    );

    // Transform inertia to world space: I_world = R * I_principal * R^T
    let world_inv_inertia = rot_mat * diag * rot_mat.transpose();

    WorldMassProperties {
        inv_inertia: glamx::Mat4::from_mat3(world_inv_inertia),
        inv_mass: local_mprops.inv_mass,
        com: world_com,
        #[cfg(feature = "dim3")]
        padding0: 0,
        #[cfg(feature = "dim3")]
        padding1: 0,
    }
}

/// Computes the linear velocity at a specific point on a rigid body.
pub fn velocity_at_point(com: Vector, vels: &Velocity, point: Vector) -> Vector {
    let lever_arm = point - com;

    #[cfg(feature = "dim2")]
    {
        // In 2D: ω × r = ω * perpendicular(r) = ω * (-r.y, r.x)
        vels.linear + vels.angular * Vec2::new(-lever_arm.y, lever_arm.x)
    }
    #[cfg(feature = "dim3")]
    {
        // v_point = v_linear + ω × (point - COM)
        vels.linear + vels.angular.cross(lever_arm)
    }
}
