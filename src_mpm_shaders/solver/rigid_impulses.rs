//! Rigid body impulse accumulation and integration kernels.
//!
//! Handles the coupling between MPM particles and rigid bodies by:
//! 1. Accumulating integer-quantized impulses from P2G transfers.
//! 2. Converting accumulated impulses to float and applying them to rigid bodies.
//! 3. Updating world-space mass properties after pose changes.

use crate::grid::grid::Grid;
use crate::nexus_shaders::dynamics::{
    apply_impulse, integrate_velocity, update_mprops, Impulse, LocalMassProperties, Velocity,
    WorldMassProperties,
};
use crate::solver::params::SimulationParams;
use crate::{ang_length, AngVector, IVector, MaybeIndexUnchecked, Pose, Vector};
use glamx::*;
use khal_derive::spirv_bindgen;
use spirv_std::spirv;

/// Integer-quantized impulse for atomic accumulation during P2G.
///
/// Uses integer representation to enable atomic add operations on the GPU.
/// The `com` field stores the center of mass to reduce binding count in P2G.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch = "spirv"), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct IntegerImpulse {
    /// Center of mass (stored here to reduce P2G binding count).
    pub com: Vector,
    #[cfg(feature = "dim3")]
    pub padding0: i32,
    /// Integer-quantized linear impulse.
    pub linear: IVector,
    /// Integer-quantized angular impulse (scalar in 2D).
    #[cfg(feature = "dim2")]
    pub angular: i32,
    pub padding1: i32,
    /// Integer-quantized angular impulse (IVec3 in 3D).
    #[cfg(feature = "dim3")]
    pub angular: IVector,
    #[cfg(feature = "dim3")]
    pub padding2: i32,
}

/// Scaling factor for float-to-integer impulse conversion.
pub const FLOAT_TO_INT_FACTOR: f32 = 1e5;

/// Converts a float value to its integer-quantized representation.
#[inline]
pub fn flt2int(flt: f32) -> i32 {
    (flt * FLOAT_TO_INT_FACTOR) as i32
}

/// Converts an integer-quantized value back to float.
#[inline]
pub fn int2flt(i: i32) -> f32 {
    i as f32 / FLOAT_TO_INT_FACTOR
}

/// Converts an integer impulse to a floating-point Impulse.
#[inline]
pub fn int_impulse_to_float(imp: &IntegerImpulse) -> Impulse {
    #[cfg(feature = "dim2")]
    {
        Impulse::new(
            Vec2::new(int2flt(imp.linear.x), int2flt(imp.linear.y)),
            int2flt(imp.angular),
        )
    }
    #[cfg(feature = "dim3")]
    {
        Impulse::new(
            Vec3::new(
                int2flt(imp.linear.x),
                int2flt(imp.linear.y),
                int2flt(imp.linear.z),
            ),
            Vec3::new(
                int2flt(imp.angular.x),
                int2flt(imp.angular.y),
                int2flt(imp.angular.z),
            ),
        )
    }
}

/// Updates rigid body velocities and poses by applying accumulated impulses.
///
/// For each rigid body:
/// 1. Converts integer impulses to float.
/// 2. Resets the integer impulse accumulator for the next substep.
/// 3. Applies the impulse to update velocity.
/// 4. Caps velocities to prevent excessive movement per substep.
/// 5. Integrates velocity to update pose.
/// 6. Applies gravity.
///
/// NOTE: numthreads(16) because we are currently limited to 16 bodies
/// due to the CPIC affinity bitmask size.
#[spirv_bindgen]
#[spirv(compute(threads(16)))]
pub fn gpu_rigid_impulses_update(
    #[spirv(global_invocation_id)] invocation_id: spirv_std::glam::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] sim_params: &SimulationParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    local_mprops: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] poses: &mut [Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] mprops: &mut [WorldMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)]
    incremental_impulses: &mut [IntegerImpulse],
) {
    let id = invocation_id.x;

    if id < vels.len() as u32 {
        let idx = id as usize;
        let inc_impulse = int_impulse_to_float(incremental_impulses.at(idx));

        // Reset the incremental impulse to zero for the next substep.
        *incremental_impulses.at_mut(idx) = IntegerImpulse::default();

        // Apply impulse and integrate.
        let current_vel = vels.read(idx);
        let current_mprops = mprops.read(idx);
        let mut new_vel = apply_impulse(&current_mprops, &current_vel, &inc_impulse);

        // Cap the velocities to not move more than a fraction of a cell-width in a given substep.
        let linvel_norm = new_vel.linear.length();
        let angvel_norm = ang_length(new_vel.angular);
        let lin_limit = 0.1 * grid.cell_width / sim_params.dt;
        let ang_limit = 1.0; // TODO: what's a good angular limit?

        let impulse_linear_len = inc_impulse.linear.length();
        let impulse_angular_len = ang_length(inc_impulse.angular);

        if impulse_linear_len != 0.0 || impulse_angular_len != 0.0 {
            if linvel_norm > lin_limit {
                new_vel.linear = new_vel.linear * (lin_limit / linvel_norm);
            }
            if angvel_norm > ang_limit {
                new_vel.angular = new_vel.angular * (ang_limit / angvel_norm);
            }
        }

        let current_pose = poses.read(idx);
        let local_mp = local_mprops.read(idx);
        let new_pose = integrate_velocity(current_pose, &new_vel, local_mp.com, sim_params.dt);

        // Apply gravity.
        // Construct a mask: 1.0 where inv_mass != 0.0, 0.0 otherwise.
        #[cfg(feature = "dim2")]
        let mass_mask = Vec2::new(
            if current_mprops.inv_mass.x != 0.0 {
                1.0
            } else {
                0.0
            },
            if current_mprops.inv_mass.y != 0.0 {
                1.0
            } else {
                0.0
            },
        );
        #[cfg(feature = "dim3")]
        let mass_mask = Vec3::new(
            if current_mprops.inv_mass.x != 0.0 {
                1.0
            } else {
                0.0
            },
            if current_mprops.inv_mass.y != 0.0 {
                1.0
            } else {
                0.0
            },
            if current_mprops.inv_mass.z != 0.0 {
                1.0
            } else {
                0.0
            },
        );
        new_vel.linear += sim_params.gravity * mass_mask * sim_params.dt;

        vels.write(idx, new_vel);
        poses.write(idx, new_pose);
    }
}

/// Updates world-space mass properties from local properties and current poses.
///
/// Also writes the updated center of mass into the incremental impulse buffer
/// so that P2G can access it without an extra binding.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_update_world_mass_properties(
    #[spirv(global_invocation_id)] invocation_id: spirv_std::glam::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    local_mprops: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] mprops: &mut [WorldMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    incremental_impulses: &mut [IntegerImpulse],
) {
    let id = invocation_id.x;

    if id < mprops.len() as u32 {
        let idx = id as usize;
        let pose = poses.read(idx);
        let local_mp = local_mprops.read(idx);
        let new_mprops = update_mprops(pose, &local_mp);
        incremental_impulses.at_mut(idx).com = new_mprops.com;
        mprops.write(idx, new_mprops);
    }
}
