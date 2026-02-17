//! Timestep bound estimation kernels.
//!
//! Computes a CFL-based timestep bound across all particles. Each thread computes
//! a per-particle bound and atomically reduces it to find the global minimum.

use crate::grid::grid::Grid;
use crate::models::default::{DefaultParticleModel, GpuParticleModel};
use crate::solver::particle::{Kinematics, ParticleProperties};
use crate::PaddingExt;
use crate::{atomic_min_u32, sqrt, Matrix, MaybeIndexUnchecked, PaddedMatrix, DIM};
use khal_derive::spirv_bindgen;
use spirv_std::spirv;

/// GPU-side timestep bound result.
///
/// Uses an atomic unsigned integer to store the minimum timestep across all particles.
/// The float timestep is converted to an integer via a fixed-point scaling factor
/// so that atomic min operations can be used.
#[derive(Clone, Copy, Default, Debug)]
#[cfg_attr(not(target_arch = "spirv"), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct GpuTimestepBounds {
    pub computed_max_dt_as_uint: u32,
}

impl GpuTimestepBounds {
    /// Conversion factor from seconds to integer representation.
    pub const FLOAT_TO_INT: f32 = 1.0e12;

    /// Converts a timestep in seconds to its integer representation.
    ///
    /// Since `secs` is always positive, truncation via `as u32` is equivalent to floor.
    #[inline]
    pub fn secs_to_int(secs: f32) -> u32 {
        (secs * Self::FLOAT_TO_INT) as u32
    }
}

pub const WORKGROUP_SIZE: u32 = 64;

/// Resets the timestep bound to the maximum possible value.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_reset_timestep_bound(
    #[spirv(global_invocation_id)] _invocation_id: spirv_std::glam::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] result: &mut [GpuTimestepBounds],
) {
    result.at_mut(0).computed_max_dt_as_uint = 0xFFFFFFFF;
}

/// Estimates the CFL-based timestep bound across all particles.
///
/// Each thread computes a per-particle timestep bound based on:
/// 1. Material model sound speed (model-specific).
/// 2. Particle velocity and APIC affine matrix contribution.
///
/// The minimum across all particles is stored atomically in `result`.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_estimate_timestep_bound(
    #[spirv(global_invocation_id)] invocation_id: spirv_std::glam::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] grid_data: &[Grid],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    particles_model: &[GpuParticleModel],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] particles_kin: &[Kinematics],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    particles_def_grad: &[PaddedMatrix],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)]
    particles_props: &[ParticleProperties],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] particles_len: &u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] result: &mut [GpuTimestepBounds],
) {
    let particle_id = invocation_id.x;

    if particle_id >= *particles_len {
        return;
    }

    let kin = particles_kin.read(particle_id as usize);

    if kin.enabled == 0 {
        return;
    }

    let def_grad = particles_def_grad.read(particle_id as usize);
    let props = particles_props.read(particle_id as usize);
    let cell_width = grid_data.at(0).cell_width;

    // Model-specific restrictions (usually based on sound speed, section 4.1).
    let density0 = kin.mass / props.init_volume;
    let def_grad = def_grad.remove_padding();
    let velocity = kin.velocity;
    let affine = kin.affine.remove_padding();
    let mass = kin.mass;

    let mut dt = DefaultParticleModel::timestep_bound(
        particles_model,
        particle_id,
        density0,
        def_grad,
        velocity,
        cell_width,
    );

    // Velocity-based restrictions (section 4.2).
    let norm_affine_squared = frobenius_norm_squared(affine);

    let d = (cell_width * cell_width) / 4.0;
    let norm_b = d * sqrt(norm_affine_squared) / mass;
    let apic_v = norm_b * 6.0 * sqrt(DIM as f32) / cell_width;
    let v = velocity.length() + apic_v;
    dt = dt.min(cell_width / v);

    let candidate = GpuTimestepBounds::secs_to_int(dt);
    atomic_min_u32(&mut result.at_mut(0).computed_max_dt_as_uint, candidate);
}

/// Computes the squared Frobenius norm of a matrix (sum of squares of all elements).
#[inline]
fn frobenius_norm_squared(m: Matrix) -> f32 {
    let mut result = 0.0;
    #[cfg(feature = "dim2")]
    {
        result += m.x_axis.x * m.x_axis.x;
        result += m.x_axis.y * m.x_axis.y;
        result += m.y_axis.x * m.y_axis.x;
        result += m.y_axis.y * m.y_axis.y;
    }
    #[cfg(feature = "dim3")]
    {
        result += m.x_axis.x * m.x_axis.x;
        result += m.x_axis.y * m.x_axis.y;
        result += m.x_axis.z * m.x_axis.z;
        result += m.y_axis.x * m.y_axis.x;
        result += m.y_axis.y * m.y_axis.y;
        result += m.y_axis.z * m.y_axis.z;
        result += m.z_axis.x * m.z_axis.x;
        result += m.z_axis.y * m.z_axis.y;
        result += m.z_axis.z * m.z_axis.z;
    }
    result
}
