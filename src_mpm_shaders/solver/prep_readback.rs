//! Readback preparation shader: computes per-particle render data on the GPU.
//!
//! This shader transforms raw particle positions and dynamics into `ReadbackData`
//! suitable for CPU rendering, avoiding the need to transfer full `Position` and
//! `Dynamics` buffers back to the CPU.

use crate::grid::grid::Grid;
use crate::solver::params::SimulationParams;
use crate::solver::particle::{Cdf, Kinematics, ParticleProperties, Position};
use crate::{Matrix, PaddedMatrix, PaddingExt, Vector, abs, acos, cos, diag, sqrt};
use glamx::*;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use crate::glamx::MatExt;

const TAU: f32 = 6.283185307179586;

const RENDER_MODE_DEFAULT: u32 = 0;
const RENDER_MODE_VOLUME: u32 = 1;
const RENDER_MODE_VELOCITY: u32 = 2;
const RENDER_MODE_PHASE: u32 = 3;
const RENDER_MODE_CDF_NORMALS: u32 = 4;
const RENDER_MODE_CDF_DISTANCES: u32 = 5;
const RENDER_MODE_CDF_SIGNS: u32 = 6;

/// Render configuration for the readback shader.
#[derive(Clone, Copy, Default)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct RenderConfig {
    pub mode: u32,
}

/// Per-particle data prepared on the GPU for CPU-side rendering.
///
/// This struct is written by the GPU readback shader and read back to the CPU.
/// Uses explicit padding to ensure layout matches between host (with SIMD alignment
/// for Vec4) and GPU (scalar block layout).
#[cfg(feature = "dim2")]
#[derive(Clone, Copy, Default)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(Debug, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct ReadbackData {
    pub color: Vec4,
    pub deformation: Mat2,
    pub position: Vec2,
    // Explicit padding: Vec4 has 16-byte alignment on host (SIMD), so the struct
    // size must be a multiple of 16. Without padding: 16+16+8 = 40, rounded to 48.
    // We add 8 bytes of explicit padding to satisfy bytemuck::Pod (no implicit padding).
    pub _pad: [f32; 2],
}

/// Per-particle data prepared on the GPU for CPU-side rendering.
///
/// Uses `PaddedMatrix` (Mat4 in 3D) instead of Mat3 to avoid SPIR-V storage buffer
/// alignment issues (Vec3 straddles 16-byte boundaries with std430 layout).
/// On the host side, use `deformation.remove_padding()` to get the Mat3.
#[cfg(feature = "dim3")]
#[derive(Clone, Copy, Default)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(Debug, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct ReadbackData {
    pub color: Vec4,
    pub deformation: PaddedMatrix,
    pub position: Vec3,
    // Vec4(16) + Mat4(64) + Vec3(12) = 92 bytes. Struct align is 16 (Vec4/Mat4).
    // Padded to 96 (next multiple of 16).
    pub _pad: f32,
}

// TODO: really needed?
#[inline]
fn fmax(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

// TODO: really needed?
#[inline]
fn fclamp(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// Compute the clamped and scaled deformation matrix for rendering.
#[cfg(feature = "dim2")]
#[inline]
fn compute_deformation(def_grad: PaddedMatrix, init_radius: f32) -> Mat2 {
    let init_def = diag(Vector::splat(init_radius * 2.0));
    let clamped = Mat2::from_cols(
        def_grad.x_axis.clamp(Vec2::splat(-4.0), Vec2::splat(4.0)),
        def_grad.y_axis.clamp(Vec2::splat(-4.0), Vec2::splat(4.0)),
    );
    init_def * clamped
}

/// Compute the clamped and scaled deformation matrix for rendering.
/// Returns `PaddedMatrix` (Mat4) to avoid SPIR-V alignment issues in storage buffers.
#[cfg(feature = "dim3")]
#[inline]
fn compute_deformation(def_grad: PaddedMatrix, init_radius: f32) -> PaddedMatrix {
    let init_def = diag(Vector::splat(init_radius * 2.0));
    let def3 = def_grad.remove_padding();
    let clamped = Mat3::from_cols(
        def3.x_axis.clamp(Vec3::splat(-4.0), Vec3::splat(4.0)),
        def3.y_axis.clamp(Vec3::splat(-4.0), Vec3::splat(4.0)),
        def3.z_axis.clamp(Vec3::splat(-4.0), Vec3::splat(4.0)),
    );
    PaddedMatrix::add_padding(init_def * clamped)
}

/// Compute the color for a particle based on the render mode.
#[cfg(feature = "dim2")]
#[inline]
fn compute_color(
    kin: &Kinematics,
    cdf: &Cdf,
    def_grad: &PaddedMatrix,
    props: &ParticleProperties,
    base_color: Vec4,
    mode: u32,
    cell_width: f32,
    dt: f32,
) -> Vec4 {
    if mode == RENDER_MODE_VELOCITY {
        let vel = kin.velocity;
        let c = Vec2::new(abs(vel.x), abs(vel.y)) * dt * 100.0 + Vec2::splat(0.2);
        Vec4::new(c.x, c.y, base_color.z, base_color.w)
    } else if mode == RENDER_MODE_VOLUME {
        let sv = def_grad.svd().s;
        let c = (Vec2::ONE - sv) / 0.005 + Vec2::splat(0.2);
        Vec4::new(c.x, c.y, base_color.z, base_color.w)
    } else if mode == RENDER_MODE_PHASE {
        let phase = props.phase;
        Vec4::new(0.0, 0.4 * phase, 0.4 * (1.0 - phase), base_color.w)
    } else if mode == RENDER_MODE_CDF_NORMALS {
        let normal = cdf.normal;
        if normal == Vec2::ZERO {
            Vec4::new(0.0, 0.0, 0.0, base_color.w)
        } else {
            let n = (normal + Vec2::ONE) * 0.5;
            Vec4::new(n.x, n.y, 0.0, base_color.w)
        }
    } else if mode == RENDER_MODE_CDF_DISTANCES {
        let d = cdf.signed_distance / (cell_width * 1.5);
        if d > 0.0 {
            Vec4::new(0.0, abs(d), 0.0, base_color.w)
        } else {
            Vec4::new(abs(d), 0.0, 0.0, base_color.w)
        }
    } else if mode == RENDER_MODE_CDF_SIGNS {
        let d = cdf.affinity;
        let a = (d.0 >> 16) & (d.0 & 0x0000ffff);
        if d.0 == 0 {
            Vec4::new(0.0, 0.0, 0.0, base_color.w)
        } else if a == 0 {
            Vec4::new(0.0, 1.0, 0.0, base_color.w)
        } else {
            Vec4::new(1.0, 0.0, 0.0, base_color.w)
        }
    } else {
        // Default mode.
        base_color
    }
}

/// Compute the color for a particle based on the render mode.
#[cfg(feature = "dim3")]
#[inline]
fn compute_color(
    kin: &Kinematics,
    cdf: &Cdf,
    def_grad: &PaddedMatrix,
    props: &ParticleProperties,
    base_color: Vec4,
    mode: u32,
    cell_width: f32,
    dt: f32,
) -> Vec4 {
    let failed = kin.enabled == 0;

    let color = if mode == RENDER_MODE_VELOCITY {
        let vel = kin.velocity;
        let c = Vec3::new(abs(vel.x), abs(vel.y), abs(vel.z)) * dt * 100.0 + Vec3::splat(0.2);
        Vec4::new(c.x, c.y, c.z, base_color.w)
    } else if mode == RENDER_MODE_VOLUME {
        let sv = def_grad.remove_padding().svd().s;
        let c = (Vec3::ONE - sv) / 0.005 + Vec3::splat(0.2);
        Vec4::new(c.x, c.y, c.z, base_color.w)
    } else if mode == RENDER_MODE_PHASE {
        let phase = props.phase;
        Vec4::new(0.0, 0.4 * phase, 0.4 * (1.0 - phase), base_color.w)
    } else if mode == RENDER_MODE_CDF_NORMALS {
        let normal = cdf.normal;
        if normal == Vec3::ZERO {
            Vec4::new(0.0, 0.0, 0.0, base_color.w)
        } else {
            let n = (normal + Vec3::ONE) * 0.5;
            Vec4::new(n.x, n.y, n.z, base_color.w)
        }
    } else if mode == RENDER_MODE_CDF_DISTANCES {
        let d = cdf.signed_distance / (cell_width * 1.5);
        if d > 0.0 {
            Vec4::new(0.0, abs(d), 0.0, base_color.w)
        } else {
            Vec4::new(abs(d), 0.0, 0.0, base_color.w)
        }
    } else if mode == RENDER_MODE_CDF_SIGNS {
        let d = cdf.affinity;
        let a = (d.0 >> 16) & (d.0 & 0x0000ffff);
        if d.0 == 0 {
            Vec4::new(0.0, 0.0, 0.0, base_color.w)
        } else if a == 0 {
            Vec4::new(0.0, 1.0, 0.0, base_color.w)
        } else {
            Vec4::new(1.0, 0.0, 0.0, base_color.w)
        }
    } else {
        // Default mode.
        base_color
    };

    // Mark disabled (failed) particles red.
    if failed {
        Vec4::new(1.0, 0.0, 0.0, 1.0)
    } else {
        color
    }
}

/// GPU kernel: prepare per-particle readback data for rendering.
///
/// Reads particle positions and dynamics, computes render color and scaled
/// deformation matrix, and writes the result to the `instances` buffer.
/// Dispatched with one thread per particle.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_prep_readback(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] instances: &mut [ReadbackData],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] particles_pos: &[Position],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] particles_kin: &[Kinematics],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] particles_def_grad: &[PaddedMatrix],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)]
    particles_props: &[ParticleProperties],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] base_colors: &[Vec4],
    #[spirv(uniform, descriptor_set = 0, binding = 7)] params: &SimulationParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] config: &[RenderConfig],
    #[spirv(uniform, descriptor_set = 0, binding = 9)] particles_len: &u32,
) {
    let particle_id = invocation_id.x;

    if particle_id >= *particles_len {
        return;
    }

    let pid = particle_id as usize;
    let kin = particles_kin.at(pid);
    let cdf = &kin.cdf;
    let def_grad = particles_def_grad.at(pid);
    let props = particles_props.at(pid);
    let pos = particles_pos.at(pid);
    let base_color = *base_colors.at(pid);
    let cell_width = grid.cell_width;
    let mode = config.at(0).mode;
    let dt = params.dt;

    let deformation = compute_deformation(*def_grad, props.init_radius);
    let color = compute_color(kin, cdf, def_grad, props, base_color, mode, cell_width, dt);

    #[cfg(feature = "dim2")]
    {
        *instances.at_mut(pid) = ReadbackData {
            color,
            deformation,
            position: pos.pt,
            _pad: [0.0; 2],
        };
    }
    #[cfg(feature = "dim3")]
    {
        *instances.at_mut(pid) = ReadbackData {
            color,
            deformation,
            position: pos.pt,
            _pad: 0.0,
        };
    }
}

/// GPU kernel: prepare per-rigid-particle readback data for rendering.
///
/// Reads rigid particle world positions and writes `ReadbackData` with a fixed
/// deformation scale (no deformation gradient) and base color from a palette.
/// Dispatched with one thread per rigid particle.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_prep_readback_rigid(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] instances: &mut [ReadbackData],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] particles_pos: &[Position],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] base_colors: &[Vec4],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] grid: &Grid,
    #[spirv(uniform, descriptor_set = 0, binding = 4)] particles_len: &u32,
) {
    let particle_id = invocation_id.x;

    if particle_id >= *particles_len {
        return;
    }

    let pid = particle_id as usize;
    let pos = particles_pos.at(pid);
    let base_color = *base_colors.at(pid);
    let cell_width = grid.cell_width;
    let scale = cell_width * 0.4;

    #[cfg(feature = "dim2")]
    {
        let deformation = diag(Vector::splat(scale));
        *instances.at_mut(pid) = ReadbackData {
            color: base_color,
            deformation,
            position: pos.pt,
            _pad: [0.0; 2],
        };
    }
    #[cfg(feature = "dim3")]
    {
        let deformation = PaddedMatrix::add_padding(diag(Vector::splat(scale)));
        *instances.at_mut(pid) = ReadbackData {
            color: base_color,
            deformation,
            position: pos.pt,
            _pad: 0.0,
        };
    }
}
