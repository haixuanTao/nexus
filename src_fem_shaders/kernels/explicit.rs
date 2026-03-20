//! Explicit solver GPU kernels (symplectic Euler integration).
//!
//! Pipeline dispatch order for one substep:
//! 1. `gpu_compute_elastic_forces` (per-element): F → P → atomic scatter forces
//! 2. `gpu_apply_forces_gravity_damping` (per-vertex): accumulated forces → velocity update
//! 3. `gpu_apply_soft_constraints` (per-vertex): spring + critical damping
//! 4. `gpu_integrate_positions` (per-vertex): x += dt * v
//! 5. `gpu_apply_hard_constraints` (per-vertex): set pos to target, zero vel
//! 6. `gpu_boundary_conditions` (per-vertex): floor collision

use crate::material::{compute_stress, precompute};
use crate::types::{
    ElementInfo, FemSimParams, VertexConstraint, VertexInfo, VertexState,
};
use crate::{
    exp_f32, pad_vec, sqrt_f32, unpad_mat, unpad_vec, Matrix, MaybeIndexUnchecked, Vector,
    VERTS_PER_ELEM,
};
use khal_derive::spirv_bindgen;
use spirv_std_macros::spirv;
use vortx_shaders::utils::atomic_add_f32;

// ── Float-atomic vector scatter via CAS loop ──

/// Per-vertex atomic vector accumulator.
/// Stores f32 values as u32 bits for CAS-based atomic add.
/// Always 16 bytes (4 x u32) regardless of dimension for GPU alignment.
#[derive(Clone, Copy, Default)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(Debug, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct AtomicForce {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub _pad: u32,
}

/// Atomically scatter a vector to a vertex's accumulator using CAS-loop float atomics.
#[inline]
pub fn scatter_force(buf: &mut [AtomicForce], vertex_idx: u32, v: Vector) {
    let slot = buf.at_mut(vertex_idx as usize);
    atomic_add_f32(&mut slot.x, v.x);
    atomic_add_f32(&mut slot.y, v.y);
    #[cfg(feature = "dim3")]
    atomic_add_f32(&mut slot.z, v.z);
}

/// Read accumulated float vector (stored as u32 bits).
#[inline]
pub fn read_force(buf: &[AtomicForce], vertex_idx: u32) -> Vector {
    let f = buf.at(vertex_idx as usize);
    #[cfg(feature = "dim2")]
    {
        Vector::new(f32::from_bits(f.x), f32::from_bits(f.y))
    }
    #[cfg(feature = "dim3")]
    {
        Vector::new(f32::from_bits(f.x), f32::from_bits(f.y), f32::from_bits(f.z))
    }
}

// ── Kernel 1: Compute elastic velocity changes (per-element) ──

/// Computes the elastic velocity change (dv) from each element and atomically
/// scatters it to the shared per-vertex accumulator.
///
/// This follows the Genesis approach: each element independently computes
/// dv_k = -(P * S[k]) * (dt / rho), avoiding mass-based force division.
///
/// For each element:
/// 1. Read vertex positions, compute deformation gradient F = D * B_inv
/// 2. Compute first Piola-Kirchhoff stress P = dΨ/dF
/// 3. For each vertex k: dv_k = -(P * S[k]) * (dt / rho), scatter via atomics
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_compute_elastic_forces(
    #[spirv(global_invocation_id)] invocation_id: vortx_shaders::glam::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &FemSimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] elem_info: &[ElementInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] vertex_state: &[VertexState],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] force_atomic: &mut [AtomicForce],
) {
    let elem_id = invocation_id.x;
    if elem_id >= params.num_elements {
        return;
    }

    let elem = elem_info.read(elem_id as usize);

    // Read vertex positions.
    let x0 = unpad_vec(vertex_state.at(elem.indices[0] as usize).pos);
    let x1 = unpad_vec(vertex_state.at(elem.indices[1] as usize).pos);
    let x2 = unpad_vec(vertex_state.at(elem.indices[2] as usize).pos);
    #[cfg(feature = "dim3")]
    let x3 = unpad_vec(vertex_state.at(elem.indices[3] as usize).pos);

    // Compute deformed shape matrix D.
    #[cfg(feature = "dim2")]
    let D = Matrix::from_cols(x1 - x0, x2 - x0);
    #[cfg(feature = "dim3")]
    let D = Matrix::from_cols(x1 - x0, x2 - x0, x3 - x0);

    // Deformation gradient F = D * B_inv.
    let B_inv = unpad_mat(elem.B_inv);
    let F = D * B_inv;

    // Precompute rotation for corotated models (identity for others).
    let precomp = precompute(F, elem.model);
    let R = unpad_mat(precomp.R);

    // Compute first Piola-Kirchhoff stress P = dΨ/dF.
    let P = compute_stress(F, elem.mu, elem.lam, elem.model, R);

    // Scatter velocity changes: dv_k = -(P * S[k]) * (dt / rho) per vertex.
    let dt_over_rho = params.dt / elem.rho;

    let dv0 = -(P * unpad_vec(elem.S[0])) * dt_over_rho;
    scatter_force(force_atomic, elem.indices[0], dv0);

    let dv1 = -(P * unpad_vec(elem.S[1])) * dt_over_rho;
    scatter_force(force_atomic, elem.indices[1], dv1);

    let dv2 = -(P * unpad_vec(elem.S[2])) * dt_over_rho;
    scatter_force(force_atomic, elem.indices[2], dv2);

    #[cfg(feature = "dim3")]
    {
        let dv3 = -(P * unpad_vec(elem.S[3])) * dt_over_rho;
        scatter_force(force_atomic, elem.indices[3], dv3);
    }
}

// ── Kernel 2: Apply accumulated dv, gravity, and damping (per-vertex) ──

/// Reads accumulated elastic velocity changes (dv), adds gravity, applies
/// damping, and updates vertex velocities.
///
/// v += accumulated_dv + gravity * dt
/// v *= exp(-alpha_rayleigh * dt) * exp(-damping * dt)
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_apply_forces_gravity_damping(
    #[spirv(global_invocation_id)] invocation_id: vortx_shaders::glam::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &FemSimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] vertex_state: &mut [VertexState],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] force_atomic: &mut [AtomicForce],
) {
    let v_id = invocation_id.x;
    if v_id >= params.num_vertices {
        return;
    }

    let state = vertex_state.at_mut(v_id as usize);
    let mut vel = unpad_vec(state.vel);

    // Read accumulated elastic dv and clear the accumulator.
    let elastic_dv = read_force(force_atomic, v_id);
    force_atomic.write(v_id as usize, AtomicForce::default());

    // Apply elastic velocity change and gravity.
    let gravity = unpad_vec(params.gravity);
    vel += elastic_dv + gravity * params.dt;

    // Rayleigh mass-proportional damping: v *= exp(-α * dt)
    vel *= exp_f32(-params.alpha_rayleigh * params.dt);

    // Simple velocity damping: v *= exp(-damping * dt)
    vel *= exp_f32(-params.damping * params.dt);

    state.vel = pad_vec(vel);
}

// ── Kernel 3: Apply soft constraints (per-vertex) ──

/// Applies soft spring constraints with critical damping.
///
/// f_spring = stiffness * (target - pos)
/// f_damping = -2 * sqrt(mass * stiffness) * vel
/// v += dt * (f_spring + f_damping) / mass
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_apply_soft_constraints(
    #[spirv(global_invocation_id)] invocation_id: vortx_shaders::glam::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &FemSimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] vertex_info: &[VertexInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] vertex_state: &mut [VertexState],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] constraints: &[VertexConstraint],
) {
    let v_id = invocation_id.x;
    if v_id >= params.num_vertices {
        return;
    }

    let c = constraints.read(v_id as usize);
    if c.is_constrained == 0 || c.is_soft == 0 {
        return;
    }

    let info = vertex_info.read(v_id as usize);
    let state = vertex_state.at_mut(v_id as usize);
    let pos = unpad_vec(state.pos);
    let mut vel = unpad_vec(state.vel);

    let target = unpad_vec(c.target_pos);
    let disp = target - pos;

    // Spring force toward target.
    let f_spring = disp * c.stiffness;

    // Critical damping force.
    let damping_coeff = 2.0 * sqrt_f32(info.mass * c.stiffness);
    let f_damping = -vel * damping_coeff;

    // Update velocity.
    if info.mass > 0.0 {
        vel += (f_spring + f_damping) * (params.dt / info.mass);
    }

    state.vel = pad_vec(vel);
}

// ── Kernel 4: Integrate positions (per-vertex) ──

/// Symplectic Euler position update: x += dt * v.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_integrate_positions(
    #[spirv(global_invocation_id)] invocation_id: vortx_shaders::glam::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &FemSimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] vertex_state: &mut [VertexState],
) {
    let v_id = invocation_id.x;
    if v_id >= params.num_vertices {
        return;
    }

    let state = vertex_state.at_mut(v_id as usize);
    let vel = unpad_vec(state.vel);
    let pos = unpad_vec(state.pos);
    state.pos = pad_vec(pos + vel * params.dt);
}

// ── Kernel 5: Apply hard constraints (per-vertex) ──

/// Enforces hard constraints by setting vertex position to target and zeroing velocity.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_apply_hard_constraints(
    #[spirv(global_invocation_id)] invocation_id: vortx_shaders::glam::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &FemSimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] vertex_state: &mut [VertexState],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] constraints: &[VertexConstraint],
) {
    let v_id = invocation_id.x;
    if v_id >= params.num_vertices {
        return;
    }

    let c = constraints.read(v_id as usize);
    if c.is_constrained == 0 || c.is_soft != 0 {
        return;
    }

    let state = vertex_state.at_mut(v_id as usize);
    state.pos = c.target_pos;
    state.vel = pad_vec(Vector::ZERO);
}

// ── Kernel 6: Boundary conditions (per-vertex) ──

/// Enforces floor collision: if vertex below floor_y, project position up
/// and zero the downward velocity component.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_boundary_conditions(
    #[spirv(global_invocation_id)] invocation_id: vortx_shaders::glam::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &FemSimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] vertex_state: &mut [VertexState],
) {
    let v_id = invocation_id.x;
    if v_id >= params.num_vertices {
        return;
    }

    let state = vertex_state.at_mut(v_id as usize);
    let mut pos = unpad_vec(state.pos);
    let mut vel = unpad_vec(state.vel);

    // Floor collision.
    if pos.y < params.floor_y {
        pos.y = params.floor_y;
        if vel.y < 0.0 {
            vel.y = 0.0;
        }
        // Floor friction: damp horizontal velocity on contact.
        vel.x *= 0.95;
        #[cfg(feature = "dim3")]
        {
            vel.z *= 0.95;
        }
    }

    state.pos = pad_vec(pos);
    state.vel = pad_vec(vel);
}
