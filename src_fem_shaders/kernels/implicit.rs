//! Implicit solver GPU kernels (Newton-PCG with backtracking line search).
//!
//! Dispatch order per substep:
//! ```text
//! gpu_init_implicit_step (per-vertex)
//! for newton_iter:
//!     gpu_precompute_material (per-element)
//!     gpu_compute_egh (per-element)
//!     gpu_scatter_elastic_force_diag (per-element)
//!     gpu_assemble_and_pcg_init (per-vertex)
//!     gpu_pcg_reduce_init (1 thread)
//!     for pcg_iter:
//!         gpu_pcg_scatter_Ap (per-element)
//!         gpu_pcg_finalize_Ap_dot (per-vertex)
//!         gpu_pcg_compute_alpha (1 thread)
//!         gpu_pcg_update_x_r_z (per-vertex)
//!         gpu_pcg_compute_beta (1 thread)
//!         gpu_pcg_update_p (per-vertex)
//!     gpu_ls_init (per-vertex)
//!     gpu_ls_energy_element (per-element)
//!     gpu_ls_finalize_init (1 thread)
//!     for ls_iter:
//!         gpu_ls_update_pos (per-vertex)
//!         gpu_ls_energy_vertex (per-vertex)
//!         gpu_ls_energy_element (per-element)
//!         gpu_ls_check_armijo (1 thread)
//! gpu_compute_velocity (per-vertex)
//! gpu_boundary_conditions (from explicit.rs)
//! ```

use super::explicit::{AtomicForce, read_force, scatter_force};
use crate::material::{compute_energy, compute_hessian_blocks, compute_stress, precompute};
use crate::types::{
    ElementEnergyGrad, ElementHessian, ElementInfo, ElementPrecomputed, FemSimParams,
    LinesearchScalars, PcgScalars, PcgVertexState, VertexConstraint, VertexInfo, VertexState,
};
use crate::{
    DIM, Matrix, PaddedVector, VERTS_PER_ELEM, Vector, abs_f32, diag, pad_mat, pad_vec, unpad_mat,
    unpad_vec,
};
use khal_std::{
    sync::{atomic_add_f32, workgroup_memory_barrier_with_group_sync},
    index::MaybeIndexUnchecked,
    macros::{spirv, spirv_bindgen},
};

// ── Workgroup-level parallel reduction for scalars ──

const WG_SIZE: usize = 64;

/// One step of tree reduction within shared memory.
#[inline]
fn wg_reduce_step(tid: usize, stride: usize, shared: &mut [f32; WG_SIZE]) {
    if tid < stride {
        let sum = shared.read(tid) + shared.read(tid + stride);
        shared.write(tid, sum);
    }
    workgroup_memory_barrier_with_group_sync();
}

/// Workgroup-level sum reduction + single atomic add to global.
///
/// All 64 threads in the workgroup must call this (OOB threads pass value=0).
/// Reduces in shared memory (log2(64) = 6 steps), then thread 0 does one
/// atomic add to the global accumulator. This reduces CAS contention from
/// N threads to N/64 workgroups.
#[inline]
fn wg_reduce_add(lid: usize, value: f32, shared: &mut [f32; WG_SIZE], global: &mut u32) {
    shared.write(lid, value);
    workgroup_memory_barrier_with_group_sync();
    wg_reduce_step(lid, 32, shared);
    wg_reduce_step(lid, 16, shared);
    wg_reduce_step(lid, 8, shared);
    wg_reduce_step(lid, 4, shared);
    wg_reduce_step(lid, 2, shared);
    wg_reduce_step(lid, 1, shared);
    if lid == 0 {
        atomic_add_f32(global, shared.read(0));
    }
    // Barrier before shared memory can be reused (e.g., for a second reduction).
    workgroup_memory_barrier_with_group_sync();
}

// ── Atomic types for diagonal Hessian block accumulation ──

/// Per-vertex atomic accumulator for diagonal Hessian block.
/// Stores f32 values as u32 bits for CAS-based atomic add.
#[derive(Clone, Copy, Default)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(Debug, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct AtomicDiag {
    #[cfg(feature = "dim2")]
    pub vals: [u32; 4],
    #[cfg(feature = "dim3")]
    pub vals: [u32; 12], // 9 used + 3 padding
}

/// Atomically scatter a matrix to a diagonal Hessian accumulator using CAS-loop float atomics.
#[inline]
fn scatter_diag(buf: &mut [AtomicDiag], vertex_idx: u32, m: Matrix) {
    let d = buf.at_mut(vertex_idx as usize);
    #[cfg(feature = "dim2")]
    {
        atomic_add_f32(&mut d.vals[0], m.x_axis.x);
        atomic_add_f32(&mut d.vals[1], m.x_axis.y);
        atomic_add_f32(&mut d.vals[2], m.y_axis.x);
        atomic_add_f32(&mut d.vals[3], m.y_axis.y);
    }
    #[cfg(feature = "dim3")]
    {
        atomic_add_f32(&mut d.vals[0], m.x_axis.x);
        atomic_add_f32(&mut d.vals[1], m.x_axis.y);
        atomic_add_f32(&mut d.vals[2], m.x_axis.z);
        atomic_add_f32(&mut d.vals[3], m.y_axis.x);
        atomic_add_f32(&mut d.vals[4], m.y_axis.y);
        atomic_add_f32(&mut d.vals[5], m.y_axis.z);
        atomic_add_f32(&mut d.vals[6], m.z_axis.x);
        atomic_add_f32(&mut d.vals[7], m.z_axis.y);
        atomic_add_f32(&mut d.vals[8], m.z_axis.z);
    }
}

/// Read accumulated diagonal Hessian block (stored as u32 bits).
#[inline]
fn read_diag(buf: &[AtomicDiag], vertex_idx: u32) -> Matrix {
    let d = buf.at(vertex_idx as usize);
    #[cfg(feature = "dim2")]
    {
        Matrix::from_cols(
            Vector::new(f32::from_bits(d.vals[0]), f32::from_bits(d.vals[1])),
            Vector::new(f32::from_bits(d.vals[2]), f32::from_bits(d.vals[3])),
        )
    }
    #[cfg(feature = "dim3")]
    {
        Matrix::from_cols(
            Vector::new(
                f32::from_bits(d.vals[0]),
                f32::from_bits(d.vals[1]),
                f32::from_bits(d.vals[2]),
            ),
            Vector::new(
                f32::from_bits(d.vals[3]),
                f32::from_bits(d.vals[4]),
                f32::from_bits(d.vals[5]),
            ),
            Vector::new(
                f32::from_bits(d.vals[6]),
                f32::from_bits(d.vals[7]),
                f32::from_bits(d.vals[8]),
            ),
        )
    }
}

// ── Scalar reduction slot indices ──

const SLOT_RTZ: usize = 0;
const SLOT_PTAP: usize = 1;
const SLOT_RTZ_NEW: usize = 2;
const SLOT_M: usize = 3;
const SLOT_ENERGY_V: usize = 4;
const SLOT_ENERGY_E: usize = 5;

// ── Helper: compute s^T * H * s block product ──

/// Computes sum_{a,b} s1[a] * H[a*DIM+b] * s2[b], returning a DIM×DIM matrix.
#[inline]
fn s_H_s_block(s1: Vector, hessian: &ElementHessian, s2: Vector) -> Matrix {
    let mut result = Matrix::ZERO;
    #[cfg(feature = "dim2")]
    {
        result += unpad_mat(hessian.blocks[0]) * (s1.x * s2.x);
        result += unpad_mat(hessian.blocks[1]) * (s1.x * s2.y);
        result += unpad_mat(hessian.blocks[2]) * (s1.y * s2.x);
        result += unpad_mat(hessian.blocks[3]) * (s1.y * s2.y);
    }
    #[cfg(feature = "dim3")]
    {
        result += unpad_mat(hessian.blocks[0]) * (s1.x * s2.x);
        result += unpad_mat(hessian.blocks[1]) * (s1.x * s2.y);
        result += unpad_mat(hessian.blocks[2]) * (s1.x * s2.z);
        result += unpad_mat(hessian.blocks[3]) * (s1.y * s2.x);
        result += unpad_mat(hessian.blocks[4]) * (s1.y * s2.y);
        result += unpad_mat(hessian.blocks[5]) * (s1.y * s2.z);
        result += unpad_mat(hessian.blocks[6]) * (s1.z * s2.x);
        result += unpad_mat(hessian.blocks[7]) * (s1.z * s2.y);
        result += unpad_mat(hessian.blocks[8]) * (s1.z * s2.z);
    }
    result
}

/// Compute deformation gradient F from element vertex positions.
#[inline]
fn compute_F(elem: &ElementInfo, vertex_state: &[VertexState]) -> Matrix {
    let x0 = unpad_vec(vertex_state.at(elem.indices[0] as usize).pos);
    let x1 = unpad_vec(vertex_state.at(elem.indices[1] as usize).pos);
    let x2 = unpad_vec(vertex_state.at(elem.indices[2] as usize).pos);
    #[cfg(feature = "dim3")]
    let x3 = unpad_vec(vertex_state.at(elem.indices[3] as usize).pos);

    #[cfg(feature = "dim2")]
    let D = Matrix::from_cols(x1 - x0, x2 - x0);
    #[cfg(feature = "dim3")]
    let D = Matrix::from_cols(x1 - x0, x2 - x0, x3 - x0);

    D * unpad_mat(elem.B_inv)
}

// ══════════════════════════════════════════════════════════════════════
//  NEWTON SETUP
// ══════════════════════════════════════════════════════════════════════

/// Initialize the implicit substep: save x_prev, compute inertia target y,
/// handle hard constraints.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_init_implicit_step(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &FemSimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] vertex_state: &mut [VertexState],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] constraints: &[VertexConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] pcg_vertex: &mut [PcgVertexState],
) {
    let v_id = invocation_id.x;
    if v_id >= params.num_vertices {
        return;
    }

    let state = vertex_state.at_mut(v_id as usize);
    let pos = unpad_vec(state.pos);
    let vel = unpad_vec(state.vel);
    let gravity = unpad_vec(params.gravity);
    let dt = params.dt;

    let pcg = pcg_vertex.at_mut(v_id as usize);
    pcg.x_prev = state.pos;

    // Inertia target: y = x + v*dt + g*dt²
    let mut y = pos + vel * dt + gravity * (dt * dt);

    // Hard constraints: set y and position to target.
    let c = constraints.read(v_id as usize);
    if c.is_constrained != 0 && c.is_soft == 0 {
        y = unpad_vec(c.target_pos);
        state.pos = c.target_pos;
    }

    // Clamp inertia target to floor — prevents the optimizer from
    // trying to push vertices below the floor.
    if y.y < params.floor_y {
        y.y = params.floor_y;
    }

    pcg.y = pad_vec(y);
}

// ══════════════════════════════════════════════════════════════════════
//  MATERIAL PRECOMPUTE & ENERGY/GRADIENT/HESSIAN
// ══════════════════════════════════════════════════════════════════════

/// Precompute per-element data (SVD rotation for corotated models).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_precompute_material(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &FemSimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] elem_info: &[ElementInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] vertex_state: &[VertexState],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    elem_precomp: &mut [ElementPrecomputed],
) {
    let elem_id = invocation_id.x;
    if elem_id >= params.num_elements {
        return;
    }

    let elem = elem_info.read(elem_id as usize);
    let F = compute_F(&elem, vertex_state);
    elem_precomp.write(elem_id as usize, precompute(F, elem.model));
}

/// Compute per-element energy, gradient, and Hessian.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_compute_egh(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &FemSimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] elem_info: &[ElementInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] vertex_state: &[VertexState],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] elem_precomp: &[ElementPrecomputed],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] elem_eg: &mut [ElementEnergyGrad],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] elem_hessian: &mut [ElementHessian],
) {
    let elem_id = invocation_id.x;
    if elem_id >= params.num_elements {
        return;
    }

    let elem = elem_info.read(elem_id as usize);
    let F = compute_F(&elem, vertex_state);
    let R = unpad_mat(elem_precomp.at(elem_id as usize).R);

    // Call energy, stress, and hessian separately to avoid returning tuples
    // containing Mat3, which causes SPIR-V struct alignment issues with naga.
    let energy = compute_energy(F, elem.mu, elem.lam, elem.model, R);
    let gradient = compute_stress(F, elem.mu, elem.lam, elem.model, R);
    let hessian_blocks = compute_hessian_blocks(elem.mu, elem.lam, elem.model, R);

    let mut eg = ElementEnergyGrad::default();
    eg.energy = energy;
    eg.gradient = pad_mat(gradient);
    elem_eg.write(elem_id as usize, eg);

    let mut eh = ElementHessian::default();
    for i in 0..(DIM * DIM) {
        eh.blocks[i] = hessian_blocks[i];
    }
    elem_hessian.write(elem_id as usize, eh);
}

/// Compute per-element energy and gradient only (no Hessian, for subsequent Newton iters
/// when Hessian is constant).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_compute_eg(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &FemSimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] elem_info: &[ElementInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] vertex_state: &[VertexState],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] elem_precomp: &[ElementPrecomputed],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] elem_eg: &mut [ElementEnergyGrad],
) {
    let elem_id = invocation_id.x;
    if elem_id >= params.num_elements {
        return;
    }

    let elem = elem_info.read(elem_id as usize);
    let F = compute_F(&elem, vertex_state);
    let R = unpad_mat(elem_precomp.at(elem_id as usize).R);

    let energy = compute_energy(F, elem.mu, elem.lam, elem.model, R);
    let gradient = compute_stress(F, elem.mu, elem.lam, elem.model, R);

    let mut eg = ElementEnergyGrad::default();
    eg.energy = energy;
    eg.gradient = pad_mat(gradient);
    elem_eg.write(elem_id as usize, eg);
}

// ══════════════════════════════════════════════════════════════════════
//  FORCE ASSEMBLY & PRECONDITIONER
// ══════════════════════════════════════════════════════════════════════

/// Per-element: scatter elastic force and diagonal Hessian contributions
/// to per-vertex atomic accumulators.
///
/// force_k = -vol * P * S[k]
/// diag_k = vol * s_H_s(S[k], H, S[k])
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_scatter_elastic_force_diag(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &FemSimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] elem_info: &[ElementInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] elem_eg: &[ElementEnergyGrad],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] elem_hessian: &[ElementHessian],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] force_atomic: &mut [AtomicForce],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] diag_atomic: &mut [AtomicDiag],
) {
    let elem_id = invocation_id.x;
    if elem_id >= params.num_elements {
        return;
    }

    let elem = elem_info.read(elem_id as usize);
    let P = unpad_mat(elem_eg.at(elem_id as usize).gradient);
    let hessian = elem_hessian.read(elem_id as usize);
    let vol = elem.vol;

    // Scatter to each vertex.
    let s0 = unpad_vec(elem.S[0]);
    scatter_force(force_atomic, elem.indices[0], -(P * s0) * vol);
    scatter_diag(
        diag_atomic,
        elem.indices[0],
        s_H_s_block(s0, &hessian, s0) * vol,
    );

    let s1 = unpad_vec(elem.S[1]);
    scatter_force(force_atomic, elem.indices[1], -(P * s1) * vol);
    scatter_diag(
        diag_atomic,
        elem.indices[1],
        s_H_s_block(s1, &hessian, s1) * vol,
    );

    let s2 = unpad_vec(elem.S[2]);
    scatter_force(force_atomic, elem.indices[2], -(P * s2) * vol);
    scatter_diag(
        diag_atomic,
        elem.indices[2],
        s_H_s_block(s2, &hessian, s2) * vol,
    );

    #[cfg(feature = "dim3")]
    {
        let s3 = unpad_vec(elem.S[3]);
        scatter_force(force_atomic, elem.indices[3], -(P * s3) * vol);
        scatter_diag(
            diag_atomic,
            elem.indices[3],
            s_H_s_block(s3, &hessian, s3) * vol,
        );
    }
}

/// Per-vertex: assemble total force and diagonal, compute preconditioner,
/// initialize PCG state, and accumulate rTz via workgroup reduction.
///
/// force = m/dt²*(y-x) - m/dt²*α*dt*(x-x_prev) + elastic_force + soft_spring
/// diag = m/dt²*(1+α*dt)*I + elastic_diag + soft_spring_stiffness*I
/// prec = diag^{-1}
/// PCG: x=0, r=force, z=prec*r, p=z
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_assemble_and_pcg_init(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(local_invocation_id)] local_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &FemSimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] vertex_info: &[VertexInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] vertex_state: &[VertexState],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] constraints: &[VertexConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] pcg_vertex: &mut [PcgVertexState],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] force_atomic: &mut [AtomicForce],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] diag_atomic: &mut [AtomicDiag],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] scalar_atomic: &mut [u32],
    #[spirv(workgroup)] shared: &mut [f32; WG_SIZE],
) {
    let v_id = invocation_id.x;
    let lid = local_id.x as usize;
    let mut rtz_contrib = 0.0f32;

    if v_id < params.num_vertices {
        let info = vertex_info.read(v_id as usize);
        let x = unpad_vec(vertex_state.at(v_id as usize).pos);
        let pcg = pcg_vertex.at_mut(v_id as usize);
        let y = unpad_vec(pcg.y);
        let x_prev = unpad_vec(pcg.x_prev);

        let m_dt2 = info.mass_over_dt2;
        let alpha_dt = params.alpha_rayleigh * params.dt;

        // Inertia force and diagonal.
        let mut force = (y - x) * m_dt2 - (x - x_prev) * (m_dt2 * alpha_dt);
        let mut diag_mat = diag(Vector::splat(m_dt2 * (1.0 + alpha_dt)));

        // Soft constraint contribution.
        let c = constraints.read(v_id as usize);
        if c.is_constrained != 0 && c.is_soft != 0 {
            let k = c.stiffness;
            force += (unpad_vec(c.target_pos) - x) * k;
            diag_mat += diag(Vector::splat(k));
        }

        // Add elastic contributions from atomic accumulators.
        force += read_force(force_atomic, v_id);
        diag_mat += read_diag(diag_atomic, v_id);

        // Clear atomic accumulators.
        force_atomic.write(v_id as usize, AtomicForce::default());
        diag_atomic.write(v_id as usize, AtomicDiag::default());

        // Floor projection: prevent downward force for vertices at/below floor.
        // This ensures the PCG search direction doesn't push floor-contact vertices down.
        if x.y <= params.floor_y + 1e-5 && force.y < 0.0 {
            force.y = 0.0;
        }

        // Compute preconditioner = inverse of diagonal block.
        let det = diag_mat.determinant();
        let prec = if abs_f32(det) > 1e-10 {
            diag_mat.inverse()
        } else {
            Matrix::IDENTITY
        };

        // Store force, diag, prec.
        pcg.force = pad_vec(force);
        pcg.diag = pad_mat(diag_mat);
        pcg.prec = pad_mat(prec);

        // PCG init: x=0, r=force, z=M^{-1}*r, p=z.
        pcg.x = pad_vec(Vector::ZERO);
        pcg.r = pad_vec(force);
        let z = prec * force;
        pcg.z = pad_vec(z);
        pcg.p = pad_vec(z);

        rtz_contrib = force.dot(z);
    }

    // Workgroup-level reduction for r·z.
    wg_reduce_add(lid, rtz_contrib, shared, scalar_atomic.at_mut(SLOT_RTZ));
}

// ══════════════════════════════════════════════════════════════════════
//  PCG SOLVER
// ══════════════════════════════════════════════════════════════════════

/// Single-thread: finalize initial rTz from atomic accumulator.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_pcg_reduce_init(
    #[spirv(global_invocation_id)] _invocation_id: khal_std::glamx::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] pcg_scalars: &mut [PcgScalars],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] scalar_atomic: &mut [u32],
) {
    let s = pcg_scalars.at_mut(0);
    s.rTz = f32::from_bits(*scalar_atomic.at(SLOT_RTZ));
    *scalar_atomic.at_mut(SLOT_RTZ) = 0u32;
}

/// Per-element: scatter stiffness contribution to Ap via atomics.
/// Ap_ki += vol * sum_km s_H_s(S[ki], H, S[km]) * p[v_km]
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_pcg_scatter_Ap(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &FemSimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] elem_info: &[ElementInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] elem_hessian: &[ElementHessian],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] pcg_vertex: &[PcgVertexState],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] Ap_atomic: &mut [AtomicForce],
) {
    let elem_id = invocation_id.x;
    if elem_id >= params.num_elements {
        return;
    }

    let elem = elem_info.read(elem_id as usize);
    let hessian = elem_hessian.read(elem_id as usize);
    let vol = elem.vol;

    // Read p for all vertices of this element.
    let p0 = unpad_vec(pcg_vertex.at(elem.indices[0] as usize).p);
    let p1 = unpad_vec(pcg_vertex.at(elem.indices[1] as usize).p);
    let p2 = unpad_vec(pcg_vertex.at(elem.indices[2] as usize).p);
    #[cfg(feature = "dim3")]
    let p3 = unpad_vec(pcg_vertex.at(elem.indices[3] as usize).p);

    let s = [
        unpad_vec(elem.S[0]),
        unpad_vec(elem.S[1]),
        unpad_vec(elem.S[2]),
    ];
    #[cfg(feature = "dim3")]
    let s3 = unpad_vec(elem.S[3]);

    // For each vertex ki, compute Ap contribution from all vertices km.
    #[cfg(feature = "dim2")]
    let p_verts = [p0, p1, p2];
    #[cfg(feature = "dim3")]
    let p_verts = [p0, p1, p2, p3];

    // Vertex 0
    let mut ap0 = s_H_s_block(s[0], &hessian, s[0]) * p0
        + s_H_s_block(s[0], &hessian, s[1]) * p1
        + s_H_s_block(s[0], &hessian, s[2]) * p2;
    #[cfg(feature = "dim3")]
    {
        ap0 += s_H_s_block(s[0], &hessian, s3) * p3;
    }
    scatter_force(Ap_atomic, elem.indices[0], ap0 * vol);

    // Vertex 1
    let mut ap1 = s_H_s_block(s[1], &hessian, s[0]) * p0
        + s_H_s_block(s[1], &hessian, s[1]) * p1
        + s_H_s_block(s[1], &hessian, s[2]) * p2;
    #[cfg(feature = "dim3")]
    {
        ap1 += s_H_s_block(s[1], &hessian, s3) * p3;
    }
    scatter_force(Ap_atomic, elem.indices[1], ap1 * vol);

    // Vertex 2
    let mut ap2 = s_H_s_block(s[2], &hessian, s[0]) * p0
        + s_H_s_block(s[2], &hessian, s[1]) * p1
        + s_H_s_block(s[2], &hessian, s[2]) * p2;
    #[cfg(feature = "dim3")]
    {
        ap2 += s_H_s_block(s[2], &hessian, s3) * p3;
    }
    scatter_force(Ap_atomic, elem.indices[2], ap2 * vol);

    // Vertex 3 (3D only)
    #[cfg(feature = "dim3")]
    {
        let ap3 = s_H_s_block(s3, &hessian, s[0]) * p0
            + s_H_s_block(s3, &hessian, s[1]) * p1
            + s_H_s_block(s3, &hessian, s[2]) * p2
            + s_H_s_block(s3, &hessian, s3) * p3;
        scatter_force(Ap_atomic, elem.indices[3], ap3 * vol);
    }
}

/// Per-vertex: finalize Ap (add inertia + atomics) and accumulate p·Ap
/// via workgroup reduction.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_pcg_finalize_Ap_dot(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(local_invocation_id)] local_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &FemSimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] vertex_info: &[VertexInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] constraints: &[VertexConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] pcg_vertex: &mut [PcgVertexState],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] Ap_atomic: &mut [AtomicForce],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] scalar_atomic: &mut [u32],
    #[spirv(workgroup)] shared: &mut [f32; WG_SIZE],
) {
    let v_id = invocation_id.x;
    let lid = local_id.x as usize;
    let mut ptap_contrib = 0.0f32;

    if v_id < params.num_vertices {
        let info = vertex_info.read(v_id as usize);
        let pcg = pcg_vertex.at_mut(v_id as usize);
        let p = unpad_vec(pcg.p);
        let alpha_dt = params.alpha_rayleigh * params.dt;

        // Inertia contribution to Ap.
        let mut Ap = p * (info.mass_over_dt2 * (1.0 + alpha_dt));

        // Soft constraint contribution.
        let c = constraints.read(v_id as usize);
        if c.is_constrained != 0 && c.is_soft != 0 {
            Ap += p * c.stiffness;
        }

        // Add stiffness from atomic scatter.
        Ap += read_force(Ap_atomic, v_id);
        Ap_atomic.write(v_id as usize, AtomicForce::default());

        pcg.Ap = pad_vec(Ap);

        ptap_contrib = p.dot(Ap);
    }

    // Workgroup-level reduction for p·Ap.
    wg_reduce_add(lid, ptap_contrib, shared, scalar_atomic.at_mut(SLOT_PTAP));
}

/// Single-thread: compute alpha = rTz / pTAp.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_pcg_compute_alpha(
    #[spirv(global_invocation_id)] _invocation_id: khal_std::glamx::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] pcg_scalars: &mut [PcgScalars],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] scalar_atomic: &mut [u32],
) {
    let s = pcg_scalars.at_mut(0);
    let pTAp = f32::from_bits(*scalar_atomic.at(SLOT_PTAP));
    *scalar_atomic.at_mut(SLOT_PTAP) = 0u32;

    s.pTAp = pTAp;
    if abs_f32(pTAp) > 1e-30 {
        s.alpha = s.rTz / pTAp;
    } else {
        s.alpha = 0.0;
    }
}

/// Per-vertex: x += α*p, r -= α*Ap, z = prec*r, accumulate new r·z
/// via workgroup reduction.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_pcg_update_x_r_z(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(local_invocation_id)] local_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &FemSimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] pcg_scalars: &[PcgScalars],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] pcg_vertex: &mut [PcgVertexState],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] scalar_atomic: &mut [u32],
    #[spirv(workgroup)] shared: &mut [f32; WG_SIZE],
) {
    let v_id = invocation_id.x;
    let lid = local_id.x as usize;
    let mut rtz_new_contrib = 0.0f32;

    if v_id < params.num_vertices {
        let alpha = pcg_scalars.at(0).alpha;
        let pcg = pcg_vertex.at_mut(v_id as usize);
        let prec = unpad_mat(pcg.prec);

        let x = unpad_vec(pcg.x) + unpad_vec(pcg.p) * alpha;
        let r = unpad_vec(pcg.r) - unpad_vec(pcg.Ap) * alpha;
        let z = prec * r;

        pcg.x = pad_vec(x);
        pcg.r = pad_vec(r);
        pcg.z = pad_vec(z);

        rtz_new_contrib = r.dot(z);
    }

    // Workgroup-level reduction for new r·z.
    wg_reduce_add(
        lid,
        rtz_new_contrib,
        shared,
        scalar_atomic.at_mut(SLOT_RTZ_NEW),
    );
}

/// Single-thread: compute beta = rTz_new / rTz, update rTz.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_pcg_compute_beta(
    #[spirv(global_invocation_id)] _invocation_id: khal_std::glamx::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] pcg_scalars: &mut [PcgScalars],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] scalar_atomic: &mut [u32],
) {
    let s = pcg_scalars.at_mut(0);
    let rTz_new = f32::from_bits(*scalar_atomic.at(SLOT_RTZ_NEW));
    *scalar_atomic.at_mut(SLOT_RTZ_NEW) = 0u32;

    s.rTz_new = rTz_new;
    if abs_f32(s.rTz) > 1e-30 {
        s.beta = rTz_new / s.rTz;
    } else {
        s.beta = 0.0;
    }
    s.rTz = rTz_new;
}

/// Per-vertex: p = z + β*p.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_pcg_update_p(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &FemSimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] pcg_scalars: &[PcgScalars],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] pcg_vertex: &mut [PcgVertexState],
) {
    let v_id = invocation_id.x;
    if v_id >= params.num_vertices {
        return;
    }

    let beta = pcg_scalars.at(0).beta;
    let pcg = pcg_vertex.at_mut(v_id as usize);
    let z = unpad_vec(pcg.z);
    let p = unpad_vec(pcg.p);
    pcg.p = pad_vec(z + p * beta);
}

// ══════════════════════════════════════════════════════════════════════
//  LINE SEARCH
// ══════════════════════════════════════════════════════════════════════

/// Per-vertex: save position for line search, compute directional derivative m = force · dx,
/// and vertex energy at current position. Uses workgroup reduction for both scalar outputs.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_ls_init(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(local_invocation_id)] local_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &FemSimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] vertex_info: &[VertexInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] vertex_state: &[VertexState],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] constraints: &[VertexConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] pcg_vertex: &[PcgVertexState],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] ls_prev_pos: &mut [PaddedVector],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] scalar_atomic: &mut [u32],
    #[spirv(workgroup)] shared: &mut [f32; WG_SIZE],
) {
    let v_id = invocation_id.x;
    let lid = local_id.x as usize;
    let mut m_contrib = 0.0f32;
    let mut ev_contrib = 0.0f32;

    if v_id < params.num_vertices {
        let x = unpad_vec(vertex_state.at(v_id as usize).pos);
        let pcg = pcg_vertex.at(v_id as usize);
        let dx = unpad_vec(pcg.x);
        let force = unpad_vec(pcg.force);
        let y = unpad_vec(pcg.y);
        let x_prev = unpad_vec(pcg.x_prev);
        let info = vertex_info.read(v_id as usize);

        // Save current position.
        ls_prev_pos.write(v_id as usize, pad_vec(x));

        // Directional derivative: m = force · dx.
        m_contrib = force.dot(dx);

        // Vertex energy: E_v = 0.5 * m/dt² * ||x - y||² + 0.5 * m/dt² * α*dt * ||x - x_prev||²
        let alpha_dt = params.alpha_rayleigh * params.dt;
        let diff_y = x - y;
        let diff_prev = x - x_prev;
        ev_contrib = 0.5 * info.mass_over_dt2 * diff_y.length_squared();
        if alpha_dt > 0.0 {
            ev_contrib += 0.5 * info.mass_over_dt2 * alpha_dt * diff_prev.length_squared();
        }

        // Soft constraint energy.
        let c = constraints.read(v_id as usize);
        if c.is_constrained != 0 && c.is_soft != 0 {
            let d = x - unpad_vec(c.target_pos);
            ev_contrib += 0.5 * c.stiffness * d.length_squared();
        }
    }

    // Two sequential workgroup reductions (shared memory reused after barrier).
    wg_reduce_add(lid, m_contrib, shared, scalar_atomic.at_mut(SLOT_M));
    wg_reduce_add(lid, ev_contrib, shared, scalar_atomic.at_mut(SLOT_ENERGY_V));
}

/// Per-element: compute elastic energy and accumulate via workgroup reduction.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_ls_energy_element(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(local_invocation_id)] local_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &FemSimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] elem_info: &[ElementInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] vertex_state: &[VertexState],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] elem_precomp: &[ElementPrecomputed],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] scalar_atomic: &mut [u32],
    #[spirv(workgroup)] shared: &mut [f32; WG_SIZE],
) {
    let elem_id = invocation_id.x;
    let lid = local_id.x as usize;
    let mut energy_contrib = 0.0f32;

    if elem_id < params.num_elements {
        let elem = elem_info.read(elem_id as usize);
        let F = compute_F(&elem, vertex_state);
        let R = unpad_mat(elem_precomp.at(elem_id as usize).R);
        let energy = compute_energy(F, elem.mu, elem.lam, elem.model, R);
        energy_contrib = elem.vol * energy;
    }

    // Workgroup-level reduction for element energy.
    wg_reduce_add(
        lid,
        energy_contrib,
        shared,
        scalar_atomic.at_mut(SLOT_ENERGY_E),
    );
}

/// Single-thread: finalize line search initialization.
/// Reads accumulated m and energy, sets prev_energy and step_size = 1.0.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_ls_finalize_init(
    #[spirv(global_invocation_id)] _invocation_id: khal_std::glamx::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    ls_scalars: &mut [LinesearchScalars],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] scalar_atomic: &mut [u32],
) {
    let s = ls_scalars.at_mut(0);
    s.m = f32::from_bits(*scalar_atomic.at(SLOT_M));
    s.prev_energy = f32::from_bits(*scalar_atomic.at(SLOT_ENERGY_V))
        + f32::from_bits(*scalar_atomic.at(SLOT_ENERGY_E));
    s.step_size = 1.0;
    s.accepted = 0;

    // Clear accumulators.
    *scalar_atomic.at_mut(SLOT_M) = 0u32;
    *scalar_atomic.at_mut(SLOT_ENERGY_V) = 0u32;
    *scalar_atomic.at_mut(SLOT_ENERGY_E) = 0u32;
}

/// Per-vertex: update position for line search trial.
/// x = ls_prev_pos + step_size * dx. Skip if already accepted.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_ls_update_pos(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &FemSimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] vertex_state: &mut [VertexState],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] constraints: &[VertexConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] pcg_vertex: &[PcgVertexState],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] ls_prev_pos: &[PaddedVector],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] ls_scalars: &[LinesearchScalars],
) {
    let v_id = invocation_id.x;
    if v_id >= params.num_vertices {
        return;
    }

    // Skip if already accepted.
    if ls_scalars.at(0).accepted != 0 {
        return;
    }

    let c = constraints.read(v_id as usize);
    let state = vertex_state.at_mut(v_id as usize);

    // Hard constraints don't move.
    if c.is_constrained != 0 && c.is_soft == 0 {
        return;
    }

    let prev = unpad_vec(ls_prev_pos.read(v_id as usize));
    let dx = unpad_vec(pcg_vertex.at(v_id as usize).x);
    let step = ls_scalars.at(0).step_size;
    let mut new_pos = prev + dx * step;

    // Clamp to floor — prevents element inversion from floor penetration.
    if new_pos.y < params.floor_y {
        new_pos.y = params.floor_y;
    }

    state.pos = pad_vec(new_pos);
}

/// Per-vertex: compute vertex energy at trial position via workgroup reduction.
/// Same energy formula as gpu_ls_init but reads current vertex_state.pos.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_ls_energy_vertex(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(local_invocation_id)] local_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &FemSimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] vertex_info: &[VertexInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] vertex_state: &[VertexState],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] constraints: &[VertexConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] pcg_vertex: &[PcgVertexState],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] scalar_atomic: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] ls_scalars: &[LinesearchScalars],
    #[spirv(workgroup)] shared: &mut [f32; WG_SIZE],
) {
    let v_id = invocation_id.x;
    let lid = local_id.x as usize;
    let mut ev_contrib = 0.0f32;

    // All threads must participate in reduction, but skip work if accepted.
    if v_id < params.num_vertices && ls_scalars.at(0).accepted == 0 {
        let x = unpad_vec(vertex_state.at(v_id as usize).pos);
        let pcg = pcg_vertex.at(v_id as usize);
        let y = unpad_vec(pcg.y);
        let x_prev = unpad_vec(pcg.x_prev);
        let info = vertex_info.read(v_id as usize);

        let alpha_dt = params.alpha_rayleigh * params.dt;
        let diff_y = x - y;
        let diff_prev = x - x_prev;
        ev_contrib = 0.5 * info.mass_over_dt2 * diff_y.length_squared();
        if alpha_dt > 0.0 {
            ev_contrib += 0.5 * info.mass_over_dt2 * alpha_dt * diff_prev.length_squared();
        }

        let c = constraints.read(v_id as usize);
        if c.is_constrained != 0 && c.is_soft != 0 {
            let d = x - unpad_vec(c.target_pos);
            ev_contrib += 0.5 * c.stiffness * d.length_squared();
        }
    }

    // Workgroup-level reduction for vertex energy.
    wg_reduce_add(lid, ev_contrib, shared, scalar_atomic.at_mut(SLOT_ENERGY_V));
}

/// Single-thread: check Armijo condition. If satisfied, set accepted=1.
/// Otherwise reduce step_size by gamma factor.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_ls_check_armijo(
    #[spirv(global_invocation_id)] _invocation_id: khal_std::glamx::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    ls_scalars: &mut [LinesearchScalars],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] scalar_atomic: &mut [u32],
) {
    let s = ls_scalars.at_mut(0);
    if s.accepted != 0 {
        return;
    }

    let energy = f32::from_bits(*scalar_atomic.at(SLOT_ENERGY_V))
        + f32::from_bits(*scalar_atomic.at(SLOT_ENERGY_E));
    s.energy = energy;

    // Clear accumulators for next iteration.
    *scalar_atomic.at_mut(SLOT_ENERGY_V) = 0u32;
    *scalar_atomic.at_mut(SLOT_ENERGY_E) = 0u32;

    // Armijo condition: E(x + s*dx) ≤ E(x) - α * s * (force · dx)
    let ls_alpha = 0.1; // TODO: pass as parameter
    let ls_gamma = 0.5; // TODO: pass as parameter
    if energy <= s.prev_energy - ls_alpha * s.step_size * s.m {
        s.accepted = 1;
    } else {
        s.step_size *= ls_gamma;
    }
}

// ══════════════════════════════════════════════════════════════════════
//  FINALIZATION
// ══════════════════════════════════════════════════════════════════════

/// Per-vertex: compute velocity from position change.
/// v = (x - x_prev) / dt
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_compute_velocity(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &FemSimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] vertex_state: &mut [VertexState],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] pcg_vertex: &[PcgVertexState],
) {
    let v_id = invocation_id.x;
    if v_id >= params.num_vertices {
        return;
    }

    let state = vertex_state.at_mut(v_id as usize);
    let x = unpad_vec(state.pos);
    let x_prev = unpad_vec(pcg_vertex.at(v_id as usize).x_prev);
    let inv_dt = 1.0 / params.dt;

    state.vel = pad_vec((x - x_prev) * inv_dt);
}
