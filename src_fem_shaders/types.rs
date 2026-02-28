//! GPU-side data structures for the FEM solver.
//!
//! All types use `#[repr(C)]` for GPU buffer compatibility.

use crate::{PaddedMatrix, PaddedVector, DIM, VERTS_PER_ELEM};

/// Per-element static data (immutable after setup).
#[derive(Clone, Copy)]
#[cfg_attr(
    not(target_arch = "spirv"),
    derive(Debug, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct ElementInfo {
    /// Vertex indices (indices[3] unused in 2D).
    pub indices: [u32; 4],
    /// Inverse reference shape matrix.
    pub B_inv: PaddedMatrix,
    /// Element mapping matrix rows: S[k] maps vertex k contribution to F.
    /// S[VERTS_PER_ELEM-1] unused slot in 2D (S[3]).
    pub S: [PaddedVector; 4],
    /// Rest volume (3D) or area (2D).
    pub vol: f32,
    /// First Lamé parameter (shear modulus).
    pub mu: f32,
    /// Second Lamé parameter.
    pub lam: f32,
    /// Material model: 0=Linear, 1=LinearCorotated, 2=StableNeoHookean.
    pub model: u32,
    /// Material density (kg/m³).
    pub rho: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

/// Per-vertex static data.
#[derive(Clone, Copy, Default)]
#[cfg_attr(
    not(target_arch = "spirv"),
    derive(Debug, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct VertexInfo {
    /// Total mass of this vertex.
    pub mass: f32,
    /// Mass divided by dt^2 (precomputed for implicit solver).
    pub mass_over_dt2: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

/// Per-vertex dynamic state (position + velocity).
#[derive(Clone, Copy, Default)]
#[cfg_attr(
    not(target_arch = "spirv"),
    derive(Debug, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct VertexState {
    /// Current position.
    pub pos: PaddedVector,
    /// Current velocity.
    pub vel: PaddedVector,
}

/// Per-vertex constraint data.
#[derive(Clone, Copy, Default)]
#[cfg_attr(
    not(target_arch = "spirv"),
    derive(Debug, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct VertexConstraint {
    /// Target position for the constraint.
    pub target_pos: PaddedVector,
    /// Spring stiffness (soft constraints only).
    pub stiffness: f32,
    /// Whether this vertex is constrained (0 or 1).
    pub is_constrained: u32,
    /// Whether this is a soft constraint (0 or 1). If 0, it's a hard constraint.
    pub is_soft: u32,
    pub _pad: u32,
}

/// Per-element precomputed data (updated each Newton iteration for corotated).
#[derive(Clone, Copy)]
#[cfg_attr(
    not(target_arch = "spirv"),
    derive(Debug, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct ElementPrecomputed {
    /// Rotation matrix from polar decomposition (identity for non-corotated models).
    pub R: PaddedMatrix,
}

impl Default for ElementPrecomputed {
    fn default() -> Self {
        Self {
            #[cfg(feature = "dim2")]
            R: glamx::Mat2::IDENTITY,
            #[cfg(feature = "dim3")]
            R: glamx::Mat4::IDENTITY,
        }
    }
}

/// Per-element energy and gradient output.
#[derive(Clone, Copy, Default)]
#[cfg_attr(
    not(target_arch = "spirv"),
    derive(Debug, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct ElementEnergyGrad {
    /// Energy density Psi(F).
    pub energy: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
    /// Gradient dPsi/dF (first Piola-Kirchhoff stress).
    pub gradient: PaddedMatrix,
}

/// Per-element Hessian stored as DIM*DIM blocks of DIM x DIM matrices.
/// H[a*DIM+b] = d²Psi / (dF_col_a dF_col_b).
/// In 3D: 9 Mat4 blocks (padded Mat3). In 2D: 4 Mat2 blocks.
#[cfg(feature = "dim3")]
#[derive(Clone, Copy, Default)]
#[cfg_attr(
    not(target_arch = "spirv"),
    derive(Debug, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct ElementHessian {
    pub blocks: [PaddedMatrix; 9],
}

#[cfg(feature = "dim2")]
#[derive(Clone, Copy, Default)]
#[cfg_attr(
    not(target_arch = "spirv"),
    derive(Debug, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct ElementHessian {
    pub blocks: [PaddedMatrix; 4],
}

/// Per-vertex PCG solver state.
#[derive(Clone, Copy, Default)]
#[cfg_attr(
    not(target_arch = "spirv"),
    derive(Debug, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct PcgVertexState {
    /// Solution vector (displacement dx).
    pub x: PaddedVector,
    /// Residual vector.
    pub r: PaddedVector,
    /// Preconditioned residual.
    pub z: PaddedVector,
    /// Search direction.
    pub p: PaddedVector,
    /// Matrix-vector product A*p.
    pub Ap: PaddedVector,
    /// Force (negative gradient of total energy).
    pub force: PaddedVector,
    /// Inertia target: y = x_prev + v*dt + g*dt².
    pub y: PaddedVector,
    /// Position at start of substep (for velocity computation and damping).
    pub x_prev: PaddedVector,
    /// Diagonal Hessian block for this vertex.
    pub diag: PaddedMatrix,
    /// Preconditioner (inverse of diag).
    pub prec: PaddedMatrix,
}

/// Global scalar state for PCG solver (single element, read/written atomically).
#[derive(Clone, Copy, Default)]
#[cfg_attr(
    not(target_arch = "spirv"),
    derive(Debug, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct PcgScalars {
    pub rTz: f32,
    pub pTAp: f32,
    pub rTr: f32,
    pub alpha: f32,
    pub beta: f32,
    pub rTz_new: f32,
    pub rTr_new: f32,
    pub _pad: f32,
}

/// Global scalar state for line search.
#[derive(Clone, Copy, Default)]
#[cfg_attr(
    not(target_arch = "spirv"),
    derive(Debug, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct LinesearchScalars {
    pub prev_energy: f32,
    pub energy: f32,
    pub step_size: f32,
    pub m: f32, // directional derivative: -dx^T * force
    pub accepted: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Simulation parameters (uniform buffer).
#[derive(Clone, Copy, Default)]
#[cfg_attr(
    not(target_arch = "spirv"),
    derive(Debug, bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct FemSimParams {
    pub dt: f32,
    pub damping: f32,
    pub alpha_rayleigh: f32,
    pub beta_rayleigh: f32,
    pub gravity: PaddedVector,
    pub floor_y: f32,
    pub num_vertices: u32,
    pub num_elements: u32,
    pub _pad: u32,
}
