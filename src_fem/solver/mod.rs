//! FEM solver types and shader wrappers.

pub mod kernels;

use crate::Vector;

/// Solver method selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverMethod {
    Explicit,
    Implicit,
}

/// Material model selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterialModel {
    Linear,
    LinearCorotated,
    StableNeoHookean,
}

impl MaterialModel {
    pub fn to_gpu_id(self) -> u32 {
        match self {
            MaterialModel::Linear => 0,
            MaterialModel::LinearCorotated => 1,
            MaterialModel::StableNeoHookean => 2,
        }
    }
}

/// FEM simulation configuration.
#[derive(Clone)]
pub struct FemConfig {
    pub dt: f32,
    pub substeps: u32,
    pub gravity: Vector,
    pub floor_y: f32,
    pub damping: f32,
    pub alpha_rayleigh: f32,
    pub beta_rayleigh: f32,
    pub method: SolverMethod,
    pub newton_iters: u32,
    pub pcg_iters: u32,
    pub pcg_tol: f32,
    pub ls_max_iters: u32,
    pub ls_alpha: f32,
    pub ls_gamma: f32,
}

impl Default for FemConfig {
    fn default() -> Self {
        Self {
            dt: 5e-3,
            substeps: 10,
            #[cfg(feature = "dim2")]
            gravity: glamx::Vec2::new(0.0, -9.81),
            #[cfg(feature = "dim3")]
            gravity: glamx::Vec3::new(0.0, -9.81, 0.0),
            floor_y: 0.0,
            damping: 5.0,
            alpha_rayleigh: 0.0,
            beta_rayleigh: 0.0,
            method: SolverMethod::Explicit,
            newton_iters: 5,
            pcg_iters: 20,
            pcg_tol: 1e-3,
            ls_max_iters: 5,
            ls_alpha: 0.1,
            ls_gamma: 0.5,
        }
    }
}

/// Material properties for mesh creation.
#[derive(Clone)]
pub struct FemMaterial {
    pub youngs_modulus: f32,
    pub poissons_ratio: f32,
    pub density: f32,
    pub model: MaterialModel,
}

impl FemMaterial {
    /// First Lamé parameter μ (shear modulus).
    pub fn mu(&self) -> f32 {
        let E = self.youngs_modulus;
        let nu = self.poissons_ratio;
        E / (2.0 * (1.0 + nu))
    }

    /// Second Lamé parameter λ.
    pub fn lambda(&self) -> f32 {
        let E = self.youngs_modulus;
        let nu = self.poissons_ratio;
        E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    }
}

/// Per-vertex constraint (host-side).
#[derive(Clone, Default)]
pub struct FemConstraint {
    pub is_constrained: bool,
    pub is_soft: bool,
    pub target_pos: Vector,
    pub stiffness: f32,
}
