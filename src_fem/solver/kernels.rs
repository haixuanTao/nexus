//! Shader wrapper structs for FEM GPU kernels.
//!
//! Each struct wraps one or more GPU compute kernels compiled from SPIR-V.
//! The `#[derive(Shader)]` macro from khal handles SPIR-V loading.

use crate::fem_shaders::kernels::explicit::{
    GpuApplyForcesGravityDamping, GpuApplyHardConstraints, GpuApplySoftConstraints,
    GpuBoundaryConditions, GpuComputeElasticForces, GpuIntegratePositions,
};
use crate::fem_shaders::kernels::implicit::{
    GpuAssembleAndPcgInit, GpuComputeEg, GpuComputeEgh, GpuComputeVelocity, GpuInitImplicitStep,
    GpuLsCheckArmijo, GpuLsEnergyElement, GpuLsEnergyVertex, GpuLsFinalizeInit, GpuLsInit,
    GpuLsUpdatePos, GpuPcgComputeAlpha, GpuPcgComputeBeta, GpuPcgFinalizeApDot, GpuPcgReduceInit,
    GpuPcgScatterAp, GpuPcgUpdateP, GpuPcgUpdateXRZ, GpuPrecomputeMaterial,
    GpuScatterElasticForceDiag,
};
use khal::Shader;

/// Explicit solver kernels (symplectic Euler).
#[derive(Shader)]
pub struct WgExplicitStep {
    pub(crate) compute_elastic_forces: GpuComputeElasticForces,
    pub(crate) apply_forces_gravity_damping: GpuApplyForcesGravityDamping,
    pub(crate) apply_soft_constraints: GpuApplySoftConstraints,
    pub(crate) integrate_positions: GpuIntegratePositions,
    pub(crate) apply_hard_constraints: GpuApplyHardConstraints,
    pub(crate) boundary_conditions: GpuBoundaryConditions,
}

/// Implicit solver kernels (Newton-PCG + line search).
#[derive(Shader)]
pub struct WgImplicitStep {
    // Newton setup
    pub(crate) init_implicit_step: GpuInitImplicitStep,
    pub(crate) precompute_material: GpuPrecomputeMaterial,
    pub(crate) compute_egh: GpuComputeEgh,
    #[allow(dead_code)]
    pub(crate) compute_eg: GpuComputeEg,
    // Force assembly
    pub(crate) scatter_elastic_force_diag: GpuScatterElasticForceDiag,
    pub(crate) assemble_and_pcg_init: GpuAssembleAndPcgInit,
    // PCG
    pub(crate) pcg_reduce_init: GpuPcgReduceInit,
    pub(crate) pcg_scatter_ap: GpuPcgScatterAp,
    pub(crate) pcg_finalize_ap_dot: GpuPcgFinalizeApDot,
    pub(crate) pcg_compute_alpha: GpuPcgComputeAlpha,
    pub(crate) pcg_update_x_r_z: GpuPcgUpdateXRZ,
    pub(crate) pcg_compute_beta: GpuPcgComputeBeta,
    pub(crate) pcg_update_p: GpuPcgUpdateP,
    // Line search
    pub(crate) ls_init: GpuLsInit,
    pub(crate) ls_energy_element: GpuLsEnergyElement,
    pub(crate) ls_finalize_init: GpuLsFinalizeInit,
    pub(crate) ls_update_pos: GpuLsUpdatePos,
    pub(crate) ls_energy_vertex: GpuLsEnergyVertex,
    pub(crate) ls_check_armijo: GpuLsCheckArmijo,
    // Finalization
    pub(crate) compute_velocity: GpuComputeVelocity,
    // Reuse boundary_conditions from explicit
    pub(crate) boundary_conditions: GpuBoundaryConditions,
}
