use crate::models::{
    DruckerPrager, DruckerPragerPlasticState, DruckerPragerPlasticity, ElasticCoefficients,
    ElasticCoefficientsExt,
};
pub use crate::mpm_shaders::models::default::GpuParticleModel;
use bytemuck::Pod;

/// Material model for MPM particles.
///
/// Defines the constitutive behavior (how stress relates to deformation) for particles.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ParticleModel {
    /// Linear elastic material (St. Venant-Kirchhoff).
    ElasticLinear(ElasticCoefficients),
    /// Neo-Hookean hyperelastic material (better for large deformations).
    ElasticNeoHookean(ElasticCoefficients),
    /// Sand/granular material with linear elasticity and Drucker-Prager plasticity.
    SandLinear(SandModel),
    /// Sand with Neo-Hookean elasticity and Drucker-Prager plasticity.
    SandNeoHookean(SandModel),
}

impl Default for ParticleModel {
    fn default() -> Self {
        Self::elastic(Self::DEFAULT_YOUNG_MODULUS, Self::DEFAULT_POISSON_RATIO)
    }
}

impl ParticleModel {
    /// Default Young's modulus for elastic materials (Pa).
    pub const DEFAULT_YOUNG_MODULUS: f32 = 1_000.0;
    /// Default Poisson's ratio for elastic materials (dimensionless).
    pub const DEFAULT_POISSON_RATIO: f32 = 0.2;

    /// Creates a linear elastic material model.
    pub fn elastic(young_modulus: f32, poisson_ratio: f32) -> Self {
        Self::ElasticLinear(ElasticCoefficients::from_young_modulus(
            young_modulus,
            poisson_ratio,
        ))
    }

    pub fn elastic_neo_hookean(young_modulus: f32, poisson_ratio: f32) -> Self {
        Self::ElasticNeoHookean(ElasticCoefficients::from_young_modulus(
            young_modulus,
            poisson_ratio,
        ))
    }

    /// Creates a sand/granular material model with Drucker-Prager plasticity.
    pub fn sand(young_modulus: f32, poisson_ratio: f32) -> Self {
        ParticleModel::SandLinear(SandModel {
            plastic_state: DruckerPragerPlasticState {
                plastic_deformation_gradient_det: 1.0,
                plastic_hardening: 1.0,
                log_vol_gain: 0.0,
            },
            plastic: DruckerPrager::new(young_modulus, poisson_ratio),
            elastic: ElasticCoefficients::from_young_modulus(young_modulus, poisson_ratio),
        })
    }

    /// Creates a sand/granular material model with Neo-Hookean elasticity.
    pub fn sand_neo_hookean(young_modulus: f32, poisson_ratio: f32) -> Self {
        ParticleModel::SandNeoHookean(SandModel {
            plastic_state: DruckerPragerPlasticState {
                plastic_deformation_gradient_det: 1.0,
                plastic_hardening: 1.0,
                log_vol_gain: 0.0,
            },
            plastic: DruckerPrager::new(young_modulus, poisson_ratio),
            elastic: ElasticCoefficients::from_young_modulus(young_modulus, poisson_ratio),
        })
    }
}

/// Combined elastic-plastic model for sand and granular materials.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct SandModel {
    /// Current plastic deformation state.
    pub plastic_state: DruckerPragerPlasticState,
    /// Drucker-Prager plasticity model parameters.
    pub plastic: DruckerPragerPlasticity,
    /// Elastic coefficients (Lamé parameters).
    pub elastic: ElasticCoefficients,
}

// IMPORTANT: this assertion ensures `GpuParticleModel` (tag + [u32; 12] = 52 bytes)
// matches the GPU-side layout.
static_assertions::assert_eq_size!(GpuParticleModel, [u8; 52]);

impl From<ParticleModel> for GpuParticleModel {
    fn from(val: ParticleModel) -> Self {
        let mut data = [0u32; 12];
        let tag = match val {
            ParticleModel::ElasticLinear(elastic) => {
                let bytes = bytemuck::bytes_of(&elastic);
                bytemuck::cast_slice_mut::<u32, u8>(&mut data)[..bytes.len()]
                    .copy_from_slice(bytes);
                0
            }
            ParticleModel::ElasticNeoHookean(elastic) => {
                let bytes = bytemuck::bytes_of(&elastic);
                bytemuck::cast_slice_mut::<u32, u8>(&mut data)[..bytes.len()]
                    .copy_from_slice(bytes);
                1
            }
            ParticleModel::SandLinear(sand) => {
                let bytes = bytemuck::bytes_of(&sand);
                bytemuck::cast_slice_mut::<u32, u8>(&mut data)[..bytes.len()]
                    .copy_from_slice(bytes);
                2
            }
            ParticleModel::SandNeoHookean(sand) => {
                let bytes = bytemuck::bytes_of(&sand);
                bytemuck::cast_slice_mut::<u32, u8>(&mut data)[..bytes.len()]
                    .copy_from_slice(bytes);
                3
            }
        };
        GpuParticleModel { tag, data }
    }
}

/// Trait for types that can be used as GPU particle model data.
pub trait GpuParticleModelData: Pod + Send + Sync {
    /// CPU-side material model type.
    type Model: Copy;
    /// Converts from CPU representation to GPU representation.
    fn from_model(model: Self::Model) -> Self;
}

impl GpuParticleModelData for GpuParticleModel {
    type Model = ParticleModel;

    fn from_model(model: Self::Model) -> Self {
        model.into()
    }
}
