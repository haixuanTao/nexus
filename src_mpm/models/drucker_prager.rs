use crate::models::lame_lambda_mu;
use crate::mpm_shaders::models::drucker_prager::DruckerPragerPlasticity;

/// CPU-side convenience wrapper for constructing `DruckerPragerPlasticity`.
pub struct DruckerPrager;

impl DruckerPrager {
    /// Creates a Drucker-Prager model with default sand parameters.
    pub fn new(young_modulus: f32, poisson_ratio: f32) -> DruckerPragerPlasticity {
        let (lambda, mu) = if young_modulus > 0.0 {
            lame_lambda_mu(young_modulus, poisson_ratio)
        } else {
            (-1.0, -1.0)
        };

        Self::from_lame(lambda, mu)
    }

    /// Creates a Drucker-Prager model from Lamé parameters with default plasticity settings.
    pub fn from_lame(lambda: f32, mu: f32) -> DruckerPragerPlasticity {
        DruckerPragerPlasticity {
            ha: 35.0f32.to_radians(),
            hb: 9.0f32.to_radians(),
            hc: 0.2,
            hd: 10.0f32.to_radians(),
            lambda,
            mu,
        }
    }
}
