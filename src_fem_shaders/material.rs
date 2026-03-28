//! Constitutive material models for FEM.
//!
//! Three models supported:
//! - **Linear**: Small-strain linear elasticity (ε = ½(F+Fᵀ) - I).
//! - **LinearCorotated**: Corotated linear elasticity with SVD-based polar decomposition.
//! - **StableNeoHookean**: Regularized Neo-Hookean (Smith et al. 2018).
//!
//! All functions are dimension-generic via cfg(dim2/dim3).

use crate::types::ElementPrecomputed;
use crate::{
    DIM, MODEL_LINEAR, MODEL_LINEAR_COROTATED, MODEL_STABLE_NEOHOOKEAN, Matrix, PaddedMatrix,
    Vector, cofactor, diag, frobenius_norm_sq, outer, pad_mat, trace, unpad_mat,
};
use glamx::*;

/// Returns true if the Hessian is constant across Newton iterations for this model.
/// Linear elasticity has a constant Hessian (no need to recompute each Newton step).
#[inline]
pub fn hessian_is_constant(model: u32) -> bool {
    model == MODEL_LINEAR
}

/// Precompute per-element data from the deformation gradient.
/// For LinearCorotated: extracts rotation R via SVD polar decomposition.
/// For others: R = identity.
#[inline]
pub fn precompute(F: Matrix, model: u32) -> ElementPrecomputed {
    if model == MODEL_LINEAR_COROTATED {
        let svd = F.svd();
        let mut u = svd.u;
        let r = u * svd.vt;
        // Ensure proper rotation: det(R) > 0
        if r.determinant() < 0.0 {
            // Flip the column corresponding to the smallest singular value (last column).
            #[cfg(feature = "dim2")]
            {
                u = Matrix::from_cols(u.col(0), -u.col(1));
            }
            #[cfg(feature = "dim3")]
            {
                u = Matrix::from_cols(u.col(0), u.col(1), -u.col(2));
            }
        }
        ElementPrecomputed {
            R: pad_mat(u * svd.vt),
        }
    } else {
        ElementPrecomputed::default()
    }
}

// ── Energy ──

/// Compute energy density Ψ(F) for the given material model.
#[inline]
pub fn compute_energy(F: Matrix, mu: f32, lam: f32, model: u32, R: Matrix) -> f32 {
    if model == MODEL_LINEAR {
        let eps = (F + F.transpose()) * 0.5 - Matrix::IDENTITY;
        let tr_eps = trace(eps);
        mu * frobenius_norm_sq(eps) + 0.5 * lam * tr_eps * tr_eps
    } else if model == MODEL_LINEAR_COROTATED {
        let F_hat = R.transpose() * F;
        let eps = (F_hat + F_hat.transpose()) * 0.5 - Matrix::IDENTITY;
        let tr_eps = trace(eps);
        mu * frobenius_norm_sq(eps) + 0.5 * lam * tr_eps * tr_eps
    } else {
        // StableNeoHookean
        let IC = frobenius_norm_sq(F);
        let J = F.determinant();
        let lambda = lam + mu;
        let alpha = 1.0 + mu / lambda;
        let J_minus_alpha = J - alpha;
        0.5 * (mu * (IC - DIM as f32) + lambda * J_minus_alpha * J_minus_alpha)
    }
}

// ── Hessian block computation ──

/// Compute Hessian blocks for the given material model.
/// Returns padded matrices to avoid SPIR-V struct alignment issues with Mat3.
///
/// hessian_blocks[a*DIM+b] = d²Ψ/(dF_col_a dF_col_b).
/// For StableNeoHookean, returns zero blocks (Hessian not implemented).
#[inline]
pub fn compute_hessian_blocks(
    mu: f32,
    lam: f32,
    model: u32,
    R: Matrix,
) -> [PaddedMatrix; DIM * DIM] {
    if model == MODEL_LINEAR {
        compute_linear_hessian_blocks(mu, lam)
    } else if model == MODEL_LINEAR_COROTATED {
        compute_corotated_hessian_blocks(mu, lam, R)
    } else {
        [PaddedMatrix::ZERO; DIM * DIM]
    }
}

/// Hessian blocks for Linear elasticity (returns padded matrices).
/// H[a*DIM+b] = μ δ_{ab} I + μ outer(e_b, e_a) + λ outer(e_a, e_b)
#[inline]
fn compute_linear_hessian_blocks(mu: f32, lam: f32) -> [PaddedMatrix; DIM * DIM] {
    let mut blocks = [PaddedMatrix::ZERO; DIM * DIM];

    #[cfg(feature = "dim2")]
    {
        let e = [Vec2::X, Vec2::Y];
        blocks[0] =
            pad_mat(diag(Vector::splat(mu)) + outer(e[0], e[0]) * mu + outer(e[0], e[0]) * lam);
        blocks[1] = pad_mat(outer(e[0], e[1]) * mu + outer(e[1], e[0]) * lam);
        blocks[2] = pad_mat(outer(e[1], e[0]) * mu + outer(e[0], e[1]) * lam);
        blocks[3] =
            pad_mat(diag(Vector::splat(mu)) + outer(e[1], e[1]) * mu + outer(e[1], e[1]) * lam);
    }

    #[cfg(feature = "dim3")]
    {
        let e = [Vec3::X, Vec3::Y, Vec3::Z];
        // a=0
        blocks[0] =
            pad_mat(diag(Vector::splat(mu)) + outer(e[0], e[0]) * mu + outer(e[0], e[0]) * lam);
        blocks[1] = pad_mat(outer(e[0], e[1]) * mu + outer(e[1], e[0]) * lam);
        blocks[2] = pad_mat(outer(e[0], e[2]) * mu + outer(e[2], e[0]) * lam);
        // a=1
        blocks[3] = pad_mat(outer(e[1], e[0]) * mu + outer(e[0], e[1]) * lam);
        blocks[4] =
            pad_mat(diag(Vector::splat(mu)) + outer(e[1], e[1]) * mu + outer(e[1], e[1]) * lam);
        blocks[5] = pad_mat(outer(e[1], e[2]) * mu + outer(e[2], e[1]) * lam);
        // a=2
        blocks[6] = pad_mat(outer(e[2], e[0]) * mu + outer(e[0], e[2]) * lam);
        blocks[7] = pad_mat(outer(e[2], e[1]) * mu + outer(e[1], e[2]) * lam);
        blocks[8] =
            pad_mat(diag(Vector::splat(mu)) + outer(e[2], e[2]) * mu + outer(e[2], e[2]) * lam);
    }

    blocks
}

/// Hessian blocks for LinearCorotated elasticity (returns padded matrices).
/// H[a*DIM+b] = μ δ_{ab} I + μ outer(R[:,b], R[:,a]) + λ outer(R[:,a], R[:,b])
#[inline]
fn compute_corotated_hessian_blocks(mu: f32, lam: f32, R: Matrix) -> [PaddedMatrix; DIM * DIM] {
    let mut blocks = [PaddedMatrix::ZERO; DIM * DIM];

    #[cfg(feature = "dim2")]
    {
        let r0 = R.col(0);
        let r1 = R.col(1);
        blocks[0] = pad_mat(diag(Vector::splat(mu)) + outer(r0, r0) * mu + outer(r0, r0) * lam);
        blocks[1] = pad_mat(outer(r0, r1) * mu + outer(r1, r0) * lam);
        blocks[2] = pad_mat(outer(r1, r0) * mu + outer(r0, r1) * lam);
        blocks[3] = pad_mat(diag(Vector::splat(mu)) + outer(r1, r1) * mu + outer(r1, r1) * lam);
    }

    #[cfg(feature = "dim3")]
    {
        let r0 = R.col(0);
        let r1 = R.col(1);
        let r2 = R.col(2);
        // a=0
        blocks[0] = pad_mat(diag(Vector::splat(mu)) + outer(r0, r0) * mu + outer(r0, r0) * lam);
        blocks[1] = pad_mat(outer(r0, r1) * mu + outer(r1, r0) * lam);
        blocks[2] = pad_mat(outer(r0, r2) * mu + outer(r2, r0) * lam);
        // a=1
        blocks[3] = pad_mat(outer(r1, r0) * mu + outer(r0, r1) * lam);
        blocks[4] = pad_mat(diag(Vector::splat(mu)) + outer(r1, r1) * mu + outer(r1, r1) * lam);
        blocks[5] = pad_mat(outer(r1, r2) * mu + outer(r2, r1) * lam);
        // a=2
        blocks[6] = pad_mat(outer(r2, r0) * mu + outer(r0, r2) * lam);
        blocks[7] = pad_mat(outer(r2, r1) * mu + outer(r1, r2) * lam);
        blocks[8] = pad_mat(diag(Vector::splat(mu)) + outer(r2, r2) * mu + outer(r2, r2) * lam);
    }

    blocks
}

/// Compute stress P = dΨ/dF for the explicit solver (gradient only, no energy).
#[inline]
pub fn compute_stress(F: Matrix, mu: f32, lam: f32, model: u32, R: Matrix) -> Matrix {
    if model == MODEL_LINEAR {
        let I = Matrix::IDENTITY;
        (F + F.transpose() - I * 2.0) * mu
            + diag(Vector::splat(lam * trace((F + F.transpose()) * 0.5 - I)))
    } else if model == MODEL_LINEAR_COROTATED {
        let F_hat = R.transpose() * F;
        let eps = (F_hat + F_hat.transpose()) * 0.5 - Matrix::IDENTITY;
        let tr_eps = trace(eps);
        R * eps * (2.0 * mu) + R * (lam * tr_eps)
    } else {
        // StableNeoHookean
        let IC = frobenius_norm_sq(F);
        let J = F.determinant();
        let alpha = 1.0 + 0.75 * mu / lam;
        let dJdF = cofactor(F);
        F * (mu * (1.0 - 1.0 / (IC + 1.0))) + dJdF * (lam * (J - alpha))
    }
}
