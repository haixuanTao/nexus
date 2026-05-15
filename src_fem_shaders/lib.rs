//! Nexus FEM (Finite Element Method) GPU shaders.
//!
//! This crate contains Rust GPU shaders for the nexus_fem solver,
//! providing GPU-accelerated FEM simulation with explicit and implicit solvers.

#![cfg_attr(target_arch_is_gpu, no_std)]
#![cfg_attr(target_arch = "spirv", feature(asm_experimental_arch))]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::manual_range_contains)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(non_snake_case)]

pub use glamx;
use glamx::*;

// ── Dimension-conditional type aliases ──

#[cfg(feature = "dim2")]
pub type Vector = Vec2;
#[cfg(feature = "dim3")]
pub type Vector = Vec3;

#[cfg(feature = "dim2")]
pub type Matrix = Mat2;
#[cfg(feature = "dim3")]
pub type Matrix = Mat3;

#[cfg(feature = "dim2")]
pub const DIM: usize = 2;
#[cfg(feature = "dim3")]
pub const DIM: usize = 3;

/// Number of vertices per element: 3 (triangle) in 2D, 4 (tetrahedron) in 3D.
#[cfg(feature = "dim2")]
pub const VERTS_PER_ELEM: usize = 3;
#[cfg(feature = "dim3")]
pub const VERTS_PER_ELEM: usize = 4;

// ── Padded types for GPU buffer alignment ──
// In 3D, Vec3 (12 bytes) needs padding to 16 bytes, Mat3 stored as Mat4.
// In 2D, no padding needed.

#[cfg(feature = "dim2")]
pub type PaddedVector = Vec2;
#[cfg(feature = "dim3")]
pub type PaddedVector = Vec4;

#[cfg(feature = "dim2")]
pub type PaddedMatrix = Mat2;
#[cfg(feature = "dim3")]
pub type PaddedMatrix = Mat4;

// ── Padding conversion helpers ──

#[cfg(feature = "dim2")]
#[inline]
pub fn pad_vec(v: Vector) -> PaddedVector {
    v
}

#[cfg(feature = "dim3")]
#[inline]
pub fn pad_vec(v: Vector) -> PaddedVector {
    Vec4::new(v.x, v.y, v.z, 0.0)
}

#[cfg(feature = "dim2")]
#[inline]
pub fn unpad_vec(v: PaddedVector) -> Vector {
    v
}

#[cfg(feature = "dim3")]
#[inline]
pub fn unpad_vec(v: PaddedVector) -> Vector {
    v.xyz()
}

#[cfg(feature = "dim2")]
#[inline]
pub fn pad_mat(m: Matrix) -> PaddedMatrix {
    m
}

#[cfg(feature = "dim3")]
#[inline]
pub fn pad_mat(m: Matrix) -> PaddedMatrix {
    Mat4::from_mat3(m)
}

#[cfg(feature = "dim2")]
#[inline]
pub fn unpad_mat(m: PaddedMatrix) -> Matrix {
    m
}

#[cfg(feature = "dim3")]
#[inline]
pub fn unpad_mat(m: PaddedMatrix) -> Matrix {
    Mat3::from_cols(m.x_axis.xyz(), m.y_axis.xyz(), m.z_axis.xyz())
}

// ── Math helpers ──

#[cfg(feature = "dim2")]
#[inline]
pub fn trace(m: Matrix) -> f32 {
    m.x_axis.x + m.y_axis.y
}

#[cfg(feature = "dim3")]
#[inline]
pub fn trace(m: Matrix) -> f32 {
    m.x_axis.x + m.y_axis.y + m.z_axis.z
}

#[cfg(feature = "dim2")]
#[inline]
pub fn diag(v: Vector) -> Matrix {
    Mat2::from_cols(Vec2::new(v.x, 0.0), Vec2::new(0.0, v.y))
}

#[cfg(feature = "dim3")]
#[inline]
pub fn diag(v: Vector) -> Matrix {
    Mat3::from_cols(
        Vec3::new(v.x, 0.0, 0.0),
        Vec3::new(0.0, v.y, 0.0),
        Vec3::new(0.0, 0.0, v.z),
    )
}

#[inline]
pub fn frobenius_norm_sq(m: Matrix) -> f32 {
    #[cfg(feature = "dim2")]
    {
        m.x_axis.length_squared() + m.y_axis.length_squared()
    }
    #[cfg(feature = "dim3")]
    {
        m.x_axis.length_squared() + m.y_axis.length_squared() + m.z_axis.length_squared()
    }
}

/// Outer product: a * b^T
#[inline]
pub fn outer(a: Vector, b: Vector) -> Matrix {
    #[cfg(feature = "dim2")]
    {
        Mat2::from_cols(a * b.x, a * b.y)
    }
    #[cfg(feature = "dim3")]
    {
        Mat3::from_cols(a * b.x, a * b.y, a * b.z)
    }
}

/// Cofactor matrix of F (dJ/dF).
/// In 2D: adjugate matrix. In 3D: cross products of column pairs.
#[cfg(feature = "dim2")]
#[inline]
pub fn cofactor(F: Matrix) -> Matrix {
    Mat2::from_cols(
        Vec2::new(F.y_axis.y, -F.y_axis.x),
        Vec2::new(-F.x_axis.y, F.x_axis.x),
    )
}

#[cfg(feature = "dim3")]
#[inline]
pub fn cofactor(F: Matrix) -> Matrix {
    let c0 = F.y_axis.cross(F.z_axis);
    let c1 = F.z_axis.cross(F.x_axis);
    let c2 = F.x_axis.cross(F.y_axis);
    Mat3::from_cols(c0, c1, c2)
}

#[inline]
pub fn abs_f32(x: f32) -> f32 {
    if x < 0.0 { -x } else { x }
}

#[inline]
pub fn exp_f32(x: f32) -> f32 {
    use khal_std::num_traits::Float;
    x.exp()
}

#[inline]
pub fn sqrt_f32(x: f32) -> f32 {
    use khal_std::num_traits::Float;
    x.sqrt()
}

// ── Material model constants ──

pub const MODEL_LINEAR: u32 = 0;
pub const MODEL_LINEAR_COROTATED: u32 = 1;
pub const MODEL_STABLE_NEOHOOKEAN: u32 = 2;

// ── Modules ──

pub mod kernels;
pub mod material;
pub mod types;
