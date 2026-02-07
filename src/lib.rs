//! GPU-accelerated rigid-body physics engine built on khal/vortx.
//!
//! **nexus** provides a high-performance physics simulation system that runs entirely on the GPU,
//! enabling massively parallel physics computation for thousands of rigid bodies. It is designed to
//! work seamlessly across platforms including web and desktop.
//!
//! # See Also
//!
//! - [`khal`]: GPU compute framework providing shader loading and dispatch.
//! - [`vortx`]: GPU tensor/buffer management.
//! - [`glamx`]: Linear algebra types (Vec2, Vec3, Pose2, Pose3, etc.).

#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
#![allow(clippy::result_large_err)]
#![allow(clippy::too_many_arguments)]

// Dimension-specific rapier alias
#[cfg(all(feature = "from_rapier", feature = "dim2"))]
use rapier2d as rapier;
#[cfg(all(feature = "from_rapier", feature = "dim3"))]
use rapier3d as rapier;

// Re-export the shader crate
#[cfg(feature = "dim2")]
pub use nexus_shaders2d as shaders;
#[cfg(feature = "dim3")]
pub use nexus_shaders3d as shaders;

use khal::re_exports::include_dir::{Dir, include_dir};

/// Embedded SPIR-V shader directory.
pub static SPIRV_DIR: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/shaders-spirv");

// Re-export commonly used types from shader crate
pub use shaders::bounding_volumes::Aabb;
pub use shaders::dynamics::{Force, Impulse, LocalMassProperties, Velocity, WorldMassProperties};
pub use shaders::shapes::Shape;
pub use shaders::{Pad, VectorWithPadding};

// Re-export glamx for users
pub use glamx;

const VALIDATE_LBVH_TOPOLOGY: bool = false;

pub mod dynamics;
pub mod pipeline;

/// Broad-phase collision detection algorithms implemented on the GPU.
///
/// Includes brute-force and LBVH (Linear Bounding Volume Hierarchy) implementations.
pub mod broad_phase;
/// Geometric query operations like ray-casting, point projection, and contact generation.
pub mod queries;
/// Geometric shape definitions and their GPU shader implementations.
pub mod shapes;
/// Utility functions and data structures, including GPU radix sort.
pub mod utils;

#[cfg(feature = "dim3")]
pub mod math {
    //! Compilation flags dependent aliases for mathematical types.
    //!
    //! Math type aliases for 3D builds.
    //!
    //! This module provides dimension-specific type aliases using glamx types.

    pub use glamx::{Mat3, Pose3, Quat, Vec3, Vec4};

    /// The default tolerance used for geometric operations.
    pub const DEFAULT_EPSILON: f32 = f32::EPSILON;

    /// The dimension of the space.
    pub const DIM: usize = 3;

    /// The dimension of the space multiplied by two.
    pub const TWO_DIM: usize = DIM * 2;

    /// The vector type.
    pub type Vector = Vec3;

    /// The angular vector type.
    pub type AngVector = Vec3;

    /// The point type (same as Vector in glamx).
    pub type Point = Vec3;

    /// The matrix type.
    pub type Matrix = Mat3;

    /// The rotation matrix type.
    pub type Rotation = Quat;

    /// The transformation type.
    pub type Pose = Pose3;

    /// The angular inertia of a rigid body.
    pub type AngularInertia = Mat3;

    /// The principal angular inertia of a rigid body.
    pub type PrincipalAngularInertia = Vec3;
}

#[cfg(feature = "dim2")]
pub mod math {
    //! Math type aliases for 2D builds.
    //!
    //! This module provides dimension-specific type aliases using glamx types.
    //!
    //! Compilation flags dependent aliases for mathematical types.

    pub use glamx::{Mat2, Pose2, Rot2, Vec2, Vec3};

    /// The default tolerance used for geometric operations.
    pub const DEFAULT_EPSILON: f32 = f32::EPSILON;

    /// The dimension of the space.
    pub const DIM: usize = 2;

    /// The dimension of the space multiplied by two.
    pub const TWO_DIM: usize = DIM * 2;

    /// The vector type.
    pub type Vector = Vec2;

    /// The angular vector type (scalar in 2D).
    pub type AngVector = f32;

    /// The point type (same as Vector in glamx).
    pub type Point = Vec2;

    /// The matrix type.
    pub type Matrix = Mat2;

    /// The rotation type.
    pub type Rotation = Rot2;

    /// The transformation type.
    pub type Pose = Pose2;

    /// The angular inertia of a rigid body (scalar in 2D).
    pub type AngularInertia = f32;

    /// The principal angular inertia of a rigid body (scalar in 2D).
    pub type PrincipalAngularInertia = f32;
}
