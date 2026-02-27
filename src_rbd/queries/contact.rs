//! Contact generation for collision response.
//!
//! This module provides GPU-accelerated contact manifold generation between pairs of
//! colliding shapes. Contact manifolds contain multiple contact points with normals
//! and penetration depths, which are essential for physics simulation and collision response.
//!
//! # Contact Manifolds
//!
//! A contact manifold represents a collision between two shapes and contains:
//! - **Contact points**: Up to 2 points in 2D, 4 points in 3D.
//! - **Contact normal**: The direction to separate the shapes.
//! - **Penetration depths**: How deep each contact point penetrates.

// Re-export contact types from the shader crate with Gpu prefix for backward compatibility
pub use crate::shaders::queries::{
    ContactManifold as GpuContactManifold, ContactPoint as GpuContactPoint,
    IndexedManifold as GpuIndexedContact,
};
