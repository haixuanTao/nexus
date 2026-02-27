//! Broad phase collision detection algorithms.
//!
//! This module provides algorithms for quickly pruning non-colliding pairs:
//! - Brute force (O(n^2), for small scenes)
//! - LBVH (Linear Bounding Volume Hierarchy, for large scenes)
//!
//! The module is organized into:
//! - Data structures and algorithms (non-kernel modules)
//! - GPU compute shader kernels (*_kernels modules)

// Data structures and algorithms
mod lbvh;

// GPU compute shader kernels
mod narrow_phase;

// Re-export non-spirv items explicitly to avoid ambiguous glob re-exports.
// The div_ceil functions have different signatures (u32 vs i32) so we pick one.
// Spirv-only items (functions and generated structs) are re-exported via glob.
pub use lbvh::*;
#[cfg(feature = "dim2")]
pub use lbvh::{expand_bits_2d, morton_2d};
pub use narrow_phase::*;
