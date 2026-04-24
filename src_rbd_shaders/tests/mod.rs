//! Tests for collision detection algorithms.
//!
//! These tests run on the CPU and call the shader functions directly.

mod epa;
mod gjk;
mod pfm_pfm;
#[cfg(feature = "dim3")]
mod linalg;
