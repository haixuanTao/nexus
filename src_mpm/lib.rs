//! Nexus MPM: GPU-accelerated Material Point Method (MPM) physics simulation.
//!
//! Provides a high-performance implementation of the Material Point Method for
//! simulating materials like fluids, sand, snow, and elastic solids. The simulation
//! runs entirely on the GPU using compute shaders, achieving real-time performance
//! for large particle systems.
//!
//! # Overview
//!
//! The MPM algorithm works by transferring data between particles (Lagrangian representation)
//! and a background grid (Eulerian representation):
//! 1. **P2G (Particle-to-Grid)**: Transfer particle mass and momentum to grid nodes
//! 2. **Grid Update**: Solve momentum equations on the grid
//! 3. **G2P (Grid-to-Particle)**: Transfer velocities back to particles and update positions
//!
//! Also supports two-way coupling with rigid bodies.
//!
//! # Features
//!
//! - `dim2`: Enable 2D simulation mode (mutually exclusive with `dim3`)
//! - `dim3`: Enable 3D simulation mode (mutually exclusive with `dim2`)

#![allow(clippy::too_many_arguments)]
#![allow(clippy::module_inception)]
#![allow(missing_docs)]

#[cfg(feature = "dim2")]
pub use nexus_mpm_shaders2d as mpm_shaders;
#[cfg(feature = "dim3")]
pub use nexus_mpm_shaders3d as mpm_shaders;

#[cfg(feature = "dim2")]
pub extern crate nexus_rbd2d as nexus_rbd;
#[cfg(feature = "dim3")]
pub extern crate nexus_rbd3d as nexus_rbd;

#[cfg(all(feature = "from_rapier", feature = "dim2"))]
pub extern crate rapier2d as rapier;
#[cfg(all(feature = "from_rapier", feature = "dim3"))]
pub extern crate rapier3d as rapier;

use khal::re_exports::include_dir::{Dir, include_dir};

/// Embedded SPIR-V shader directory.
pub static SPIRV_DIR: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/shaders-spirv");

pub mod grid;
pub mod models;
pub mod pipeline;
#[cfg(feature = "from_rapier")]
pub(crate) mod sampling;
pub mod solver;
#[cfg(all(feature = "from_rapier", feature = "dim3"))]
pub mod trimesh;

/// Reinterprets a `&Tensor<T>` as `&Tensor<U>` when `T` and `U` have the same size.
///
/// # Safety
/// This is safe when both `T` and `U` are `Pod` and have the same size, meaning
/// the underlying GPU buffer has an identical memory layout regardless of type.
pub(crate) fn cast_tensor<T: bytemuck::Pod + Send + Sync, U: bytemuck::Pod + Send + Sync>(
    tensor: &vortx::tensor::Tensor<T>,
) -> &vortx::tensor::Tensor<U> {
    assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<U>());
    unsafe { &*(tensor as *const vortx::tensor::Tensor<T> as *const vortx::tensor::Tensor<U>) }
}

/// Reinterprets a `&mut Tensor<T>` as `&mut Tensor<U>` when `T` and `U` have the same size.
pub(crate) fn cast_tensor_mut<T: bytemuck::Pod + Send + Sync, U: bytemuck::Pod + Send + Sync>(
    tensor: &mut vortx::tensor::Tensor<T>,
) -> &mut vortx::tensor::Tensor<U> {
    assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<U>());
    unsafe { &mut *(tensor as *mut vortx::tensor::Tensor<T> as *mut vortx::tensor::Tensor<U>) }
}
