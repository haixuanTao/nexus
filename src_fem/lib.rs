//! Nexus FEM (Finite Element Method) solver.
//!
//! GPU-accelerated FEM simulation with explicit and implicit (Newton-PCG) solvers.
//! Supports 2D (triangles) and 3D (tetrahedra).

#![allow(non_snake_case)]

#[cfg(feature = "dim2")]
pub use nexus_fem_shaders2d as fem_shaders;
#[cfg(feature = "dim3")]
pub use nexus_fem_shaders3d as fem_shaders;

pub use fem_shaders::glamx;
pub use fem_shaders::types;
pub use fem_shaders::{DIM, Matrix, PaddedMatrix, PaddedVector, VERTS_PER_ELEM, Vector};

use khal::re_exports::include_dir::{Dir, include_dir};
pub static SPIRV_DIR: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/shaders-spirv");

/// Cast a Tensor<T> reference to Tensor<U> via bytemuck.
#[allow(dead_code)]
pub(crate) fn cast_tensor<T: bytemuck::Pod + Send + Sync, U: bytemuck::Pod + Send + Sync>(
    tensor: &vortx::tensor::Tensor<T>,
) -> &vortx::tensor::Tensor<U> {
    assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<U>());
    unsafe { &*(tensor as *const vortx::tensor::Tensor<T> as *const vortx::tensor::Tensor<U>) }
}

/// Cast a Tensor<T> mutable reference to Tensor<U> via bytemuck.
#[allow(dead_code)]
pub(crate) fn cast_tensor_mut<T: bytemuck::Pod + Send + Sync, U: bytemuck::Pod + Send + Sync>(
    tensor: &mut vortx::tensor::Tensor<T>,
) -> &mut vortx::tensor::Tensor<U> {
    assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<U>());
    unsafe { &mut *(tensor as *mut vortx::tensor::Tensor<T> as *mut vortx::tensor::Tensor<U>) }
}

pub mod mesh;
pub mod pipeline;
pub mod solver;
