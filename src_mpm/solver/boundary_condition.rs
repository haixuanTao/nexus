pub use crate::mpm_shaders::solver::boundary_condition::BoundaryCondition;
use khal::BufferUsages;
use khal::backend::{GpuBackend, GpuBackendError};
use vortx::tensor::Tensor;

/// Convenience constructors for `BoundaryCondition`.
pub trait BoundaryConditionExt {
    fn stick() -> Self;
    fn slip() -> Self;
    fn separate(friction: f32) -> Self;
}

impl BoundaryConditionExt for BoundaryCondition {
    fn stick() -> BoundaryCondition {
        BoundaryCondition { ty: 0, friction: 0.0 }
    }

    fn slip() -> BoundaryCondition {
        BoundaryCondition { ty: 1, friction: 0.0 }
    }

    fn separate(friction: f32) -> BoundaryCondition {
        BoundaryCondition { ty: 2, friction }
    }
}

/// GPU buffers for storing boundary conditions per rigid-body.
pub struct GpuMaterials {
    pub materials: Tensor<BoundaryCondition>,
}

impl GpuMaterials {
    /// Creates material buffers.
    ///
    /// Allocates space for up to 16 bodies (CPIC limitation).
    pub fn new(backend: &GpuBackend, materials: &[BoundaryCondition]) -> Result<Self, GpuBackendError> {
        assert!(
            materials.len() <= 16,
            "CPIC only supports up to 16 colliders"
        );
        Ok(Self {
            materials: Tensor::vector(backend, materials, BufferUsages::STORAGE)?,
        })
    }
}
