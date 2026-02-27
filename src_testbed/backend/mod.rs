mod cpu;
mod gpu;

pub use cpu::CpuBackend;
pub use gpu::GpuBackend;

#[cfg(feature = "dim2")]
use nexus_rbd2d as nexus_rbd;
#[cfg(feature = "dim3")]
use nexus_rbd3d as nexus_rbd;

use khal::backend::GpuBackend as KhalGpuBackend;
use nexus_rbd::math::Pose;
use nexus_rbd::pipeline::RunStats;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BackendType {
    Cpu,
    Gpu,
}

/// Trait for physics simulation backends (CPU or GPU)
pub trait SimulationBackend {
    /// Get the current poses for rendering
    fn poses(&self) -> &[Pose];
    fn num_bodies(&self) -> usize;
    fn num_joints(&self) -> usize;

    /// Step the simulation
    async fn step(&mut self, gpu: Option<&KhalGpuBackend>) -> RunStats;
}

#[allow(clippy::large_enum_variant)]
pub enum PhysicsBackend {
    Cpu(CpuBackend),
    Gpu(GpuBackend),
}

impl PhysicsBackend {
    pub async fn step(&mut self, gpu: Option<&KhalGpuBackend>) -> RunStats {
        match self {
            PhysicsBackend::Cpu(backend) => backend.step(gpu).await,
            PhysicsBackend::Gpu(backend) => backend.step(gpu).await,
        }
    }

    pub fn poses(&self) -> &[Pose] {
        match self {
            PhysicsBackend::Cpu(backend) => backend.poses(),
            PhysicsBackend::Gpu(backend) => backend.poses(),
        }
    }

    pub fn num_bodies(&self) -> usize {
        match self {
            PhysicsBackend::Cpu(backend) => backend.num_bodies(),
            PhysicsBackend::Gpu(backend) => backend.num_bodies(),
        }
    }

    pub fn num_joints(&self) -> usize {
        match self {
            PhysicsBackend::Cpu(backend) => backend.num_joints(),
            PhysicsBackend::Gpu(backend) => backend.num_joints(),
        }
    }
}
