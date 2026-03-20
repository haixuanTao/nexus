mod cpu;
mod gpu;

pub use cpu::CpuBackend;
pub use gpu::GpuBackend;

use khal::backend::GpuBackend as KhalGpuBackend;
use nexus::rbd::math::Pose;
use nexus::rbd::pipeline::RunStats;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BackendType {
    /// CPU physics using rapier.
    Rapier,
    /// GPU-accelerated physics using nexus + WebGPU.
    Gpu,
    /// CPU physics using nexus (same pipeline as GPU, executed on CPU).
    Cpu,
    /// GPU-accelerated physics using nexus + CUDA.
    #[cfg(feature = "cuda")]
    Cuda,
}

/// Trait for physics simulation backends (CPU or GPU)
pub trait SimulationBackend {
    /// Get the current poses for rendering
    fn poses(&self) -> &[Pose];
    fn num_bodies(&self) -> usize;
    fn num_joints(&self) -> usize;
    fn num_batches(&self) -> usize;

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

    pub fn num_batches(&self) -> usize {
        match self {
            PhysicsBackend::Cpu(backend) => backend.num_batches(),
            PhysicsBackend::Gpu(backend) => backend.num_batches(),
        }
    }
}
