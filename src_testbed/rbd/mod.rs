pub mod backend;
pub mod graphics;

pub use backend::{BackendType, CpuBackend, GpuBackend, PhysicsBackend};
pub use graphics::{RenderContext, setup_graphics, update_instances};

use khal::backend::GpuBackend as KhalGpuBackend;
use nexus::rbd::pipeline::GpuPhysicsPipeline;
use rapier::geometry::ColliderSet;
use rapier::prelude::{ImpulseJointSet, RigidBodySet};

pub struct SimulationState {
    pub bodies: RigidBodySet,
    pub colliders: ColliderSet,
    pub impulse_joints: ImpulseJointSet,
}

pub struct PhysicsContext {
    pub backend: PhysicsBackend,
}

impl PhysicsContext {
    pub fn new(backend: PhysicsBackend) -> Self {
        Self { backend }
    }
}

pub async fn setup_physics(
    gpu: Option<&KhalGpuBackend>,
    phys: &SimulationState,
    backend_type: BackendType,
    gpu_error: &mut Option<String>,
    cached_pipeline: &mut Option<GpuPhysicsPipeline>,
) -> PhysicsContext {
    let backend = match backend_type {
        BackendType::Gpu => {
            let gpu = gpu.unwrap();

            if let Some(pipeline) = cached_pipeline.take() {
                let gpu_backend = GpuBackend::with_pipeline(gpu, pipeline, phys).await;
                PhysicsBackend::Gpu(gpu_backend)
            } else {
                match GpuBackend::try_new(gpu, phys).await {
                    Ok(gpu_backend) => PhysicsBackend::Gpu(gpu_backend),
                    Err(e) => {
                        *gpu_error = Some(format!(
                            "GPU backend initialization failed: {}. Using CPU backend.",
                            e
                        ));
                        PhysicsBackend::Cpu(CpuBackend::new(SimulationState {
                            bodies: phys.bodies.clone(),
                            colliders: phys.colliders.clone(),
                            impulse_joints: phys.impulse_joints.clone(),
                        }))
                    }
                }
            }
        }
        BackendType::Cpu => PhysicsBackend::Cpu(CpuBackend::new(SimulationState {
            bodies: phys.bodies.clone(),
            colliders: phys.colliders.clone(),
            impulse_joints: phys.impulse_joints.clone(),
        })),
    };

    PhysicsContext::new(backend)
}
