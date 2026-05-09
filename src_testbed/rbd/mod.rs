pub mod backend;
pub mod graphics;

pub use backend::{BackendType, CpuBackend, GpuBackend, PhysicsBackend};
pub use graphics::{RenderContext, setup_graphics, update_instances};

use khal::backend::GpuBackend as KhalGpuBackend;
use nexus::rbd::dynamics::GpuSimParams;
use nexus::rbd::pipeline::GpuPhysicsPipeline;
use nexus::rbd::math::Pose;
use rapier::geometry::{ColliderHandle, ColliderSet, SharedShape};
use rapier::prelude::{ImpulseJointSet, MultibodyJointSet, RigidBodySet};
use std::collections::HashMap;

/// Custom visual shape that overrides a collider's default rendering. The shape is
/// drawn at the collider's world pose composed with [`Self::local_pose`] (handy when
/// the collider's shape is a proxy — e.g. an OBB — and the visual mesh sits in a
/// different local frame).
#[derive(Clone)]
pub struct VisualShape {
    pub shape: SharedShape,
    pub local_pose: Pose,
}

impl VisualShape {
    pub fn new(shape: SharedShape) -> Self {
        Self {
            shape,
            local_pose: Pose::IDENTITY,
        }
    }

    pub fn with_local_pose(shape: SharedShape, local_pose: Pose) -> Self {
        Self { shape, local_pose }
    }
}

pub struct BatchEnvironment {
    pub bodies: RigidBodySet,
    pub colliders: ColliderSet,
    pub impulse_joints: ImpulseJointSet,
    pub multibody_joints: MultibodyJointSet,
    pub sim_params: GpuSimParams,
    /// Optional per-collider visual override. When a collider handle is present in
    /// this map its [`VisualShape`] is rendered instead of the collider's own shape.
    pub visuals: HashMap<ColliderHandle, VisualShape>,
}

pub struct SimulationState {
    pub environments: Vec<BatchEnvironment>,
    /// Number of physics steps run per render frame (default: `1`).
    pub num_steps_per_frame: u32,
}

impl SimulationState {
    pub fn from_environments(environments: Vec<BatchEnvironment>) -> Self {
        Self {
            environments,
            num_steps_per_frame: 1,
        }
    }

    pub fn single(
        bodies: RigidBodySet,
        colliders: ColliderSet,
        impulse_joints: ImpulseJointSet,
    ) -> Self {
        Self::single_with_multibody(bodies, colliders, impulse_joints, MultibodyJointSet::new())
    }

    pub fn single_with_multibody(
        bodies: RigidBodySet,
        colliders: ColliderSet,
        impulse_joints: ImpulseJointSet,
        multibody_joints: MultibodyJointSet,
    ) -> Self {
        Self::single_with_multibody_and_visuals(
            bodies,
            colliders,
            impulse_joints,
            multibody_joints,
            HashMap::new(),
        )
    }

    pub fn single_with_multibody_and_visuals(
        bodies: RigidBodySet,
        colliders: ColliderSet,
        impulse_joints: ImpulseJointSet,
        multibody_joints: MultibodyJointSet,
        visuals: HashMap<ColliderHandle, VisualShape>,
    ) -> Self {
        Self::from_environments(vec![BatchEnvironment {
            bodies,
            colliders,
            impulse_joints,
            multibody_joints,
            sim_params: GpuSimParams::default(),
            visuals,
        }])
    }

    /// Sets `sim_params.dt` on every environment to `dt`.
    pub fn with_dt(mut self, dt: f32) -> Self {
        for env in &mut self.environments {
            env.sim_params.dt = dt;
        }
        self
    }

    /// Sets the number of physics steps run between two renders.
    pub fn with_num_steps_per_frame(mut self, num_steps_per_frame: u32) -> Self {
        self.num_steps_per_frame = num_steps_per_frame;
        self
    }
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
                        PhysicsBackend::Cpu(CpuBackend::new(phys))
                    }
                }
            }
        }
        BackendType::Cpu => {
            #[cfg(feature = "cpu")]
            {
                let cpu_backend = KhalGpuBackend::Cpu;
                match GpuBackend::try_new(&cpu_backend, phys).await {
                    Ok(gpu_backend) => PhysicsBackend::Gpu(gpu_backend),
                    Err(e) => {
                        *gpu_error = Some(format!(
                            "Nexus CPU backend initialization failed: {}. Using rapier CPU backend.",
                            e
                        ));
                        PhysicsBackend::Cpu(CpuBackend::new(phys))
                    }
                }
            }
            #[cfg(not(feature = "cpu"))]
            {
                *gpu_error =
                    Some("CPU backend not available (compiled without 'cpu' feature).".to_string());
                PhysicsBackend::Cpu(CpuBackend::new(phys))
            }
        }
        #[cfg(feature = "cuda")]
        BackendType::Cuda => {
            let gpu = gpu.expect("Cuda device initialization failed");

            if let Some(pipeline) = cached_pipeline.take() {
                let gpu_backend = GpuBackend::with_pipeline(gpu, pipeline, phys).await;
                PhysicsBackend::Gpu(gpu_backend)
            } else {
                match GpuBackend::try_new(gpu, phys).await {
                    Ok(gpu_backend) => PhysicsBackend::Gpu(gpu_backend),
                    Err(e) => {
                        *gpu_error = Some(format!(
                            "CUDA backend initialization failed: {}. Using CPU backend.",
                            e
                        ));
                        PhysicsBackend::Cpu(CpuBackend::new(phys))
                    }
                }
            }
        }
        #[cfg(feature = "metal")]
        BackendType::Metal => {
            let gpu = gpu.expect("Metal device initialization failed");

            if let Some(pipeline) = cached_pipeline.take() {
                let gpu_backend = GpuBackend::with_pipeline(gpu, pipeline, phys).await;
                PhysicsBackend::Gpu(gpu_backend)
            } else {
                match GpuBackend::try_new(gpu, phys).await {
                    Ok(gpu_backend) => PhysicsBackend::Gpu(gpu_backend),
                    Err(e) => {
                        *gpu_error = Some(format!(
                            "Metal backend initialization failed: {}. Using CPU backend.",
                            e
                        ));
                        PhysicsBackend::Cpu(CpuBackend::new(phys))
                    }
                }
            }
        }
        BackendType::Rapier => PhysicsBackend::Cpu(CpuBackend::new(phys)),
    };

    PhysicsContext::new(backend)
}
