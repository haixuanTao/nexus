#[cfg(feature = "dim2")]
use nexus2d as nexus;
#[cfg(feature = "dim3")]
use nexus3d as nexus;
#[cfg(feature = "dim2")]
use rapier2d as rapier;
#[cfg(feature = "dim3")]
use rapier3d as rapier;

use super::SimulationBackend;
use crate::SimulationState;
use khal::backend::GpuBackend as KhalGpuBackend;
use nexus::math::Pose;
use nexus::pipeline::RunStats;
use rapier::dynamics::{CCDSolver, IntegrationParameters, IslandManager};
use rapier::geometry::{BroadPhaseBvh, ColliderSet, NarrowPhase};
use rapier::prelude::{ImpulseJointSet, MultibodyJointSet, PhysicsPipeline, RigidBodySet};

/// CPU-based physics backend using rapier
pub struct CpuBackend {
    pipeline: PhysicsPipeline,
    integration_parameters: IntegrationParameters,
    islands: IslandManager,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd_solver: CCDSolver,
    poses_cache: Vec<Pose>,
}

impl CpuBackend {
    pub fn new(phys: SimulationState) -> Self {
        let mut poses_cache = Vec::new();
        let mut shapes_cache = Vec::new();

        // Build initial poses and shapes from the simulation state
        for (_, co) in phys.colliders.iter() {
            poses_cache.push(*co.position());
            shapes_cache.push(co.shared_shape().clone());
        }

        #[allow(unused_mut)] // mut not needed in 2D but needed in 3d.
        let mut params = IntegrationParameters::default();
        Self {
            pipeline: PhysicsPipeline::new(),
            integration_parameters: params,
            islands: IslandManager::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            bodies: phys.bodies,
            colliders: phys.colliders,
            impulse_joints: phys.impulse_joints,
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            poses_cache,
        }
    }
}

impl SimulationBackend for CpuBackend {
    fn poses(&self) -> &[Pose] {
        &self.poses_cache
    }
    fn num_bodies(&self) -> usize {
        self.poses().len()
    }
    fn num_joints(&self) -> usize {
        self.impulse_joints.len()
    }

    async fn step(&mut self, _gpu: Option<&KhalGpuBackend>) -> RunStats {
        let t0 = web_time::Instant::now();

        #[cfg(feature = "dim2")]
        let gravity = glamx::Vec2::Y * -9.81;
        #[cfg(feature = "dim3")]
        let gravity = glamx::Vec3::Y * -9.81;

        self.pipeline.step(
            gravity,
            &self.integration_parameters,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            &(),
            &(),
        );
        let total_sim_time = t0.elapsed();

        // Update poses cache
        self.poses_cache.clear();
        for (_, co) in self.colliders.iter() {
            self.poses_cache.push(*co.position());
        }

        RunStats {
            total_simulation_time_with_readback: total_sim_time,
            ..Default::default()
        }
    }
}
