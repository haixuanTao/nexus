use crate::mpm::step::SimulationStepResult;
use khal::backend::GpuBackend;
use nexus::mpm::pipeline::{MpmState, MpmPipeline};
use nexus::mpm::solver::{GpuParticleModel};
use rapier::prelude::{
    CCDSolver, ColliderSet, DefaultBroadPhase, ImpulseJointSet, IntegrationParameters,
    IslandManager, MultibodyJointSet, NarrowPhase, PhysicsPipeline, RigidBodySet,
};
use std::any::Any;

pub struct MpmAppState {
    pub render_mode: RenderMode,
    pub pipeline: MpmPipeline,
    pub min_num_substeps: u32,
    pub max_num_substeps: u32,
    pub num_substeps: u32,
    pub gravity_factor: f32,
    pub restarting: bool,
    pub show_rigid_particles: bool,
    pub use_cpic: bool,
}

#[derive(Default)]
pub struct RapierData {
    pub bodies: RigidBodySet,
    pub colliders: ColliderSet,
    pub impulse_joints: ImpulseJointSet,
    pub multibody_joints: MultibodyJointSet,
    pub params: IntegrationParameters,
    pub physics_pipeline: PhysicsPipeline,
    pub narrow_phase: NarrowPhase,
    pub broad_phase: DefaultBroadPhase,
    pub ccd_solver: CCDSolver,
    pub islands: IslandManager,
}

pub struct PhysicsState<'a> {
    pub backend: &'a GpuBackend,
    pub data: &'a mut MpmState,
    pub results: &'a SimulationStepResult,
    pub(crate) step_id: usize,
}

impl PhysicsState<'_> {
    pub fn step_id(&self) -> usize {
        self.step_id
    }
}

pub struct MpmPhysicsContext {
    pub data: MpmState,
    pub rapier_data: RapierData,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum RenderMode {
    Default = 0,
    Volume = 1,
    Velocity = 2,
    Phase = 3,
    CdfNormals = 4,
    CdfDistances = 5,
    CdfSigns = 6,
}

impl RenderMode {
    pub const ALL: &'static [RenderMode] = &[
        Self::Default,
        Self::Volume,
        Self::Velocity,
        Self::Phase,
        Self::CdfNormals,
        Self::CdfDistances,
        Self::CdfSigns,
    ];

    pub fn text(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Volume => "volume",
            Self::Velocity => "velocity",
            Self::Phase => "phase",
            Self::CdfNormals => "cdf (normals)",
            Self::CdfDistances => "cdf (distances)",
            Self::CdfSigns => "cdf (signs)",
        }
    }
}
