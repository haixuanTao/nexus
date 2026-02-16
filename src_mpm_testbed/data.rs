use crate::step::SimulationStepResult;
use khal::backend::GpuBackend;
use nexus_mpm::pipeline::{MpmData, MpmPipeline};
use nexus_mpm::solver::{GpuParticleModel, GpuParticleModelData};
use rapier::prelude::{
    CCDSolver, ColliderSet, DefaultBroadPhase, ImpulseJointSet, IntegrationParameters,
    IslandManager, MultibodyJointSet, NarrowPhase, PhysicsPipeline, RigidBodySet,
};
use std::any::Any;

pub struct AppState<GpuModel: GpuParticleModelData = GpuParticleModel> {
    pub run_state: RunState,
    pub render_mode: RenderMode,
    pub pipeline: MpmPipeline<GpuModel>,
    pub min_num_substeps: u32,
    pub max_num_substeps: u32,
    pub num_substeps: u32,
    pub gravity_factor: f32,
    pub restarting: bool,
    pub show_rigid_particles: bool,
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

pub trait PhysicsCallback<GpuModel: GpuParticleModelData> {
    fn update(&mut self, state: &mut PhysicsState<'_, GpuModel>);
}

impl<GpuModel: GpuParticleModelData, F: FnMut(&mut PhysicsState<GpuModel>)>
    PhysicsCallback<GpuModel> for F
{
    fn update(&mut self, state: &mut PhysicsState<'_, GpuModel>) {
        (*self)(state);
    }
}

pub struct PhysicsState<'a, GpuModel: GpuParticleModelData = GpuParticleModel> {
    pub backend: &'a GpuBackend,
    pub data: &'a mut MpmData<GpuModel>,
    pub results: &'a SimulationStepResult,
    pub(crate) step_id: usize,
}

impl<GpuModel: GpuParticleModelData> PhysicsState<'_, GpuModel> {
    pub fn step_id(&self) -> usize {
        self.step_id
    }
}

pub struct PhysicsContext<GpuModel: GpuParticleModelData = GpuParticleModel> {
    pub data: MpmData<GpuModel>,
    pub rapier_data: RapierData,
    pub callbacks: Vec<Box<dyn PhysicsCallback<GpuModel>>>,
    pub hooks_state: Option<Box<dyn Any>>,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum RunState {
    Running,
    Paused,
    Step,
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
