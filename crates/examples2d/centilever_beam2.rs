use nexus_testbed2d::DemoBuilder;
use nexus_testbed2d::mpm::{MpmAppState, MpmPhysicsContext, RapierData};
use nexus_testbed2d::{nexus, rapier};

use khal::backend::GpuBackend;
use nexus::mpm::pipeline::MpmData;
use nexus::mpm::solver::{
    BoundaryCondition, BoundaryConditionExt, Particle, ParticleModel, SimulationParams,
};
use rapier::prelude::{ColliderBuilder, RigidBodyBuilder};

#[allow(dead_code)]
fn main() {
    panic!("Run the `all_examples2` binary instead.");
}

pub fn builder() -> DemoBuilder {
    DemoBuilder::mpm("Cantilever beam", build)
}

fn build(backend: &GpuBackend, app_state: &mut MpmAppState) -> MpmPhysicsContext {
    let mut rapier_data = RapierData::default();

    let width = 10.0;
    let height = 2.0;
    let fixed_part = 1.0;
    let cell_width = 0.2;
    let particle_per_cell_dim = 2;
    let young_modulus = 1.0e8;
    let poisson_ratio = 0.3;

    let diameter = cell_width / particle_per_cell_dim as f32;
    let ni = ((width + fixed_part) / diameter).ceil() as usize;
    let nj = (height / diameter).ceil() as usize;

    let mut particles = vec![];
    for i in 0..ni {
        for j in 0..nj {
            let position = glamx::vec2(i as f32, j as f32) * diameter;
            let density = 1000.0;
            let radius = diameter / 2.0;
            let model = ParticleModel::elastic_neo_hookean(young_modulus, poisson_ratio);
            particles.push(Particle::new(position, radius, density, model));
        }
    }

    if !app_state.restarting {
        app_state.min_num_substeps = 150;
        app_state.max_num_substeps = 150;
        app_state.gravity_factor = 1.0;
    };

    let params = SimulationParams {
        gravity: glamx::vec2(0.0, -9.81) * app_state.gravity_factor,
        padding: 0.0,
        dt: 1.0 / 60.0,
    };

    let rb = RigidBodyBuilder::fixed()
        .translation(glamx::vec2(0.0, height / 2.0))
        .build();
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(fixed_part, height);
    let ground = rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    let data = MpmData::new(
        backend,
        params,
        &particles,
        &rapier_data.bodies,
        &rapier_data.colliders,
        &[(ground, BoundaryCondition::stick())],
        cell_width,
        30_000,
    )
    .unwrap();
    MpmPhysicsContext {
        data,
        rapier_data,
        callbacks: vec![],
        hooks_state: None,
    }
}
