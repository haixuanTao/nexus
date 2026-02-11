use nexus_mpm_testbed2d::{nexus_mpm, rapier, AppState, PhysicsContext, RapierData};

use glamx::Vec2;
use khal::backend::GpuBackend;
use nexus_mpm::pipeline::MpmData;
use nexus_mpm::solver::{Particle, ParticleModel, SimulationParams};
use rapier::prelude::{ColliderBuilder, RigidBodyBuilder};

#[allow(dead_code)]
fn main() {
    panic!("Run the `mpm_testbed2` binary instead.");
}

pub fn elasticity_demo(backend: &GpuBackend, app_state: &mut AppState) -> PhysicsContext {
    let mut rapier_data = RapierData::default();

    let offset_y = 10.0;
    // let cell_width = 0.1;
    let cell_width = 0.2;
    let mut particles = vec![];
    for i in 0..700 {
        for j in 0..700 {
            let position =
                glamx::vec2(i as f32 + 0.5 + (i / 50) as f32 * 2.0, j as f32 + 0.5)
                    * cell_width
                    / 2.0
                    + Vec2::Y * offset_y;
            let density = 1000.0;
            let radius = cell_width / 4.0;
            let model = ParticleModel::elastic(5.0e6, 0.2);
            particles.push(Particle::new(position, radius, density, model));
        }
    }

    if !app_state.restarting {
        app_state.max_num_substeps = 15;
        app_state.gravity_factor = 2.0;
    };

    let params = SimulationParams {
        gravity: glamx::vec2(0.0, -9.81) * app_state.gravity_factor,
        padding: 0.0,
        dt: 1.0 / 60.0,
    };

    let rb = RigidBodyBuilder::fixed().translation(glamx::vec2(0.0, -1.0));
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(1000.0, 1.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    let rb = RigidBodyBuilder::fixed()
        .translation(glamx::vec2(-20.0, 0.0))
        .rotation(0.5);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(1.0, 60.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    let rb = RigidBodyBuilder::fixed()
        .translation(glamx::vec2(90.0, 0.0))
        .rotation(-0.5);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(1.0, 60.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    let data = MpmData::new(
        backend,
        params,
        &particles,
        &rapier_data.bodies,
        &rapier_data.colliders,
        &[],
        cell_width,
        30_000,
    )
    .unwrap();
    PhysicsContext {
        data,
        rapier_data,
        callbacks: vec![],
        hooks_state: None,
    }
}
