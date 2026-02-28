use nexus_testbed2d::mpm::{MpmAppState, MpmPhysicsContext, RapierData};
use nexus_testbed2d::{nexus, rapier};

use glamx::Vec2;
use khal::backend::GpuBackend;
use nexus::mpm::pipeline::MpmData;
use nexus::mpm::solver::{Particle, ParticleModel, SimulationParams};
use rapier::prelude::{ColliderBuilder, RigidBodyBuilder};

#[allow(dead_code)]
fn main() {
    panic!("Run the `all_examples2` binary instead.");
}

pub fn sand_demo(backend: &GpuBackend, app_state: &mut MpmAppState) -> MpmPhysicsContext {
    let mut rapier_data = RapierData::default();

    let offset_y = 46.0;
    // let cell_width = 0.1;
    let cell_width = 0.2;
    let mut particles = vec![];
    for i in 0..700 {
        for j in 0..700 {
            let position =
                glamx::vec2(i as f32 + 0.5, j as f32 + 0.5) * cell_width / 2.0 + Vec2::Y * offset_y;
            let density = 1000.0;
            let radius = cell_width / 4.0;
            let young_modulus = 1.0e7;
            let poisson_ratio = 0.2;
            let model = ParticleModel::sand(young_modulus, poisson_ratio);

            particles.push(Particle::new(position, radius, density, model));
        }
    }

    if !app_state.restarting {
        app_state.min_num_substeps = 10;
        app_state.max_num_substeps = 10;
        app_state.gravity_factor = 1.0;
    };

    let params = SimulationParams {
        gravity: glamx::vec2(0.0, -9.81) * app_state.gravity_factor,
        padding: 0.0,
        dt: 1.0 / 60.0,
    };

    const ANGVEL: f32 = 2.0;

    /*
     * Static platforms.
     */
    let rb = RigidBodyBuilder::fixed().translation(glamx::vec2(35.0, -1.0));
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(42.0, 1.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    let rb = RigidBodyBuilder::fixed()
        .translation(glamx::vec2(-25.0, 45.0))
        .rotation(0.5);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(1.0, 52.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    let rb = RigidBodyBuilder::fixed()
        .translation(glamx::vec2(95.0, 45.0))
        .rotation(-0.5);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(1.0, 52.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    /*
     * Rotating platforms.
     */
    let rb = RigidBodyBuilder::kinematic_velocity_based()
        .translation(glamx::vec2(5.0, 35.0))
        .angvel(ANGVEL);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(1.0, 10.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    let rb = RigidBodyBuilder::kinematic_velocity_based()
        .translation(glamx::vec2(35.0, 35.0))
        .angvel(-ANGVEL);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(10.0, 1.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    let rb = RigidBodyBuilder::kinematic_velocity_based()
        .translation(glamx::vec2(65.0, 35.0))
        .angvel(ANGVEL);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(1.0, 10.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    let rb = RigidBodyBuilder::kinematic_velocity_based()
        .translation(glamx::vec2(20.0, 20.0))
        .angvel(-ANGVEL);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::ball(5.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    let rb = RigidBodyBuilder::kinematic_velocity_based()
        .translation(glamx::vec2(50.0, 20.0))
        .angvel(-ANGVEL);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::capsule_y(5.0, 3.0);
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
    MpmPhysicsContext {
        data,
        rapier_data,
        callbacks: vec![],
        hooks_state: None,
    }
}
