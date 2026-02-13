use nexus_mpm_testbed3d::{RapierData, nexus_mpm};

use glamx::vec3;
use khal::backend::GpuBackend;
use nexus_mpm::{
    pipeline::MpmData,
    solver::{Particle, ParticleModel, SimulationParams},
};
use nexus_mpm_testbed3d::{AppState, PhysicsContext};
use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder};

#[allow(dead_code)]
fn main() {
    panic!("Run the `mpm_testbed3` binary instead.");
}

const DENSITY: f32 = 2700.0;
const YOUNG_MODULUS: f32 = 2.0e9;
const POISSON_RATIO: f32 = 0.2;

pub fn sand_demo(backend: &GpuBackend, app_state: &mut AppState) -> PhysicsContext {
    let mut rapier_data = RapierData::default();

    let nxz = 45;
    let cell_width = 1.0;
    let mut particles = vec![];
    for i in 0..nxz {
        for j in 0..100 {
            for k in 0..nxz {
                let position = vec3(
                    i as f32 + 0.5 - nxz as f32 / 2.0,
                    j as f32 + 0.5 + 10.0,
                    k as f32 + 0.5 - nxz as f32 / 2.0,
                ) * cell_width
                    / 2.0;
                let radius = cell_width / 4.0;
                let model = ParticleModel::sand(YOUNG_MODULUS, POISSON_RATIO);
                let particle = Particle::new(position, radius, DENSITY, model);
                particles.push(particle);
            }
        }
    }

    if !app_state.restarting {
        app_state.min_num_substeps = 20;
        app_state.max_num_substeps = 20;
        app_state.gravity_factor = 1.0;
    };

    let params = SimulationParams {
        gravity: vec3(0.0, -9.81, 0.0) * app_state.gravity_factor,
        dt: 1.0 / 60.0,
    };

    // Floor
    let rb = RigidBodyBuilder::fixed().translation(vec3(0.0, -4.0, 0.0));
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(100.0, 4.0, 100.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    // Wall -Z
    let rb = RigidBodyBuilder::fixed().translation(vec3(0.0, 5.0, -35.0));
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(35.0, 5.0, 0.5);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    // Wall +Z
    let rb = RigidBodyBuilder::fixed().translation(vec3(0.0, 5.0, 35.0));
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(35.0, 5.0, 0.5);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    // Wall -X
    let rb = RigidBodyBuilder::fixed().translation(vec3(-35.0, 5.0, 0.0));
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(0.5, 5.0, 35.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    // Wall +X
    let rb = RigidBodyBuilder::fixed().translation(vec3(35.0, 5.0, 0.0));
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(0.5, 5.0, 35.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    // Rotating blade
    let rb = RigidBodyBuilder::kinematic_velocity_based()
        .translation(vec3(0.0, 2.0, 0.0))
        .rotation(vec3(0.0, 0.0, -0.5))
        .angvel(vec3(0.0, -1.0, 0.0));
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(0.5, 2.0, 30.0);
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
