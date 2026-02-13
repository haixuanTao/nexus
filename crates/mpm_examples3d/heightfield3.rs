use nexus_mpm_testbed3d::{RapierData, nexus_mpm};

use glamx::vec3;
use khal::backend::GpuBackend;
use nexus_mpm::{
    pipeline::MpmData,
    solver::{BoundaryCondition, BoundaryConditionExt, Particle, ParticleModel, SimulationParams},
};
use nexus_mpm_testbed3d::{AppState, PhysicsContext};
use rapier3d::parry::utils::Array2;
use rapier3d::prelude::{ColliderBuilder, HeightField, RigidBodyBuilder};

#[allow(dead_code)]
fn main() {
    panic!("Run the `mpm_testbed3` binary instead.");
}

pub fn heightfield_demo(backend: &GpuBackend, app_state: &mut AppState) -> PhysicsContext {
    let mut rapier_data = RapierData::default();

    let nxz = 45;
    let cell_width = 1.0;
    let mut particles = vec![];
    for i in 0..nxz {
        for j in 0..100 {
            for k in 0..nxz {
                let position = vec3(
                    i as f32 + 0.5 - nxz as f32 / 2.0,
                    j as f32 + 0.5 + 14.0,
                    k as f32 + 0.5 - nxz as f32 / 2.0,
                ) * cell_width
                    / 2.0;
                let density = 2700.0;
                let radius = cell_width / 4.0;
                let model = ParticleModel::sand(2.0e9, 0.2);
                particles.push(Particle::new(position, radius, density, model));
            }
        }
    }

    if !app_state.restarting {
        app_state.min_num_substeps = 10;
        app_state.max_num_substeps = 40;
        app_state.gravity_factor = 1.0;
    };

    let params = SimulationParams {
        gravity: vec3(0.0, -9.81, 0.0) * app_state.gravity_factor,
        dt: 1.0 / 60.0,
    };

    // Sinusoidal heightfield terrain
    let heights = Array2::from_fn(200, 200, |i, j| {
        (i as f32 / 10.0).sin() * (j as f32 / 10.0).cos()
    });
    let heightfield = HeightField::new(heights, vec3(100.0, 5.0, 100.0));
    let (vtx, idx) = heightfield.to_trimesh();
    let rb = RigidBodyBuilder::fixed();
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::trimesh(vtx, idx).unwrap();
    let floor = rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    let data = MpmData::new(
        backend,
        params,
        &particles,
        &rapier_data.bodies,
        &rapier_data.colliders,
        &[(floor, BoundaryCondition::stick())],
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
