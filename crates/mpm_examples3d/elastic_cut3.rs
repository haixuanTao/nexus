use nexus_mpm_testbed3d::{RapierData, nexus};

use glamx::{Pose3, vec3};
use khal::backend::GpuBackend;
use nexus::mpm::{
    pipeline::MpmData,
    solver::{Particle, ParticleModel, SimulationParams},
};
use nexus_mpm_testbed3d::{AppState, PhysicsContext};
use rapier3d::parry::utils::Array2;
use rapier3d::prelude::{ColliderBuilder, HeightField, RigidBodyBuilder, TriMeshFlags};

#[allow(dead_code)]
fn main() {
    panic!("Run the `mpm_testbed3` binary instead.");
}

pub fn elastic_cut_demo(backend: &GpuBackend, app_state: &mut AppState) -> PhysicsContext {
    let mut rapier_data = RapierData::default();

    let nxz = 50;
    let cell_width = 1.0;
    let mut particles = vec![];
    for i in 0..nxz {
        for j in 0..30 {
            for k in 0..nxz {
                let position = vec3(
                    i as f32 + 0.5 - nxz as f32 / 2.0,
                    j as f32 + 0.5 + 60.0,
                    k as f32 + 0.5 - nxz as f32 / 2.0,
                ) * cell_width
                    / 2.0;
                let density = 2700.0;
                let radius = cell_width / 4.0;
                let model = ParticleModel::elastic(1.0e7, 0.2);
                particles.push(Particle::new(position, radius, density, model));
            }
        }
    }

    if !app_state.restarting {
        app_state.min_num_substeps = 10;
        app_state.max_num_substeps = 40;
        app_state.gravity_factor = 4.0;
    };

    let params = SimulationParams {
        gravity: vec3(0.0, -9.81, 0.0) * app_state.gravity_factor,
        dt: 1.0 / 60.0,
    };

    // Floor
    let rb = RigidBodyBuilder::fixed().translation(vec3(0.0, -4.0, 0.0));
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(100.0, 1.0, 100.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    // Cutting planes (3 heightfield trimeshes)
    for k in 0..3 {
        let heights = Array2::zeros(10, 10);
        let heightfield = HeightField::new(heights, vec3(35.0, 1.0, 10.0));
        let (mut vtx, idx) = heightfield.to_trimesh();
        vtx.iter_mut().for_each(|pt| {
            *pt =
                Pose3::rotation(vec3(1.3, 0.0, 0.0)) * *pt + vec3(0.0, 10.0, k as f32 * 10.0 - 10.0)
        });
        let rb = RigidBodyBuilder::fixed();
        let rb_handle = rapier_data.bodies.insert(rb);
        let co = ColliderBuilder::trimesh_with_flags(vtx, idx, TriMeshFlags::ORIENTED).unwrap();
        rapier_data
            .colliders
            .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);
    }

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
