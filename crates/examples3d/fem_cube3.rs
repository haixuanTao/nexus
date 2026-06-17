use glamx::vec3;
use khal::backend::GpuBackend;
use nexus_testbed3d::{Viewer, nexus};

use nexus::fem::{
    mesh::FemMesh,
    pipeline::FemState,
    solver::{FemConfig, FemMaterial, MaterialModel, SolverMethod},
};

pub async fn run(viewer: &mut Viewer) {
    viewer.set_camera(vec3(2.0, 2.0, 2.0), vec3(0.5, 0.3, 0.5));
    let mut scene = viewer.set_fem(build).await;
    while viewer.render(&mut scene).await {
        scene.simulate(viewer).await;
    }
    scene.detach(viewer);
}

fn build(backend: &GpuBackend) -> FemState {
    let mesh = FemMesh::generate_grid([8, 8, 8], vec3(0.3, 0.3, 0.3), vec3(0.7, 0.7, 0.7));

    let material = FemMaterial {
        youngs_modulus: 1e6,
        poissons_ratio: 0.3,
        density: 1000.0,
        model: MaterialModel::LinearCorotated,
    };

    const USE_IMPLICIT: bool = false;

    let config = if USE_IMPLICIT {
        FemConfig {
            dt: 16e-4,
            substeps: 1, // 10,
            floor_y: 0.05,
            damping: 5.0,
            method: SolverMethod::Implicit,
            pcg_iters: 50,
            ls_max_iters: 1,
            newton_iters: 10,
            ..Default::default()
        }
    } else {
        FemConfig {
            dt: 16e-4,
            substeps: 10,
            floor_y: 0.05,
            damping: 5.0,
            method: SolverMethod::Explicit,
            pcg_iters: 50,
            ls_max_iters: 1,
            newton_iters: 10,
            ..Default::default()
        }
    };

    FemState::new(backend, &[(mesh, material)], &config).unwrap()
}
