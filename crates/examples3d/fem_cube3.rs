use glamx::vec3;
use khal::backend::GpuBackend;
use nexus_testbed3d::{nexus, DemoBuilder};

use nexus::fem::{
    mesh::FemMesh,
    pipeline::FemData,
    solver::{FemConfig, FemMaterial, MaterialModel, SolverMethod},
};

#[allow(dead_code)]
fn main() {
    panic!("Run the `all_examples3` binary instead.");
}

pub fn builder() -> DemoBuilder {
    DemoBuilder::fem("FEM cube", build)
        .with_camera(vec3(2.0, 2.0, 2.0), vec3(0.5, 0.3, 0.5))
}

fn build(backend: &GpuBackend) -> FemData {
    // Match Genesis reference: 0.4m cube in [0.3, 0.7]^3
    let mesh = FemMesh::generate_grid(
        [8, 8, 8],
        vec3(0.3, 0.3, 0.3),
        vec3(0.7, 0.7, 0.7),
    );

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

    FemData::new(backend, &[(mesh, material)], &config).unwrap()
}
