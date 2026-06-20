use glamx::vec3;
use khal::backend::GpuTimestamps;
use nexus_testbed3d::NexusViewer;

use nexus3d::fem::mesh::FemMesh;
use nexus3d::fem::solver::{FemConfig, FemMaterial, MaterialModel, SolverMethod};
use nexus3d::prelude::{NexusState, NexusPipeline};

pub async fn run(viewer: &mut NexusViewer, pipeline: &mut NexusPipeline) -> anyhow::Result<NexusState> {
    viewer.set_camera(vec3(2.0, 2.0, 2.0), vec3(0.5, 0.3, 0.5));

    let mut state = NexusState::default();

    let mesh = FemMesh::generate_grid([8, 24, 8], vec3(0.3, 0.3, 0.3), vec3(0.7, 1.5, 0.7));
    let material = FemMaterial {
        youngs_modulus: 1e6,
        poissons_ratio: 0.3,
        density: 1000.0,
        model: MaterialModel::LinearCorotated,
    };
    let config = FemConfig {
        dt: 16e-4,
        substeps: 10,
        floor_y: 0.05,
        damping: 5.0,
        method: SolverMethod::Explicit,
        pcg_iters: 50,
        ls_max_iters: 1,
        newton_iters: 10,
        ..Default::default()
    };

    state.insert_fem(viewer.backend(), &[(mesh, material)], &config)?;

    let mut timestamps = GpuTimestamps::new(viewer.backend(), 2048);
    state.finalize(viewer.backend()).await?;

    while viewer.render_frame().await {
        if viewer.simulating() {
            pipeline.simulate(viewer.backend(), &mut state,Some(&mut timestamps)).await;
        }
        viewer.sync(&mut state).await;
    }

    Ok(state)
}
