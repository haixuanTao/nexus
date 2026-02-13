//! Grid node update kernel.
//!
//! Updates grid node velocities by applying forces (gravity, boundary conditions)
//! and solving momentum equations on the grid.

use crate::grid::grid::{GpuGrid, indirect_dispatch_tensor};
use crate::mpm_shaders::solver::grid_update::GpuGridUpdate;
use crate::solver::GpuSimulationParams;
use khal::Shader;
use khal::backend::{GpuBackendError, GpuPass};

/// GPU compute kernel for updating grid node velocities.
///
/// Applies external forces (gravity), boundary conditions (sticky/slip walls),
/// and solves momentum equations on grid nodes. Runs between P2G and G2P stages.
#[derive(Shader)]
pub struct WgGridUpdate {
    /// Compiled grid update compute shader.
    grid_update: GpuGridUpdate,
}

impl WgGridUpdate {
    /// Launches the grid update kernel.
    ///
    /// # Arguments
    ///
    /// * `pass` - Compute pass to record commands into
    /// * `sim_params` - Simulation parameters (gravity, timestep)
    /// * `grid` - Grid with nodes to update
    pub fn launch(
        &self,
        pass: &mut GpuPass,
        sim_params: &GpuSimulationParams,
        grid: &mut GpuGrid,
    ) -> Result<(), GpuBackendError> {
        self.grid_update.call(
            pass,
            indirect_dispatch_tensor(&grid.indirect_n_g2p_p2g_groups),
            &sim_params.params,
            &grid.meta,
            &grid.active_blocks,
            &mut grid.nodes,
        )
    }
}
