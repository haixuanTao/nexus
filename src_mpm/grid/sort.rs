//! Particle sorting kernels for spatial acceleration.
//!
//! These kernels handle spatial hashing and sorting to group particles by grid block
//! for efficient neighbor queries during P2G/G2P.

use crate::grid::grid::GpuGrid;
use crate::mpm_shaders::grid::sort::{
    GpuCopyParticlesLenToScanValue, GpuCopyScanValuesToFirstParticles, GpuFinalizeParticlesSort,
    GpuMarkRigidParticlesNeedingBlock, GpuSortRigidParticles, GpuTouchParticleBlocks,
    GpuTouchRigidParticleBlocks, GpuUpdateBlockParticleCount,
};
use crate::solver::GpuRigidParticles;
use khal::Shader;

/// GPU compute kernels for sorting particles into grid cells.
///
/// Implements spatial hashing and sorting to group particles by grid block
/// for efficient neighbor queries during P2G/G2P.
#[derive(Shader)]
pub struct WgSort {
    pub(crate) touch_particle_blocks: GpuTouchParticleBlocks,
    pub(crate) touch_rigid_particle_blocks: GpuTouchRigidParticleBlocks,
    pub(crate) mark_rigid_particles_needing_block: GpuMarkRigidParticlesNeedingBlock,
    pub(crate) update_block_particle_count: GpuUpdateBlockParticleCount,
    pub(crate) copy_particles_len_to_scan_value: GpuCopyParticlesLenToScanValue,
    pub(crate) copy_scan_values_to_first_particles: GpuCopyScanValuesToFirstParticles,
    pub(crate) finalize_particles_sort: GpuFinalizeParticlesSort,
    pub(crate) sort_rigid_particles: GpuSortRigidParticles,
}

impl WgSort {
    /// Sorts rigid body particles into grid cells.
    pub fn launch_sort_rigid_particles(
        &self,
        pass: &mut khal::backend::GpuPass,
        rigid_particles: &mut GpuRigidParticles,
        grid: &mut GpuGrid,
    ) -> Result<(), khal::backend::GpuBackendError> {
        if !rigid_particles.is_empty() {
            self.sort_rigid_particles.call(
                pass,
                rigid_particles.len() as u32,
                &grid.meta,
                &grid.hmap_entries,
                &rigid_particles.sample_points,
                &mut grid.rigid_nodes_linked_lists,
                &mut rigid_particles.node_linked_lists,
            )
        } else {
            Ok(())
        }
    }
}
