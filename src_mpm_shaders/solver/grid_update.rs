//! Grid update kernel: converts grid momentum to velocity and applies gravity.
//!
//! After P2G transfers momentum onto the grid, this kernel converts momentum to
//! velocity (dividing by mass), applies gravity, and clamps velocities so no node
//! moves more than one cell width per timestep.

use crate::grid::grid::*;
use crate::solver::params::SimulationParams;
use crate::{
    scalar_part, vector_part, vector_plus_one, MaybeIndexUnchecked, Vector, VectorPlusOne,
};
use glamx::*;
use khal_derive::spirv_bindgen;
use spirv_std::spirv;

/// GPU kernel: grid update.
///
/// Converts grid momentum to velocity, applies gravity, and clamps velocities.
/// Dispatched with one workgroup per active block, one thread per node.
#[spirv_bindgen]
#[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8))))]
#[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
pub fn gpu_grid_update(
    #[spirv(workgroup_id)] block_id: spirv_std::glam::UVec3,
    #[spirv(local_invocation_id)] tid: spirv_std::glam::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] sim_params: &SimulationParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] grid_data: &[Grid],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] active_blocks: &[ActiveBlockHeader],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] nodes: &mut [Node],
) {
    let bid = block_id.x;
    let vid = active_blocks.at(bid as usize).virtual_id;
    let cell_width = grid_data.at(0).cell_width;

    let global_chunk_id = block_header_id_to_physical_id(BlockHeaderId { id: bid });

    #[cfg(feature = "dim2")]
    let global_node_id = node_id(global_chunk_id, UVec2::new(tid.x, tid.y));
    #[cfg(feature = "dim3")]
    let global_node_id = node_id(global_chunk_id, UVec3::new(tid.x, tid.y, tid.z));

    #[cfg(feature = "dim2")]
    let cell_pos = Vec2::new(
        (vid.id.x * 8 + tid.x as i32) as f32,
        (vid.id.y * 8 + tid.y as i32) as f32,
    ) * cell_width;
    #[cfg(feature = "dim3")]
    let cell_pos = Vec3::new(
        (vid.id.x * 4 + tid.x as i32) as f32,
        (vid.id.y * 4 + tid.y as i32) as f32,
        (vid.id.z * 4 + tid.z as i32) as f32,
    ) * cell_width;

    let global_id = global_node_id.id as usize;
    let momentum_velocity_mass = nodes.at(global_id).momentum_velocity_mass;
    let momentum_velocity_mass_incompatible =
        nodes.at(global_id).momentum_velocity_mass_incompatible;
    let new_grid_velocity_mass =
        update_single_cell(sim_params, cell_width, cell_pos, momentum_velocity_mass);
    let new_grid_velocity_mass_incompatible = update_single_cell(
        sim_params,
        cell_width,
        cell_pos,
        momentum_velocity_mass_incompatible,
    );
    nodes.at_mut(global_id).momentum_velocity_mass = new_grid_velocity_mass;
    nodes.at_mut(global_id).momentum_velocity_mass_incompatible =
        new_grid_velocity_mass_incompatible;
}

/// Updates a single cell's momentum+mass to velocity+mass.
///
/// Converts momentum to velocity by dividing by mass, adds gravity,
/// and clamps velocity to at most one cell width per timestep.
#[inline]
fn update_single_cell(
    sim_params: &SimulationParams,
    cell_width: f32,
    _cell_pos: Vector,
    momentum_velocity_mass: VectorPlusOne,
) -> VectorPlusOne {
    let mass = scalar_part(momentum_velocity_mass);
    let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
    let momentum = vector_part(momentum_velocity_mass);
    let mut velocity = (momentum + sim_params.gravity * (mass * sim_params.dt)) * inv_mass;

    // Clamp the velocity so it doesn't exceed 1 grid cell in one step.
    let vel_limit = Vector::splat(cell_width / sim_params.dt);
    velocity = velocity.clamp(-vel_limit, vel_limit);

    vector_plus_one(velocity, mass)
}
