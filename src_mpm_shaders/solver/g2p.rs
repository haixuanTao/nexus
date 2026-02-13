//! Grid-to-Particle (G2P) transfer kernel.
//!
//! This kernel transfers grid node velocities back to particles using APIC
//! (Affine Particle-In-Cell) interpolation. It handles CPIC compatibility
//! checks, computes velocity gradients for the affine matrix, and accumulates
//! rigid body velocities for particles near colliders.
//!
//! Corresponds to `g2p.slang`.

use crate::grid::grid::*;
use crate::grid::kernel::*;
use crate::nexus_shaders::dynamics::{
    velocity_at_point, Velocity as BodyVelocity, WorldMassProperties as BodyMassProperties,
};
use crate::solver::boundary_condition::BoundaryCondition;
use crate::solver::params::SimulationParams;
use crate::solver::particle::{
    associated_cell_index_in_block_off_by_one, dir_to_associated_grid_node, Dynamics, Position,
};
use crate::PaddingExt;
use crate::{
    scalar_part, vector_part, vector_plus_one, Matrix, MaybeIndexUnchecked, PaddedMatrix, Vector,
    VectorPlusOne,
};
use glamx::*;
use khal_derive::spirv_bindgen;
use spirv_std::arch::workgroup_memory_barrier_with_group_sync;
use spirv_std::spirv;
use unroll::unroll_for_loops;
/*
 * Constants.
 */

#[cfg(feature = "dim2")]
const NUM_SHARED_CELLS: usize = 100; // 10 * 10
#[cfg(feature = "dim3")]
const NUM_SHARED_CELLS: usize = 216; // 6 * 6 * 6

const WORKGROUP_SIZE: u32 = 64;

/*
 * Global -> shared memory transfer.
 */

#[inline]
#[unroll_for_loops]
fn global_shared_memory_transfers(
    grid_data: &[Grid],
    hmap_entries: &[GridHashMapEntry],
    nodes: &[Node],
    tid: spirv_std::glam::UVec3,
    active_block_vid: BlockVirtualId,
    shared_nodes_vel_mass: &mut [VectorPlusOne; NUM_SHARED_CELLS],
    shared_nodes_vel_mass_incompatible: &mut [VectorPlusOne; NUM_SHARED_CELLS],
    shared_nodes_cdf: &mut [NodeCdf; NUM_SHARED_CELLS],
) {
    let base_block_pos_int = active_block_vid.id;

    #[cfg(feature = "dim2")]
    {
        for i_loop in 0..2 {
            for j_loop in 0..2 {
                if !((i_loop == 1 && tid.x > 1) || (j_loop == 1 && tid.y > 1)) {
                    let octant = UVec2::new(i_loop as u32, j_loop as u32);
                    let octant_hid = find_block_header_id(
                        grid_data,
                        hmap_entries,
                        &BlockVirtualId {
                            id: base_block_pos_int + IVec2::new(octant.x as i32, octant.y as i32),
                        },
                    );
                    let shared_index = octant * 8 + UVec2::new(tid.x, tid.y);
                    let flat_shared_index =
                        flatten_shared_index(shared_index.x, shared_index.y) as usize;

                    if octant_hid.id != NONE {
                        let global_chunk_id = block_header_id_to_physical_id(octant_hid);
                        let tid_xy = UVec2::new(tid.x, tid.y);
                        let global_node_id = node_id(global_chunk_id, tid_xy);
                        let node = nodes.read(global_node_id.id as usize);
                        shared_nodes_vel_mass[flat_shared_index] = node.momentum_velocity_mass;
                        shared_nodes_vel_mass_incompatible[flat_shared_index] =
                            node.momentum_velocity_mass_incompatible;
                        shared_nodes_cdf[flat_shared_index] = node.cdf;
                    } else {
                        shared_nodes_vel_mass[flat_shared_index] = VectorPlusOne::ZERO;
                        shared_nodes_vel_mass_incompatible[flat_shared_index] = VectorPlusOne::ZERO;
                        shared_nodes_cdf[flat_shared_index] = NodeCdf::new(0.0, 0, NONE);
                    }
                }
            }
        }
    }

    #[cfg(feature = "dim3")]
    {
        for i_loop in 0..2 {
            for j_loop in 0..2 {
                for k_loop in 0..2 {
                    if !((i_loop == 1 && tid.x > 1)
                        || (j_loop == 1 && tid.y > 1)
                        || (k_loop == 1 && tid.z > 1))
                    {
                        let octant = UVec3::new(i_loop, j_loop, k_loop);
                        let octant_hid = find_block_header_id(
                            grid_data,
                            hmap_entries,
                            &BlockVirtualId::new(
                                base_block_pos_int
                                    + IVec3::new(octant.x as i32, octant.y as i32, octant.z as i32),
                            ),
                        );
                        let tid_xyz = UVec3::new(tid.x, tid.y, tid.z);
                        let shared_index = octant * 4 + tid_xyz;
                        let flat_shared_index =
                            flatten_shared_index(shared_index.x, shared_index.y, shared_index.z)
                                as usize;

                        if octant_hid.id != NONE {
                            let global_chunk_id = block_header_id_to_physical_id(octant_hid);
                            let global_node_id = node_id(global_chunk_id, tid_xyz);
                            let node = nodes.read(global_node_id.id as usize);
                            shared_nodes_vel_mass[flat_shared_index] = node.momentum_velocity_mass;
                            shared_nodes_vel_mass_incompatible[flat_shared_index] =
                                node.momentum_velocity_mass_incompatible;
                            shared_nodes_cdf[flat_shared_index] = node.cdf;
                        } else {
                            shared_nodes_vel_mass[flat_shared_index] = VectorPlusOne::ZERO;
                            shared_nodes_vel_mass_incompatible[flat_shared_index] =
                                VectorPlusOne::ZERO;
                            shared_nodes_cdf[flat_shared_index] = NodeCdf::new(0.0, 0, NONE);
                        }
                    }
                }
            }
        }
    }
}

/*
 * Per-particle G2P interpolation.
 */

#[inline]
#[allow(clippy::too_many_arguments)]
#[unroll_for_loops]
fn particle_g2p(
    body_vels: &[BodyVelocity],
    body_mprops: &[BodyMassProperties],
    body_materials: &[BoundaryCondition],
    particles_pos: &[Position],
    particles_dyn: &mut [Dynamics],
    particle_id: u32,
    cell_width: f32,
    _dt: f32,
    shared_nodes_vel_mass: &[VectorPlusOne; NUM_SHARED_CELLS],
    shared_nodes_vel_mass_incompatible: &[VectorPlusOne; NUM_SHARED_CELLS],
    shared_nodes_cdf: &[NodeCdf; NUM_SHARED_CELLS],
) {
    let mut rigid_vel = Vector::ZERO;
    let mut momentum_velocity_mass = VectorPlusOne::ZERO;
    let mut velocity_gradient = Matrix::ZERO;
    let mut vel_grad_det = 0.0f32;

    // G2P
    if particles_dyn.at(particle_id as usize).enabled != 0 {
        let particle_pos = particles_pos.read(particle_id as usize);
        let particle_vel = particles_dyn.at(particle_id as usize).velocity;
        let particle_cdf = particles_dyn.at(particle_id as usize).cdf;

        let inv_d = QuadraticKernel::inv_d(cell_width);
        let ref_elt_pos_minus_particle_pos = dir_to_associated_grid_node(&particle_pos, cell_width);
        let w = QuadraticKernel::precompute_weights(ref_elt_pos_minus_particle_pos, cell_width);

        let assoc_cell_index_in_block =
            associated_cell_index_in_block_off_by_one(&particle_pos, cell_width);

        #[cfg(feature = "dim2")]
        let packed_cell_index_in_block =
            flatten_shared_index(assoc_cell_index_in_block.x, assoc_cell_index_in_block.y);
        #[cfg(feature = "dim3")]
        let packed_cell_index_in_block = flatten_shared_index(
            assoc_cell_index_in_block.x,
            assoc_cell_index_in_block.y,
            assoc_cell_index_in_block.z,
        );

        for i in 0..NBH_LEN as u32 {
            let shift = NBH_SHIFTS.read(i as usize);
            let packed_shift = NBH_SHIFT_SHARED.read(i as usize);
            let shared_id = (packed_cell_index_in_block + packed_shift) as usize;
            let cell_data = shared_nodes_vel_mass[shared_id];
            let cell_cdf = shared_nodes_cdf[shared_id];
            let is_compatible =
                affinities_are_compatible(particle_cdf.affinity, cell_cdf.affinities);

            #[cfg(feature = "dim2")]
            let dpt = ref_elt_pos_minus_particle_pos
                + Vec2::new(shift.x as f32, shift.y as f32) * cell_width;
            #[cfg(feature = "dim3")]
            let dpt = ref_elt_pos_minus_particle_pos
                + Vec3::new(shift.x as f32, shift.y as f32, shift.z as f32) * cell_width;

            let mut cpic_cell_data = cell_data;

            if !is_compatible {
                cpic_cell_data = shared_nodes_vel_mass_incompatible[shared_id];

                if cell_cdf.closest_id != NONE {
                    let body_vel = body_vels.read(cell_cdf.closest_id as usize);
                    let body_com = body_mprops.at(cell_cdf.closest_id as usize).com;
                    let body_material = body_materials.read(cell_cdf.closest_id as usize);
                    let cell_center = dpt + particle_pos.pt;
                    let body_pt_vel = velocity_at_point(body_com, &body_vel, cell_center);

                    let cpic_vel = vector_part(cpic_cell_data);
                    let particle_ghost_vel = body_pt_vel
                        + body_material
                            .project_velocity(cpic_vel - body_pt_vel, particle_cdf.normal);
                    cpic_cell_data =
                        vector_plus_one(particle_ghost_vel, scalar_part(cpic_cell_data));
                }
            }

            #[cfg(feature = "dim2")]
            let weight = vec3_extract(w[0], shift.x) * vec3_extract(w[1], shift.y);
            #[cfg(feature = "dim3")]
            let weight = vec3_extract(w[0], shift.x)
                * vec3_extract(w[1], shift.y)
                * vec3_extract(w[2], shift.z);

            let cpic_vel = vector_part(cpic_cell_data);

            momentum_velocity_mass += cpic_cell_data * weight;
            velocity_gradient += outer_product(cpic_vel, dpt) * (weight * inv_d);
            vel_grad_det += weight * inv_d * cpic_vel.dot(dpt);
        }

        // Accumulate rigid body velocities for all affinity-linked colliders.
        for i_collider in 0..16 {
            if affinity_bit(i_collider as u32, particle_cdf.affinity) {
                let body_vel = body_vels.read(i_collider as usize);
                let body_com = body_mprops.at(i_collider as usize).com;
                rigid_vel += velocity_at_point(body_com, &body_vel, particle_pos.pt);
            }
        }
    }

    particles_dyn.at_mut(particle_id as usize).cdf.rigid_vel = rigid_vel;
    // Set the particle velocity, and store the velocity gradient into the affine matrix.
    // The rest will be dealt with in the particle update kernel(s).
    particles_dyn.at_mut(particle_id as usize).affine =
        PaddedMatrix::add_padding(velocity_gradient);
    particles_dyn.at_mut(particle_id as usize).vel_grad_det = vel_grad_det;
    particles_dyn.at_mut(particle_id as usize).velocity = vector_part(momentum_velocity_mass);
}

/*
 * GPU entry points.
 */

/// GPU kernel: G2P transfer (2D).
///
/// Transfers grid node velocities back to particles using APIC interpolation.
/// Dispatched with one workgroup per active block.
#[spirv_bindgen]
#[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8))))]
#[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
#[unroll_for_loops]
pub fn gpu_g2p(
    #[spirv(workgroup_id)] block_id: spirv_std::glam::UVec3,
    #[spirv(local_invocation_id)] tid: spirv_std::glam::UVec3,
    #[spirv(local_invocation_index)] tid_flat: u32,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &SimulationParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] grid_data: &[Grid],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] hmap_entries: &[GridHashMapEntry],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] active_blocks: &[ActiveBlockHeader],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] nodes: &[Node],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] sorted_particle_ids: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] particles_pos: &[Position],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] particles_dyn: &mut [Dynamics],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] body_vels: &[BodyVelocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] body_mprops: &[BodyMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 10)]
    body_materials: &[BoundaryCondition],
    // Shared memory.
    #[spirv(workgroup)] shared_nodes_vel_mass: &mut [VectorPlusOne; NUM_SHARED_CELLS],
    #[spirv(workgroup)] shared_nodes_vel_mass_incompatible: &mut [VectorPlusOne; NUM_SHARED_CELLS],
    #[spirv(workgroup)] shared_nodes_cdf: &mut [NodeCdf; NUM_SHARED_CELLS],
) {
    let bid = block_id.x;
    // Force copy of the virtual ID (naga bug workaround).
    let vid_ = active_blocks.at(bid as usize).virtual_id.id;
    let vid = BlockVirtualId::new(vid_);

    // Block -> shared memory transfer.
    global_shared_memory_transfers(
        grid_data,
        hmap_entries,
        nodes,
        tid,
        vid,
        shared_nodes_vel_mass,
        shared_nodes_vel_mass_incompatible,
        shared_nodes_cdf,
    );

    // Sync after shared memory initialization.
    workgroup_memory_barrier_with_group_sync();

    // Particle update. Runs g2p on shared memory only.
    let first_particle = active_blocks.at(bid as usize).first_particle;
    let max_particle_id = first_particle + active_blocks.at(bid as usize).num_particles;

    let num_block_particles = max_particle_id - first_particle;
    let max_iters = (num_block_particles + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    let mut sorted_particle_id = first_particle + tid_flat;
    for _ in 0..max_iters {
        if sorted_particle_id >= max_particle_id {
            break;
        }
        let particle_id = sorted_particle_ids.read(sorted_particle_id as usize);
        particle_g2p(
            body_vels,
            body_mprops,
            body_materials,
            particles_pos,
            particles_dyn,
            particle_id,
            grid_data.at(0).cell_width,
            params.dt,
            shared_nodes_vel_mass,
            shared_nodes_vel_mass_incompatible,
            shared_nodes_cdf,
        );
        sorted_particle_id += WORKGROUP_SIZE;
    }
}

/*
 * Shared memory flatten helpers for G2P.
 * Note: different from P2G -- no shift subtraction since the truncated blocks
 * are in the higher-index quadrants.
 */

#[cfg(feature = "dim2")]
#[inline]
fn flatten_shared_index(x: u32, y: u32) -> u32 {
    x + y * 10
}

#[cfg(feature = "dim3")]
#[inline]
fn flatten_shared_index(x: u32, y: u32, z: u32) -> u32 {
    x + y * 6 + z * 6 * 6
}

/*
 * Outer product helper.
 */

#[cfg(feature = "dim2")]
#[inline]
fn outer_product(a: Vec2, b: Vec2) -> Mat2 {
    Mat2::from_cols(a * b.x, a * b.y)
}

#[cfg(feature = "dim3")]
#[inline]
fn outer_product(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}
