//! Particle-to-Grid (P2G) transfer kernel.
//!
//! This is the core MPM kernel that transfers particle data (momentum, mass, affine matrix)
//! onto the grid nodes. Uses shared memory, workgroup barriers, and linked list traversal
//! to efficiently gather particle contributions from neighboring nodes.
//!
//! The kernel also handles CPIC (Compatible Particle-In-Cell) affinity checks:
//! particles that are incompatible with a node (different side of a collider) contribute
//! to the node's `incompatible` momentum field instead, and impulses are computed for
//! the rigid body coupling.

use crate::grid::grid::*;
use crate::grid::kernel::*;
use crate::nexus_rbd_shaders::dynamics::{Velocity as BodyVelocity, velocity_at_point};
use crate::solver::boundary_condition::BoundaryCondition;
use crate::solver::particle::{Kinematics, Position, dir_to_associated_grid_node};
use crate::{AngVector, Matrix, PaddingExt, TWO_WAYS_COUPLING_ENABLED, Vector};
use core::ops::Range;
use glamx::*;
use khal_std::arch::{
    atomic_add_i32, atomic_load_u32_workgroup, atomic_max_u32_workgroup, atomic_store_u32_workgroup,
};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::{
    arch::workgroup_memory_barrier_with_group_sync,
    macros::{spirv, spirv_bindgen},
};
use unroll::unroll_for_loops;
/*
 * Shared memory layout constants.
 */

/// Number of shared memory cells.
/// In 2D: (8+2)^2 = 100. In 3D: (4+2)^3 = 216.
#[cfg(feature = "dim2")]
const NUM_SHARED_CELLS: usize = 10 * 10;
#[cfg(feature = "dim3")]
const NUM_SHARED_CELLS: usize = 6 * 6 * 6;

/// A shared-memory node entry: particle ID and global node ID.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct SharedNode {
    particle_id: u32,
    global_id: u32,
}

/// Result of a single P2G step (one particle iteration).
#[derive(Clone, Copy)]
struct P2GStepResult {
    new_momentum_velocity: Vector,
    new_mass: f32,
    new_momentum_velocity_incompatible: Vector,
    new_mass_incompatible: f32,
    impulse: Vector,
    ang_impulse: AngVector,
}

impl P2GStepResult {
    fn zero() -> Self {
        Self {
            new_momentum_velocity: Vector::ZERO,
            new_mass: 0.0,
            new_momentum_velocity_incompatible: Vector::ZERO,
            new_mass_incompatible: 0.0,
            impulse: Vector::ZERO,
            #[cfg(feature = "dim2")]
            ang_impulse: 0.0,
            #[cfg(feature = "dim3")]
            ang_impulse: Vec3::ZERO,
        }
    }
}

/// Integer impulse atomic struct for accumulating impulses across threads.
///
/// Uses integer atomics to avoid floating-point atomic limitations on GPU.
/// The COM (center of mass) is stored alongside to reduce binding count.
#[derive(Clone, Copy, Default)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct IntegerImpulseAtomic {
    pub com: Vector,
    pub linear_x: i32,
    pub linear_y: i32,
    #[cfg(feature = "dim3")]
    pub linear_z: i32,
    #[cfg(feature = "dim3")]
    pub _padding_a: i32,
    #[cfg(feature = "dim2")]
    pub angular: i32,
    #[cfg(feature = "dim2")]
    pub _padding: i32,
    #[cfg(feature = "dim3")]
    pub angular_x: i32,
    #[cfg(feature = "dim3")]
    pub angular_y: i32,
    #[cfg(feature = "dim3")]
    pub angular_z: i32,
    #[cfg(feature = "dim3")]
    pub _padding_b: [i32; 2],
}

const FLOAT_TO_INT_FACTOR: f32 = 1e5;

/*
 * P2G step: compute the contribution of all particles in the neighborhood
 * to a single grid node.
 */

#[inline]
fn p2g_step<const USE_CPIC: bool>(
    body_vels: &[BodyVelocity],
    body_impulses: &mut [IntegerImpulseAtomic],
    body_materials: &[BoundaryCondition],
    packed_cell_index_in_block: u32,
    cell_width: f32,
    node_affinity: AffinityBits,
    collider_id: u32,
    shared_pos: &[Position; NUM_SHARED_CELLS],
    shared_vel_mass: &[(Vector, f32); NUM_SHARED_CELLS],
    // shared_mass: &[f32; NUM_SHARED_CELLS],
    shared_affine: &[Matrix; NUM_SHARED_CELLS],
    shared_affinities: &[AffinityBits; NUM_SHARED_CELLS],
    shared_normals: &[Vector; NUM_SHARED_CELLS],
) -> P2GStepResult {
    // Shift to reach the first node with particles contributing to the current cell's data.
    #[cfg(feature = "dim2")]
    let bottommost_contributing_node = flatten_shared_shift(2, 2);
    #[cfg(feature = "dim3")]
    let bottommost_contributing_node = flatten_shared_shift(2, 2, 2);

    let mut new_momentum_velocity = Vector::ZERO;
    let mut new_mass = 0.0f32;
    let mut new_momentum_velocity_incompatible = Vector::ZERO;
    let mut new_mass_incompatible = 0.0f32;
    let mut impulse = Vector::ZERO;
    #[cfg(feature = "dim2")]
    let mut ang_impulse: AngVector = 0.0;
    #[cfg(feature = "dim3")]
    let mut ang_impulse: AngVector = Vec3::ZERO;

    for i in 0..NBH_LEN as u32 {
        let packed_shift = NBH_SHIFT_SHARED.read(i as usize);
        let nbh_shared_index =
            (packed_cell_index_in_block - bottommost_contributing_node + packed_shift) as usize;
        let particle_pos = shared_pos.read(nbh_shared_index);
        let (particle_vel, particle_mass) = shared_vel_mass.read(nbh_shared_index);
        // let particle_mass = shared_mass.read(nbh_shared_index);
        let particle_affine = shared_affine.read(nbh_shared_index);
        let ref_elt_pos_minus_particle_pos = dir_to_associated_grid_node(&particle_pos, cell_width);
        let w = QuadraticKernel::precompute_weights(ref_elt_pos_minus_particle_pos, cell_width);

        let shift = NBH_SHIFTS.read(i as usize);

        #[cfg(feature = "dim2")]
        let inv_shift = UVec2::new(2, 2) - shift;
        #[cfg(feature = "dim2")]
        let momentum = particle_vel * particle_mass;
        #[cfg(feature = "dim2")]
        let dpt = ref_elt_pos_minus_particle_pos
            + Vec2::new(inv_shift.x as f32, inv_shift.y as f32) * cell_width;
        #[cfg(feature = "dim2")]
        let weight = vec3_extract(w[0], inv_shift.x) * vec3_extract(w[1], inv_shift.y);

        #[cfg(feature = "dim3")]
        let inv_shift = UVec3::new(2, 2, 2) - shift;
        #[cfg(feature = "dim3")]
        let momentum = particle_vel * particle_mass;
        #[cfg(feature = "dim3")]
        let dpt = ref_elt_pos_minus_particle_pos
            + Vec3::new(inv_shift.x as f32, inv_shift.y as f32, inv_shift.z as f32) * cell_width;
        #[cfg(feature = "dim3")]
        let weight = vec3_extract(w[0], inv_shift.x)
            * vec3_extract(w[1], inv_shift.y)
            * vec3_extract(w[2], inv_shift.z);

        let vel_contribution = (particle_affine * dpt + momentum) * weight;
        let mass_contribution = particle_mass * weight;

        if USE_CPIC {
            let particle_affinity = shared_affinities.read(nbh_shared_index);
            if !particle_affinity.is_compatible(node_affinity) {
                if TWO_WAYS_COUPLING_ENABLED && collider_id != NONE {
                    let particle_normal = shared_normals.read(nbh_shared_index);
                    let body_vel = body_vels.read(collider_id as usize);
                    let body_com = body_impulses.at(collider_id as usize).com;
                    let body_material = body_materials.read(collider_id as usize);
                    let cell_center = dpt + particle_pos.pt;
                    let body_pt_vel = velocity_at_point(body_com, &body_vel, cell_center);
                    let particle_ghost_vel = body_pt_vel
                        + body_material
                            .project_velocity(particle_vel - body_pt_vel, particle_normal);
                    let delta_impulse =
                        (particle_vel - particle_ghost_vel) * (weight * particle_mass);

                    let lever_arm = body_com - cell_center;

                    #[cfg(feature = "dim2")]
                    {
                        let delta_ang_impulse =
                            delta_impulse.dot(Vec2::new(lever_arm.y, -lever_arm.x));
                        ang_impulse += delta_ang_impulse;
                    }
                    #[cfg(feature = "dim3")]
                    {
                        let delta_ang_impulse = delta_impulse.cross(lever_arm);
                        ang_impulse += delta_ang_impulse;
                    }

                    impulse += delta_impulse;
                }

                new_momentum_velocity_incompatible += vel_contribution;
                new_mass_incompatible += mass_contribution;
            } else {
                new_momentum_velocity += vel_contribution;
                new_mass += mass_contribution;
            }
        } else {
            new_momentum_velocity += vel_contribution;
            new_mass += mass_contribution;
        }
    }

    P2GStepResult {
        new_momentum_velocity,
        new_mass,
        new_momentum_velocity_incompatible,
        new_mass_incompatible,
        impulse,
        ang_impulse,
    }
}

/*
 * fetch_max_linked_lists_length: compute the maximum linked list length across
 * all nodes in the 2x2 (2D) or 2x2x2 (3D) neighborhood of the current block.
 */

#[inline]
#[unroll_for_loops]
fn fetch_max_linked_lists_length(
    grid: &Grid,
    hmap_entries: &[GridHashMapEntry],
    nodes_linked_lists: &[NodeLinkedList],
    tid: khal_std::glamx::UVec3,
    active_block_vid: BlockVirtualId,
    _bid: u32,
    max_linked_list_length: &mut u32,
) {
    #[cfg(feature = "dim2")]
    let base_block_pos_int = active_block_vid.id - IVec2::new(1, 1);
    #[cfg(feature = "dim3")]
    let base_block_pos_int = active_block_vid.id - IVec3::new(1, 1, 1);

    for i_loop in 0..2 {
        for j_loop in 0..2 {
            #[cfg(feature = "dim2")]
            {
                if !((i_loop == 0 && tid.x < 6) || (j_loop == 0 && tid.y < 6)) {
                    let octant = UVec2::new(i_loop as u32, j_loop as u32);
                    let octant_hid = grid.find_block_header_id(
                        hmap_entries,
                        &BlockVirtualId {
                            id: base_block_pos_int + IVec2::new(octant.x as i32, octant.y as i32),
                        },
                    );
                    if octant_hid.id != NONE {
                        let global_chunk_id = octant_hid.physical_id();
                        let tid_xy = UVec2::new(tid.x, tid.y);
                        let global_node_id = global_chunk_id.node_id(tid_xy);
                        let len = nodes_linked_lists.at(global_node_id.id as usize).len;
                        atomic_max_u32_workgroup(max_linked_list_length, len);
                    }
                }
            }
            #[cfg(feature = "dim3")]
            for k_loop in 0..2 {
                if !((i_loop == 0 && tid.x < 2)
                    || (j_loop == 0 && tid.y < 2)
                    || (k_loop == 0 && tid.z < 2))
                {
                    let octant = UVec3::new(i_loop as u32, j_loop as u32, k_loop as u32);
                    let octant_hid = grid.find_block_header_id(
                        hmap_entries,
                        &BlockVirtualId {
                            id: base_block_pos_int
                                + IVec3::new(octant.x as i32, octant.y as i32, octant.z as i32),
                            padding: 0,
                        },
                    );
                    if octant_hid.id != NONE {
                        let global_chunk_id = octant_hid.physical_id();
                        let tid_xyz = UVec3::new(tid.x, tid.y, tid.z);
                        let global_node_id = global_chunk_id.node_id(tid_xyz);
                        let len = nodes_linked_lists.at(global_node_id.id as usize).len;
                        atomic_max_u32_workgroup(max_linked_list_length, len);
                    }
                }
            }
        }
    }
}

/*
 * fetch_nodes: load the initial linked-list heads and global node IDs into shared memory.
 */

#[inline]
#[unroll_for_loops]
fn fetch_nodes(
    grid: &Grid,
    hmap_entries: &[GridHashMapEntry],
    nodes_linked_lists: &[NodeLinkedList],
    tid: khal_std::glamx::UVec3,
    active_block_vid: BlockVirtualId,
    _bid: u32,
    shared_nodes: &mut [SharedNode; NUM_SHARED_CELLS],
) {
    #[cfg(feature = "dim2")]
    let base_block_pos_int = active_block_vid.id - IVec2::new(1, 1);
    #[cfg(feature = "dim3")]
    let base_block_pos_int = active_block_vid.id - IVec3::new(1, 1, 1);

    for i_loop in 0..2 {
        for j_loop in 0..2 {
            #[cfg(feature = "dim2")]
            {
                if !((i_loop == 0 && tid.x < 6) || (j_loop == 0 && tid.y < 6)) {
                    let octant = UVec2::new(i_loop as u32, j_loop as u32);
                    let octant_hid = grid.find_block_header_id(
                        hmap_entries,
                        &BlockVirtualId {
                            id: base_block_pos_int + IVec2::new(octant.x as i32, octant.y as i32),
                        },
                    );
                    let shared_index = octant * 8 + UVec2::new(tid.x, tid.y);
                    let shared_node_index =
                        flatten_shared_index(shared_index.x, shared_index.y) as usize;

                    if octant_hid.id != NONE {
                        let global_chunk_id = octant_hid.physical_id();
                        let tid_xy = UVec2::new(tid.x, tid.y);
                        let global_node_id = global_chunk_id.node_id(tid_xy);
                        let particle_id = nodes_linked_lists.at(global_node_id.id as usize).head;
                        shared_nodes.at_mut(shared_node_index).particle_id = particle_id;
                        shared_nodes.at_mut(shared_node_index).global_id = global_node_id.id;
                    } else {
                        shared_nodes.at_mut(shared_node_index).particle_id = NONE;
                    }
                }
            }
            #[cfg(feature = "dim3")]
            for k_loop in 0..2 {
                if !((i_loop == 0 && tid.x < 2)
                    || (j_loop == 0 && tid.y < 2)
                    || (k_loop == 0 && tid.z < 2))
                {
                    let octant = UVec3::new(i_loop as u32, j_loop as u32, k_loop as u32);
                    let octant_hid = grid.find_block_header_id(
                        hmap_entries,
                        &BlockVirtualId {
                            id: base_block_pos_int
                                + IVec3::new(octant.x as i32, octant.y as i32, octant.z as i32),
                            padding: 0,
                        },
                    );
                    let tid_xyz = UVec3::new(tid.x, tid.y, tid.z);
                    let shared_index = octant * 4 + tid_xyz;
                    let shared_node_index =
                        flatten_shared_index(shared_index.x, shared_index.y, shared_index.z)
                            as usize;

                    if octant_hid.id != NONE {
                        let global_chunk_id = octant_hid.physical_id();
                        let global_node_id = global_chunk_id.node_id(tid_xyz);
                        let particle_id = nodes_linked_lists.at(global_node_id.id as usize).head;
                        shared_nodes.at_mut(shared_node_index).particle_id = particle_id;
                        shared_nodes.at_mut(shared_node_index).global_id = global_node_id.id;
                    } else {
                        shared_nodes.at_mut(shared_node_index).particle_id = NONE;
                    }
                }
            }
        }
    }
}

/*
 * fetch_next_particle: advance each shared-memory slot's linked list by one,
 * loading the current particle's data into shared memory arrays.
 */

#[inline]
#[allow(clippy::too_many_arguments)]
#[unroll_for_loops]
fn fetch_next_particle<const USE_CPIC: bool>(
    particles_pos: &[Position],
    particles_kin: &[Kinematics],
    particle_node_linked_lists: &[u32],
    tid: khal_std::glamx::UVec3,
    shared_nodes: &mut [SharedNode; NUM_SHARED_CELLS],
    shared_pos: &mut [Position; NUM_SHARED_CELLS],
    shared_vel_mass: &mut [(Vector, f32); NUM_SHARED_CELLS],
    shared_affine: &mut [Matrix; NUM_SHARED_CELLS],
    shared_affinities: &mut [AffinityBits; NUM_SHARED_CELLS],
    shared_normals: &mut [Vector; NUM_SHARED_CELLS],
) {
    for i_loop in 0..2 {
        for j_loop in 0..2 {
            for k_loop in 0..2 {
                #[cfg(feature = "dim2")]
                let skip = k_loop != 0 || (i_loop == 0 && tid.x < 6) || (j_loop == 0 && tid.y < 6);
                #[cfg(feature = "dim3")]
                let skip = (i_loop == 0 && tid.x < 2)
                    || (j_loop == 0 && tid.y < 2)
                    || (k_loop == 0 && tid.z < 2);

                if !skip {
                    #[cfg(feature = "dim2")]
                    let shared_flat_index = {
                        let octant = UVec2::new(i_loop as u32, j_loop as u32);
                        let shared_index = octant * 8 + UVec2::new(tid.x, tid.y);
                        flatten_shared_index(shared_index.x, shared_index.y) as usize
                    };
                    #[cfg(feature = "dim3")]
                    let shared_flat_index = {
                        let octant = UVec3::new(i_loop as u32, j_loop as u32, k_loop as u32);
                        let tid_xyz = UVec3::new(tid.x, tid.y, tid.z);
                        let shared_index = octant * 4 + tid_xyz;
                        flatten_shared_index(shared_index.x, shared_index.y, shared_index.z)
                            as usize
                    };

                    let curr_particle_id = shared_nodes.at(shared_flat_index).particle_id;

                    if curr_particle_id != NONE
                        && particles_kin.at(curr_particle_id as usize).enabled != 0
                    {
                        let pkin = particles_kin.read(curr_particle_id as usize);

                        if USE_CPIC {
                            shared_affinities.write(shared_flat_index, pkin.cdf.affinity);
                            shared_normals.write(shared_flat_index, pkin.cdf.normal);
                        }
                        shared_pos.write(
                            shared_flat_index,
                            particles_pos.read(curr_particle_id as usize),
                        );
                        shared_affine.write(shared_flat_index, pkin.affine.remove_padding());
                        shared_vel_mass.write(shared_flat_index, (pkin.velocity, pkin.mass));
                    } else {
                        if USE_CPIC {
                            shared_affinities.write(shared_flat_index, AffinityBits::EMPTY);
                            shared_normals.write(shared_flat_index, Vector::ZERO);
                        }

                        shared_pos.at_mut(shared_flat_index).pt = Vector::ZERO;
                        shared_affine.write(shared_flat_index, Matrix::ZERO);
                        shared_vel_mass.write(shared_flat_index, (Vector::ZERO, 0.0));
                    }

                    if curr_particle_id != NONE {
                        // Advance the linked list even if the particle is disabled.
                        let next_particle_id =
                            particle_node_linked_lists.read(curr_particle_id as usize);
                        shared_nodes.at_mut(shared_flat_index).particle_id = next_particle_id;
                    }
                }
            }
        }
    }
}

/*
 * GPU entry points.
 */
/// GPU kernel: P2G transfer (2D).
///
/// Transfers particle momentum, mass, and affine contributions onto grid nodes.
/// Handles CPIC compatibility checks and rigid body impulse accumulation.
///
/// Dispatched with one workgroup per active block.
pub fn gpu_p2g_generic<const USE_CPIC: bool>(
    block_id: khal_std::glamx::UVec3,
    tid: khal_std::glamx::UVec3,
    tid_flat: u32,
    grid: &Grid,
    hmap_entries: &[GridHashMapEntry],
    active_blocks: &[ActiveBlockHeader],
    nodes_linked_lists: &[NodeLinkedList],
    particle_node_linked_lists: &[u32],
    particles_pos: &[Position],
    particles_kin: &[Kinematics],
    nodes: &mut [Node],
    body_vels: &[BodyVelocity],
    body_impulses: &mut [IntegerImpulseAtomic],
    body_materials: &[BoundaryCondition],
    // Shared memory arrays.
    shared_vel_mass: &mut [(Vector, f32); NUM_SHARED_CELLS],
    shared_affine: &mut [Matrix; NUM_SHARED_CELLS],
    shared_nodes: &mut [SharedNode; NUM_SHARED_CELLS],
    shared_pos: &mut [Position; NUM_SHARED_CELLS],
    shared_affinities: &mut [AffinityBits; NUM_SHARED_CELLS],
    shared_normals: &mut [Vector; NUM_SHARED_CELLS],
    max_linked_list_length: &mut u32,
    max_linked_list_length_uniform: &mut u32,
) {
    let bid = block_id.x;
    // Force copy of the virtual ID (naga bug workaround).
    let vid_ = active_blocks.at(bid as usize).virtual_id.id;
    let vid = BlockVirtualId::new(vid_);

    // Initialize max linked list length to 0.
    if tid_flat == 0 {
        atomic_store_u32_workgroup(max_linked_list_length, 0);
    }

    workgroup_memory_barrier_with_group_sync();
    fetch_max_linked_lists_length(
        grid,
        hmap_entries,
        nodes_linked_lists,
        tid,
        vid,
        bid,
        max_linked_list_length,
    );
    workgroup_memory_barrier_with_group_sync();

    *max_linked_list_length_uniform = atomic_load_u32_workgroup(max_linked_list_length);

    // Block -> shared memory transfer.
    fetch_nodes(
        grid,
        hmap_entries,
        nodes_linked_lists,
        tid,
        vid,
        bid,
        shared_nodes,
    );

    // Compute the packed cell index for the current thread's node.
    #[cfg(feature = "dim2")]
    let packed_cell_index_in_block = flatten_shared_index(tid.x + 8, tid.y + 8);
    #[cfg(feature = "dim3")]
    let packed_cell_index_in_block = flatten_shared_index(tid.x + 4, tid.y + 4, tid.z + 4);

    let global_id = shared_nodes
        .at(packed_cell_index_in_block as usize)
        .global_id;
    let node_affinities = nodes.at(global_id as usize).cdf.affinities;
    let collider_id = if USE_CPIC {
        nodes.at(global_id as usize).cdf.closest_id
    } else {
        0
    };
    let mut total_result = P2GStepResult::zero();

    // Iterate through linked lists with uniform control flow.
    let len = *max_linked_list_length_uniform;

    #[cfg(feature = "web-compat")] // Need to cap the iteration count on the web.
    const K_RANGE: Range<u32> = 0..64;
    #[cfg(not(feature = "web-compat"))]
    let K_RANGE: Range<u32> = 0..len;

    for _k in K_RANGE {
        #[cfg(feature = "web-compat")]
        let ok = _k < len;
        #[cfg(not(feature = "web-compat"))]
        const ok: bool = true;

        workgroup_memory_barrier_with_group_sync();
        if ok {
            fetch_next_particle::<USE_CPIC>(
                particles_pos,
                particles_kin,
                particle_node_linked_lists,
                tid,
                shared_nodes,
                shared_pos,
                shared_vel_mass,
                shared_affine,
                shared_affinities,
                shared_normals,
            );
        }

        workgroup_memory_barrier_with_group_sync();

        if ok {
            let partial_result = p2g_step::<USE_CPIC>(
                body_vels,
                body_impulses,
                body_materials,
                packed_cell_index_in_block,
                grid.cell_width,
                node_affinities,
                collider_id,
                shared_pos,
                shared_vel_mass,
                shared_affine,
                shared_affinities,
                shared_normals,
            );
            total_result.new_momentum_velocity += partial_result.new_momentum_velocity;
            total_result.new_mass += partial_result.new_mass;

            if USE_CPIC {
                total_result.new_momentum_velocity_incompatible +=
                    partial_result.new_momentum_velocity_incompatible;
                total_result.new_mass_incompatible += partial_result.new_mass_incompatible;
                total_result.impulse += partial_result.impulse;
                total_result.ang_impulse += partial_result.ang_impulse;
            }
        }
    }

    // Write the node state to global memory.
    nodes.at_mut(global_id as usize).momentum_velocity = total_result.new_momentum_velocity;
    nodes.at_mut(global_id as usize).mass = total_result.new_mass;

    if USE_CPIC {
        nodes
            .at_mut(global_id as usize)
            .momentum_velocity_incompatible = total_result.new_momentum_velocity_incompatible;
        nodes.at_mut(global_id as usize).mass_incompatible = total_result.new_mass_incompatible;

        // Apply the impulse to the closest body using integer atomics.
        if TWO_WAYS_COUPLING_ENABLED && collider_id != NONE {
            let ci = collider_id as usize;
            #[cfg(feature = "dim2")]
            {
                atomic_add_i32(
                    &mut body_impulses.at_mut(ci).linear_x,
                    flt2int(total_result.impulse.x),
                );
                atomic_add_i32(
                    &mut body_impulses.at_mut(ci).linear_y,
                    flt2int(total_result.impulse.y),
                );
                atomic_add_i32(
                    &mut body_impulses.at_mut(ci).angular,
                    flt2int(total_result.ang_impulse),
                );
            }
            #[cfg(feature = "dim3")]
            {
                atomic_add_i32(
                    &mut body_impulses.at_mut(ci).linear_x,
                    flt2int(total_result.impulse.x),
                );
                atomic_add_i32(
                    &mut body_impulses.at_mut(ci).linear_y,
                    flt2int(total_result.impulse.y),
                );
                atomic_add_i32(
                    &mut body_impulses.at_mut(ci).linear_z,
                    flt2int(total_result.impulse.z),
                );
                atomic_add_i32(
                    &mut body_impulses.at_mut(ci).angular_x,
                    flt2int(total_result.ang_impulse.x),
                );
                atomic_add_i32(
                    &mut body_impulses.at_mut(ci).angular_y,
                    flt2int(total_result.ang_impulse.y),
                );
                atomic_add_i32(
                    &mut body_impulses.at_mut(ci).angular_z,
                    flt2int(total_result.ang_impulse.z),
                );
            }
        }
    }
}

/// Converts a float to an integer for atomic accumulation.
#[inline]
fn flt2int(flt: f32) -> i32 {
    (flt * FLOAT_TO_INT_FACTOR) as i32
}

/*
 * Shared memory flatten helpers
 */

#[cfg(feature = "dim2")]
#[inline]
fn flatten_shared_index(x: u32, y: u32) -> u32 {
    (x - 6) + (y - 6) * 10
}

#[cfg(feature = "dim2")]
#[inline]
fn flatten_shared_shift(x: u32, y: u32) -> u32 {
    x + y * 10
}

#[cfg(feature = "dim3")]
#[inline]
fn flatten_shared_index(x: u32, y: u32, z: u32) -> u32 {
    (x - 2) + (y - 2) * 6 + (z - 2) * 6 * 6
}

#[cfg(feature = "dim3")]
#[inline]
fn flatten_shared_shift(x: u32, y: u32, z: u32) -> u32 {
    x + y * 6 + z * 6 * 6
}

/*
 Entrypoint specializations (with our without CPIC)
*/

#[spirv_bindgen]
#[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8))))]
#[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
pub fn gpu_p2g(
    #[spirv(workgroup_id)] block_id: khal_std::glamx::UVec3,
    #[spirv(local_invocation_id)] tid: khal_std::glamx::UVec3,
    #[spirv(local_invocation_index)] tid_flat: u32,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] hmap_entries: &[GridHashMapEntry],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] active_blocks: &[ActiveBlockHeader],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    nodes_linked_lists: &[NodeLinkedList],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] particle_node_linked_lists: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] particles_pos: &[Position],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] particles_kin: &[Kinematics],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] nodes: &mut [Node],
    // Shared memory arrays.
    // TODO PERF: analyze shared memory access patterns to avoid bank conflicts
    // (https://feldmann.nyc/blog/smem-microbenchmarks)
    #[spirv(workgroup)] shared_vel_mass: &mut [(Vector, f32); NUM_SHARED_CELLS], // P2G runs 10ms slower in the 3D sand demo unless we group vel and mass.
    #[spirv(workgroup)] shared_affine: &mut [Matrix; NUM_SHARED_CELLS],
    #[spirv(workgroup)] shared_nodes: &mut [SharedNode; NUM_SHARED_CELLS],
    #[spirv(workgroup)] shared_pos: &mut [Position; NUM_SHARED_CELLS],
    #[spirv(workgroup)] shared_affinities: &mut [AffinityBits; NUM_SHARED_CELLS],
    #[spirv(workgroup)] shared_normals: &mut [Vector; NUM_SHARED_CELLS],
    #[spirv(workgroup)] max_linked_list_length: &mut u32,
    #[spirv(workgroup)] max_linked_list_length_uniform: &mut u32,
) {
    gpu_p2g_generic::<false>(
        block_id,
        tid,
        tid_flat,
        grid,
        hmap_entries,
        active_blocks,
        nodes_linked_lists,
        particle_node_linked_lists,
        particles_pos,
        particles_kin,
        nodes,
        &[],
        &mut [],
        &[],
        shared_vel_mass,
        shared_affine,
        shared_nodes,
        shared_pos,
        shared_affinities,
        shared_normals,
        max_linked_list_length,
        max_linked_list_length_uniform,
    );
}

/// GPU kernel: P2G transfer (2D).
///
/// Transfers particle momentum, mass, and affine contributions onto grid nodes.
/// Handles CPIC compatibility checks and rigid body impulse accumulation.
///
/// Dispatched with one workgroup per active block.
#[spirv_bindgen]
#[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8))))]
#[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
pub fn gpu_p2g_cpic(
    #[spirv(workgroup_id)] block_id: khal_std::glamx::UVec3,
    #[spirv(local_invocation_id)] tid: khal_std::glamx::UVec3,
    #[spirv(local_invocation_index)] tid_flat: u32,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] hmap_entries: &[GridHashMapEntry],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] active_blocks: &[ActiveBlockHeader],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    nodes_linked_lists: &[NodeLinkedList],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] particle_node_linked_lists: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] particles_pos: &[Position],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] particles_kin: &[Kinematics],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] nodes: &mut [Node],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] body_vels: &[BodyVelocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)]
    body_materials: &[BoundaryCondition],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 10)]
    body_impulses: &mut [IntegerImpulseAtomic],
    // Shared memory arrays.
    // TODO PERF: analyze shared memory access patterns to avoid bank conflicts
    // (https://feldmann.nyc/blog/smem-microbenchmarks)
    #[spirv(workgroup)] shared_vel_mass: &mut [(Vector, f32); NUM_SHARED_CELLS], // P2G runs 10ms slower in the 3D sand demo unless we group vel and mass.
    #[spirv(workgroup)] shared_affine: &mut [Matrix; NUM_SHARED_CELLS],
    #[spirv(workgroup)] shared_nodes: &mut [SharedNode; NUM_SHARED_CELLS],
    #[spirv(workgroup)] shared_pos: &mut [Position; NUM_SHARED_CELLS],
    #[spirv(workgroup)] shared_affinities: &mut [AffinityBits; NUM_SHARED_CELLS],
    #[spirv(workgroup)] shared_normals: &mut [Vector; NUM_SHARED_CELLS],
    #[spirv(workgroup)] max_linked_list_length: &mut u32,
    #[spirv(workgroup)] max_linked_list_length_uniform: &mut u32,
) {
    gpu_p2g_generic::<true>(
        block_id,
        tid,
        tid_flat,
        grid,
        hmap_entries,
        active_blocks,
        nodes_linked_lists,
        particle_node_linked_lists,
        particles_pos,
        particles_kin,
        nodes,
        body_vels,
        body_impulses,
        body_materials,
        shared_vel_mass,
        shared_affine,
        shared_nodes,
        shared_pos,
        shared_affinities,
        shared_normals,
        max_linked_list_length,
        max_linked_list_length_uniform,
    );
}
