//! Grid-to-Particle CDF (Contact Distance Field) transfer kernel.
//!
//! This kernel transfers the grid-level contact distance field data back to particles.
//! It uses MLS (Moving Least Squares) reconstruction to compute per-particle signed
//! distances and contact normals from the surrounding grid nodes' CDF data.
//!
//! The kernel also determines per-particle affinity bits and sign bits used by CPIC
//! for compatible particle-grid transfers in subsequent timesteps.
//!
//! Corresponds to `g2p_cdf.slang`.

use crate::grid::grid::*;
use crate::grid::kernel::*;
use crate::solver::params::SimulationParams;
use crate::solver::particle::{
    associated_cell_index_in_block_off_by_one, dir_to_associated_grid_node, Cdf, Position,
};
use crate::{abs, MaybeIndexUnchecked, Vector};
use crunchy::unroll;
use glamx::*;
use khal_derive::spirv_bindgen;
use spirv_std::arch::workgroup_memory_barrier_with_group_sync;
use spirv_std::spirv;
use unroll::unroll_for_loops;
/*
 * Constants.
 */

#[cfg(feature = "dim2")]
const NUM_SHARED_CELLS: usize = 10 * 10;
#[cfg(feature = "dim3")]
const NUM_SHARED_CELLS: usize = 6 * 6 * 6;

const WORKGROUP_SIZE: u32 = 64;

/*
 * Shared memory flatten helpers for G2P (no subtraction, same as g2p.rs).
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
 * Outer product helpers for the MLS reconstruction.
 * In 2D: Vec3 x Vec3 -> Mat3 (homogeneous coordinates).
 * In 3D: Vec4 x Vec4 -> Mat4 (homogeneous coordinates).
 */

#[cfg(feature = "dim2")]
#[inline]
fn outer_product_3(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}

#[cfg(feature = "dim3")]
#[inline]
fn outer_product_4(a: Vec4, b: Vec4) -> Mat4 {
    Mat4::from_cols(a * b.x, a * b.y, a * b.z, a * b.w)
}

/*
 * Helper: whether a shape has a solid interior (for sign bit computation).
 */

#[inline]
fn shape_has_solid_interior(_i_collider: u32) -> bool {
    // TODO: needs to be false for unoriented trimeshes and polylines,
    //       true for geometric primitives.
    false
}

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
    shared_nodes: &mut [NodeCdf; NUM_SHARED_CELLS],
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
                    let flat_id = flatten_shared_index(shared_index.x, shared_index.y) as usize;

                    if octant_hid.id != NONE {
                        let global_chunk_id = block_header_id_to_physical_id(octant_hid);
                        let tid_xy = UVec2::new(tid.x, tid.y);
                        let global_node_id = node_id(global_chunk_id, tid_xy);
                        shared_nodes[flat_id] = nodes.at(global_node_id.id as usize).cdf;
                    } else {
                        shared_nodes[flat_id] = NodeCdf::new(0.0, 0, NONE);
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
                        let octant = UVec3::new(i_loop as u32, j_loop as u32, k_loop as u32);
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
                        let shared_node_id =
                            flatten_shared_index(shared_index.x, shared_index.y, shared_index.z)
                                as usize;

                        if octant_hid.id != NONE {
                            let global_chunk_id = block_header_id_to_physical_id(octant_hid);
                            let global_node_id = node_id(global_chunk_id, tid_xyz);
                            shared_nodes[shared_node_id] = nodes.at(global_node_id.id as usize).cdf;
                        } else {
                            shared_nodes[shared_node_id] = NodeCdf::new(0.0, 0, NONE);
                        }
                    }
                }
            }
        }
    }
}

/*
 * Per-particle G2P CDF interpolation.
 */

#[inline]
#[unroll_for_loops]
fn particle_g2p(
    particles_pos: &[Position],
    particles_cdf: &mut [Cdf],
    particle_id: u32,
    cell_width: f32,
    _dt: f32,
    shared_nodes: &[NodeCdf; NUM_SHARED_CELLS],
) {
    let mut contact_dist = 0.0f32;
    let mut particle_affinity = 0u32;
    let mut affinity_signs = [0.0f32; 16];

    let prev_affinity = particles_cdf.at(particle_id as usize).affinity;
    let particle_pos = particles_pos.read(particle_id as usize);
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

    // Pass 1: Determine sign bits (Eqn. 21) and combine affinity masks.
    for i in 0..27 { // For loop unrolling, use the fixed bound (the maximum one between 2D and 3D).
        if i < NBH_LEN {
            let shift = NBH_SHIFTS.read(i as usize);
            let packed_shift = NBH_SHIFT_SHARED.read(i as usize);
            let cell_data = shared_nodes[(packed_cell_index_in_block + packed_shift) as usize];
            particle_affinity |= cell_data.affinities & AFFINITY_BITS_MASK;

            #[cfg(feature = "dim2")]
            let weight = vec3_extract(w[0], shift.x) * vec3_extract(w[1], shift.y);
            #[cfg(feature = "dim3")]
            let weight =
                vec3_extract(w[0], shift.x) * vec3_extract(w[1], shift.y) * vec3_extract(w[2], shift.z);

            // Unrolled inner loop over 16 colliders.
            // NOTE: `unroll_for_loops` doesn’t see through the closure so we use crunchy::unroll instead.
            unroll! {
                for i_collider in 0..16 {
                    let compatible = if affinity_bit(i_collider as u32, cell_data.affinities) {
                        1.0f32
                    } else {
                        0.0f32
                    };
                    let sign = if sign_bit(i_collider as u32, cell_data.affinities)
                        && !shape_has_solid_interior(i_collider as u32)
                    {
                        -1.0f32
                    } else {
                        1.0f32
                    };
                    affinity_signs[i_collider as usize] += compatible * weight * sign * cell_data.distance;
                }
            }
        }
    }

    // Convert the affinity signs to bits.
    for i_collider in 0..16 {
        let mask = 1u32 << (i_collider as u32 + SIGN_BITS_SHIFT);
        if (prev_affinity & (1u32 << i_collider)) == 0 {
            // Only set the sign bit for affinities that didn't exist before.
            let sgn_bit = if affinity_signs[i_collider as usize] < 0.0 {
                mask
            } else {
                0u32
            };
            particle_affinity |= sgn_bit;
        } else {
            particle_affinity |= prev_affinity & mask;
        }
    }

    // Pass 2: MLS reconstruction of the contact distance/normal (Eq. 4).
    #[cfg(feature = "dim2")]
    let mut qtq = Mat3::ZERO;
    #[cfg(feature = "dim2")]
    let mut qtu = Vec3::ZERO;
    #[cfg(feature = "dim3")]
    let mut qtq = Mat4::ZERO;
    #[cfg(feature = "dim3")]
    let mut qtu = Vec4::ZERO;

    for i in 0..27 { // For loop unrolling, use the fixed bound (the maximum one between 2D and 3D).
        if i < NBH_LEN {
            let shift = NBH_SHIFTS.read(i as usize);
            let packed_shift = NBH_SHIFT_SHARED.read(i as usize);
            let cell_data = shared_nodes[(packed_cell_index_in_block + packed_shift) as usize];

            #[cfg(feature = "dim2")]
            let dpt =
                ref_elt_pos_minus_particle_pos + Vec2::new(shift.x as f32, shift.y as f32) * cell_width;
            #[cfg(feature = "dim2")]
            let weight = vec3_extract(w[0], shift.x) * vec3_extract(w[1], shift.y);
            #[cfg(feature = "dim3")]
            let dpt = ref_elt_pos_minus_particle_pos
                + Vec3::new(shift.x as f32, shift.y as f32, shift.z as f32) * cell_width;
            #[cfg(feature = "dim3")]
            let weight =
                vec3_extract(w[0], shift.x) * vec3_extract(w[1], shift.y) * vec3_extract(w[2], shift.z);

            let combined_affinity = cell_data.affinities & particle_affinity & AFFINITY_BITS_MASK;
            let sign_differences = ((cell_data.affinities >> SIGN_BITS_SHIFT)
                ^ (particle_affinity >> SIGN_BITS_SHIFT))
                & combined_affinity;

            #[cfg(feature = "dim2")]
            let p = Vec3::new(dpt.x, dpt.y, 1.0);
            #[cfg(feature = "dim3")]
            let p = Vec4::new(dpt.x, dpt.y, dpt.z, 1.0);

            if combined_affinity != 0 {
                if sign_differences == 0 {
                    // All signs match: positive distance.
                    #[cfg(feature = "dim2")]
                    {
                        qtq += outer_product_3(p, p) * weight;
                        qtu += p * (weight * cell_data.distance);
                    }
                    #[cfg(feature = "dim3")]
                    {
                        qtq += outer_product_4(p, p) * weight;
                        qtu += p * (weight * cell_data.distance);
                    }
                } else {
                    // Sign difference: negative distance.
                    #[cfg(feature = "dim2")]
                    {
                        qtq += outer_product_3(p, p) * weight;
                        qtu += p * (weight * -cell_data.distance);
                    }
                    #[cfg(feature = "dim3")]
                    {
                        qtq += outer_product_4(p, p) * weight;
                        qtu += p * (weight * -cell_data.distance);
                    }
                }
            }
        }
    }

    if qtq.determinant() > 1.0e-8 {
        #[cfg(feature = "dim2")]
        {
            let result = qtq.inverse() * qtu;
            let len = Vec2::new(result.x, result.y).length();
            let normal = if len > 1.0e-6 {
                Vec2::new(result.x, result.y) / len
            } else {
                Vec2::ZERO
            };
            *particles_cdf.at_mut(particle_id as usize) = Cdf::new(
                normal,
                Vec2::ZERO, // PERF: init the rigid-velocities here instead of in g2p?
                result.z,
                particle_affinity,
            );
        }
        #[cfg(feature = "dim3")]
        {
            let result = qtq.inverse() * qtu;
            let normal_vec = Vec3::new(result.x, result.y, result.z);
            let len = normal_vec.length();
            let normal = if len > 1.0e-6 {
                normal_vec / len
            } else {
                Vec3::ZERO
            };
            *particles_cdf.at_mut(particle_id as usize) =
                Cdf::new(normal, Vec3::ZERO, result.w, particle_affinity);
        }
    } else {
        // TODO: store the affinity in this case too?
        *particles_cdf.at_mut(particle_id as usize) = Cdf::zero();
    }
}

/*
 * GPU entry points.
 */

/// GPU kernel: G2P CDF transfer (2D).
///
/// Transfers grid CDF data back to particles using MLS reconstruction.
#[spirv_bindgen]
#[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8))))]
#[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
pub fn gpu_g2p_cdf(
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] particles_cdf: &mut [Cdf],
    // Shared memory.
    #[spirv(workgroup)] shared_nodes: &mut [NodeCdf; NUM_SHARED_CELLS],
) {
    let bid = block_id.x;
    let vid_ = active_blocks.at(bid as usize).virtual_id.id;
    let vid = BlockVirtualId::new(vid_);

    // Block -> shared memory transfer.
    global_shared_memory_transfers(grid_data, hmap_entries, nodes, tid, vid, shared_nodes);

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
            particles_pos,
            particles_cdf,
            particle_id,
            grid_data.at(0).cell_width,
            params.dt,
            shared_nodes,
        );
        sorted_particle_id += WORKGROUP_SIZE;
    }
}
