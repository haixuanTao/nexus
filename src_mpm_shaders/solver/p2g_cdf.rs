//! Particle-to-Grid CDF (Contact Distance Field) transfer kernel.
//!
//! This kernel transfers collision primitives (segments in 2D, triangles in 3D)
//! from rigid body surface particles onto nearby grid nodes. For each grid node,
//! it projects the node position onto each nearby primitive to compute the signed
//! distance and affinity bits used by the CPIC method.
//!
//! Uses the same linked-list traversal and shared-memory pattern as P2G, but
//! transfers geometry primitives instead of particle dynamics.

use core::ops::Range;
use crate::grid::grid::*;
use crate::grid::kernel::*;
use crate::solver::particle::{Position, RigidParticleIndices};
use crate::{abs, MaybeIndexUnchecked, Vector};
use glamx::*;
use khal_derive::spirv_bindgen;
use nexus_rbd_shaders::PaddedVector;
use vortx_shaders::arch::workgroup_memory_barrier_with_group_sync;
use spirv_std::spirv;
use unroll::unroll_for_loops;
use vortx_shaders::utils::{atomic_load_u32_workgroup, atomic_max_u32_workgroup, atomic_store_u32_workgroup};
/*
 * Constants.
 */

#[cfg(feature = "dim2")]
const NUM_SHARED_CELLS: usize = 10 * 10;
#[cfg(feature = "dim3")]
const NUM_SHARED_CELLS: usize = 6 * 6 * 6;

/*
 * Shared memory types.
 */

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct SharedNode {
    particle_id: u32,
    global_id: u32,
}

/// A collision primitive stored in shared memory.
/// In 2D: segment (two endpoints). In 3D: triangle (three vertices).
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct SharedPrimitive {
    a: Vector,
    b: Vector,
    #[cfg(feature = "dim3")]
    c: Vector,
}

/*
 * Segment projection helper (2D).
 */

#[cfg(feature = "dim2")]
#[inline]
fn project_local_point_on_segment(a: Vec2, b: Vec2, point: Vec2) -> Vec2 {
    let ab = b - a;
    let ap = point - a;
    let ab_sqnorm = ab.dot(ab);

    if ab_sqnorm < 1.0e-10 {
        return a;
    }

    let t = ap.dot(ab) / ab_sqnorm;
    let t = t.clamp(0.0, 1.0);
    a + ab * t
}

/*
 * P2G CDF step: project grid node position onto each nearby primitive.
 */

#[inline]
fn p2g_cdf_step(
    packed_cell_index_in_block: u32,
    cell_width: f32,
    cell_pos: Vector,
    shared_primitives: &[SharedPrimitive; NUM_SHARED_CELLS],
    shared_collider_ids: &[u32; NUM_SHARED_CELLS],
) -> NodeCdf {
    #[cfg(feature = "dim2")]
    let bottommost_contributing_node = flatten_shared_shift(2, 2);
    #[cfg(feature = "dim3")]
    let bottommost_contributing_node = flatten_shared_shift(2, 2, 2);

    let mut result = NodeCdf::NONE;

    for i in 0..NBH_LEN as u32 {
        let packed_shift = NBH_SHIFT_SHARED.read(i as usize);
        let nbh_shared_index =
            (packed_cell_index_in_block - bottommost_contributing_node + packed_shift) as usize;

        let collider_id = shared_collider_ids.read(nbh_shared_index);

        if collider_id == NONE {
            continue;
        }

        let primitive = shared_primitives.read(nbh_shared_index);

        #[cfg(feature = "dim2")]
        {
            // Project on Segment.
            let proj = project_local_point_on_segment(primitive.a, primitive.b, cell_pos);
            // Check if this is a valid projection (not clamped to an endpoint).
            let not_at_a = proj.x != primitive.a.x || proj.y != primitive.a.y;
            let not_at_b = proj.x != primitive.b.x || proj.y != primitive.b.y;
            if not_at_a && not_at_b {
                let dpt = cell_pos - proj;
                let distance = dpt.length();
                let ab = primitive.b - primitive.a;
                let sign = dpt.dot(Vec2::new(-ab.y, ab.x)) < 0.0;
                result.affinities.set_bit(collider_id, sign);

                if distance < result.distance {
                    result.distance = distance;
                    result.closest_id = collider_id;
                }
            }
        }

        #[cfg(feature = "dim3")]
        {
            // Project on Triangle.
            let ap = cell_pos - primitive.a;
            let bp = cell_pos - primitive.b;
            let cp = cell_pos - primitive.c;
            let ab = primitive.b - primitive.a;
            let ac = primitive.c - primitive.a;
            let bc = primitive.c - primitive.b;
            let n = ab.cross(ac);
            let n_length = n.length();

            if n_length != 0.0
                && ab.cross(n).dot(ap) <= 0.0
                && bc.cross(n).dot(bp) <= 0.0
                && ac.cross(n).dot(cp) >= 0.0
            // Positive sign due to `ac` instead of `ca`.
            {
                // Valid projection on the face interior.
                let signed_dist = n.dot(ap) / n_length;
                let distance = abs(signed_dist);
                result.affinities.set_bit(collider_id, signed_dist < 0.0);

                if distance < result.distance {
                    result.distance = distance;
                    result.closest_id = collider_id;
                }
            }
        }
    }

    result
}

/*
 * Fetch functions
 */

#[inline]
#[unroll_for_loops]
fn fetch_max_linked_lists_length(
    grid: &Grid,
    hmap_entries: &[GridHashMapEntry],
    rigid_nodes_linked_lists: &[NodeLinkedList],
    tid: spirv_std::glam::UVec3,
    active_block_vid: BlockVirtualId,
    max_linked_list_length: &mut u32,
) {
    #[cfg(feature = "dim2")]
    let base_block_pos_int = active_block_vid.id - IVec2::new(1, 1);
    #[cfg(feature = "dim3")]
    let base_block_pos_int = active_block_vid.id - IVec3::new(1, 1, 1);

    for i_loop in 0..2u32 {
        for j_loop in 0..2u32 {
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
                        let len = rigid_nodes_linked_lists.at(global_node_id.id as usize).len;
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
                        &BlockVirtualId::new(
                            base_block_pos_int
                                + IVec3::new(octant.x as i32, octant.y as i32, octant.z as i32),
                        ),
                    );
                    if octant_hid.id != NONE {
                        let global_chunk_id = octant_hid.physical_id();
                        let tid_xyz = UVec3::new(tid.x, tid.y, tid.z);
                        let global_node_id = global_chunk_id.node_id(tid_xyz);
                        let len = rigid_nodes_linked_lists.at(global_node_id.id as usize).len;
                        atomic_max_u32_workgroup(max_linked_list_length, len);
                    }
                }
            }
        }
    }
}

#[inline]
#[unroll_for_loops]
fn fetch_nodes(
    grid: &Grid,
    hmap_entries: &[GridHashMapEntry],
    rigid_nodes_linked_lists: &[NodeLinkedList],
    tid: spirv_std::glam::UVec3,
    active_block_vid: BlockVirtualId,
    shared_nodes: &mut [SharedNode; NUM_SHARED_CELLS],
) {
    #[cfg(feature = "dim2")]
    let base_block_pos_int = active_block_vid.id - IVec2::new(1, 1);
    #[cfg(feature = "dim3")]
    let base_block_pos_int = active_block_vid.id - IVec3::new(1, 1, 1);

    for i_loop in 0..2u32 {
        for j_loop in 0..2u32 {
            for k_loop in 0..2 {
                #[cfg(feature = "dim2")]
                let (skip, shared_node_index) = {
                    if k_loop != 0 || (i_loop == 0 && tid.x < 6) || (j_loop == 0 && tid.y < 6) {
                        (true, 0usize)
                    } else {
                        let octant = UVec2::new(i_loop as u32, j_loop as u32);
                        let shared_index = octant * 8 + UVec2::new(tid.x, tid.y);
                        (
                            false,
                            flatten_shared_index(shared_index.x, shared_index.y) as usize,
                        )
                    }
                };
                #[cfg(feature = "dim3")]
                let (skip, shared_node_index) = {
                    if (i_loop == 0 && tid.x < 2)
                        || (j_loop == 0 && tid.y < 2)
                        || (k_loop == 0 && tid.z < 2)
                    {
                        (true, 0usize)
                    } else {
                        let octant = UVec3::new(i_loop as u32, j_loop as u32, k_loop as u32);
                        let tid_xyz = UVec3::new(tid.x, tid.y, tid.z);
                        let shared_index = octant * 4 + tid_xyz;
                        (
                            false,
                            flatten_shared_index(shared_index.x, shared_index.y, shared_index.z)
                                as usize,
                        )
                    }
                };

                if !skip {
                    #[cfg(feature = "dim2")]
                    let octant_hid = {
                        let octant = UVec2::new(i_loop as u32, j_loop as u32);
                        grid.find_block_header_id(
                            hmap_entries,
                            &BlockVirtualId {
                                id: base_block_pos_int + IVec2::new(octant.x as i32, octant.y as i32),
                            },
                        )
                    };
                    #[cfg(feature = "dim3")]
                    let octant_hid = {
                        let octant = UVec3::new(i_loop as u32, j_loop as u32, k_loop as u32);
                        grid.find_block_header_id(
                            hmap_entries,
                            &BlockVirtualId::new(
                                base_block_pos_int
                                    + IVec3::new(octant.x as i32, octant.y as i32, octant.z as i32),
                            ),
                        )
                    };

                    if octant_hid.id != NONE {
                        let global_chunk_id = octant_hid.physical_id();
                        #[cfg(feature = "dim2")]
                        let global_node_id = global_chunk_id.node_id(UVec2::new(tid.x, tid.y));
                        #[cfg(feature = "dim3")]
                        let global_node_id = global_chunk_id.node_id(UVec3::new(tid.x, tid.y, tid.z));
                        let particle_id = rigid_nodes_linked_lists.at(global_node_id.id as usize).head;
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

#[inline]
fn fetch_next_particle(
    particle_node_linked_lists: &[u32],
    collider_vertices: &[PaddedVector],
    rigid_particle_indices: &[RigidParticleIndices],
    tid: spirv_std::glam::UVec3,
    shared_nodes: &mut [SharedNode; NUM_SHARED_CELLS],
    shared_primitives: &mut [SharedPrimitive; NUM_SHARED_CELLS],
    shared_collider_ids: &mut [u32; NUM_SHARED_CELLS],
) {
    for i_loop in 0..2u32 {
        for j_loop in 0..2u32 {
            for k_loop in 0..2u32 {
                #[cfg(feature = "dim2")]
                let skip = k_loop != 0 || (i_loop == 0 && tid.x < 6) || (j_loop == 0 && tid.y < 6);
                #[cfg(feature = "dim3")]
                let skip = (i_loop == 0 && tid.x < 2)
                    || (j_loop == 0 && tid.y < 2)
                    || (k_loop == 0 && tid.z < 2);

                if skip {
                    continue;
                }

                #[cfg(feature = "dim2")]
                let shared_flat_index = {
                    let octant = UVec2::new(i_loop, j_loop);
                    let shared_index = octant * 8 + UVec2::new(tid.x, tid.y);
                    flatten_shared_index(shared_index.x, shared_index.y) as usize
                };
                #[cfg(feature = "dim3")]
                let shared_flat_index = {
                    let octant = UVec3::new(i_loop, j_loop, k_loop);
                    let tid_xyz = UVec3::new(tid.x, tid.y, tid.z);
                    let shared_index = octant * 4 + tid_xyz;
                    flatten_shared_index(shared_index.x, shared_index.y, shared_index.z) as usize
                };

                let curr_particle_id = shared_nodes.at(shared_flat_index).particle_id;

                if curr_particle_id != NONE {
                    let rigid_idx = rigid_particle_indices.read(curr_particle_id as usize);
                    shared_collider_ids.write(shared_flat_index, rigid_idx.collider);

                    #[cfg(feature = "dim2")]
                    {
                        shared_primitives.write(shared_flat_index, SharedPrimitive {
                            a: collider_vertices.read(rigid_idx.segment.x as usize).0,
                            b: collider_vertices.read(rigid_idx.segment.y as usize).0,
                        });
                    }
                    #[cfg(feature = "dim3")]
                    {
                        shared_primitives.write(shared_flat_index, SharedPrimitive {
                            a: collider_vertices.read(rigid_idx.triangle.x as usize).0,
                            b: collider_vertices.read(rigid_idx.triangle.y as usize).0,
                            c: collider_vertices.read(rigid_idx.triangle.z as usize).0,
                        });
                    }

                    let next_particle_id =
                        particle_node_linked_lists.read(curr_particle_id as usize);
                    shared_nodes.at_mut(shared_flat_index).particle_id = next_particle_id;
                } else {
                    shared_collider_ids.write(shared_flat_index, NONE);
                    shared_primitives.write(shared_flat_index, SharedPrimitive {
                        a: Vector::ZERO,
                        b: Vector::ZERO,
                        #[cfg(feature = "dim3")]
                        c: Vector::ZERO,
                    });
                }
            }
        }
    }
}

/*
 * GPU entry points.
 */

/// GPU kernel: P2G CDF transfer
#[spirv_bindgen]
#[cfg_attr(feature = "dim2", spirv(compute(threads(8, 8))))]
#[cfg_attr(feature = "dim3", spirv(compute(threads(4, 4, 4))))]
pub fn gpu_p2g_cdf(
    #[spirv(workgroup_id)] block_id: spirv_std::glam::UVec3,
    #[spirv(local_invocation_id)] tid: spirv_std::glam::UVec3,
    #[spirv(local_invocation_index)] tid_flat: u32,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] hmap_entries: &[GridHashMapEntry],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] active_blocks: &[ActiveBlockHeader],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    rigid_nodes_linked_lists: &[NodeLinkedList],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] particle_node_linked_lists: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)]
    collider_vertices: &[PaddedVector],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)]
    rigid_particle_indices: &[RigidParticleIndices],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] nodes: &mut [Node],
    // Shared memory.
    #[spirv(workgroup)] shared_nodes: &mut [SharedNode; NUM_SHARED_CELLS],
    #[spirv(workgroup)] shared_primitives: &mut [SharedPrimitive; NUM_SHARED_CELLS],
    #[spirv(workgroup)] shared_collider_ids: &mut [u32; NUM_SHARED_CELLS],
    #[spirv(workgroup)] max_linked_list_length: &mut u32,
    #[spirv(workgroup)] max_linked_list_length_uniform: &mut u32,
) {
    let bid = block_id.x;
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
        rigid_nodes_linked_lists,
        tid,
        vid,
        max_linked_list_length,
    );
    workgroup_memory_barrier_with_group_sync();

    *max_linked_list_length_uniform = atomic_load_u32_workgroup(max_linked_list_length);

    // Block -> shared memory transfer.
    fetch_nodes(
        grid,
        hmap_entries,
        rigid_nodes_linked_lists,
        tid,
        vid,
        shared_nodes,
    );

    // Compute the packed cell index and cell position for the current thread's node.
    #[cfg(feature = "dim2")]
    let packed_cell_index_in_block = flatten_shared_index(tid.x + 8, tid.y + 8);
    #[cfg(feature = "dim2")]
    let cell_pos = Vec2::new(
        (vid.id.x * 8 + tid.x as i32) as f32,
        (vid.id.y * 8 + tid.y as i32) as f32,
    ) * grid.cell_width;
    #[cfg(feature = "dim3")]
    let packed_cell_index_in_block = flatten_shared_index(tid.x + 4, tid.y + 4, tid.z + 4);
    #[cfg(feature = "dim3")]
    let cell_pos = Vec3::new(
        (vid.id.x * 4 + tid.x as i32) as f32,
        (vid.id.y * 4 + tid.y as i32) as f32,
        (vid.id.z * 4 + tid.z as i32) as f32,
    ) * grid.cell_width;

    let global_id = shared_nodes.at(packed_cell_index_in_block as usize).global_id;
    let mut node_cdf = nodes.at(global_id as usize).cdf;

    // Iterate through the linked list with uniform control flow.
    let len = *max_linked_list_length_uniform;

    // Need to cap the iteration count on the web.
    #[cfg(feature = "web-compat")]
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
            fetch_next_particle(
                particle_node_linked_lists,
                collider_vertices,
                rigid_particle_indices,
                tid,
                shared_nodes,
                shared_primitives,
                shared_collider_ids,
            );
        }
        workgroup_memory_barrier_with_group_sync();
        if ok {
            let partial_result = p2g_cdf_step(
                packed_cell_index_in_block,
                grid.cell_width,
                cell_pos,
                shared_primitives,
                shared_collider_ids,
            );

            if partial_result.closest_id != NONE {
                node_cdf.affinities |= partial_result.affinities;

                if partial_result.distance < node_cdf.distance {
                    node_cdf.distance = partial_result.distance;
                    node_cdf.closest_id = partial_result.closest_id;
                }
            }
        }
    }

    // Write the node cdf to global memory.
    nodes.at_mut(global_id as usize).cdf = node_cdf;
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
