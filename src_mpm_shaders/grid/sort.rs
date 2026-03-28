//! Particle sorting kernels for the sparse MPM grid.
//!
//! These kernels handle:
//! 1. Marking blocks as active based on particle positions.
//! 2. Counting particles per block.
//! 3. Building sorted particle arrays and per-node linked lists.
//!
//! The sorting pipeline runs in multiple passes:
//! 1. `touch_particle_blocks` / `touch_rigid_particle_blocks` - mark active blocks
//! 2. `mark_rigid_particles_needing_block` - flag rigid particles near block boundaries
//! 3. `update_block_particle_count` - count particles per active block
//! 4. `copy_particles_len_to_scan_value` - prepare prefix sum input
//! 5. prefix sum (external) - compute exclusive scan of particle counts
//! 6. `copy_scan_values_to_first_particles` - write back sorted offsets
//! 7. `finalize_particles_sort` - place particles in sorted order and build linked lists
//! 8. `sort_rigid_particles` - build per-node linked lists for rigid particles

use crate::grid::grid::*;
use crate::solver::particle::{Position, associated_cell_index_in_block_off_by_one};
use khal_std::arch::atomic_add_u32;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

/// Marks all blocks associated with each particle as active.
///
/// For each particle, computes the set of blocks whose stencil could overlap
/// the particle, and inserts them into the hashmap. This must be run before
/// any per-block operations.
// TODO HACK: enabling spirv-passthrough for this shader since naga panics
//            on the spv backend because of https://github.com/gfx-rs/wgpu/issues/7315
//            (in our case, it’s caused by the lines involving the atomic compare-exchange).
#[spirv_bindgen(spirv_passthrough)]
#[spirv(compute(threads(64)))]
pub fn gpu_touch_particle_blocks(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] grid: &mut Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    hmap_entries: &mut [GridHashMapEntry],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    active_blocks: &mut [ActiveBlockHeader],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] particles_pos: &[Position],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] particles_len: &u32,
) {
    let id = invocation_id.x;
    if id < *particles_len {
        let cell_width = grid.cell_width;
        let particle = particles_pos.read(id as usize);
        let blocks = BlockVirtualId::blocks_associated_to_point(cell_width, particle.pt);
        for i in 0..NUM_ASSOC_BLOCKS {
            grid.mark_block_as_active(hmap_entries, active_blocks, &blocks[i]);
        }
    }
}

/// Marks all blocks associated with each rigid particle as active.
///
/// Similar to `gpu_touch_particle_blocks`, but operates on rigid body surface
/// particles. Only touches blocks for rigid particles that are flagged as needing
/// a block (via the `rigid_particle_needs_block` bitfield).
// TODO HACK: enabling spirv-passthrough for this shader since naga panics
//            on the spv backend because of https://github.com/gfx-rs/wgpu/issues/7315
//            (in our case, it’s caused by the lines involving the atomic compare-exchange).
#[spirv_bindgen(spirv_passthrough)]
#[spirv(compute(threads(64)))]
pub fn gpu_touch_rigid_particle_blocks(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] grid: &mut Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    hmap_entries: &mut [GridHashMapEntry],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    active_blocks: &mut [ActiveBlockHeader],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] rigid_particles_pos: &[Position],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] rigid_particle_needs_block: &[u32],
) {
    let id = invocation_id.x;
    if id < rigid_particles_pos.len() as u32 {
        let cell_width = grid.cell_width;
        let entry_id = (id / 32) as usize;
        let entry_bit = 1u32 << (id % 32);
        let needs_block = (rigid_particle_needs_block.read(entry_id) & entry_bit) != 0;

        if needs_block {
            let particle = rigid_particles_pos.read(id as usize);
            let block = BlockVirtualId::block_associated_to_point(cell_width, particle.pt);
            grid.mark_block_as_active(hmap_entries, active_blocks, &block);
        }
    }
}

/// Flags rigid particles that need their own block activated.
///
/// A rigid particle needs its own block if at least one (but not all) of its
/// associated blocks are already active. This means the particle is near a
/// block boundary and its contributions would be lost without an additional block.
///
/// The result is stored as a bitfield in `rigid_particle_needs_block`, where
/// each u32 holds flags for 32 rigid particles.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mark_rigid_particles_needing_block(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] hmap_entries: &[GridHashMapEntry],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] rigid_particles_pos: &[Position],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    rigid_particle_needs_block: &mut [u32],
) {
    let id = invocation_id.x;
    if id < rigid_particles_pos.len() as u32 {
        let cell_width = grid.cell_width;
        let particle = rigid_particles_pos.read(id as usize);
        let blocks = BlockVirtualId::blocks_associated_to_point(cell_width, particle.pt);

        // Find the first block that already has a header in the hashmap.
        let mut i = 0u32;
        for _ in 0..NUM_ASSOC_BLOCKS {
            if grid
                .find_block_header_id(hmap_entries, &blocks[i as usize])
                .id
                != NONE
            {
                break;
            }
            i += 1;
        }

        let entry_id = (id / 32) as usize;
        let entry_bit = 1u32 << (id % 32);

        // If some but not all associated blocks are active, the rigid particle
        // needs its own block to ensure proper grid transfers.
        if i > 0 && i < NUM_ASSOC_BLOCKS as u32 {
            // Set the bit atomically.
            khal_std::arch::atomic_or_u32(
                &mut rigid_particle_needs_block.at_mut(entry_id),
                entry_bit,
            );
        } else {
            // Clear the bit atomically.
            khal_std::arch::atomic_and_u32(
                &mut rigid_particle_needs_block.at_mut(entry_id),
                !entry_bit,
            );
        }
    }
}

/// Counts the number of particles in each active block.
///
/// Each thread processes one particle, finds its associated block, and
/// atomically increments that block's `num_particles` counter.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_update_block_particle_count(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] hmap_entries: &[GridHashMapEntry],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] particles_pos: &[Position],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] particles_len: &u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)]
    active_blocks: &mut [ActiveBlockHeader],
) {
    let id = invocation_id.x;
    if id < *particles_len {
        let cell_width = grid.cell_width;
        let particle = particles_pos.read(id as usize);
        let block_vid = BlockVirtualId::block_associated_to_point(cell_width, particle.pt);
        let active_block_id = grid.find_block_header_id(hmap_entries, &block_vid);
        atomic_add_u32(
            &mut active_blocks
                .at_mut(active_block_id.id as usize)
                .num_particles,
            1,
        );
    }
}

/// Copies each active block's particle count into the scan_values buffer.
///
/// This prepares the input for the prefix sum pass. After the prefix sum,
/// `scan_values[i]` will contain the global offset for block `i`'s particles.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_copy_particles_len_to_scan_value(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] active_blocks: &[ActiveBlockHeader],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] scan_values: &mut [u32],
) {
    let id = invocation_id.x;
    if id < grid.num_active_blocks {
        scan_values.write(id as usize, active_blocks.at(id as usize).num_particles);
    }
}

/// Writes the prefix sum results back as `first_particle` offsets and resets particle counts.
///
/// After the prefix sum, `scan_values[i]` contains the exclusive scan result.
/// This kernel copies it into `active_blocks[i].first_particle`.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_copy_scan_values_to_first_particles(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] scan_values: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    active_blocks: &mut [ActiveBlockHeader],
) {
    let id = invocation_id.x;
    if id < grid.num_active_blocks {
        let idx = id as usize;
        active_blocks.at_mut(idx).first_particle = scan_values.read(idx);
    }
}

/// Places particles into their sorted positions and builds per-node linked lists.
///
/// Each thread processes one particle:
/// 1. Finds the particle's active block via the hashmap.
/// 2. Atomically claims a slot in the sorted array (using `scan_values` as a counter).
/// 3. Writes the particle's original index into `sorted_particle_ids`.
/// 4. Inserts the particle into the per-node linked list for its closest grid node.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_finalize_particles_sort(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] hmap_entries: &[GridHashMapEntry],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] particles_pos: &[Position],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] particles_len: &u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] scan_values: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)]
    nodes_linked_lists: &mut [NodeLinkedList],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)]
    particle_node_linked_lists: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] sorted_particle_ids: &mut [u32],
) {
    let id = invocation_id.x;
    if id < *particles_len {
        let cell_width = grid.cell_width;
        let particle = particles_pos.read(id as usize);
        let block_vid = BlockVirtualId::block_associated_to_point(cell_width, particle.pt);

        // Place the particle at its sorted position.
        let active_block_id = grid.find_block_header_id(hmap_entries, &block_vid);
        let target_index = atomic_add_u32(&mut scan_values.at_mut(active_block_id.id as usize), 1);
        sorted_particle_ids.write(target_index as usize, id);

        // Build per-node particle linked list.
        let node_local_id = associated_cell_index_in_block_off_by_one(&particle, cell_width);
        let node_global_id = active_block_id.physical_id().node_id(node_local_id);
        let prev_head = khal_std::arch::atomic_exchange_u32(
            &mut nodes_linked_lists.at_mut(node_global_id.id as usize).head,
            id,
        );
        atomic_add_u32(
            &mut nodes_linked_lists.at_mut(node_global_id.id as usize).len,
            1,
        );
        particle_node_linked_lists.write(id as usize, prev_head);
    }
}

/// Builds per-node linked lists for rigid body surface particles.
///
/// Similar to the finalization pass for deformable particles, but only
/// builds the linked list (no sorting into a sorted array). Rigid particles
/// that don't map to any active block are silently skipped.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_sort_rigid_particles(
    #[spirv(global_invocation_id)] invocation_id: khal_std::glamx::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] grid: &Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] hmap_entries: &[GridHashMapEntry],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] rigid_particles_pos: &[Position],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    rigid_nodes_linked_lists: &mut [NodeLinkedList],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)]
    rigid_particle_node_linked_lists: &mut [u32],
) {
    let id = invocation_id.x;
    if id < rigid_particles_pos.len() as u32 {
        let cell_width = grid.cell_width;
        let particle = rigid_particles_pos.read(id as usize);
        let block_vid = BlockVirtualId::block_associated_to_point(cell_width, particle.pt);

        let active_block_id = grid.find_block_header_id(hmap_entries, &block_vid);

        // If the rigid particle doesn't map to any active block, we can just ignore it
        // as it won't affect the simulation.
        if active_block_id.id != NONE {
            // Build per-node rigid particle linked list.
            let node_local_id = associated_cell_index_in_block_off_by_one(&particle, cell_width);
            let node_global_id = active_block_id.physical_id().node_id(node_local_id);
            let prev_head = khal_std::arch::atomic_exchange_u32(
                &mut rigid_nodes_linked_lists
                    .at_mut(node_global_id.id as usize)
                    .head,
                id,
            );
            atomic_add_u32(
                &mut rigid_nodes_linked_lists
                    .at_mut(node_global_id.id as usize)
                    .len,
                1,
            );
            rigid_particle_node_linked_lists.write(id as usize, prev_head);
        }
    }
}
