//! Sparse grid data structures and hashmap for MPM.
//!
//! The MPM grid is stored as a sparse set of active blocks. Each block contains
//! a fixed number of grid nodes (8x8 in 2D, 4x4x4 in 3D = 64 nodes per block).
//!
//! Active blocks are tracked via a GPU hashmap that maps virtual block coordinates
//! to physical storage indices. The hashmap uses open addressing with linear probing
//! and atomic compare-exchange for lock-free insertion.

use crate::{atomic_add_u32, IVector, MaybeIndexUnchecked, Vector, VectorPlusOne};
use glamx::*;
use khal_derive::spirv_bindgen;
use spirv_std::spirv;

/*
 * Constants.
 */

/// Workgroup size for P2G and G2P transfers.
pub const G2P_P2G_WORKGROUP_SIZE: u32 = 64;

/// Number of cells (nodes) per block.
/// 8 * 8 = 64 in 2D, 4 * 4 * 4 = 64 in 3D.
pub const NUM_CELL_PER_BLOCK: u32 = 64;

/// Workgroup size for grid-level operations.
/// Must match `NUM_CELL_PER_BLOCK` because some kernels (like reset) rely on it.
pub const GRID_WORKGROUP_SIZE: u32 = NUM_CELL_PER_BLOCK;

/// Sentinel value indicating "no entry" / "empty slot" in the hashmap and linked lists.
pub const NONE: u32 = 0xFFFFFFFF;

/// Number of blocks associated with each particle/point.
/// In 2D a particle can straddle 4 blocks (2x2), in 3D it can straddle 8 blocks (2x2x2).
#[cfg(feature = "dim2")]
pub const NUM_ASSOC_BLOCKS: usize = 4;
/// Number of blocks associated with each particle/point.
#[cfg(feature = "dim3")]
pub const NUM_ASSOC_BLOCKS: usize = 8;

/// Offset applied when computing cell indices within a block.
pub const OFF_BY_ONE: i32 = 1;

/// Mask for the lower 16 affinity bits in the CDF affinity field.
pub const AFFINITY_BITS_MASK: u32 = 0x0000FFFF;

/// Bit shift to access the sign bits in the upper 16 bits of the affinity field.
pub const SIGN_BITS_SHIFT: u32 = 16;

/*
 * Index newtypes.
 */

/// Virtual (logical) block coordinate in the sparse grid.
///
/// This is an integer vector (IVec2 in 2D, IVec3 in 3D) identifying a block's
/// position in the infinite virtual grid.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch = "spirv"), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct BlockVirtualId {
    pub id: IVector,
    #[cfg(feature = "dim3")]
    pub padding: u32,
}

impl BlockVirtualId {
    pub fn new(id: IVector) -> Self {
        Self {
            id,
            #[cfg(feature = "dim3")]
            padding: 0,
        }
    }
}

/// Index into the active block headers array.
///
/// After insertion into the hashmap, each active block is assigned a header ID
/// that serves as its index in the `active_blocks` array.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch = "spirv"), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct BlockHeaderId {
    pub id: u32,
}

/// Physical (storage) index for a block's first node.
///
/// Computed as `header_id * NUM_CELL_PER_BLOCK`. Used to index into the flat
/// node arrays.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch = "spirv"), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct BlockPhysicalId {
    pub id: u32,
}

/// Physical (storage) index for a single grid node.
///
/// Computed as `block_physical_id + local_offset_in_block`.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch = "spirv"), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct NodePhysicalId {
    pub id: u32,
}

/*
 * Data structures.
 */

/// Per-node linked list head for particle sorting.
///
/// Each grid node maintains a linked list of particles that map to it.
/// The `head` field points to the first particle, and the `len` field
/// counts the number of particles in the list.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch = "spirv"), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct NodeLinkedList {
    pub head: u32,
    pub len: u32,
}

/// An entry in the GPU hashmap that maps block virtual IDs to header IDs.
///
/// The hashmap uses open addressing with linear probing. The `state` field
/// serves double duty: `NONE` means the slot is empty, otherwise it stores
/// the packed key for comparison during probing.
///
/// IMPORTANT: if this struct is changed (including its layout), be sure to
///            modify the corresponding host-side struct to ensure it has the
///            right size. Otherwise the hashmap will break.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch = "spirv"), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct GridHashMapEntry {
    /// The virtual block ID key.
    pub key: BlockVirtualId,
    /// The associated block header ID value.
    pub value: BlockHeaderId,
    /// The packed key stored in this slot, or `NONE` if the slot is empty.
    pub state: u32,
    /// Ownership flag for weak-CAS correctness.
    /// Reset to 0 each frame; the first thread to `atomic_exchange` it to 1
    /// becomes the slot's owner and allocates the block header.
    pub ownership: u32,
    pub padding: [u32; 1],
}

/// Header for an active block in the sparse grid.
///
/// Stores the virtual ID (for computing world-space positions) and
/// particle sorting information.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch = "spirv"), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct ActiveBlockHeader {
    /// The virtual block coordinate needed to compute world-space node positions.
    pub virtual_id: BlockVirtualId,
    /// Index of the first particle belonging to this block in the sorted array.
    pub first_particle: u32,
    /// Number of particles belonging to this block.
    pub num_particles: u32,
    #[cfg(feature = "dim3")]
    pub padding: [u32; 2],
}

/// Top-level grid metadata.
///
/// Contains the current number of active blocks and configuration parameters.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch = "spirv"), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct Grid {
    /// Current number of active blocks (modified atomically during insertion).
    pub num_active_blocks: u32,
    /// The uniform cell width (grid spacing).
    pub cell_width: f32,
    /// Capacity of the hashmap (must be a power of 2).
    pub hmap_capacity: u32,
    /// Maximum number of blocks that can be stored.
    pub capacity: u32,
}

/// Contact distance field data stored per grid node.
///
/// Used by the CPIC (Compatible Particle-In-Cell) method to handle
/// rigid body coupling through affinity-based compatibility checks.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch = "spirv"), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct NodeCdf {
    /// Signed distance to the closest collider surface.
    pub distance: f32,
    /// Affinity bits: lower 16 bits are affinity flags, upper 16 bits are sign flags.
    /// Two bits per collider.
    pub affinities: u32,
    /// Index of the closest collider, or `NONE` if no collider is nearby.
    pub closest_id: u32,
}

impl NodeCdf {
    /// Creates a new `NodeCdf` with the given values.
    #[inline]
    pub fn new(distance: f32, affinities: u32, closest_id: u32) -> Self {
        Self {
            distance,
            affinities,
            closest_id,
        }
    }
}

/// A single grid node's state.
///
/// Stores momentum/velocity packed with mass, plus CDF data for rigid body coupling.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch = "spirv"), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct Node {
    /// The first DIM components contain either momentum or velocity (depending on context).
    /// The last component contains the node's mass.
    pub momentum_velocity_mass: VectorPlusOne,
    /// SPIR-V padding: Vec3 has align(16) so stride must be multiple of 16.
    #[cfg(feature = "dim2")]
    pub _pad0: u32,
    /// Momentum/velocity/mass for particles that are incompatible with this node
    /// (per CPIC's affinity-based compatibility). This ensures P2G/G2P transfers
    /// on incompatible nodes still work properly without losing contributions from
    /// other compatible particles.
    pub momentum_velocity_mass_incompatible: VectorPlusOne,
    /// SPIR-V padding: Vec3 has align(16) so stride must be multiple of 16.
    #[cfg(feature = "dim2")]
    pub _pad1: u32,
    /// Contact distance field data for rigid body coupling.
    pub cdf: NodeCdf,
    /// SPIR-V padding: NodeCdf is 12 bytes, pad to 16 for struct stride alignment.
    pub _padding: u32,
}

/*
 * Hashmap functions.
 */

/// Packs a virtual block ID into a single u32 for use as a hashmap key.
///
/// In 2D: 16 bits for X, 16 bits for Y.
/// In 3D: 11 bits for X, 10 bits for Y, 11 bits for Z (Y gets fewer bits
///         assuming Y-up, since the vertical extent is typically smaller).
#[cfg(feature = "dim2")]
#[inline]
pub fn pack_key(key: &BlockVirtualId) -> u32 {
    ((key.id.x + 0x00007FFF) as u32 & 0x0000FFFF)
        | (((key.id.y + 0x00007FFF) as u32 & 0x0000FFFF) << 16)
}

/// Packs a virtual block ID into a single u32 for use as a hashmap key.
#[cfg(feature = "dim3")]
#[inline]
pub fn pack_key(key: &BlockVirtualId) -> u32 {
    ((key.id.x + 0x000003FF) as u32 & 0x000007FF)
        | (((key.id.y + 0x000001FF) as u32 & 0x000003FF) << 11)
        | (((key.id.z + 0x000003FF) as u32 & 0x000007FF) << 21)
}

/// Computes a Murmur3-based hash of a packed key.
///
/// The hash is used to determine the initial probe slot in the hashmap.
#[inline]
pub fn hash(packed_key: u32) -> u32 {
    let mut key = packed_key;
    key = key.wrapping_mul(0xCC9E2D51);
    key = (key << 15) | (key >> 17);
    key = key.wrapping_mul(0x1B873593);
    key
}

/// Attempts to insert a block into the hashmap using atomic compare-exchange.
///
/// Uses open addressing with linear probing. Returns the slot index if a new
/// entry was created, or `NONE` if the key already exists or the hashmap is full.
///
/// This function handles weak CAS semantics (as found on WebGPU/WGSL targets
/// where SPIR-V's `OpAtomicCompareExchange` is translated to
/// `atomicCompareExchangeWeak`):
/// - After a CAS that returns `NONE`, it verifies the write via `atomic_load`
///   (which is always strong). On spurious failure (load still shows `NONE`),
///   the same slot is retried on the next loop iteration.
/// - Uses `atomic_exchange` on the entry's `ownership` field to resolve races
///   where multiple threads with the same key all see `NONE` from CAS. Only
///   the thread that exchanges `0 → 1` is considered the inserter.
///
/// The hashmap implementation is inspired by:
/// <https://nosferalatu.com/SimpleGPUHashTable.html>
#[inline]
pub fn insertion_index(
    hmap_entries: &mut [GridHashMapEntry],
    capacity: u32,
    key: &BlockVirtualId,
) -> u32 {
    let packed_key = pack_key(key);
    let mut slot = hash(packed_key) & (capacity - 1);

    // NOTE: if there is no more room in the hashmap to store the data, we just do nothing.
    // It is up to the user to detect the high occupancy, resize the hashmap, and re-run
    // the failed insertion.
    for _ in 0..capacity {
        let old_value = unsafe {
            spirv_std::arch::atomic_compare_exchange::<
                u32,
                { spirv_std::memory::Scope::QueueFamily as u32 },
                { spirv_std::memory::Semantics::NONE.bits() },
                { spirv_std::memory::Semantics::NONE.bits() },
            >(
                &mut hmap_entries.at_mut(slot as usize).state,
                packed_key,
                NONE,
            )
        };

        if old_value == packed_key {
            // The entry already exists.
            return NONE;
        }

        if old_value != NONE {
            // Slot occupied by a different key. Probe next.
            slot = (slot + 1) & (capacity - 1);
            continue;
        }

        // CAS returned NONE. Either we wrote successfully, or it was a spurious
        // failure (weak CAS on WGSL/Metal). Verify with atomic_load (which is always strong).
        let current = unsafe {
            spirv_std::arch::atomic_load::<
                u32,
                { spirv_std::memory::Scope::QueueFamily as u32 },
                { spirv_std::memory::Semantics::NONE.bits() },
            >(&hmap_entries.at(slot as usize).state)
        };

        if current == packed_key {
            // Slot contains our key (we wrote it, or a same-key thread did).
            // Use atomic_exchange on ownership to determine the unique owner.
            // atomic_exchange is always strong (no weak variant in WGSL).
            hmap_entries.at_mut(slot as usize).key = *key;
            let prev = unsafe {
                spirv_std::arch::atomic_exchange::<
                    u32,
                    { spirv_std::memory::Scope::QueueFamily as u32 },
                    { spirv_std::memory::Semantics::NONE.bits() },
                >(&mut hmap_entries.at_mut(slot as usize).ownership, 1)
            };
            if prev == 0 {
                return slot; // We are the owner (new insertion).
            }
            return NONE; // Another thread owns this slot.
        }

        if current != NONE {
            // A different key was written between our CAS and load. Probe next.
            slot = (slot + 1) & (capacity - 1);
            continue;
        }

        // current == NONE: spurious CAS failure. Retry the same slot on the
        // next iteration (slot is not advanced). This wastes one iteration of
        // the capacity-bounded loop but spurious failures are extremely rare.
    }

    NONE
}

/// Looks up a block's header ID in the hashmap.
///
/// Returns the `BlockHeaderId` for the given virtual block coordinate,
/// or a `BlockHeaderId` with `id == NONE` if the block is not active.
#[inline]
pub fn find_block_header_id(
    grid: &[Grid],
    hmap_entries: &[GridHashMapEntry],
    key: &BlockVirtualId,
) -> BlockHeaderId {
    let packed_key = pack_key(key);
    let capacity = grid.at(0).hmap_capacity;
    let mut slot = hash(packed_key) & (capacity - 1);

    for _ in 0..capacity {
        let state = hmap_entries.at(slot as usize).state;
        if state == packed_key {
            return hmap_entries.at(slot as usize).value;
        } else if state == NONE {
            break;
        }

        slot = (slot + 1) & (capacity - 1);
    }

    BlockHeaderId { id: NONE }
}

/// Marks a block as active by inserting it into the hashmap and allocating a header.
///
/// If the block is successfully inserted (i.e., it was not already active),
/// a new `ActiveBlockHeader` entry is created and the hashmap entry is linked
/// to it via an atomically-assigned header ID.
#[inline]
pub fn mark_block_as_active(
    grid: &mut [Grid],
    hmap_entries: &mut [GridHashMapEntry],
    active_blocks: &mut [ActiveBlockHeader],
    block: &BlockVirtualId,
) {
    let slot = insertion_index(hmap_entries, grid.at(0).hmap_capacity, block);

    if slot != NONE {
        let block_header_id = atomic_add_u32(&mut grid.at_mut(0).num_active_blocks, 1);
        active_blocks.at_mut(block_header_id as usize).virtual_id = *block;
        active_blocks
            .at_mut(block_header_id as usize)
            .first_particle = 0;
        active_blocks.at_mut(block_header_id as usize).num_particles = 0;
        hmap_entries.at_mut(slot as usize).value = BlockHeaderId {
            id: block_header_id,
        };
    }
}

/*
 * Index conversion functions.
 */

/// Converts a block header ID to a physical storage ID.
///
/// The physical ID is the index of the block's first node in the flat node arrays.
#[inline]
pub fn block_header_id_to_physical_id(hid: BlockHeaderId) -> BlockPhysicalId {
    BlockPhysicalId {
        id: hid.id * NUM_CELL_PER_BLOCK,
    }
}

/// Computes the physical node ID from a block's physical ID and a local offset within the block.
///
/// In 2D: nodes are laid out in row-major order within 8x8 blocks.
/// In 3D: nodes are laid out in row-major order within 4x4x4 blocks.
#[cfg(feature = "dim2")]
#[inline]
pub fn node_id(pid: BlockPhysicalId, shift_in_block: UVec2) -> NodePhysicalId {
    NodePhysicalId {
        id: pid.id + shift_in_block.x + shift_in_block.y * 8,
    }
}

/// Computes the physical node ID from a block's physical ID and a local offset within the block.
#[cfg(feature = "dim3")]
#[inline]
pub fn node_id(pid: BlockPhysicalId, shift_in_block: UVec3) -> NodePhysicalId {
    NodePhysicalId {
        id: pid.id + shift_in_block.x + shift_in_block.y * 4 + shift_in_block.z * 4 * 4,
    }
}

/*
 * Block-to-point association functions.
 */

/// Returns the primary block associated with a world-space point.
///
/// The associated block is determined by rounding the point's position to the nearest
/// cell, subtracting one (for the quadratic kernel offset), then dividing by the block
/// size (8 in 2D, 4 in 3D).
#[cfg(feature = "dim2")]
#[inline]
pub fn block_associated_to_point(cell_width: f32, pt: Vector) -> BlockVirtualId {
    let assoc_cell = (pt / cell_width).round() - Vector::ONE;
    let assoc_block = (assoc_cell / 8.0).floor();
    BlockVirtualId {
        id: IVec2::new(assoc_block.x as i32, assoc_block.y as i32),
    }
}

/// Returns the primary block associated with a world-space point.
#[cfg(feature = "dim3")]
#[inline]
pub fn block_associated_to_point(cell_width: f32, pt: Vector) -> BlockVirtualId {
    let assoc_cell = (pt / cell_width).round() - Vector::ONE;
    let assoc_block = (assoc_cell / 4.0).floor();
    BlockVirtualId {
        id: IVec3::new(
            assoc_block.x as i32,
            assoc_block.y as i32,
            assoc_block.z as i32,
        ),
        #[cfg(feature = "dim3")]
        padding: 0,
    }
}

/// Returns all blocks associated with a world-space point.
///
/// A particle's quadratic kernel stencil can span into neighboring blocks,
/// so we need to mark all potentially affected blocks as active.
/// This returns 4 blocks in 2D or 8 blocks in 3D.
#[inline]
pub fn blocks_associated_to_point(
    cell_width: f32,
    pt: Vector,
) -> [BlockVirtualId; NUM_ASSOC_BLOCKS] {
    let main_block = block_associated_to_point(cell_width, pt);
    blocks_associated_to_block(&main_block)
}

/// Returns all blocks neighboring a given block (including itself).
///
/// For a main block at position B, returns all blocks in the 2x2 (2D) or 2x2x2 (3D)
/// neighborhood starting at B.
#[cfg(feature = "dim2")]
#[inline]
pub fn blocks_associated_to_block(block: &BlockVirtualId) -> [BlockVirtualId; NUM_ASSOC_BLOCKS] {
    [
        BlockVirtualId {
            id: block.id + IVec2::new(0, 0),
        },
        BlockVirtualId {
            id: block.id + IVec2::new(0, 1),
        },
        BlockVirtualId {
            id: block.id + IVec2::new(1, 0),
        },
        BlockVirtualId {
            id: block.id + IVec2::new(1, 1),
        },
    ]
}

/// Returns all blocks neighboring a given block (including itself).
#[cfg(feature = "dim3")]
#[inline]
pub fn blocks_associated_to_block(block: &BlockVirtualId) -> [BlockVirtualId; NUM_ASSOC_BLOCKS] {
    [
        BlockVirtualId {
            id: block.id + IVec3::new(0, 0, 0),
            padding: 0,
        },
        BlockVirtualId {
            id: block.id + IVec3::new(0, 0, 1),
            padding: 0,
        },
        BlockVirtualId {
            id: block.id + IVec3::new(0, 1, 0),
            padding: 0,
        },
        BlockVirtualId {
            id: block.id + IVec3::new(0, 1, 1),
            padding: 0,
        },
        BlockVirtualId {
            id: block.id + IVec3::new(1, 0, 0),
            padding: 0,
        },
        BlockVirtualId {
            id: block.id + IVec3::new(1, 0, 1),
            padding: 0,
        },
        BlockVirtualId {
            id: block.id + IVec3::new(1, 1, 0),
            padding: 0,
        },
        BlockVirtualId {
            id: block.id + IVec3::new(1, 1, 1),
            padding: 0,
        },
    ]
}

/*
 * Affinity functions for CPIC.
 */

/// Checks if a specific collider's affinity bit is set.
#[inline]
pub fn affinity_bit(i_collider: u32, affinity: u32) -> bool {
    (affinity & (1 << i_collider)) != 0
}

/// Checks if a specific collider's sign bit is set.
#[inline]
pub fn sign_bit(i_collider: u32, affinity: u32) -> bool {
    ((affinity >> SIGN_BITS_SHIFT) & (1 << i_collider)) != 0
}

/// Checks if two affinity fields are compatible (same sign for all shared affinities).
///
/// Two nodes/particles are compatible if, for every collider they both have affinity to,
/// they agree on the sign (i.e., they are on the same side of the collider surface).
#[inline]
pub fn affinities_are_compatible(affinity1: u32, affinity2: u32) -> bool {
    let affinities_in_common = affinity1 & affinity2 & AFFINITY_BITS_MASK;
    let signs1 = (affinity1 >> SIGN_BITS_SHIFT) & affinities_in_common;
    let signs2 = (affinity2 >> SIGN_BITS_SHIFT) & affinities_in_common;
    signs1 == signs2
}

/*
 * Helper: integer division rounding up.
 */
#[inline]
fn div_ceil(x: u32, y: u32) -> u32 {
    (x + y - 1) / y
}

/*
 * Entry points.
 */

/// Resets all hashmap entries to the empty state and clears the active block count.
///
/// Each thread resets one hashmap slot (state, key, value, and ownership flag).
/// Thread 0 also resets `num_active_blocks` to 0.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_reset_hmap(
    #[spirv(global_invocation_id)] invocation_id: spirv_std::glam::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] grid_data: &mut Grid,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    hmap_entries: &mut [GridHashMapEntry],
) {
    let id = invocation_id.x;

    if id < grid_data.hmap_capacity {
        let entry = hmap_entries.at_mut(id as usize);
        entry.state = NONE;
        // Reset ownership so the next frame's insertions can claim slots.
        entry.ownership = 0;
        // Resetting the following isn't necessary for correctness,
        // but it makes debugging easier.
        entry.key = BlockVirtualId {
            id: IVector::ZERO,
            #[cfg(feature = "dim3")]
            padding: 0,
        };
        entry.value = BlockHeaderId { id: 0 };
    }
    if id == 0 {
        grid_data.num_active_blocks = 0;
    }
}

/// Computes indirect dispatch sizes based on the number of active blocks.
///
/// Produces two sets of dispatch arguments:
/// - `n_block_groups`: for per-block dispatches (ceil(num_active_blocks / GRID_WORKGROUP_SIZE))
/// - `n_g2p_p2g_groups`: for P2G/G2P dispatches (one workgroup per active block)
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_init_indirect_workgroups(
    #[spirv(global_invocation_id)] _invocation_id: spirv_std::glam::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] grid_data: &[Grid],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] n_block_groups: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] n_g2p_p2g_groups: &mut [u32],
) {
    let num_active_blocks = grid_data.at(0).num_active_blocks;
    n_block_groups.write(0, div_ceil(num_active_blocks, GRID_WORKGROUP_SIZE));
    n_block_groups.write(1, 1);
    n_block_groups.write(2, 1);
    n_g2p_p2g_groups.write(0, num_active_blocks);
    n_g2p_p2g_groups.write(1, 1);
    n_g2p_p2g_groups.write(2, 1);
}

/// Resets all grid nodes and linked lists for the current set of active blocks.
///
/// Each thread resets one node. Clears momentum, velocity, mass, CDF data,
/// and both particle and rigid particle linked lists.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_reset(
    #[spirv(global_invocation_id)] invocation_id: spirv_std::glam::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] grid_data: &[Grid],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] nodes: &mut [Node],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    nodes_linked_lists: &mut [NodeLinkedList],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    rigid_nodes_linked_lists: &mut [NodeLinkedList],
) {
    let i = invocation_id.x;
    let num_nodes = grid_data.at(0).num_active_blocks * NUM_CELL_PER_BLOCK;
    if i < num_nodes {
        let idx = i as usize;
        let node = nodes.at_mut(idx);
        node.momentum_velocity_mass = VectorPlusOne::ZERO;
        node.momentum_velocity_mass_incompatible = VectorPlusOne::ZERO;
        node.cdf = NodeCdf::new(0.0, 0, NONE);

        let ll = nodes_linked_lists.at_mut(idx);
        ll.head = NONE;
        ll.len = 0;

        let rll = rigid_nodes_linked_lists.at_mut(idx);
        rll.head = NONE;
        rll.len = 0;
    }
}
