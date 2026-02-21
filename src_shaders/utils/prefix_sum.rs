//! Prefix sum compute shader kernels
//!
//! This module contains the actual GPU compute shader entry points for parallel prefix sum.
//!
//! What is Prefix Sum?
//! Given an input array [a0, a1, a2, ...], compute output [0, a0, a0+a1, a0+a1+a2, ...]
//! Each output element is the sum of the element with all elements before it.
//!
//! Use in rapier:
//! Used to compute per-body constraint ranges from constraint counts:
//! - Input: [3, 2, 5, 1] (number of constraints per body)
//! - Output: [0, 3, 5, 10] (end index of constraints for each body)
//! - Body i's constraints are at indices [output[i-1], output[i])

use khal_derive::spirv_bindgen;
use spirv_std::arch::workgroup_memory_barrier_with_group_sync;
use spirv_std::glam::UVec3;
use spirv_std::spirv;

use crate::MaybeIndexUnchecked;

/// Workgroup size: number of elements processed per workgroup.
pub const WORKGROUP_SIZE: usize = 256;

/// Performs exclusive prefix sum on a segment of the data array.
///
/// This kernel uses workgroup-shared memory for the tree-based scan algorithm.
/// Each workgroup processes WORKGROUP_SIZE elements.
///
/// Algorithm:
/// - Phase 1: Up-sweep (reduce) - builds a tree of partial sums
/// - Phase 2: Down-sweep - transforms the tree into an exclusive scan
#[spirv_bindgen]
#[spirv(compute(threads(256)))]
pub fn gpu_prefix_sum_sweep(
    #[spirv(local_invocation_id)] thread_id: UVec3,
    #[spirv(workgroup_id)] block_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] aux: &mut [u32],
    #[spirv(workgroup)] workspace: &mut [u32; 256],
) {
    let bid = block_id.x as usize;
    let tid = thread_id.x as usize;
    let data_len = data.len();

    // Global index for this thread's element
    let elt_id = tid + bid * WORKGROUP_SIZE;

    // Phase 0: Load data into shared memory
    if elt_id < data_len {
        workspace.write(tid, data.read(elt_id));
    } else {
        // Pad with zeros for out-of-bounds threads
        workspace.write(tid, 0);
    }

    // Phase 1: Up-sweep (reduce) - build tree of partial sums
    // After this phase, workspace[WORKGROUP_SIZE-1] contains the total sum.
    // NOTE: we always process the full WORKGROUP_SIZE (a power of two) instead of
    //       rounding up the actual block length. Out-of-bounds elements are zero-padded
    //       so the result is correct, and using a compile-time constant avoids non-uniform
    //       control flow that breaks workgroup barriers on WebGPU.
    {
        let mut d = WORKGROUP_SIZE / 2;
        let mut offset = 1usize;
        // log2(256) = 8 iterations
        for _ in 0..8u32 {
            workgroup_memory_barrier_with_group_sync();
            if tid < d {
                let ia = tid * 2 * offset + offset - 1;
                let ib = (tid * 2 + 1) * offset + offset - 1;

                let sum = workspace.read(ia) + workspace.read(ib);
                workspace.write(ib, sum);
            }

            d /= 2;
            offset *= 2;
        }
    }

    // Thread 0 saves the total sum and clears the root for down-sweep
    if tid == 0 {
        let total_sum = workspace.read(WORKGROUP_SIZE - 1);
        aux.write(bid, total_sum);
        workspace.write(WORKGROUP_SIZE - 1, 0);
    }

    // Phase 2: Down-sweep - propagate partial sums down the tree
    // Transforms the tree into an exclusive scan
    {
        let mut d = 1usize;
        let mut offset = WORKGROUP_SIZE / 2;
        // log2(256) = 8 iterations
        for _ in 0..8u32 {
            workgroup_memory_barrier_with_group_sync();
            if tid < d {
                let ia = tid * 2 * offset + offset - 1;
                let ib = (tid * 2 + 1) * offset + offset - 1;

                let a = workspace.read(ia);
                let b = workspace.read(ib);

                workspace.write(ia, b);
                workspace.write(ib, a + b);
            }

            d *= 2;
            offset /= 2;
        }
    }

    // Synchronize before writing results
    workgroup_memory_barrier_with_group_sync();

    // Write results back to global memory
    if elt_id < data_len {
        data.write(elt_id, workspace.read(tid));
    }
}

/// Adds per-block offsets to complete multi-block prefix sum.
///
/// After each block computes its local prefix sum, we need to add the total
/// sum from all previous blocks to each element. This kernel adds aux[bid-1]
/// (the sum of all blocks before this one) to each element in block bid.
#[spirv_bindgen]
#[spirv(compute(threads(256)))]
pub fn gpu_add_data_grp(
    #[spirv(global_invocation_id)] thread_id: UVec3,
    #[spirv(workgroup_id)] block_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] aux: &[u32],
) {
    let tid = thread_id.x as usize;
    let bid = block_id.x as usize;
    let data_len = data.len();

    if tid < data_len {
        // Add the cumulative sum from all previous blocks
        *data.at_mut(tid) += aux.read(bid);
    }
}
