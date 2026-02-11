//! Parallel prefix sum (exclusive scan) compute shaders.
//!
//! Implements a two-pass parallel prefix sum using workgroup shared memory.
//! The algorithm works in three stages:
//! 1. `gpu_prefix_sum`: Each workgroup computes a local exclusive scan and writes
//!    its total sum to an auxiliary buffer.
//! 2. The auxiliary buffer is itself scanned (by running `gpu_prefix_sum` again on it).
//! 3. `gpu_add_data_grp`: Each workgroup adds the corresponding auxiliary value to
//!    complete the global prefix sum.
//!
//! Uses the Blelloch up-sweep / down-sweep algorithm within each workgroup.

use crate::MaybeIndexUnchecked;
use khal_derive::spirv_bindgen;
use spirv_std::arch::workgroup_memory_barrier_with_group_sync;
use spirv_std::glam::UVec3;
use spirv_std::spirv;

/// Workgroup size for the prefix sum kernels.
pub const WORKGROUP_SIZE: usize = 256;

/// Computes the next power of two greater than or equal to `val`.
///
/// See Bit Twiddling Hacks:
/// <https://graphics.stanford.edu/%7Eseander/bithacks.html#RoundUpPowerOf2>
#[inline]
fn next_power_of_two(val: u32) -> u32 {
    let mut v = val;
    v = v.wrapping_sub(1);
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v = v.wrapping_add(1);
    v
}

/// Performs an exclusive prefix sum (scan) on a segment of the data array.
///
/// Each workgroup processes `WORKGROUP_SIZE` elements using workgroup shared memory.
/// The algorithm uses an up-sweep (reduce) phase followed by a down-sweep phase
/// to compute the exclusive scan. The total sum for each workgroup is written to
/// the `aux` buffer for the next pass.
#[spirv_bindgen]
#[spirv(compute(threads(256)))]
pub fn gpu_prefix_sum(
    #[spirv(local_invocation_id)] thread_id: UVec3,
    #[spirv(workgroup_id)] block_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] aux: &mut [u32],
    #[spirv(workgroup)] workspace: &mut [u32; 256],
) {
    let bid = block_id.x as usize;
    let tid = thread_id.x as usize;
    let data_len = data.len();

    if bid * WORKGROUP_SIZE >= data_len {
        return;
    }

    let data_block_len = (data_len - bid * WORKGROUP_SIZE) as u32;
    let shared_len = if next_power_of_two(data_block_len) < 1 {
        1u32
    } else if next_power_of_two(data_block_len) > WORKGROUP_SIZE as u32 {
        WORKGROUP_SIZE as u32
    } else {
        next_power_of_two(data_block_len)
    };
    let shared_len = shared_len as usize;
    let elt_id = tid + bid * WORKGROUP_SIZE;

    // Load data into shared memory.
    if elt_id < data_len {
        workspace.write(tid, data.read(elt_id));
    } else {
        workspace.write(tid, 0);
    }

    // Up-sweep (reduce) phase.
    {
        let mut d = shared_len / 2;
        let mut offset = 1usize;
        for _ in 0..8 {
            if d == 0 {
                break;
            }
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

    if tid == 0 {
        let total_sum = workspace.read(shared_len - 1);
        aux.write(bid, total_sum);
        workspace.write(shared_len - 1, 0);
    }

    // Down-sweep phase.
    {
        let mut d = 1usize;
        let mut offset = shared_len / 2;
        for _ in 0..8 {
            if d >= shared_len {
                break;
            }
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

    workgroup_memory_barrier_with_group_sync();

    if elt_id < data_len {
        data.write(elt_id, workspace.read(tid));
    }
}

/// Adds per-workgroup offsets to complete the multi-pass prefix sum.
///
/// After each workgroup has computed its local prefix sum, this kernel adds the
/// cumulative sum from all previous workgroups (stored in `aux`) to each element.
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
        *data.at_mut(tid) += aux.read(bid);
    }
}
