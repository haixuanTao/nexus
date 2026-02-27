//! Initialize Indirect Dispatch Arguments for Radix Sort
//!
//! Sets up the indirect dispatch arguments based on the number of keys to sort.

use khal_derive::spirv_bindgen;
use spirv_std::spirv;

use crate::MaybeIndexUnchecked;

use super::sorting::{div_ceil, BIN_COUNT, BLOCK_SIZE};

/// GPU kernel to initialize indirect dispatch arguments for radix sort.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_init_sort_dispatch(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] num_keys_arr: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] sort_dispatch: &mut [u32; 3],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] reduce_dispatch: &mut [u32; 3],
) {
    let num_keys = num_keys_arr.read(0);
    let num_wgs = div_ceil(num_keys, BLOCK_SIZE);
    let reduce_wgs = BIN_COUNT * div_ceil(num_wgs, BLOCK_SIZE);

    sort_dispatch.write(0, num_wgs);
    sort_dispatch.write(1, 1);
    sort_dispatch.write(2, 1);
    reduce_dispatch.write(0, if reduce_wgs > 0 { reduce_wgs } else { 1 });
    reduce_dispatch.write(1, 1);
    reduce_dispatch.write(2, 1);
}
