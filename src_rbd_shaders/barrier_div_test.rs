//! Bisect B: TWO sequential `if lane==0`+barrier sections, fully INLINE (no fn call).
use khal_std::glamx::UVec3;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;
use khal_std::index::MaybeIndexUnchecked;

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_barrier_div_test(
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] out: &mut [u32],
    #[spirv(workgroup)] s: &mut [u32; 64],
) {
    let lane = lid.x;
    if lane == 0 {
        let mut i = 0u32;
        while i < 64 { s.write(i as usize, i.wrapping_mul(7)); i += 1; }
    }
    workgroup_memory_barrier_with_group_sync();
    let a = s.read(lane as usize);
    if lane == 0 {
        let mut i = 0u32;
        while i < 64 { s.write(i as usize, i.wrapping_mul(3)); i += 1; }
    }
    workgroup_memory_barrier_with_group_sync();
    let b = s.read(lane as usize);
    out.write(lane as usize, a.wrapping_add(b));
}
