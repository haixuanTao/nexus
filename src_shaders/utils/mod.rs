//! Utility algorithms.

mod basis;
pub mod radix_sort;
pub mod prefix_sum;

pub use basis::*;

/// Division with ceiling (signed).
pub fn div_ceil(x: i32, y: i32) -> i32 {
    (x + y - 1) / y
}

/// Division with ceiling (unsigned).
pub fn udiv_ceil(x: u32, y: u32) -> u32 {
    (x + y - 1) / y
}