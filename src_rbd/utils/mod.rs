//! Utility functions and data structures for GPU-accelerated algorithms.
//!
//! This module provides general-purpose GPU algorithms that support the collision
//! detection and physics simulation pipelines.

pub use radix_sort::{RadixSort, RadixSortWorkspace};
pub use prefix_sum::{GpuPrefixSum, PrefixSumWorkspace};

mod radix_sort;
mod prefix_sum;