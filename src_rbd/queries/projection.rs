//! Point projection queries for distance and closest point calculations.
//!
//! Point projection finds the closest point on a shape's surface to a given query point.
//! This is fundamental for:
//! - Distance queries between shapes
//! - Penetration depth calculations
//! - Closest point computations
//! - Contact point generation

use crate::math::Point;

/// Result of a point projection query, GPU-compatible layout.
///
/// Contains the closest point on the shape's surface and whether the query point
/// was inside the shape.
///
/// # Fields
///
/// - `point`: The projected point (closest point on the shape's surface)
/// - `is_inside`: Whether the query point was inside the shape (non-zero = inside, 0 = outside)
/// - `padding` (2D only): Alignment padding for GPU buffer compatibility
///
/// # Memory Layout
///
/// This struct is `#[repr(C)]` and implements `bytemuck::Pod`, making it suitable for
/// direct GPU buffer uploads and downloads.
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GpuProjectionResult {
    /// The projected point on the shape's surface.
    ///
    /// This is the point on the shape's boundary that is closest to the query point.
    pub point: Point,

    /// Whether the query point was inside the shape.
    ///
    /// - `0`: Query point is outside the shape
    /// - Non-zero: Query point is inside the shape
    pub is_inside: u32,

    #[cfg(feature = "dim2")]
    /// Padding for 2D builds to maintain alignment (unused).
    pub padding: u32,
}
