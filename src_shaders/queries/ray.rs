//! Ray Definition and Operations
//!
//! This module provides the ray structure and basic ray operations.
//!
//! A ray is defined by an origin point and a direction vector.
//! The ray represents all points: origin + t * dir for t >= 0.

use crate::Vector;

/// A ray defined by an origin point and direction.
///
/// Represents a half-line starting at 'origin' and extending infinitely
/// in the 'dir' direction.
///
/// Note: The direction vector does not need to be normalized.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Ray {
    /// The ray's starting point.
    pub origin: Vector,
    /// The ray's direction vector.
    pub dir: Vector,
}

impl Ray {
    /// Creates a new ray from origin and direction.
    #[inline]
    pub fn new(origin: Vector, dir: Vector) -> Self {
        Self { origin, dir }
    }

    /// Computes a point on the ray at parameter t.
    ///
    /// Returns: The point at `origin + t * dir`.
    #[inline]
    pub fn point_at(&self, t: f32) -> Vector {
        self.origin + self.dir * t
    }
}

/// Computes a point on the ray at parameter t.
///
/// Parameters:
/// - ray: The ray.
/// - t: The parameter (t >= 0 for points on the ray).
///
/// Returns: The point at `origin + t * dir`.
#[inline]
pub fn pt_at(ray: &Ray, t: f32) -> Vector {
    ray.point_at(t)
}
