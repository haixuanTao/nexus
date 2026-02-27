//! Ball (Sphere/Circle) Shape Module
//!
//! This module provides geometric operations for balls (spheres in 3D, circles in 2D).
//! A ball is defined by its radius and is centered at the origin in its local frame.

use crate::bounding_volumes::Aabb;
use crate::queries::ProjectionResult;
use crate::{Pose, Vector};

/// A ball (sphere in 3D, circle in 2D) defined by its radius.
///
/// The ball is centered at the origin in its local coordinate frame.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Ball {
    /// The radius of the self. Must be positive.
    pub radius: f32,
}

impl Ball {
    /// Creates a new ball with the given radius.
    #[inline]
    pub fn new(radius: f32) -> Self {
        Self { radius }
    }

    /// Projects a point onto a ball in its local coordinate frame.
    ///
    /// If the point is inside the ball, it is returned unchanged.
    /// Otherwise, the closest point on the ball's surface is returned.
    ///
    /// Parameters:
    /// - ball: The ball shape.
    /// - pt: The point to project (in the ball's local frame).
    ///
    /// Returns: The projected point (in the ball's local frame).
    #[inline]
    pub fn project_local_point(&self, pt: Vector) -> Vector {
        let dist = pt.length();

        if dist > self.radius && dist != 0.0 {
            pt * (self.radius / dist)
        } else {
            pt
        }
    }

    /// Projects a point onto a transformed ball in world space.
    ///
    /// Parameters:
    /// - ball: The ball shape.
    /// - pose: The ball's world-space pose.
    /// - pt: The point to project (in world space).
    ///
    /// Returns: The projected point (in world space).
    #[inline]
    pub fn project_point(&self, pose: Pose, pt: Vector) -> Vector {
        let local_pt = pose.inverse_transform_point(pt);
        pose * self.project_local_point(local_pt)
    }

    /// Projects a point onto the boundary (surface) of a ball in its local frame.
    ///
    /// This always projects onto the ball's surface, even if the point is inside.
    ///
    /// Parameters:
    /// - ball: The ball shape.
    /// - pt: The point to project (in the ball's local frame).
    ///
    /// Returns: ProjectionResult containing the surface point and is_inside flag.
    ///
    /// Special case: If the point is at the ball's center (origin),
    /// an arbitrary direction is chosen for projection.
    #[inline]
    pub fn project_local_point_on_boundary(&self, pt: Vector) -> ProjectionResult {
        let dist = pt.length();

        if dist != 0.0 {
            let is_inside = dist < self.radius;
            ProjectionResult::new(pt * (self.radius / dist), is_inside)
        } else {
            // Point is at the center, pick an arbitrary direction
            let proj = Vector::Y * self.radius;
            ProjectionResult::new(proj, true)
        }
    }

    /// Projects a point onto the boundary of a transformed ball in world space.
    ///
    /// Parameters:
    /// - ball: The ball shape.
    /// - pose: The ball's world-space pose.
    /// - pt: The point to project (in world space).
    ///
    /// Returns: ProjectionResult with point in world space and is_inside flag.
    #[inline]
    pub fn project_point_on_boundary(&self, pose: Pose, pt: Vector) -> ProjectionResult {
        let local_pt = pose.inverse_transform_point(pt);
        let mut result = self.project_local_point_on_boundary(local_pt);
        result.point = pose * result.point;
        result
    }

    /// Computes the local support point of a ball in a given direction.
    ///
    /// The support point is the point on the ball's surface that is
    /// furthest in the given direction.
    ///
    /// Parameters:
    /// - ball: The ball shape.
    /// - dir: The search direction (does not need to be normalized).
    ///
    /// Returns: The support point in local coordinates.
    #[inline]
    pub fn local_support_point(&self, dir: Vector) -> Vector {
        let dir_len = dir.length();

        if dir_len != 0.0 {
            dir * (self.radius / dir_len)
        } else {
            Vector::Y * self.radius
        }
    }

    /// Computes the support point of a transformed ball in a given direction.
    ///
    /// Parameters:
    /// - ball: The ball shape.
    /// - pose: The ball's pose.
    /// - dir: The search direction in world space.
    ///
    /// Returns: The support point in world coordinates.
    #[inline]
    pub fn support_point(&self, pose: Pose, dir: Vector) -> Vector {
        pose * self.local_support_point(dir)
    }

    /// Computes the AABB of a ball (centered at origin in local space).
    #[inline]
    pub fn aabb(&self) -> Aabb {
        let half_extents = Vector::splat(self.radius);
        Aabb::new(-half_extents, half_extents)
    }
}
