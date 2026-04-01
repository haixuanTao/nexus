//! Ball (sphere in 3D, circle in 2D) shape.

use crate::bounding_volumes::Aabb;
use crate::queries::ProjectionResult;
use crate::{Pose, Vector};

/// A ball (sphere in 3D, circle in 2D) defined by its radius.
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
    #[inline]
    pub fn project_point(&self, pose: Pose, pt: Vector) -> Vector {
        let local_pt = pose.inverse_transform_point(pt);
        pose * self.project_local_point(local_pt)
    }

    /// Projects a point onto the boundary (surface) of a ball in its local frame.
    ///
    /// Always projects onto the surface, even if the point is inside.
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
    #[inline]
    pub fn project_point_on_boundary(&self, pose: Pose, pt: Vector) -> ProjectionResult {
        let local_pt = pose.inverse_transform_point(pt);
        let mut result = self.project_local_point_on_boundary(local_pt);
        result.point = pose * result.point;
        result
    }

    /// Computes the local support point of a ball in a given direction.
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
