//! Cuboid (Box/Rectangle) Shape Module
//!
//! This module provides geometric operations for axis-aligned boxes.
//! A cuboid is defined by its half-extents along each dimension.

use crate::queries::{PolygonalFeature, ProjectionResult};
use crate::{Pose, Vector};

/// A cuboid (box in 3D, rectangle in 2D) defined by half-extents.
///
/// The cuboid is centered at the origin in its local frame.
/// Each dimension extends from -halfExtents to +halfExtents.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Cuboid {
    /// Half-widths along each axis.
    /// e.g., halfExtents = (1, 2, 3) means the box extends from
    /// (-1, -2, -3) to (1, 2, 3) in local coordinates.
    pub half_extents: Vector,
}

impl Cuboid {
    /// Creates a new cuboid with the given half-extents.
    #[inline]
    pub fn new(half_extents: Vector) -> Self {
        Self { half_extents }
    }

    /// Projects a point on a box.
    ///
    /// If the point is inside the box, the point itself is returned.
    #[inline]
    pub fn project_local_point(&self, pt: Vector) -> Vector {
        let mins = -self.half_extents;
        let maxs = self.half_extents;

        let mins_pt = mins - pt; // -hext - pt
        let pt_maxs = pt - maxs; // pt - hext
        let shift = mins_pt.max(Vector::ZERO) - pt_maxs.max(Vector::ZERO);

        pt + shift
    }

    /// Projects a point on a transformed box.
    ///
    /// If the point is inside the box, the point itself is returned.
    #[inline]
    pub fn project_point(&self, pose: Pose, pt: Vector) -> Vector {
        let local_pt = pose.inverse_transform_point(pt);
        pose * self.project_local_point(local_pt)
    }

    /// Projects a point on the boundary of a box.
    #[inline]
    pub fn project_local_point_on_boundary(&self, pt: Vector) -> ProjectionResult {
        let out_proj = self.project_local_point(pt);

        // Projection if the point is inside the box.
        // Note: Vector::copysign uses llvm.copysign which isn't supported by NVVM.
        // Use a manual implementation that works on all targets.
        let pt_sgn_with_zero = Vector::select(pt.cmpge(Vector::ZERO), Vector::ONE, -Vector::ONE);
        // This is the sign of pt, or -1 for components that were zero.
        // This bias is arbitrary (we could have picked +1), but we picked it so
        // it matches the bias that's in parry.
        let pt_sgn = pt_sgn_with_zero + (pt_sgn_with_zero.abs() - Vector::ONE);
        let diff = self.half_extents - pt_sgn * pt;

        #[cfg(feature = "dim2")]
        let in_proj = {
            let pick_x = diff.x <= diff.y;
            let shift_x = Vector::new(diff.x * pt_sgn.x, 0.0);
            let shift_y = Vector::new(0.0, diff.y * pt_sgn.y);
            let pen_shift = if pick_x { shift_x } else { shift_y };
            pt + pen_shift
        };

        #[cfg(feature = "dim3")]
        let in_proj = {
            let pick_x = diff.x <= diff.y && diff.x <= diff.z;
            let pick_y = diff.y <= diff.x && diff.y <= diff.z;
            let shift_x = Vector::new(diff.x * pt_sgn.x, 0.0, 0.0);
            let shift_y = Vector::new(0.0, diff.y * pt_sgn.y, 0.0);
            let shift_z = Vector::new(0.0, 0.0, diff.z * pt_sgn.z);
            let pen_shift = if pick_x {
                shift_x
            } else if pick_y {
                shift_y
            } else {
                shift_z
            };
            pt + pen_shift
        };

        // Select between in and out proj.
        let is_inside = pt == out_proj;
        ProjectionResult::new(if is_inside { in_proj } else { out_proj }, is_inside)
    }

    /// Project a point of a transformed box's boundary.
    ///
    /// If the point is inside of the box, it will be projected on its boundary but
    /// `ProjectionResult::is_inside` will be set to `true`.
    #[inline]
    pub fn project_point_on_boundary(&self, pose: Pose, pt: Vector) -> ProjectionResult {
        let local_pt = pose.inverse_transform_point(pt);
        let mut result = self.project_local_point_on_boundary(local_pt);
        result.point = pose * result.point;
        result
    }

    /// Computes the local support point of a cuboid.
    #[inline]
    pub fn local_support_point(&self, axis: Vector) -> Vector {
        #[cfg(feature = "dim2")]
        {
            Vector::new(
                if axis.x >= 0.0 {
                    self.half_extents.x
                } else {
                    -self.half_extents.x
                },
                if axis.y >= 0.0 {
                    self.half_extents.y
                } else {
                    -self.half_extents.y
                },
            )
        }
        #[cfg(feature = "dim3")]
        {
            Vector::new(
                if axis.x >= 0.0 {
                    self.half_extents.x
                } else {
                    -self.half_extents.x
                },
                if axis.y >= 0.0 {
                    self.half_extents.y
                } else {
                    -self.half_extents.y
                },
                if axis.z >= 0.0 {
                    self.half_extents.z
                } else {
                    -self.half_extents.z
                },
            )
        }
    }

    /// Computes the support face of a 2D cuboid (rectangle).
    #[cfg(feature = "dim2")]
    #[inline]
    pub fn support_face(&self, axis: Vector) -> PolygonalFeature {
        let he = self.half_extents;
        let abs_dir = axis.abs();

        if abs_dir.x >= abs_dir.y {
            let sign = if axis.x > 0.0 { 1.0 } else { -1.0 };
            PolygonalFeature {
                vertices: [
                    Vector::new(he.x * sign, he.y),
                    Vector::new(he.x * sign, -he.y),
                ],
                num_vertices: 2,
            }
        } else {
            let sign = if axis.y > 0.0 { 1.0 } else { -1.0 };
            PolygonalFeature {
                vertices: [
                    Vector::new(he.x, he.y * sign),
                    Vector::new(-he.x, he.y * sign),
                ],
                num_vertices: 2,
            }
        }
    }

    /// Computes the support face of a 3D cuboid (box).
    #[cfg(feature = "dim3")]
    #[inline]
    pub fn support_face(&self, axis: Vector) -> PolygonalFeature {
        let he = self.half_extents;
        let abs_dir = axis.abs();
        let mut iamax = 2u32;

        if abs_dir.x >= abs_dir.y && abs_dir.x >= abs_dir.z {
            iamax = 0;
        } else if abs_dir.y >= abs_dir.x && abs_dir.y >= abs_dir.z {
            iamax = 1;
        }

        let sign = if match iamax {
            0 => axis.x,
            1 => axis.y,
            _ => axis.z,
        } > 0.0
        {
            1.0
        } else {
            -1.0
        };

        // TODO PERF: avoid branching using some index arithmetic?
        match iamax {
            0 => PolygonalFeature {
                vertices: [
                    Vector::new(he.x * sign, he.y, he.z),
                    Vector::new(he.x * sign, -he.y, he.z),
                    Vector::new(he.x * sign, -he.y, -he.z),
                    Vector::new(he.x * sign, he.y, -he.z),
                ],
                num_vertices: 4,
            },
            1 => PolygonalFeature {
                vertices: [
                    Vector::new(he.x, he.y * sign, he.z),
                    Vector::new(-he.x, he.y * sign, he.z),
                    Vector::new(-he.x, he.y * sign, -he.z),
                    Vector::new(he.x, he.y * sign, -he.z),
                ],
                num_vertices: 4,
            },
            _ => PolygonalFeature {
                vertices: [
                    Vector::new(he.x, he.y, he.z * sign),
                    Vector::new(he.x, -he.y, he.z * sign),
                    Vector::new(-he.x, -he.y, he.z * sign),
                    Vector::new(-he.x, he.y, he.z * sign),
                ],
                num_vertices: 4,
            },
        }
    }
}
