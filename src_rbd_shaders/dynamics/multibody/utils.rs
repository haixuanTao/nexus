//! Small math / coordinate helpers shared across multibody kernels.

#[cfg(feature = "dim3")]
use glamx::Vec3;
#[cfg(feature = "dim2")]
use glamx::Vec2;

use khal_std::index::MaybeIndexUnchecked;
use parry::math::VectorExt;
use crate::dynamics::joint::{ANG_AXES_MASK, LIN_AXES_MASK};
use crate::{DIM, Pose, Rotation, Vector};

use super::types::{MAX_JOINT_DOFS, MultibodyLinkStatic, MultibodyLinkWorkspace};

/// Number of free DOFs implied by a `locked_axes` bitmask.
#[inline]
pub fn count_free_dofs(locked: u32) -> u32 {
    crate::dynamics::joint::SPATIAL_DIM as u32
        - (locked & ((LIN_AXES_MASK | ANG_AXES_MASK) as u32)).count_ones()
}

/// Number of free linear DOFs (bits 0..DIM).
#[inline]
pub fn count_free_lin_dofs(locked: u32) -> u32 {
    DIM - (locked & LIN_AXES_MASK).count_ones()
}

/// Number of free angular DOFs.
#[inline]
pub fn count_free_ang_dofs(locked: u32) -> u32 {
    crate::ANG_DIM - (locked & ANG_AXES_MASK).count_ones()
}

/// Compute the link's `local_to_parent` pose given its current joint coords/rotation.
///
/// Mirrors rapier's `MultibodyJoint::body_to_parent`: starts from `joint_rot * local_frame_b⁻¹`,
/// prepends a translation for each free linear DOF, and finally composes with `local_frame_a`.
pub fn body_to_parent(stat: &MultibodyLinkStatic, link: &MultibodyLinkWorkspace) -> Pose {
    let locked = stat.data.locked_axes;
    let mut transform =
        Pose::from_parts(Vector::ZERO, link.joint_rot) * stat.data.local_frame_b.inverse();

    for i in 0..DIM {
        if (locked & (1 << i)) == 0 {
            let t = Vector::ith(i as usize, link.coords.read(i as usize));
            transform = Pose::from_parts(t, Rotation::IDENTITY) * transform;
        }
    }

    stat.data.local_frame_a * transform
}
