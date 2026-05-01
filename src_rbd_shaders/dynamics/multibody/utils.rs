//! Small math / coordinate helpers shared across multibody kernels.

use glamx::{Quat, Vec3};

use crate::Pose;
use crate::dynamics::joint::{ANG_AXES_MASK, LIN_AXES_MASK};

use super::types::{MAX_JOINT_DOFS, MultibodyLinkStatic, MultibodyLinkWorkspace};

/// i-th cartesian basis vector. Branches on value to avoid SPIR-V pointer phis.
#[inline]
pub fn basis_vec3(i: u32) -> Vec3 {
    if i == 0 {
        Vec3::X
    } else if i == 1 {
        Vec3::Y
    } else {
        Vec3::Z
    }
}

/// Read index `i` (0..=5) of a `[f32; MAX_JOINT_DOFS]` by value.
#[inline]
pub fn coord_get(arr: &[f32; MAX_JOINT_DOFS], i: u32) -> f32 {
    if i == 0 {
        arr[0]
    } else if i == 1 {
        arr[1]
    } else if i == 2 {
        arr[2]
    } else if i == 3 {
        arr[3]
    } else if i == 4 {
        arr[4]
    } else {
        arr[5]
    }
}

/// Write index `i` (0..=5) of a `[f32; MAX_JOINT_DOFS]`.
#[inline]
pub fn coord_set(arr: &mut [f32; MAX_JOINT_DOFS], i: u32, v: f32) {
    if i == 0 {
        arr[0] = v;
    } else if i == 1 {
        arr[1] = v;
    } else if i == 2 {
        arr[2] = v;
    } else if i == 3 {
        arr[3] = v;
    } else if i == 4 {
        arr[4] = v;
    } else {
        arr[5] = v;
    }
}

/// Number of free DOFs implied by a `locked_axes` bitmask.
#[inline]
pub fn count_free_dofs(locked: u32) -> u32 {
    6 - (locked & 0x3f).count_ones()
}

/// Number of free linear DOFs (bits 0..3).
#[inline]
pub fn count_free_lin_dofs(locked: u32) -> u32 {
    3 - (locked & LIN_AXES_MASK).count_ones()
}

/// Number of free angular DOFs (bits 3..6).
#[inline]
pub fn count_free_ang_dofs(locked: u32) -> u32 {
    3 - ((locked & ANG_AXES_MASK) >> 3).count_ones()
}

/// Compute the link's `local_to_parent` pose given its current joint coords/rotation.
///
/// Mirrors rapier's `MultibodyJoint::body_to_parent`: starts from `joint_rot * local_frame_b⁻¹`,
/// prepends a translation for each free linear DOF, and finally composes with `local_frame_a`.
pub fn body_to_parent(stat: &MultibodyLinkStatic, ws: &MultibodyLinkWorkspace) -> Pose {
    let locked = stat.data.locked_axes;
    let mut transform = Pose::from_parts(Vec3::ZERO, ws.joint_rot) * stat.data.local_frame_b.inverse();

    for i in 0u32..3 {
        if (locked & (1 << i)) == 0 {
            let t = basis_vec3(i) * coord_get(&ws.coords, i);
            transform = Pose::from_parts(t, Quat::IDENTITY) * transform;
        }
    }

    stat.data.local_frame_a * transform
}
