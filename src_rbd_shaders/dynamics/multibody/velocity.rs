//! Velocity propagation (rapier's `update_dynamics` velocity phase).
//!
//! Computes per-link world-space `joint_velocity` and total `rb_vels` by walking
//! links parent-before-child, so that the Coriolis assembly can read them.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

use parry::math::VectorExt;
use crate::dynamics::body::{LocalMassProperties, Velocity};
use crate::utils::{Slice, SliceMut};
use crate::{ANG_DIM, AngVector, DIM, Vector, gcross_av};

use super::types::{MultibodyInfo, MultibodyLinkStatic, MultibodyLinkWorkspace};

/// Body-local velocity contributed by this joint, reading the joint's free-DOF
/// velocities directly from `vel_slice[assembly_id..]` rather than via a stack
/// array. The stack version forces a `[f32; MAX_JOINT_DOFS]` Function-storage
/// variable which spills to private memory.
#[inline]
fn jacobian_mul_coordinates(
    locked_axes: u32,
    assembly_id: u32,
    vel_slice: &Slice<f32>,
) -> (Vector, AngVector) {
    let mut lin = Vector::ZERO;
    #[cfg(feature = "dim3")]
    let mut ang = AngVector::ZERO;
    #[cfg(feature = "dim2")]
    let mut ang: AngVector = 0.0;
    let mut curr = 0u32;

    for i in 0..DIM {
        if (locked_axes & (1 << i)) == 0 {
            let v = vel_slice.read((assembly_id + curr) as usize);
            lin += Vector::ith(i as usize, v);
            curr += 1;
        }
    }

    let ang_locked = (locked_axes >> DIM) & ((1 << ANG_DIM) - 1);
    let num_ang = ANG_DIM - ang_locked.count_ones();
    if num_ang == 1 {
        #[cfg(feature = "dim3")]
        {
            let dof_id = (!ang_locked & 0x7).trailing_zeros();
            let v = vel_slice.read((assembly_id + curr) as usize);
            ang += Vector::ith(dof_id as usize, v);
        }
        #[cfg(feature = "dim2")]
        {
            let v = vel_slice.read((assembly_id + curr) as usize);
            ang += v;
        }
    } else if num_ang == 3 {
        #[cfg(feature = "dim3")]
        {
            let vx = vel_slice.read((assembly_id + curr) as usize);
            let vy = vel_slice.read((assembly_id + curr + 1) as usize);
            let vz = vel_slice.read((assembly_id + curr + 2) as usize);
            ang += AngVector::new(vx, vy, vz);
        }
    }
    (lin, ang)
}
