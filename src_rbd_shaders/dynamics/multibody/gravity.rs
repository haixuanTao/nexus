//! Coriolis-aware generalized force assembly.
//!
//! Mirrors rapier's `update_acceleration` pre-solve logic (equations 42–45):
//! per link we build a kinematic acceleration `acc` recursively from the parent's,
//! then form the "external force" with the inertial / gyroscopic corrections:
//!
//!   acc[i] = acc[parent] + 2·parent_ω × joint_vel.linvel + parent_ω × joint_vel.angvel  (3D only)
//!          + parent_ω × (parent_ω × shift02) + parent_α × shift02
//!   acc[i].linvel += rb.ω × (rb.ω × shift23) + acc[i].angvel × shift23
//!   gyroscopic     = rb.ω × (I · rb.ω)            (zero in 2D)
//!   f_ext_lin  = rb.F_lin - m · acc.linvel        (here rb.F_lin = m·g)
//!   f_ext_ang  = rb.τ     - gyroscopic - I · acc.angvel
//!   τ         += J_iᵀ · (f_ext_lin, f_ext_ang)
//!
//! Finally, `τ -= damping ⊙ velocities`, matching
//!   `self.accelerations.cmpy(-1.0, &self.damping, &self.velocities, 1.0)`.

use glamx::{Vec2, Vec3};
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

use crate::dynamics::body::{LocalMassProperties, Velocity};
use crate::dynamics::joint::SPATIAL_DIM;
use crate::utils::linalg::{MatSlice, fill, gemv_tr_spatial_split};
use crate::utils::{Slice, SliceMut};
use crate::{AngVector, Vector, gcross_av};

use super::mass_matrix::link_world_inertia;
use super::types::{MultibodyInfo, MultibodyLinkStatic, MultibodyLinkWorkspace};

#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_apply_gravity_with_coriolis(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] links_workspace: &mut [MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] links_local_mprops: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] body_jacobians: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] gen_forces: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] dof_velocities: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] damping: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] num_multibodies: &[u32],
    // TODO: this is only worth keepign as a storage buffer (instead of an uniform)
    //       if we can have different gravity per batch.
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] gravity: &[f32; 3],
    #[spirv(uniform, descriptor_set = 0, binding = 10)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 11)] links_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 12)] jacobians_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 13)] dof_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let links_start = batch_id * *links_batch_capacity as usize;
    let jac_start = batch_id * *jacobians_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let num_links = mb.num_links;
    let ndofs = mb.ndofs;
    let first_link_global = links_start + mb.first_link as usize;
    let mb_jac_base = jac_start + mb.jacobian_offset as usize;
    let gen_base = dof_start + mb.first_dof as usize;

    let stat_slice = Slice(links_static, first_link_global);
    let mut ws_slice = SliceMut(links_workspace, first_link_global);
    let local_mprops_slice = Slice(links_local_mprops, first_link_global);
    let vel_slice = Slice(dof_velocities, gen_base);
    let damping_slice = Slice(damping, gen_base);

    // accelerations.fill(0.0) for this multibody.
    let accelerations = MatSlice::dense(gen_base, ndofs, 1);
    fill(gen_forces, accelerations, 0.0);

    let _ = stat_slice; // not currently used; kept for future kinematic-DOF gating.

    #[cfg(feature = "dim3")]
    let g = Vec3::new(gravity[0], gravity[1], gravity[2]);
    #[cfg(feature = "dim2")]
    let g = Vec2::new(gravity[0], gravity[1]);

    for k in 0..num_links {
        // Reference-only access to `ws`. Gather just the small fields we
        // need; full-struct read materialises a 240 B copy in Function memory.
        let (
            self_joint_vel_lin,
            self_joint_vel_ang,
            self_shift02,
            self_shift23,
            self_local_to_world,
            self_rb_ang,
        ) = {
            let ws = ws_slice.at(k as usize);
            (
                ws.joint_velocity.linear,
                ws.joint_velocity.angular,
                ws.shift02,
                ws.shift23,
                ws.local_to_world,
                ws.rb_vels.angular,
            )
        };

        // Build kinematic acceleration `acc` (eqs 42–45).
        let mut acc_lin = Vector::ZERO;
        #[cfg(feature = "dim3")]
        let mut acc_ang: AngVector = AngVector::ZERO;
        #[cfg(feature = "dim2")]
        let mut acc_ang: AngVector = 0.0;

        if k != 0 {
            let stat = stat_slice.read(k as usize);
            // Parent workspace is read-only here; reference avoids a 240 B copy.
            let parent_ws = ws_slice.at(stat.parent_link_id as usize);
            let parent_acc_lin = parent_ws.kinematic_acc.linear;
            let parent_acc_ang = parent_ws.kinematic_acc.angular;
            let parent_ang = parent_ws.rb_vels.angular;

            acc_lin = parent_acc_lin;
            acc_ang = parent_acc_ang;

            // 2 · parent_ω × joint_vel.linvel
            acc_lin += gcross_av(parent_ang, self_joint_vel_lin) * 2.0;
            // parent_ω × joint_vel.angvel — vanishes in 2D (angular is scalar).
            #[cfg(feature = "dim3")]
            {
                acc_ang += parent_ang.cross(self_joint_vel_ang);
            }
            #[cfg(feature = "dim2")]
            {
                let _ = self_joint_vel_ang;
            }
            // parent_ω × (parent_ω × shift02)
            acc_lin += gcross_av(parent_ang, gcross_av(parent_ang, self_shift02));
            // parent_α × shift02
            acc_lin += gcross_av(parent_acc_ang, self_shift02);
        } else {
            let _ = self_joint_vel_ang;
            let _ = self_shift02;
        }
        // Self-shift: rb.ω × (rb.ω × shift23), acc.ω × shift23.
        let rb_ang = self_rb_ang;
        acc_lin += gcross_av(rb_ang, gcross_av(rb_ang, self_shift23));
        acc_lin += gcross_av(acc_ang, self_shift23);

        // Field-targeted write — only `kinematic_acc` changes.
        ws_slice.at_mut(k as usize).kinematic_acc = Velocity::new(acc_lin, acc_ang);

        let lmp = local_mprops_slice.read(k as usize);
        let inv_mass_x = lmp.inv_mass.x;
        if inv_mass_x == 0.0 {
            continue;
        }
        let mass = 1.0 / inv_mass_x;
        // `link_world_inertia` only reads `local_to_world.rotation`.
        // Reuse the value we already pulled out; the helper takes a
        // workspace ref so synthesise one via `at()`.
        let rb_inertia = link_world_inertia(ws_slice.at(k as usize), &lmp);
        let _ = self_local_to_world;

        // Gyroscopic torque: `rb.ω × (I · rb.ω)` in 3D, 0 in 2D.
        #[cfg(feature = "dim3")]
        let gyroscopic = {
            let i_omega = rb_inertia * rb_ang;
            rb_ang.cross(i_omega)
        };
        #[cfg(feature = "dim2")]
        let gyroscopic: AngVector = 0.0;

        // I · acc.angvel — Mat3·Vec3 in 3D, scalar·scalar in 2D.
        let i_acc_ang = rb_inertia * acc_ang;

        // f_ext_lin = m·g - m·acc_lin.
        #[cfg(feature = "dim3")]
        let f_lin = (g - acc_lin) * mass;
        #[cfg(feature = "dim2")]
        let f_lin = (g - acc_lin) * mass;
        let f_ang = -gyroscopic - i_acc_ang;

        let body_jacobian = MatSlice::dense(
            mb_jac_base + (k as usize) * SPATIAL_DIM * (ndofs as usize),
            SPATIAL_DIM as u32,
            ndofs,
        );

        // Split form: takes Vector + AngVector, no `[f32; SPATIAL_DIM]`
        // Function-storage scratch.
        gemv_tr_spatial_split(
            gen_forces,
            gen_base,
            1.0,
            body_jacobians,
            body_jacobian,
            f_lin,
            f_ang,
            1.0,
        );
    }

    // `accelerations.cmpy(-1.0, &damping, &velocities, 1.0)`.
    for i in 0..ndofs {
        let idx = gen_base + i as usize;
        let cur = gen_forces.read(idx);
        gen_forces.write(
            idx,
            cur - damping_slice.read(i as usize) * vel_slice.read(i as usize),
        );
    }
}
