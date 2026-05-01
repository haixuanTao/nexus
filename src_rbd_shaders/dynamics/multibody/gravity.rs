//! Coriolis-aware generalized force assembly.
//!
//! Mirrors rapier's `update_acceleration` pre-solve logic (equations 42–45):
//! per link we build a kinematic acceleration `acc` recursively from the parent's,
//! then form the "external force" with the inertial / gyroscopic corrections:
//!
//!   acc[i] = acc[parent] + 2·parent_ω × joint_vel.linvel + parent_ω × joint_vel.angvel
//!          + parent_ω × (parent_ω × shift02) + parent_α × shift02
//!   acc[i].linvel += rb.ω × (rb.ω × shift23) + acc[i].angvel × shift23
//!   gyroscopic     = rb.ω × (I · rb.ω)
//!   f_ext_lin  = rb.F_lin - m · acc.linvel        (here rb.F_lin = m·g)
//!   f_ext_ang  = rb.τ     - gyroscopic - I · acc.angvel
//!   τ         += J_iᵀ · (f_ext_lin, f_ext_ang)
//!
//! Finally, `τ -= damping ⊙ velocities`, matching
//!   `self.accelerations.cmpy(-1.0, &self.damping, &self.velocities, 1.0)`.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

use glamx::Vec3;

use crate::dynamics::body::{LocalMassProperties, Velocity};
use crate::utils::linalg::{MatSlice, fill, gemv_tr_spatial};
use crate::utils::{Slice, SliceMut};

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

    let gx = gravity[0];
    let gy = gravity[1];
    let gz = gravity[2];

    for k in 0..num_links {
        let stat = stat_slice.read(k as usize);
        let mut ws = ws_slice.read(k as usize);

        // Build kinematic acceleration `acc` (eqs 42–45).
        let mut acc_lin = Vec3::ZERO;
        let mut acc_ang = Vec3::ZERO;
        if k != 0 {
            let parent_ws = ws_slice.read(stat.parent_link_id as usize);
            let parent_acc = parent_ws.kinematic_acc;
            let parent_ang = parent_ws.rb_vels.angular;

            acc_lin = parent_acc.linear;
            acc_ang = parent_acc.angular;

            // 2 · parent_ω × joint_vel.linvel
            acc_lin += parent_ang.cross(ws.joint_velocity.linear) * 2.0;
            // parent_ω × joint_vel.angvel
            acc_ang += parent_ang.cross(ws.joint_velocity.angular);
            // parent_ω × (parent_ω × shift02)
            acc_lin += parent_ang.cross(parent_ang.cross(ws.shift02));
            // parent_α × shift02
            acc_lin += parent_acc.angular.cross(ws.shift02);
        }
        // Self-shift: rb.ω × (rb.ω × shift23), acc.ω × shift23.
        let rb_ang = ws.rb_vels.angular;
        acc_lin += rb_ang.cross(rb_ang.cross(ws.shift23));
        acc_lin += acc_ang.cross(ws.shift23);

        ws.kinematic_acc = Velocity::new(acc_lin, acc_ang);
        ws_slice.write(k as usize, ws);

        let lmp = local_mprops_slice.read(k as usize);
        let inv_mass_x = lmp.inv_mass.x;
        if inv_mass_x == 0.0 {
            continue;
        }
        let mass = 1.0 / inv_mass_x;
        let rb_inertia = link_world_inertia(&ws, &lmp);

        // rb.forces = (m·g, 0). Build `external_forces` per rapier.
        let gyroscopic = {
            let i_omega = rb_inertia * rb_ang;
            rb_ang.cross(i_omega)
        };
        let i_acc_ang = rb_inertia * acc_ang;
        let f_lin = Vec3::new(mass * gx, mass * gy, mass * gz) - acc_lin * mass;
        let f_ang = -gyroscopic - i_acc_ang;
        let external_forces = [f_lin.x, f_lin.y, f_lin.z, f_ang.x, f_ang.y, f_ang.z];

        let body_jacobian = MatSlice::dense(
            mb_jac_base + (k as usize) * 6 * (ndofs as usize),
            6,
            ndofs,
        );

        gemv_tr_spatial(
            gen_forces,
            gen_base,
            1.0,
            body_jacobians,
            body_jacobian,
            external_forces,
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
