//! Solver compute shader kernels
//!
//! This module contains the actual GPU compute shader entry points for the physics solver.

use khal_std::glamx::UVec3;
use khal_std::macros::{spirv, spirv_bindgen};

use khal_std::{iter::StepRng, arch::atomic_add_u32, index::MaybeIndexUnchecked};
use crate::{AngVector, Pose, Vector};

use super::body::{integrate_velocity, LocalMassProperties, Velocity, WorldMassProperties};
use super::constraint::{TwoBodyConstraint, TwoBodyConstraintBuilder};
use super::sim_params::SimParams;
use super::solver_utils::{
    contact_to_constraint, remove_cfm_and_bias, solve_constraint_gauss_seidel, update_constraint,
    warmstart_body, warmstart_constraint,
};

use crate::queries::IndexedManifold;
use crate::utils::{Slice, SliceMut};

const WORKGROUP_SIZE: u32 = 64;

/// Resets the current color to 1 (for graph coloring).
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_solver_reset_color(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] curr_color: &mut u32,
) {
    // NOTE: this `for` loop is silly. It doesn't do anything
    //       more than a `*curr_color = 1` in a convoluted
    //       way because otherwise rustgpu apparently does not generate
    //       the spirv for this kernel (seems to happen if the kernel is
    //       too trivial.
    for k in 0..1 {
        // NOTE: our first colors start at 1 instead of 0.
        *curr_color = 1 + k;
    }
}

/// Increments the current color.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_solver_inc_color(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] curr_color: &mut u32,
) {
    // NOTE: this `for` loop is silly. It doesn't do anything
    //       more than a `*curr_color += 1` in a convoluted
    //       way because otherwise rustgpu apparently does not generate
    //       the spirv for this kernel (seems to happen if the kernel is
    //       too trivial.
    for k in 0..1 {
        *curr_color += 1 + k;
    }
}

/// Initializes constraints from contact manifolds.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_solver_init_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] contacts: &[IndexedManifold],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] contacts_len: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    constraints: &mut [TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    constraint_builders: &mut [TwoBodyConstraintBuilder],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] body_constraint_counts: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] vels: &[Velocity],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 2)] mprops: &[WorldMassProperties],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 3)] all_params: &[SimParams],
    #[spirv(uniform, descriptor_set = 1, binding = 4)] contacts_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 1, binding = 5)] colliders_batch_capacity: &u32,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y as usize;
    let contacts_start = batch_id * *contacts_batch_capacity as usize;
    let colliders_start = batch_id * *colliders_batch_capacity as usize;
    let params = all_params.at(batch_id);

    let contacts = Slice(contacts, contacts_start);
    let mut constraints = SliceMut(constraints, contacts_start);
    let mut constraint_builders = SliceMut(constraint_builders, contacts_start);
    let mut body_constraint_counts = SliceMut(body_constraint_counts, colliders_start);
    let poses = Slice(poses, colliders_start);
    let vels = Slice(vels, colliders_start);
    let mprops = Slice(mprops, colliders_start);
    let len = contacts_len.read(batch_id);

    for i in StepRng::new(invocation_id.x..len, num_threads) {
        contact_to_constraint(
            contacts.at(i as usize),
            &mprops,
            &poses,
            &vels,
            params,
            constraints.at_mut(i as usize),
            constraint_builders.at_mut(i as usize),
        );

        let body1 = contacts.at(i as usize).colliders.x;
        let body2 = contacts.at(i as usize).colliders.y;

        // HACK: add a better way of identifying static bodies.
        if mprops.at(body1 as usize).inv_mass != Vector::ZERO {
            atomic_add_u32(body_constraint_counts.at_mut(body1 as usize), 1);
        }

        if mprops.at(body2 as usize).inv_mass != Vector::ZERO {
            atomic_add_u32(body_constraint_counts.at_mut(body2 as usize), 1);
        }
    }
}

/// Updates constraints for a new substep.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_solver_update_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    constraints: &mut [TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    constraint_builders: &[TwoBodyConstraintBuilder],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contacts_len: &[u32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] all_params: &[SimParams],
    #[spirv(uniform, descriptor_set = 1, binding = 2)] contacts_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 1, binding = 3)] colliders_batch_capacity: &u32,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y as usize;
    let contacts_start = batch_id * *contacts_batch_capacity as usize;
    let colliders_start = batch_id * *colliders_batch_capacity as usize;
    let params = all_params.at(batch_id);

    let mut constraints = SliceMut(constraints, contacts_start);
    let constraint_builders = Slice(constraint_builders, contacts_start);
    let poses = Slice(poses, colliders_start);
    let len = contacts_len.read(batch_id);

    for i in StepRng::new(invocation_id.x..len, num_threads) {
        update_constraint(
            constraints.at_mut(i as usize),
            constraint_builders.at(i as usize),
            &poses,
            params,
        );
    }
}

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_solver_sort_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] body_constraint_counts: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mprops: &[WorldMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contacts: &[IndexedManifold],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] contacts_len: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] body_constraint_ids: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] contacts_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] colliders_batch_capacity: &u32,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y as usize;
    let contacts_start = batch_id * *contacts_batch_capacity as usize;
    let colliders_start = batch_id * *colliders_batch_capacity as usize;
    let bci_start = batch_id * 2 * *contacts_batch_capacity as usize;

    let contacts = Slice(contacts, contacts_start);
    let mut body_constraint_counts = SliceMut(body_constraint_counts, colliders_start);
    let mprops = Slice(mprops, colliders_start);
    let mut body_constraint_ids = SliceMut(body_constraint_ids, bci_start);
    let len = contacts_len.read(batch_id);

    for i in StepRng::new(invocation_id.x..len, num_threads) {
        let body1 = contacts.at(i as usize).colliders.x as usize;
        let body2 = contacts.at(i as usize).colliders.y as usize;

        // HACK: add a better way of identifying static bodies.
        if mprops.at(body1).inv_mass != Vector::ZERO {
            let id1 = atomic_add_u32(body_constraint_counts.at_mut(body1), 1);
            body_constraint_ids.write(id1 as usize, i);
        }

        // HACK: add a better way of identifying static bodies.
        if mprops.at(body2).inv_mass != Vector::ZERO {
            let id2 = atomic_add_u32(body_constraint_counts.at_mut(body2), 1);
            body_constraint_ids.write(id2 as usize, i);
        }
    }
}

/// Cleans up solver state and initializes solver velocities.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_solver_cleanup(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] body_constraint_counts: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] solver_vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] vels: &[Velocity],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] mprops: &[WorldMassProperties],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 2)] num_colliders: &[u32],
    #[spirv(uniform, descriptor_set = 1, binding = 3)] colliders_batch_capacity: &u32,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y as usize;
    let colliders_start = batch_id * *colliders_batch_capacity as usize;
    let num_bodies = num_colliders.read(batch_id);

    let mut body_constraint_counts = SliceMut(body_constraint_counts, colliders_start);
    let mut solver_vels = SliceMut(solver_vels, colliders_start);
    let vels = Slice(vels, colliders_start);
    let mprops = Slice(mprops, colliders_start);

    for i in StepRng::new(invocation_id.x..num_bodies, num_threads) {
        let idx = i as usize;
        body_constraint_counts.write(idx, 0);

        // HACK: to handle static bodies.
        if mprops.at(idx).inv_mass != Vector::ZERO {
            solver_vels.at_mut(idx).linear = vels.at(idx).linear;
            solver_vels.at_mut(idx).angular = vels.at(idx).angular;
        } else {
            solver_vels.at_mut(idx).linear = Vector::ZERO;
            solver_vels.at_mut(idx).angular = AngVector::default();
        }
    }
}

/// Initializes solver velocity increments (gravity, external forces).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_init_solver_vels_inc(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] solver_vels_inc: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mprops: &[WorldMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] num_colliders: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] all_params: &[SimParams],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] colliders_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let colliders_start = batch_id * *colliders_batch_capacity as usize;
    let params = all_params.at(batch_id);
    let i = invocation_id.x;

    let num_colliders = num_colliders.read(batch_id);
    let mut solver_vels_inc = SliceMut(solver_vels_inc, colliders_start);
    let mprops = Slice(mprops, colliders_start);

    if i < num_colliders {
        let idx = i as usize;
        solver_vels_inc.at_mut(idx).linear = Vector::ZERO;
        solver_vels_inc.at_mut(idx).angular = AngVector::default();

        // TODO: this isn't a very pretty way of detecting static bodies.
        if mprops.at(idx).inv_mass != Vector::ZERO {
            // TODO: this currently only handles gravity.
            // TODO: make the gravity configurable
            let gravity = Vector::Y * -9.81;
            solver_vels_inc.at_mut(idx).linear = gravity * params.dt;
        }
    }
}

/// Applies solver velocity increments to solver velocities.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_apply_solver_vels_inc(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] solver_vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] solver_vels_inc: &[Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] num_colliders: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] colliders_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let colliders_start = batch_id * *colliders_batch_capacity as usize;
    let i = invocation_id.x;

    let num_colliders = num_colliders.read(batch_id);
    let mut solver_vels = SliceMut(solver_vels, colliders_start);
    let solver_vels_inc = Slice(solver_vels_inc, colliders_start);

    if i < num_colliders {
        let idx = i as usize;
        solver_vels.at_mut(idx).linear += solver_vels_inc.at(idx).linear;
        solver_vels.at_mut(idx).angular += solver_vels_inc.at(idx).angular;
    }
}

/// Applies warmstart impulses without graph coloring (gather-style per body).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_warmstart_without_colors(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] body_constraint_counts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] body_constraint_ids: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] constraints: &[TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] solver_vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] num_colliders: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] colliders_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] contacts_batch_capacity: &u32,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y as usize;
    let colliders_start = batch_id * *colliders_batch_capacity as usize;
    let contacts_start = batch_id * *contacts_batch_capacity as usize;
    let bci_start = batch_id * 2 * *contacts_batch_capacity as usize;
    let num_bodies = num_colliders.read(batch_id);

    let body_constraint_counts = Slice(body_constraint_counts, colliders_start);
    let body_constraint_ids = Slice(body_constraint_ids, bci_start);
    let constraints = Slice(constraints, contacts_start);
    let mut solver_vels = SliceMut(solver_vels, colliders_start);

    for body_id in StepRng::new(invocation_id.x..num_bodies, num_threads) {
        let mut solver_vel = solver_vels.read(body_id as usize);
        warmstart_body(
            body_id,
            &body_constraint_counts,
            &body_constraint_ids,
            &constraints,
            &mut solver_vel,
        );
        solver_vels.write(body_id as usize, solver_vel);
    }
}

/// Applies warmstart impulses with graph coloring (scatter-style per constraint).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_warmstart(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] constraints: &[TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] solver_vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] constraints_colors: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] contacts_len: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] curr_color: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 5)] contacts_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] colliders_batch_capacity: &u32,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y as usize;
    let contacts_start = batch_id * *contacts_batch_capacity as usize;
    let colliders_start = batch_id * *colliders_batch_capacity as usize;

    let constraints = Slice(constraints, contacts_start);
    let constraints_colors = Slice(constraints_colors, contacts_start);
    let mut solver_vels = SliceMut(solver_vels, colliders_start);
    let len = contacts_len.read(batch_id);
    let color = *curr_color;

    for i in StepRng::new(invocation_id.x..len, num_threads) {
        if constraints_colors.read(i as usize) == color {
            let constraint = constraints.at(i as usize);
            let solver_id1 = constraint.solver_body_a as usize;
            let solver_id2 = constraint.solver_body_b as usize;

            let mut solver_vel1 = solver_vels.read(solver_id1);
            let mut solver_vel2 = solver_vels.read(solver_id2);

            warmstart_constraint(constraint, &mut solver_vel1, &mut solver_vel2);

            solver_vels.write(solver_id1, solver_vel1);
            solver_vels.write(solver_id2, solver_vel2);
        }
    }
}

/// Main constraint solver iteration kernel (Projected Gauss-Seidel).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_step_gauss_seidel(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    constraints: &mut [TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] solver_vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] constraints_colors: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] contacts_len: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] curr_color: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 5)] contacts_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] colliders_batch_capacity: &u32,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y as usize;
    let contacts_start = batch_id * *contacts_batch_capacity as usize;
    let colliders_start = batch_id * *colliders_batch_capacity as usize;

    let mut constraints = SliceMut(constraints, contacts_start);
    let constraints_colors = Slice(constraints_colors, contacts_start);
    let mut solver_vels = SliceMut(solver_vels, colliders_start);
    let len = contacts_len.read(batch_id);
    let color = *curr_color;

    for i in StepRng::new(invocation_id.x..len, num_threads) {
        // Only process constraints of the current color (for parallelization)
        if constraints_colors.read(i as usize) == color {
            let solver_id1 = constraints.at(i as usize).solver_body_a as usize;
            let solver_id2 = constraints.at(i as usize).solver_body_b as usize;

            let mut solver_vel1 = solver_vels.read(solver_id1);
            let mut solver_vel2 = solver_vels.read(solver_id2);

            solve_constraint_gauss_seidel(
                constraints.at_mut(i as usize),
                &mut solver_vel1,
                &mut solver_vel2,
            );

            solver_vels.write(solver_id1, solver_vel1);
            solver_vels.write(solver_id2, solver_vel2);
        }
    }
}

/// Integrates velocity to update poses.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_integrate(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] poses: &mut [Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] solver_vels: &[Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    local_mprops: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] num_colliders: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] all_params: &[SimParams],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] colliders_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let colliders_start = batch_id * *colliders_batch_capacity as usize;
    let params = all_params.at(batch_id);
    let i = invocation_id.x;

    let num_colliders = num_colliders.read(batch_id);
    let mut poses = SliceMut(poses, colliders_start);
    let solver_vels = Slice(solver_vels, colliders_start);
    let local_mprops = Slice(local_mprops, colliders_start);

    if i < num_colliders {
        let idx = i as usize;
        let vels = solver_vels.at(idx);
        poses.write(
            idx,
            integrate_velocity(poses.read(idx), vels, local_mprops.at(idx).com, params.dt),
        );
    }
}

/// Finalizes solver by copying solver velocities back to body velocities.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_solver_finalize(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] solver_vels: &[Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] num_colliders: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] colliders_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let colliders_start = batch_id * *colliders_batch_capacity as usize;
    let i = invocation_id.x;

    let num_colliders = num_colliders.read(batch_id);
    let mut vels = SliceMut(vels, colliders_start);
    let solver_vels = Slice(solver_vels, colliders_start);

    if i < num_colliders {
        let idx = i as usize;
        vels.at_mut(idx).linear = solver_vels.at(idx).linear;
        vels.at_mut(idx).angular = solver_vels.at(idx).angular;
    }
}

/// Removes CFM and bias from constraints for the final substep iteration.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_remove_cfm_and_bias_kernel(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    constraints: &mut [TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] contacts_len: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] contacts_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let contacts_start = batch_id * *contacts_batch_capacity as usize;
    let i = invocation_id.x;

    let mut constraints = SliceMut(constraints, contacts_start);
    let len = contacts_len.read(batch_id);

    if i < len {
        remove_cfm_and_bias(constraints.at_mut(i as usize));
    }
}
