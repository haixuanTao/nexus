//! Solver compute shader kernels
//!
//! This module contains the actual GPU compute shader entry points for the physics solver.

use khal_std::glamx::UVec3;
use khal_std::macros::{spirv, spirv_bindgen};

use crate::{AngVector, Pose, Vector};
use khal_std::{sync::atomic_add_u32, index::MaybeIndexUnchecked, iter::StepRng};

use super::body::{LocalMassProperties, Velocity, WorldMassProperties};
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    constraints: &mut [TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    constraint_builders: &mut [TwoBodyConstraintBuilder],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] body_constraint_counts: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] body_group: &[u32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] collider_world_poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] solver_body_poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 2)] vels: &[Velocity],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 3)] mprops: &[WorldMassProperties],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 4)] all_params: &[SimParams],
    #[spirv(uniform, descriptor_set = 1, binding = 5)] contacts_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 1, binding = 6)] colliders_batch_capacity: &u32,
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
    let body_group = Slice(body_group, colliders_start);
    let collider_world_poses = Slice(collider_world_poses, colliders_start);
    let solver_body_poses = Slice(solver_body_poses, colliders_start);
    let vels = Slice(vels, colliders_start);
    let mprops = Slice(mprops, colliders_start);
    // Iterating to `cap` (instead of `contacts_len[batch]`) lets us drop the
    // `contacts_len` storage binding to fit WebGPU's 10-storage-per-stage
    // limit. Empty / unused contact slots have `contact.len == 0` and are
    // skipped — narrow-phase zero-initialises the buffer so the sentinel is
    // reliable.
    let cap = *contacts_batch_capacity;

    for i in StepRng::new(invocation_id.x..cap, num_threads) {
        let im = contacts.at(i as usize);
        if im.contact.len == 0 {
            continue;
        }
        contact_to_constraint(
            im,
            &mprops,
            &collider_world_poses,
            &solver_body_poses,
            &vels,
            params,
            constraints.at_mut(i as usize),
            constraint_builders.at_mut(i as usize),
        );

        let body1 = im.colliders.x;
        let body2 = im.colliders.y;
        let group1 = body_group.read(body1 as usize);
        let group2 = body_group.read(body2 as usize);

        // Count toward the body's GROUP slot. A body is "active" for the
        // graph-coloring graph if it's a free dynamic body (inv_mass != 0) OR
        // it's part of a multibody (group != self — the multibody handles its
        // own dynamics but its bodies still need correct coloring so contacts
        // touching different links of the same multibody never share a color).
        let is_mb1 = group1 != body1;
        if mprops.at(body1 as usize).inv_mass != Vector::ZERO || is_mb1 {
            atomic_add_u32(body_constraint_counts.at_mut(group1 as usize), 1);
        }
        let is_mb2 = group2 != body2;
        if mprops.at(body2 as usize).inv_mass != Vector::ZERO || is_mb2 {
            atomic_add_u32(body_constraint_counts.at_mut(group2 as usize), 1);
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
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] solver_body_poses: &[Pose],
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
    let solver_body_poses = Slice(solver_body_poses, colliders_start);
    let len = contacts_len.read(batch_id);

    for i in StepRng::new(invocation_id.x..len, num_threads) {
        update_constraint(
            constraints.at_mut(i as usize),
            constraint_builders.at(i as usize),
            &solver_body_poses,
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] body_group: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 6)] contacts_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 7)] colliders_batch_capacity: &u32,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y as usize;
    let contacts_start = batch_id * *contacts_batch_capacity as usize;
    let colliders_start = batch_id * *colliders_batch_capacity as usize;
    let bci_start = batch_id * 2 * *contacts_batch_capacity as usize;

    let contacts = Slice(contacts, contacts_start);
    let mut body_constraint_counts = SliceMut(body_constraint_counts, colliders_start);
    let body_group = Slice(body_group, colliders_start);
    let mprops = Slice(mprops, colliders_start);
    let mut body_constraint_ids = SliceMut(body_constraint_ids, bci_start);
    let len = contacts_len.read(batch_id);

    for i in StepRng::new(invocation_id.x..len, num_threads) {
        let body1 = contacts.at(i as usize).colliders.x as usize;
        let body2 = contacts.at(i as usize).colliders.y as usize;
        let group1 = body_group.read(body1) as usize;
        let group2 = body_group.read(body2) as usize;

        let is_mb1 = group1 != body1;
        if mprops.at(body1).inv_mass != Vector::ZERO || is_mb1 {
            let id1 = atomic_add_u32(body_constraint_counts.at_mut(group1), 1);
            body_constraint_ids.write(id1 as usize, i);
        }

        let is_mb2 = group2 != body2;
        if mprops.at(body2).inv_mass != Vector::ZERO || is_mb2 {
            let id2 = atomic_add_u32(body_constraint_counts.at_mut(group2), 1);
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
pub fn gpu_integrate_linearized(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] poses: &mut [Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] solver_vels: &[Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] num_colliders: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] all_params: &[SimParams],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] colliders_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let colliders_start = batch_id * *colliders_batch_capacity as usize;
    let params = all_params.at(batch_id);
    let i = invocation_id.x;

    let num_colliders = num_colliders.read(batch_id);
    let mut poses = SliceMut(poses, colliders_start);
    let solver_vels = Slice(solver_vels, colliders_start);

    if i < num_colliders {
        let idx = i as usize;
        let vels = solver_vels.at(idx);
        let pose = poses.at_mut(idx);
        vels.integrate_linearized(params.dt, &mut pose.translation, &mut pose.rotation);
    }
}

/// Initializes the solver-bodies' COM-centered poses from the body world poses.
///
/// `solver_body_pose = body_pose.prepend_translation(local_com)`. Mirrors
/// rapier's `SolverBodies::copy_from`: the solver works in a frame whose
/// origin is the body's center of mass and whose rotation is the body's, so
/// every constraint Jacobian "world COM" entry can simply read
/// `solver_body_pose.translation`.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_init_solver_bodies(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] body_poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    local_mprops: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] solver_body_poses: &mut [Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] num_colliders: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] colliders_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let colliders_start = batch_id * *colliders_batch_capacity as usize;
    let i = invocation_id.x;

    let num_colliders = num_colliders.read(batch_id);
    let body_poses = Slice(body_poses, colliders_start);
    let local_mprops = Slice(local_mprops, colliders_start);
    let mut solver_body_poses = SliceMut(solver_body_poses, colliders_start);

    if i < num_colliders {
        let idx = i as usize;
        solver_body_poses.write(
            idx,
            body_poses.read(idx).prepend_translation(local_mprops.at(idx).com),
        );
    }
}

/// Finalizes solver by copying solver velocities back to body velocities and
/// converting the COM-centered solver poses back to body-origin poses.
///
/// `body_pose = solver_body_pose.prepend_translation(-local_com)`. Mirrors
/// rapier's `velocity_solver::writeback_bodies` (which assigns
/// `next_position = solver_pose.prepend_translation(-local_com)`).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_solver_finalize(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] solver_vels: &[Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] body_poses: &mut [Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] solver_body_poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] local_mprops: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] num_colliders: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 6)] colliders_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let colliders_start = batch_id * *colliders_batch_capacity as usize;
    let i = invocation_id.x;

    let num_colliders = num_colliders.read(batch_id);
    let mut vels = SliceMut(vels, colliders_start);
    let solver_vels = Slice(solver_vels, colliders_start);
    let mut body_poses = SliceMut(body_poses, colliders_start);
    let solver_body_poses = Slice(solver_body_poses, colliders_start);
    let local_mprops = Slice(local_mprops, colliders_start);

    if i < num_colliders {
        let idx = i as usize;
        vels.at_mut(idx).linear = solver_vels.at(idx).linear;
        vels.at_mut(idx).angular = solver_vels.at(idx).angular;
        *body_poses.at_mut(idx) = solver_body_poses
            .read(idx)
            .prepend_translation(-local_mprops.at(idx).com);
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
