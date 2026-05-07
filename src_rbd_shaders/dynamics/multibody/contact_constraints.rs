//! Multibody contact constraints.
//!
//! Mirrors rapier's `RigidBodyMultibodyContactConstraint` flow for contacts
//! where one or both sides are a multibody link. Each contact point produces
//! one normal (non-penetration) slot plus `DIM-1` Coulomb-friction tangent
//! slots in the per-multibody constraint slab; the tangent slots clamp their
//! impulse to `±μ · normal_impulse` at solve time (independent per-tangent
//! clamp — i.e. box friction; rapier's circular-cone joint clamp via
//! `cap_magnitude` is a future refinement).
//!
//! Pipeline, called once per substep from `apply_substep`:
//!
//!   1. `gpu_mb_init_contact_constraints` — scan the contacts buffer; for each
//!      contact point touching a link of this multibody, emit a normal-direction
//!      constraint and `DIM-1` tangent constraints (using
//!      `OrthonormalBasis::orthonormal_vector(normal)` for the first tangent
//!      and `normal × tangent0` for the second), and write each constraint's
//!      multibody-side `Jᵀ` row into `contact_constraint_jacs`.
//!   2. `gpu_mb_finalize_contact_constraints` — for each emitted constraint,
//!      LU back-solve `M · column = Jᵀ` (writing the column into
//!      `contact_constraint_columns`) and set `inv_lhs = 1 / (Jᵀ·column +
//!      free_body_inv_r)`.
//!   3. `gpu_mb_solve_contact_constraints` — one PGS sweep updating both
//!      `dof_velocities` (multibody side) and `solver_vels` (free body side).
//!   4. `gpu_mb_remove_contact_constraint_bias` — strip the positional bias
//!      from `rhs` (mirrors `gpu_mb_remove_joint_constraint_bias`) for the
//!      stabilization sweep.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

use crate::dynamics::body::{Velocity, WorldMassProperties};
use crate::dynamics::joint::SPATIAL_DIM;
use crate::queries::IndexedManifold;
use crate::utils::Slice;
use crate::utils::linalg::{MatSlice, lu_solve_in_place};
use crate::{ANG_DIM, AngVector, DIM, Pose, Vector, gcross, gdot};

use super::types::{
    CONTACT_CONSTRAINTS_PER_POINT, MAX_MB_CONTACT_CONSTRAINTS_PER_MB, MAX_MB_CONTACTS_PER_MB,
    MB_CONTACT_KIND_INACTIVE, MB_CONTACT_KIND_NORMAL, MB_CONTACT_KIND_TANGENT,
    MultibodyContactConstraint, MultibodyInfo, MultibodyLinkStatic, MultibodyLinkWorkspace,
};

#[cfg(feature = "dim3")]
use glamx::Vec3;
#[cfg(feature = "dim2")]
use glamx::Vec2;

/// Default Coulomb friction coefficient — matches the rb-rb default in
/// `solver_utils::contact_to_constraint`. Per-collider material props are a
/// TODO for both code paths.
const FRICTION_DEFAULT: f32 = 0.5;

/// Compute an arbitrary unit vector orthogonal to `v` (assumed unit length).
/// Mirrors rapier's `OrthonormalBasis::orthonormal_vector` fallback used when
/// the relative tangent velocity is too small to drive friction direction
/// selection.
#[cfg(feature = "dim3")]
#[inline]
fn orthonormal_vector(v: Vec3) -> Vec3 {
    let sign = if v.z < 0.0 { -1.0 } else { 1.0 };
    let a = -1.0 / (sign + v.z);
    let b = v.x * v.y * a;
    Vec3::new(b, sign + v.y * v.y * a, -v.y)
}

#[cfg(feature = "dim2")]
#[inline]
fn orthonormal_vector(v: Vec2) -> Vec2 {
    Vec2::new(-v.y, v.x)
}

/// Read the `link_id`-th column block of the multibody's body jacobian and
/// project it through the per-side `(unit_force, unit_torque)` pair,
/// **adding** the resulting `Jᵀ` row to `out_jacs[col_offset ..]` (so two
/// calls accumulate — used by self-collisions, which combine the two
/// touched links into a single net `Jᵀ` row).
///
/// Mirrors rapier's `Multibody::fill_jacobians` (the scalar inner kernel),
/// returning `j · invm_j` is deferred to the finalize kernel; here we just
/// pack the row.
#[inline]
fn fill_contact_jac_row(
    body_jacobians: &[f32],
    mb_jac_base: usize,
    ndofs: u32,
    link_id: u32,
    unit_force: Vector,
    unit_torque: AngVector,
    out_jacs: &mut [f32],
    col_offset: usize,
    accumulate: bool,
) {
    // Per-link SPATIAL_DIM × ndofs jacobian (rows 0..DIM = J_v, rows
    // DIM..SPATIAL_DIM = J_w).
    let link_jac_base = mb_jac_base + (link_id as usize) * SPATIAL_DIM * (ndofs as usize);
    let link_j = MatSlice::dense(link_jac_base, SPATIAL_DIM as u32, ndofs);
    let (link_j_v, link_j_w) = link_j.rows_range_pair(0, DIM, DIM, ANG_DIM);
    for j in 0..ndofs {
        // Linear contribution: `unit_force · J_v[:, j]`.
        let dot;
        #[cfg(feature = "dim3")]
        {
            let jv0 = body_jacobians.read(link_j_v.idx(0, j));
            let jv1 = body_jacobians.read(link_j_v.idx(1, j));
            let jv2 = body_jacobians.read(link_j_v.idx(2, j));
            let jw0 = body_jacobians.read(link_j_w.idx(0, j));
            let jw1 = body_jacobians.read(link_j_w.idx(1, j));
            let jw2 = body_jacobians.read(link_j_w.idx(2, j));
            dot = unit_force.x * jv0
                + unit_force.y * jv1
                + unit_force.z * jv2
                + unit_torque.x * jw0
                + unit_torque.y * jw1
                + unit_torque.z * jw2;
        }
        #[cfg(feature = "dim2")]
        {
            let jv0 = body_jacobians.read(link_j_v.idx(0, j));
            let jv1 = body_jacobians.read(link_j_v.idx(1, j));
            let jw0 = body_jacobians.read(link_j_w.idx(0, j));
            dot = unit_force.x * jv0 + unit_force.y * jv1 + unit_torque * jw0;
        }
        let prev = if accumulate {
            out_jacs.read(col_offset + j as usize)
        } else {
            0.0
        };
        out_jacs.write(col_offset + j as usize, prev + dot);
    }
}

/// Pack the per-link world-space contact point into the constraint.
///
/// Pass 1: scans every contact in `contacts[batch]` and, for each contact
/// point touching a link of this multibody, emits a normal-direction
/// `MultibodyContactConstraint` and writes the multibody-side `Jᵀ` row
/// (`mb_normal · J_v + (mb_shift × mb_normal) · J_w`) into
/// `contact_constraint_jacs`. Friction tangents and multibody-multibody
/// contacts are not yet handled — such contacts are skipped.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_init_contact_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] links_workspace: &[MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] body_jacobians: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] body_to_link: &[[u32; 2]],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] contact_constraint_jacs: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] contact_constraint_count: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] num_multibodies: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] dt_buf: &[f32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] mprops: &[WorldMassProperties],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 2)] contacts: &[IndexedManifold],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 3)] contacts_len: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 10)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 11)] links_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 12)] jacobians_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 13)] dof_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 14)] contact_constraints_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 15)] contact_constraint_columns_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 1, binding = 4)] contacts_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 1, binding = 5)] colliders_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 1, binding = 6)] body_to_link_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }
    let dt = dt_buf.read(0);
    let inv_dt = if dt != 0.0 { 1.0 / dt } else { 0.0 };
    let erp_inv_dt = inv_dt;
    let allowed_lin_err = 0.001f32;
    let max_corr_velocity = 10.0f32;

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let links_start = batch_id * *links_batch_capacity as usize;
    let jac_start = batch_id * *jacobians_batch_capacity as usize;
    let cons_start = batch_id * *contact_constraints_batch_capacity as usize;
    let col_start = batch_id * *contact_constraint_columns_batch_capacity as usize;
    let contacts_start = batch_id * *contacts_batch_capacity as usize;
    let colliders_start = batch_id * *colliders_batch_capacity as usize;
    let b2l_start = batch_id * *body_to_link_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let ndofs = mb.ndofs;
    if ndofs == 0 {
        contact_constraint_count.write(mb_start + mb_idx as usize, 0);
        return;
    }
    let mb_jac_base = jac_start + mb.jacobian_offset as usize;
    let cons_base = cons_start + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize);
    // Each constraint slot reserves `dof_batch_capacity` floats in the
    // column buffer (matches the allocation in `from_rapier` and avoids any
    // overlap between multibodies of differing `ndofs`).
    let dofs_stride = *dof_batch_capacity as usize;
    let col_base =
        col_start + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize) * dofs_stride;

    let ws_slice = Slice(links_workspace, links_start + mb.first_link as usize);
    let _ = links_static;

    let n_contacts = contacts_len.read(batch_id);
    let mut count = 0u32;

    for ci in 0..n_contacts {
        if count >= MAX_MB_CONTACTS_PER_MB {
            break;
        }
        let im = contacts.read(contacts_start + ci as usize);
        let id1 = im.colliders.x;
        let id2 = im.colliders.y;

        let l1 = body_to_link.read(b2l_start + id1 as usize);
        let l2 = body_to_link.read(b2l_start + id2 as usize);
        let mb_on_1 = l1[0] == mb_idx;
        let mb_on_2 = l2[0] == mb_idx;

        if !mb_on_1 && !mb_on_2 {
            continue;
        }
        // Inter-multibody contacts (each side is a DIFFERENT multibody) are
        // not yet handled — skip them. Self-collisions (both sides on this
        // SAME multibody) are handled below.
        if l1[0] != u32::MAX && l2[0] != u32::MAX && l1[0] != l2[0] {
            continue;
        }

        let is_self = mb_on_1 && mb_on_2;
        let (mb_link_id_a, mb_link_id_b, free_body_id) = if is_self {
            (l1[1], l2[1], u32::MAX)
        } else if mb_on_1 {
            (l1[1], u32::MAX, id2)
        } else {
            (l2[1], u32::MAX, id1)
        };

        // Skip degenerate self-contacts on the same link.
        if is_self && mb_link_id_a == mb_link_id_b {
            continue;
        }

        let pose1 = poses.read(colliders_start + id1 as usize);
        let world_normal = pose1.rotation * im.contact.normal_a;
        let lin_jac = if is_self || mb_on_1 { world_normal } else { -world_normal };
        let mb_normal = -lin_jac;

        let free_mp = if is_self {
            WorldMassProperties::default()
        } else {
            mprops.read(colliders_start + free_body_id as usize)
        };
        let free_im = if is_self { 0.0 } else { free_mp.inv_mass.x };

        let link_origin_a = ws_slice.at(mb_link_id_a as usize).local_to_world.translation;
        let link_origin_b_default = link_origin_a;

        for k in 0..im.contact.len {
            // One contact point produces 1 normal + (DIM-1) friction slots.
            if count + CONTACT_CONSTRAINTS_PER_POINT > MAX_MB_CONTACT_CONSTRAINTS_PER_MB {
                break;
            }
            let pt_local = im.contact.points_a.read(k as usize).pt;
            let dist = im.contact.points_a.read(k as usize).dist;
            let pt_world = pose1 * (pt_local + im.contact.normal_a * (dist * 0.5));

            // Tangent basis — matches rapier's fallback path
            // (`OrthonormalBasis::orthonormal_vector(force_dir1)` then
            // `dir1.cross(tangent1)`). Velocity-driven tangent selection
            // (rapier's preferred path when `|tangent_relvel|` is large) is
            // skipped for now — the fallback is correct, just less optimal.
            let mb_tangent0 = orthonormal_vector(mb_normal);
            #[cfg(feature = "dim3")]
            let mb_tangent1 = mb_normal.cross(mb_tangent0);

            // A-side (link `mb_link_id_a`, rapier's body 1): impulse along
            // `force_dir1 = -world_normal_a = mb_normal`.
            let shift_a = pt_world - link_origin_a;
            let torque_a_normal = gcross(shift_a, mb_normal);
            let torque_a_t0 = gcross(shift_a, mb_tangent0);
            #[cfg(feature = "dim3")]
            let torque_a_t1 = gcross(shift_a, mb_tangent1);

            let rhs_bias =
                (erp_inv_dt * (dist + allowed_lin_err)).clamp(-max_corr_velocity, 0.0);
            let rhs_wo_bias = if dist > 0.0 { dist * inv_dt } else { 0.0 };

            let normal_slot = count;
            let normal_col_offset = col_base + (normal_slot as usize) * dofs_stride;

            fill_contact_jac_row(
                body_jacobians,
                mb_jac_base,
                ndofs,
                mb_link_id_a,
                mb_normal,
                torque_a_normal,
                contact_constraint_jacs,
                normal_col_offset,
                false,
            );

            // B-side fold-in for self-contacts, free body for the rest. The
            // ang_jac fields below describe the FREE body side; for self
            // contacts they collapse to zero because both sides are folded
            // into `J_mb` already.
            let (ang_jac_normal, ii_ang_jac_normal) = if is_self {
                let link_origin_b = ws_slice.at(mb_link_id_b as usize).local_to_world.translation;
                let _ = link_origin_b_default;
                let shift_b = pt_world - link_origin_b;
                let torque_b_normal = gcross(shift_b, lin_jac);
                fill_contact_jac_row(
                    body_jacobians,
                    mb_jac_base,
                    ndofs,
                    mb_link_id_b,
                    lin_jac,
                    torque_b_normal,
                    contact_constraint_jacs,
                    normal_col_offset,
                    true,
                );
                #[cfg(feature = "dim3")]
                {
                    (AngVector::ZERO, AngVector::ZERO)
                }
                #[cfg(feature = "dim2")]
                {
                    (0.0f32, 0.0f32)
                }
            } else {
                let _ = link_origin_b_default;
                let free_shift = pt_world - free_mp.com;
                let aj = gcross(free_shift, lin_jac);
                let iiaj = free_mp.inv_inertia_mul(aj);
                (aj, iiaj)
            };

            // Normal constraint slot.
            #[cfg(feature = "dim3")]
            let normal_cons = MultibodyContactConstraint {
                multibody_id: mb_idx,
                link_id: mb_link_id_a,
                kind: MB_CONTACT_KIND_NORMAL,
                free_body_id,
                free_body_im: free_im,
                friction_coeff: FRICTION_DEFAULT,
                normal_constraint_slot: normal_slot,
                _pad0: 0,
                lin_jac,
                _pad1: 0,
                ang_jac: ang_jac_normal,
                _pad2: 0,
                ii_ang_jac: ii_ang_jac_normal,
                _pad3: 0,
                inv_lhs: 0.0,
                rhs: rhs_wo_bias + rhs_bias,
                rhs_wo_bias,
                impulse: 0.0,
                cfm_coeff: 0.0,
                cfm_gain: 0.0,
                _pad4: [0; 2],
            };
            #[cfg(feature = "dim2")]
            let normal_cons = MultibodyContactConstraint {
                multibody_id: mb_idx,
                link_id: mb_link_id_a,
                kind: MB_CONTACT_KIND_NORMAL,
                free_body_id,
                free_body_im: free_im,
                ang_jac: ang_jac_normal,
                ii_ang_jac: ii_ang_jac_normal,
                friction_coeff: FRICTION_DEFAULT,
                normal_constraint_slot: normal_slot,
                _pad0: [0; 1],
                lin_jac,
                inv_lhs: 0.0,
                rhs: rhs_wo_bias + rhs_bias,
                rhs_wo_bias,
                impulse: 0.0,
                cfm_coeff: 0.0,
                cfm_gain: 0.0,
            };
            contact_constraints.write(cons_base + normal_slot as usize, normal_cons);
            count += 1;

            // Friction tangent constraints — same contact point, tangent
            // direction. The MB-side `Jᵀ` row is written into the next slab
            // column; the free-side jacobians are stored on the constraint.
            // Limit `±μ · normal_impulse` is computed at solve time by
            // looking up `cons[normal_constraint_slot].impulse`.
            for tang_idx in 0..(CONTACT_CONSTRAINTS_PER_POINT - 1) {
                let mb_tangent = if tang_idx == 0 { mb_tangent0 } else {
                    #[cfg(feature = "dim3")]
                    {
                        mb_tangent1
                    }
                    #[cfg(feature = "dim2")]
                    {
                        // Unreachable in 2D (loop count = 0).
                        mb_tangent0
                    }
                };
                let torque_a_tang = if tang_idx == 0 { torque_a_t0 } else {
                    #[cfg(feature = "dim3")]
                    {
                        torque_a_t1
                    }
                    #[cfg(feature = "dim2")]
                    {
                        torque_a_t0
                    }
                };
                let free_tangent = -mb_tangent;
                let tang_slot = count;
                let tang_col_offset = col_base + (tang_slot as usize) * dofs_stride;

                fill_contact_jac_row(
                    body_jacobians,
                    mb_jac_base,
                    ndofs,
                    mb_link_id_a,
                    mb_tangent,
                    torque_a_tang,
                    contact_constraint_jacs,
                    tang_col_offset,
                    false,
                );

                let (ang_jac_tang, ii_ang_jac_tang) = if is_self {
                    let link_origin_b = ws_slice.at(mb_link_id_b as usize).local_to_world.translation;
                    let shift_b = pt_world - link_origin_b;
                    let torque_b_tang = gcross(shift_b, free_tangent);
                    fill_contact_jac_row(
                        body_jacobians,
                        mb_jac_base,
                        ndofs,
                        mb_link_id_b,
                        free_tangent,
                        torque_b_tang,
                        contact_constraint_jacs,
                        tang_col_offset,
                        true,
                    );
                    #[cfg(feature = "dim3")]
                    {
                        (AngVector::ZERO, AngVector::ZERO)
                    }
                    #[cfg(feature = "dim2")]
                    {
                        (0.0f32, 0.0f32)
                    }
                } else {
                    let free_shift = pt_world - free_mp.com;
                    let aj = gcross(free_shift, free_tangent);
                    let iiaj = free_mp.inv_inertia_mul(aj);
                    (aj, iiaj)
                };

                // No surface velocity (TODO: conveyor belts) → rhs = 0.
                #[cfg(feature = "dim3")]
                let tang_cons = MultibodyContactConstraint {
                    multibody_id: mb_idx,
                    link_id: mb_link_id_a,
                    kind: MB_CONTACT_KIND_TANGENT,
                    free_body_id,
                    free_body_im: free_im,
                    friction_coeff: FRICTION_DEFAULT,
                    normal_constraint_slot: normal_slot,
                    _pad0: 0,
                    lin_jac: free_tangent,
                    _pad1: 0,
                    ang_jac: ang_jac_tang,
                    _pad2: 0,
                    ii_ang_jac: ii_ang_jac_tang,
                    _pad3: 0,
                    inv_lhs: 0.0,
                    rhs: 0.0,
                    rhs_wo_bias: 0.0,
                    impulse: 0.0,
                    cfm_coeff: 0.0,
                    cfm_gain: 0.0,
                    _pad4: [0; 2],
                };
                #[cfg(feature = "dim2")]
                let tang_cons = MultibodyContactConstraint {
                    multibody_id: mb_idx,
                    link_id: mb_link_id_a,
                    kind: MB_CONTACT_KIND_TANGENT,
                    free_body_id,
                    free_body_im: free_im,
                    ang_jac: ang_jac_tang,
                    ii_ang_jac: ii_ang_jac_tang,
                    friction_coeff: FRICTION_DEFAULT,
                    normal_constraint_slot: normal_slot,
                    _pad0: [0; 1],
                    lin_jac: free_tangent,
                    inv_lhs: 0.0,
                    rhs: 0.0,
                    rhs_wo_bias: 0.0,
                    impulse: 0.0,
                    cfm_coeff: 0.0,
                    cfm_gain: 0.0,
                };
                contact_constraints.write(cons_base + tang_slot as usize, tang_cons);
                count += 1;
            }
        }
    }

    // Mark surplus slots as inactive so the solve sweep skips them.
    for s in count..MAX_MB_CONTACT_CONSTRAINTS_PER_MB {
        let mut cz = contact_constraints.read(cons_base + s as usize);
        cz.kind = MB_CONTACT_KIND_INACTIVE;
        cz.impulse = 0.0;
        contact_constraints.write(cons_base + s as usize, cz);
    }
    contact_constraint_count.write(mb_start + mb_idx as usize, count);
}

/// Pass 2: for each emitted constraint, LU back-solve `M · column = Jᵀ`
/// (the row produced by the init kernel) and set `inv_lhs = 1 / (Jᵀ ·
/// column + free_body_inv_r)`.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_finalize_contact_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mass_matrices: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] lu_pivots: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] contact_constraint_jacs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] contact_constraint_columns: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] contact_constraint_count: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 8)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 9)] mass_matrix_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 10)] dof_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 11)] contact_constraints_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 12)] contact_constraint_columns_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let mm_start = batch_id * *mass_matrix_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;
    let cons_start = batch_id * *contact_constraints_batch_capacity as usize;
    let col_start = batch_id * *contact_constraint_columns_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let ndofs = mb.ndofs;
    if ndofs == 0 {
        return;
    }
    let mb_mm_base = mm_start + mb.mass_matrix_offset as usize;
    let piv_offset = dof_start + mb.first_dof as usize;
    let cons_base = cons_start + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize);
    let dofs_stride = *dof_batch_capacity as usize;
    let col_base =
        col_start + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize) * dofs_stride;

    let m = MatSlice::dense(mb_mm_base, ndofs, ndofs);
    let count = contact_constraint_count.read(mb_start + mb_idx as usize);

    for s in 0..count {
        let col_offset = col_base + (s as usize) * dofs_stride;
        // 1) Copy J^T row into the column buffer (it'll be overwritten by the
        //    LU solve with the M⁻¹·Jᵀ result).
        for i in 0..ndofs {
            let v = contact_constraint_jacs.read(col_offset + i as usize);
            contact_constraint_columns.write(col_offset + i as usize, v);
        }
        // 2) Solve M · column = J^T  (in place).
        lu_solve_in_place(
            mass_matrices,
            m,
            lu_pivots,
            piv_offset,
            contact_constraint_columns,
            col_offset,
        );
        // 3) inv_r_mb = J · column.
        let mut inv_r_mb = 0.0f32;
        for i in 0..ndofs {
            let j = contact_constraint_jacs.read(col_offset + i as usize);
            let c = contact_constraint_columns.read(col_offset + i as usize);
            inv_r_mb += j * c;
        }
        // 4) Add free body's contribution: im (since lin_jac is unit) +
        //    ang_jac · ii_ang_jac. For self-contacts the B-side is folded into
        //    `J_mb`, so there's no free-body term.
        let mut cons = contact_constraints.read(cons_base + s as usize);
        let is_self = cons.free_body_id == u32::MAX;
        let inv_r_free = if is_self {
            0.0
        } else {
            cons.free_body_im + gdot(cons.ang_jac, cons.ii_ang_jac)
        };
        let total = inv_r_mb + inv_r_free;
        cons.inv_lhs = if total > 0.0 { 1.0 / total } else { 0.0 };
        contact_constraints.write(cons_base + s as usize, cons);
    }
}

/// One PGS sweep over the multibody's active contact constraints. Updates
/// the multibody's `dof_velocities` and the free body's `solver_vels`.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_solve_contact_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contact_constraint_jacs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] contact_constraint_columns: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] contact_constraint_count: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] dof_velocities: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] solver_vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 8)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 9)] dof_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 10)] contact_constraints_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 11)] contact_constraint_columns_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 12)] colliders_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;
    let cons_start = batch_id * *contact_constraints_batch_capacity as usize;
    let col_start = batch_id * *contact_constraint_columns_batch_capacity as usize;
    let colliders_start = batch_id * *colliders_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let ndofs = mb.ndofs;
    if ndofs == 0 {
        return;
    }
    let v_base = dof_start + mb.first_dof as usize;
    let cons_base = cons_start + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize);
    let dofs_stride = *dof_batch_capacity as usize;
    let col_base =
        col_start + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize) * dofs_stride;

    let count = contact_constraint_count.read(mb_start + mb_idx as usize);
    for s in 0..count {
        let mut cons = contact_constraints.read(cons_base + s as usize);
        if cons.kind == MB_CONTACT_KIND_INACTIVE {
            continue;
        }
        let col_offset = col_base + (s as usize) * dofs_stride;

        // J · u = J_mb · v_mb_dofs + J_free · v_free.
        let is_self = cons.free_body_id == u32::MAX;
        let mut j_dot_v = 0.0f32;
        for i in 0..ndofs {
            let j = contact_constraint_jacs.read(col_offset + i as usize);
            let v = dof_velocities.read(v_base + i as usize);
            j_dot_v += j * v;
        }
        let free = if is_self {
            Velocity::default()
        } else {
            solver_vels.read(colliders_start + cons.free_body_id as usize)
        };
        if !is_self {
            j_dot_v += cons.lin_jac.dot(free.linear) + gdot(cons.ang_jac, free.angular);
        }

        let rhs_total = j_dot_v + cons.rhs;
        let raw_imp = cons.impulse
            - cons.inv_lhs * (rhs_total + cons.cfm_gain * cons.impulse);

        // Normal: clamp to ≥ 0 (no separation impulse). Friction tangent:
        // clamp to `±μ · normal_impulse` — looks up the paired normal slot
        // for the current accumulated impulse. Mirrors rapier's
        // `ContactConstraintNormalPart::generic_solve` /
        // `ContactConstraintTangentPart::generic_solve` (independent
        // per-tangent clamp, i.e. box friction; rapier's circular-cone
        // joint clamp is a future refinement).
        let new_imp = if cons.kind == MB_CONTACT_KIND_TANGENT {
            let normal = contact_constraints
                .read(cons_base + cons.normal_constraint_slot as usize);
            let limit = cons.friction_coeff * normal.impulse;
            if raw_imp > limit {
                limit
            } else if raw_imp < -limit {
                -limit
            } else {
                raw_imp
            }
        } else if raw_imp < 0.0 {
            0.0
        } else {
            raw_imp
        };
        let delta = new_imp - cons.impulse;
        cons.impulse = new_imp;
        contact_constraints.write(cons_base + s as usize, cons);

        if delta != 0.0 {
            for i in 0..ndofs {
                let v_idx = v_base + i as usize;
                let cur = dof_velocities.read(v_idx);
                let col = contact_constraint_columns.read(col_offset + i as usize);
                dof_velocities.write(v_idx, cur + delta * col);
            }
            if !is_self {
                let mut new_free = free;
                new_free.linear =
                    new_free.linear + cons.lin_jac * (cons.free_body_im * delta);
                new_free.angular = new_free.angular + cons.ii_ang_jac * delta;
                solver_vels.write(colliders_start + cons.free_body_id as usize, new_free);
            }
        }
    }
}

/// Strip the positional bias from each active contact constraint's `rhs`,
/// matching `gpu_mb_remove_joint_constraint_bias`.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_remove_contact_constraint_bias(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] contact_constraint_count: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 4)] contact_constraints_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let cons_start = batch_id * *contact_constraints_batch_capacity as usize;
    let cons_base = cons_start + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize);
    let count = contact_constraint_count.read(mb_start + mb_idx as usize);

    for s in 0..count {
        let mut cons = contact_constraints.read(cons_base + s as usize);
        if cons.kind == 0 {
            continue;
        }
        cons.rhs = cons.rhs_wo_bias;
        contact_constraints.write(cons_base + s as usize, cons);
    }
}
