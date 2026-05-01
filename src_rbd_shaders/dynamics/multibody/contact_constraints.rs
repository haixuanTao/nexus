//! Multibody contact constraints.
//!
//! Mirrors rapier's `RigidBodyMultibodyContactConstraint` flow but currently
//! limited to the **normal** component (no friction) of contacts where exactly
//! one side is a multibody (the other is a free rigid body).
//!
//! Pipeline, called once per substep from `apply_substep`:
//!
//!   1. `gpu_mb_init_contact_constraints` — scan the contacts buffer; for each
//!      contact point touching a link of this multibody, emit a normal-direction
//!      constraint and write the multibody-side `Jᵀ` row into
//!      `contact_constraint_jacs`.
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

use glamx::Vec3;

use crate::Pose;
use crate::dynamics::body::{Velocity, WorldMassProperties};
use crate::queries::IndexedManifold;
use crate::utils::Slice;
use crate::utils::linalg::{MatSlice, lu_solve_in_place};

use super::types::{
    MAX_MB_CONTACTS_PER_MB, MultibodyContactConstraint, MultibodyInfo, MultibodyLinkStatic,
    MultibodyLinkWorkspace,
};

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
    unit_force: Vec3,
    unit_torque: Vec3,
    out_jacs: &mut [f32],
    col_offset: usize,
    accumulate: bool,
) {
    // Per-link 6×ndofs jacobian (rows 0-2 = J_v, rows 3-5 = J_w).
    let link_jac_base = mb_jac_base + (link_id as usize) * 6 * (ndofs as usize);
    let link_j = MatSlice::dense(link_jac_base, 6, ndofs);
    let (link_j_v, link_j_w) = link_j.rows_range_pair(0, 3, 3, 3);
    for j in 0..ndofs {
        let jv0 = body_jacobians.read(link_j_v.idx(0, j));
        let jv1 = body_jacobians.read(link_j_v.idx(1, j));
        let jv2 = body_jacobians.read(link_j_v.idx(2, j));
        let jw0 = body_jacobians.read(link_j_w.idx(0, j));
        let jw1 = body_jacobians.read(link_j_w.idx(1, j));
        let jw2 = body_jacobians.read(link_j_w.idx(2, j));
        let dot = unit_force.x * jv0
            + unit_force.y * jv1
            + unit_force.z * jv2
            + unit_torque.x * jw0
            + unit_torque.y * jw1
            + unit_torque.z * jw2;
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
    let cons_base = cons_start + (mb_idx as usize) * (MAX_MB_CONTACTS_PER_MB as usize);
    // Each constraint slot reserves `dof_batch_capacity` floats in the
    // column buffer (matches the allocation in `from_rapier` and avoids any
    // overlap between multibodies of differing `ndofs`).
    let dofs_stride = *dof_batch_capacity as usize;
    let col_base =
        col_start + (mb_idx as usize) * (MAX_MB_CONTACTS_PER_MB as usize) * dofs_stride;

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

        // Skip degenerate self-contacts on the same link (geometry shouldn't
        // produce these, but be defensive).
        if is_self && mb_link_id_a == mb_link_id_b {
            continue;
        }

        let pose1 = poses.read(colliders_start + id1 as usize);
        let world_normal = pose1.rotation * im.contact.normal_a;
        // Convention: `lin_jac` = impulse direction on the "B-side" body.
        //   - Free contact: B-side = free body. lin_jac = +world_normal_a if
        //     mb=1, -world_normal_a if mb=2.
        //   - Self-contact: B-side = link `mb_link_id_b` (= rapier's body 2).
        //     lin_jac = +world_normal_a (impulse on body 2 = -force_dir1).
        let lin_jac = if is_self || mb_on_1 { world_normal } else { -world_normal };
        let mb_normal = -lin_jac;

        // Free-body mass-properties (only valid for the free-contact path).
        let free_mp = if is_self {
            WorldMassProperties::default()
        } else {
            mprops.read(colliders_start + free_body_id as usize)
        };
        let free_im = if is_self { 0.0 } else { free_mp.inv_mass.x };

        let link_ws_a = ws_slice.read(mb_link_id_a as usize);
        let link_origin_a = link_ws_a.local_to_world.translation;
        // For self-contacts the second link's origin is read inside the
        // contact-point loop; for free contacts this is unused.
        let link_origin_b_default = link_origin_a;

        for k in 0..im.contact.len {
            if count >= MAX_MB_CONTACTS_PER_MB {
                break;
            }
            let pt_local = im.contact.points_a.read(k as usize).pt;
            let dist = im.contact.points_a.read(k as usize).dist;
            // World contact point — mid-point between the two surfaces, matching
            // rapier's `pose1 * (pt_a + normal_a * dist / 2)`.
            let pt_world = pose1 * (pt_local + im.contact.normal_a * (dist * 0.5));

            // A-side (link `mb_link_id_a`, rapier's body 1): impulse along
            // `force_dir1 = -world_normal_a = mb_normal`.
            let shift_a = pt_world - link_origin_a;
            let torque_a = shift_a.cross(mb_normal);

            // Penetration bias: rapier's clamped `erp_inv_dt · (dist + allowed_lin_err)`.
            let rhs_bias =
                (erp_inv_dt * (dist + allowed_lin_err)).clamp(-max_corr_velocity, 0.0);
            // Repulsion against any positive distance — clears float drift.
            let rhs_wo_bias = if dist > 0.0 { dist * inv_dt } else { 0.0 };

            let slot = count;
            let col_offset = col_base + (slot as usize) * dofs_stride;

            // Always start by writing A-side jacobian (overwriting any prior
            // slot content). For self-contacts we then accumulate the B-side.
            fill_contact_jac_row(
                body_jacobians,
                mb_jac_base,
                ndofs,
                mb_link_id_a,
                mb_normal,
                torque_a,
                contact_constraint_jacs,
                col_offset,
                false,
            );

            // For free contacts, the free body's J row is encoded via
            // `lin_jac` / `ang_jac` / `ii_ang_jac` on the constraint and
            // applied directly to `solver_vels` during solve. For
            // self-contacts, both sides go through the same multibody, so
            // the B-side jacobian must be added into the same `Jᵀ` row.
            let (ang_jac, ii_ang_jac, link_id_for_struct) = if is_self {
                let link_ws_b = ws_slice.read(mb_link_id_b as usize);
                let link_origin_b = link_ws_b.local_to_world.translation;
                let _ = link_origin_b_default;
                let shift_b = pt_world - link_origin_b;
                // B-side (link `mb_link_id_b`, rapier's body 2): impulse
                // along `+world_normal_a = lin_jac`. Torque arg matches
                // rapier's `torque_dir2 = dp2 × (-force_dir1) = shift_b ×
                // lin_jac`.
                let torque_b = shift_b.cross(lin_jac);
                fill_contact_jac_row(
                    body_jacobians,
                    mb_jac_base,
                    ndofs,
                    mb_link_id_b,
                    lin_jac,
                    torque_b,
                    contact_constraint_jacs,
                    col_offset,
                    true,
                );
                // ang_jac / ii_ang_jac aren't used by the solve path when
                // `free_body_id == u32::MAX`; keep them zero for clarity.
                (Vec3::ZERO, Vec3::ZERO, mb_link_id_a)
            } else {
                let _ = link_origin_b_default;
                let free_shift = pt_world - free_mp.com;
                let aj = free_shift.cross(lin_jac);
                let iiaj = free_mp.inv_inertia_mul(aj);
                (aj, iiaj, mb_link_id_a)
            };

            let cons = MultibodyContactConstraint {
                multibody_id: mb_idx,
                link_id: link_id_for_struct,
                kind: 1,
                free_body_id,
                free_body_im: free_im,
                _pad0: [0; 3],
                lin_jac,
                _pad1: 0,
                ang_jac,
                _pad2: 0,
                ii_ang_jac,
                _pad3: 0,
                inv_lhs: 0.0,
                rhs: rhs_wo_bias + rhs_bias,
                rhs_wo_bias,
                impulse: 0.0,
                cfm_coeff: 0.0,
                cfm_gain: 0.0,
                _pad4: [0; 2],
            };
            contact_constraints.write(cons_base + slot as usize, cons);
            count += 1;
        }
    }

    // Mark surplus slots as inactive so the solve sweep skips them.
    for s in count..MAX_MB_CONTACTS_PER_MB {
        let mut cz = contact_constraints.read(cons_base + s as usize);
        cz.kind = 0;
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
    let cons_base = cons_start + (mb_idx as usize) * (MAX_MB_CONTACTS_PER_MB as usize);
    let dofs_stride = *dof_batch_capacity as usize;
    let col_base =
        col_start + (mb_idx as usize) * (MAX_MB_CONTACTS_PER_MB as usize) * dofs_stride;

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
        //    ang_jac · ii_ang_jac. For self-contacts both sides are already
        //    folded into the multibody-side `Jᵀ`, so there's no free-body
        //    term — `inv_lhs` is just `1 / (Jᵀ·column)`.
        let mut cons = contact_constraints.read(cons_base + s as usize);
        let is_self = cons.free_body_id == u32::MAX;
        let inv_r_free = if is_self {
            0.0
        } else {
            cons.free_body_im + cons.ang_jac.dot(cons.ii_ang_jac)
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
    let cons_base = cons_start + (mb_idx as usize) * (MAX_MB_CONTACTS_PER_MB as usize);
    let dofs_stride = *dof_batch_capacity as usize;
    let col_base =
        col_start + (mb_idx as usize) * (MAX_MB_CONTACTS_PER_MB as usize) * dofs_stride;

    let count = contact_constraint_count.read(mb_start + mb_idx as usize);
    for s in 0..count {
        let mut cons = contact_constraints.read(cons_base + s as usize);
        if cons.kind == 0 {
            continue;
        }
        let col_offset = col_base + (s as usize) * dofs_stride;

        // J · u = J_mb · v_mb_dofs + J_free · v_free.
        // For self-contacts (`free_body_id == u32::MAX`), the B-side jacobian
        // is folded into `J_mb` already, so there's no separate free-body
        // term to add.
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
            j_dot_v += cons.lin_jac.dot(free.linear) + cons.ang_jac.dot(free.angular);
        }

        // rapier's contact PGS step: `dlambda = -r · (dvel + cfm·λ)` where
        // `dvel = J·u + rhs`. Note this is the OPPOSITE sign convention from
        // the joint-limit kernel, which uses `+r · (dvel - cfm·λ)`. The
        // difference: joint limits encode "target velocity = -rhs" with
        // positive rhs at excess; contacts encode "target separation = -rhs"
        // with negative rhs at penetration.
        let rhs_total = j_dot_v + cons.rhs;
        let raw_imp = cons.impulse
            - cons.inv_lhs * (rhs_total + cons.cfm_gain * cons.impulse);
        // Normal impulse must be ≥ 0 (no pulling apart).
        let new_imp = if raw_imp < 0.0 { 0.0 } else { raw_imp };
        let delta = new_imp - cons.impulse;
        cons.impulse = new_imp;
        contact_constraints.write(cons_base + s as usize, cons);

        if delta != 0.0 {
            // Multibody side: `Jᵀ` was packed with `mb_normal = -lin_jac` for
            // the A-side (and accumulated with `+lin_jac` for the B-side on
            // self-contacts). To push the multibody apart, add `delta·column`.
            for i in 0..ndofs {
                let v_idx = v_base + i as usize;
                let cur = dof_velocities.read(v_idx);
                let col = contact_constraint_columns.read(col_offset + i as usize);
                dof_velocities.write(v_idx, cur + delta * col);
            }
            if !is_self {
                // Free body side: solver_vels += delta · M_free⁻¹ · J_free^T.
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
    let cons_base = cons_start + (mb_idx as usize) * (MAX_MB_CONTACTS_PER_MB as usize);
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
