//! Fused multibody PGS sweep: joint limit/motor constraints followed by
//! contact constraints, in ONE dispatch per substep phase.
//!
//! Replaces the former `gpu_mb_solve_joint_constraints` /
//! `gpu_mb_remove_solve_joint_no_bias` / `gpu_mb_solve_contact_constraints` /
//! `gpu_mb_remove_contact_constraint_bias` chain (2-3 dispatches per phase,
//! each a fully serial one-thread-per-multibody loop):
//!
//! - One 64-lane workgroup per (multibody, batch); the multibody's
//!   generalized velocities live in WORKGROUP memory for the whole sweep, so
//!   the per-constraint `Δv = delta · column` updates and the `J·v` products
//!   run one-DOF-per-lane against shared memory instead of serial storage
//!   round-trips.
//! - The bias removal that used to be a separate read-modify-write dispatch
//!   is a `use_bias` uniform: the stabilization sweep simply reads
//!   `rhs_wo_bias` (the next substep re-initializes every constraint, so the
//!   persistent rewrite was never needed).
//!
//! The arithmetic per constraint is IDENTICAL to the serial kernels (same
//! product/sum order — lane 0 accumulates the lane products in DOF order), so
//! results are bit-exact with the former chain.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::iter::StepRng;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

use crate::dynamics::body::Velocity;
use crate::gdot;
use crate::utils::BatchIndices;
use crate::utils::linalg::MAX_MB_DOFS;

use super::types::{
    MAX_MB_CONTACT_CONSTRAINTS_PER_MB, MB_CONTACT_KIND_NORMAL, MB_CONTACT_KIND_TANGENT, MB_JOINT_KIND_LIMIT,
    MB_JOINT_KIND_MOTOR, MultibodyContactConstraint, MultibodyInfo, MultibodyJointConstraint,
};

const LANES: u32 = 64;

/// One PGS sweep over a multibody's joint (limit/motor) constraints followed
/// by its contact constraints. `use_bias = 0` runs the stabilization form
/// (`rhs_wo_bias`); non-zero runs the biased form (`rhs`).
///
/// Dispatch: one 64-lane workgroup per (multibody, batch) — thread grid
/// `[multibodies_per_batch · 64, num_batches, 1]`.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_solve_constraints(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    joint_constraints: &mut [MultibodyJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] joint_constraint_columns: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] contact_constraint_jacs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] contact_constraint_columns: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 6)] use_bias: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 7)] batch_ids: &BatchIndices,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] dof_state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] solver_vels: &mut [Velocity],
    #[spirv(workgroup)] dof_v: &mut [f32; MAX_MB_DOFS as usize],
    #[spirv(workgroup)] scratch: &mut [f32; LANES as usize],
    #[spirv(workgroup)] imp_shared: &mut [f32; MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize],
    #[spirv(workgroup)] delta_shared: &mut f32,
) {
    let batch_id = workgroup_id.y;
    let mb_idx = workgroup_id.x;
    let lane = local_id.x;
    let num_mb = batch_ids.multibodies_len;
    if mb_idx >= num_mb {
        return;
    }

    let mb = multibody_info.read(batch_ids.mbi(batch_id, mb_idx as usize));
    let ndofs = mb.ndofs;
    // Uniform per workgroup: every lane of this group returns together.
    if ndofs == 0 {
        return;
    }
    let use_bias = *use_bias != 0;

    let v_base = mb.first_dof as usize;
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let colliders_start = batch_ids.coll_start(batch_id);

    let jcons_base =
        batch_ids.mb_joint_constraints_start(batch_id) + mb.first_constraint as usize;
    let jcol_base = batch_ids.mb_joint_constraint_columns_start(batch_id)
        + (mb.first_constraint as usize) * dofs_stride;

    let ccons_base = batch_ids.mb_contact_constraints_start(batch_id)
        + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize);
    let ccol_base = batch_ids.mb_contact_constraint_columns_start(batch_id)
        + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize) * dofs_stride;

    let contact_count = mb.contact_constraint_count;
    // Nothing to solve (common for freely-swinging multibodies in tiny
    // batched environments): skip the load/store round-trip entirely.
    // Workgroup-uniform.
    if mb.max_constraints == 0 && contact_count == 0 {
        return;
    }

    // Load the generalized velocities and accumulated contact impulses into
    // workgroup memory. Impulses stay shared for the whole sweep so the
    // tangent clamp can read its normal's impulse without a storage fence.
    if lane < ndofs {
        dof_v.write(lane as usize, dof_state.read(batch_ids.mbi(batch_id, v_base + lane as usize)));
    }
    for s in StepRng::new(lane..contact_count, LANES) {
        imp_shared.write(s as usize, contact_constraints.read(ccons_base + s as usize).impulse);
    }
    workgroup_memory_barrier_with_group_sync();

    /*
     * Joint (limit/motor) sweep — mirrors the former
     * `solve_joint_constraints_body`, one constraint at a time.
     */
    for s in 0..mb.max_constraints {
        // Every lane reads the same constraint: all per-constraint scalars
        // below are workgroup-uniform, so no broadcast is needed.
        let cons = joint_constraints.read(jcons_base + s as usize);
        if cons.kind != MB_JOINT_KIND_LIMIT && cons.kind != MB_JOINT_KIND_MOTOR {
            // Unused slot or inactive limit. Uniform skip: all lanes take it
            // together (barrier-safe).
            continue;
        }

        let rhs = if use_bias { cons.rhs } else { cons.rhs_wo_bias };
        let v_d = dof_v.read(cons.dof_id as usize);
        let rhs_total = v_d + rhs;
        let raw_imp = cons.impulse + cons.inv_lhs * (rhs_total - cons.cfm_gain * cons.impulse);
        let mut new_imp = raw_imp;
        if new_imp < cons.impulse_lo {
            new_imp = cons.impulse_lo;
        }
        if new_imp > cons.impulse_hi {
            new_imp = cons.impulse_hi;
        }
        let delta = new_imp - cons.impulse;

        if lane == 0 {
            let mut cons = cons;
            cons.impulse = new_imp;
            joint_constraints.write(jcons_base + s as usize, cons);
        }

        // All lanes read `dof_v[dof_id]` above; sync before overwriting it.
        workgroup_memory_barrier_with_group_sync();
        if lane < ndofs {
            let col = joint_constraint_columns
                .read(jcol_base + (s as usize) * dofs_stride + lane as usize);
            dof_v.write(lane as usize, dof_v.read(lane as usize) - (delta * col));
        }
        workgroup_memory_barrier_with_group_sync();
    }

    /*
     * Contact sweep — mirrors the former `gpu_mb_solve_contact_constraints`,
     * one constraint at a time; the `J·v` products run one-DOF-per-lane.
     */
    for s in 0..contact_count {
        let cons = contact_constraints.read(ccons_base + s as usize);
        let col_offset = ccol_base + (s as usize) * dofs_stride;
        let is_self = cons.free_body_id == u32::MAX;

        // Multibody side of J · u, one product per lane; lane 0 sums them in
        // DOF order (bit-identical to the old serial accumulation).
        scratch.write(lane as usize, if lane < ndofs {
            contact_constraint_jacs.read(col_offset + lane as usize) * dof_v.read(lane as usize)
        } else {
            0.0
        });
        workgroup_memory_barrier_with_group_sync();

        if lane == 0 {
            let mut j_dot_v = 0.0f32;
            for i in 0..ndofs {
                j_dot_v += scratch.read(i as usize);
            }
            // Free-body side stays lane-0-local (reads its own prior writes in
            // program order, so no storage fence is needed within the sweep).
            let free = if is_self {
                Velocity::default()
            } else {
                solver_vels.read(colliders_start + cons.free_body_id as usize)
            };
            if !is_self {
                j_dot_v += cons.lin_jac.dot(free.linear) + gdot(cons.ang_jac, free.angular);
            }

            let rhs = if use_bias { cons.rhs } else { cons.rhs_wo_bias };
            let impulse = imp_shared.read(s as usize);
            let rhs_total = j_dot_v + rhs;
            // CFM-factor form (rapier's `*ContactConstraintNormalPart::generic_solve`).
            // NORMAL ONLY: rapier's TangentPart::solve has no cfm_factor — friction
            // is rigid. Applying the soft-contact cfm to tangents multiplied the
            // effective μ by cfm_factor (≈0.02 at the NF=30/ζ=5 config): μ_eff≈0.03
            // regardless of collider friction — the standing-creep bug.
            let unsoft_imp = impulse - cons.inv_lhs * rhs_total;
            let raw_imp = if cons.kind == MB_CONTACT_KIND_TANGENT {
                unsoft_imp
            } else {
                cons.cfm_factor * unsoft_imp
            };

            // Normal: clamp to ≥ 0. Friction tangent: clamp to
            // `±μ · normal_impulse` (box friction), reading the paired normal
            // slot's CURRENT impulse from shared memory.
            let new_imp = if cons.kind == MB_CONTACT_KIND_TANGENT {
                let limit =
                    cons.friction_coeff * imp_shared.read(cons.normal_constraint_slot as usize);
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
            let delta = new_imp - impulse;
            imp_shared.write(s as usize, new_imp);
            // Broadcast through a dedicated slot (NOT `scratch[0]`): the next
            // iteration's product writes into `scratch` must not race with the
            // other lanes' read of the broadcast below, and keeping them on
            // separate locations makes the two barriers per iteration enough.
            *delta_shared = delta;

            if delta != 0.0 && !is_self {
                let mut new_free = free;
                new_free.linear += cons.lin_jac * (cons.free_body_im * delta);
                new_free.angular += cons.ii_ang_jac * delta;
                solver_vels.write(colliders_start + cons.free_body_id as usize, new_free);
            }
        }
        workgroup_memory_barrier_with_group_sync();

        // Per-lane `dof_v[lane]` update: no lane reads another lane's DOF in
        // this loop (only `scratch` crosses lanes), so no end-of-iteration
        // barrier is needed — the next iteration's `scratch[lane]` write is
        // ordered by the barrier above.
        let delta = *delta_shared;
        if delta != 0.0 && lane < ndofs {
            let col = contact_constraint_columns.read(col_offset + lane as usize);
            dof_v.write(lane as usize, dof_v.read(lane as usize) + (delta * col));
        }
    }

    /*
     * Writeback: generalized velocities and accumulated contact impulses.
     */
    if lane < ndofs {
        dof_state.write(
            batch_ids.mbi(batch_id, v_base + lane as usize),
            dof_v.read(lane as usize),
        );
    }
    for s in StepRng::new(lane..contact_count, LANES) {
        let mut cons = contact_constraints.read(ccons_base + s as usize);
        cons.impulse = imp_shared.read(s as usize);
        contact_constraints.write(ccons_base + s as usize, cons);
    }
}

/*
 * Constraint-space (Delassus) contact solve — used instead of the fused
 * joint+contact sweep above when the per-multibody Delassus blocks are
 * allocated (small total multibody counts; the blocks are
 * `MAX_MB_CONTACT_CONSTRAINTS_PER_MB²` floats per multibody, so huge batched
 * scenes keep the dof-space path).
 *
 * The PGS recurrence for contact `s` needs `a[s] = J_s · u` under the CURRENT
 * velocities. The dof-space sweep recomputes that dot product from the
 * generalized velocities every iteration — an `O(ndofs)` latency chain per
 * constraint. In constraint space, `a[·]` is tracked incrementally instead:
 * applying `delta` at constraint `s` changes `a[j]` by exactly
 * `delta · D[s][j]` where
 *
 *   D[s][j] = jac_j · (M⁻¹ jac_sᵀ)                       (multibody coupling)
 *           + [same free body] im · lin_j·lin_s + ang_j · (I⁻¹ ang_s)
 *
 * `D` is precomputed by `gpu_mb_build_contact_delassus` right after the
 * columns are finalized (once per step in explicit-coriolis mode). The sweep
 * then runs on shared-memory scalars with one lane-parallel `D`-row update
 * per constraint. Algebraically identical to the dof-space sweep, but the
 * floating-point summation order differs (incremental vs recomputed), so the
 * two paths are NOT bit-identical.
 */

/// Joint-only PGS sweep (the joint half of [`gpu_mb_solve_constraints`]),
/// used by the Delassus path where the contact half runs in constraint space
/// as a separate dispatch (binding one kernel to both the joint and the
/// Delassus buffers would exceed the 8-storage-buffer budget).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_solve_joints(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    joint_constraints: &mut [MultibodyJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] joint_constraint_columns: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dof_state: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] use_bias: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 5)] batch_ids: &BatchIndices,
    #[spirv(workgroup)] dof_v: &mut [f32; MAX_MB_DOFS as usize],
) {
    let batch_id = workgroup_id.y;
    let mb_idx = workgroup_id.x;
    let lane = local_id.x;
    let num_mb = batch_ids.multibodies_len;

    let mb = multibody_info.read(batch_ids.mbi(batch_id, mb_idx as usize));
    let ndofs = mb.ndofs;
    // No early returns before the barriers: naga lowers them into breaks that
    // jump past the barriers, and Tint (browser WGSL) then rejects the whole
    // module. Guard the WORK on `live`; barriers run unconditionally.
    let live = mb_idx < num_mb && ndofs != 0 && mb.max_constraints != 0;
    let use_bias = *use_bias != 0;

    let v_base = mb.first_dof as usize;
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let jcons_base =
        batch_ids.mb_joint_constraints_start(batch_id) + mb.first_constraint as usize;
    let jcol_base = batch_ids.mb_joint_constraint_columns_start(batch_id)
        + (mb.first_constraint as usize) * dofs_stride;

    if live && lane < ndofs {
        dof_v.write(lane as usize, dof_state.read(batch_ids.mbi(batch_id, v_base + lane as usize)));
    }
    workgroup_memory_barrier_with_group_sync();

    // The loop carries barriers, so its bound must be provably uniform for the
    // browser's WGSL analysis: `mb.max_constraints` is a storage read (Tint
    // treats those as non-uniform), so the strict build loops to the
    // uniform-sourced batch capacity and guards the work on the real count.
    #[cfg(feature = "strict_uniformity")]
    let bound = batch_ids.mb_joint_constraints_batch_capacity;
    #[cfg(not(feature = "strict_uniformity"))]
    let bound = mb.max_constraints;
    for s in 0..bound {
        // Every lane reads the same constraint: all per-constraint scalars
        // below are workgroup-uniform, so no broadcast is needed. Unused
        // slots / inactive limits keep `active == false` — the lanes still
        // reach both barriers below (no uniform `continue`, see above).
        let in_range = live && s < mb.max_constraints;
        let cons = if in_range {
            joint_constraints.read(jcons_base + s as usize)
        } else {
            MultibodyJointConstraint::default()
        };
        let active =
            in_range && (cons.kind == MB_JOINT_KIND_LIMIT || cons.kind == MB_JOINT_KIND_MOTOR);

        let rhs = if use_bias { cons.rhs } else { cons.rhs_wo_bias };
        let v_d = dof_v.read(cons.dof_id as usize);
        let rhs_total = v_d + rhs;
        let raw_imp = cons.impulse + cons.inv_lhs * (rhs_total - cons.cfm_gain * cons.impulse);
        let mut new_imp = raw_imp;
        if new_imp < cons.impulse_lo {
            new_imp = cons.impulse_lo;
        }
        if new_imp > cons.impulse_hi {
            new_imp = cons.impulse_hi;
        }
        let delta = new_imp - cons.impulse;

        if active && lane == 0 {
            let mut cons = cons;
            cons.impulse = new_imp;
            joint_constraints.write(jcons_base + s as usize, cons);
        }

        // All lanes read `dof_v[dof_id]` above; sync before overwriting it.
        workgroup_memory_barrier_with_group_sync();
        if active && lane < ndofs {
            let col = joint_constraint_columns
                .read(jcol_base + (s as usize) * dofs_stride + lane as usize);
            dof_v.write(lane as usize, dof_v.read(lane as usize) - (delta * col));
        }
        workgroup_memory_barrier_with_group_sync();
    }

    if live && lane < ndofs {
        dof_state.write(
            batch_ids.mbi(batch_id, v_base + lane as usize),
            dof_v.read(lane as usize),
        );
    }
}

/// Fills the per-multibody Delassus block `D[s][j] = ∂a[j]/∂impulse[s]` (row
/// `s` = the effect OF constraint `s`, laid out row-contiguously so the solve
/// kernel's per-iteration row update reads coalesced). Runs right after
/// `gpu_mb_finalize_contact_constraints` (it consumes the M⁻¹Jᵀ columns).
///
/// The free-body coupling term is only nonzero for two constraints on the
/// SAME free body, and vanishes automatically for static free bodies (their
/// `free_body_im` and `ii_ang_jac` are zero).
///
/// One 64-lane workgroup per (multibody, batch); lanes stride the
/// `count × count` pairs.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_build_contact_delassus(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    contact_constraints: &[MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contact_constraint_jacs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] contact_constraint_columns: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] delassus: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] batch_ids: &BatchIndices,
) {
    const MAXC: u32 = MAX_MB_CONTACT_CONSTRAINTS_PER_MB;
    let batch_id = workgroup_id.y;
    let mb_idx = workgroup_id.x;
    let lane = local_id.x;
    let num_mb = batch_ids.multibodies_len;
    if mb_idx >= num_mb {
        return;
    }

    let mb = multibody_info.read(batch_ids.mbi(batch_id, mb_idx as usize));
    let ndofs = mb.ndofs;
    let count = mb.contact_constraint_count;
    if ndofs == 0 || count == 0 {
        return;
    }

    let cons_base = batch_ids.mb_contact_constraints_start(batch_id)
        + (mb_idx as usize) * (MAXC as usize);
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let col_base = batch_ids.mb_contact_constraint_columns_start(batch_id)
        + (mb_idx as usize) * (MAXC as usize) * dofs_stride;
    let d_base = ((batch_id * batch_ids.multibodies_batch_capacity + mb_idx) as usize)
        * (MAXC as usize)
        * (MAXC as usize);

    // Pair `p = s · count + j`: consecutive lanes share the source row `s`
    // and vary the target `j`, so the column reads of `s` broadcast and the
    // `D` writes coalesce.
    let num_pairs = count * count;
    for p in StepRng::new(lane..num_pairs, LANES) {
        let (s, j) = crate::div_rem_nz(p, count);

        // Multibody coupling: jac_j · (M⁻¹ jac_sᵀ).
        let jac_j_off = col_base + (j as usize) * dofs_stride;
        let col_s_off = col_base + (s as usize) * dofs_stride;
        let mut v = 0.0f32;
        for i in 0..ndofs {
            let jj = contact_constraint_jacs.read(jac_j_off + i as usize);
            let cs = contact_constraint_columns.read(col_s_off + i as usize);
            v += jj * cs;
        }

        // Free-body coupling (impulse at `s` moves the shared free body,
        // which feeds `a[j]`'s free-side term). Zero for self-contacts and
        // static free bodies.
        let cons_s = contact_constraints.read(cons_base + s as usize);
        let cons_j = contact_constraints.read(cons_base + j as usize);
        if cons_s.free_body_id != u32::MAX && cons_s.free_body_id == cons_j.free_body_id {
            v += cons_s.free_body_im * cons_j.lin_jac.dot(cons_s.lin_jac)
                + gdot(cons_j.ang_jac, cons_s.ii_ang_jac);
        }

        delassus.write(d_base + (s * MAXC + j) as usize, v);
    }
}

/// Constraint-space contact sweep: tracks `a[s] = J_s · u` incrementally in
/// workgroup memory using the precomputed Delassus rows, so each PGS
/// iteration is a couple of shared-memory scalars plus one lane-parallel row
/// update — instead of the dof-space `O(ndofs)` product/update latency chain.
///
/// Runs AFTER [`gpu_mb_solve_joints`] (whose dof-velocity changes are folded
/// in by the fresh `a[·]` evaluation below). One 64-lane workgroup per
/// (multibody, batch).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_solve_contacts_delassus(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contact_constraint_jacs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] contact_constraint_columns: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] delassus: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] use_bias: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] batch_ids: &BatchIndices,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] dof_state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] solver_vels: &mut [Velocity],
    #[spirv(workgroup)] dof_v: &mut [f32; MAX_MB_DOFS as usize],
    #[spirv(workgroup)] a_shared: &mut [f32; MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize],
    #[spirv(workgroup)] imp_shared: &mut [f32; MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize],
    #[spirv(workgroup)] rhs_shared: &mut [f32; MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize],
    #[spirv(workgroup)] inv_lhs_shared: &mut [f32; MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize],
    #[spirv(workgroup)] cfm_shared: &mut [f32; MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize],
    #[spirv(workgroup)] friction_shared: &mut [f32; MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize],
    #[spirv(workgroup)] meta_shared: &mut [u32; MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize],
) {
    const MAXC: u32 = MAX_MB_CONTACT_CONSTRAINTS_PER_MB;
    let batch_id = workgroup_id.y;
    let mb_idx = workgroup_id.x;
    let lane = local_id.x;
    let num_mb = batch_ids.multibodies_len;

    let mb = multibody_info.read(batch_ids.mbi(batch_id, mb_idx as usize));
    let ndofs = mb.ndofs;
    // No early returns before the barriers: naga lowers them into breaks that
    // jump past the barriers, and Tint (browser WGSL) then rejects the whole
    // module. Guard the WORK on `live`/`count`; barriers run unconditionally.
    let live = mb_idx < num_mb && ndofs != 0;
    let count = if live { mb.contact_constraint_count } else { 0 };
    let use_bias = *use_bias != 0;

    let v_base = mb.first_dof as usize;
    let colliders_start = batch_ids.coll_start(batch_id);
    let cons_base = batch_ids.mb_contact_constraints_start(batch_id)
        + (mb_idx as usize) * (MAXC as usize);
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let col_base = batch_ids.mb_contact_constraint_columns_start(batch_id)
        + (mb_idx as usize) * (MAXC as usize) * dofs_stride;
    let d_base = ((batch_id * batch_ids.multibodies_batch_capacity + mb_idx) as usize)
        * (MAXC as usize)
        * (MAXC as usize);

    if live && lane < ndofs {
        dof_v.write(lane as usize, dof_state.read(batch_ids.mbi(batch_id, v_base + lane as usize)));
    }

    // Preload the per-constraint solve scalars into shared SoA arrays. The
    // serial recurrence below then never touches storage on its critical
    // path. `meta` packs the kind, the paired normal slot, and whether the
    // free-body side needs the (storage) fire-and-forget velocity update
    // (dynamic free body only — zero-mass floors skip it entirely).
    for s in StepRng::new(lane..count, LANES) {
        let cons = contact_constraints.read(cons_base + s as usize);
        imp_shared.write(s as usize, cons.impulse);
        rhs_shared.write(s as usize, if use_bias { cons.rhs } else { cons.rhs_wo_bias });
        // Stiction anchor: pull the contact back toward where it gripped at the
        // step's manifold build. Positional bias — with-bias pass only.
        #[cfg(feature = "dim3")]
        if use_bias && cons.kind == MB_CONTACT_KIND_TANGENT {
            let gain = f32::from_bits(cons._pad4[0]);
            let mc = f32::from_bits(cons._pad4[1]);
            let b = cons._unused_cfm * gain;
            rhs_shared.write(s as usize, rhs_shared.read(s as usize) + (if b > mc { mc } else if b < -mc { -mc } else { b }));
        }
        // Live-depth ERP (explicit mode's per-substep contact refresh): normal
        // rows recompute their positional bias each substep from the integrated
        // depth estimate (dist drifts as the link moves between manifold
        // rebuilds — the stale per-step bias is what makes coarse sim steps
        // topple a statically-loaded stance). friction_coeff on normal rows
        // carries erp_inv_dt; _unused_cfm is (dist + allowed), integrated below.
        #[cfg(feature = "dim3")]
        if use_bias && cons.kind == MB_CONTACT_KIND_NORMAL {
            let erp = cons.friction_coeff;
            let mc = f32::from_bits(cons._pad4[1]);
            let b = cons._unused_cfm * erp;
            let bias = if b < -mc { -mc } else if b > 0.0 { 0.0 } else { b };
            rhs_shared.write(s as usize, cons.rhs_wo_bias + bias);
        }
        inv_lhs_shared.write(s as usize, cons.inv_lhs);
        cfm_shared.write(s as usize, cons.cfm_factor);
        friction_shared.write(s as usize, cons.friction_coeff);
        let is_self = cons.free_body_id == u32::MAX;
        let free_active = !is_self
            && (cons.free_body_im != 0.0 || gdot(cons.ii_ang_jac, cons.ii_ang_jac) != 0.0);
        meta_shared.write(s as usize, (cons.kind & 0xff)
            | ((cons.normal_constraint_slot & 0xffff) << 8)
            | (if free_active { 1 << 24 } else { 0 }));
    }
    workgroup_memory_barrier_with_group_sync();

    // Fresh `a[s] = J_s · u` under the current (post-joint-sweep, post-
    // warmstart) velocities. One slot per lane, strided.
    for s in StepRng::new(lane..count, LANES) {
        let jac_off = col_base + (s as usize) * dofs_stride;
        let mut dot = 0.0f32;
        for i in 0..ndofs {
            dot += contact_constraint_jacs.read(jac_off + i as usize) * dof_v.read(i as usize);
        }
        let cons = contact_constraints.read(cons_base + s as usize);
        if cons.free_body_id != u32::MAX {
            let free = solver_vels.read(colliders_start + cons.free_body_id as usize);
            dot += cons.lin_jac.dot(free.linear) + gdot(cons.ang_jac, free.angular);
        }
        a_shared.write(s as usize, dot);
    }
    workgroup_memory_barrier_with_group_sync();

    // The sweep carries a barrier, so under `strict_uniformity` its bound must
    // be a compile-time constant (`count` is a storage read, and even `delta`
    // below is shared-memory-derived — Tint treats both as non-uniform) and
    // the per-iteration barrier runs unconditionally.
    #[cfg(feature = "strict_uniformity")]
    let sweep_bound = MAXC;
    #[cfg(not(feature = "strict_uniformity"))]
    let sweep_bound = count;
    for s in 0..sweep_bound {
        let sweep_active = s < count;
        // Every lane computes the same scalar recurrence from shared memory —
        // all inputs are workgroup-uniform, so no broadcast is needed.
        let (new_imp, delta) = if sweep_active {
            let meta = meta_shared.read(s as usize);
            let kind = meta & 0xff;
            let normal_slot = (meta >> 8) & 0xffff;

            let impulse = imp_shared.read(s as usize);
            let rhs_total = a_shared.read(s as usize) + rhs_shared.read(s as usize);
            // CFM-factor form (rapier's `*ContactConstraintNormalPart::generic_solve`).
            // NORMAL ONLY — see the scalar sweep above: friction tangents are rigid.
            let unsoft_imp = impulse - inv_lhs_shared.read(s as usize) * rhs_total;
            let raw_imp = if kind == MB_CONTACT_KIND_TANGENT {
                unsoft_imp
            } else {
                cfm_shared.read(s as usize) * unsoft_imp
            };

            let new_imp = if kind == MB_CONTACT_KIND_TANGENT {
                let limit =
                    friction_shared.read(s as usize) * imp_shared.read(normal_slot as usize);
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
            (new_imp, new_imp - impulse)
        } else {
            (0.0, 0.0)
        };

        // Uniform skip (delta is workgroup-uniform): nothing changed — and in
        // the non-strict build no barrier is needed either, since no shared
        // writes happened this iteration.
        if delta != 0.0 {
            if lane == 0 {
                imp_shared.write(s as usize, new_imp);
                let free_active = (meta_shared.read(s as usize) >> 24) != 0;
                // Fire-and-forget free-body velocity update (only when the
                // free body is dynamic). Not on the recurrence's critical
                // path: `a`'s free-side coupling is already inside `D`.
                if free_active {
                    let cons = contact_constraints.read(cons_base + s as usize);
                    let mut free =
                        solver_vels.read(colliders_start + cons.free_body_id as usize);
                    free.linear += cons.lin_jac * (cons.free_body_im * delta);
                    free.angular += cons.ii_ang_jac * delta;
                    solver_vels.write(colliders_start + cons.free_body_id as usize, free);
                }
            }
            // Lane-parallel Delassus row update (row `s` is contiguous), plus
            // the off-path dof update (each lane owns its DOF; nothing reads
            // `dof_v` again until the writeback).
            let d_row = d_base + (s * MAXC) as usize;
            for j in StepRng::new(lane..count, LANES) {
                a_shared.write(j as usize, a_shared.read(j as usize) + (delta * delassus.read(d_row + j as usize)));
            }
            if lane < ndofs {
                let col = contact_constraint_columns
                    .read(col_base + (s as usize) * dofs_stride + lane as usize);
                dof_v.write(lane as usize, dof_v.read(lane as usize) + (delta * col));
            }
            #[cfg(not(feature = "strict_uniformity"))]
            workgroup_memory_barrier_with_group_sync();
        }
        #[cfg(feature = "strict_uniformity")]
        workgroup_memory_barrier_with_group_sync();
    }

    /*
     * Writeback: generalized velocities and accumulated contact impulses.
     */
    if lane < ndofs {
        dof_state.write(
            batch_ids.mbi(batch_id, v_base + lane as usize),
            dof_v.read(lane as usize),
        );
    }
    for s in StepRng::new(lane..count, LANES) {
        let mut cons = contact_constraints.read(ccons_writeback_guard(cons_base, s));
        cons.impulse = imp_shared.read(s as usize);
        // Stiction anchor: integrate this substep's residual tangential motion
        // (a_shared[s] = J·u under the POST-solve velocities — kept current by
        // the sweep's Delassus row updates) into the slip accumulator. Once per
        // substep: on the stabilization (wo-bias) pass, which runs last.
        #[cfg(feature = "dim3")]
        if !use_bias
            && (cons.kind == MB_CONTACT_KIND_TANGENT || cons.kind == MB_CONTACT_KIND_NORMAL)
        {
            let gain = f32::from_bits(cons._pad4[0]);
            if gain > 0.0 {
                // tangent: slip += J·v·dt′ ; normal: depth += J·v·dt′
                cons._unused_cfm += a_shared.read(s as usize) / gain;
            }
        }
        contact_constraints.write(ccons_writeback_guard(cons_base, s), cons);
    }
}

/// Trivial index helper (keeps the writeback lines within the formatter's
/// width).
#[inline(always)]
fn ccons_writeback_guard(cons_base: usize, s: u32) -> usize {
    cons_base + s as usize
}
