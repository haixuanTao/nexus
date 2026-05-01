//! Multibody joint limit / motor constraints.
//!
//! Mirrors rapier's `unit_joint_limit_constraint` + `unit_joint_motor_constraint`
//! + the PGS solver. Each constraint targets a single generalized DOF `d`:
//!
//!   * jacobian = e_d (1 in slot d, 0 elsewhere)
//!   * inv_lhs  = 1 / (e_dᵀ · M⁻¹ · e_d)
//!   * column   = M⁻¹ · e_d         (full ndofs vector — used to update v)
//!
//! The solver iterates PGS sweeps:
//!
//!   rhs_total = J · v + self.rhs                   (= v[d] + bias)
//!   new_imp   = clamp(impulse + inv_lhs * (rhs_total - cfm_gain * impulse), bounds)
//!   Δimp      = new_imp - impulse
//!   v         -= Δimp · column                     (subtract: rapier's sign convention)
//!
//! Per-multibody, all constraint slots are scanned (`kind == 0` ones are skipped).

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

use crate::utils::Slice;
use crate::utils::linalg::{MatSlice, lu_solve_in_place};

use super::types::{
    MultibodyInfo, MultibodyJointConstraint, MultibodyLinkStatic, MultibodyLinkWorkspace,
};
use super::utils::coord_get;

/// Compute joint motor parameters mirroring rapier's `JointMotor::motor_params`.
#[inline]
fn motor_params(motor: &crate::dynamics::joint::JointMotor, dt: f32) -> (f32, f32, f32, f32, f32) {
    // Returns (erp_inv_dt, cfm_coeff, cfm_gain, target_vel_clamp_inv_dt, max_impulse).
    let inv_dt = if dt != 0.0 { 1.0 / dt } else { 0.0 };
    let mp = crate::dynamics::joint::motor_params(motor, dt);
    (
        mp.erp_inv_dt,
        mp.cfm_coeff,
        mp.cfm_gain,
        inv_dt,
        mp.max_impulse,
    )
}

/// Solve `M · x = e_d` in place (writes `x` into `dst[0..n]`). Uses the same
/// LU factor + pivots produced by `gpu_mb_lu_decompose`.
#[inline]
fn lu_solve_unit(
    buf_m: &[f32],
    m: MatSlice,
    buf_pivots: &[u32],
    pivots_offset: usize,
    dst: &mut [f32],
    dst_offset: usize,
    dof_id: u32,
) {
    let n = m.rows;
    // dst[0..n] := e_{dof_id}  (then permuted by lu_solve_in_place).
    for i in 0..n {
        dst[dst_offset + i as usize] = if i == dof_id { 1.0 } else { 0.0 };
    }
    lu_solve_in_place(buf_m, m, buf_pivots, pivots_offset, dst, dst_offset);
}

/// Initialize the multibody's joint-limit / joint-motor unit constraints.
///
/// For each link, scans every free DOF that has either `limit_axes` or `motor_axes`
/// set, and emits one `MultibodyJointConstraint` per active limit and one per
/// active motor (rapier emits these separately even when both are on the same axis).
///
/// Must run after `gpu_mb_lu_decompose` — the LU factors of `M` are used to compute
/// the per-constraint M⁻¹ column and effective inverse mass.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_init_joint_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] links_workspace: &[MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] mass_matrices: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] lu_pivots: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] joint_constraints: &mut [MultibodyJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] joint_constraint_columns: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] num_multibodies: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] dt_buf: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 9)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 10)] links_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 11)] mass_matrix_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 12)] dof_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 13)] joint_constraints_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 14)] joint_constraint_columns_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }
    let dt = dt_buf.read(0);

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let links_start = batch_id * *links_batch_capacity as usize;
    let mm_start = batch_id * *mass_matrix_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;
    let cons_start = batch_id * *joint_constraints_batch_capacity as usize;
    let col_start = batch_id * *joint_constraint_columns_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let num_links = mb.num_links;
    let ndofs = mb.ndofs;
    if ndofs == 0 {
        return;
    }
    let first_link_global = links_start + mb.first_link as usize;
    let mb_mm_base = mm_start + mb.mass_matrix_offset as usize;
    let piv_offset = dof_start + mb.first_dof as usize;
    let cons_base = cons_start + mb.first_constraint as usize;
    // One column of M⁻¹ per constraint slot, ndofs floats each.
    let col_base = col_start + (mb.first_constraint as usize) * (ndofs as usize);

    let stat_slice = Slice(links_static, first_link_global);
    let ws_slice = Slice(links_workspace, first_link_global);
    let m = MatSlice::dense(mb_mm_base, ndofs, ndofs);

    // Mark all slots as inactive; the loop below activates the live ones.
    for s in 0..mb.max_constraints {
        let mut cz: MultibodyJointConstraint = joint_constraints.read(cons_base + s as usize);
        cz.kind = 0;
        cz.impulse = 0.0;
        joint_constraints.write(cons_base + s as usize, cz);
    }

    let inv_dt = if dt != 0.0 { 1.0 / dt } else { 0.0 };

    let mut slot = 0u32;
    for k in 0..num_links {
        let stat = stat_slice.read(k as usize);
        let ws = ws_slice.read(k as usize);
        let locked = stat.data.locked_axes;
        let limit_axes = stat.data.limit_axes & !locked;
        let motor_axes = stat.data.motor_axes & !locked;
        if limit_axes == 0 && motor_axes == 0 {
            continue;
        }
        if stat.kinematic != 0 {
            continue;
        }

        // Walk free axes in DOF order, mirroring `MultibodyJoint::velocity_constraints`.
        // `curr_free_dof` tracks the position within this joint's slice of the
        // multibody's generalized-velocity vector; the absolute index is
        // `stat.assembly_id + curr_free_dof`.
        let mut curr_free_dof = 0u32;

        // Linear DOFs first.
        for axis in 0u32..3 {
            if (locked & (1 << axis)) != 0 {
                continue;
            }
            let abs_dof = stat.assembly_id + curr_free_dof;
            let curr_pos = coord_get(&ws.coords, axis);

            if (motor_axes & (1 << axis)) != 0 {
                let has_limits = (limit_axes & (1 << axis)) != 0;
                let limit_min = stat.data.limits[axis as usize].min;
                let limit_max = stat.data.limits[axis as usize].max;
                emit_motor_constraint(
                    joint_constraints,
                    joint_constraint_columns,
                    cons_base,
                    col_base,
                    slot,
                    abs_dof,
                    ndofs,
                    curr_pos,
                    inv_dt,
                    dt,
                    &stat.data.motors[axis as usize],
                    has_limits,
                    limit_min,
                    limit_max,
                    mass_matrices,
                    m,
                    lu_pivots,
                    piv_offset,
                );
                slot += 1;
            }
            if (limit_axes & (1 << axis)) != 0 {
                emit_limit_constraint(
                    joint_constraints,
                    joint_constraint_columns,
                    cons_base,
                    col_base,
                    slot,
                    abs_dof,
                    ndofs,
                    curr_pos,
                    [stat.data.limits[axis as usize].min, stat.data.limits[axis as usize].max],
                    dt,
                    mass_matrices,
                    m,
                    lu_pivots,
                    piv_offset,
                );
                slot += 1;
            }
            curr_free_dof += 1;
        }

        // Angular DOFs.
        for axis in 3u32..6 {
            if (locked & (1 << axis)) != 0 {
                continue;
            }
            let abs_dof = stat.assembly_id + curr_free_dof;
            let curr_pos = coord_get(&ws.coords, axis);

            if (limit_axes & (1 << axis)) != 0 {
                emit_limit_constraint(
                    joint_constraints,
                    joint_constraint_columns,
                    cons_base,
                    col_base,
                    slot,
                    abs_dof,
                    ndofs,
                    curr_pos,
                    [stat.data.limits[axis as usize].min, stat.data.limits[axis as usize].max],
                    dt,
                    mass_matrices,
                    m,
                    lu_pivots,
                    piv_offset,
                );
                slot += 1;
            }
            if (motor_axes & (1 << axis)) != 0 {
                let has_limits = (limit_axes & (1 << axis)) != 0;
                let limit_min = stat.data.limits[axis as usize].min;
                let limit_max = stat.data.limits[axis as usize].max;
                emit_motor_constraint(
                    joint_constraints,
                    joint_constraint_columns,
                    cons_base,
                    col_base,
                    slot,
                    abs_dof,
                    ndofs,
                    curr_pos,
                    inv_dt,
                    dt,
                    &stat.data.motors[axis as usize],
                    has_limits,
                    limit_min,
                    limit_max,
                    mass_matrices,
                    m,
                    lu_pivots,
                    piv_offset,
                );
                slot += 1;
            }
            curr_free_dof += 1;
        }
    }
}

/// Solve `M · column = e_{dof_id}` and pack `inv_lhs = 1 / column[dof_id]`,
/// matching `inv_lhs = 1 / (Jᵀ M⁻¹ J)` for J = e_{dof_id}.
#[inline]
fn compute_constraint_column(
    joint_constraint_columns: &mut [f32],
    col_base: usize,
    slot: u32,
    ndofs: u32,
    dof_id: u32,
    mass_matrices: &[f32],
    m: MatSlice,
    lu_pivots: &[u32],
    piv_offset: usize,
) -> f32 {
    let col_offset = col_base + (slot as usize) * (ndofs as usize);
    lu_solve_unit(
        mass_matrices,
        m,
        lu_pivots,
        piv_offset,
        joint_constraint_columns,
        col_offset,
        dof_id,
    );
    let lhs = joint_constraint_columns.read(col_offset + dof_id as usize);
    if lhs != 0.0 { 1.0 / lhs } else { 0.0 }
}

/// Initialize a single limit constraint slot. Mirrors rapier's
/// `unit_joint_limit_constraint`.
#[inline]
fn emit_limit_constraint(
    joint_constraints: &mut [MultibodyJointConstraint],
    joint_constraint_columns: &mut [f32],
    cons_base: usize,
    col_base: usize,
    slot: u32,
    dof_id: u32,
    ndofs: u32,
    curr_pos: f32,
    limits: [f32; 2],
    dt: f32,
    mass_matrices: &[f32],
    m: MatSlice,
    lu_pivots: &[u32],
    piv_offset: usize,
) {
    // Fixed regularization values matching rapier's defaults for joint softness:
    // erp_inv_dt = 1 / dt, cfm_coeff = 0 — full positional bias, no compliance.
    let erp_inv_dt = if dt != 0.0 { 1.0 / dt } else { 0.0 };
    let cfm_coeff = 0.0f32;

    let min_enabled = curr_pos < limits[0];
    let max_enabled = limits[1] < curr_pos;
    let lo_excess = (limits[0] - curr_pos).max(0.0);
    let hi_excess = (curr_pos - limits[1]).max(0.0);
    let rhs_bias = (hi_excess - lo_excess) * erp_inv_dt;
    let rhs_wo_bias = 0.0f32;

    let inv_lhs = compute_constraint_column(
        joint_constraint_columns,
        col_base,
        slot,
        ndofs,
        dof_id,
        mass_matrices,
        m,
        lu_pivots,
        piv_offset,
    );

    let max_neg_impulse = if min_enabled { -1.0e30f32 } else { 0.0 };
    let max_pos_impulse = if max_enabled { 1.0e30f32 } else { 0.0 };

    let cons = MultibodyJointConstraint {
        dof_id,
        kind: 1,
        _kind_extra: 0,
        _pad0: 0,
        rhs: rhs_wo_bias + rhs_bias,
        rhs_wo_bias,
        inv_lhs,
        impulse: 0.0,
        impulse_lo: max_neg_impulse,
        impulse_hi: max_pos_impulse,
        cfm_coeff,
        cfm_gain: 0.0,
    };
    joint_constraints.write(cons_base + slot as usize, cons);
}

/// Initialize a single motor constraint slot. Mirrors rapier's
/// `unit_joint_motor_constraint`. `has_limits` + `(limit_min, limit_max)` flatten
/// rapier's `Option<[Real; 2]>` parameter (rust-gpu can't represent enums).
#[inline]
fn emit_motor_constraint(
    joint_constraints: &mut [MultibodyJointConstraint],
    joint_constraint_columns: &mut [f32],
    cons_base: usize,
    col_base: usize,
    slot: u32,
    dof_id: u32,
    ndofs: u32,
    curr_pos: f32,
    inv_dt: f32,
    dt: f32,
    motor: &crate::dynamics::joint::JointMotor,
    has_limits: bool,
    limit_min: f32,
    limit_max: f32,
    mass_matrices: &[f32],
    m: MatSlice,
    lu_pivots: &[u32],
    piv_offset: usize,
) {
    let (erp_inv_dt, cfm_coeff, cfm_gain, _, max_impulse) = motor_params(motor, dt);

    let mut rhs_wo_bias = 0.0f32;
    if erp_inv_dt != 0.0 {
        rhs_wo_bias += (curr_pos - motor.target_pos) * erp_inv_dt;
    }

    let mut target_vel = motor.target_vel;
    if has_limits {
        let lo = (limit_min - curr_pos) * inv_dt;
        let hi = (limit_max - curr_pos) * inv_dt;
        if target_vel < lo {
            target_vel = lo;
        }
        if target_vel > hi {
            target_vel = hi;
        }
    }
    rhs_wo_bias += -target_vel;

    let inv_lhs = compute_constraint_column(
        joint_constraint_columns,
        col_base,
        slot,
        ndofs,
        dof_id,
        mass_matrices,
        m,
        lu_pivots,
        piv_offset,
    );

    let cons = MultibodyJointConstraint {
        dof_id,
        kind: 2,
        _kind_extra: 0,
        _pad0: 0,
        rhs: rhs_wo_bias,
        rhs_wo_bias,
        inv_lhs,
        impulse: 0.0,
        impulse_lo: -max_impulse,
        impulse_hi: max_impulse,
        cfm_coeff,
        cfm_gain,
    };
    joint_constraints.write(cons_base + slot as usize, cons);
}

/// Replace each active constraint's `rhs` with `rhs_wo_bias`, mirroring rapier's
/// `GenericJointConstraint::remove_bias_from_rhs`.
///
/// Used by the TGS-soft substep loop: bias-driven PGS happens before position
/// integration, then `remove_bias` runs and a final PGS sweep settles velocity
/// along constrained DOFs to zero (no rebound from positional bias).
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_remove_joint_constraint_bias(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] joint_constraints: &mut [MultibodyJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 4)] joint_constraints_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let cons_start = batch_id * *joint_constraints_batch_capacity as usize;
    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let cons_base = cons_start + mb.first_constraint as usize;

    for s in 0..mb.max_constraints {
        let mut cons = joint_constraints.read(cons_base + s as usize);
        if cons.kind == 0 {
            continue;
        }
        cons.rhs = cons.rhs_wo_bias;
        joint_constraints.write(cons_base + s as usize, cons);
    }
}

/// One PGS sweep: iterates the multibody's active limit/motor constraints and
/// updates `dof_velocities` in place. Mirrors rapier's `JointConstraint::solve_generic`
/// for a 1-DOF jacobian.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_solve_joint_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] joint_constraints: &mut [MultibodyJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] joint_constraint_columns: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dof_velocities: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] dof_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 7)] joint_constraints_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 8)] joint_constraint_columns_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;
    let cons_start = batch_id * *joint_constraints_batch_capacity as usize;
    let col_start = batch_id * *joint_constraint_columns_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let ndofs = mb.ndofs;
    if ndofs == 0 || mb.max_constraints == 0 {
        return;
    }
    let v_base = dof_start + mb.first_dof as usize;
    let cons_base = cons_start + mb.first_constraint as usize;
    let col_base = col_start + (mb.first_constraint as usize) * (ndofs as usize);

    for s in 0..mb.max_constraints {
        let mut cons = joint_constraints.read(cons_base + s as usize);
        if cons.kind == 0 {
            continue;
        }

        // J · v for J = e_{dof_id} is just v[dof_id].
        let v_d = dof_velocities.read(v_base + cons.dof_id as usize);
        let rhs_total = v_d + cons.rhs;
        let raw_imp = cons.impulse + cons.inv_lhs * (rhs_total - cons.cfm_gain * cons.impulse);
        let mut new_imp = raw_imp;
        if new_imp < cons.impulse_lo {
            new_imp = cons.impulse_lo;
        }
        if new_imp > cons.impulse_hi {
            new_imp = cons.impulse_hi;
        }
        let delta = new_imp - cons.impulse;
        cons.impulse = new_imp;
        joint_constraints.write(cons_base + s as usize, cons);

        // v -= delta · column   (column = M⁻¹ · e_d).
        for i in 0..ndofs {
            let v_idx = v_base + i as usize;
            let cur = dof_velocities.read(v_idx);
            let col = joint_constraint_columns.read(col_base + (s as usize) * (ndofs as usize) + i as usize);
            dof_velocities.write(v_idx, cur - delta * col);
        }
    }
}
