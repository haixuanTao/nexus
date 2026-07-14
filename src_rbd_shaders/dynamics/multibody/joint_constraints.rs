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
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

/// Lanes per workgroup for the cooperative fused `init_solve_joint` kernel: one
/// workgroup per articulation, the independent per-constraint init back-solves
/// split across these lanes (the serial PGS solve then runs on lane 0).
pub const MB_JOINT_INIT_LANES: u32 = 32;

use crate::DIM;
use crate::dynamics::joint::SPATIAL_DIM;
use crate::utils::BatchIndices;
use crate::utils::linalg::{MatSlice, lu_solve_in_place};

use super::types::{
    MultibodyInfo, MultibodyJointConstraint, MultibodyLinkStatic, MultibodyLinkWorkspace,
};

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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    links_workspace: &[MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] mass_matrices: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] lu_pivots: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)]
    joint_constraints: &mut [MultibodyJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)]
    joint_constraint_columns: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 8)] dt_uniform: &f32,
    #[spirv(uniform, descriptor_set = 0, binding = 9)] batch_ids: &BatchIndices,
) {
    let batch_id = invocation_id.y;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id as usize);
    if mb_idx >= num_mb {
        return;
    }
    init_joint_constraints_body(
        multibody_info,
        links_static,
        links_workspace,
        mass_matrices,
        lu_pivots,
        joint_constraints,
        joint_constraint_columns,
        batch_id,
        mb_idx,
        *dt_uniform,
        batch_ids,
        false,
    );
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    joint_constraints: &mut [MultibodyJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
) {
    let batch_id = invocation_id.y;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id as usize);
    if mb_idx >= num_mb {
        return;
    }

    let mb = batch_ids
        .mb_batch(batch_id, multibody_info)
        .read(mb_idx as usize);
    let cons_base = batch_ids.mb_joint_constraints_start(batch_id) + mb.first_constraint as usize;

    for s in 0..mb.max_constraints {
        let mut cons = joint_constraints.read(cons_base + s as usize);
        if cons.kind == 0 {
            continue;
        }
        cons.rhs = cons.rhs_wo_bias;
        joint_constraints.write(cons_base + s as usize, cons);
    }
}

/// PGS sweep body — shared between `gpu_mb_solve_joint_constraints` and
/// the fused init+solve / remove-bias+solve kernels. Writes back `cons`
/// before subtracting `delta · column` from `v`.
#[inline]
fn solve_joint_constraints_body(
    multibody_info: &[MultibodyInfo],
    joint_constraints: &mut [MultibodyJointConstraint],
    joint_constraint_columns: &[f32],
    dof_state: &mut [f32],
    batch_id: u32,
    mb_idx: u32,
    batch_ids: &BatchIndices,
) {
    let mb = batch_ids
        .mb_batch(batch_id, multibody_info)
        .read(mb_idx as usize);
    let ndofs = mb.ndofs;
    if ndofs == 0 || mb.max_constraints == 0 {
        return;
    }
    let v_base = batch_ids.dof_start(batch_id) + mb.first_dof as usize;
    let cons_base = batch_ids.mb_joint_constraints_start(batch_id) + mb.first_constraint as usize;
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let col_base = batch_ids.mb_joint_constraint_columns_start(batch_id)
        + (mb.first_constraint as usize) * dofs_stride;

    for s in 0..mb.max_constraints {
        let mut cons = joint_constraints.read(cons_base + s as usize);
        if cons.kind == 0 {
            continue;
        }

        let v_d = dof_state.read(v_base + cons.dof_id as usize);
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

        for i in 0..ndofs {
            let v_idx = v_base + i as usize;
            let cur = dof_state.read(v_idx);
            let col =
                joint_constraint_columns.read(col_base + (s as usize) * dofs_stride + i as usize);
            dof_state.write(v_idx, cur - delta * col);
        }
    }
}

/// Cooperative (threads(N)) PGS sweep — bit-identical to
/// `solve_joint_constraints_body` but the per-constraint `O(ndofs)` velocity
/// update is split across the workgroup's lanes. The sweep is Gauss-Seidel
/// (constraint `s+1` reads velocities written by `s`), so the constraint loop
/// stays serial: lane 0 computes the scalar impulse `delta` + writes the
/// constraint, broadcasts `delta` through `shared_delta`, then all lanes apply
/// disjoint DOFs. A barrier each iteration keeps the update visible before the
/// next constraint reads it. EVERY lane iterates EVERY constraint (no divergent
/// `continue`) so the barriers stay workgroup-uniform — inactive slots just
/// broadcast `delta = 0`.
#[inline]
fn solve_joint_constraints_par(
    multibody_info: &[MultibodyInfo],
    joint_constraints: &mut [MultibodyJointConstraint],
    joint_constraint_columns: &[f32],
    dof_state: &mut [f32],
    batch_id: u32,
    mb_idx: u32,
    batch_ids: &BatchIndices,
    lane: u32,
    num_lanes: u32,
    shared_delta: &mut impl MaybeIndexUnchecked<f32>,
) {
    let mb = batch_ids
        .mb_batch(batch_id, multibody_info)
        .read(mb_idx as usize);
    let ndofs = mb.ndofs;
    if ndofs == 0 || mb.max_constraints == 0 {
        return;
    }
    let v_base = batch_ids.dof_start(batch_id) + mb.first_dof as usize;
    let cons_base = batch_ids.mb_joint_constraints_start(batch_id) + mb.first_constraint as usize;
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let col_base = batch_ids.mb_joint_constraint_columns_start(batch_id)
        + (mb.first_constraint as usize) * dofs_stride;

    for s in 0..mb.max_constraints {
        if lane == 0 {
            let mut cons = joint_constraints.read(cons_base + s as usize);
            let delta = if cons.kind != 0 {
                let v_d = dof_state.read(v_base + cons.dof_id as usize);
                let rhs_total = v_d + cons.rhs;
                let raw_imp =
                    cons.impulse + cons.inv_lhs * (rhs_total - cons.cfm_gain * cons.impulse);
                let mut new_imp = raw_imp;
                if new_imp < cons.impulse_lo {
                    new_imp = cons.impulse_lo;
                }
                if new_imp > cons.impulse_hi {
                    new_imp = cons.impulse_hi;
                }
                let d = new_imp - cons.impulse;
                cons.impulse = new_imp;
                joint_constraints.write(cons_base + s as usize, cons);
                d
            } else {
                0.0
            };
            shared_delta.write(0, delta);
        }
        workgroup_memory_barrier_with_group_sync();
        let delta = shared_delta.read(0);
        if delta != 0.0 {
            let mut i = lane;
            while i < ndofs {
                let v_idx = v_base + i as usize;
                let cur = dof_state.read(v_idx);
                let col = joint_constraint_columns
                    .read(col_base + (s as usize) * dofs_stride + i as usize);
                dof_state.write(v_idx, cur - delta * col);
                i += num_lanes;
            }
        }
        // Update visible before the next constraint's lane-0 read; also guards
        // `shared_delta` before lane 0 overwrites it next iteration.
        workgroup_memory_barrier_with_group_sync();
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    joint_constraints: &mut [MultibodyJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] joint_constraint_columns: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dof_state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] batch_ids: &BatchIndices,
) {
    let batch_id = invocation_id.y;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id as usize);
    if mb_idx >= num_mb {
        return;
    }
    solve_joint_constraints_body(
        multibody_info,
        joint_constraints,
        joint_constraint_columns,
        dof_state,
        batch_id,
        mb_idx,
        batch_ids,
    );
}

#[inline]
/// Serial per-articulation constraint-discovery walk: emits one constraint slot
/// per active limit/motor axis. `defer_column`: when true, the expensive LU
/// back-solve is skipped (a later parallel phase fills `inv_lhs`/column from each
/// slot's `dof_id`) — used by the cooperative fused kernel so the back-solves
/// don't run serially inside this single-threaded walk. Pass `false` for the
/// standalone single-thread kernel (back-solve inline).
fn init_joint_constraints_body(
    multibody_info: &[MultibodyInfo],
    links_static: &[MultibodyLinkStatic],
    links_workspace: &[MultibodyLinkWorkspace],
    mass_matrices: &[f32],
    lu_pivots: &[u32],
    joint_constraints: &mut [MultibodyJointConstraint],
    joint_constraint_columns: &mut [f32],
    batch_id: u32,
    mb_idx: u32,
    dt: f32,
    batch_ids: &BatchIndices,
    defer_column: bool,
) {
    let mb = batch_ids
        .mb_batch(batch_id, multibody_info)
        .read(mb_idx as usize);
    let num_links = mb.num_links;
    let ndofs = mb.ndofs;
    if ndofs == 0 {
        return;
    }
    let mb_mm_base = batch_ids.mm_start(batch_id) + mb.mass_matrix_offset as usize;
    let piv_offset = batch_ids.dof_start(batch_id) + mb.first_dof as usize;
    let cons_base = batch_ids.mb_joint_constraints_start(batch_id) + mb.first_constraint as usize;
    // One column of M⁻¹ per constraint slot — `dof_batch_capacity` floats
    // per slot (only the first `ndofs` of each are meaningful, but we use
    // the batch-wide max as the stride to match the host allocation
    // `cons_col_cap = cons_cap * dofs_cap` and to avoid two multibodies
    // with different ndofs stomping on each other's columns).
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let col_base = batch_ids.mb_joint_constraint_columns_start(batch_id)
        + (mb.first_constraint as usize) * dofs_stride;

    let stat_slice = batch_ids
        .mb_links_batch(batch_id, links_static)
        .offset(mb.first_link as usize);
    let ws_slice = batch_ids
        .mb_links_batch(batch_id, links_workspace)
        .offset(mb.first_link as usize);
    let m = MatSlice::dense(mb_mm_base, ndofs, ndofs);

    for s in 0..mb.max_constraints {
        let mut cz: MultibodyJointConstraint = joint_constraints.read(cons_base + s as usize);
        cz.kind = 0;
        cz.impulse = 0.0;
        joint_constraints.write(cons_base + s as usize, cz);
    }

    let inv_dt = if dt != 0.0 { 1.0 / dt } else { 0.0 };

    let mut slot = 0u32;
    for k in 0..num_links {
        let stat = &stat_slice[k as usize];
        let ws = &ws_slice[k as usize];
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
        for axis in 0..DIM {
            if (locked & (1 << axis)) != 0 {
                continue;
            }
            let abs_dof = stat.assembly_id + curr_free_dof;
            let curr_pos = ws.coords.read(axis as usize);

            if (motor_axes & (1 << axis)) != 0 {
                let has_limits = (limit_axes & (1 << axis)) != 0;
                let limit_min = stat.data.limits[axis as usize].min;
                let limit_max = stat.data.limits[axis as usize].max;
                // Copy the motor BY VALUE before taking a reference: the
                // cuda-oxide codegen drops the dynamic index when lowering
                // `&stat.data.motors[axis]` as a call operand (it passes
                // `&motors[0]`), which made every motor read slot 0's default
                // (stiffness 0 ⇒ position targets ignored). By-value element
                // loads keep the index on both backends.
                let motor = stat.data.motors[axis as usize];
                emit_motor_constraint(
                    joint_constraints,
                    joint_constraint_columns,
                    cons_base,
                    col_base,
                    dofs_stride,
                    slot,
                    abs_dof,
                    ndofs,
                    curr_pos,
                    inv_dt,
                    dt,
                    &motor,
                    has_limits,
                    limit_min,
                    limit_max,
                    mass_matrices,
                    m,
                    lu_pivots,
                    piv_offset,
                    defer_column,
                );
                slot += 1;
            }
            if (limit_axes & (1 << axis)) != 0 {
                emit_limit_constraint(
                    joint_constraints,
                    joint_constraint_columns,
                    cons_base,
                    col_base,
                    dofs_stride,
                    slot,
                    abs_dof,
                    ndofs,
                    curr_pos,
                    [
                        stat.data.limits[axis as usize].min,
                        stat.data.limits[axis as usize].max,
                    ],
                    dt,
                    mass_matrices,
                    m,
                    lu_pivots,
                    piv_offset,
                    defer_column,
                );
                slot += 1;
            }
            curr_free_dof += 1;
        }

        // Angular DOFs.
        for axis in DIM..(SPATIAL_DIM as u32) {
            if (locked & (1 << axis)) != 0 {
                continue;
            }
            let abs_dof = stat.assembly_id + curr_free_dof;
            let curr_pos = ws.coords.read(axis as usize);

            if (limit_axes & (1 << axis)) != 0 {
                emit_limit_constraint(
                    joint_constraints,
                    joint_constraint_columns,
                    cons_base,
                    col_base,
                    dofs_stride,
                    slot,
                    abs_dof,
                    ndofs,
                    curr_pos,
                    [
                        stat.data.limits[axis as usize].min,
                        stat.data.limits[axis as usize].max,
                    ],
                    dt,
                    mass_matrices,
                    m,
                    lu_pivots,
                    piv_offset,
                    defer_column,
                );
                slot += 1;
            }
            if (motor_axes & (1 << axis)) != 0 {
                let has_limits = (limit_axes & (1 << axis)) != 0;
                let limit_min = stat.data.limits[axis as usize].min;
                let limit_max = stat.data.limits[axis as usize].max;
                // By-value copy — see the matching comment in the linear loop
                // (cuda-oxide drops the dynamic index on `&motors[axis]`).
                let motor = stat.data.motors[axis as usize];
                emit_motor_constraint(
                    joint_constraints,
                    joint_constraint_columns,
                    cons_base,
                    col_base,
                    dofs_stride,
                    slot,
                    abs_dof,
                    ndofs,
                    curr_pos,
                    inv_dt,
                    dt,
                    &motor,
                    has_limits,
                    limit_min,
                    limit_max,
                    mass_matrices,
                    m,
                    lu_pivots,
                    piv_offset,
                    defer_column,
                );
                slot += 1;
            }
            curr_free_dof += 1;
        }
    }
}

/// Parallel back-solve phase (pairs with `init_joint_constraints_body(.., defer_column=true)`).
/// Each lane owns a disjoint set of constraint slots (`s = lane, lane+num_lanes,
/// …`) and computes that slot's `M⁻¹·e_{dof_id}` column + `inv_lhs`. The slots
/// are independent (disjoint columns, read-only shared `M`), so this needs no
/// barriers. Mirrors the cooperative `gpu_mb_finalize_contact_constraints`.
#[inline]
fn compute_joint_columns_par(
    multibody_info: &[MultibodyInfo],
    joint_constraints: &mut [MultibodyJointConstraint],
    joint_constraint_columns: &mut [f32],
    mass_matrices: &[f32],
    lu_pivots: &[u32],
    batch_id: u32,
    mb_idx: u32,
    batch_ids: &BatchIndices,
    lane: u32,
    num_lanes: u32,
) {
    let mb = batch_ids
        .mb_batch(batch_id, multibody_info)
        .read(mb_idx as usize);
    let ndofs = mb.ndofs;
    if ndofs == 0 {
        return;
    }
    let mb_mm_base = batch_ids.mm_start(batch_id) + mb.mass_matrix_offset as usize;
    let piv_offset = batch_ids.dof_start(batch_id) + mb.first_dof as usize;
    let cons_base = batch_ids.mb_joint_constraints_start(batch_id) + mb.first_constraint as usize;
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let col_base = batch_ids.mb_joint_constraint_columns_start(batch_id)
        + (mb.first_constraint as usize) * dofs_stride;
    let m = MatSlice::dense(mb_mm_base, ndofs, ndofs);

    let mut s = lane;
    while s < mb.max_constraints {
        let mut cons = joint_constraints.read(cons_base + s as usize);
        if cons.kind != 0 {
            let inv_lhs = compute_constraint_column(
                joint_constraint_columns,
                col_base,
                s,
                dofs_stride,
                ndofs,
                cons.dof_id,
                mass_matrices,
                m,
                lu_pivots,
                piv_offset,
            );
            cons.inv_lhs = inv_lhs;
            joint_constraints.write(cons_base + s as usize, cons);
        }
        s += num_lanes;
    }
}

/// Solve `M · column = e_{dof_id}` and pack `inv_lhs = 1 / column[dof_id]`,
/// matching `inv_lhs = 1 / (Jᵀ M⁻¹ J)` for J = e_{dof_id}.
///
/// Per-constraint column stride is `dofs_stride` (= batch-wide
/// `dof_batch_capacity`), not this multibody's `ndofs` — see the matching
/// `col_base` computation in `gpu_mb_init_joint_constraints` for why.
#[inline]
fn compute_constraint_column(
    joint_constraint_columns: &mut [f32],
    col_base: usize,
    slot: u32,
    dofs_stride: usize,
    ndofs: u32,
    dof_id: u32,
    mass_matrices: &[f32],
    m: MatSlice,
    lu_pivots: &[u32],
    piv_offset: usize,
) -> f32 {
    let _ = ndofs;
    let col_offset = col_base + (slot as usize) * dofs_stride;
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
    dofs_stride: usize,
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
    // When true, skip the (expensive) LU back-solve here and leave `inv_lhs = 0`;
    // a later parallel phase recomputes the column from `dof_id`. Used by the
    // cooperative fused kernel so the back-solves don't run serially inside the
    // single-threaded discovery walk.
    defer_column: bool,
) {
    // Fixed regularization values matching rapier's defaults for joint softness:
    // erp_inv_dt = 1 / dt, cfm_coeff = 0 — full positional bias, no compliance.
    let erp_inv_dt = if dt != 0.0 { 1.0 / dt } else { 0.0 };
    let cfm_coeff = 0.0f32;

    let min_enabled = curr_pos < limits[0];
    let max_enabled = limits[1] < curr_pos;
    // INACTIVE limit (joint strictly inside its bounds): the row would carry
    // zero positional bias AND both impulse clamps at 0, so it can never apply
    // any impulse — dead weight in every PGS sweep. Skip emission: the slot
    // stays `kind = 0` (pre-zeroed by the discovery walk), which the deferred
    // column back-solve (`compute_joint_columns_par`) and both PGS sweeps
    // already skip. Result-identical, and it saves the O(ndofs) back-solve
    // column per limit per substep — a standing robot has essentially every
    // joint inside its limits every substep, so the joint-constraint kernel
    // drops from #joints rows to #violated rows (~0). Constraints are rebuilt
    // from current coords each substep, so a joint reaching its bound re-emits
    // on the next substep exactly as before.
    if !min_enabled && !max_enabled {
        return;
    }
    let lo_excess = (limits[0] - curr_pos).max(0.0);
    let hi_excess = (curr_pos - limits[1]).max(0.0);
    let rhs_bias = (hi_excess - lo_excess) * erp_inv_dt;
    let rhs_wo_bias = 0.0f32;

    let inv_lhs = if defer_column {
        0.0
    } else {
        compute_constraint_column(
            joint_constraint_columns,
            col_base,
            slot,
            dofs_stride,
            ndofs,
            dof_id,
            mass_matrices,
            m,
            lu_pivots,
            piv_offset,
        )
    };

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
    dofs_stride: usize,
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
    // See `emit_limit_constraint`: defer the LU back-solve to a parallel phase.
    defer_column: bool,
) {
    // Force-based motors are NOT emitted as soft constraints here — they are
    // applied as an explicit generalized-force PD torque in
    // `gpu_mb_gravity_and_lu` (`gen_forces += clamp(kp·(target−q) − kd·q̇,
    // ±max_force)`). The soft cfm_gain motor constraint under-realizes kp on
    // low-inertia joints (the ankles), so the articulation sags / buckles under
    // gravity. Leaving this slot untouched keeps it at the inactive `kind = 0`
    // written by the init zeroing pass, so the solver and the parallel
    // back-solve (`if cons.kind != 0`) both skip it — slot accounting is
    // unchanged.
    if motor.model == crate::dynamics::joint::FORCE_BASED {
        return;
    }
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

    let inv_lhs = if defer_column {
        0.0
    } else {
        compute_constraint_column(
            joint_constraint_columns,
            col_base,
            slot,
            dofs_stride,
            ndofs,
            dof_id,
            mass_matrices,
            m,
            lu_pivots,
            piv_offset,
        )
    };

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

/// Fused `init + solve_with_bias` for joint limit/motor constraints. Runs the
/// init logic (build constraints, LU back-solve to get M⁻¹·e_d columns) and
/// immediately follows with one PGS sweep WITH bias. Replaces two threads(1)
/// dispatches with one — at the cost of ~zero extra work, since both kernels
/// already iterate the same per-multibody slot loop.
#[spirv_bindgen]
#[spirv(compute(threads(32)))]
pub fn gpu_mb_init_solve_joint_with_bias(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    links_workspace: &[MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] mass_matrices: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] lu_pivots: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)]
    joint_constraints: &mut [MultibodyJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)]
    joint_constraint_columns: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] num_multibodies: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] dof_state: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 9)] dt_uniform: &f32,
    #[spirv(uniform, descriptor_set = 0, binding = 10)] batch_ids: &BatchIndices,
    #[spirv(workgroup)] shared_delta: &mut [f32; 1],
) {
    let batch_id = wg_id.y;
    let mb_idx = wg_id.x;
    let lane = lid.x;
    let num_mb = num_multibodies.read(batch_id as usize);
    // Uniform across the workgroup (all lanes share `mb_idx`), so every lane
    // takes the same branch — no barrier-divergence hazard below.
    if mb_idx >= num_mb {
        return;
    }
    // Phase 1 (lane 0): serial constraint-discovery walk, writing each slot's
    // metadata but DEFERRING the expensive LU back-solve. (Gating the back-solve
    // inside this serial walk wouldn't parallelize it — the walk runs in SIMT
    // lockstep, so one active lane per slot = still serial. Hence the split.)
    if lane == 0 {
        init_joint_constraints_body(
            multibody_info,
            links_static,
            links_workspace,
            mass_matrices,
            lu_pivots,
            joint_constraints,
            joint_constraint_columns,
            batch_id,
            mb_idx,
            *dt_uniform,
            batch_ids,
            true,
        );
    }
    // Make lane 0's constraint metadata visible to all lanes.
    workgroup_memory_barrier_with_group_sync();
    // Phase 2 (all lanes): the independent per-slot back-solves run in parallel,
    // each lane owning a disjoint set of slots (this is the actual win — same
    // structure as the cooperative finalize_contact kernel).
    compute_joint_columns_par(
        multibody_info,
        joint_constraints,
        joint_constraint_columns,
        mass_matrices,
        lu_pivots,
        batch_id,
        mb_idx,
        batch_ids,
        lane,
        MB_JOINT_INIT_LANES,
    );
    // Make every lane's column/inv_lhs writes visible before the PGS sweep.
    workgroup_memory_barrier_with_group_sync();
    // Phase 3: cooperative PGS solve — Gauss-Seidel across constraints (serial),
    // but the per-constraint DOF-apply is split across lanes.
    solve_joint_constraints_par(
        multibody_info,
        joint_constraints,
        joint_constraint_columns,
        dof_state,
        batch_id,
        mb_idx,
        batch_ids,
        lane,
        MB_JOINT_INIT_LANES,
        shared_delta,
    );
}

/// Fused `remove_bias + solve_without_bias` for joint constraints — runs once
/// per substep at the end of the substep, after position integration. Drops
/// one threads(1) dispatch per substep.
#[spirv_bindgen]
#[spirv(compute(threads(32)))]
pub fn gpu_mb_remove_solve_joint_no_bias(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    joint_constraints: &mut [MultibodyJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] joint_constraint_columns: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dof_state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] batch_ids: &BatchIndices,
    #[spirv(workgroup)] shared_delta: &mut [f32; 1],
) {
    let batch_id = wg_id.y;
    let mb_idx = wg_id.x;
    let lane = lid.x;
    let num_mb = num_multibodies.read(batch_id as usize);
    if mb_idx >= num_mb {
        return;
    }

    let mb = batch_ids
        .mb_batch(batch_id, multibody_info)
        .read(mb_idx as usize);
    let cons_base = batch_ids.mb_joint_constraints_start(batch_id) + mb.first_constraint as usize;

    // Inlined `remove_bias`: replace `rhs` with `rhs_wo_bias` for active slots.
    // Independent per slot → lane-split.
    let mut s = lane;
    while s < mb.max_constraints {
        let mut cons = joint_constraints.read(cons_base + s as usize);
        if cons.kind != 0 {
            cons.rhs = cons.rhs_wo_bias;
            joint_constraints.write(cons_base + s as usize, cons);
        }
        s += 32;
    }
    // Make every lane's `rhs` write visible before the cooperative solve reads
    // the constraints.
    workgroup_memory_barrier_with_group_sync();

    solve_joint_constraints_par(
        multibody_info,
        joint_constraints,
        joint_constraint_columns,
        dof_state,
        batch_id,
        mb_idx,
        batch_ids,
        lane,
        32,
        shared_delta,
    );
}
