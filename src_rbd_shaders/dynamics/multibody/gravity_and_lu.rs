//! Fused `apply_gravity_with_coriolis` + LU factor + LU solve kernel.
//!
//! Replaces a 3-dispatch chain (`gpu_mb_apply_gravity_with_coriolis` →
//! `gpu_mb_lu_factor_and_solve`) with a single workgroup-parallel kernel,
//! removing two WebGPU dispatch round-trips per `compute_dynamics` call.
//!
//! Lane partitioning matches the originals: gravity assembly is per-link
//! sequential (kinematic_acc chained parent → child) with the per-link
//! `Aᵀ·f` scatter parallelised across DOFs; LU then operates on a workgroup
//! tile of the mass matrix.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

use glamx::{Vec2, Vec3, Vec4};

use crate::dynamics::body::{LocalMassProperties, Velocity};
use crate::dynamics::joint::SPATIAL_DIM;
use crate::utils::linalg::{MAX_MB_DOFS, MatSlice, fill_par};
#[cfg(feature = "dim2")]
use crate::utils::linalg::gemv_tr_spatial_split_par;
#[cfg(feature = "dim3")]
use crate::utils::linalg::gemv_tr_spatial_split_sparse_par;
use crate::utils::{BatchIndices, Slice};
use crate::{AngVector, Vector, gcross_av};

use super::lu::{LANES, NO_PARENT, ltdl_factor_in_shared, ltdl_solve_in_shared, sm_idx};
use super::mass_matrix::link_world_inertia;
use super::types::{MultibodyInfo, MultibodyLinkStatic, MultibodyLinkWorkspace};

/// Fused gravity / Coriolis-force assembly + LU factor + LU solve.
///
/// Bindings count (`14` storage max in 3D, `10` in 2D):
///   - `multibody_info`, `links_static`, `links_workspace`,
///     `links_local_mprops`, `body_jacobians`, `gen_forces`, `mass_matrices`,
///     `lu_pivots`, `dof_velocities`, `damping`, `num_multibodies`, `gravity`
///   = 12 storage buffers, plus 4 uniform `*_batch_capacity` slots.
///
/// Workgroup memory: matrix tile (`32×32 f32 = 4 KiB`) + rhs (`32 f32`) +
/// per-lane reduction scratch (`32 f32`) + two scalar broadcast slots ≈ 4.5
/// KiB, well under the 19 904 B limit configured by the testbed.
#[spirv_bindgen]
#[spirv(compute(threads(32, 1, 1)))]
pub fn gpu_mb_gravity_and_lu(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    links_workspace: &mut [MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    links_local_mprops: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] body_jacobians: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] gen_forces: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] mass_matrices: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] lu_pivots: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] dof_state: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 10)] gravity: &Vec4,
    #[spirv(uniform, descriptor_set = 0, binding = 11)] batch_ids: &BatchIndices,
    // Per-DOF Coulomb joint friction (MJCF `frictionloss`, N·m). Same per-env-per-
    // DOF layout as the velocity section (indexed `gen_base + i`); 0 for the root
    // and any unactuated DOF.
    #[spirv(storage_buffer, descriptor_set = 0, binding = 12)] dof_frictionloss: &[f32],
    // Actuator-delay state, per batch, stride `2 + links_per_batch`:
    //   [0] tick — physics-step counter within the control step (f32; the
    //       host zeroes it once per control step, THIS kernel bumps it),
    //   [1] k — delay in physics steps (0 = no delay, the default),
    //   [2 + link] — the PREVIOUS control step's motor position target for
    //       that link's motorized axis.
    // While tick < k the PD uses the previous target (WBC-AGILE's
    // DelayedPDActuator semantics) with ZERO mid-step host writes — the old
    // host-side restage stalled the stream on a pageable H2D copy per substep.
    // NOTE: the tick bump assumes ONE multibody per batch (zealot's layout);
    // with several, extra multibodies race the bump by one step at worst.
    #[spirv(storage_buffer, descriptor_set = 0, binding = 13)] motor_delay_state: &mut [f32],
    // Mass-matrix tile in shared memory.
    #[spirv(workgroup)] mat: &mut [f32; (MAX_MB_DOFS * MAX_MB_DOFS) as usize],
    // RHS / solution vector.
    #[spirv(workgroup)] x: &mut [f32; MAX_MB_DOFS],
    // Per-lane partial sums for the triangular-solve tree reduction.
    #[spirv(workgroup)] partial: &mut [f32; LANES as usize],
    // Lane-0 → all-lanes broadcast slots used by the LU pivot step.
    #[spirv(workgroup)] pivot_row_shared: &mut u32,
    #[spirv(workgroup)] inv_akk_shared: &mut f32,
) {
    let batch_id = wg_id.y;
    let mb_idx = wg_id.x;
    let lane = lid.x;

    let mb = batch_ids
        .mb_batch(batch_id, multibody_info)
        .read(mb_idx as usize);
    let num_links = mb.num_links;
    let ndofs = mb.ndofs;
    let mb_jac_base = batch_ids.jac_start(batch_id) + mb.jacobian_offset as usize;
    let gen_base = batch_ids.dof_start(batch_id) + mb.first_dof as usize;
    let mb_mm_base = batch_ids.mm_start(batch_id) + mb.mass_matrix_offset as usize;
    let piv_offset = gen_base;
    let rhs_offset = gen_base;

    let stat_slice = batch_ids
        .mb_links_batch(batch_id, links_static)
        .offset(mb.first_link as usize);
    let mut ws_slice = batch_ids
        .mb_links_batch_mut(batch_id, links_workspace)
        .offset(mb.first_link as usize);
    let local_mprops_slice = batch_ids
        .mb_links_batch(batch_id, links_local_mprops)
        .offset(mb.first_link as usize);
    let vel_slice = Slice(dof_state, gen_base);
    let damping_slice = Slice(
        dof_state,
        batch_ids.dof_damping_section_offset as usize + gen_base,
    );
    let frictionloss_slice = Slice(dof_frictionloss, gen_base);

    // ---- Phase 1: zero the generalized-force vector (parallel across DOFs). ----
    let accelerations = MatSlice::dense(gen_base, ndofs, 1);
    // TODO(perf): up to a certain number of degrees of freedom, we could actually run all the
    //             calculations in shared memory and only write the result in the end.
    //             Currently, the max number of dofs is 32 but we still accumulate forces/accelerations
    //             in global memory whereas it could be done in shared memory instead.
    fill_par(gen_forces, accelerations, 0.0, lane, LANES);
    workgroup_memory_barrier_with_group_sync();

    let _ = stat_slice;

    #[cfg(feature = "dim3")]
    let g = Vec3::new(gravity.x, gravity.y, gravity.z);
    #[cfg(feature = "dim2")]
    let g = Vec2::new(gravity.x, gravity.y);

    // ---- Phase 2: per-link gravity / Coriolis-force assembly. ----
    // NOTE: fixed number of iterations for uniform control flow.
    // TODO(PERF): on non-web platforms we could just use `num_links` as the upper bound.
    for k in 0..MAX_MB_DOFS as u32 {
        let active = k < num_links;
        let mut acc_lin = Vector::ZERO;
        #[cfg(feature = "dim3")]
        let mut acc_ang: AngVector = AngVector::ZERO;
        #[cfg(feature = "dim2")]
        let mut acc_ang: AngVector = 0.0;

        if active {
            let (
                self_joint_vel_lin,
                self_joint_vel_ang,
                self_shift02,
                self_shift23,
                _self_local_to_world,
                self_rb_ang,
            ) = {
                let ws = &ws_slice[k as usize];
                (
                    ws.joint_velocity.linear,
                    ws.joint_velocity.angular,
                    ws.shift02,
                    ws.shift23,
                    ws.local_to_world,
                    ws.rb_vels.angular,
                )
            };

            if k != 0 {
                let stat = stat_slice[k as usize];
                let parent_ws = &ws_slice[stat.parent_link_id as usize];
                let parent_acc_lin = parent_ws.kinematic_acc.linear;
                let parent_acc_ang = parent_ws.kinematic_acc.angular;
                let parent_ang = parent_ws.rb_vels.angular;

                acc_lin = parent_acc_lin;
                acc_ang = parent_acc_ang;

                acc_lin += gcross_av(parent_ang, self_joint_vel_lin) * 2.0;
                #[cfg(feature = "dim3")]
                {
                    acc_ang += parent_ang.cross(self_joint_vel_ang);
                }
                #[cfg(feature = "dim2")]
                {
                    let _ = self_joint_vel_ang;
                }
                acc_lin += gcross_av(parent_ang, gcross_av(parent_ang, self_shift02));
                acc_lin += gcross_av(parent_acc_ang, self_shift02);
            } else {
                let _ = self_joint_vel_ang;
                let _ = self_shift02;
            }
            let rb_ang = self_rb_ang;
            acc_lin += gcross_av(rb_ang, gcross_av(rb_ang, self_shift23));
            acc_lin += gcross_av(acc_ang, self_shift23);

            if lane == 0 {
                ws_slice[k as usize].kinematic_acc = Velocity::new(acc_lin, acc_ang);
            }
        }

        // Top-level barrier: reached uniformly by every lane on every outer
        // iteration so children see the just-published `kinematic_acc`.
        workgroup_memory_barrier_with_group_sync();

        if active {
            let rb_ang = ws_slice[k as usize].rb_vels.angular;
            let lmp = local_mprops_slice[k as usize];
            let inv_mass_x = lmp.inv_mass.x;
            if inv_mass_x != 0.0 {
                let mass = 1.0 / inv_mass_x;
                let rb_inertia = link_world_inertia(&ws_slice[k as usize], &lmp);

                #[cfg(feature = "dim3")]
                let gyroscopic = {
                    let i_omega = rb_inertia * rb_ang;
                    rb_ang.cross(i_omega)
                };
                #[cfg(feature = "dim2")]
                let gyroscopic: AngVector = 0.0;

                let i_acc_ang = rb_inertia * acc_ang;

                #[cfg(feature = "dim3")]
                let f_lin = (g - acc_lin) * mass;
                #[cfg(feature = "dim2")]
                let f_lin = (g - acc_lin) * mass;
                let f_ang = -gyroscopic - i_acc_ang;

                // Chain-sparse jacobian: read only the stored ancestor-chain
                // columns (lane = global DOF, rank via popcount); off-chain
                // DOFs have exactly-zero formal columns and are skipped.
                #[cfg(feature = "dim3")]
                {
                    let stat = stat_slice[k as usize];
                    let body_jacobian = MatSlice::dense(
                        mb_jac_base + stat.jac_offset as usize,
                        SPATIAL_DIM as u32,
                        stat.jac_chain_mask.count_ones(),
                    );
                    gemv_tr_spatial_split_sparse_par(
                        gen_forces,
                        gen_base,
                        1.0,
                        body_jacobians,
                        body_jacobian,
                        stat.jac_chain_mask,
                        f_lin,
                        f_ang,
                        1.0,
                        lane,
                        LANES,
                    );
                }
                #[cfg(feature = "dim2")]
                {
                    let body_jacobian = MatSlice::dense(
                        mb_jac_base + (k as usize) * SPATIAL_DIM * (ndofs as usize),
                        SPATIAL_DIM as u32,
                        ndofs,
                    );
                    gemv_tr_spatial_split_par(
                        gen_forces,
                        gen_base,
                        1.0,
                        body_jacobians,
                        body_jacobian,
                        f_lin,
                        f_ang,
                        1.0,
                        lane,
                        LANES,
                    );
                }
            }
        }
    }

    // Damping subtraction is handled at solve time via the LU rhs (we still
    // need the read-modify-write done by lane d): we don't have `dt` here, so
    // damping is applied identically to the original kernel: rhs -=
    // damping[d] * v[d] regardless of `dt` (matches rapier's `cmpy(-1.0,
    // damping, velocities, 1.0)` form, which does not include `dt`).
    workgroup_memory_barrier_with_group_sync();
    let i = lane;
    if i < ndofs {
        let idx = gen_base + i as usize;
        let cur = gen_forces.read(idx);
        let v = vel_slice[i as usize];
        // Viscous damping (linear) + Coulomb joint friction (frictionloss). The
        // Coulomb term -fl·sign(v) is smoothed near v=0 via clamp(v/eps) (eps=1
        // rad/s) so it tapers to viscous instead of chattering across zero. It's
        // applied explicitly here (not folded into M like viscous damping) since
        // Coulomb isn't linear in v.
        let coulomb = frictionloss_slice[i as usize] * (v / 1.0).clamp(-1.0, 1.0);
        gen_forces.write(idx, cur - damping_slice[i as usize] * v - coulomb);
    }
    workgroup_memory_barrier_with_group_sync();

    // ---- Explicit force-based motor PD torque ----
    // For motors with `model == FORCE_BASED`, apply the actuator torque
    // τ = clamp(stiffness·(target − q) − damping·q̇, ±max_force) DIRECTLY as a
    // generalized force (added to `gen_forces` before the LU solve, so it enters
    // the accelerations like any applied force), instead of as the soft
    // cfm_gain motor constraint (which under-realizes kp on the low-inertia leg
    // joints, so the robot sags / buckles under gravity). This matches the real
    // robot and MuJoCo's position actuator exactly: τ = kp·err − kv·q̇ with fixed
    // gains. AccelerationBased motors (model 0) are untouched — they still go
    // through the constraint path. Run serially on lane 0 over the links
    // (`num_links` is small, ~13); each motor maps to a distinct DOF so there's
    // no double-write, and `gen_forces` for this mb is final after the damping
    // pass above.
    if lane == 0 {
        // Actuator delay: while this control step's physics-step counter is
        // below the batch's delay, the PD tracks the PREVIOUS target.
        let delay_stride = 2 + batch_ids.links_batch_capacity as usize;
        let delay_base = batch_id as usize * delay_stride;
        let tick = motor_delay_state.read(delay_base);
        let delay_k = motor_delay_state.read(delay_base + 1);
        let use_prev = tick < delay_k;
        for k in 0..num_links {
            let stat = stat_slice[k as usize];
            if stat.kinematic != 0 {
                continue;
            }
            let locked = stat.data.locked_axes;
            let motor_axes = stat.data.motor_axes & !locked;
            if motor_axes == 0 {
                continue;
            }
            let ws = &ws_slice[k as usize];
            // Walk the free axes in DOF order (mirrors `init_joint_constraints`),
            // tracking the DOF offset within this joint's slice.
            let mut curr_free_dof = 0u32;
            for axis in 0..(SPATIAL_DIM as u32) {
                if (locked & (1 << axis)) != 0 {
                    continue;
                }
                if (motor_axes & (1 << axis)) != 0 {
                    // By-value element load — cuda-oxide drops the dynamic index
                    // on `&motors[axis]` (see init_joint_constraints).
                    let motor = stat.data.motors[axis as usize];
                    if motor.model == crate::dynamics::joint::FORCE_BASED {
                        let q = ws.coords.read(axis as usize);
                        let abs_dof = stat.assembly_id + curr_free_dof;
                        let v = vel_slice[abs_dof as usize];
                        let target = if use_prev {
                            motor_delay_state
                                .read(delay_base + 2 + (mb.first_link + k) as usize)
                        } else {
                            motor.target_pos
                        };
                        let tau = (motor.stiffness * (target - q)
                            - motor.damping * v)
                            .clamp(-motor.max_force, motor.max_force);
                        let idx = gen_base + abs_dof as usize;
                        gen_forces.write(idx, gen_forces.read(idx) + tau);
                    }
                }
                curr_free_dof += 1;
            }
        }
        // Bump the per-control-step physics-step counter (single writer: one
        // multibody per batch — see the binding comment).
        if mb_idx == 0 {
            motor_delay_state.write(delay_base, tick + 1.0);
        }
    }
    workgroup_memory_barrier_with_group_sync();

    // ---- Phase 3: load M into shared memory, factor in place. ----
    let m_view = MatSlice::dense(mb_mm_base, ndofs, ndofs);
    if lane < ndofs {
        for r in 0..ndofs {
            mat.write(sm_idx(r, lane), mass_matrices.read(m_view.idx(r, lane)));
        }
        x.write(lane as usize, gen_forces.read(rhs_offset + lane as usize));
    }

    // Per-DOF parent array (tree metadata for the sparse LᵀDL factor/solves),
    // written into the old pivots buffer so downstream solves read the tree
    // from the binding they already have. A joint's DOFs chain internally;
    // its first DOF hangs off the last DOF of the nearest ancestor link that
    // has any (welded links contribute none).
    if lane == 0 {
        let mut k = 0u32;
        while k < num_links {
            let stat = stat_slice[k as usize];
            if stat.ndofs > 0 {
                let mut p = NO_PARENT;
                if k != 0 {
                    let mut a = stat.parent_link_id;
                    for _ in 0..MAX_MB_DOFS as u32 {
                        let s = stat_slice[a as usize];
                        if s.ndofs > 0 {
                            p = s.assembly_id + s.ndofs - 1;
                            break;
                        }
                        if a == 0 {
                            break;
                        }
                        a = s.parent_link_id;
                    }
                }
                lu_pivots.write(piv_offset + stat.assembly_id as usize, p);
                for t in 1..stat.ndofs {
                    lu_pivots.write(
                        piv_offset + (stat.assembly_id + t) as usize,
                        stat.assembly_id + t - 1,
                    );
                }
            }
            k += 1;
        }
    }
    workgroup_memory_barrier_with_group_sync();

    ltdl_factor_in_shared(ndofs, lane, mat, lu_pivots, piv_offset);

    // Persist the LᵀDL factors to global memory (joint / contact constraint
    // init reuses them for unit-RHS solves).
    if lane < ndofs {
        for r in 0..ndofs {
            mass_matrices.write(m_view.idx(r, lane), mat.read(sm_idx(r, lane)));
        }
    }

    // ---- Phase 4: solve M·x = τ for the gravity rhs. ----
    ltdl_solve_in_shared(ndofs, lane, mat, lu_pivots, piv_offset, x);

    let _ = partial;
    let _ = pivot_row_shared;
    let _ = inv_akk_shared;
    if lane < ndofs {
        gen_forces.write(rhs_offset + lane as usize, x.read(lane as usize));
    }
}

/// Phases 1–2 of `gpu_mb_gravity_and_lu` only: gravity / Coriolis force
/// assembly + damping + explicit PD motors + per-DOF parent-array build —
/// NO factor, NO solve. Used by the `NEXUS_ENV_PER_LANE=1` path, where the
/// tree-sparse LᵀDL factor + gravity solve run in `gpu_mb_ltdl_lanes`
/// (one multibody per LANE on global memory) instead of serially on lane 0
/// of this kernel's workgroup.
#[spirv_bindgen]
#[spirv(compute(threads(32, 1, 1)))]
pub fn gpu_mb_gravity_rhs(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    links_workspace: &mut [MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    links_local_mprops: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] body_jacobians: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] gen_forces: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] lu_pivots: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] dof_state: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 9)] gravity: &Vec4,
    #[spirv(uniform, descriptor_set = 0, binding = 10)] batch_ids: &BatchIndices,
    // Per-DOF Coulomb joint friction (MJCF `frictionloss`, N·m). Same per-env-per-
    // DOF layout as the velocity section (indexed `gen_base + i`); 0 for the root
    // and any unactuated DOF.
    #[spirv(storage_buffer, descriptor_set = 0, binding = 11)] dof_frictionloss: &[f32],
    // Actuator-delay state, per batch, stride `2 + links_per_batch`:
    //   [0] tick — physics-step counter within the control step (f32; the
    //       host zeroes it once per control step, THIS kernel bumps it),
    //   [1] k — delay in physics steps (0 = no delay, the default),
    //   [2 + link] — the PREVIOUS control step's motor position target for
    //       that link's motorized axis.
    // While tick < k the PD uses the previous target (WBC-AGILE's
    // DelayedPDActuator semantics) with ZERO mid-step host writes — the old
    // host-side restage stalled the stream on a pageable H2D copy per substep.
    // NOTE: the tick bump assumes ONE multibody per batch (zealot's layout);
    // with several, extra multibodies race the bump by one step at worst.
    #[spirv(storage_buffer, descriptor_set = 0, binding = 12)] motor_delay_state: &mut [f32],
) {
    let batch_id = wg_id.y;
    let mb_idx = wg_id.x;
    let lane = lid.x;

    let mb = batch_ids
        .mb_batch(batch_id, multibody_info)
        .read(mb_idx as usize);
    let num_links = mb.num_links;
    let ndofs = mb.ndofs;
    let mb_jac_base = batch_ids.jac_start(batch_id) + mb.jacobian_offset as usize;
    let gen_base = batch_ids.dof_start(batch_id) + mb.first_dof as usize;
    let mb_mm_base = batch_ids.mm_start(batch_id) + mb.mass_matrix_offset as usize;
    let piv_offset = gen_base;
    let rhs_offset = gen_base;

    let stat_slice = batch_ids
        .mb_links_batch(batch_id, links_static)
        .offset(mb.first_link as usize);
    let mut ws_slice = batch_ids
        .mb_links_batch_mut(batch_id, links_workspace)
        .offset(mb.first_link as usize);
    let local_mprops_slice = batch_ids
        .mb_links_batch(batch_id, links_local_mprops)
        .offset(mb.first_link as usize);
    let vel_slice = Slice(dof_state, gen_base);
    let damping_slice = Slice(
        dof_state,
        batch_ids.dof_damping_section_offset as usize + gen_base,
    );
    let frictionloss_slice = Slice(dof_frictionloss, gen_base);

    // ---- Phase 1: zero the generalized-force vector (parallel across DOFs). ----
    let accelerations = MatSlice::dense(gen_base, ndofs, 1);
    // TODO(perf): up to a certain number of degrees of freedom, we could actually run all the
    //             calculations in shared memory and only write the result in the end.
    //             Currently, the max number of dofs is 32 but we still accumulate forces/accelerations
    //             in global memory whereas it could be done in shared memory instead.
    fill_par(gen_forces, accelerations, 0.0, lane, LANES);
    workgroup_memory_barrier_with_group_sync();

    let _ = stat_slice;

    #[cfg(feature = "dim3")]
    let g = Vec3::new(gravity.x, gravity.y, gravity.z);
    #[cfg(feature = "dim2")]
    let g = Vec2::new(gravity.x, gravity.y);

    // ---- Phase 2: per-link gravity / Coriolis-force assembly. ----
    // NOTE: fixed number of iterations for uniform control flow.
    // TODO(PERF): on non-web platforms we could just use `num_links` as the upper bound.
    for k in 0..MAX_MB_DOFS as u32 {
        let active = k < num_links;
        let mut acc_lin = Vector::ZERO;
        #[cfg(feature = "dim3")]
        let mut acc_ang: AngVector = AngVector::ZERO;
        #[cfg(feature = "dim2")]
        let mut acc_ang: AngVector = 0.0;

        if active {
            let (
                self_joint_vel_lin,
                self_joint_vel_ang,
                self_shift02,
                self_shift23,
                _self_local_to_world,
                self_rb_ang,
            ) = {
                let ws = &ws_slice[k as usize];
                (
                    ws.joint_velocity.linear,
                    ws.joint_velocity.angular,
                    ws.shift02,
                    ws.shift23,
                    ws.local_to_world,
                    ws.rb_vels.angular,
                )
            };

            if k != 0 {
                let stat = stat_slice[k as usize];
                let parent_ws = &ws_slice[stat.parent_link_id as usize];
                let parent_acc_lin = parent_ws.kinematic_acc.linear;
                let parent_acc_ang = parent_ws.kinematic_acc.angular;
                let parent_ang = parent_ws.rb_vels.angular;

                acc_lin = parent_acc_lin;
                acc_ang = parent_acc_ang;

                acc_lin += gcross_av(parent_ang, self_joint_vel_lin) * 2.0;
                #[cfg(feature = "dim3")]
                {
                    acc_ang += parent_ang.cross(self_joint_vel_ang);
                }
                #[cfg(feature = "dim2")]
                {
                    let _ = self_joint_vel_ang;
                }
                acc_lin += gcross_av(parent_ang, gcross_av(parent_ang, self_shift02));
                acc_lin += gcross_av(parent_acc_ang, self_shift02);
            } else {
                let _ = self_joint_vel_ang;
                let _ = self_shift02;
            }
            let rb_ang = self_rb_ang;
            acc_lin += gcross_av(rb_ang, gcross_av(rb_ang, self_shift23));
            acc_lin += gcross_av(acc_ang, self_shift23);

            if lane == 0 {
                ws_slice[k as usize].kinematic_acc = Velocity::new(acc_lin, acc_ang);
            }
        }

        // Top-level barrier: reached uniformly by every lane on every outer
        // iteration so children see the just-published `kinematic_acc`.
        workgroup_memory_barrier_with_group_sync();

        if active {
            let rb_ang = ws_slice[k as usize].rb_vels.angular;
            let lmp = local_mprops_slice[k as usize];
            let inv_mass_x = lmp.inv_mass.x;
            if inv_mass_x != 0.0 {
                let mass = 1.0 / inv_mass_x;
                let rb_inertia = link_world_inertia(&ws_slice[k as usize], &lmp);

                #[cfg(feature = "dim3")]
                let gyroscopic = {
                    let i_omega = rb_inertia * rb_ang;
                    rb_ang.cross(i_omega)
                };
                #[cfg(feature = "dim2")]
                let gyroscopic: AngVector = 0.0;

                let i_acc_ang = rb_inertia * acc_ang;

                #[cfg(feature = "dim3")]
                let f_lin = (g - acc_lin) * mass;
                #[cfg(feature = "dim2")]
                let f_lin = (g - acc_lin) * mass;
                let f_ang = -gyroscopic - i_acc_ang;

                // Chain-sparse jacobian: read only the stored ancestor-chain
                // columns (lane = global DOF, rank via popcount); off-chain
                // DOFs have exactly-zero formal columns and are skipped.
                #[cfg(feature = "dim3")]
                {
                    let stat = stat_slice[k as usize];
                    let body_jacobian = MatSlice::dense(
                        mb_jac_base + stat.jac_offset as usize,
                        SPATIAL_DIM as u32,
                        stat.jac_chain_mask.count_ones(),
                    );
                    gemv_tr_spatial_split_sparse_par(
                        gen_forces,
                        gen_base,
                        1.0,
                        body_jacobians,
                        body_jacobian,
                        stat.jac_chain_mask,
                        f_lin,
                        f_ang,
                        1.0,
                        lane,
                        LANES,
                    );
                }
                #[cfg(feature = "dim2")]
                {
                    let body_jacobian = MatSlice::dense(
                        mb_jac_base + (k as usize) * SPATIAL_DIM * (ndofs as usize),
                        SPATIAL_DIM as u32,
                        ndofs,
                    );
                    gemv_tr_spatial_split_par(
                        gen_forces,
                        gen_base,
                        1.0,
                        body_jacobians,
                        body_jacobian,
                        f_lin,
                        f_ang,
                        1.0,
                        lane,
                        LANES,
                    );
                }
            }
        }
    }

    // Damping subtraction is handled at solve time via the LU rhs (we still
    // need the read-modify-write done by lane d): we don't have `dt` here, so
    // damping is applied identically to the original kernel: rhs -=
    // damping[d] * v[d] regardless of `dt` (matches rapier's `cmpy(-1.0,
    // damping, velocities, 1.0)` form, which does not include `dt`).
    workgroup_memory_barrier_with_group_sync();
    let i = lane;
    if i < ndofs {
        let idx = gen_base + i as usize;
        let cur = gen_forces.read(idx);
        let v = vel_slice[i as usize];
        // Viscous damping (linear) + Coulomb joint friction (frictionloss). The
        // Coulomb term -fl·sign(v) is smoothed near v=0 via clamp(v/eps) (eps=1
        // rad/s) so it tapers to viscous instead of chattering across zero. It's
        // applied explicitly here (not folded into M like viscous damping) since
        // Coulomb isn't linear in v.
        let coulomb = frictionloss_slice[i as usize] * (v / 1.0).clamp(-1.0, 1.0);
        gen_forces.write(idx, cur - damping_slice[i as usize] * v - coulomb);
    }
    workgroup_memory_barrier_with_group_sync();

    // ---- Explicit force-based motor PD torque ----
    // For motors with `model == FORCE_BASED`, apply the actuator torque
    // τ = clamp(stiffness·(target − q) − damping·q̇, ±max_force) DIRECTLY as a
    // generalized force (added to `gen_forces` before the LU solve, so it enters
    // the accelerations like any applied force), instead of as the soft
    // cfm_gain motor constraint (which under-realizes kp on the low-inertia leg
    // joints, so the robot sags / buckles under gravity). This matches the real
    // robot and MuJoCo's position actuator exactly: τ = kp·err − kv·q̇ with fixed
    // gains. AccelerationBased motors (model 0) are untouched — they still go
    // through the constraint path. Run serially on lane 0 over the links
    // (`num_links` is small, ~13); each motor maps to a distinct DOF so there's
    // no double-write, and `gen_forces` for this mb is final after the damping
    // pass above.
    if lane == 0 {
        // Actuator delay: while this control step's physics-step counter is
        // below the batch's delay, the PD tracks the PREVIOUS target.
        let delay_stride = 2 + batch_ids.links_batch_capacity as usize;
        let delay_base = batch_id as usize * delay_stride;
        let tick = motor_delay_state.read(delay_base);
        let delay_k = motor_delay_state.read(delay_base + 1);
        let use_prev = tick < delay_k;
        for k in 0..num_links {
            let stat = stat_slice[k as usize];
            if stat.kinematic != 0 {
                continue;
            }
            let locked = stat.data.locked_axes;
            let motor_axes = stat.data.motor_axes & !locked;
            if motor_axes == 0 {
                continue;
            }
            let ws = &ws_slice[k as usize];
            // Walk the free axes in DOF order (mirrors `init_joint_constraints`),
            // tracking the DOF offset within this joint's slice.
            let mut curr_free_dof = 0u32;
            for axis in 0..(SPATIAL_DIM as u32) {
                if (locked & (1 << axis)) != 0 {
                    continue;
                }
                if (motor_axes & (1 << axis)) != 0 {
                    // By-value element load — cuda-oxide drops the dynamic index
                    // on `&motors[axis]` (see init_joint_constraints).
                    let motor = stat.data.motors[axis as usize];
                    if motor.model == crate::dynamics::joint::FORCE_BASED {
                        let q = ws.coords.read(axis as usize);
                        let abs_dof = stat.assembly_id + curr_free_dof;
                        let v = vel_slice[abs_dof as usize];
                        let target = if use_prev {
                            motor_delay_state
                                .read(delay_base + 2 + (mb.first_link + k) as usize)
                        } else {
                            motor.target_pos
                        };
                        let tau = (motor.stiffness * (target - q)
                            - motor.damping * v)
                            .clamp(-motor.max_force, motor.max_force);
                        let idx = gen_base + abs_dof as usize;
                        gen_forces.write(idx, gen_forces.read(idx) + tau);
                    }
                }
                curr_free_dof += 1;
            }
        }
        // Bump the per-control-step physics-step counter (single writer: one
        // multibody per batch — see the binding comment).
        if mb_idx == 0 {
            motor_delay_state.write(delay_base, tick + 1.0);
        }
    }
    workgroup_memory_barrier_with_group_sync();

    // Per-DOF parent array (tree metadata for the sparse LᵀDL factor/solves),
    // written into the old pivots buffer so downstream solves read the tree
    // from the binding they already have. A joint's DOFs chain internally;
    // its first DOF hangs off the last DOF of the nearest ancestor link that
    // has any (welded links contribute none).
    if lane == 0 {
        let mut k = 0u32;
        while k < num_links {
            let stat = stat_slice[k as usize];
            if stat.ndofs > 0 {
                let mut p = NO_PARENT;
                if k != 0 {
                    let mut a = stat.parent_link_id;
                    for _ in 0..MAX_MB_DOFS as u32 {
                        let s = stat_slice[a as usize];
                        if s.ndofs > 0 {
                            p = s.assembly_id + s.ndofs - 1;
                            break;
                        }
                        if a == 0 {
                            break;
                        }
                        a = s.parent_link_id;
                    }
                }
                lu_pivots.write(piv_offset + stat.assembly_id as usize, p);
                for t in 1..stat.ndofs {
                    lu_pivots.write(
                        piv_offset + (stat.assembly_id + t) as usize,
                        stat.assembly_id + t - 1,
                    );
                }
            }
            k += 1;
        }
    }
}

/// Gravity / Coriolis-force assembly, ONE MULTIBODY PER LANE (3D).
///
/// Drop-in replacement for `gpu_mb_gravity_rhs` on the
/// `NEXUS_ENV_PER_LANE=1` path. The warp-per-multibody original pays for
/// structure, not bytes (proven by the chain-sparse storage migration
/// moving it 0%): every lane redundantly computes the kinematic
/// acceleration, the per-link loop runs to the uniform bound
/// `MAX_MB_DOFS` (= 64) with TWO barriers per iteration (~128 barriers
/// for ~30 real links), and the motor-PD + pivot phases run serially on
/// lane 0 with 31 lanes idle. Here thread `t = wg.x·32 + lane` owns
/// multibody `(batch = t / mb_cap, mb = t % mb_cap)` outright: the walk
/// runs once, stops at the real `num_links`, and needs ZERO barriers —
/// 32 multibodies per warp, same operation order per DOF (ascending `k`,
/// one contribution each) → bit-identical forces.
///
/// Dispatch: `[num_batches · multibodies_batch_capacity, 1, 1]` threads
/// (like `gpu_mb_ltdl_lanes`, which this precedes).
#[cfg(feature = "dim3")]
#[spirv_bindgen]
#[spirv(compute(threads(32, 1, 1)))]
pub fn gpu_mb_gravity_rhs_lanes(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    links_workspace: &mut [MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    links_local_mprops: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] body_jacobians: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] gen_forces: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] lu_pivots: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] dof_state: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 9)] gravity: &Vec4,
    #[spirv(uniform, descriptor_set = 0, binding = 10)] batch_ids: &BatchIndices,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 11)] dof_frictionloss: &[f32],
    // See `gpu_mb_gravity_rhs` for the delay-state layout / tick-bump rules.
    #[spirv(storage_buffer, descriptor_set = 0, binding = 12)] motor_delay_state: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 13)] num_batches_u: &u32,
) {
    let mb_cap = batch_ids.multibodies_batch_capacity;
    let t = wg_id.x * LANES + lid.x;
    let total = *num_batches_u * mb_cap;
    if t >= total {
        return;
    }
    let batch_id = t / mb_cap;
    let mb_idx = t % mb_cap;
    if mb_idx >= num_multibodies.read(batch_id as usize) {
        return;
    }
    let mb = batch_ids
        .mb_batch(batch_id, multibody_info)
        .read(mb_idx as usize);
    let num_links = mb.num_links;
    let ndofs = mb.ndofs;
    if ndofs == 0 {
        return;
    }
    let mb_jac_base = batch_ids.jac_start(batch_id) + mb.jacobian_offset as usize;
    let gen_base = batch_ids.dof_start(batch_id) + mb.first_dof as usize;
    let piv_offset = gen_base;

    let stat_slice = batch_ids
        .mb_links_batch(batch_id, links_static)
        .offset(mb.first_link as usize);
    let mut ws_slice = batch_ids
        .mb_links_batch_mut(batch_id, links_workspace)
        .offset(mb.first_link as usize);
    let local_mprops_slice = batch_ids
        .mb_links_batch(batch_id, links_local_mprops)
        .offset(mb.first_link as usize);
    let vel_slice = Slice(dof_state, gen_base);
    let damping_slice = Slice(
        dof_state,
        batch_ids.dof_damping_section_offset as usize + gen_base,
    );
    let frictionloss_slice = Slice(dof_frictionloss, gen_base);

    // ---- Phase 1: zero the generalized-force vector. ----
    for i in 0..ndofs {
        gen_forces.write(gen_base + i as usize, 0.0);
    }

    let g = Vec3::new(gravity.x, gravity.y, gravity.z);

    // ---- Phase 2: per-link gravity / Coriolis-force assembly (serial —
    // the walk is parent→child dependent anyway; parallelism comes from
    // 32 multibodies per warp). Same math and per-DOF accumulation order
    // as `gpu_mb_gravity_rhs`.
    for k in 0..num_links {
        let (
            self_joint_vel_lin,
            self_joint_vel_ang,
            self_shift02,
            self_shift23,
            self_rb_ang,
        ) = {
            let ws = &ws_slice[k as usize];
            (
                ws.joint_velocity.linear,
                ws.joint_velocity.angular,
                ws.shift02,
                ws.shift23,
                ws.rb_vels.angular,
            )
        };

        let mut acc_lin = Vector::ZERO;
        let mut acc_ang: AngVector = AngVector::ZERO;
        if k != 0 {
            let stat = stat_slice[k as usize];
            let parent_ws = &ws_slice[stat.parent_link_id as usize];
            let parent_acc_lin = parent_ws.kinematic_acc.linear;
            let parent_acc_ang = parent_ws.kinematic_acc.angular;
            let parent_ang = parent_ws.rb_vels.angular;

            acc_lin = parent_acc_lin;
            acc_ang = parent_acc_ang;

            acc_lin += gcross_av(parent_ang, self_joint_vel_lin) * 2.0;
            acc_ang += parent_ang.cross(self_joint_vel_ang);
            acc_lin += gcross_av(parent_ang, gcross_av(parent_ang, self_shift02));
            acc_lin += gcross_av(parent_acc_ang, self_shift02);
        }
        let rb_ang = self_rb_ang;
        acc_lin += gcross_av(rb_ang, gcross_av(rb_ang, self_shift23));
        acc_lin += gcross_av(acc_ang, self_shift23);

        ws_slice[k as usize].kinematic_acc = Velocity::new(acc_lin, acc_ang);

        let lmp = local_mprops_slice[k as usize];
        let inv_mass_x = lmp.inv_mass.x;
        if inv_mass_x != 0.0 {
            let mass = 1.0 / inv_mass_x;
            let rb_inertia = link_world_inertia(&ws_slice[k as usize], &lmp);

            let gyroscopic = {
                let i_omega = rb_inertia * rb_ang;
                rb_ang.cross(i_omega)
            };
            let i_acc_ang = rb_inertia * acc_ang;

            let f_lin = (g - acc_lin) * mass;
            let f_ang = -gyroscopic - i_acc_ang;

            // Chain-sparse Jᵀ·f scatter: iterate the stored columns (set
            // bits of the chain mask, ascending). Same dot expression and
            // `cur + s` accumulation as `gemv_tr_spatial_split_sparse_par`.
            let stat = stat_slice[k as usize];
            let jac = MatSlice::dense(
                mb_jac_base + stat.jac_offset as usize,
                SPATIAL_DIM as u32,
                stat.jac_chain_mask.count_ones(),
            );
            let mut m = stat.jac_chain_mask;
            for sc in 0..32u32 {
                if m == 0 {
                    break;
                }
                let d = m.trailing_zeros();
                m &= m - 1;
                let s = body_jacobians.read(jac.idx(0, sc)) * f_lin.x
                    + body_jacobians.read(jac.idx(1, sc)) * f_lin.y
                    + body_jacobians.read(jac.idx(2, sc)) * f_lin.z
                    + body_jacobians.read(jac.idx(3, sc)) * f_ang.x
                    + body_jacobians.read(jac.idx(4, sc)) * f_ang.y
                    + body_jacobians.read(jac.idx(5, sc)) * f_ang.z;
                let idx = gen_base + d as usize;
                let cur = gen_forces.read(idx);
                gen_forces.write(idx, 1.0 * cur + 1.0 * s);
            }
        }
    }

    // ---- Damping + Coulomb joint friction (see `gpu_mb_gravity_rhs`). ----
    for i in 0..ndofs {
        let idx = gen_base + i as usize;
        let cur = gen_forces.read(idx);
        let v = vel_slice[i as usize];
        let coulomb = frictionloss_slice[i as usize] * (v / 1.0).clamp(-1.0, 1.0);
        gen_forces.write(idx, cur - damping_slice[i as usize] * v - coulomb);
    }

    // ---- Explicit force-based motor PD torque (see `gpu_mb_gravity_rhs`;
    // identical loop, previously serial on lane 0 with 31 idle lanes). ----
    {
        let delay_stride = 2 + batch_ids.links_batch_capacity as usize;
        let delay_base = batch_id as usize * delay_stride;
        let tick = motor_delay_state.read(delay_base);
        let delay_k = motor_delay_state.read(delay_base + 1);
        let use_prev = tick < delay_k;
        for k in 0..num_links {
            let stat = stat_slice[k as usize];
            if stat.kinematic != 0 {
                continue;
            }
            let locked = stat.data.locked_axes;
            let motor_axes = stat.data.motor_axes & !locked;
            if motor_axes == 0 {
                continue;
            }
            let ws = &ws_slice[k as usize];
            let mut curr_free_dof = 0u32;
            for axis in 0..(SPATIAL_DIM as u32) {
                if (locked & (1 << axis)) != 0 {
                    continue;
                }
                if (motor_axes & (1 << axis)) != 0 {
                    let motor = stat.data.motors[axis as usize];
                    if motor.model == crate::dynamics::joint::FORCE_BASED {
                        let q = ws.coords.read(axis as usize);
                        let abs_dof = stat.assembly_id + curr_free_dof;
                        let v = vel_slice[abs_dof as usize];
                        let target = if use_prev {
                            motor_delay_state
                                .read(delay_base + 2 + (mb.first_link + k) as usize)
                        } else {
                            motor.target_pos
                        };
                        let tau = (motor.stiffness * (target - q)
                            - motor.damping * v)
                            .clamp(-motor.max_force, motor.max_force);
                        let idx = gen_base + abs_dof as usize;
                        gen_forces.write(idx, gen_forces.read(idx) + tau);
                    }
                }
                curr_free_dof += 1;
            }
        }
        // Tick bump: one writer per batch (multibody 0), matching the
        // warp version's `mb_idx == 0` rule.
        if mb_idx == 0 {
            motor_delay_state.write(delay_base, tick + 1.0);
        }
    }

    // ---- Per-DOF parent array (see `gpu_mb_gravity_rhs`). ----
    for k in 0..num_links {
        let stat = stat_slice[k as usize];
        if stat.ndofs > 0 {
            let mut p = NO_PARENT;
            if k != 0 {
                let mut a = stat.parent_link_id;
                for _ in 0..MAX_MB_DOFS as u32 {
                    let s = stat_slice[a as usize];
                    if s.ndofs > 0 {
                        p = s.assembly_id + s.ndofs - 1;
                        break;
                    }
                    if a == 0 {
                        break;
                    }
                    a = s.parent_link_id;
                }
            }
            lu_pivots.write(piv_offset + stat.assembly_id as usize, p);
            for t in 1..stat.ndofs {
                lu_pivots.write(
                    piv_offset + (stat.assembly_id + t) as usize,
                    stat.assembly_id + t - 1,
                );
            }
        }
    }
}

/// Tree-sparse LᵀDL factor + gravity solve, ONE MULTIBODY PER LANE.
///
/// The serial factor/solve loops from `gpu_mb_gravity_and_lu` run on lane 0
/// with 31 lanes idle — fine when the dense math needed the other lanes,
/// pure waste for the barrier-free sparse path. Here thread
/// `t = wg.x·32 + lane` owns multibody `(batch = t / mb_cap,
/// mb = t % mb_cap)` and factors/solves in GLOBAL memory: 32 multibodies
/// per warp, zero barriers, zero shared memory. Same operation order as the
/// in-shared path → bit-identical factors and accelerations.
///
/// Dispatch: `[num_batches · multibodies_batch_capacity, 1, 1]` threads.
/// Requires `gpu_mb_gravity_rhs` to have run (rhs in `gen_forces`, parent
/// array in `lu_pivots`).
#[spirv_bindgen]
#[spirv(compute(threads(32, 1, 1)))]
pub fn gpu_mb_ltdl_lanes(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mass_matrices: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] gen_forces: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] lu_pivots: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] num_batches_u: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] batch_ids: &BatchIndices,
) {
    let mb_cap = batch_ids.multibodies_batch_capacity;
    let t = wg_id.x * LANES + lid.x;
    let total = *num_batches_u * mb_cap;
    if t >= total {
        return;
    }
    let batch_id = t / mb_cap;
    let mb_idx = t % mb_cap;
    if mb_idx >= num_multibodies.read(batch_id as usize) {
        return;
    }
    let mb = batch_ids
        .mb_batch(batch_id, multibody_info)
        .read(mb_idx as usize);
    let ndofs = mb.ndofs;
    if ndofs == 0 {
        return;
    }
    let gen_base = batch_ids.dof_start(batch_id) + mb.first_dof as usize;
    let mb_mm_base = batch_ids.mm_start(batch_id) + mb.mass_matrix_offset as usize;
    let m_view = MatSlice::dense(mb_mm_base, ndofs, ndofs);

    if ndofs <= 32 {
        // Factor + solve with the workspace in per-thread LOCAL memory
        // instead of read-modify-writing global storage: the elimination's
        // ~4k dependent RMW steps each cost an L2 round-trip in global (GPU
        // L1 is write-evict for global stores — the finalize-kernel lesson,
        // 2.36×), while local memory is L1-backed AND, because every lane
        // walks the SAME tree topology in lockstep, the interleaved
        // per-thread layout coalesces each step's 32 lane accesses into one
        // transaction. Copy-in/write-back are dependence-free streaming.
        // Same operation order as `ltdl_factor_global`/`ltdl_solve_global`
        // → bit-identical factors and accelerations.
        let n = ndofs;
        let n2 = (n * n) as usize;
        let mut mloc = [0.0f32; 1024];
        for f in 0..n2 {
            mloc[f] = mass_matrices.read(mb_mm_base + f);
        }
        let mut par = [NO_PARENT; 32];
        for i in 0..n {
            par[i as usize] = lu_pivots.read(gen_base + i as usize);
        }

        // ---- Factor M = Lᵀ·D·L (elimination order of `ltdl_factor_global`).
        for step in 0..n {
            let k = n - 1 - step;
            let d = mloc[(k * n + k) as usize];
            let inv_d = if d != 0.0 { 1.0 / d } else { 0.0 };
            let mut i = par[k as usize];
            // Bounded loops (parents strictly decrease) so corrupt data
            // can't hang the GPU.
            for _ in 0..32u32 {
                if i == NO_PARENT {
                    break;
                }
                let a = mloc[(i * n + k) as usize] * inv_d;
                let mut j = i;
                for _ in 0..32u32 {
                    if j == NO_PARENT {
                        break;
                    }
                    let v = mloc[(j * n + i) as usize] - a * mloc[(j * n + k) as usize];
                    mloc[(j * n + i) as usize] = v;
                    j = par[j as usize];
                }
                mloc[(i * n + k) as usize] = a;
                i = par[i as usize];
            }
        }

        // Persist the factors (constraint init / finalize reuse them).
        for f in 0..n2 {
            mass_matrices.write(mb_mm_base + f, mloc[f]);
        }

        // ---- Solve M·x = τ with the rhs in registers/local.
        let mut x = [0.0f32; 32];
        for i in 0..n {
            x[i as usize] = gen_forces.read(gen_base + i as usize);
        }
        // Lᵀ·z = b: scatter descending.
        for step in 0..n {
            let i = n - 1 - step;
            let xi = x[i as usize];
            let mut j = par[i as usize];
            for _ in 0..32u32 {
                if j == NO_PARENT {
                    break;
                }
                x[j as usize] -= mloc[(j * n + i) as usize] * xi;
                j = par[j as usize];
            }
        }
        // z = D⁻¹·z.
        for i in 0..n {
            let d = mloc[(i * n + i) as usize];
            x[i as usize] = if d != 0.0 { x[i as usize] / d } else { 0.0 };
        }
        // L·x = z: gather ascending.
        for i in 0..n {
            let mut s = x[i as usize];
            let mut j = par[i as usize];
            for _ in 0..32u32 {
                if j == NO_PARENT {
                    break;
                }
                s -= mloc[(j * n + i) as usize] * x[j as usize];
                j = par[j as usize];
            }
            x[i as usize] = s;
        }
        for i in 0..n {
            gen_forces.write(gen_base + i as usize, x[i as usize]);
        }
    } else {
        // Oversized fallback (ndofs > 32 can't happen with the warp-32
        // kernels, but keep the global path total).
        super::lu::ltdl_factor_global(mass_matrices, m_view, ndofs, lu_pivots, gen_base);
        super::lu::ltdl_solve_global(
            mass_matrices,
            m_view,
            ndofs,
            lu_pivots,
            gen_base,
            gen_forces,
            gen_base,
        );
    }
}
