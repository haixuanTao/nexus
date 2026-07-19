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
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

/// Lanes per workgroup for the cooperative contact-constraint kernels.
pub const MB_CONTACT_SOLVE_LANES: u32 = 32;

use crate::dynamics::body::{Velocity, WorldMassProperties};
use crate::dynamics::joint::SPATIAL_DIM;
use crate::dynamics::sim_params::SimParams;
use crate::queries::IndexedManifold;
use crate::utils::BatchIndices;
use crate::utils::linalg::MatSlice;
use crate::{ANG_DIM, AngVector, DIM, Pose, Vector, gcross, gdot};

use super::types::{
    CONTACT_CONSTRAINTS_PER_POINT, MAX_MB_CONTACT_CONSTRAINTS_PER_MB, MAX_MB_CONTACTS_PER_MB,
    MB_CONTACT_KIND_NORMAL, MB_CONTACT_KIND_TANGENT, MultibodyContactConstraint, MultibodyInfo,
};

#[cfg(feature = "dim2")]
use glamx::Vec2;
#[cfg(feature = "dim3")]
use glamx::Vec3;

/// Default Coulomb friction coefficient — the host's `MultibodyInfo::friction`
/// fallback when no collider friction is found. Contacts now read
/// `mb.friction` (per-env, from the rapier collider material); this constant is
/// retained for reference and as the documented default. The rigid-body path in
/// `solver_utils::contact_to_constraint` still hardcodes 0.5 (separate TODO).
#[allow(dead_code)]
const FRICTION_DEFAULT: f32 = 0.5;

/// Angular-friction length scale (metres). The torsional/rolling friction limit
/// at a contact is `mb.friction * ANG_FRICTION_LEN * normal_impulse` — i.e. the
/// effective lever a real contact PATCH would provide. Gives a single contact
/// point a rotational constraint so a loaded foot resists pivoting/screwing
/// about it (the sim-to-real foot "sliding-rotating"). ~half the foot footprint.
/// Tunable: too small → foot still pivots; too large → foot can't articulate.
const ANG_FRICTION_LEN: f32 = 0.0;

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
    jac_meta: &[[u32; 2]],
    links_base: usize,
    mb_jac_base: usize,
    ndofs: u32,
    link_id: u32,
    unit_force: Vector,
    unit_torque: AngVector,
    out_jacs: &mut [f32],
    col_offset: usize,
    // Distance between consecutive DOF elements of one row: 1 for the
    // env-major (AoS) layout, `num_batches` for the batch-interleaved SoA
    // layout (`NEXUS_SOA_CONTACTS=1`).
    elem_stride: usize,
    accumulate: bool,
) {
    // 3D: the link's jacobian is stored chain-sparse (`6 × chain_len`,
    // columns = ascending set bits of the chain mask, metadata in the 8-byte
    // `jac_meta` record — NOT `links_static`, whose embedded `GenericJoint`
    // makes a by-value load per constraint row cost more than the sparse
    // reads save). Zero the row up front (skipped when accumulating into an
    // already-written row), then add only the stored chain columns.
    #[cfg(feature = "dim3")]
    {
        let meta = jac_meta.read(links_base + link_id as usize);
        let mask = meta[1];
        let link_j = MatSlice::dense(
            mb_jac_base + meta[0] as usize,
            SPATIAL_DIM as u32,
            mask.count_ones(),
        );
        let (link_j_v, link_j_w) = link_j.rows_range_pair(0, DIM, DIM, ANG_DIM);
        if !accumulate {
            for j in 0..ndofs {
                out_jacs.write(col_offset + (j as usize) * elem_stride, 0.0);
            }
        }
        // Set-bit iteration: `sc` is the stored column, `j` its global DOF.
        // Bounded `for` + `break` (structured control flow — see the
        // rust-gpu `while` note in `utils::linalg`).
        let mut m = mask;
        for sc in 0..32u32 {
            if m == 0 {
                break;
            }
            let j = m.trailing_zeros();
            m &= m - 1;
            let jv0 = body_jacobians.read(link_j_v.idx(0, sc));
            let jv1 = body_jacobians.read(link_j_v.idx(1, sc));
            let jv2 = body_jacobians.read(link_j_v.idx(2, sc));
            let jw0 = body_jacobians.read(link_j_w.idx(0, sc));
            let jw1 = body_jacobians.read(link_j_w.idx(1, sc));
            let jw2 = body_jacobians.read(link_j_w.idx(2, sc));
            let dot = unit_force.x * jv0
                + unit_force.y * jv1
                + unit_force.z * jv2
                + unit_torque.x * jw0
                + unit_torque.y * jw1
                + unit_torque.z * jw2;
            let idx = col_offset + (j as usize) * elem_stride;
            out_jacs.write(idx, out_jacs.read(idx) + dot);
        }
    }
    // 2D keeps the dense per-link layout (no GPU multibody host path in 2D).
    #[cfg(feature = "dim2")]
    {
        let _ = (jac_meta, links_base);
        let link_jac_base = mb_jac_base + (link_id as usize) * SPATIAL_DIM * (ndofs as usize);
        let link_j = MatSlice::dense(link_jac_base, SPATIAL_DIM as u32, ndofs);
        let (link_j_v, link_j_w) = link_j.rows_range_pair(0, DIM, DIM, ANG_DIM);
        for j in 0..ndofs {
            let jv0 = body_jacobians.read(link_j_v.idx(0, j));
            let jv1 = body_jacobians.read(link_j_v.idx(1, j));
            let jw0 = body_jacobians.read(link_j_w.idx(0, j));
            let dot = unit_force.x * jv0 + unit_force.y * jv1 + unit_torque * jw0;
            let prev = if accumulate {
                out_jacs.read(col_offset + (j as usize) * elem_stride)
            } else {
                0.0
            };
            out_jacs.write(col_offset + (j as usize) * elem_stride, prev + dot);
        }
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
/// Body of `gpu_mb_init_contact_constraints`, shared by the classic
/// env-major entry and the `NEXUS_SOA_CONTACTS=1` batch-interleaved entry.
/// Layout parameters: a jac row element `(slot s, dof j)` lives at
/// `col_base + s·slot_stride + j·elem_stride`.
///   - env-major: `col_base = col_start(b) + mb·MAXC·ds`, `slot_stride = ds`,
///     `elem_stride = 1`.
///   - SoA:       `col_base = (mb·MAXC·ds)·NB + b`, `slot_stride = ds·NB`,
///     `elem_stride = NB`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn init_contact_constraints_body(
    batch_id: u32,
    mb_idx: u32,
    dt: f32,
    col_base: usize,
    slot_stride: usize,
    elem_stride: usize,
    multibody_info: &[MultibodyInfo],
    body_jacobians: &[f32],
    jac_meta: &[[u32; 2]],
    body_to_link: &[[u32; 2]],
    contact_constraints: &mut [MultibodyContactConstraint],
    contact_constraint_jacs: &mut [f32],
    contact_constraint_count: &mut [u32],
    mprops: &[WorldMassProperties],
    poses: &[Pose],
    contacts: &[IndexedManifold],
    contacts_len: &[u32],
    all_params: &[SimParams],
    batch_ids: &BatchIndices,
) {
    let inv_dt = if dt != 0.0 { 1.0 / dt } else { 0.0 };
    // Damped Baumgarte position correction from this env's `SimParams`, matching
    // the rigid-body path's `contact_erp_inv_dt` = ω / (dt·ω + 2ζ). Replaces the
    // old `erp_inv_dt = inv_dt` (ERP=1.0, which tried to undo ALL penetration in
    // one step → a penetrating foot got a separating bias up to
    // `max_corr_velocity` and LAUNCHED the articulation; energy injection that
    // worsened with more solver iterations). Reading SimParams here also makes
    // per-env `contact_natural_frequency` / `contact_damping_ratio` domain
    // randomization actually reach the multibody contact solver.
    // Damped Baumgarte position correction (mirrors the rigid-body path's
    // `contact_erp_inv_dt`): erp_inv_dt = ω / (dt·ω + 2ζ). NOT `erp_inv_dt =
    // inv_dt` (ERP=1.0) — that undoes ALL penetration in one step, so a
    // penetrating foot got a separating bias up to `max_corr_velocity` and
    // LAUNCHED the articulation (energy injection that worsened with more solver
    // iterations). Now reads per-env `contact_natural_frequency` /
    // `contact_damping_ratio` from this batch's `SimParams` (set 1, binding 4 —
    // the same per-env tensor the free-body solver reads via `all_params.at`),
    // so contact-stiffness domain randomization reaches the multibody contact
    // solver. `dt` still comes from the dedicated `dt_uniform` (the substep dt),
    // NOT from `params.dt`, so with DR off (all envs at the 30/5 default) the
    // erp/cfm are bit-identical to the previous hardcoded path.
    let params = all_params.at(batch_id as usize);
    let contact_natural_frequency = params.contact_natural_frequency;
    let contact_damping_ratio = params.contact_damping_ratio;
    let ang_freq = contact_natural_frequency * core::f32::consts::TAU;
    let erp_inv_dt = ang_freq / (dt * ang_freq + 2.0 * contact_damping_ratio);
    // Contact CFM (constraint-force-mixing) coefficient — SOFT/compliant normal
    // contact. The multibody contact path was fully HARD (cfm=0), so the first
    // PGS-solved point of the foot's manifold absorbed the whole load and the
    // others stayed at ~0 (single load-bearing point → the foot rocks). A soft
    // normal makes each point's force ∝ its penetration, so the box's manifold
    // points SHARE load and the foot settles flat (a patch) — what the rigid-body
    // path already does (sim_params::contact_cfm_factor) and what MuJoCo's
    // solref/solimp gives. The solve divides the impulse update by (1+cfm_coeff).
    // BIPED_CONTACT_CFM_SCALE (default 1) tunes the softness up for stronger
    // load-sharing without a recompile path change.
    let contact_erp = dt * erp_inv_dt;
    let contact_cfm_coeff = if contact_erp > 0.0 {
        let ie = 1.0 / contact_erp - 1.0;
        ie * ie / ((1.0 + ie) * 4.0 * contact_damping_ratio * contact_damping_ratio)
    } else {
        0.0
    };
    let allowed_lin_err = 0.001f32;
    let max_corr_velocity = 10.0f32;

    let mb_start = batch_ids.mb_start(batch_id);
    let cons_start = batch_ids.mb_contact_constraints_start(batch_id);
    let col_start = batch_ids.mb_contact_constraint_columns_start(batch_id);
    let colliders_start = batch_ids.coll_start(batch_id);
    // `body_to_link` is laid out with stride = colliders_batch_capacity.
    let b2l_start = colliders_start;

    // Per-multibody early-out: padding multibody slots have `ndofs == 0`,
    // which we use here as the sentinel (replaces the `num_multibodies`
    // storage binding the kernel used to read).
    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let ndofs = mb.ndofs;
    if ndofs == 0 {
        contact_constraint_count.write(mb_start + mb_idx as usize, 0);
        return;
    }
    let mb_jac_base = batch_ids.jac_start(batch_id) + mb.jacobian_offset as usize;
    // This multibody's links in the per-batch static-links slab (chain-sparse
    // jacobian metadata lives in `MultibodyLinkStatic`).
    let links_base = batch_ids.links_start(batch_id) + mb.first_link as usize;
    let cons_base = cons_start + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize);
    // Each constraint slot reserves `dof_batch_capacity` floats in the
    // column buffer (matches the allocation in `from_rapier` and avoids any
    // overlap between multibodies of differing `ndofs`).
    let dofs_stride = batch_ids.dof_batch_capacity as usize;

    let contacts_slice = batch_ids.contact_batch(batch_id, contacts);
    // Clamp to the allocated capacity: the narrow-phase atomic over-counts past
        // capacity (writes are skipped beyond it), so iterating to the raw
        // count would read unwritten/out-of-bounds slots. WebGPU clamps such
        // reads; native CUDA (unsafe_remove_boundchecks) would fault.
        let n_contacts = contacts_len.read(batch_id as usize).min(batch_ids.contacts_batch_capacity);
    let mut count = 0u32;

    for ci in 0..n_contacts {
        if count >= MAX_MB_CONTACTS_PER_MB {
            break;
        }
        let im = contacts_slice[ci as usize];
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
        let pose2 = poses.read(colliders_start + id2 as usize);
        let world_normal = pose1.rotation * im.contact.normal_a;
        let lin_jac = if is_self || mb_on_1 {
            world_normal
        } else {
            -world_normal
        };
        let mb_normal = -lin_jac;

        let free_mp = if is_self {
            WorldMassProperties::default()
        } else {
            mprops.read(colliders_start + free_body_id as usize)
        };
        let free_im = if is_self { 0.0 } else { free_mp.inv_mass.x };

        // Multibody-link origins come from the collider poses buffer instead
        // of `links_workspace` (which we no longer bind in this kernel — see
        // binding-count cap discussion). For each link, the body that holds
        // it has the same world pose as the link itself (the FK pass writes
        // both). `mb_link_id_a` always corresponds to body `id1` when
        // `mb_on_1 || is_self`, and to `id2` otherwise. `mb_link_id_b` is
        // only used in the self-contact case where it corresponds to `id2`.
        let link_origin_a = if is_self || mb_on_1 {
            pose1.translation
        } else {
            pose2.translation
        };
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

            let rhs_bias = (erp_inv_dt * (dist + allowed_lin_err)).clamp(-max_corr_velocity, 0.0);
            let rhs_wo_bias = if dist > 0.0 { dist * inv_dt } else { 0.0 };

            let normal_slot = count;
            let normal_col_offset = col_base + (normal_slot as usize) * slot_stride;

            fill_contact_jac_row(
                body_jacobians,
                jac_meta,
                links_base,
                mb_jac_base,
                ndofs,
                mb_link_id_a,
                mb_normal,
                torque_a_normal,
                contact_constraint_jacs,
                normal_col_offset,
                elem_stride,
                false,
            );

            // B-side fold-in for self-contacts, free body for the rest. The
            // ang_jac fields below describe the FREE body side; for self
            // contacts they collapse to zero because both sides are folded
            // into `J_mb` already.
            let (ang_jac_normal, ii_ang_jac_normal) = if is_self {
                // Self-contact: B-side link is the collider at `id2`, so its
                // world pose is `pose2` (already loaded). Avoids a
                // `links_workspace` binding.
                let link_origin_b = pose2.translation;
                let _ = link_origin_b_default;
                let shift_b = pt_world - link_origin_b;
                let torque_b_normal = gcross(shift_b, lin_jac);
                fill_contact_jac_row(
                    body_jacobians,
                    jac_meta,
                    links_base,
                    mb_jac_base,
                    ndofs,
                    mb_link_id_b,
                    lin_jac,
                    torque_b_normal,
                    contact_constraint_jacs,
                    normal_col_offset,
                    elem_stride,
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
                friction_coeff: mb.friction,
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
                // SOFT normal contact (load-sharing across the foot's manifold).
                cfm_coeff: contact_cfm_coeff,
                cfm_gain: 0.0,
                // DEBUG: stash the contact-point world XY so the host per-substep
                // trace can see WHICH point is load-bearing each substep — testing
                // the "single contact point dances between candidate points →
                // ratchets the foot forward" hypothesis. [foot-pivot trace]
                _pad4: [pt_world.x.to_bits(), pt_world.y.to_bits()],
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
                friction_coeff: mb.friction,
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
            for tang_idx in 0..(DIM - 1) {
                let mb_tangent = if tang_idx == 0 {
                    mb_tangent0
                } else {
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
                let torque_a_tang = if tang_idx == 0 {
                    torque_a_t0
                } else {
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
                let tang_col_offset = col_base + (tang_slot as usize) * slot_stride;

                fill_contact_jac_row(
                    body_jacobians,
                    jac_meta,
                    links_base,
                    mb_jac_base,
                    ndofs,
                    mb_link_id_a,
                    mb_tangent,
                    torque_a_tang,
                    contact_constraint_jacs,
                    tang_col_offset,
                    elem_stride,
                    false,
                );

                let (ang_jac_tang, ii_ang_jac_tang) = if is_self {
                    let link_origin_b = pose2.translation;
                    let shift_b = pt_world - link_origin_b;
                    let torque_b_tang = gcross(shift_b, free_tangent);
                    fill_contact_jac_row(
                        body_jacobians,
                        jac_meta,
                        links_base,
                        mb_jac_base,
                        ndofs,
                        mb_link_id_b,
                        free_tangent,
                        torque_b_tang,
                        contact_constraint_jacs,
                        tang_col_offset,
                        elem_stride,
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
                    friction_coeff: mb.friction,
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
                    friction_coeff: mb.friction,
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

            // Angular friction (torsional about the contact normal + rolling
            // about each tangent — the MuJoCo condim=6 analog). A single contact
            // point otherwise gives NO rotational constraint, so a loaded foot
            // PIVOTS/screws about it (the sim-to-real foot "sliding-rotating":
            // contact point sticks, foot rotates about it, origin sweeps). Each
            // constraint resists the link's angular velocity about its axis
            // (pure-angular jacobian: zero linear force, torque = axis), clamped
            // (kind=TANGENT) to ±(mb.friction·ANG_FRICTION_LEN)·normal_impulse.
            // Slots are already reserved by the per-point capacity check above
            // (CONTACT_CONSTRAINTS_PER_POINT = 1 normal + (DIM-1) linear + ANG_DIM).
            #[cfg(feature = "dim3")]
            {
                let ang_axes = [mb_normal, mb_tangent0, mb_tangent1];
                let ang_mu = mb.friction * ANG_FRICTION_LEN;
                for ax_i in 0..ANG_DIM {
                    let ax = ang_axes[ax_i as usize];
                    let ang_slot = count;
                    let ang_col_offset = col_base + (ang_slot as usize) * slot_stride;
                    fill_contact_jac_row(
                        body_jacobians,
                        jac_meta,
                        links_base,
                        mb_jac_base,
                        ndofs,
                        mb_link_id_a,
                        Vector::ZERO,
                        ax,
                        contact_constraint_jacs,
                        ang_col_offset,
                        elem_stride,
                        false,
                    );
                    let (afj, iiafj) = if is_self {
                        fill_contact_jac_row(
                            body_jacobians,
                            jac_meta,
                            links_base,
                            mb_jac_base,
                            ndofs,
                            mb_link_id_b,
                            Vector::ZERO,
                            -ax,
                            contact_constraint_jacs,
                            ang_col_offset,
                            elem_stride,
                            true,
                        );
                        (AngVector::ZERO, AngVector::ZERO)
                    } else {
                        let aj = -ax;
                        (aj, free_mp.inv_inertia_mul(aj))
                    };
                    let ang_cons = MultibodyContactConstraint {
                        multibody_id: mb_idx,
                        link_id: mb_link_id_a,
                        kind: MB_CONTACT_KIND_TANGENT,
                        free_body_id,
                        free_body_im: free_im,
                        friction_coeff: ang_mu,
                        normal_constraint_slot: normal_slot,
                        _pad0: 0,
                        lin_jac: Vector::ZERO,
                        _pad1: 0,
                        ang_jac: afj,
                        _pad2: 0,
                        ii_ang_jac: iiafj,
                        _pad3: 0,
                        inv_lhs: 0.0,
                        rhs: 0.0,
                        rhs_wo_bias: 0.0,
                        impulse: 0.0,
                        cfm_coeff: 0.0,
                        cfm_gain: 0.0,
                        _pad4: [0; 2],
                    };
                    contact_constraints.write(cons_base + ang_slot as usize, ang_cons);
                    count += 1;
                }
            }
        }
    }

    // The solve / finalize / remove-bias kernels iterate `0..count` so we
    // don't need to mark surplus slots inactive — they're never read.
    contact_constraint_count.write(mb_start + mb_idx as usize, count);
}

#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_init_contact_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] body_jacobians: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] jac_meta: &[[u32; 2]],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] body_to_link: &[[u32; 2]],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)]
    contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] contact_constraint_jacs: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)]
    contact_constraint_count: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 7)] dt_uniform: &f32,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] mprops: &[WorldMassProperties],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 2)] contacts: &[IndexedManifold],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 3)] contacts_len: &[u32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 4)] all_params: &[SimParams],
    #[spirv(uniform, descriptor_set = 0, binding = 8)] batch_ids: &BatchIndices,
) {
    let batch_id = invocation_id.y;
    let mb_idx = invocation_id.x;
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let col_start = batch_ids.mb_contact_constraint_columns_start(batch_id);
    let col_base =
        col_start + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize) * dofs_stride;
    init_contact_constraints_body(
        batch_id,
        mb_idx,
        *dt_uniform,
        col_base,
        dofs_stride,
        1,
        multibody_info,
        body_jacobians,
        jac_meta,
        body_to_link,
        contact_constraints,
        contact_constraint_jacs,
        contact_constraint_count,
        mprops,
        poses,
        contacts,
        contacts_len,
        all_params,
        batch_ids,
    );
}

/// `NEXUS_SOA_CONTACTS=1` entry: identical scan, but jac rows are written in
/// the batch-interleaved SoA layout (`elem·NB + batch`), so the SoA
/// finalize/solve kernels read them fully coalesced.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_init_contact_constraints_soa(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] body_jacobians: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] jac_meta: &[[u32; 2]],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] body_to_link: &[[u32; 2]],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)]
    contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] contact_constraint_jacs: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)]
    contact_constraint_count: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 7)] dt_uniform: &f32,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] mprops: &[WorldMassProperties],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 2)] contacts: &[IndexedManifold],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 3)] contacts_len: &[u32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 4)] all_params: &[SimParams],
    #[spirv(uniform, descriptor_set = 0, binding = 9)] batch_ids: &BatchIndices,
    #[spirv(uniform, descriptor_set = 0, binding = 10)] num_batches_u: &u32,
) {
    let batch_id = invocation_id.y;
    let mb_idx = invocation_id.x;
    let nb = *num_batches_u as usize;
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let col_base = ((mb_idx as usize)
        * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize)
        * dofs_stride)
        * nb
        + batch_id as usize;
    init_contact_constraints_body(
        batch_id,
        mb_idx,
        *dt_uniform,
        col_base,
        dofs_stride * nb,
        nb,
        multibody_info,
        body_jacobians,
        jac_meta,
        body_to_link,
        contact_constraints,
        contact_constraint_jacs,
        contact_constraint_count,
        mprops,
        poses,
        contacts,
        contacts_len,
        all_params,
        batch_ids,
    );
}

/// Pass 2: for each emitted constraint, LU back-solve `M · column = Jᵀ`
/// (the row produced by the init kernel) and set `inv_lhs = 1 / (Jᵀ ·
/// column + free_body_inv_r)`.
/// Lanes per workgroup for the cooperative `*_finalize_contact_constraints` /
/// per-constraint kernels: one workgroup per articulation, the `count`
/// independent constraint back-solves are split across these lanes (grid-stride,
/// no barriers — each constraint writes its own disjoint column + slot).
pub const MB_CONTACT_FINALIZE_LANES: u32 = 32;

#[spirv_bindgen]
#[spirv(compute(threads(32)))]
pub fn gpu_mb_finalize_contact_constraints(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mass_matrices: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] lu_pivots: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] contact_constraint_jacs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)]
    contact_constraint_columns: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] contact_constraint_count: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 8)] batch_ids: &BatchIndices,
) {
    let batch_id = wg_id.y;
    let mb_idx = wg_id.x;
    let lane = lid.x;
    let num_mb = num_multibodies.read(batch_id as usize);
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_ids.mb_start(batch_id);
    let cons_start = batch_ids.mb_contact_constraints_start(batch_id);
    let col_start = batch_ids.mb_contact_constraint_columns_start(batch_id);

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let ndofs = mb.ndofs;
    if ndofs == 0 {
        return;
    }
    let mb_mm_base = batch_ids.mm_start(batch_id) + mb.mass_matrix_offset as usize;
    let piv_offset = batch_ids.dof_start(batch_id) + mb.first_dof as usize;
    let cons_base = cons_start + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize);
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let col_base =
        col_start + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize) * dofs_stride;

    let m = MatSlice::dense(mb_mm_base, ndofs, ndofs);
    let count = contact_constraint_count.read(mb_start + mb_idx as usize);

    // Each constraint's back-solve is independent (disjoint column + slot, only
    // read-only shared M/jacs), so split the slot loop across the workgroup's
    // lanes with NO barriers — lane `lane` owns constraints
    // `lane, lane+LANES, …`.
    let mut s = lane;
    while s < count {
        let col_offset = col_base + (s as usize) * dofs_stride;
        // The back-solve's dependence chain used to read-modify-write the
        // column in GLOBAL memory (each step waiting on an L2 round-trip:
        // GPU L1 is write-evict for global stores). Solve in a per-lane
        // local array instead — the chain stays in registers / L1-backed
        // local memory — and write the finished column out once. Same
        // floats, same operation order as `lu_solve_in_place`.
        let mut col = [0.0f32; 32];
        // 1) J^T row into the local column.
        for i in 0..ndofs {
            col[i as usize] = contact_constraint_jacs.read(col_offset + i as usize);
        }
        // 2) Solve M · column = J^T in place (tree-sparse LᵀDL; factor +
        //    parents read from global — they're workgroup-shared and stay
        //    cache-hot).
        {
            const NO_PARENT: u32 = u32::MAX;
            // Lᵀ·z = b: scatter descending (descendants before ancestors).
            for step in 0..ndofs {
                let i = ndofs - 1 - step;
                let xi = col[i as usize];
                let mut j = lu_pivots.read(piv_offset + i as usize);
                // Bounded (parents strictly decrease) so corrupt data can't hang.
                for _ in 0..ndofs {
                    if j == NO_PARENT {
                        break;
                    }
                    col[j as usize] -= mass_matrices.read(m.idx(i, j)) * xi;
                    j = lu_pivots.read(piv_offset + j as usize);
                }
            }
            // z = D⁻¹·z.
            for i in 0..ndofs {
                let d = mass_matrices.read(m.idx(i, i));
                col[i as usize] = if d != 0.0 { col[i as usize] / d } else { 0.0 };
            }
            // L·x = z: gather ascending (ancestors before descendants).
            for i in 0..ndofs {
                let mut acc = col[i as usize];
                let mut j = lu_pivots.read(piv_offset + i as usize);
                for _ in 0..ndofs {
                    if j == NO_PARENT {
                        break;
                    }
                    acc -= mass_matrices.read(m.idx(i, j)) * col[j as usize];
                    j = lu_pivots.read(piv_offset + j as usize);
                }
                col[i as usize] = acc;
            }
        }
        // 3) Write the column out + inv_r_mb = J · column in one pass.
        let mut inv_r_mb = 0.0f32;
        for i in 0..ndofs {
            let c = col[i as usize];
            contact_constraint_columns.write(col_offset + i as usize, c);
            let j = contact_constraint_jacs.read(col_offset + i as usize);
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
        s += MB_CONTACT_FINALIZE_LANES;
    }
}

/// Cooperative (threads(N)) contact PGS sweep — bit-identical to the serial
/// loop in `gpu_mb_solve_contact_constraints`, but each constraint's two
/// `O(ndofs)` passes are split across the workgroup's lanes: `J_mb · v_mb` is a
/// shared-memory tree reduction, then lane 0 forms the scalar impulse `delta`
/// (incl. the free-body term, the friction clamp against the paired normal slot,
/// and the free-body `solver_vels` update) and broadcasts it, then all lanes
/// apply disjoint DOFs. The sweep is Gauss-Seidel (constraint `s+1` reads
/// velocities written by `s`, and a tangent reads its normal's impulse), so the
/// constraint loop stays serial with a barrier each iteration. `count` is
/// workgroup-uniform, so every lane runs every iteration → barriers stay
/// uniform.
#[inline]
fn solve_contact_constraints_par(
    multibody_info: &[MultibodyInfo],
    contact_constraints: &mut [MultibodyContactConstraint],
    contact_constraint_jacs: &[f32],
    contact_constraint_columns: &[f32],
    contact_constraint_count: &[u32],
    dof_state: &mut [f32],
    solver_vels: &mut [Velocity],
    batch_id: u32,
    mb_idx: u32,
    batch_ids: &BatchIndices,
    lane: u32,
    num_lanes: u32,
    partial: &mut impl MaybeIndexUnchecked<f32>,
    shared_delta: &mut impl MaybeIndexUnchecked<f32>,
) {
    let mb_start = batch_ids.mb_start(batch_id);
    let cons_start = batch_ids.mb_contact_constraints_start(batch_id);
    let col_start = batch_ids.mb_contact_constraint_columns_start(batch_id);
    let colliders_start = batch_ids.coll_start(batch_id);
    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let ndofs = mb.ndofs;
    if ndofs == 0 {
        return;
    }
    let v_base = batch_ids.dof_start(batch_id) + mb.first_dof as usize;
    let cons_base = cons_start + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize);
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let col_base =
        col_start + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize) * dofs_stride;
    let count = contact_constraint_count.read(mb_start + mb_idx as usize);

    for s in 0..count {
        let col_offset = col_base + (s as usize) * dofs_stride;

        // J_mb · v_mb — parallel partial sums, then a 32-lane tree reduction.
        let mut local = 0.0f32;
        let mut i = lane;
        while i < ndofs {
            local += contact_constraint_jacs.read(col_offset + i as usize)
                * dof_state.read(v_base + i as usize);
            i += num_lanes;
        }
        partial.write(lane as usize, local);
        workgroup_memory_barrier_with_group_sync();
        for step in 0..5u32 {
            let stride = 1u32 << (4 - step);
            if lane < stride {
                let v = partial.read(lane as usize) + partial.read((lane + stride) as usize);
                partial.write(lane as usize, v);
            }
            workgroup_memory_barrier_with_group_sync();
        }

        if lane == 0 {
            let mut cons = contact_constraints.read(cons_base + s as usize);
            let is_self = cons.free_body_id == u32::MAX;
            let mut j_dot_v = partial.read(0);
            let free = if is_self {
                Velocity::default()
            } else {
                solver_vels.read(colliders_start + cons.free_body_id as usize)
            };
            if !is_self {
                j_dot_v += cons.lin_jac.dot(free.linear) + gdot(cons.ang_jac, free.angular);
            }
            let rhs_total = j_dot_v + cons.rhs;
            // Soft contact: ÷(1+cfm_coeff) makes the normal force compliant
            // (∝ penetration) so the foot's manifold points share load. cfm_coeff=0
            // (tangent / angular / hard) ⇒ factor 1 ⇒ unchanged hard solve.
            let raw_imp = (cons.impulse - cons.inv_lhs * (rhs_total + cons.cfm_gain * cons.impulse))
                / (1.0 + cons.cfm_coeff);
            let new_imp = if cons.kind == MB_CONTACT_KIND_TANGENT {
                let normal =
                    contact_constraints.read(cons_base + cons.normal_constraint_slot as usize);
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
            if delta != 0.0 && !is_self {
                let mut new_free = free;
                new_free.linear = new_free.linear + cons.lin_jac * (cons.free_body_im * delta);
                new_free.angular = new_free.angular + cons.ii_ang_jac * delta;
                solver_vels.write(colliders_start + cons.free_body_id as usize, new_free);
            }
            shared_delta.write(0, delta);
        }
        workgroup_memory_barrier_with_group_sync();

        let delta = shared_delta.read(0);
        if delta != 0.0 {
            let mut i = lane;
            while i < ndofs {
                let v_idx = v_base + i as usize;
                let cur = dof_state.read(v_idx);
                let col = contact_constraint_columns.read(col_offset + i as usize);
                dof_state.write(v_idx, cur + delta * col);
                i += num_lanes;
            }
        }
        // Update visible before next constraint's reduction; guards `partial` /
        // `shared_delta` before the next iteration reuses them.
        workgroup_memory_barrier_with_group_sync();
    }
}

/// One PGS sweep over the multibody's active contact constraints. Updates
/// the multibody's `dof_velocities` and the free body's `solver_vels`.
#[spirv_bindgen]
#[spirv(compute(threads(32)))]
pub fn gpu_mb_solve_contact_constraints(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contact_constraint_jacs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] contact_constraint_columns: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] contact_constraint_count: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] dof_state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] solver_vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 8)] batch_ids: &BatchIndices,
    #[spirv(workgroup)] partial: &mut [f32; MB_CONTACT_SOLVE_LANES as usize],
    #[spirv(workgroup)] shared_delta: &mut [f32; 1],
) {
    let batch_id = wg_id.y;
    let mb_idx = wg_id.x;
    let num_mb = num_multibodies.read(batch_id as usize);
    if mb_idx >= num_mb {
        return;
    }
    solve_contact_constraints_par(
        multibody_info,
        contact_constraints,
        contact_constraint_jacs,
        contact_constraint_columns,
        contact_constraint_count,
        dof_state,
        solver_vels,
        batch_id,
        mb_idx,
        batch_ids,
        lid.x,
        MB_CONTACT_SOLVE_LANES,
        partial,
        shared_delta,
    );
}

/// Strip the positional bias from each active contact constraint's `rhs`,
/// matching `gpu_mb_remove_joint_constraint_bias`.
///
/// NOTE: keep this kernel around for now (compat / standalone use), but the
/// hot end-of-substep path uses the fused
/// `gpu_mb_remove_solve_contact_no_bias` below, which inlines this loop
/// before the PGS sweep so the two threads(1) dispatches collapse into one.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_remove_contact_constraint_bias(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] contact_constraint_count: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
) {
    let batch_id = invocation_id.y;
    let mb_idx = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id as usize);
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_ids.mb_start(batch_id);
    let cons_start = batch_ids.mb_contact_constraints_start(batch_id);
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

/// Fused end-of-substep stabilization: strip the positional bias from each
/// active contact constraint's `rhs` (was `gpu_mb_remove_contact_constraint_bias`),
/// then run one PGS sweep WITHOUT bias (was `gpu_mb_solve_contact_constraints`).
///
/// Drops one `threads(1)` dispatch per substep × 4 substeps = ~4 dispatches
/// per ctrl step on the multibody side, and avoids one round-trip through
/// global memory for the `cons` struct in the bias-strip pass (the loaded
/// value is immediately reused by the solve pass).
///
/// Mirrors the existing `gpu_mb_remove_solve_joint_no_bias` fusion for joint
/// constraints (`joint_constraints.rs:669`).
#[spirv_bindgen]
#[spirv(compute(threads(32)))]
pub fn gpu_mb_remove_solve_contact_no_bias(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contact_constraint_jacs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] contact_constraint_columns: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] contact_constraint_count: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] dof_state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] solver_vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 8)] batch_ids: &BatchIndices,
    #[spirv(workgroup)] partial: &mut [f32; MB_CONTACT_SOLVE_LANES as usize],
    #[spirv(workgroup)] shared_delta: &mut [f32; 1],
) {
    let batch_id = wg_id.y;
    let mb_idx = wg_id.x;
    let lane = lid.x;
    let num_mb = num_multibodies.read(batch_id as usize);
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_ids.mb_start(batch_id);
    let cons_start = batch_ids.mb_contact_constraints_start(batch_id);
    let cons_base = cons_start + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize);
    let count = contact_constraint_count.read(mb_start + mb_idx as usize);

    // Strip the positional bias (`rhs = rhs_wo_bias`) — independent per slot, so
    // lane-split. Then a barrier so the solve sees the post-strip `rhs`.
    let mut s = lane;
    while s < count {
        let mut cons = contact_constraints.read(cons_base + s as usize);
        if cons.kind != 0 {
            cons.rhs = cons.rhs_wo_bias;
            contact_constraints.write(cons_base + s as usize, cons);
        }
        s += 32;
    }
    workgroup_memory_barrier_with_group_sync();

    solve_contact_constraints_par(
        multibody_info,
        contact_constraints,
        contact_constraint_jacs,
        contact_constraint_columns,
        contact_constraint_count,
        dof_state,
        solver_vels,
        batch_id,
        mb_idx,
        batch_ids,
        lane,
        MB_CONTACT_SOLVE_LANES,
        partial,
        shared_delta,
    );
}
