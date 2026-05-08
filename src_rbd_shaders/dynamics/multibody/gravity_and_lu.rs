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

use glamx::{Vec2, Vec3};

use crate::dynamics::body::{LocalMassProperties, Velocity};
use crate::dynamics::joint::SPATIAL_DIM;
use crate::utils::{Slice, SliceMut};
use crate::utils::linalg::{MAX_MB_DOFS, MatSlice, fill_par, gemv_tr_spatial_split_par};
use crate::{AngVector, Vector, gcross_av};

use super::lu::{
    LANES, lu_apply_pivots, lu_factor_in_shared, lu_triangular_solve_in_place, sm_idx,
};
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    links_workspace: &mut [MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] links_local_mprops: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] body_jacobians: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] gen_forces: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] mass_matrices: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] lu_pivots: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] dof_velocities: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] damping: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 10)] num_multibodies: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 11)] gravity: &[f32; 3],
    #[spirv(uniform, descriptor_set = 0, binding = 12)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 13)] links_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 14)] jacobians_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 15)] mass_matrix_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 16)] dof_batch_capacity: &u32,
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
    let batch_id = wg_id.y as usize;
    let mb_idx = wg_id.x;
    let lane = lid.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let links_start = batch_id * *links_batch_capacity as usize;
    let jac_start = batch_id * *jacobians_batch_capacity as usize;
    let mm_start = batch_id * *mass_matrix_batch_capacity as usize;
    let dof_start = batch_id * *dof_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let num_links = mb.num_links;
    let ndofs = mb.ndofs;
    let first_link_global = links_start + mb.first_link as usize;
    let mb_jac_base = jac_start + mb.jacobian_offset as usize;
    let gen_base = dof_start + mb.first_dof as usize;
    let mb_mm_base = mm_start + mb.mass_matrix_offset as usize;
    let piv_offset = gen_base;
    let rhs_offset = gen_base;

    let stat_slice = Slice(links_static, first_link_global);
    let mut ws_slice = SliceMut(links_workspace, first_link_global);
    let local_mprops_slice = Slice(links_local_mprops, first_link_global);
    let vel_slice = Slice(dof_velocities, gen_base);
    let damping_slice = Slice(damping, gen_base);

    // ---- Phase 1: zero the generalized-force vector (parallel across DOFs). ----
    let accelerations = MatSlice::dense(gen_base, ndofs, 1);
    fill_par(gen_forces, accelerations, 0.0, lane, LANES);
    workgroup_memory_barrier_with_group_sync();

    let _ = stat_slice;

    #[cfg(feature = "dim3")]
    let g = Vec3::new(gravity[0], gravity[1], gravity[2]);
    #[cfg(feature = "dim2")]
    let g = Vec2::new(gravity[0], gravity[1]);

    // ---- Phase 2: per-link gravity / Coriolis-force assembly. ----
    for k in 0..num_links {
        let (
            self_joint_vel_lin,
            self_joint_vel_ang,
            self_shift02,
            self_shift23,
            self_local_to_world,
            self_rb_ang,
        ) = {
            let ws = ws_slice.at(k as usize);
            (
                ws.joint_velocity.linear,
                ws.joint_velocity.angular,
                ws.shift02,
                ws.shift23,
                ws.local_to_world,
                ws.rb_vels.angular,
            )
        };

        let mut acc_lin = Vector::ZERO;
        #[cfg(feature = "dim3")]
        let mut acc_ang: AngVector = AngVector::ZERO;
        #[cfg(feature = "dim2")]
        let mut acc_ang: AngVector = 0.0;

        if k != 0 {
            let stat = stat_slice.read(k as usize);
            let parent_ws = ws_slice.at(stat.parent_link_id as usize);
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
            ws_slice.at_mut(k as usize).kinematic_acc = Velocity::new(acc_lin, acc_ang);
        }
        workgroup_memory_barrier_with_group_sync();

        let lmp = local_mprops_slice.read(k as usize);
        let inv_mass_x = lmp.inv_mass.x;
        if inv_mass_x == 0.0 {
            continue;
        }
        let mass = 1.0 / inv_mass_x;
        let rb_inertia = link_world_inertia(ws_slice.at(k as usize), &lmp);
        let _ = self_local_to_world;

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
        gen_forces.write(
            idx,
            cur - damping_slice.read(i as usize) * vel_slice.read(i as usize),
        );
    }
    workgroup_memory_barrier_with_group_sync();

    // ---- Phase 3: load M into shared memory, factor in place. ----
    if ndofs == 0 {
        return;
    }
    let m_view = MatSlice::dense(mb_mm_base, ndofs, ndofs);
    if lane < ndofs {
        for r in 0..ndofs {
            mat.write(sm_idx(r, lane), mass_matrices.read(m_view.idx(r, lane)));
        }
        x.write(lane as usize, gen_forces.read(rhs_offset + lane as usize));
    }
    workgroup_memory_barrier_with_group_sync();

    lu_factor_in_shared(
        ndofs,
        lane,
        mat,
        lu_pivots,
        piv_offset,
        pivot_row_shared,
        inv_akk_shared,
    );

    // Persist LU factors to global memory (joint / contact constraint init
    // reuses them for unit-RHS solves).
    if lane < ndofs {
        for r in 0..ndofs {
            mass_matrices.write(m_view.idx(r, lane), mat.read(sm_idx(r, lane)));
        }
    }

    // ---- Phase 4: solve M·x = τ for the gravity rhs. ----
    lu_apply_pivots(ndofs, lane, lu_pivots, piv_offset, x);
    lu_triangular_solve_in_place(ndofs, lane, mat, x, partial);

    if lane < ndofs {
        gen_forces.write(rhs_offset + lane as usize, x.read(lane as usize));
    }
}
