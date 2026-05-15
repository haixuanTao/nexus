//! Augmented mass matrix assembly via CRBA, with optional Coriolis /
//! gyroscopic terms.
//!
//! Rapier:
//!     self.augmented_mass.quadform(1.0, &rb_mass_matrix_wo_gyro, body_jacobian, 1.0);
//!
//! Here we use `quadform_spatial` which exploits the block-diagonal structure of the
//! per-link spatial mass to avoid forming the full SPATIAL_DIM × SPATIAL_DIM matrix.
//! World-space inertia is recomputed from the link's current orientation.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

#[cfg(feature = "dim3")]
use glamx::{Mat3, Vec3};

use crate::dynamics::body::LocalMassProperties;
use crate::dynamics::joint::SPATIAL_DIM;
#[cfg(feature = "dim3")]
use crate::rotation_to_matrix;
use crate::utils::Slice;
use crate::utils::linalg::{
    MAX_MB_DOFS, MatSlice, copy_from, copy_from_par, fill, fill_par, gemm_inertia_lhs_cross_buf,
    gemm_inertia_lhs_par, gemm_omega_skew_tr_cross_buf, gemm_omega_skew_tr_cross_buf_par,
    gemm_skew_tr_lhs_cross_buf, gemm_skew_tr_lhs_cross_buf_par, gemm_skew_tr_lhs_par, gemm_tr,
    gemm_tr_par, quadform_spatial, quadform_spatial_par,
};
#[cfg(feature = "dim3")]
use crate::utils::linalg::{gemm_skew_lhs_cross_buf, gemm_skew_lhs_cross_buf_par};
use crate::{ANG_DIM, DIM};

use super::jacobian::joint_jacobian_column;
use super::types::{MultibodyInfo, MultibodyLinkStatic, MultibodyLinkWorkspace};

/// Workgroup width for the parallelised mass-matrix kernel. Must match the
/// `MB_MM_LANES` constant on the host side and the `threads(...)` attribute on
/// `gpu_mb_mass_matrix_with_coriolis`.
const LANES: u32 = 32;

/// World-space inertia for this link.
///
/// In 3D returns a `Mat3` (`I_world = R · diag(principal_inertia) · Rᵀ`). In 2D
/// returns the scalar moment of inertia (already in world frame because there
/// is only one rotational DOF).
#[cfg(feature = "dim3")]
#[inline]
pub(super) fn link_world_inertia(ws: &MultibodyLinkWorkspace, lmp: &LocalMassProperties) -> Mat3 {
    let ipi = lmp.inv_principal_inertia;
    let px = if ipi.x != 0.0 { 1.0 / ipi.x } else { 0.0 };
    let py = if ipi.y != 0.0 { 1.0 / ipi.y } else { 0.0 };
    let pz = if ipi.z != 0.0 { 1.0 / ipi.z } else { 0.0 };
    let r = rotation_to_matrix(ws.local_to_world.rotation * lmp.inertia_ref_frame);
    // M = r · diag(px, py, pz) (column-scale); I = M · rᵀ.
    let m = Mat3::from_cols(r.x_axis * px, r.y_axis * py, r.z_axis * pz);
    m * r.transpose()
}

#[cfg(feature = "dim2")]
#[inline]
pub(super) fn link_world_inertia(_ws: &MultibodyLinkWorkspace, lmp: &LocalMassProperties) -> f32 {
    if lmp.inv_inertia != 0.0 {
        1.0 / lmp.inv_inertia
    } else {
        0.0
    }
}

//
// Mass matrix with Coriolis + gyroscopic terms.
//
// Mirrors rapier's `update_inertias`. In 3D this includes a gyroscopic
// derivative `[ω]_× · I − [Iω]_×` on the augmented inertia and the full
// `coriolis_w` propagation; in 2D the gyroscopic term is zero and
// `coriolis_w` collapses to a 1-row block.

/// Scale each column of `dst_v` (`DIM × ndofs`) by a scalar, in place:
/// `dst_v := scale · src_v`.
#[inline]
fn scaled_copy_lin_dim(
    buf_dst: &mut [f32],
    dst: MatSlice,
    scale: f32,
    buf_src: &[f32],
    src: MatSlice,
) {
    for c in 0..dst.cols {
        for r in 0..DIM {
            buf_dst.write(dst.idx(r, c), scale * buf_src.read(src.idx(r, c)));
        }
    }
}
