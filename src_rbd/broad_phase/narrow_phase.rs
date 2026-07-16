//! Narrow-phase collision detection: generates contact manifolds from broad-phase pairs.

use crate::math::Pose;
use crate::queries::GpuIndexedContact;
use crate::shaders::PaddedVector;
use crate::shaders::broad_phase::{
    GpuInitPfmPfmDispatch, GpuNarrowPhaseInitContactsDispatch, GpuNarrowPhasePfmPfm,
    GpuNarrowPhaseShapeShape, GpuResetNarrowPhase, NarrowPhasePfmPair,
};
#[cfg(feature = "dim3")]
use crate::shaders::broad_phase::GpuReduceContacts;
use crate::shaders::shapes::Shape;
use khal::Shader;
use khal::backend::{GpuBackendError, GpuPass};
use vortx::tensor::Tensor;

/// GPU shader for narrow-phase collision detection.
#[derive(Shader)]
pub struct GpuNarrowPhase {
    reset_narrow_phase: GpuResetNarrowPhase,
    narrow_phase: GpuNarrowPhaseShapeShape,
    narrow_phase_pfm_pfm: GpuNarrowPhasePfmPfm,
    #[cfg(feature = "dim3")]
    reduce_contacts: GpuReduceContacts,
    init_pfm_pfm_indirect_args: GpuInitPfmPfmDispatch,
    init_contacts_indirect_args: GpuNarrowPhaseInitContactsDispatch,
}

impl GpuNarrowPhase {
    /// Dispatches the narrow-phase collision detection pipeline.
    pub fn dispatch(
        &self,
        pass: &mut GpuPass,
        _num_colliders: u32,
        poses: &Tensor<Pose>,
        shapes: &Tensor<Shape>,
        vertices: &Tensor<PaddedVector>,
        indices: &Tensor<u32>,
        collision_pairs: &Tensor<[u32; 2]>,
        collision_pairs_len: &Tensor<u32>,
        collision_pairs_indirect: &Tensor<[u32; 3]>,
        contacts: &mut Tensor<GpuIndexedContact>,
        contacts_len: &mut Tensor<u32>,
        contacts_indirect: &mut Tensor<[u32; 3]>,
        pfm_pairs: &mut Tensor<NarrowPhasePfmPair>,
        pfm_pairs_len: &mut Tensor<u32>,
        pfm_pairs_indirect: &mut Tensor<[u32; 3]>,
        batch_indices: &Tensor<crate::shaders::utils::BatchIndices>,
        // OPTIONAL training-grade contact reduction: merge all manifolds of a
        // collider pair (per-triangle trimesh contacts) to one ≤4-point
        // manifold before the solvers see them. Single-manifold pairs pass
        // through bit-identically; `false` skips the kernel entirely.
        reduce_contacts: bool,
    ) -> Result<(), GpuBackendError> {
        let num_batches = contacts_len.len() as u32;
        // Fixed capacity-based grids (BIPED_FIXED_GRID) — both kernels grid-stride
        // over the true per-batch count, so `[ceil(cap/64), num_batches, 1]` is
        // correct and avoids the per-dispatch GPU-drain of indirect dispatch.
        let nb = num_batches.max(1);
        let pairs_grid = [
            (collision_pairs.len() as u32 / nb).max(1).div_ceil(64),
            num_batches,
            1,
        ];
        let pfm_grid = [(pfm_pairs.len() as u32 / nb).max(1).div_ceil(64), num_batches, 1];
        self.reset_narrow_phase
            .call(pass, [1u32, num_batches, 1], contacts_len, pfm_pairs_len)?;

        self.narrow_phase.call(
            pass,
            crate::dispatch_grid(collision_pairs_indirect, pairs_grid),
            collision_pairs,
            collision_pairs_len,
            poses,
            shapes,
            contacts,
            contacts_len,
            pfm_pairs,
            pfm_pairs_len,
            batch_indices,
            vertices,
            indices,
        )?;

        self.init_pfm_pfm_indirect_args
            .call(pass, 1u32, pfm_pairs_len, pfm_pairs_indirect)?;
        self.narrow_phase_pfm_pfm.call(
            pass,
            crate::dispatch_grid(&*pfm_pairs_indirect, pfm_grid),
            contacts,
            contacts_len,
            pfm_pairs,
            pfm_pairs_len,
            batch_indices,
            vertices,
            indices,
        )?;
        #[cfg(feature = "dim3")]
        if reduce_contacts {
            self.reduce_contacts
                .call(pass, [1u32, num_batches, 1], contacts, contacts_len, batch_indices)?;
        }
        #[cfg(not(feature = "dim3"))]
        let _ = reduce_contacts;
        self.init_contacts_indirect_args
            .call(pass, 1u32, contacts_len, contacts_indirect)?;

        Ok(())
    }
}
