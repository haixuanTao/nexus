//! Narrow-phase collision detection: generates contact manifolds from broad-phase pairs.

use crate::math::Pose;
use crate::queries::GpuIndexedContact;
use crate::shaders::PaddedVector;
use crate::shaders::broad_phase::{
    GpuInitPfmPfmDispatch, GpuNarrowPhaseInitContactsDispatch, GpuNarrowPhasePfmPfm,
    GpuNarrowPhaseShapeShape, GpuResetNarrowPhase, NarrowPhasePfmPair,
};
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
        contacts_batch_capacity: &Tensor<u32>,
        colliders_batch_capacity: &Tensor<u32>,
    ) -> Result<(), GpuBackendError> {
        let num_batches = contacts_len.len() as u32;
        self.reset_narrow_phase
            .call(pass, [1u32, num_batches, 1], contacts_len, pfm_pairs_len)?;

        self.narrow_phase.call(
            pass,
            collision_pairs_indirect,
            collision_pairs,
            collision_pairs_len,
            poses,
            shapes,
            contacts,
            contacts_len,
            pfm_pairs,
            pfm_pairs_len,
            contacts_batch_capacity,
            colliders_batch_capacity,
            vertices,
            indices,
        )?;

        self.init_pfm_pfm_indirect_args
            .call(pass, 1u32, pfm_pairs_len, pfm_pairs_indirect)?;

        self.narrow_phase_pfm_pfm.call(
            pass,
            &*pfm_pairs_indirect,
            contacts,
            contacts_len,
            pfm_pairs,
            pfm_pairs_len,
            contacts_batch_capacity,
            vertices,
            indices,
        )?;

        self.init_contacts_indirect_args
            .call(pass, 1u32, contacts_len, contacts_indirect)?;

        Ok(())
    }
}
