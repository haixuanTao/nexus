//! Narrow-phase collision detection for generating contact manifolds.
//!
//! After the broad-phase identifies potentially colliding pairs using AABBs, the narrow-phase
//! performs detailed collision tests to generate contact manifolds. These manifolds contain
//! precise contact point information needed for physics simulation.
//!
//! The narrow-phase:
//! 1. Takes collision pairs from the broad-phase.
//! 2. Performs shape-specific collision tests (ball-ball, cuboid-cuboid, etc.)
//! 3. Generates contact manifolds with points, normals, and penetration depths.
//! 4. Outputs indexed contacts for the physics solver.

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
///
/// This shader performs detailed collision tests on potentially colliding pairs identified
/// by the broad-phase. It generates contact manifolds containing:
/// - Contact points (up to 2 in 2D, 4 in 3D)
/// - Contact normals
/// - Penetration depths
///
/// # Pipeline Stages
///
/// The narrow-phase executes in two stages:
/// 1. **Main**: Processes collision pairs and generates contacts.
/// 2. **Init indirect args**: Prepares dispatch arguments for subsequent kernels.
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
    ///
    /// # Parameters
    ///
    /// - `pass`: The compute pass to record commands into
    /// - `_num_colliders`: Total number of colliders (unused currently)
    /// - `poses`: Collider poses (positions and rotations)
    /// - `shapes`: Collider shapes
    /// - `vertices`: Vertex buffer for mesh shapes
    /// - `indices`: Index buffer for mesh shapes
    /// - `collision_pairs`: Potentially colliding pairs from broad-phase
    /// - `collision_pairs_len`: Number of collision pairs
    /// - `collision_pairs_indirect`: Indirect dispatch arguments for collision pairs
    /// - `contacts`: Output buffer for contact manifolds
    /// - `contacts_len`: Output count of generated contacts
    /// - `contacts_indirect`: Indirect dispatch arguments for contacts
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
        self.reset_narrow_phase
            .call(pass, 1, contacts_len, pfm_pairs_len)?;

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
