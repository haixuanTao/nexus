//! Narrow-phase collision detection: generates contact manifolds from broad-phase pairs.

use crate::math::Pose;
use crate::queries::GpuIndexedContact;
use crate::shaders::PaddedVector;
#[cfg(feature = "dim3")]
use crate::shaders::broad_phase::GpuReduceContacts;
use crate::shaders::broad_phase::{
    CollisionPair, GpuFlattenBatchesDispatch, GpuNarrowPhaseInitContactsDispatch,
    GpuNarrowPhasePfmPfm, GpuNarrowPhaseShapeShape, GpuNarrowPhaseShapeShapeDeferred,
    GpuResetNarrowPhase, NarrowPhasePfmPair,
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
    /// Pass 2: defers complex shape pairs (PFM / trimesh / polyline) into the
    /// `pfm_pairs` work-list. Split from `narrow_phase` to fit 8 storage buffers.
    narrow_phase_deferred: GpuNarrowPhaseShapeShapeDeferred,
    narrow_phase_pfm_pfm: GpuNarrowPhasePfmPfm,
    /// Builds the flat 1-D dispatch grid + prefix offsets for a per-batch
    /// work-list (used for both the collision pairs and the PFM pairs), so the
    /// kernels pack items from many batches into full warps instead of one
    /// mostly-idle workgroup per batch.
    flatten_batches: GpuFlattenBatchesDispatch,
    #[cfg(feature = "dim3")]
    reduce_contacts: GpuReduceContacts,
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
        collision_pairs: &Tensor<CollisionPair>,
        collision_pairs_len: &mut Tensor<u32>,
        collision_pairs_indirect: &mut Tensor<[u32; 3]>,
        contacts: &mut Tensor<GpuIndexedContact>,
        contacts_len: &mut Tensor<u32>,
        contacts_indirect: &mut Tensor<[u32; 3]>,
        pfm_pairs: &mut Tensor<NarrowPhasePfmPair>,
        pfm_pairs_len: &mut Tensor<u32>,
        pfm_pairs_indirect: &mut Tensor<[u32; 3]>,
        batch_indices: &Tensor<crate::shaders::utils::BatchIndices>,
        collider_parent: &Tensor<u32>,
        collider_materials: &Tensor<crate::shaders::queries::ColliderMaterial>,
        pairs_offsets: &mut Tensor<u32>,
        pfm_offsets: &mut Tensor<u32>,
        // Optional: merge each collider pair's manifolds into one before the
        // solvers see them. `false` skips the kernel entirely.
        reduce_contacts: bool,
    ) -> Result<(), GpuBackendError> {
        let num_batches = contacts_len.len() as u32;
        // Capacity-based grids for fixed-grid dispatch (see `crate::dispatch_grid`).
        // These kernels are FLAT 1-D (post-#21 flatten): they thread off
        // `global_invocation_id.x` with a grid-stride loop over ALL batches'
        // pairs, using `num_threads = num_workgroups.x * WORKGROUP_SIZE` — the
        // y/z dims are never read. The fixed grid MUST therefore be 1-D covering
        // the whole flat capacity; the old `[x, num_batches, 1]` shape made every
        // y-slice re-run the identical flat loop, so the atomic-append emitters
        // (deferred site 7 / pfm_pfm site 8) wrote each contact `num_batches`
        // times → contact/coloring blow-up (~19s/step, the known fixed-grid
        // pathology). The indirect path was already 1-D (built by
        // `flatten_batches`); this matches it.
        let pairs_grid = [(collision_pairs.len() as u32).max(1).div_ceil(64), 1, 1];
        let pfm_grid = [(pfm_pairs.len() as u32).max(1).div_ceil(64), 1, 1];
        self.reset_narrow_phase
            .call(pass, [1u32, num_batches, 1], contacts_len, pfm_pairs_len)?;

        // The broad phase wrote a `[max/64, num_batches, 1]` grid into
        // `collision_pairs_indirect`; rewrite it (and derive the offsets) for
        // the flat layout. Nothing else consumes the batched form.
        self.flatten_batches.call(
            pass,
            1u32,
            collision_pairs_len,
            pairs_offsets,
            collision_pairs_indirect,
            batch_indices,
        )?;

        self.narrow_phase.call(
            pass,
            crate::dispatch_grid_tagged(collision_pairs_indirect, pairs_grid, 6),
            collision_pairs,
            pairs_offsets,
            poses,
            shapes,
            contacts,
            contacts_len,
            batch_indices,
            collider_parent,
            collider_materials,
        )?;

        // Pass 2: defer the complex shape pairs into `pfm_pairs` (kept as a
        // separate dispatch so each pass fits 8 storage buffers).
        self.narrow_phase_deferred.call(
            pass,
            // FIXED (was default-indirect): the "~19s/step, 100% host CPU"
            // pathology was a stale 2-D fixed grid `[x, num_batches, 1]` — these
            // flat kernels only thread off `num_workgroups.x`, so every y-slice
            // re-ran the whole flat loop and the atomic-append emitters wrote each
            // contact `num_batches` times, exploding contact/coloring. The grid is
            // now 1-D (see above), so fixed dispatch is correct here and drops the
            // per-step host sync that blocked CUDA-graph capture.
            crate::dispatch_grid_tagged(collision_pairs_indirect, pairs_grid, 7),
            collision_pairs,
            pairs_offsets,
            poses,
            shapes,
            pfm_pairs,
            pfm_pairs_len,
            batch_indices,
            vertices,
            indices,
        )?;

        self.flatten_batches.call(
            pass,
            1u32,
            pfm_pairs_len,
            pfm_offsets,
            pfm_pairs_indirect,
            batch_indices,
        )?;
        self.narrow_phase_pfm_pfm.call(
            pass,
            crate::dispatch_grid_tagged(&*pfm_pairs_indirect, pfm_grid, 8),
            contacts,
            contacts_len,
            pfm_pairs,
            pfm_offsets,
            batch_indices,
            vertices,
            indices,
            collider_parent,
            collider_materials,
        )?;
        #[cfg(feature = "dim3")]
        if reduce_contacts {
            self.reduce_contacts.call(
                pass,
                [1u32, num_batches, 1],
                contacts,
                contacts_len,
                batch_indices,
            )?;
        }
        #[cfg(not(feature = "dim3"))]
        let _ = reduce_contacts;
        self.init_contacts_indirect_args
            .call(pass, 1u32, contacts_len, contacts_indirect)?;

        Ok(())
    }
}
