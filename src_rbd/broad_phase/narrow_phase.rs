//! Narrow-phase collision detection: generates contact manifolds from broad-phase pairs.

use crate::math::Pose;
use crate::queries::GpuIndexedContact;
use crate::shaders::PaddedVector;
#[cfg(feature = "dim3")]
use crate::shaders::broad_phase::GpuReduceContacts;
use crate::shaders::broad_phase::{
    CollisionPair, GpuContactSortGather, GpuContactSortKeys, GpuContactSortPairKeys,
    GpuInitPfmPfmDispatch, GpuNarrowPhaseInitContactsDispatch, GpuNarrowPhasePfmPfm,
    GpuNarrowPhaseShapeShape, GpuNarrowPhaseShapeShapeDeferred, GpuResetNarrowPhase,
    NarrowPhasePfmPair,
};
use crate::utils::{RadixSort, RadixSortWorkspace};
use khal::backend::GpuBackend;
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
    #[cfg(feature = "dim3")]
    reduce_contacts: GpuReduceContacts,
    init_pfm_pfm_indirect_args: GpuInitPfmPfmDispatch,
    init_contacts_indirect_args: GpuNarrowPhaseInitContactsDispatch,
    contact_sort_keys: GpuContactSortKeys,
    contact_sort_pair_keys: GpuContactSortPairKeys,
    contact_sort_gather: GpuContactSortGather,
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
        collision_pairs_len: &Tensor<u32>,
        collision_pairs_indirect: &Tensor<[u32; 3]>,
        contacts: &mut Tensor<GpuIndexedContact>,
        contacts_len: &mut Tensor<u32>,
        contacts_indirect: &mut Tensor<[u32; 3]>,
        mb_sweep_indirect: &mut Tensor<[u32; 3]>,
        pfm_pairs: &mut Tensor<NarrowPhasePfmPair>,
        pfm_pairs_len: &mut Tensor<u32>,
        pfm_pairs_indirect: &mut Tensor<[u32; 3]>,
        batch_indices: &Tensor<crate::shaders::utils::BatchIndices>,
        collider_parent: &Tensor<u32>,
        collider_materials: &Tensor<crate::shaders::queries::ColliderMaterial>,
        // Optional: merge each collider pair's manifolds into one before the
        // solvers see them. `false` skips the kernel entirely.
        reduce_contacts: bool,
    ) -> Result<(), GpuBackendError> {
        let num_batches = contacts_len.len() as u32;
        // Capacity-based grids for fixed-grid dispatch (see `crate::dispatch_grid`).
        let nb = num_batches.max(1);
        let pairs_grid = [
            (collision_pairs.len() as u32 / nb).max(1).div_ceil(64),
            num_batches,
            1,
        ];
        let pfm_grid = [(pfm_pairs.len() as u32 / nb).max(1).div_ceil(64), num_batches, 1];
        self.reset_narrow_phase
            .call(pass, [num_batches, 1, 1], contacts_len, pfm_pairs_len)?;

        self.narrow_phase.call(
            pass,
            crate::dispatch_grid(collision_pairs_indirect, pairs_grid),
            collision_pairs,
            collision_pairs_len,
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
            crate::dispatch_grid(collision_pairs_indirect, pairs_grid),
            collision_pairs,
            collision_pairs_len,
            poses,
            shapes,
            pfm_pairs,
            pfm_pairs_len,
            batch_indices,
            vertices,
            indices,
        )?;

        // Single 256-lane workgroup: parallel max over the per-batch counts.
        self.init_pfm_pfm_indirect_args
            .call(pass, 256u32, pfm_pairs_len, pfm_pairs_indirect)?;
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
        // Single 256-lane workgroup: parallel max over the per-batch counts.
        // Also emits the (possibly zero-workgroup) grid for the per-multibody
        // contact-constraint dispatches.
        self.init_contacts_indirect_args.call(
            pass,
            256u32,
            contacts_len,
            contacts_indirect,
            mb_sweep_indirect,
            batch_indices,
        )?;

        Ok(())
    }

    /// Canonicalizes the contact-buffer order (see
    /// `RbdPipeline::deterministic_contacts`) to (collider pair, feature)
    /// lexicographic: a stable radix sort by feature id (LSD stage), a re-key
    /// of that permutation by packed collider pair, a second stable radix
    /// sort, then a gather into `contacts_scratch`. The caller copies the
    /// scratch back over `contacts` with an encoder buffer copy (an in-place
    /// gather would race), and runs contact reduction AFTER that copy — the
    /// greedy reduce merges in buffer order, so it must see the sorted order.
    ///
    /// `colliders_batch_capacity^2` must fit in `u32` (the packed sort key);
    /// the caller gates on that.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_contact_sort(
        &self,
        backend: &GpuBackend,
        pass: &mut GpuPass,
        sort: &RadixSort,
        workspace: &mut RadixSortWorkspace,
        contacts: &Tensor<GpuIndexedContact>,
        contacts_len: &Tensor<u32>,
        batch_indices: &Tensor<crate::shaders::utils::BatchIndices>,
        sort_keys: &mut Tensor<u32>,
        sort_vals: &mut Tensor<u32>,
        sorted_keys: &mut Tensor<u32>,
        sorted_vals: &mut Tensor<u32>,
        contacts_scratch: &mut Tensor<GpuIndexedContact>,
        colliders_batch_capacity: u32,
    ) -> Result<(), GpuBackendError> {
        let num_batches = contacts_len.len() as u32;
        let nb = num_batches.max(1);
        let cap = (contacts.len() as u32 / nb).max(1);
        let grid = [cap, num_batches, 1];
        // Stage 1 (LSD): stable sort by feature id (trimesh/polyline BVH leaf,
        // 0 for analytic contacts). Feature ids are unbounded — full 32 bits.
        self.contact_sort_keys.call(
            pass,
            grid,
            contacts,
            contacts_len,
            sort_keys,
            sort_vals,
            batch_indices,
        )?;
        sort.dispatch(
            backend,
            pass,
            workspace,
            sort_keys,
            sort_vals,
            contacts_len,
            32,
            num_batches,
            sorted_keys,
            sorted_vals,
        )?;
        // Stage 2 (MSD): re-key the feature-ranked permutation by packed
        // collider pair and stable-sort again — net order is (pair, feature)
        // lexicographic, which is unique per contact record.
        self.contact_sort_pair_keys.call(
            pass,
            grid,
            contacts,
            contacts_len,
            sorted_vals,
            sort_keys,
            sort_vals,
            batch_indices,
        )?;
        // Bits needed for the packed pair key (max value cap^2 - 1), floored
        // at one radix pass — the zero-pass path would skip the batched init
        // that writes the outputs.
        let key_max = (colliders_batch_capacity as u64 * colliders_batch_capacity as u64)
            .saturating_sub(1)
            .max(1) as u32;
        let sorting_bits = (32 - key_max.leading_zeros()).max(4);
        sort.dispatch(
            backend,
            pass,
            workspace,
            sort_keys,
            sort_vals,
            contacts_len,
            sorting_bits,
            num_batches,
            sorted_keys,
            sorted_vals,
        )?;
        self.contact_sort_gather.call(
            pass,
            grid,
            contacts,
            contacts_len,
            sorted_vals,
            contacts_scratch,
            batch_indices,
        )?;
        Ok(())
    }

    /// Standalone contact reduction, for the deterministic-order path: the
    /// greedy reduce merges each collider pair's manifolds in buffer order,
    /// so `RbdPipeline` runs it AFTER the canonical sort has been copied back
    /// (the in-dispatch reduce is skipped in that mode). Pre-reduce indirect
    /// grids stay valid — reduction only shrinks per-batch counts.
    #[cfg(feature = "dim3")]
    pub fn dispatch_reduce_contacts(
        &self,
        pass: &mut GpuPass,
        contacts: &mut Tensor<GpuIndexedContact>,
        contacts_len: &mut Tensor<u32>,
        batch_indices: &Tensor<crate::shaders::utils::BatchIndices>,
    ) -> Result<(), GpuBackendError> {
        let num_batches = contacts_len.len() as u32;
        self.reduce_contacts.call(
            pass,
            [1u32, num_batches, 1],
            contacts,
            contacts_len,
            batch_indices,
        )
    }
}
