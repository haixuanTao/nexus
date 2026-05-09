//! Linear Bounding Volume Hierarchy (LBVH) broad-phase collision detection.
//!
//! Implements the Karras 2012 parallel LBVH construction algorithm on the GPU,
//! providing O(n log n) collision detection suitable for large dynamic scenes.

use crate::math::Pose;
use crate::shaders::PaddedVector;
use crate::shaders::bounding_volumes::Aabb;
use crate::shaders::broad_phase::{
    GpuLbvhBuild, GpuLbvhComputeDomain, GpuLbvhComputeMorton, GpuLbvhFindCollisionPairs,
    GpuLbvhInitDispatch, GpuLbvhRefitInternal, GpuLbvhRefitLeaves, GpuLbvhResetCollisionPairs,
    LbvhNode,
};
use crate::shaders::shapes::Shape;
use crate::utils::{RadixSort, RadixSortWorkspace};
use khal::backend::{Encoder, GpuBackend, GpuBackendError, GpuEncoder, GpuPass, GpuTimestamps};
use khal::{BufferUsages, Shader};
use vortx::tensor::Tensor;

/// GPU shader for Linear Bounding Volume Hierarchy (LBVH) construction and traversal.
///
/// Implements the Karras 2012 parallel LBVH construction algorithm on the GPU, providing
/// O(n log n) collision detection suitable for large dynamic scenes.
#[derive(Shader)]
pub struct GpuLbvh {
    compute_domain: GpuLbvhComputeDomain,
    compute_morton: GpuLbvhComputeMorton,
    build: GpuLbvhBuild,
    refit_leaves: GpuLbvhRefitLeaves,
    refit_internal: GpuLbvhRefitInternal,
    reset_collision_pairs: GpuLbvhResetCollisionPairs,
    find_collision_pairs: GpuLbvhFindCollisionPairs,
    lbvh_init_indirect_args: GpuLbvhInitDispatch,
}

/// GPU-resident state for LBVH construction and queries.
///
/// Buffers automatically resize when the number of colliders changes.
pub struct LbvhState {
    buffer_usages: BufferUsages,
    domain_aabb: Tensor<Aabb>,
    n_sort: Tensor<u32>,
    unsorted_morton_keys: Tensor<u32>,
    sorted_morton_keys: Tensor<u32>,
    unsorted_colliders: Tensor<u32>,
    sorted_colliders: Tensor<u32>,
    tree: Tensor<LbvhNode>,
    sort_workspace: RadixSortWorkspace,
}

/// High-level LBVH broad-phase interface.
pub struct Lbvh {
    shaders: GpuLbvh,
    sort: RadixSort,
}

impl LbvhState {
    /// Creates a new LBVH state with default buffer usage flags.
    pub fn new(backend: &GpuBackend) -> Self {
        Self::with_usages(backend, BufferUsages::STORAGE)
    }

    /// Creates a new LBVH state with custom buffer usage flags.
    pub fn with_usages(backend: &GpuBackend, usages: BufferUsages) -> Self {
        Self {
            n_sort: Tensor::scalar(backend, 0, usages).unwrap(),
            domain_aabb: Tensor::scalar_uninit(backend, usages).unwrap(),
            unsorted_morton_keys: Tensor::vector_uninit(backend, 0, usages).unwrap(),
            sorted_morton_keys: Tensor::vector_uninit(backend, 0, usages).unwrap(),
            unsorted_colliders: Tensor::vector_uninit(backend, 0, usages).unwrap(),
            sorted_colliders: Tensor::vector_uninit(backend, 0, usages).unwrap(),
            tree: Tensor::vector_uninit(backend, 0, usages).unwrap(),
            sort_workspace: RadixSortWorkspace::new(backend),
            buffer_usages: usages,
        }
    }

    pub(crate) fn tree(&self) -> &Tensor<LbvhNode> {
        &self.tree
    }

    pub(crate) fn sorted_colliders(&self) -> &Tensor<u32> {
        &self.sorted_colliders
    }

    fn resize_buffers(&mut self, backend: &GpuBackend, colliders_len: u32, num_batches: u32) {
        if (self.domain_aabb.len() as u32) < num_batches {
            self.domain_aabb =
                Tensor::vector_uninit(backend, num_batches, self.buffer_usages).unwrap();
        }

        // NOTE: colliders_len is the total colliders count, already taking all batches into account.
        if (self.tree.len() as u32) < 2 * colliders_len {
            self.unsorted_morton_keys =
                Tensor::vector_uninit(backend, colliders_len, self.buffer_usages).unwrap();
            self.sorted_morton_keys =
                Tensor::vector_uninit(backend, colliders_len, self.buffer_usages).unwrap();
            // Use per-batch LOCAL indices so that after sorting, each batch's
            // sorted_colliders slice contains local indices usable with per-batch
            // Slice offsets in the shaders.
            let colliders_per_batch = colliders_len / num_batches;
            let unsorted_colliders: Vec<_> = (0..num_batches)
                .flat_map(|_| 0..colliders_per_batch)
                .collect();
            self.unsorted_colliders =
                Tensor::vector(backend, &unsorted_colliders, self.buffer_usages).unwrap();
            self.sorted_colliders =
                Tensor::vector_uninit(backend, colliders_len, self.buffer_usages).unwrap();
            self.tree =
                Tensor::vector_uninit(backend, 2 * colliders_len, self.buffer_usages).unwrap();

            // FIXME: we should instead write the len into the existing buffer at each frame
            //        to handle dynamic body/collider insertion/removal.
            // FIXME: this doesn’t account for batches having mismatched numbers of colliders.
            // n_sort is a per-batch vector: each element is the per-batch key count.
            // The radix sort init kernel infers num_batches from n_sort.len().
            let n_sort_data = vec![colliders_per_batch; num_batches as usize];
            self.n_sort = Tensor::vector(backend, &n_sort_data, self.buffer_usages).unwrap();
        }
    }
}

impl Lbvh {
    /// Creates a new LBVH instance by loading shaders on the given backend.
    pub fn from_backend(backend: &GpuBackend) -> Self {
        Self {
            shaders: GpuLbvh::from_backend(backend).unwrap(),
            sort: RadixSort::from_backend(backend).unwrap(),
        }
    }

    /// Rebuilds the LBVH tree from current collider poses and shapes.
    ///
    /// Should be called each frame before [`find_pairs`](Self::find_pairs) if colliders have moved.
    pub fn update_tree(
        &self,
        backend: &GpuBackend,
        encoder: &mut GpuEncoder,
        state: &mut LbvhState,
        colliders_len: u32,
        num_batches: u32,
        poses: &Tensor<Pose>,
        vertex_buffers: &Tensor<PaddedVector>,
        shapes: &Tensor<Shape>,
        num_shapes: &Tensor<u32>,
        colliders_batch_capacity: &Tensor<u32>,
        mut timestamps: Option<&mut GpuTimestamps>,
    ) -> Result<(), GpuBackendError> {
        state.resize_buffers(backend, colliders_len, num_batches);

        let colliders_per_batch = colliders_len / num_batches;

        let mut pass = encoder.begin_pass("lbvh-compute-domain", timestamps.as_deref_mut());
        self.shaders.compute_domain.call(
            &mut pass,
            [1u32, num_batches, 1],
            poses,
            &mut state.domain_aabb,
            num_shapes,
            colliders_batch_capacity,
        )?;
        pass.memory_barrier();
        drop(pass);

        let mut pass = encoder.begin_pass("lbvh-compute-morton", timestamps.as_deref_mut());
        self.shaders.compute_morton.call(
            &mut pass,
            [colliders_per_batch, num_batches, 1],
            poses,
            &state.domain_aabb,
            &mut state.unsorted_morton_keys,
            num_shapes,
            colliders_batch_capacity,
        )?;
        pass.memory_barrier();
        drop(pass);

        let mut pass = encoder.begin_pass("lbvh-sort-dispatch", timestamps.as_deref_mut());
        self.sort.dispatch(
            backend,
            &mut pass,
            &mut state.sort_workspace,
            &state.unsorted_morton_keys,
            &state.unsorted_colliders,
            &state.n_sort,
            32,
            num_batches,
            &mut state.sorted_morton_keys,
            &mut state.sorted_colliders,
        )?;
        drop(pass);

        let mut pass = encoder.begin_pass("lbvh-build", timestamps.as_deref_mut());
        self.shaders.build.call(
            &mut pass,
            [colliders_per_batch.saturating_sub(1), num_batches, 1],
            &state.sorted_morton_keys,
            &mut state.tree,
            num_shapes,
            colliders_batch_capacity,
        )?;
        pass.memory_barrier();
        drop(pass);

        let mut pass = encoder.begin_pass("lbvh-refit_leaves", timestamps.as_deref_mut());
        self.shaders.refit_leaves.call(
            &mut pass,
            [colliders_per_batch, num_batches, 1],
            poses,
            shapes,
            &state.sorted_colliders,
            &mut state.tree,
            num_shapes,
            colliders_batch_capacity,
            vertex_buffers,
        )?;
        pass.memory_barrier();
        drop(pass);

        let mut pass = encoder.begin_pass("lbvh-refit-internal", timestamps.as_deref_mut());
        self.shaders.refit_internal.call(
            &mut pass,
            [1u32, num_batches, 1],
            &mut state.tree,
            num_shapes,
            colliders_batch_capacity,
        )?;
        pass.memory_barrier();
        drop(pass);

        Ok(())
    }

    /// Traverses the LBVH tree to find potentially colliding pairs.
    ///
    /// After the tree has been built with [`update_tree`](Self::update_tree), this method
    /// traverses it to identify pairs of colliders whose AABBs overlap.
    pub fn find_pairs(
        &self,
        pass: &mut GpuPass,
        state: &mut LbvhState,
        colliders_len: u32,
        num_batches: u32,
        num_shapes: &Tensor<u32>,
        colliders_batch_capacity: &Tensor<u32>,
        collision_pairs_batch_capacity: &Tensor<u32>,
        collision_pairs: &mut Tensor<[u32; 2]>,
        collision_pairs_len: &mut Tensor<u32>,
        collision_pairs_indirect: &mut Tensor<[u32; 3]>,
        collision_groups: &Tensor<crate::rapier::geometry::InteractionGroups>,
    ) -> Result<(), GpuBackendError> {
        let colliders_per_batch = colliders_len / num_batches;

        self.shaders.reset_collision_pairs.call(
            pass,
            [1u32, num_batches, 1],
            collision_pairs_len,
        )?;
        // find_collision_pairs reads collision_pairs_len (as atomic counter)
        // immediately after reset_collision_pairs zeroed it.
        pass.memory_barrier();
        self.shaders.find_collision_pairs.call(
            pass,
            [colliders_per_batch, num_batches, 1],
            &state.tree,
            collision_pairs,
            collision_pairs_len,
            num_shapes,
            colliders_batch_capacity,
            collision_pairs_batch_capacity,
            collision_groups,
        )?;
        // lbvh_init_indirect_args reads the final atomic counter written
        // by find_collision_pairs to populate collision_pairs_indirect.
        pass.memory_barrier();
        self.shaders.lbvh_init_indirect_args.call(
            pass,
            1u32,
            collision_pairs_len,
            collision_pairs_indirect,
        )?;
        pass.memory_barrier();
        Ok(())
    }
}
