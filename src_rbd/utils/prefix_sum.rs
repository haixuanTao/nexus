//! GPU parallel prefix sum (scan) algorithm.
//!
//! This module implements an efficient parallel prefix sum on the GPU using a work-efficient
//! algorithm. Prefix sum is a fundamental parallel primitive used throughout the physics engine.
//!
//! # What is Prefix Sum?
//!
//! Given an input array `[a₀, a₁, a₂, ..., aₙ]`, the prefix sum produces:
//! `[0, a₀, a₀+a₁, a₀+a₁+a₂, ..., a₀+a₁+...+aₙ₋₁]`
//!
//! Note the special variant used here: a 0 is prepended as the first element, which is useful
//! for computing array indices and offsets.

use crate::shaders::utils::prefix_sum::{GpuAddDataGrp, GpuPrefixSumSweep};
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};
use khal::{BufferUsages, Shader};
use vortx::tensor::Tensor;

/// GPU shader for parallel prefix sum.
///
/// This shader implements a work-efficient parallel scan algorithm optimized for GPUs.
#[derive(Shader)]
pub struct GpuPrefixSum {
    /// Main prefix sum kernel (both up-sweep and down-sweep).
    prefix_sum_kernel: GpuPrefixSumSweep,
    /// Kernel for adding partial sums from coarser levels.
    add_data_grp_kernel: GpuAddDataGrp,
}

impl GpuPrefixSum {
    const THREADS: u32 = 256;

    /// Dispatches the prefix sum algorithm on GPU data.
    ///
    /// # Parameters
    ///
    /// - `backend`: The GPU backend
    /// - `pass`: The compute pass to record commands into
    /// - `workspace`: Workspace containing auxiliary buffers (resized automatically if needed)
    /// - `data`: Input/output buffer (modified in-place)
    ///
    /// # Panics
    ///
    /// Panics if `THREADS` is not 256, as the shared memory size is hardcoded in the shader.
    pub fn launch(
        &self,
        backend: &GpuBackend,
        pass: &mut GpuPass,
        workspace: &mut PrefixSumWorkspace,
        data: &mut Tensor<u32>,
    ) -> Result<(), GpuBackendError> {
        // If this assert fails, the kernel launches below must be changed because we are using
        // a fixed size for the shared memory currently.
        assert_eq!(
            Self::THREADS,
            256,
            "Internal error: prefix sum assumes a thread count equal to 256"
        );

        workspace.reserve(backend, data.len() as u32);

        let ngroups0 = workspace.stages[0].buffer.len() as u32;

        self.prefix_sum_kernel.call(
            pass,
            [ngroups0 * Self::THREADS, 1, 1],
            data,
            &mut workspace.stages[0].buffer,
        )?;

        for i in 0..workspace.num_stages - 1 {
            let (left, right) = workspace.stages.split_at_mut(i + 1);
            let data_stage = &mut left[i];
            let aux_stage = &mut right[0];
            let ngroups = aux_stage.buffer.len() as u32;

            self.prefix_sum_kernel.call(
                pass,
                [ngroups * Self::THREADS, 1, 1],
                &mut data_stage.buffer,
                &mut aux_stage.buffer,
            )?;
        }

        if workspace.num_stages > 2 {
            for i in (0..workspace.num_stages - 2).rev() {
                let (left, right) = workspace.stages.split_at_mut(i + 1);
                let data_stage = &mut left[i];
                let aux_stage = &right[0];
                let ngroups = aux_stage.buffer.len() as u32;

                self.add_data_grp_kernel.call(
                    pass,
                    [ngroups * Self::THREADS, 1, 1],
                    &mut data_stage.buffer,
                    &aux_stage.buffer,
                )?;
            }
        }

        if workspace.num_stages > 1 {
            self.add_data_grp_kernel.call(
                pass,
                [ngroups0 * Self::THREADS, 1, 1],
                data,
                &workspace.stages[0].buffer,
            )?;
        }

        Ok(())
    }

    /// CPU reference implementation of the prefix sum algorithm.
    ///
    /// This method computes the same result as the GPU version but on the CPU.
    /// Useful for testing and verification.
    ///
    /// # Parameters
    ///
    /// - `v`: Input/output vector (modified in-place)
    pub fn eval_cpu(&self, v: &mut [u32]) {
        for i in 0..v.len() - 1 {
            v[i + 1] += v[i];
        }

        // NOTE: we actually have a special variant of the prefix-sum
        //       where the result is as if a 0 was appended to the input vector.
        for i in (1..v.len()).rev() {
            v[i] = v[i - 1];
        }

        v[0] = 0;
    }
}

/// One stage in the multi-level prefix sum hierarchy.
struct PrefixSumStage {
    /// Maximum number of elements this stage can handle.
    capacity: u32,
    /// GPU buffer for storing partial sums at this level.
    buffer: Tensor<u32>,
}

/// Workspace containing auxiliary buffers for hierarchical prefix sum.
///
/// The workspace maintains a hierarchy of buffers for the multi-level scan algorithm.
/// It automatically resizes when the input data size changes.
#[derive(Default)]
pub struct PrefixSumWorkspace {
    stages: Vec<PrefixSumStage>,
    num_stages: usize,
}

impl PrefixSumWorkspace {
    /// Creates a new empty workspace.
    pub fn new() -> Self {
        Self {
            stages: vec![],
            num_stages: 0,
        }
    }

    /// Creates a workspace pre-allocated for a specific buffer size.
    ///
    /// # Parameters
    ///
    /// - `backend`: The GPU backend for allocating buffers
    /// - `buffer_len`: Size of the data buffer that will be scanned
    pub fn with_capacity(backend: &GpuBackend, buffer_len: u32) -> Self {
        let mut result = Self {
            stages: vec![],
            num_stages: 0,
        };
        result.reserve(backend, buffer_len);
        result
    }

    /// Ensures the workspace has sufficient capacity for a given buffer size.
    ///
    /// Resizes auxiliary buffers if needed. This is called automatically by [`GpuPrefixSum::launch`].
    ///
    /// # Parameters
    ///
    /// - `backend`: The GPU backend for allocating buffers
    /// - `buffer_len`: Size of the data buffer that will be scanned
    pub fn reserve(&mut self, backend: &GpuBackend, buffer_len: u32) {
        let mut stage_len = buffer_len.div_ceil(GpuPrefixSum::THREADS);

        if self.stages.is_empty() || self.stages[0].capacity < stage_len {
            // Reinitialize the auxiliary buffers.
            self.stages.clear();

            while stage_len != 1 {
                let buffer = Tensor::vector(
                    backend,
                    vec![0u32; stage_len as usize],
                    BufferUsages::STORAGE,
                )
                .unwrap();
                self.stages.push(PrefixSumStage {
                    capacity: stage_len,
                    buffer,
                });

                stage_len = stage_len.div_ceil(GpuPrefixSum::THREADS);
            }

            // The last stage always has only 1 element.
            self.stages.push(PrefixSumStage {
                capacity: 1,
                buffer: Tensor::vector(backend, [0u32], BufferUsages::STORAGE).unwrap(),
            });
            self.num_stages = self.stages.len();
        } else if self.stages[0].buffer.len() as u32 != stage_len {
            // The stages have big enough buffers, but we need to adjust their length.
            self.num_stages = 0;
            while stage_len != 1 {
                self.num_stages += 1;
                stage_len = stage_len.div_ceil(GpuPrefixSum::THREADS);
            }

            // The last stage always has only 1 element.
            self.num_stages += 1;
        }
    }
}
