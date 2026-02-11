//! Parallel prefix sum (scan) implementation for GPU.
//!
//! This module implements an efficient parallel prefix sum on the GPU using a work-efficient
//! algorithm. Prefix sum is a fundamental parallel primitive used throughout the MPM pipeline,
//! particularly for computing particle index offsets during grid sorting.
//!
//! # What is Prefix Sum?
//!
//! Given an input array `[a0, a1, a2, ..., an]`, the prefix sum produces:
//! `[0, a0, a0+a1, a0+a1+a2, ..., a0+a1+...+a(n-1)]`
//!
//! Note the special variant used here: a 0 is prepended as the first element, which is useful
//! for computing array indices and offsets.

use crate::mpm_shaders::grid::prefix_sum::{GpuAddDataGrp, GpuPrefixSum};
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};
use khal::{BufferUsages, Shader};
use vortx::tensor::Tensor;

/// GPU compute kernels for parallel prefix sum.
///
/// This is a special variant that produces results as if a 0 was prepended
/// to the input vector. Used for computing particle index offsets in blocks.
#[derive(Shader)]
pub struct WgPrefixSum {
    prefix_sum: GpuPrefixSum,
    add_data_grp: GpuAddDataGrp,
}

impl WgPrefixSum {
    // TODO: figure out a way to read this from the shader.
    const THREADS: u32 = 256;

    /// Computes parallel prefix sum on GPU data.
    ///
    /// Uses a multi-stage algorithm to handle arbitrary-length arrays.
    /// The result is equivalent to a CPU scan with a 0 prepended.
    ///
    /// # Arguments
    ///
    /// * `backend` - GPU backend
    /// * `pass` - Compute pass
    /// * `workspace` - Auxiliary buffers for multi-stage scan
    /// * `data` - Input/output buffer to scan (modified in-place)
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

        self.prefix_sum.call(
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

            self.prefix_sum.call(
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

                self.add_data_grp.call(
                    pass,
                    [ngroups * Self::THREADS, 1, 1],
                    &mut data_stage.buffer,
                    &aux_stage.buffer,
                )?;
            }
        }

        if workspace.num_stages > 1 {
            self.add_data_grp.call(
                pass,
                [ngroups0 * Self::THREADS, 1, 1],
                data,
                &workspace.stages[0].buffer,
            )?;
        }

        Ok(())
    }

    /// CPU reference implementation of the prefix sum for testing/validation.
    ///
    /// Applies the same algorithm as the GPU version but on CPU.
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

/// Workspace buffers for multi-stage prefix sum.
///
/// Stores auxiliary buffers needed for hierarchical scan of large arrays.
#[derive(Default)]
pub struct PrefixSumWorkspace {
    stages: Vec<PrefixSumStage>,
    num_stages: usize,
}

impl PrefixSumWorkspace {
    /// Creates an empty workspace.
    pub fn new() -> Self {
        Self {
            stages: vec![],
            num_stages: 0,
        }
    }

    /// Creates a workspace with capacity for the given buffer length.
    ///
    /// Allocates all necessary auxiliary buffers upfront.
    pub fn with_capacity(backend: &GpuBackend, buffer_len: u32) -> Self {
        let mut result = Self {
            stages: vec![],
            num_stages: 0,
        };
        result.reserve(backend, buffer_len);
        result
    }

    /// Ensures workspace has capacity for the given buffer length.
    ///
    /// Reallocates auxiliary buffers if needed.
    pub fn reserve(&mut self, backend: &GpuBackend, buffer_len: u32) {
        let mut stage_len = buffer_len.div_ceil(WgPrefixSum::THREADS);

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

                stage_len = stage_len.div_ceil(WgPrefixSum::THREADS);
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
                stage_len = stage_len.div_ceil(WgPrefixSum::THREADS);
            }

            // The last stage always has only 1 element.
            self.num_stages += 1;
        }
    }
}
