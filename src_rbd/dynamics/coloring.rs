//! Graph coloring algorithms for parallel constraint solving.
//!
//! This module implements two graph coloring algorithms that enable parallel constraint solving
//! on the GPU:
//!
//! # TOPO-GC (Topological Graph Coloring)
//!
//! A fast, coloring algorithm that typically produces fewer colors and converges
//! in fewer iterations. This is the primary algorithm used by default.
//!
//! **Algorithm**: Iteratively assigns colors to constraints based on local topology. Conflicts
//! are detected and resolved in each iteration until convergence.
//!
//! **Advantages**:
//! - Fast convergence (typically < 10 iterations).
//! - Produces fewer colors (better parallelism for constraints resolution).
//!
//! **Disadvantages**:
//! - May fail to converge for highly complex constraint graphs (if too many colors are needed).
//! - Falls back to Luby if it doesn't converge within iteration limit.
//!
//! # Luby's Algorithm
//!
//! A randomized coloring algorithm used as a fallback when TOPO-GC fails or for very complex
//! constraint graphs.
//!
//! **Algorithm**: Each constraint randomly selects itself or neighbors in each iteration.
//! Selected constraints that don't conflict get the same color.
//!
//! **Advantages**:
//! - Always converges (probabilistically).
//! - Handles arbitrary constraint graphs.
//!
//! **Disadvantages**:
//! - Slower convergence.
//! - May produce more colors (less parallelism for constraints resolution).

use crate::pipeline::RunStats;
use crate::shaders::dynamics::TwoBodyConstraint;
use crate::shaders::dynamics::{
    GpuFixConflictsTopoGc, GpuResetCompletionFlagTopoGc, GpuResetLuby, GpuResetTopoGc,
    GpuStepGraphColoringLuby, GpuStepGraphColoringTopoGc,
};
use khal::Shader;
use khal::backend::{Backend, Encoder, GpuBackend, GpuBackendError, GpuPass, GpuTimestamps};
use vortx::tensor::Tensor;

/// GPU shaders for constraint graph coloring.
///
/// Contains compute pipelines for both TOPO-GC and Luby's algorithm.
#[derive(Shader)]
pub struct GpuColoring {
    /// Initializes state for Luby's algorithm.
    reset_luby_kernel: GpuResetLuby,
    /// One iteration of Luby's coloring.
    step_graph_coloring_luby_kernel: GpuStepGraphColoringLuby,
    /// Initializes state for TOPO-GC algorithm.
    reset_topo_gc_kernel: GpuResetTopoGc,
    /// One iteration of TOPO-GC coloring.
    step_graph_coloring_topo_gc_kernel: GpuStepGraphColoringTopoGc,
    /// Detects and fixes conflicts in TOPO-GC coloring.
    fix_conflicts_topo_gc_kernel: GpuFixConflictsTopoGc,
    reset_completion_flag_topo_gc: GpuResetCompletionFlagTopoGc,
}

/// Arguments for graph coloring dispatch.
///
/// Contains all GPU buffers needed by the coloring algorithms.
pub struct ColoringArgs<'a> {
    /// Indirect dispatch arguments based on contact count.
    pub contacts_len_indirect: &'a Tensor<[u32; 3]>,
    /// Number of constraints per body.
    pub body_constraint_counts: &'a Tensor<u32>,
    /// Constraint IDs associated with each body.
    pub body_constraint_ids: &'a Tensor<u32>,
    /// The constraints to be colored.
    pub constraints: &'a Tensor<TwoBodyConstraint>,
    /// Output: color assigned to each constraint.
    pub constraints_colors: &'a mut Tensor<u32>,
    /// Random values for Luby's algorithm.
    pub constraints_rands: &'a mut Tensor<u32>,
    /// Current color being assigned.
    pub curr_color: &'a mut Tensor<u32>,
    /// Count of uncolored constraints (or changed flag for TOPO-GC).
    pub uncolored: &'a mut Tensor<u32>,
    /// Staging buffer for reading uncolored count on CPU.
    pub uncolored_staging: &'a Tensor<u32>,
    /// Total number of contacts.
    pub contacts_len: &'a Tensor<u32>,
    /// Buffer tracking which constraints are colored.
    pub colored: &'a mut Tensor<u32>,
    /// Maximum contacts per batch (stride between batches in contact buffers).
    pub contacts_batch_capacity: &'a Tensor<u32>,
    /// Maximum colliders per batch (stride between batches in body buffers).
    pub colliders_batch_capacity: &'a Tensor<u32>,
    /// Per-body graph-coloring group id (multibody-aware): contacts touching
    /// different bodies of the same multibody share a group and never share
    /// a color. For free bodies, `body_group[i] = i`.
    pub body_group: &'a Tensor<u32>,
}

impl GpuColoring {
    /// Dispatches the reset_luby kernel.
    fn dispatch_reset_luby(
        &self,
        pass: &mut GpuPass,
        args: &mut ColoringArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        self.reset_luby_kernel.call(
            pass,
            args.contacts_len_indirect,
            args.constraints_colors,
            args.constraints_rands,
            args.contacts_len,
            args.contacts_batch_capacity,
        )?;
        pass.memory_barrier();
        Ok(())
    }

    /// Dispatches the step_graph_coloring_luby kernel.
    fn dispatch_step_luby(
        &self,
        pass: &mut GpuPass,
        args: &mut ColoringArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        self.step_graph_coloring_luby_kernel.call(
            pass,
            args.contacts_len_indirect,
            args.body_constraint_counts,
            args.body_constraint_ids,
            args.constraints,
            args.constraints_colors,
            args.constraints_rands,
            args.uncolored,
            args.body_group,
            args.curr_color,
            args.contacts_len,
            args.contacts_batch_capacity,
            args.colliders_batch_capacity,
        )?;
        pass.memory_barrier();
        Ok(())
    }

    /// Dispatches the reset_topo_gc kernel.
    fn dispatch_reset_topo_gc(
        &self,
        pass: &mut GpuPass,
        args: &mut ColoringArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        self.reset_topo_gc_kernel.call(
            pass,
            args.contacts_len_indirect,
            args.constraints_colors,
            args.colored,
            args.contacts_len,
            args.contacts_batch_capacity,
        )?;
        pass.memory_barrier();
        Ok(())
    }

    /// Dispatches the step_graph_coloring_topo_gc kernel.
    fn dispatch_step_topo_gc(
        &self,
        pass: &mut GpuPass,
        args: &mut ColoringArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        self.step_graph_coloring_topo_gc_kernel.call(
            pass,
            args.contacts_len_indirect,
            args.body_constraint_counts,
            args.body_constraint_ids,
            args.constraints,
            args.constraints_colors,
            args.colored,
            args.uncolored,
            args.contacts_len,
            args.body_group,
            args.contacts_batch_capacity,
            args.colliders_batch_capacity,
        )?;
        pass.memory_barrier();
        Ok(())
    }

    /// Dispatches the fix_conflicts_topo_gc kernel.
    fn dispatch_fix_conflicts_topo_gc(
        &self,
        pass: &mut GpuPass,
        args: &mut ColoringArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        self.fix_conflicts_topo_gc_kernel.call(
            pass,
            args.contacts_len_indirect,
            args.body_constraint_counts,
            args.body_constraint_ids,
            args.constraints,
            args.constraints_colors,
            args.colored,
            args.uncolored,
            args.contacts_len,
            args.body_group,
            args.contacts_batch_capacity,
            args.colliders_batch_capacity,
        )?;
        pass.memory_barrier();
        Ok(())
    }

    /// Executes Luby's randomized graph coloring algorithm.
    ///
    /// Returns the total number of colors used (1-indexed).
    pub async fn dispatch_luby<'a>(
        &self,
        backend: &GpuBackend,
        mut args: ColoringArgs<'a>,
        stats: &mut RunStats,
    ) -> u32 {
        let t0 = web_time::Instant::now();

        // Initialize coloring state
        {
            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass("luby-coloring-reset", None);
            self.dispatch_reset_luby(&mut pass, &mut args).unwrap();
            drop(pass);
            backend.submit(encoder).unwrap();
        }

        let mut num_colors = 0;
        for color in 1u32.. {
            backend
                .write_buffer(args.curr_color.buffer_mut(), 0, &[color])
                .unwrap();
            backend
                .write_buffer(args.uncolored.buffer_mut(), 0, &[0u32])
                .unwrap();

            {
                let mut encoder = backend.begin_encoding();
                let mut pass = encoder.begin_pass("luby-coloring-step", None);
                self.dispatch_step_luby(&mut pass, &mut args).unwrap();
                drop(pass);
                backend.submit(encoder).unwrap();
            }

            let uncolored = backend
                .slow_read_vec(args.uncolored.buffer())
                .await
                .unwrap()[0];

            if uncolored == 0 {
                num_colors = color + 1;
                break;
            }
        }

        stats.num_colors = num_colors;
        stats.coloring_fallback_time = t0.elapsed();
        num_colors
    }

    /// Runs a fixed number of iterations of the topo-gc coloring.
    pub fn dispatch_topo_gc_bounded<'a>(
        &self,
        pass: &mut GpuPass,
        mut args: ColoringArgs<'a>,
        max_colors: u32,
    ) -> Result<(), GpuBackendError> {
        // Reset coloring state.
        self.dispatch_reset_topo_gc(pass, &mut args)?;
        for _ in 0..max_colors {
            // reset_completion_flag reads/writes uncolored after the previous
            // step / fix_conflicts iteration may have written it.
            pass.memory_barrier();
            self.reset_completion_flag_topo_gc
                .call(pass, 1u32, args.uncolored)?;
            // step_topo_gc reads uncolored just zeroed.
            pass.memory_barrier();
            self.dispatch_step_topo_gc(pass, &mut args)?;
            // fix_conflicts reads constraints_colors written by step.
            pass.memory_barrier();
            self.dispatch_fix_conflicts_topo_gc(pass, &mut args)?;
        }

        Ok(())
    }

    /// Executes the TOPO-GC (Topological Graph Coloring) algorithm.
    ///
    /// Returns `Some(num_colors)` (1-indexed) on success, or `None` if convergence
    /// fails (caller should fall back to [`dispatch_luby`](Self::dispatch_luby)).
    pub async fn dispatch_topo_gc<'a>(
        &self,
        backend: &GpuBackend,
        mut args: ColoringArgs<'a>,
        stats: &mut RunStats,
        mut timestamps: Option<&mut GpuTimestamps>,
    ) -> Option<u32> {
        let t0 = web_time::Instant::now();

        // Initialize TOPO-GC state
        {
            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass("topo-gc-coloring-reset", timestamps.as_deref_mut());
            self.dispatch_reset_topo_gc(&mut pass, &mut args).unwrap();
            drop(pass);
            backend.submit(encoder).unwrap();
        }

        let mut num_loops = 0;
        loop {
            num_loops += 1;
            if num_loops > 64 {
                stats.coloring_time = t0.elapsed();
                return None;
            }

            // Batch multiple iterations to reduce CPU-GPU sync overhead
            {
                let mut encoder = backend.begin_encoding();
                let mut pass =
                    encoder.begin_pass("topo-gc-coloring-step", timestamps.as_deref_mut());
                for _ in 0..10 {
                    // Reset completion flag
                    self.reset_completion_flag_topo_gc
                        .call(&mut pass, 1u32, args.uncolored)
                        .unwrap();

                    // Step coloring
                    self.dispatch_step_topo_gc(&mut pass, &mut args).unwrap();

                    // Fix conflicts
                    self.dispatch_fix_conflicts_topo_gc(&mut pass, &mut args)
                        .unwrap();
                }
                drop(pass);
                backend.submit(encoder).unwrap();
            }

            let max_color = backend
                .slow_read_vec(args.uncolored.buffer())
                .await
                .unwrap()[0];

            if max_color != 0 {
                stats.coloring_time = t0.elapsed();
                stats.num_colors = max_color;
                stats.coloring_iterations = num_loops;
                return Some(max_color + 1); // NOTE: color indices are 1-based.
            }
        }
    }
}
