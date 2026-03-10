//! Warmstarting mechanism for constraint solver temporal coherence.
//!
//! Warmstarting reuses impulses from the previous simulation frame to initialize the
//! constraint solver, significantly improving convergence speed and stability. This exploits
//! temporal coherence - the observation that adjacent frames in a simulation tend to have
//! similar contact configurations and required impulses.
//!
//! # How It Works
//!
//! 1. After solving constraints in frame N, impulse accumulators are stored.
//! 2. In frame N+1, new contacts are matched against old contacts.
//! 3. Matching contacts inherit their previous impulses as starting guesses.
//! 4. The solver converges faster since it starts closer to the solution.

use crate::shaders::dynamics::{
    GpuTransferWarmstartImpulses, TwoBodyConstraint, TwoBodyConstraintBuilder,
};
use khal::Shader;
use khal::backend::{GpuBackendError, GpuPass};
use vortx::tensor::Tensor;

/// GPU shader for transferring warmstart impulses between frames.
///
/// This shader matches new contacts against old contacts and transfers impulse
/// accumulators when a match is found.
#[derive(Shader)]
pub struct GpuWarmstart {
    /// Compute pipeline that matches contacts and transfers impulses.
    transfer_warmstart_impulses_kernel: GpuTransferWarmstartImpulses,
}

/// Arguments for warmstart dispatch.
///
/// Contains buffers for both old (previous frame) and new (current frame) constraint data.
pub struct WarmstartArgs<'a> {
    /// Number of contacts in current frame.
    pub contacts_len: &'a Tensor<u32>,
    /// Constraint counts per body from previous frame.
    pub old_body_constraint_counts: &'a Tensor<u32>,
    /// Constraint IDs per body from previous frame.
    pub old_body_constraint_ids: &'a Tensor<u32>,
    /// Solver constraints from previous frame.
    pub old_constraints: &'a Tensor<TwoBodyConstraint>,
    /// Constraint builders from previous frame.
    pub old_constraint_builders: &'a Tensor<TwoBodyConstraintBuilder>,
    /// Solver constraints for current frame (to be warmstarted).
    pub new_constraints: &'a mut Tensor<TwoBodyConstraint>,
    /// Constraint builders for current frame.
    pub new_constraint_builders: &'a Tensor<TwoBodyConstraintBuilder>,
    /// Indirect dispatch arguments based on contact count.
    pub contacts_len_indirect: &'a Tensor<[u32; 3]>,
    /// Maximum contacts per batch (stride between batches in contact buffers).
    pub contacts_batch_capacity: &'a Tensor<u32>,
    /// Maximum colliders per batch (stride between batches in body buffers).
    pub colliders_batch_capacity: &'a Tensor<u32>,
}

impl GpuWarmstart {
    /// Transfers warmstart impulses from old constraints to new constraints.
    ///
    /// This method dispatches a compute shader that searches for matching contacts
    /// between the previous and current frames. When a match is found, the impulse
    /// accumulator from the old contact is copied to the new contact.
    ///
    /// # Parameters
    ///
    /// - `pass`: The compute pass to record commands into.
    /// - `args`: Warmstart arguments containing old and new constraint buffers.
    pub fn transfer_warmstart_impulses<'a>(
        &self,
        pass: &mut GpuPass,
        args: WarmstartArgs<'a>,
    ) -> Result<(), GpuBackendError> {
        self.transfer_warmstart_impulses_kernel.call(
            pass,
            args.contacts_len_indirect,
            args.old_body_constraint_counts,
            args.old_body_constraint_ids,
            args.old_constraints,
            args.old_constraint_builders,
            args.new_constraints,
            args.new_constraint_builders,
            args.contacts_len,
            args.contacts_batch_capacity,
            args.colliders_batch_capacity,
        )
    }
}
