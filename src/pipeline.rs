#[cfg(feature = "fem")]
use crate::fem::pipeline::FemPipeline;
#[cfg(feature = "mpm")]
use crate::mpm::pipeline::MpmPipeline;
#[cfg(feature = "rbd")]
use crate::rbd::pipeline::RbdPipeline;
use crate::state::NexusState;
use khal::backend::{GpuBackend, GpuBackendError, GpuTimestamps};

/// A multiphysics pipeline combining all the supported
/// physics modelings (with coupling whne supported).
pub struct NexusPipeline {
    #[cfg(feature = "rbd")]
    rbd: RbdPipeline,
    #[cfg(feature = "mpm")]
    mpm: MpmPipeline,
    #[cfg(feature = "fem")]
    fem: FemPipeline,
}

impl NexusPipeline {
    pub async fn step(&self, backend: &GpuBackend, state: &mut NexusState, mut timestamps: Option<&mut GpuTimestamps>)
        -> Result<(), GpuBackendError>{
        #[cfg(feature = "rbd")]
        if let Some(rbd) = state.rbd.as_mut() {
            self.rbd.step(backend, rbd, timestamps.as_deref_mut())?;
        }
        #[cfg(feature = "mpm")]
        if let Some(mpm) = state.mpm.as_mut() {
            self.mpm.step(backend, mpm, timestamps.as_deref_mut())?;
        }
        #[cfg(feature = "fem")]
        if let Some(fem) = state.fem.as_mut() {
            self.fem.step(backend, fem, timestamps)?;
        }

        Ok(())
    }
}