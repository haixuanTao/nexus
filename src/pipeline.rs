use crate::fem::pipeline::FemPipeline;
use crate::mpm::pipeline::MpmPipeline;
use crate::rbd::pipeline::RbdPipeline;
use crate::state::NexusState;
use khal::backend::{GpuBackend, GpuBackendError, GpuTimestamps};

bitflags::bitflags! {
    /// A bit mask identifying nexus pipelines.
    #[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    pub struct NexusPipelineMask: u8 {
        const RBD = 1 << 0;
        const MPM = 1 << 1;
        const FEM = 1 << 2;
    }
}

#[derive(Default)]
pub struct NexusPipeline {
    pub rbd_pipeline: Option<RbdPipeline>,
    pub mpm_pipeline: Option<MpmPipeline>,
    pub fem_pipeline: Option<FemPipeline>,
}

impl NexusPipeline {
    pub fn preload_pipelines(
        &mut self,
        backend: &GpuBackend,
        pipelines: NexusPipelineMask,
    ) -> Result<(), GpuBackendError> {
        if pipelines.contains(NexusPipelineMask::RBD) && self.rbd_pipeline.is_none() {
            self.rbd_pipeline = Some(RbdPipeline::new(backend)?);
        }
        if pipelines.contains(NexusPipelineMask::MPM) && self.mpm_pipeline.is_none() {
            self.mpm_pipeline = Some(MpmPipeline::new(backend)?);
        }
        if pipelines.contains(NexusPipelineMask::FEM) && self.fem_pipeline.is_none() {
            self.fem_pipeline = Some(FemPipeline::new(backend)?);
        }
        Ok(())
    }

    /// Advances the physics simulation by one GPU timestep.
    ///
    /// The compute pipelines are compiled lazily the first time their
    /// sub-state is stepped, so the initial call is more expensive (shader
    /// compilation) than the subsequent ones.
    ///
    /// In addition, resources are loaded lazily on the GPU, so the first step
    /// after inserting/removing entities can be slower too. Call `Self::finalize`
    /// to pay that cost upfront.
    pub async fn simulate(
        &mut self,
        backend: &GpuBackend,
        state: &mut NexusState,
        mut timestamps: Option<&mut GpuTimestamps>,
    ) -> Result<(), GpuBackendError> {
        state.finalize(backend).await?;

        let t0 = web_time::Instant::now();

        // Rigid-bodies. `auto_resize_buffers` grows the collision-pair / coloring
        // buffers when the previous step overflowed them.
        if let Some(rbd) = state.rbd.as_mut() {
            self.preload_pipelines(backend, NexusPipelineMask::RBD)?;
            let pipeline = self.rbd_pipeline.as_mut().unwrap_or_else(|| unreachable!());
            let steps = state.rbd_steps_per_frame.max(1);
            for _ in 0..steps {
                state.run_stats = pipeline.step(backend, rbd, timestamps.as_deref_mut())?;
            }
            pipeline.auto_resize_buffers(backend, rbd);
        }

        // MPM continuum.
        if let Some(mpm) = state.mpm.as_mut() {
            self.preload_pipelines(backend, NexusPipelineMask::MPM)?;
            let pipeline = self.mpm_pipeline.as_mut().unwrap_or_else(|| unreachable!());

            // MPM needs many small substeps per visible frame for stability.
            // Upload the per-substep dt once, then run the substep loop.
            let substeps = state.mpm_substeps.max(1);
            let _ = mpm.write_substep_params(backend, substeps);
            for _ in 0..substeps {
                let _ = pipeline.step(backend, mpm, timestamps.as_deref_mut());
            }
        }

        // FEM soft-bodies.
        if let Some(fem) = state.fem.as_mut() {
            self.preload_pipelines(backend, NexusPipelineMask::FEM)?;
            let pipeline = self.fem_pipeline.as_mut().unwrap_or_else(|| unreachable!());

            for _ in 0..fem.num_substeps {
                let _ = pipeline.step(backend, fem, timestamps.as_deref_mut());
            }
        }

        state.run_stats.encoding_time = t0.elapsed();

        Ok(())
    }
}
