use super::SimulationBackend;
use crate::SimulationState;
use khal::backend::{Backend, GpuBackend as KhalGpuBackend};
use nexus::rbd::math::Pose;
use nexus::rbd::pipeline::{GpuPhysicsPipeline, GpuPhysicsState, RunStats};

/// GPU-based physics backend using nexus
pub struct GpuBackend {
    pipeline: GpuPhysicsPipeline,
    state: GpuPhysicsState,
    poses_cache: Vec<Pose>,
}

impl GpuBackend {
    /// Reads poses from GPU buffer, handling dimension-specific conversion.
    async fn read_poses(
        gpu: &KhalGpuBackend,
        state: &GpuPhysicsState,
    ) -> Result<Vec<Pose>, String> {
        gpu.slow_read_vec(state.poses().buffer())
            .await
            .map_err(|e| format!("Failed to read poses: {:?}", e))
    }

    /// Reads poses into an existing buffer, handling dimension-specific conversion.
    async fn read_poses_into(
        gpu: &KhalGpuBackend,
        state: &GpuPhysicsState,
        poses_cache: &mut Vec<Pose>,
    ) {
        poses_cache.resize(state.poses().len() as usize, Pose::default());
        let _ = gpu
            .slow_read_buffer(state.poses().buffer(), poses_cache)
            .await;
    }

    /// Attempts to create a new GPU backend, returning an error if initialization fails.
    ///
    /// This method can fail if:
    /// - Shader compilation fails
    /// - GPU device doesn't support required features
    /// - Memory allocation fails
    pub async fn try_new(gpu: &KhalGpuBackend, phys: &SimulationState) -> Result<Self, String> {
        let pipeline = GpuPhysicsPipeline::from_backend(gpu);
        let state =
            GpuPhysicsState::from_rapier(gpu, &phys.bodies, &phys.colliders, &phys.impulse_joints);
        let poses_cache = Self::read_poses(gpu, &state).await?;

        Ok(Self {
            pipeline,
            state,
            poses_cache,
        })
    }

    /// Creates a new GPU backend with a pre-compiled pipeline.
    ///
    /// This is faster than [`try_new`](Self::try_new) when switching demos because
    /// it reuses the existing pipeline instead of recompiling shaders.
    pub async fn with_pipeline(
        gpu: &KhalGpuBackend,
        pipeline: GpuPhysicsPipeline,
        phys: &SimulationState,
    ) -> Self {
        let state =
            GpuPhysicsState::from_rapier(gpu, &phys.bodies, &phys.colliders, &phys.impulse_joints);
        let poses_cache = Self::read_poses(gpu, &state).await.unwrap_or_default();

        Self {
            pipeline,
            state,
            poses_cache,
        }
    }

    /// Extracts the pipeline from this backend, consuming it.
    ///
    /// Useful for reusing the pipeline when switching demos.
    pub fn into_pipeline(self) -> GpuPhysicsPipeline {
        self.pipeline
    }

    /// Creates a new GPU backend, panicking if initialization fails.
    ///
    /// Use [`try_new`](Self::try_new) for error handling.
    #[allow(dead_code)]
    pub async fn new(gpu: &KhalGpuBackend, phys: &SimulationState) -> Self {
        Self::try_new(gpu, phys).await.unwrap()
    }
}

impl SimulationBackend for GpuBackend {
    fn poses(&self) -> &[Pose] {
        &self.poses_cache
    }
    fn num_bodies(&self) -> usize {
        self.poses_cache.len()
    }
    fn num_joints(&self) -> usize {
        self.state.joints().len()
    }

    async fn step(&mut self, gpu: Option<&KhalGpuBackend>) -> RunStats {
        let gpu = gpu.unwrap();

        let t0 = web_time::Instant::now();
        let mut run_stats = self.pipeline.step(gpu, &mut self.state).await;

        // Read back poses from GPU
        Self::read_poses_into(gpu, &self.state, &mut self.poses_cache).await;

        run_stats.total_simulation_time_with_readback = t0.elapsed();

        run_stats
    }
}
