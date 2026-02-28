use khal::backend::{Backend, GpuBackend as KhalGpuBackend, GpuTimestamps};
use kiss3d::prelude::*;
use nexus::fem::pipeline::{FemData, FemPipeline};

pub type FemSceneBuildFn = fn(&KhalGpuBackend) -> FemData;
pub type FemSceneBuilders = Vec<(String, FemSceneBuildFn)>;

#[derive(Default)]
pub struct FemStepTimings {
    pub total_step_time: f32,
    pub encoding_time: f32,
    pub readback_time: f32,
    pub gpu_pass_times: Vec<(String, f64)>,
    pub gpu_total_time: f64,
}

pub struct FemStage {
    pub(crate) gpu: KhalGpuBackend,
    pub(crate) selected_demo: usize,
    pub(crate) builders: FemSceneBuilders,
    pub(crate) pipeline: FemPipeline,
    pub(crate) data: FemData,
    pub(crate) timestamps: GpuTimestamps,
    pub(crate) timings: FemStepTimings,
    #[cfg(feature = "dim2")]
    pub(crate) instances: Vec<InstanceData2d>,
    #[cfg(feature = "dim3")]
    pub(crate) instances: Vec<InstanceData3d>,
}

impl FemStage {
    pub async fn new(gpu: KhalGpuBackend, builders: FemSceneBuilders) -> Self {
        let pipeline = FemPipeline::new(&gpu).unwrap();
        let data = (builders[0].1)(&gpu);
        let timestamps = GpuTimestamps::new(&gpu, 2048);

        let mut stage = Self {
            gpu,
            selected_demo: 0,
            builders,
            pipeline,
            data,
            timestamps,
            timings: FemStepTimings::default(),
            instances: vec![],
        };

        // Initial readback so vertices are visible before first step.
        stage.readback_positions().await;
        stage
    }

    pub fn set_demo(&mut self, demo_id: usize) {
        self.selected_demo = demo_id;
        self.data = (self.builders[demo_id].1)(&self.gpu);
    }

    pub async fn update(&mut self) {
        let t_total = web_time::Instant::now();
        let t_encoding = web_time::Instant::now();
        self.timestamps.reset();

        // Run substeps.
        for _ in 0..self.data.num_substeps {
            self.pipeline
                .launch_step(&mut self.gpu, &mut self.data, Some(&mut self.timestamps))
                .unwrap();
        }

        // Readback vertex positions.
        let mut encoder = self.gpu.begin_encoding();
        self.data.launch_readback(&mut encoder).unwrap();
        self.timestamps.resolve(&mut encoder);
        self.gpu.submit(encoder).unwrap();
        let t_encoding = t_encoding.elapsed().as_secs_f32() * 1000.0;

        println!("sync");
        self.gpu.synchronize().unwrap();
        let t_total_step = t_total.elapsed().as_secs_f32() * 1000.0;

        // Read timestamps.
        let (gpu_pass_times, gpu_total_time) =
            if let Ok(results) = self.timestamps.read(&self.gpu).await {
                let mut aggregated: Vec<(String, f64)> = vec![];
                for r in &results {
                    if let Some(existing) =
                        aggregated.iter_mut().find(|(label, _)| label == &r.label)
                    {
                        existing.1 += r.duration_ms;
                    } else {
                        aggregated.push((r.label.clone(), r.duration_ms));
                    }
                }
                let total = aggregated.iter().map(|e| e.1).sum();
                (aggregated, total)
            } else {
                (Vec::new(), 0.0)
            };

        // Read positions from GPU.
        let t_readback = web_time::Instant::now();
        let positions = self.data.read_positions(&self.gpu).await.unwrap();
        let t_readback = t_readback.elapsed().as_secs_f32() * 1000.0;

        self.timings = FemStepTimings {
            total_step_time: t_total_step,
            encoding_time: t_encoding,
            readback_time: t_readback,
            gpu_pass_times,
            gpu_total_time,
        };

        self.build_instances(&positions);
    }

    async fn readback_positions(&mut self) {
        let mut encoder = self.gpu.begin_encoding();
        self.data.launch_readback(&mut encoder).unwrap();
        self.gpu.submit(encoder).unwrap();
        self.gpu.synchronize().unwrap();

        let positions = self.data.read_positions(&self.gpu).await.unwrap();
        self.build_instances(&positions);
    }

    fn build_instances(&mut self, positions: &[nexus::fem::Vector]) {
        self.instances.clear();

        // Estimate vertex spacing from bounding box and vertex count.
        let point_scale = Self::estimate_point_scale(positions);

        #[cfg(feature = "dim2")]
        {
            let scale = glamx::Mat2::from_diagonal(glamx::Vec2::splat(point_scale));
            for pos in positions {
                self.instances.push(InstanceData2d {
                    position: *pos,
                    color: [0.35, 0.55, 0.82, 1.0],
                    deformation: scale,
                    ..Default::default()
                });
            }
        }

        #[cfg(feature = "dim3")]
        {
            let scale = glamx::Mat3::from_diagonal(glamx::Vec3::splat(point_scale));
            let color = Color::new(0.35, 0.55, 0.82, 1.0);
            for pos in positions {
                self.instances.push(InstanceData3d {
                    position: *pos,
                    color,
                    deformation: scale,
                    ..Default::default()
                });
            }
        }
    }

    /// Estimate a good point scale from vertex positions.
    /// Uses bounding box diagonal / num_vertices^(1/DIM) as approximate vertex spacing,
    /// then scales to ~25% of that spacing for visual clarity.
    fn estimate_point_scale(positions: &[nexus::fem::Vector]) -> f32 {
        if positions.is_empty() {
            return 0.1;
        }
        let mut lo = positions[0];
        let mut hi = positions[0];
        for &p in positions {
            lo = lo.min(p);
            hi = hi.max(p);
        }
        let extent = hi - lo;
        #[cfg(feature = "dim2")]
        let avg_extent = (extent.x + extent.y) / 2.0;
        #[cfg(feature = "dim3")]
        let avg_extent = (extent.x + extent.y + extent.z) / 3.0;

        #[cfg(feature = "dim2")]
        let n_per_side = (positions.len() as f32).sqrt();
        #[cfg(feature = "dim3")]
        let n_per_side = (positions.len() as f32).cbrt();

        let spacing = avg_extent / n_per_side.max(1.0);
        spacing * 0.25
    }
}
