#![allow(clippy::result_large_err)]
#![allow(clippy::too_many_arguments)]

#[cfg(feature = "dim2")]
use nexus2d as nexus;
#[cfg(feature = "dim3")]
use nexus3d as nexus;
#[cfg(feature = "dim2")]
use rapier2d as rapier;
#[cfg(feature = "dim3")]
use rapier3d as rapier;

mod backend;
mod graphics;
mod ui;

use backend::{BackendType, CpuBackend, GpuBackend, PhysicsBackend};
use graphics::{RenderContext, setup_graphics, update_instances};
use khal::backend::{GpuBackend as KhalGpuBackend, WebGpu};
use khal::re_exports::wgpu::Limits;
use nexus::pipeline::{GpuPhysicsPipeline, RunStats};
use ui::{PhysicsContext, RunState, render_compiling_message, render_ui};

#[cfg(feature = "dim3")]
use kiss3d::camera::FixedView2d;
#[cfg(feature = "dim2")]
use kiss3d::camera::FixedView3d;
#[cfg(feature = "dim3")]
use kiss3d::camera::OrbitCamera3d;
#[cfg(feature = "dim2")]
use kiss3d::camera::PanZoomCamera2d;
use kiss3d::scene::{SceneNode2d, SceneNode3d};
use kiss3d::window::Window;
use rapier::geometry::ColliderSet;
use rapier::prelude::{ImpulseJointSet, RigidBodySet};

pub struct SimulationState {
    pub bodies: RigidBodySet,
    pub colliders: ColliderSet,
    pub impulse_joints: ImpulseJointSet,
}

pub type SimulationBuilders = Vec<(&'static str, fn() -> SimulationState)>;

pub struct Testbed {
    builders: SimulationBuilders,
    selected_demo: usize,
    backend_type: BackendType,
    run_state: RunState,
    run_stats: RunStats,
    gpu_init_error: Option<String>,
    /// Cached GPU pipeline to avoid recompilation when switching demos
    cached_gpu_pipeline: Option<GpuPhysicsPipeline>,
}

impl Testbed {
    pub fn from_builders(builders: SimulationBuilders) -> Self {
        Self {
            builders,
            selected_demo: 0,
            backend_type: BackendType::Gpu,
            run_state: RunState::Paused,
            run_stats: RunStats::default(),
            gpu_init_error: None,
            cached_gpu_pipeline: None,
        }
    }

    pub fn with_backend(mut self, backend_type: BackendType) -> Self {
        self.backend_type = backend_type;
        self
    }

    pub async fn run(mut self) {
        let mut window = Window::new("nexus demos").await;

        // Set up cameras first so we can render the "compiling" message
        #[cfg(feature = "dim2")]
        let (mut camera2d, mut camera3d) = {
            let mut sidescroll = PanZoomCamera2d::default();
            sidescroll.look_at(glamx::Vec2::new(0.0, 100.0), 7.5);
            (sidescroll, FixedView3d::default())
        };
        #[cfg(feature = "dim3")]
        let (mut camera2d, mut camera3d) = {
            let arc_ball = OrbitCamera3d::new(
                glamx::Vec3::new(-100.0, 100.0, -100.0),
                glamx::Vec3::new(0.0, 40.0, 0.0),
            );
            (FixedView2d::default(), arc_ball)
        };

        let mut scene3d = SceneNode3d::empty();
        let mut scene2d = SceneNode2d::empty();

        // Add a default light to the 3D scene
        scene3d.add_directional_light(glamx::Vec3::new(-1.0, -1.0, -1.0));
        scene3d.add_directional_light(glamx::Vec3::new(1.0, 1.0, 1.0));

        // Try to initialize GPU, fallback to CPU if it fails
        let limits = Limits {
            max_buffer_size: 600_000_000,
            max_storage_buffer_binding_size: 600_000_000,
            max_storage_buffers_per_shader_stage: 10, // For narrow phase.
            ..Default::default()
        };
        let gpu = match WebGpu::new(Default::default(), limits)
            .await
            .map(|mut wgpu| {
                wgpu.force_buffer_copy_src = true; // TODO: this is for debugging, remove later.
                KhalGpuBackend::WebGpu(wgpu)
            }) {
            Ok(gpu) => Some(gpu),
            Err(e) => {
                // GPU initialization failed, force CPU backend
                self.gpu_init_error = Some(format!(
                    "GPU backend not available, initialization failed:\n\"{}\"\n",
                    e
                ));
                self.backend_type = BackendType::Cpu;
                None
            }
        };

        // Check if we need to compile shaders (GPU backend without cached pipeline).
        let needs_shader_compilation = matches!(self.backend_type, BackendType::Gpu { .. })
            && self.cached_gpu_pipeline.is_none();

        // Render a "compiling shaders" message before doing the actual compilation.
        // The app will freeze during the compilation, so we need to draw this before.
        if needs_shader_compilation {
            // Don't run a single render pass. It can take a few frames for the window/canvas to
            // show up so we don't want the app to freeze before the message is actually visible.
            for _ in 0..100 {
                window
                    .render(
                        Some(&mut scene3d),
                        Some(&mut scene2d),
                        Some(&mut camera3d),
                        Some(&mut camera2d),
                        None,
                        None,
                    )
                    .await;
                render_compiling_message(&mut window);
            }
        }

        let phys = (self.builders[0].1)();
        let mut physics = setup_physics(
            gpu.as_ref(),
            &phys,
            self.backend_type,
            &mut self.gpu_init_error,
            &mut self.cached_gpu_pipeline,
        )
        .await;

        let mut render_ctx = setup_graphics(&mut scene2d, &mut scene3d, &phys).await;

        while window
            .render(
                Some(&mut scene3d),
                Some(&mut scene2d),
                Some(&mut camera3d),
                Some(&mut camera2d),
                None,
                None,
            )
            .await
        {
            let ui_res = render_ui(
                &mut window,
                &self.builders,
                &mut self.selected_demo,
                &mut self.backend_type,
                &mut self.run_state,
                &self.run_stats,
                &mut physics,
                gpu.as_ref(),
                &self.gpu_init_error,
            );

            if let Some(new_demo) = ui_res.new_selected_demo {
                self.selected_demo = new_demo;
                let phys = (self.builders[new_demo].1)();
                render_ctx.clear();

                // Extract pipeline from current GPU backend if present
                if let PhysicsBackend::Gpu(gpu_backend) = physics.backend {
                    self.cached_gpu_pipeline = Some(gpu_backend.into_pipeline());
                }

                physics = setup_physics(
                    gpu.as_ref(),
                    &phys,
                    self.backend_type,
                    &mut self.gpu_init_error,
                    &mut self.cached_gpu_pipeline,
                )
                .await;

                render_ctx = setup_graphics(&mut scene2d, &mut scene3d, &phys).await;
            }

            self.step_simulation(gpu.as_ref(), &mut physics, &mut render_ctx)
                .await;
        }
    }

    async fn step_simulation(
        &mut self,
        gpu: Option<&KhalGpuBackend>,
        physics: &mut PhysicsContext,
        render_ctx: &mut RenderContext,
    ) {
        if self.run_state != RunState::Paused {
            self.run_stats = physics.backend.step(gpu).await;
        }

        if self.run_state == RunState::Step {
            self.run_state = RunState::Paused;
        }

        // Update instances using set_instances for efficient rendering
        update_instances(render_ctx, &physics.backend);
    }
}

async fn setup_physics(
    gpu: Option<&KhalGpuBackend>,
    phys: &SimulationState,
    backend_type: BackendType,
    gpu_error: &mut Option<String>,
    cached_pipeline: &mut Option<GpuPhysicsPipeline>,
) -> PhysicsContext {
    let backend = match backend_type {
        BackendType::Gpu => {
            // Try to create GPU backend, fallback to CPU if it fails
            let gpu = gpu.unwrap();

            // Try to reuse cached pipeline or create a new one
            if let Some(pipeline) = cached_pipeline.take() {
                // Fast path: reuse existing pipeline
                let gpu_backend = GpuBackend::with_pipeline(gpu, pipeline, phys).await;
                PhysicsBackend::Gpu(gpu_backend)
            } else {
                // Slow path: compile shaders for the first time
                match GpuBackend::try_new(gpu, phys).await {
                    Ok(gpu_backend) => PhysicsBackend::Gpu(gpu_backend),
                    Err(e) => {
                        // GPU backend creation failed, fallback to CPU
                        *gpu_error = Some(format!(
                            "GPU backend initialization failed: {}. Using CPU backend.",
                            e
                        ));
                        PhysicsBackend::Cpu(CpuBackend::new(SimulationState {
                            bodies: phys.bodies.clone(),
                            colliders: phys.colliders.clone(),
                            impulse_joints: phys.impulse_joints.clone(),
                        }))
                    }
                }
            }
        }
        BackendType::Cpu => PhysicsBackend::Cpu(CpuBackend::new(SimulationState {
            bodies: phys.bodies.clone(),
            colliders: phys.colliders.clone(),
            impulse_joints: phys.impulse_joints.clone(),
        })),
    };

    PhysicsContext::new(backend)
}
