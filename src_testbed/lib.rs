#![allow(clippy::result_large_err)]
#![allow(clippy::too_many_arguments)]

#[cfg(feature = "dim2")]
pub extern crate nexus2d as nexus;
#[cfg(feature = "dim3")]
pub extern crate nexus3d as nexus;
#[cfg(feature = "dim2")]
pub extern crate rapier2d as rapier;
#[cfg(feature = "dim3")]
pub extern crate rapier3d as rapier;

pub mod mpm;
pub mod rbd;
mod ui;

use std::collections::HashMap;

use mpm::MpmStage;
use rbd::{BackendType, PhysicsBackend, RenderContext, setup_graphics, setup_physics, update_instances};
use khal::backend::{GpuBackend as KhalGpuBackend, WebGpu};
use khal::re_exports::wgpu::Limits;
use nexus::mpm::solver::GpuParticleModel;
use nexus::rbd::pipeline::{GpuPhysicsPipeline, RunStats};
use ui::{render_compiling_message, render_ui};

#[cfg(feature = "dim3")]
use kiss3d::camera::FixedView2d;
#[cfg(feature = "dim2")]
use kiss3d::camera::FixedView3d;
#[cfg(feature = "dim3")]
use kiss3d::camera::OrbitCamera3d;
#[cfg(feature = "dim2")]
use kiss3d::camera::PanZoomCamera2d;
use kiss3d::prelude::Color;
use kiss3d::scene::{SceneNode2d, SceneNode3d};
use kiss3d::window::Window;
use rapier::geometry::ColliderHandle;

pub use rbd::SimulationState;

#[cfg(feature = "dim2")]
type RenderNode = SceneNode2d;
#[cfg(feature = "dim3")]
type RenderNode = SceneNode3d;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum RunState {
    Running,
    Paused,
    Step,
}

#[derive(Copy, Clone)]
pub struct UiSections {
    pub show_examples: bool,
    pub show_settings: bool,
    pub show_performance: bool,
}

pub enum DemoBuilder {
    Rbd(&'static str, fn() -> SimulationState),
    Mpm(String, mpm::MpmSceneBuildFn<GpuParticleModel>),
}

impl DemoBuilder {
    pub fn name(&self) -> &str {
        match self {
            DemoBuilder::Rbd(name, _) => name,
            DemoBuilder::Mpm(name, _) => name.as_str(),
        }
    }
}

enum ActiveDemo {
    Rbd {
        physics: rbd::PhysicsContext,
        render_ctx: RenderContext,
    },
    Mpm {
        stage: MpmStage<GpuParticleModel>,
        colliders_gfx: HashMap<ColliderHandle, RenderNode>,
        particle_node: RenderNode,
        rigid_particle_node: RenderNode,
    },
}

pub struct Testbed {
    builders: Vec<DemoBuilder>,
    selected_demo: usize,
    ui_sections: UiSections,
    backend_type: BackendType,
    run_state: RunState,
    run_stats: RunStats,
    gpu_init_error: Option<String>,
    cached_gpu_pipeline: Option<GpuPhysicsPipeline>,
}

impl Testbed {
    pub fn from_builders(builders: Vec<DemoBuilder>) -> Self {
        Self {
            builders,
            selected_demo: 0,
            ui_sections: UiSections {
                show_examples: true,
                show_settings: false,
                show_performance: true,
            },
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
        window.set_background_color(Color::new(245.0 / 255.0, 245.0 / 255.0, 236.0 / 255.0, 1.0));

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

        scene3d.add_directional_light(glamx::Vec3::new(-1.0, -1.0, -1.0));
        scene3d.add_directional_light(glamx::Vec3::new(1.0, 1.0, 1.0));

        // Initialize GPU (shared between RBD and MPM).
        let limits = Limits {
            max_buffer_size: 1_000_000_000,
            max_storage_buffer_binding_size: 1_000_000_000,
            #[cfg(target_arch = "wasm32")]
            max_storage_buffers_per_shader_stage: 10,
            #[cfg(not(target_arch = "wasm32"))]
            max_storage_buffers_per_shader_stage: 12,
            max_compute_workgroup_storage_size: 19904,
            ..Default::default()
        };
        let gpu = match WebGpu::new(Default::default(), limits)
            .await
            .map(|mut wgpu| {
                wgpu.force_buffer_copy_src = true;
                KhalGpuBackend::WebGpu(wgpu)
            }) {
            Ok(gpu) => Some(gpu),
            Err(e) => {
                self.gpu_init_error = Some(format!(
                    "GPU backend not available, initialization failed:\n\"{}\"\n",
                    e
                ));
                self.backend_type = BackendType::Cpu;
                None
            }
        };

        // Show compiling message for RBD GPU demos if needed.
        let is_rbd = matches!(self.builders[0], DemoBuilder::Rbd(..));
        let needs_shader_compilation = is_rbd
            && matches!(self.backend_type, BackendType::Gpu { .. })
            && self.cached_gpu_pipeline.is_none();

        if needs_shader_compilation {
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

        // Create initial active demo.
        let mut active_demo = self
            .create_active_demo(
                gpu.as_ref(),
                &mut scene2d,
                &mut scene3d,
            )
            .await;

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
                &mut self.ui_sections,
                &mut self.backend_type,
                &mut self.run_state,
                &self.run_stats,
                &mut active_demo,
                gpu.as_ref(),
                &self.gpu_init_error,
            );

            if let Some(new_demo) = ui_res.new_selected_demo {
                self.selected_demo = new_demo;

                // Clean up old active demo and extract GPU pipeline for caching.
                match active_demo {
                    ActiveDemo::Rbd { physics, mut render_ctx } => {
                        render_ctx.clear();
                        if let PhysicsBackend::Gpu(gpu_backend) = physics.backend {
                            self.cached_gpu_pipeline = Some(gpu_backend.into_pipeline());
                        }
                    }
                    ActiveDemo::Mpm {
                        mut colliders_gfx,
                        mut particle_node,
                        mut rigid_particle_node,
                        ..
                    } => {
                        for (_, mut node) in colliders_gfx.drain() {
                            node.detach();
                        }
                        particle_node.detach();
                        rigid_particle_node.detach();
                    }
                }

                active_demo = self
                    .create_active_demo(
                        gpu.as_ref(),
                        &mut scene2d,
                        &mut scene3d,
                    )
                    .await;
            }

            self.step_simulation(gpu.as_ref(), &mut active_demo).await;
        }
    }

    async fn create_active_demo(
        &mut self,
        gpu: Option<&KhalGpuBackend>,
        scene2d: &mut SceneNode2d,
        scene3d: &mut SceneNode3d,
    ) -> ActiveDemo {
        match &self.builders[self.selected_demo] {
            DemoBuilder::Rbd(_, builder) => {
                let phys = builder();
                let physics = setup_physics(
                    gpu,
                    &phys,
                    self.backend_type,
                    &mut self.gpu_init_error,
                    &mut self.cached_gpu_pipeline,
                )
                .await;
                let render_ctx = setup_graphics(scene2d, scene3d, &phys).await;
                ActiveDemo::Rbd {
                    physics,
                    render_ctx,
                }
            }
            DemoBuilder::Mpm(_, _builder) => {
                let _builder = *_builder;
                let gpu = gpu.expect("GPU required for MPM demos");

                // Create a new MpmStage with just the single selected builder.
                // We pass all MPM builders so set_demo works.
                let mpm_builders: Vec<_> = self
                    .builders
                    .iter()
                    .filter_map(|b| match b {
                        DemoBuilder::Mpm(name, f) => Some((name.clone(), *f)),
                        _ => None,
                    })
                    .collect();

                // Find the index of the current MPM demo within MPM-only builders.
                let current_name = self.builders[self.selected_demo].name();
                let mpm_demo_idx = mpm_builders
                    .iter()
                    .position(|(name, _)| name == current_name)
                    .unwrap_or(0);

                // Create a temporary single-builder list for initialization,
                // then swap in the full list.
                let init_builders = vec![mpm_builders[mpm_demo_idx].clone()];
                let mut stage = MpmStage::new(
                    KhalGpuBackend::WebGpu(
                        match gpu {
                            KhalGpuBackend::WebGpu(w) => w.clone(),
                            #[allow(unreachable_patterns)]
                            _ => panic!("Expected WebGpu backend"),
                        },
                    ),
                    |_| Box::new(()),
                    init_builders,
                )
                .await;

                // Replace builders with full MPM list and fix selected index.
                stage.builders = mpm_builders;
                stage.selected_demo = mpm_demo_idx;

                // Run an initial readback (0 substeps) so particles are visible
                // before the first simulation step.
                stage.update().await;

                let mut colliders_gfx = HashMap::new();
                mpm::render_colliders(
                    scene2d,
                    scene3d,
                    &stage.physics,
                    &mut colliders_gfx,
                );

                #[cfg(feature = "dim2")]
                let particle_node = scene2d.add_rectangle(1.0, 1.0);
                #[cfg(feature = "dim3")]
                let particle_node = scene3d.add_cube(1.0, 1.0, 1.0);

                #[cfg(feature = "dim2")]
                let rigid_particle_node = scene2d.add_rectangle(1.0, 1.0);
                #[cfg(feature = "dim3")]
                let rigid_particle_node = scene3d.add_cube(1.0, 1.0, 1.0);

                ActiveDemo::Mpm {
                    stage,
                    colliders_gfx,
                    particle_node,
                    rigid_particle_node,
                }
            }
        }
    }

    async fn step_simulation(
        &mut self,
        gpu: Option<&KhalGpuBackend>,
        active_demo: &mut ActiveDemo,
    ) {
        match active_demo {
            ActiveDemo::Rbd {
                physics,
                render_ctx,
            } => {
                if self.run_state != RunState::Paused {
                    self.run_stats = physics.backend.step(gpu).await;
                }
                update_instances(render_ctx, &physics.backend);
            }
            ActiveDemo::Mpm {
                stage,
                colliders_gfx,
                particle_node,
                rigid_particle_node,
                ..
            } => {
                if self.run_state != RunState::Paused {
                    stage.update().await;
                }
                mpm::update_colliders(&stage.physics, colliders_gfx);
                particle_node.set_instances(&stage.instances);
                rigid_particle_node.set_instances(&stage.rigid_instances);
            }
        }

        if self.run_state == RunState::Step {
            self.run_state = RunState::Paused;
        }
    }
}
