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

pub mod fem;
pub mod mpm;
pub mod rbd;
mod ui;

use std::collections::HashMap;

use fem::FemStage;
use khal::backend::{GpuBackend as KhalGpuBackend, WebGpu};
use khal::re_exports::wgpu::Limits;
use mpm::MpmStage;
use nexus::mpm::solver::GpuParticleModel;
use nexus::rbd::pipeline::{GpuPhysicsPipeline, RunStats};
pub use rbd::BackendType;
use rbd::{PhysicsBackend, RenderContext, setup_graphics, setup_physics, update_instances};
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

pub use rbd::{BatchEnvironment, SimulationState};

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

/// Initial camera position for a demo.
#[derive(Clone, Copy)]
pub struct CameraSetup {
    pub eye: glamx::Vec3,
    pub target: glamx::Vec3,
}

pub enum DemoBuilder {
    Rbd(&'static str, fn() -> SimulationState, Option<CameraSetup>),
    Mpm(
        String,
        mpm::MpmSceneBuildFn<GpuParticleModel>,
        Option<CameraSetup>,
    ),
    Fem(String, fem::FemSceneBuildFn, Option<CameraSetup>),
}

impl DemoBuilder {
    pub fn rbd(name: &'static str, f: fn() -> SimulationState) -> Self {
        Self::Rbd(name, f, None)
    }

    pub fn mpm(name: impl Into<String>, f: mpm::MpmSceneBuildFn<GpuParticleModel>) -> Self {
        Self::Mpm(name.into(), f, None)
    }

    pub fn fem(name: impl Into<String>, f: fem::FemSceneBuildFn) -> Self {
        Self::Fem(name.into(), f, None)
    }

    pub fn with_camera(mut self, eye: glamx::Vec3, target: glamx::Vec3) -> Self {
        match &mut self {
            Self::Rbd(_, _, c) | Self::Mpm(_, _, c) | Self::Fem(_, _, c) => {
                *c = Some(CameraSetup { eye, target });
            }
        }
        self
    }

    pub fn name(&self) -> &str {
        match self {
            DemoBuilder::Rbd(name, ..) => name,
            DemoBuilder::Mpm(name, ..) => name.as_str(),
            DemoBuilder::Fem(name, ..) => name.as_str(),
        }
    }

    pub fn camera(&self) -> Option<&CameraSetup> {
        match self {
            DemoBuilder::Rbd(_, _, c) | DemoBuilder::Mpm(_, _, c) | DemoBuilder::Fem(_, _, c) => {
                c.as_ref()
            }
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
    Fem {
        stage: FemStage,
        vertex_node: RenderNode,
    },
}

pub struct Testbed {
    builders: Vec<DemoBuilder>,
    selected_demo: usize,
    ui_sections: UiSections,
    backend_type: BackendType,
    prev_backend_type: BackendType,
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
            prev_backend_type: BackendType::Gpu,
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

    pub fn with_cpu(mut self) -> Self {
        self.backend_type = BackendType::Cpu;
        self
    }

    async fn init_webgpu(&mut self) -> Option<KhalGpuBackend> {
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
        match WebGpu::new(Default::default(), limits)
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
                None
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn init_cuda(&mut self) -> Option<KhalGpuBackend> {
        match khal::backend::cuda::Cuda::new(0) {
            Ok(cuda) => Some(KhalGpuBackend::Cuda(cuda)),
            Err(e) => {
                self.gpu_init_error = Some(format!(
                    "CUDA backend not available, initialization failed:\n\"{:?}\"\n",
                    e
                ));
                None
            }
        }
    }

    pub fn with_running(mut self) -> Self {
        self.run_state = RunState::Running;
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

        // Always probe the WebGPU backend at startup so the UI knows whether
        // to offer it as an option, independent of the initial backend choice.
        // Other backends (CUDA) are still lazily initialized on demand.
        let mut webgpu: Option<KhalGpuBackend> = self.init_webgpu().await;

        #[cfg(feature = "cuda")]
        let mut cuda: Option<KhalGpuBackend> = if matches!(self.backend_type, BackendType::Cuda) {
            self.init_cuda()
        } else {
            None
        };

        /// Returns the active GPU backend based on the given backend type.
        fn pick_gpu<'a>(
            backend_type: BackendType,
            webgpu: &'a Option<KhalGpuBackend>,
            #[cfg(feature = "cuda")] cuda: &'a Option<KhalGpuBackend>,
        ) -> Option<&'a KhalGpuBackend> {
            match backend_type {
                BackendType::Gpu => webgpu.as_ref(),
                #[cfg(feature = "cuda")]
                BackendType::Cuda => cuda.as_ref(),
                _ => None,
            }
        }

        // Show compiling message for RBD GPU demos if needed.
        let is_rbd = matches!(self.builders[0], DemoBuilder::Rbd(..));
        let needs_shader_compilation = is_rbd
            && matches!(self.backend_type, BackendType::Gpu)
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
                {
                    let bt = self.backend_type;
                    pick_gpu(
                        bt,
                        &webgpu,
                        #[cfg(feature = "cuda")]
                        &cuda,
                    )
                },
                &mut scene2d,
                &mut scene3d,
            )
            .await;

        // Apply initial camera if the demo specifies one.
        #[cfg(feature = "dim3")]
        if let Some(cam) = self.builders[self.selected_demo].camera() {
            camera3d = OrbitCamera3d::new(cam.eye, cam.target);
        }

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
            // The UI uses `webgpu` directly (not `pick_gpu`) to decide whether
            // to offer the GPU radio: availability must not depend on the
            // currently-selected backend.
            let ui_res = render_ui(
                &mut window,
                &self.builders,
                &mut self.selected_demo,
                &mut self.ui_sections,
                &mut self.backend_type,
                &mut self.run_state,
                &self.run_stats,
                &mut active_demo,
                webgpu.as_ref(),
                &self.gpu_init_error,
            );

            if let Some(new_demo) = ui_res.new_selected_demo {
                self.selected_demo = new_demo;

                // Invalidate cached pipeline if backend type changed.
                let backend_changed = self.backend_type != self.prev_backend_type;
                self.prev_backend_type = self.backend_type;

                // Lazily initialize the GPU backend if switching to one that hasn't been created yet.
                if backend_changed {
                    match self.backend_type {
                        BackendType::Gpu if webgpu.is_none() => {
                            webgpu = self.init_webgpu().await;
                        }
                        #[cfg(feature = "cuda")]
                        BackendType::Cuda if cuda.is_none() => {
                            cuda = self.init_cuda();
                        }
                        _ => {}
                    }
                }

                // Clean up old active demo and extract GPU pipeline for caching.
                match active_demo {
                    ActiveDemo::Rbd {
                        physics,
                        mut render_ctx,
                    } => {
                        render_ctx.clear();
                        if !backend_changed {
                            if let PhysicsBackend::Gpu(gpu_backend) = physics.backend {
                                self.cached_gpu_pipeline = Some(gpu_backend.into_pipeline());
                            }
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
                    ActiveDemo::Fem {
                        mut vertex_node, ..
                    } => {
                        vertex_node.detach();
                    }
                }

                active_demo = self
                    .create_active_demo(
                        {
                            let bt = self.backend_type;
                            pick_gpu(
                                bt,
                                &webgpu,
                                #[cfg(feature = "cuda")]
                                &cuda,
                            )
                        },
                        &mut scene2d,
                        &mut scene3d,
                    )
                    .await;

                // Apply camera if the new demo specifies one.
                #[cfg(feature = "dim3")]
                if let Some(cam) = self.builders[self.selected_demo].camera() {
                    camera3d = OrbitCamera3d::new(cam.eye, cam.target);
                }
            }

            self.step_simulation(
                {
                    let bt = self.backend_type;
                    pick_gpu(
                        bt,
                        &webgpu,
                        #[cfg(feature = "cuda")]
                        &cuda,
                    )
                },
                &mut active_demo,
            )
            .await;
        }
    }

    async fn create_active_demo(
        &mut self,
        gpu: Option<&KhalGpuBackend>,
        scene2d: &mut SceneNode2d,
        scene3d: &mut SceneNode3d,
    ) -> ActiveDemo {
        /// Pick the khal backend for MPM/FEM demos based on the current backend
        /// selection. `Rapier` is treated as `Cpu` so the user's Rapier choice
        /// for RBD demos is preserved across demo switches.
        fn cpu_or_gpu_backend(
            kind: &str,
            backend_type: BackendType,
            gpu: Option<&KhalGpuBackend>,
        ) -> KhalGpuBackend {
            match backend_type {
                #[cfg(feature = "cpu")]
                BackendType::Cpu | BackendType::Rapier => KhalGpuBackend::Cpu,
                #[cfg(not(feature = "cpu"))]
                BackendType::Cpu | BackendType::Rapier => {
                    panic!("CPU backend not available: compile with the 'cpu' feature")
                }
                _ => gpu
                    .cloned()
                    .unwrap_or_else(|| panic!("GPU required for {} demos", kind)),
            }
        }

        match &self.builders[self.selected_demo] {
            DemoBuilder::Rbd(_, builder, _) => {
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
            DemoBuilder::Mpm(_, _, _) => {
                let khal_backend = cpu_or_gpu_backend("MPM", self.backend_type, gpu);

                // Create a new MpmStage with just the single selected builder.
                // We pass all MPM builders so set_demo works.
                let mpm_builders: Vec<_> = self
                    .builders
                    .iter()
                    .filter_map(|b| match b {
                        DemoBuilder::Mpm(name, f, _) => Some((name.clone(), *f)),
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
                let mut stage = MpmStage::new(khal_backend, |_| Box::new(()), init_builders).await;

                // Replace builders with full MPM list and fix selected index.
                stage.builders = mpm_builders;
                stage.selected_demo = mpm_demo_idx;

                // Run an initial readback (0 substeps) so particles are visible
                // before the first simulation step.
                stage.update().await;

                let mut colliders_gfx = HashMap::new();
                mpm::render_colliders(scene2d, scene3d, &stage.physics, &mut colliders_gfx);

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
            DemoBuilder::Fem(_, _, _) => {
                let khal_backend = cpu_or_gpu_backend("FEM", self.backend_type, gpu);

                let fem_builders: Vec<_> = self
                    .builders
                    .iter()
                    .filter_map(|b| match b {
                        DemoBuilder::Fem(name, f, _) => Some((name.clone(), *f)),
                        _ => None,
                    })
                    .collect();

                let current_name = self.builders[self.selected_demo].name();
                let fem_demo_idx = fem_builders
                    .iter()
                    .position(|(name, _)| name == current_name)
                    .unwrap_or(0);

                let init_builders = vec![fem_builders[fem_demo_idx].clone()];
                let mut stage = FemStage::new(khal_backend, init_builders).await;

                stage.builders = fem_builders;
                stage.selected_demo = fem_demo_idx;

                #[cfg(feature = "dim2")]
                let vertex_node = scene2d.add_rectangle(1.0, 1.0);
                #[cfg(feature = "dim3")]
                let vertex_node = scene3d.add_cube(1.0, 1.0, 1.0);

                ActiveDemo::Fem { stage, vertex_node }
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
            ActiveDemo::Fem { stage, vertex_node } => {
                if self.run_state != RunState::Paused {
                    stage.update().await;
                }
                vertex_node.set_instances(&stage.instances);
            }
        }

        if self.run_state == RunState::Step {
            self.run_state = RunState::Paused;
        }
    }
}
