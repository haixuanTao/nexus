//! Generic, scene-agnostic rendering/runtime resources.
//!
//! [`Viewer`] is the testbed analogue of kiss3d's `Window`: it owns the window,
//! cameras, GPU backends and UI state, but knows nothing about a particular
//! physics scene. Examples build a scene from rapier resources via
//! [`Viewer::set_rbd`] / [`set_mpm`](Viewer::set_mpm) / [`set_fem`](Viewer::set_fem)
//! and then own the loop:
//!
//! ```ignore
//! let mut viewer = Viewer::new(vec![]).await;
//! let mut scene = viewer.set_rbd(state).await;
//! while viewer.render(&mut scene).await {
//!     scene.simulate(&mut viewer).await;
//! }
//! scene.detach(&mut viewer);
//! ```

use std::collections::HashMap;

use khal::backend::{GpuBackend as KhalGpuBackend, WebGpu};
use khal::re_exports::wgpu::Limits;

use kiss3d::prelude::Color;
use kiss3d::scene::{SceneNode2d, SceneNode3d};
use kiss3d::window::Window;

#[cfg(feature = "dim3")]
use kiss3d::camera::{FixedView2d, OrbitCamera3d};
#[cfg(feature = "dim2")]
use kiss3d::camera::{FixedView3d, PanZoomCamera2d};

use nexus::mpm::solver::GpuParticleModel;
use nexus::rbd::pipeline::{GpuPhysicsPipeline, RunStats};

use crate::fem::{FemScene, FemSceneBuildFn};
use crate::mpm::{self, MpmScene, MpmSceneBuildFn};
use crate::rbd::{BackendType, RbdScene, SimulationState, setup_graphics, setup_physics};
use crate::{DemoKind, RunState, Transition, UiSections};

/// UI / runtime state that is independent from the GPU/window resources. Kept in
/// its own struct so [`Viewer::render`] can split-borrow it from `window`.
pub struct UiState {
    pub run_state: RunState,
    pub run_stats: RunStats,
    pub ui_sections: UiSections,
    pub backend_type: BackendType,
    pub gpu_init_error: Option<String>,
    /// Names + kinds of all registered demos, used to populate the demo picker.
    pub demos: Vec<(String, DemoKind)>,
    /// Index (into `demos`) of the currently selected demo.
    pub selected_demo: usize,
    /// Pending loop transition requested via the UI (demo switch / quit).
    pub(crate) transition: Option<Transition>,
}

pub struct Viewer {
    window: Window,
    scene2d: SceneNode2d,
    scene3d: SceneNode3d,
    #[cfg(feature = "dim3")]
    camera3d: OrbitCamera3d,
    #[cfg(feature = "dim3")]
    camera2d: FixedView2d,
    #[cfg(feature = "dim2")]
    camera3d: FixedView3d,
    #[cfg(feature = "dim2")]
    camera2d: PanZoomCamera2d,
    webgpu: Option<KhalGpuBackend>,
    #[cfg(feature = "cuda")]
    cuda: Option<KhalGpuBackend>,
    #[cfg(feature = "metal")]
    metal: Option<KhalGpuBackend>,
    cached_gpu_pipeline: Option<GpuPhysicsPipeline>,
    pub ui: UiState,
}

impl Viewer {
    /// Creates a viewer, opening the window and probing the WebGPU backend.
    ///
    /// `demos` is the list of `(name, kind)` shown in the demo picker; pass an
    /// empty vec for a standalone single-example viewer (no picker).
    pub async fn new(demos: Vec<(String, DemoKind)>) -> Self {
        let mut window = Window::new("nexus demos").await;
        window.set_background_color(Color::new(245.0 / 255.0, 245.0 / 255.0, 236.0 / 255.0, 1.0));
        window.set_shadows_enabled(false);

        #[cfg(feature = "dim2")]
        let (camera2d, camera3d) = {
            let mut sidescroll = PanZoomCamera2d::default();
            sidescroll.look_at(glamx::Vec2::new(0.0, 100.0), 7.5);
            (sidescroll, FixedView3d::default())
        };
        #[cfg(feature = "dim3")]
        let (camera2d, camera3d) = {
            let arc_ball = OrbitCamera3d::new(
                glamx::Vec3::new(-100.0, 100.0, -100.0),
                glamx::Vec3::new(0.0, 40.0, 0.0),
            );
            (FixedView2d::default(), arc_ball)
        };

        let mut scene3d = SceneNode3d::empty();
        let scene2d = SceneNode2d::empty();

        scene3d.add_directional_light(glamx::Vec3::new(-1.0, -1.0, -1.0));
        scene3d.add_directional_light(glamx::Vec3::new(1.0, 1.0, 1.0));

        let mut viewer = Self {
            window,
            scene2d,
            scene3d,
            camera3d,
            camera2d,
            webgpu: None,
            #[cfg(feature = "cuda")]
            cuda: None,
            #[cfg(feature = "metal")]
            metal: None,
            cached_gpu_pipeline: None,
            ui: UiState {
                run_state: RunState::Paused,
                run_stats: RunStats::default(),
                ui_sections: UiSections {
                    show_examples: true,
                    show_settings: false,
                    show_performance: true,
                },
                backend_type: BackendType::Gpu,
                gpu_init_error: None,
                demos,
                selected_demo: 0,
                transition: None,
            },
        };

        // Always probe the WebGPU backend at startup so the UI knows whether to
        // offer it as an option, independent of the initial backend choice.
        viewer.webgpu = viewer.init_webgpu().await;
        viewer
    }

    pub fn with_backend(mut self, backend_type: BackendType) -> Self {
        self.ui.backend_type = backend_type;
        self
    }

    pub fn with_cpu(mut self) -> Self {
        self.ui.backend_type = BackendType::Cpu;
        self
    }

    pub fn with_running(mut self) -> Self {
        self.ui.run_state = RunState::Running;
        self
    }

    pub fn with_selected_demo(mut self, idx: usize) -> Self {
        self.ui.selected_demo = idx;
        self
    }

    pub fn selected_demo(&self) -> usize {
        self.ui.selected_demo
    }

    /// Whether the loop should stop entirely (window closed).
    pub fn quitting(&self) -> bool {
        matches!(self.ui.transition, Some(Transition::Quit))
    }

    /// Clears a pending demo-switch transition. Call between two example runs.
    pub fn clear_transition(&mut self) {
        self.ui.transition = None;
    }

    async fn init_webgpu(&mut self) -> Option<KhalGpuBackend> {
        let limits = Limits {
            max_buffer_size: 1_200_000_000,
            max_storage_buffer_binding_size: 1_200_000_000,
            #[cfg(target_arch = "wasm32")]
            max_storage_buffers_per_shader_stage: 10,
            #[cfg(not(target_arch = "wasm32"))]
            max_storage_buffers_per_shader_stage: 14,
            max_compute_workgroup_storage_size: 19904,
            ..Default::default()
        };
        match WebGpu::new(Default::default(), limits).await.map(|mut wgpu| {
            wgpu.force_buffer_copy_src = true;
            KhalGpuBackend::WebGpu(wgpu)
        }) {
            Ok(gpu) => Some(gpu),
            Err(e) => {
                self.ui.gpu_init_error = Some(format!(
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
                self.ui.gpu_init_error = Some(format!(
                    "CUDA backend not available, initialization failed:\n\"{:?}\"\n",
                    e
                ));
                None
            }
        }
    }

    #[cfg(feature = "metal")]
    fn init_metal(&mut self) -> Option<KhalGpuBackend> {
        match khal::backend::metal::Metal::new() {
            Ok(metal) => Some(KhalGpuBackend::Metal(metal)),
            Err(e) => {
                self.ui.gpu_init_error = Some(format!(
                    "Metal backend not available, initialization failed:\n\"{:?}\"\n",
                    e
                ));
                None
            }
        }
    }

    /// Lazily initializes the GPU backend matching the currently selected backend
    /// type if it has not been created yet. WebGPU is always probed at startup, so
    /// only the (synchronous) CUDA/Metal backends are created on demand here.
    fn ensure_backend_initialized(&mut self) {
        match self.ui.backend_type {
            #[cfg(feature = "cuda")]
            BackendType::Cuda if self.cuda.is_none() => {
                self.cuda = self.init_cuda();
            }
            #[cfg(feature = "metal")]
            BackendType::Metal if self.metal.is_none() => {
                self.metal = self.init_metal();
            }
            _ => {}
        }
    }

    /// Returns the active GPU backend for the current backend type, if available.
    pub fn gpu(&self) -> Option<&KhalGpuBackend> {
        match self.ui.backend_type {
            BackendType::Gpu => self.webgpu.as_ref(),
            #[cfg(feature = "cuda")]
            BackendType::Cuda => self.cuda.as_ref(),
            #[cfg(feature = "metal")]
            BackendType::Metal => self.metal.as_ref(),
            _ => None,
        }
    }

    /// Picks the khal backend for MPM/FEM scenes. `Rapier` is treated as `Cpu`
    /// so the user's Rapier choice for RBD scenes is preserved across switches.
    fn cpu_or_gpu_backend(&self, kind: &str) -> KhalGpuBackend {
        match self.ui.backend_type {
            #[cfg(feature = "cpu")]
            BackendType::Cpu | BackendType::Rapier => KhalGpuBackend::Cpu,
            #[cfg(not(feature = "cpu"))]
            BackendType::Cpu | BackendType::Rapier => {
                panic!("CPU backend not available: compile with the 'cpu' feature")
            }
            _ => self
                .gpu()
                .cloned()
                .unwrap_or_else(|| panic!("GPU required for {} demos", kind)),
        }
    }

    /// Whether the WebGPU backend is available (used by the UI backend selector).
    pub fn gpu_available(&self) -> bool {
        self.webgpu.is_some()
    }

    #[cfg(feature = "dim3")]
    pub fn set_camera(&mut self, eye: glamx::Vec3, target: glamx::Vec3) {
        self.camera3d = OrbitCamera3d::new(eye, target);
    }

    #[cfg(feature = "dim2")]
    pub fn set_camera_2d(&mut self, center: glamx::Vec2, zoom: f32) {
        self.camera2d.look_at(center, zoom);
    }

    /// Renders one frame and the UI. Returns `false` when the loop should end
    /// (window closed or a new demo selected). This is the loop condition.
    pub async fn render<S: crate::Scene>(&mut self, scene: &mut S) -> bool {
        let cont = self
            .window
            .render(
                Some(&mut self.scene3d),
                Some(&mut self.scene2d),
                Some(&mut self.camera3d),
                Some(&mut self.camera2d),
                None,
                None,
            )
            .await;

        if !cont {
            self.ui.transition = Some(Transition::Quit);
            return false;
        }

        let gpu_available = self.webgpu.is_some();
        // Disjoint closure capture (edition 2024): the closure borrows `self.ui`
        // and `scene` while `self.window` is the receiver.
        self.window.draw_ui(|ctx| {
            crate::ui::setup_custom_theme(ctx);
            crate::ui::main_panel(ctx, &mut self.ui, gpu_available, scene);
        });

        self.ui.transition.is_none()
    }

    /// Renders a few frames showing the "compiling shaders" message. Used before
    /// an RBD GPU scene is built and the pipeline must be compiled (which freezes
    /// the app for a few seconds).
    async fn maybe_show_compiling(&mut self) {
        let compiling = !matches!(self.ui.backend_type, BackendType::Rapier | BackendType::Cpu)
            && self.cached_gpu_pipeline.is_none();
        if !compiling {
            return;
        }
        for _ in 0..40 {
            self.window
                .render(
                    Some(&mut self.scene3d),
                    Some(&mut self.scene2d),
                    Some(&mut self.camera3d),
                    Some(&mut self.camera2d),
                    None,
                    None,
                )
                .await;
            crate::ui::render_compiling_message(&mut self.window);
        }
    }

    /// Builds a rigid-body scene from rapier resources.
    pub async fn set_rbd(&mut self, phys: SimulationState) -> RbdScene {
        self.ensure_backend_initialized();
        self.maybe_show_compiling().await;

        let dt = phys.environments[0].sim_params.dt;
        let num_steps_per_frame = phys.num_steps_per_frame.max(1);
        let created_backend = self.ui.backend_type;

        // Inline GPU selection so the borrow only touches the relevant fields,
        // leaving `gpu_init_error` and `cached_gpu_pipeline` free to be borrowed.
        let gpu = match created_backend {
            BackendType::Gpu => self.webgpu.as_ref(),
            #[cfg(feature = "cuda")]
            BackendType::Cuda => self.cuda.as_ref(),
            #[cfg(feature = "metal")]
            BackendType::Metal => self.metal.as_ref(),
            _ => None,
        };
        let physics = setup_physics(
            gpu,
            &phys,
            created_backend,
            &mut self.ui.gpu_init_error,
            &mut self.cached_gpu_pipeline,
        )
        .await;
        let render_ctx = setup_graphics(&mut self.scene2d, &mut self.scene3d, &phys).await;

        RbdScene {
            physics,
            render_ctx,
            sim_time: 0.0,
            dt,
            num_steps_per_frame,
            created_backend,
        }
    }

    /// Caches the GPU pipeline extracted from a finished RBD scene so the next
    /// RBD scene on the same backend can reuse it (skipping shader compilation).
    pub(crate) fn cache_pipeline(&mut self, pipeline: GpuPhysicsPipeline) {
        self.cached_gpu_pipeline = Some(pipeline);
    }

    /// Builds a MPM scene from a build function (as in the example `build`).
    pub async fn set_mpm(&mut self, build: MpmSceneBuildFn<GpuParticleModel>) -> MpmScene {
        self.ensure_backend_initialized();
        let khal = self.cpu_or_gpu_backend("MPM");
        let mut stage =
            crate::mpm::MpmStage::new(khal, |_| Box::new(()), vec![("demo".to_string(), build)])
                .await;
        // Initial readback (0 substeps) so particles are visible before the first step.
        stage.update().await;

        let mut colliders_gfx = HashMap::new();
        mpm::render_colliders(
            &mut self.scene2d,
            &mut self.scene3d,
            &stage.physics,
            &mut colliders_gfx,
        );

        #[cfg(feature = "dim2")]
        let particle_node = self.scene2d.add_rectangle(1.0, 1.0);
        #[cfg(feature = "dim3")]
        let particle_node = self.scene3d.add_cube(1.0, 1.0, 1.0);
        #[cfg(feature = "dim2")]
        let rigid_particle_node = self.scene2d.add_rectangle(1.0, 1.0);
        #[cfg(feature = "dim3")]
        let rigid_particle_node = self.scene3d.add_cube(1.0, 1.0, 1.0);

        MpmScene {
            stage,
            colliders_gfx,
            particle_node,
            rigid_particle_node,
        }
    }

    /// Builds a FEM scene from a build function.
    pub async fn set_fem(&mut self, build: FemSceneBuildFn) -> FemScene {
        self.ensure_backend_initialized();
        let khal = self.cpu_or_gpu_backend("FEM");
        let stage = crate::fem::FemStage::new(khal, vec![("demo".to_string(), build)]).await;

        #[cfg(feature = "dim2")]
        let vertex_node = self.scene2d.add_rectangle(1.0, 1.0);
        #[cfg(feature = "dim3")]
        let vertex_node = self.scene3d.add_cube(1.0, 1.0, 1.0);

        FemScene { stage, vertex_node }
    }
}
