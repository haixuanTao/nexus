//! Generic, scene-agnostic rendering/runtime resources.
//!
//! [`NexusViewer`] is the testbed analogue of kiss3d's `Window`: it owns the window,
//! cameras, GPU backends and UI state, but knows nothing about a particular
//! physics scene. Examples build a scene from rapier resources via
//! [`NexusViewer::set_rbd`] / [`set_mpm`](NexusViewer::set_mpm) / [`set_fem`](NexusViewer::set_fem)
//! and then own the loop:
//!
//! ```ignore
//! let mut viewer = NexusViewer::new(vec![]).await;
//! let mut scene = viewer.set_rbd(state).await;
//! while viewer.render(&mut scene).await {
//!     scene.simulate(&mut viewer).await;
//! }
//! scene.detach(&mut viewer);
//! ```

use std::collections::HashMap;

use khal::backend::{Backend, GpuBackend as KhalGpuBackend, GpuTimestamps, WebGpu};
use khal::re_exports::wgpu::Limits;

use kiss3d::prelude::Color;
use kiss3d::scene::{SceneNode2d, SceneNode3d};
use kiss3d::window::Window;

#[cfg(feature = "dim3")]
use kiss3d::camera::{FixedView2d, OrbitCamera3d};
#[cfg(feature = "dim2")]
use kiss3d::camera::{FixedView3d, PanZoomCamera2d};
use rapier::prelude::{RigidBodyHandle, SharedShape};
use nexus::mpm::solver::GpuParticleModel;
use nexus::rbd::math::Pose;
use nexus::rbd::pipeline::{RbdPipeline, RbdStats};
use nexus::state::{NexusRbdHandle, NexusState, RbdCoupling};
use crate::fem::{FemScene, FemSceneBuildFn};
use crate::mpm::{self, MpmScene, MpmSceneBuildFn};
use crate::rapier::prelude::{Collider, ColliderSet, ImpulseJointSet, RigidBody, RigidBodySet};
use crate::rbd::{
    BackendType, RbdScene, RenderContext, SimulationState, setup_physics,
};
use crate::{DemoKind, RunState, Scene, Transition, UiSections};

/// UI / runtime state that is independent from the GPU/window resources. Kept in
/// its own struct so [`NexusViewer::render`] can split-borrow it from `window`.
pub struct UiState {
    pub run_state: RunState,
    pub run_stats: RbdStats,
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

/// Minimal [`Scene`] used to draw the UI panels for a viewer-owned
/// [`NexusState`] scene, which (unlike [`RbdScene`]) has no scene object of its
/// own to hand to [`NexusViewer::render_frame`].
struct NexusSceneUi;

impl Scene for NexusSceneUi {
    fn is_rbd(&self) -> bool {
        true
    }
}

#[cfg(feature = "dim2")]
pub type SceneNode = SceneNode2d;
#[cfg(feature = "dim3")]
pub type SceneNode = SceneNode3d;

pub struct ViewerNode {
    node: SceneNode,
    instance_id: usize,
}

pub struct NexusViewer {
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
    // TODO: the backends shouldn’t be stored by the viewer.
    webgpu: Option<KhalGpuBackend>,
    #[cfg(feature = "cuda")]
    cuda: Option<KhalGpuBackend>,
    #[cfg(feature = "metal")]
    metal: Option<KhalGpuBackend>,
    /// CPU backend, stored so [`Self::backend`] can hand out a reference for the
    /// `Cpu`/`Rapier` selections. `None` when compiled without the `cpu` feature.
    cpu: Option<KhalGpuBackend>,
    // TODO: the Rbdpipeline shouldn’t be stored by the viewer.
    cached_gpu_pipeline: Option<RbdPipeline>,
    nexus_render: RenderContext,
    pub ui: UiState,
}

impl NexusViewer {
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
            cpu: {
                #[cfg(feature = "cpu")]
                {
                    Some(KhalGpuBackend::Cpu)
                }
                #[cfg(not(feature = "cpu"))]
                {
                    None
                }
            },
            cached_gpu_pipeline: None,
            nexus_render: RenderContext::new(),
            ui: UiState {
                run_state: RunState::Paused,
                run_stats: RbdStats::default(),
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

    pub fn backend(&self) -> &KhalGpuBackend {
        match self.ui.backend_type {
            BackendType::Gpu => self.webgpu.as_ref().unwrap(),
            #[cfg(feature = "cuda")]
            BackendType::Cuda => self.cuda.as_ref().unwrap(),
            #[cfg(feature = "metal")]
            BackendType::Metal => self.metal.as_ref().unwrap(),
            // Both CPU selections run the nexus pipeline on the CPU backend.
            BackendType::Cpu | BackendType::Rapier => self
                .cpu
                .as_ref()
                .expect("CPU backend unavailable: compile with the 'cpu' feature"),
            #[allow(unreachable_patterns)]
            _ => panic!("selected backend is not available in this build"),
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

    /// Ensures the GPU backend for the current backend type exists. Call once,
    /// before the demo loop, so [`Self::backend`] is usable by the examples that
    /// drive a [`NexusState`] directly.
    pub fn init_backend(&mut self) {
        self.ensure_backend_initialized();
    }

    /// Registers a render shape for a body in environment 0.
    pub fn insert_shape(&mut self, handle: RigidBodyHandle, shape: &SharedShape) {
        self.insert_shape_in(0, handle, shape)
    }

    /// Registers a render shape for a body in environment `env` (batch).
    pub fn insert_shape_in(&mut self, env: u32, handle: RigidBodyHandle, shape: &SharedShape) {
        self.nexus_render.insert_shape(
            &mut self.scene2d,
            &mut self.scene3d,
            env,
            handle,
            shape,
            Pose::IDENTITY,
        )
    }

    /// Registers a render shape with a body-local pose offset (e.g. a URDF
    /// visual mesh whose frame differs from its proxy collider).
    pub fn insert_visual_shape(
        &mut self,
        env: u32,
        handle: RigidBodyHandle,
        shape: &SharedShape,
        local_pose: Pose,
    ) {
        self.nexus_render.insert_shape(
            &mut self.scene2d,
            &mut self.scene3d,
            env,
            handle,
            shape,
            local_pose,
        )
    }

    /// Reads the latest collider poses from a [`NexusState`] back from the GPU
    /// and pushes them into the viewer-owned render instances.
    pub async fn sync(&mut self, state: &NexusState) {
        let Some(rbd) = state.rbd.as_ref() else {
            return;
        };
        let poses = rbd.poses();
        let mut cache = vec![Pose::default(); poses.len() as usize];
        let _ = self.backend().slow_read_buffer(poses.buffer(), &mut cache).await;
        self.nexus_render.update_instances_from_poses(state, &cache);
        self.ui.run_stats = state.rbd_stats.clone();
    }

    /// Tears down the viewer-owned `NexusState` render nodes. A no-op for legacy
    /// [`RbdScene`] demos (which detach their own nodes). Call between two runs.
    pub fn clear_scene(&mut self) {
        self.nexus_render.clear();
    }

    /// Whether the simulation should advance this frame, honoring the
    /// run/pause/step UI state. A pending single-step (`Step`) is consumed: this
    /// returns `true` once and then latches the run state back to `Paused`.
    ///
    /// Examples driving a [`NexusState`] gate their `simulate` call on this, the
    /// way [`RbdScene::simulate`](crate::RbdScene::simulate) does internally for
    /// legacy demos.
    pub fn simulating(&mut self) -> bool {
        match self.ui.run_state {
            RunState::Paused => false,
            RunState::Running => true,
            RunState::Step => {
                self.ui.run_state = RunState::Paused;
                true
            }
        }
    }

    /// Renders one frame of the viewer-owned `NexusState` scene and the UI.
    /// Returns `false` when the loop should end (window closed or a new demo
    /// selected). This is the no-scene-argument counterpart of [`Self::render`].
    pub async fn render_frame(&mut self) -> bool {
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
        let mut scene_ui = NexusSceneUi;
        // Disjoint closure capture (edition 2024): the closure borrows `self.ui`
        // and `scene_ui` while `self.window` is the receiver.
        self.window.draw_ui(|ctx| {
            crate::ui::setup_custom_theme(ctx);
            crate::ui::main_panel(ctx, &mut self.ui, gpu_available, &mut scene_ui);
        });

        self.ui.transition.is_none()
    }

    /// Builds a rigid-body scene from rapier resources.
    pub async fn set_rbd(&mut self, phys: SimulationState) -> RbdScene {
        todo!()
        // self.ensure_backend_initialized();
        // self.maybe_show_compiling().await;
        //
        // let dt = phys.environments[0].sim_params.dt;
        // let num_steps_per_frame = phys.num_steps_per_frame.max(1);
        // let created_backend = self.ui.backend_type;
        //
        // // Inline GPU selection so the borrow only touches the relevant fields,
        // // leaving `gpu_init_error` and `cached_gpu_pipeline` free to be borrowed.
        // let gpu = match created_backend {
        //     BackendType::Gpu => self.webgpu.as_ref(),
        //     #[cfg(feature = "cuda")]
        //     BackendType::Cuda => self.cuda.as_ref(),
        //     #[cfg(feature = "metal")]
        //     BackendType::Metal => self.metal.as_ref(),
        //     _ => None,
        // };
        // let physics = setup_physics(
        //     gpu,
        //     &phys,
        //     created_backend,
        //     &mut self.ui.gpu_init_error,
        //     &mut self.cached_gpu_pipeline,
        // )
        // .await;
        // self.nexus_render.setup_graphics(&mut self.scene2d, &mut self.scene3d, &phys);
        //
        // RbdScene {
        //     physics,
        //     render_ctx,
        //     sim_time: 0.0,
        //     dt,
        //     num_steps_per_frame,
        //     created_backend,
        // }
    }

    /// Caches the GPU pipeline extracted from a finished RBD scene so the next
    /// RBD scene on the same backend can reuse it (skipping shader compilation).
    pub(crate) fn cache_pipeline(&mut self, pipeline: RbdPipeline) {
        self.cached_gpu_pipeline = Some(pipeline);
    }

    /// Builds a MPM scene from a build function (as in the example `build`).
    pub async fn set_mpm(&mut self, build: MpmSceneBuildFn) -> MpmScene {
        self.ensure_backend_initialized();
        let khal = self.cpu_or_gpu_backend("MPM");
        let mut stage =
            crate::mpm::MpmStage::new(khal, vec![("demo".to_string(), build)])
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
