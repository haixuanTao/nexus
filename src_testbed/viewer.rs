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
#[cfg(feature = "dim2")]
use kiss3d::scene::InstanceData2d;
#[cfg(feature = "dim3")]
use kiss3d::scene::InstanceData3d;
use kiss3d::window::Window;

/// Viewer-owned scene node type for the active dimension.
#[cfg(feature = "dim2")]
type SceneNodeX = SceneNode2d;
#[cfg(feature = "dim3")]
type SceneNodeX = SceneNode3d;

#[cfg(feature = "dim3")]
use kiss3d::camera::{FixedView2d, OrbitCamera3d};
#[cfg(feature = "dim2")]
use kiss3d::camera::{FixedView3d, PanZoomCamera2d};
use rapier::prelude::{RigidBodyHandle, SharedShape};
use nexus::mpm::solver::GpuParticleModel;
use nexus::rbd::math::{Pose, Vector};
use nexus::rbd::pipeline::{RbdPipeline, RunStats};
use nexus::state::{NexusCounts, NexusRbdHandle, NexusState, RbdCoupling};
// use crate::fem::{FemScene, FemSceneBuildFn};
// use crate::mpm::{self, MpmScene, MpmSceneBuildFn};
use crate::rapier::prelude::{Collider, ColliderSet, ImpulseJointSet, RigidBody, RigidBodySet};
// use crate::rbd::{
//     BackendType, RbdScene, RenderContext, SimulationState, setup_physics,
// };
use crate::{DemoKind, RunState, Transition, UiSections};
use crate::backend::BackendType;
use crate::graphics::RenderContext;

/// UI / runtime state that is independent from the GPU/window resources. Kept in
/// its own struct so [`NexusViewer::render`] can split-borrow it from `window`.
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
    /// User-editable per-scene simulation settings, surfaced in the Settings panel.
    pub sim_settings: SimSettings,
    /// Index of the demo [`sim_settings`](Self::sim_settings) was seeded from.
    /// The settings are pulled FROM the scene only when this differs from the
    /// selected demo (i.e. a genuine demo switch) and pushed back TO the scene
    /// otherwise — so restarts and backend switches keep the user's edits.
    pub(crate) settings_demo: Option<usize>,
    /// Which sub-systems the current scene contains (drives which settings show).
    pub(crate) has_mpm: bool,
    pub(crate) has_fem: bool,
    pub(crate) has_rbd: bool,
    /// Current scene entity counts, refreshed every `sync` for the UI.
    pub(crate) counts: NexusCounts,
}

/// Editable simulation settings exposed in the testbed UI. The viewer pulls
/// these from the [`NexusState`] when a demo loads and pushes edits back each
/// frame (see [`NexusViewer::sync`]).
#[derive(Clone)]
pub struct SimSettings {
    /// MPM substeps per rendered frame.
    pub mpm_substeps: u32,
    /// MPM CPIC rigid-body coupling toggle.
    pub mpm_use_cpic: bool,
    /// Gravity applied to the MPM continuum.
    pub mpm_gravity: Vector,
    /// FEM substeps per rendered frame.
    pub fem_substeps: u32,
    /// Gravity applied to the FEM soft bodies.
    pub fem_gravity: Vector,
    /// FEM mass-proportional damping.
    pub fem_damping: f32,
    /// Rigid-body solver steps advanced per rendered frame.
    pub rbd_steps_per_frame: u32,
}

impl Default for SimSettings {
    fn default() -> Self {
        Self {
            mpm_substeps: 20,
            mpm_use_cpic: true,
            mpm_gravity: Vector::ZERO,
            fem_substeps: 10,
            fem_gravity: Vector::ZERO,
            fem_damping: 5.0,
            rbd_steps_per_frame: 1,
        }
    }
}

/// Minimal [`Scene`] used to draw the UI panels for a viewer-owned
/// [`NexusState`] scene, which (unlike [`RbdScene`]) has no scene object of its
/// own to hand to [`NexusViewer::render_frame`].
struct NexusSceneUi;


#[cfg(feature = "dim2")]
pub type SceneNode = SceneNode2d;
#[cfg(feature = "dim3")]
pub type SceneNode = SceneNode3d;

pub struct ViewerNode {
    node: SceneNode,
    instance_id: usize,
}

/// Number of frames to render the "compiling shaders" banner before the
/// (blocking) pipeline preload, so the browser actually *presents* it first. On
/// the web, `create_compute_pipeline` stalls the JS thread without yielding, so
/// a banner drawn but not yet composited would never reach the screen before
/// the freeze; rendering a few real frames first forces the paint.
const COMPILE_BANNER_PRESENT_FRAMES: u32 = 10;

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
    /// Viewer-owned point-cloud node for MPM particles (lazily created in `sync`).
    mpm_node: Option<SceneNodeX>,
    /// Viewer-owned point-cloud node for FEM vertices (lazily created in `sync`).
    fem_node: Option<SceneNodeX>,
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
            mpm_node: None,
            fem_node: None,
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
                sim_settings: SimSettings::default(),
                settings_demo: None,
                has_mpm: false,
                has_fem: false,
                has_rbd: false,
                counts: NexusCounts::default(),
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
            BackendType::Cpu => self
                .cpu
                .as_ref()
                .expect("CPU backend unavailable: compile with the 'cpu' feature"),
            #[allow(unreachable_patterns)]
            _ => panic!("selected backend is not available in this build"),
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

    /// Reads the latest state from a [`NexusState`] back from the GPU and pushes
    /// it into the viewer-owned render instances: rigid-body collider poses, the
    /// MPM particle point cloud, and the FEM vertex point cloud.
    pub async fn sync(&mut self, state: &mut NexusState) {
        // Simulation stats (incl. GPU pass timestamps) are aggregated into
        // `run_stats` for every sub-state, so surface them regardless of whether
        // the scene has any rigid bodies (FEM/MPM-only scenes have none).
        self.ui.run_stats = state.run_stats.clone();
        self.ui.counts = state.counts();

        // Settings: seed the UI from the scene only when a different demo is
        // loaded; on a restart / backend switch (same demo) keep the user's
        // current settings and push them back into the freshly-built scene.
        if self.ui.settings_demo != Some(self.ui.selected_demo) {
            self.ui.has_rbd = state.rbd.is_some();
            // Use `has_mpm()` (not `mpm.is_some()`) so emitters that start with
            // no particles — MPM allocated lazily on the first emit — still
            // surface their MPM settings in the UI.
            self.ui.has_mpm = state.has_mpm();
            self.ui.has_fem = state.fem.is_some();
            self.ui.sim_settings.mpm_substeps = state.mpm_substeps();
            self.ui.sim_settings.mpm_use_cpic = state.mpm_use_cpic();
            self.ui.sim_settings.mpm_gravity = state.mpm_gravity();
            self.ui.sim_settings.fem_substeps = state.fem_substeps();
            self.ui.sim_settings.fem_gravity = state.fem_gravity();
            self.ui.sim_settings.fem_damping = state.fem_damping();
            self.ui.sim_settings.rbd_steps_per_frame = state.rbd_steps_per_frame();
            self.ui.settings_demo = Some(self.ui.selected_demo);
        } else {
            let s = self.ui.sim_settings.clone();
            state.set_mpm_substeps(s.mpm_substeps);
            state.set_mpm_use_cpic(s.mpm_use_cpic);
            state.set_mpm_gravity(s.mpm_gravity);
            state.set_fem_substeps(s.fem_substeps);
            let _ = state.set_fem_gravity(self.backend(), s.fem_gravity);
            let _ = state.set_fem_damping(self.backend(), s.fem_damping);
            state.set_rbd_steps_per_frame(s.rbd_steps_per_frame);
        }

        // Rigid bodies: collider poses → instanced shapes.
        if let Some(rbd) = state.rbd.as_ref() {
            let poses = rbd.poses();
            let mut cache = vec![Pose::default(); poses.len() as usize];
            let _ = self.backend().slow_read_buffer(poses.buffer(), &mut cache).await;
            self.nexus_render.update_instances_from_poses(state, &cache);
        }

        // MPM particles: world positions → a point cloud sized to the grid.
        if let Some(mpm) = state.mpm.as_ref() {
            let positions = mpm
                .particles
                .read_positions(self.backend())
                .await
                .unwrap_or_default();
            let scale = state.mpm_cell_width() * 0.5;
            if self.mpm_node.is_none() {
                self.mpm_node = Some(self.new_point_node());
            }
            let data = build_point_instances(&positions, scale, [0.65, 0.5, 0.35, 1.0]);
            self.mpm_node.as_mut().unwrap().set_instances(&data);
        }

        // FEM vertices: read back positions and render as a point cloud.
        if let Some(fem) = state.fem.as_mut() {
            let mut enc = self.backend().begin_encoding();
            let _ = fem.launch_readback(&mut enc);
            let _ = self.backend().submit(enc);
            let _ = self.backend().synchronize();
            let positions = fem.read_positions(self.backend()).await.unwrap_or_default();
            let scale = estimate_point_scale(&positions);
            if self.fem_node.is_none() {
                self.fem_node = Some(self.new_point_node());
            }
            let data = build_point_instances(&positions, scale, [0.35, 0.55, 0.82, 1.0]);
            self.fem_node.as_mut().unwrap().set_instances(&data);
        }
    }

    /// Creates a unit point-cloud base node (a cube in 3D, a rectangle in 2D)
    /// that subsequent per-particle/per-vertex instances are drawn from.
    fn new_point_node(&mut self) -> SceneNodeX {
        #[cfg(feature = "dim2")]
        {
            self.scene2d.add_rectangle(1.0, 1.0)
        }
        #[cfg(feature = "dim3")]
        {
            self.scene3d.add_cube(1.0, 1.0, 1.0)
        }
    }

    /// The backend currently selected in the UI.
    pub fn backend_type(&self) -> BackendType {
        self.ui.backend_type
    }

    /// Renders the "Compiling shaders…" overlay for a few frames and presents
    /// them. Call right before a blocking pipeline compilation so the banner is
    /// on screen during the freeze — and, on the web, actually composited first:
    /// rendering several real frames forces the browser to paint before the
    /// (blocking, non-yielding) `create_compute_pipeline` (see
    /// [`COMPILE_BANNER_PRESENT_FRAMES`]).
    pub async fn show_compile_banner(&mut self) {
        for _ in 0..COMPILE_BANNER_PRESENT_FRAMES {
            let _ = self
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
            let gpu_available = self.webgpu.is_some();
            self.window.draw_ui(|ctx| {
                crate::ui::setup_custom_theme(ctx);
                crate::ui::main_panel(ctx, &mut self.ui, gpu_available);
                crate::ui::compiling_overlay(ctx);
            });
        }
    }

    /// Tears down the viewer-owned `NexusState` render nodes. A no-op for legacy
    /// [`RbdScene`] demos (which detach their own nodes). Call between two runs.
    pub fn clear_scene(&mut self) {
        self.nexus_render.clear();
        if let Some(mut node) = self.mpm_node.take() {
            node.detach();
        }
        if let Some(mut node) = self.fem_node.take() {
            node.detach();
        }
        // NOTE: `settings_demo` is intentionally NOT reset here, so a restart or
        // backend switch (same demo) preserves the user's settings. It is re-seeded
        // in `sync` only when the selected demo actually changes.
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
        // Disjoint closure capture (edition 2024): the closure borrows `self.ui`
        // and `scene_ui` while `self.window` is the receiver.
        self.window.draw_ui(|ctx| {
            crate::ui::setup_custom_theme(ctx);
            crate::ui::main_panel(ctx, &mut self.ui, gpu_available);
        });

        self.ui.transition.is_none()
    }

    /// Caches the GPU pipeline extracted from a finished RBD scene so the next
    /// RBD scene on the same backend can reuse it (skipping shader compilation).
    pub(crate) fn cache_pipeline(&mut self, pipeline: RbdPipeline) {
        self.cached_gpu_pipeline = Some(pipeline);
    }
}

/// Builds point-cloud render instances (one per position) at a uniform scale and
/// color — used for MPM particles and FEM vertices.
#[cfg(feature = "dim3")]
fn build_point_instances(
    positions: &[glamx::Vec3],
    scale: f32,
    color: [f32; 4],
) -> Vec<InstanceData3d> {
    let deformation = glamx::Mat3::from_diagonal(glamx::Vec3::splat(scale));
    let color = Color::new(color[0], color[1], color[2], color[3]);
    positions
        .iter()
        .map(|p| InstanceData3d {
            position: *p,
            color,
            deformation,
            ..Default::default()
        })
        .collect()
}

#[cfg(feature = "dim2")]
fn build_point_instances(
    positions: &[glamx::Vec2],
    scale: f32,
    color: [f32; 4],
) -> Vec<InstanceData2d> {
    let deformation = glamx::Mat2::from_diagonal(glamx::Vec2::splat(scale));
    positions
        .iter()
        .map(|p| InstanceData2d {
            position: *p,
            color,
            deformation,
            ..Default::default()
        })
        .collect()
}

/// Estimates a point sprite size from the vertex spacing (bounding-box diagonal
/// divided by the per-axis count). Used for FEM vertices, whose spacing varies.
#[cfg(feature = "dim3")]
fn estimate_point_scale(positions: &[glamx::Vec3]) -> f32 {
    if positions.len() < 2 {
        return 0.05;
    }
    let (mut lo, mut hi) = (positions[0], positions[0]);
    for p in positions {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    let diag = (hi - lo).length().max(1e-4);
    (diag / (positions.len() as f32).cbrt()).max(1e-4) * 0.6
}

#[cfg(feature = "dim2")]
fn estimate_point_scale(positions: &[glamx::Vec2]) -> f32 {
    if positions.len() < 2 {
        return 0.05;
    }
    let (mut lo, mut hi) = (positions[0], positions[0]);
    for p in positions {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    let diag = (hi - lo).length().max(1e-4);
    (diag / (positions.len() as f32).sqrt()).max(1e-4) * 0.6
}
