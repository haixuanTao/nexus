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

pub use data::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

mod data;
pub mod step;

use crate::step::{GpuReadbackData, RenderConfig, SimulationStepResult, WgPrepReadback};
use khal::Shader;
use khal::backend::{Backend, GpuBackend as KhalGpuBackend, GpuTimestamps, WebGpu};
use khal::re_exports::wgpu::Limits;
use kiss3d::egui;
use kiss3d::prelude::*;
use nexus::mpm::pipeline::{MpmPipeline, MpmPipelineHooks};
use nexus::mpm::solver::GpuParticleModelData;
use rapier::geometry::{ColliderHandle, Shape, ShapeType};

#[cfg(feature = "dim3")]
use kiss3d::camera::{FixedView2d, OrbitCamera3d};
#[cfg(feature = "dim2")]
use kiss3d::camera::{FixedView3d, PanZoomCamera2d};
use kiss3d::scene::{SceneNode2d, SceneNode3d};

pub type SceneBuilders<GpuModel> = Vec<(String, SceneBuildFn<GpuModel>)>;
pub type SceneBuildFn<GpuModel> =
    fn(&KhalGpuBackend, &mut AppState<GpuModel>) -> PhysicsContext<GpuModel>;

#[cfg(feature = "dim2")]
type RenderNode = SceneNode2d;
#[cfg(feature = "dim3")]
type RenderNode = SceneNode3d;

pub(crate) struct Stage<GpuModel: GpuParticleModelData> {
    pub(crate) gpu: KhalGpuBackend,
    pub(crate) selected_demo: usize,
    pub(crate) builders: SceneBuilders<GpuModel>,
    pub(crate) physics: PhysicsContext<GpuModel>,
    pub(crate) hooks: Box<dyn MpmPipelineHooks<GpuModel>>,
    pub(crate) app_state: AppState<GpuModel>,
    pub(crate) step_id: usize,
    pub(crate) step_result: SimulationStepResult,
    pub(crate) readback_shader: WgPrepReadback,
    pub(crate) readback: GpuReadbackData,
    pub(crate) timestamps: GpuTimestamps,
    #[cfg(feature = "dim2")]
    instances: Vec<InstanceData2d>,
    #[cfg(feature = "dim3")]
    instances: Vec<InstanceData3d>,
    #[cfg(feature = "dim2")]
    rigid_instances: Vec<InstanceData2d>,
    #[cfg(feature = "dim3")]
    rigid_instances: Vec<InstanceData3d>,
}

impl<GpuModel: GpuParticleModelData> Stage<GpuModel> {
    pub async fn new(
        hooks: impl FnOnce(&KhalGpuBackend) -> Box<dyn MpmPipelineHooks<GpuModel>>,
        builders: SceneBuilders<GpuModel>,
    ) -> Stage<GpuModel> {
        let limits = Limits {
            #[cfg(target_arch = "wasm32")]
            max_storage_buffers_per_shader_stage: 10,
            #[cfg(not(target_arch = "wasm32"))]
            max_storage_buffers_per_shader_stage: 12,
            max_buffer_size: 1_000_000_000,
            max_storage_buffer_binding_size: 1_000_000_000,
            max_compute_workgroup_storage_size: 19904, // For P2G
            ..Limits::default()
        };
        let gpu = WebGpu::new(Default::default(), limits).await.unwrap();
        let gpu = KhalGpuBackend::WebGpu(gpu);

        let mpm_pipeline = MpmPipeline::new(&gpu).unwrap();
        let mut app_state = AppState {
            pipeline: mpm_pipeline,
            run_state: RunState::Paused,
            render_mode: RenderMode::Default,
            max_num_substeps: 1,
            min_num_substeps: 1,
            num_substeps: 1,
            gravity_factor: 1.0,
            restarting: false,
            show_rigid_particles: true,
            use_cpic: true,
        };
        let hooks = hooks(&gpu);
        let physics = (builders[0].1)(&gpu, &mut app_state);
        app_state.num_substeps = 0;

        let readback_shader = WgPrepReadback::from_backend(&gpu).unwrap();
        let num_rigid = physics.data.rigid_particles.len() as usize;
        let readback = GpuReadbackData::new(
            &gpu,
            physics.data.particles.len(),
            num_rigid,
            RenderMode::Default as u32,
        )
        .unwrap();
        let timestamps = GpuTimestamps::new(&gpu, 2048);
        let mut step_result = SimulationStepResult::default();
        step_result
            .instances
            .resize(physics.data.particles.len(), Default::default());
        step_result
            .rigid_instances
            .resize(num_rigid, Default::default());

        Stage {
            builders,
            instances: vec![],
            rigid_instances: vec![],
            readback_shader,
            readback,
            timestamps,
            gpu,
            physics,
            hooks,
            app_state,
            step_result,
            step_id: 0,
            selected_demo: 0,
        }
    }

    pub fn set_demo(&mut self, demo_id: usize) {
        self.selected_demo = demo_id;
        self.physics = (self.builders[demo_id]).1(&self.gpu, &mut self.app_state);
        let num_rigid = self.physics.data.rigid_particles.len() as usize;
        self.readback = GpuReadbackData::new(
            &self.gpu,
            self.physics.data.particles.len(),
            num_rigid,
            self.app_state.render_mode as u32,
        )
        .unwrap();
        self.app_state.num_substeps = 1;
        self.step_result
            .instances
            .resize(self.physics.data.particles.len(), Default::default());
        self.step_result
            .rigid_instances
            .resize(num_rigid, Default::default());
    }

    async fn update(&mut self) {
        if !self.step_simulation().await {
            return;
        }

        self.instances.clear();
        #[cfg(feature = "dim2")]
        self.instances
            .extend(self.step_result.instances.iter().map(|d| InstanceData2d {
                position: d.position,
                color: [d.color.x, d.color.y, d.color.z, d.color.w],
                deformation: d.deformation,
                ..Default::default()
            }));
        #[cfg(feature = "dim3")]
        self.instances
            .extend(self.step_result.instances.iter().map(|d| {
                use nexus::mpm::mpm_shaders::PaddingExt;
                InstanceData3d {
                    position: d.position,
                    color: Color::new(d.color.x, d.color.y, d.color.z, d.color.w),
                    deformation: d.deformation.remove_padding(),
                    ..Default::default()
                }
            }));

        self.rigid_instances.clear();
        if self.app_state.show_rigid_particles {
            #[cfg(feature = "dim2")]
            self.rigid_instances
                .extend(
                    self.step_result
                        .rigid_instances
                        .iter()
                        .map(|d| InstanceData2d {
                            position: d.position,
                            color: [d.color.x, d.color.y, d.color.z, d.color.w],
                            deformation: d.deformation,
                            ..Default::default()
                        }),
                );
            #[cfg(feature = "dim3")]
            self.rigid_instances
                .extend(self.step_result.rigid_instances.iter().map(|d| {
                    use nexus::mpm::mpm_shaders::PaddingExt;
                    InstanceData3d {
                        position: d.position,
                        color: Color::new(d.color.x, d.color.y, d.color.z, d.color.w),
                        deformation: d.deformation.remove_padding(),
                        ..Default::default()
                    }
                }));
        }
    }
}

pub async fn run<GpuModel: GpuParticleModelData>(scene_builders: SceneBuilders<GpuModel>) {
    #[cfg(feature = "dim2")]
    {
        run_with_hooks(|_| Box::new(()), scene_builders).await;
    }
    #[cfg(feature = "dim3")]
    {
        run_with_hooks(|_| Box::new(()), scene_builders).await;
    }
}

pub async fn run_with_hooks<GpuModel: GpuParticleModelData>(
    hooks: impl FnOnce(&KhalGpuBackend) -> Box<dyn MpmPipelineHooks<GpuModel>>,
    scene_builders: SceneBuilders<GpuModel>,
) {
    let mut colliders_gfx: HashMap<ColliderHandle, RenderNode> = HashMap::new();
    let mut stage = Stage::new(hooks, scene_builders).await;

    let mut window = Window::new("nexus mpm testbed").await;
    let mut scene3d = SceneNode3d::empty();
    let mut scene2d = SceneNode2d::empty();

    #[cfg(feature = "dim3")]
    {
        scene3d.add_directional_light(glamx::Vec3::new(-1.0, -1.0, -1.0));
        scene3d.add_directional_light(glamx::Vec3::new(1.0, -1.0, 1.0));
    }

    render_colliders(
        &mut scene2d,
        &mut scene3d,
        &stage.physics,
        &mut colliders_gfx,
    );

    // Create particle instance nodes.
    #[cfg(feature = "dim2")]
    let mut particle_node = scene2d.add_rectangle(1.0, 1.0);
    #[cfg(feature = "dim3")]
    let mut particle_node = scene3d.add_cube(1.0, 1.0, 1.0);

    // Create rigid particle instance node.
    #[cfg(feature = "dim2")]
    let mut rigid_particle_node = scene2d.add_rectangle(1.0, 1.0);
    #[cfg(feature = "dim3")]
    let mut rigid_particle_node = scene3d.add_cube(1.0, 1.0, 1.0);

    #[cfg(feature = "dim2")]
    let (mut camera2d, mut camera3d) = {
        let mut sidescroll = PanZoomCamera2d::default();
        sidescroll.look_at(glamx::Vec2::new(35.0, 35.0), 5.0);
        (sidescroll, FixedView3d::default())
    };
    #[cfg(feature = "dim3")]
    let (mut camera2d, mut camera3d) = {
        let arc_ball = OrbitCamera3d::new(glamx::Vec3::new(40.0, 40.0, 40.0), glamx::Vec3::ZERO);
        (FixedView2d::default(), arc_ball)
    };

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
        let mut new_selected_demo = None;

        // Step simulation.
        stage.update().await;

        // Update collider rendering.
        update_colliders(&stage.physics, &mut colliders_gfx);
        particle_node.set_instances(&stage.instances);
        rigid_particle_node.set_instances(&stage.rigid_instances);

        // UI
        window.draw_ui(|ctx| {
            egui::Window::new("Settings").show(ctx, |ui| {
                let mut changed = false;
                egui::ComboBox::from_label("selected sample")
                    .selected_text(&stage.builders[stage.selected_demo].0)
                    .show_ui(ui, |ui| {
                        for (i, (name, _)) in stage.builders.iter().enumerate() {
                            changed = ui
                                .selectable_value(&mut stage.selected_demo, i, name)
                                .changed()
                                || changed;
                        }
                    });
                if changed {
                    new_selected_demo = Some(stage.selected_demo);
                }

                let prev_render_mode = stage.app_state.render_mode;
                egui::ComboBox::from_label("render mode")
                    .selected_text(stage.app_state.render_mode.text())
                    .show_ui(ui, |ui| {
                        for mode in RenderMode::ALL {
                            ui.selectable_value(
                                &mut stage.app_state.render_mode,
                                *mode,
                                mode.text(),
                            );
                        }
                    });

                if stage.app_state.render_mode != prev_render_mode {
                    stage
                        .gpu
                        .write_buffer(
                            stage.readback.mode.buffer_mut(),
                            0,
                            &[RenderConfig {
                                mode: stage.app_state.render_mode as u32,
                            }],
                        )
                        .unwrap();
                }

                ui.checkbox(
                    &mut stage.app_state.show_rigid_particles,
                    "show rigid particles",
                );
                ui.checkbox(&mut stage.app_state.use_cpic, "use CPIC");

                ui.label(format!(
                    "total: {:.1}ms (encoding: {:.1}ms)",
                    stage.step_result.timings.total_step_time,
                    stage.step_result.timings.encoding_time
                ));
                ui.label(format!(
                    "readback: {:.1}ms",
                    stage.step_result.timings.readback_time
                ));
                ui.label(format!("particles: {}", stage.physics.data.particles.len()));
                ui.label(format!("substeps: {}", stage.app_state.num_substeps));

                if !stage.step_result.timings.gpu_pass_times.is_empty() {
                    ui.separator();
                    ui.label(format!(
                        "GPU total: {:.2}ms",
                        stage.step_result.timings.gpu_total_time
                    ));
                    for (label, ms) in &stage.step_result.timings.gpu_pass_times {
                        ui.label(format!("  {}: {:.2}ms", label, ms));
                    }
                }

                ui.horizontal(|ui| {
                    let play_pause_label = if stage.app_state.run_state == RunState::Running {
                        "Pause"
                    } else {
                        "Play"
                    };
                    if ui.button(play_pause_label).clicked() {
                        if stage.app_state.run_state == RunState::Running {
                            stage.app_state.run_state = RunState::Paused;
                        } else {
                            stage.app_state.run_state = RunState::Running;
                        }
                    }
                    if ui.button("Step").clicked() {
                        stage.app_state.run_state = RunState::Step;
                    }
                    if ui.button("Restart").clicked() {
                        new_selected_demo = Some(stage.selected_demo);
                    }
                });
            });
        });

        if let Some(demo) = new_selected_demo {
            stage.set_demo(demo);
            // Remove old colliders and re-render.
            for (_, mut node) in colliders_gfx.drain() {
                node.detach();
            }
            render_colliders(
                &mut scene2d,
                &mut scene3d,
                &stage.physics,
                &mut colliders_gfx,
            );
        }
    }
}

fn update_colliders<GpuModel: GpuParticleModelData>(
    physics: &PhysicsContext<GpuModel>,
    colliders: &mut HashMap<ColliderHandle, RenderNode>,
) {
    for (handle, node) in colliders.iter_mut() {
        if let Some(collider) = physics.rapier_data.colliders.get(*handle) {
            let pose = collider.position();
            let tra = pose.translation;

            #[cfg(feature = "dim2")]
            {
                node.set_position(tra);
                node.set_rotation(pose.rotation.angle());
            }
            #[cfg(feature = "dim3")]
            {
                node.set_position(tra);
                node.set_rotation(pose.rotation);
            }
        }
    }
}

fn render_colliders<GpuModel: GpuParticleModelData>(
    scene2d: &mut SceneNode2d,
    scene3d: &mut SceneNode3d,
    physics: &PhysicsContext<GpuModel>,
    colliders: &mut HashMap<ColliderHandle, RenderNode>,
) {
    for (handle, collider) in physics.rapier_data.colliders.iter() {
        if let Some(node) = generate_collider_node(scene2d, scene3d, collider.shape()) {
            colliders.insert(handle, node);
        }
    }
}

#[cfg(feature = "dim2")]
fn generate_collider_node(
    scene2d: &mut SceneNode2d,
    _scene3d: &mut SceneNode3d,
    co_shape: &dyn Shape,
) -> Option<RenderNode> {
    match co_shape.shape_type() {
        ShapeType::Cuboid => {
            let cuboid = co_shape.as_cuboid().unwrap();
            let polyline = cuboid.to_polyline();
            let mesh = kiss3d_mesh_from_polyline_2d(&polyline);
            Some(scene2d.add_mesh(Rc::new(RefCell::new(mesh)), glamx::Vec2::ONE))
        }
        ShapeType::Ball => {
            let ball = co_shape.as_ball().unwrap();
            let polyline = ball.to_polyline(40);
            let mesh = kiss3d_mesh_from_polyline_2d(&polyline);
            Some(scene2d.add_mesh(Rc::new(RefCell::new(mesh)), glamx::Vec2::ONE))
        }
        ShapeType::Capsule => {
            let capsule = co_shape.as_capsule().unwrap();
            let polyline = capsule.to_polyline(40);
            let mesh = kiss3d_mesh_from_polyline_2d(&polyline);
            Some(scene2d.add_mesh(Rc::new(RefCell::new(mesh)), glamx::Vec2::ONE))
        }
        ShapeType::Triangle => {
            let tri = co_shape.as_triangle().unwrap();
            let vtx = vec![tri.a, tri.b, tri.c];
            let mesh = kiss3d_mesh_2d(&vtx, &[[0, 1, 2]]);
            Some(scene2d.add_mesh(Rc::new(RefCell::new(mesh)), glamx::Vec2::ONE))
        }
        ShapeType::ConvexPolygon => {
            let poly = co_shape.as_convex_polygon().unwrap();
            let polyline = poly.points().to_vec();
            let mesh = kiss3d_mesh_from_polyline_2d(&polyline);
            Some(scene2d.add_mesh(Rc::new(RefCell::new(mesh)), glamx::Vec2::ONE))
        }
        ShapeType::Polyline => {
            // Polylines are rendered as thin triangles.
            let polyline = co_shape.as_polyline().unwrap();
            let vertices = polyline.vertices();
            if vertices.len() < 2 {
                return None;
            }
            let mesh = kiss3d_mesh_from_polyline_2d(vertices);
            Some(scene2d.add_mesh(Rc::new(RefCell::new(mesh)), glamx::Vec2::ONE))
        }
        _ => None,
    }
}

#[cfg(feature = "dim3")]
fn generate_collider_node(
    _scene2d: &mut SceneNode2d,
    scene3d: &mut SceneNode3d,
    co_shape: &dyn Shape,
) -> Option<RenderNode> {
    match co_shape.shape_type() {
        ShapeType::Ball => {
            let ball = co_shape.as_ball().unwrap();
            let (vtx, idx) = ball.to_trimesh(10, 10);
            let mesh = kiss3d_mesh_3d(&vtx, &idx);
            Some(scene3d.add_mesh(Rc::new(RefCell::new(mesh)), glamx::Vec3::ONE))
        }
        ShapeType::Cuboid => {
            let cuboid = co_shape.as_cuboid().unwrap();
            let sz = cuboid.half_extents * 2.0;
            Some(scene3d.add_cube(sz.x, sz.y, sz.z))
        }
        ShapeType::Capsule => {
            let capsule = co_shape.as_capsule().unwrap();
            let (vtx, idx) = capsule.to_trimesh(20, 10);
            let mesh = kiss3d_mesh_3d(&vtx, &idx);
            Some(scene3d.add_mesh(Rc::new(RefCell::new(mesh)), glamx::Vec3::ONE))
        }
        ShapeType::Triangle => {
            let tri = co_shape.as_triangle().unwrap();
            let mesh = kiss3d_mesh_3d(&[tri.a, tri.b, tri.c], &[[0, 1, 2], [0, 2, 1]]);
            Some(scene3d.add_mesh(Rc::new(RefCell::new(mesh)), glamx::Vec3::ONE))
        }
        ShapeType::TriMesh => {
            let trimesh = co_shape.as_trimesh().unwrap();
            let mesh = kiss3d_mesh_3d(trimesh.vertices(), trimesh.indices());
            Some(scene3d.add_mesh(Rc::new(RefCell::new(mesh)), glamx::Vec3::ONE))
        }
        ShapeType::HeightField => {
            let heightfield = co_shape.as_heightfield().unwrap();
            let (vtx, idx) = heightfield.to_trimesh();
            let mesh = kiss3d_mesh_3d(&vtx, &idx);
            Some(scene3d.add_mesh(Rc::new(RefCell::new(mesh)), glamx::Vec3::ONE))
        }
        ShapeType::ConvexPolyhedron => {
            let poly = co_shape.as_convex_polyhedron().unwrap();
            let (vtx, idx) = poly.to_trimesh();
            let mesh = kiss3d_mesh_3d(&vtx, &idx);
            Some(scene3d.add_mesh(Rc::new(RefCell::new(mesh)), glamx::Vec3::ONE))
        }
        _ => None,
    }
}

#[cfg(feature = "dim2")]
fn kiss3d_mesh_from_polyline_2d(vertices: &[glamx::Vec2]) -> GpuMesh2d {
    let n = vertices.len();
    if n < 3 {
        return GpuMesh2d::new(vertices.to_vec(), vec![], None, false);
    }
    let idx: Vec<_> = (1..n as u32 - 1).map(|i| [0, i, i + 1]).collect();
    kiss3d_mesh_2d(vertices, &idx)
}

#[cfg(feature = "dim2")]
fn kiss3d_mesh_2d(vertices: &[glamx::Vec2], indices: &[[u32; 3]]) -> GpuMesh2d {
    GpuMesh2d::new(vertices.to_vec(), indices.to_vec(), None, false)
}

#[cfg(feature = "dim3")]
fn kiss3d_mesh_3d(vertices: &[glamx::Vec3], indices: &[[u32; 3]]) -> GpuMesh3d {
    GpuMesh3d::new(vertices.to_vec(), indices.to_vec(), None, None, false)
}
