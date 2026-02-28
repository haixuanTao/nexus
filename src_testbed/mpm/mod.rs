pub mod data;
pub mod step;

pub use data::*;

use crate::RunState;
use step::{GpuReadbackData, SimulationStepResult, WgPrepReadback};
use khal::Shader;
use khal::backend::{GpuBackend as KhalGpuBackend, GpuTimestamps};
use kiss3d::prelude::*;
use nexus::mpm::pipeline::{MpmPipeline, MpmPipelineHooks};
use nexus::mpm::solver::GpuParticleModelData;
use rapier::geometry::{ColliderHandle, Shape, ShapeType};

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use kiss3d::scene::{SceneNode2d, SceneNode3d};

#[cfg(feature = "dim2")]
type RenderNode = SceneNode2d;
#[cfg(feature = "dim3")]
type RenderNode = SceneNode3d;

pub type MpmSceneBuilders<GpuModel> = Vec<(String, MpmSceneBuildFn<GpuModel>)>;
pub type MpmSceneBuildFn<GpuModel> =
    fn(&KhalGpuBackend, &mut MpmAppState<GpuModel>) -> MpmPhysicsContext<GpuModel>;

pub struct MpmStage<GpuModel: GpuParticleModelData> {
    pub(crate) gpu: KhalGpuBackend,
    pub(crate) selected_demo: usize,
    pub(crate) builders: MpmSceneBuilders<GpuModel>,
    pub(crate) physics: MpmPhysicsContext<GpuModel>,
    pub(crate) hooks: Box<dyn MpmPipelineHooks<GpuModel>>,
    pub(crate) app_state: MpmAppState<GpuModel>,
    pub(crate) step_id: usize,
    pub(crate) step_result: SimulationStepResult,
    pub(crate) readback_shader: WgPrepReadback,
    pub(crate) readback: GpuReadbackData,
    pub(crate) timestamps: GpuTimestamps,
    #[cfg(feature = "dim2")]
    pub(crate) instances: Vec<InstanceData2d>,
    #[cfg(feature = "dim3")]
    pub(crate) instances: Vec<InstanceData3d>,
    #[cfg(feature = "dim2")]
    pub(crate) rigid_instances: Vec<InstanceData2d>,
    #[cfg(feature = "dim3")]
    pub(crate) rigid_instances: Vec<InstanceData3d>,
}

impl<GpuModel: GpuParticleModelData> MpmStage<GpuModel> {
    pub async fn new(
        gpu: KhalGpuBackend,
        hooks: impl FnOnce(&KhalGpuBackend) -> Box<dyn MpmPipelineHooks<GpuModel>>,
        builders: MpmSceneBuilders<GpuModel>,
    ) -> MpmStage<GpuModel> {
        let mpm_pipeline = MpmPipeline::new(&gpu).unwrap();
        let mut app_state = MpmAppState {
            pipeline: mpm_pipeline,
            render_mode: RenderMode::Default,
            max_num_substeps: 1,
            min_num_substeps: 1,
            num_substeps: 1,
            gravity_factor: 1.0,
            restarting: false,
            show_rigid_particles: false,
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

        MpmStage {
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

    pub async fn update(&mut self) {
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

// Collider rendering for MPM scenes (static/kinematic bodies).

pub fn update_colliders<GpuModel: GpuParticleModelData>(
    physics: &MpmPhysicsContext<GpuModel>,
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

pub fn render_colliders<GpuModel: GpuParticleModelData>(
    scene2d: &mut SceneNode2d,
    scene3d: &mut SceneNode3d,
    physics: &MpmPhysicsContext<GpuModel>,
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
