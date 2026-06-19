use glamx::Vec3;
use nexus::rbd::math::Pose;

#[cfg(feature = "dim3")]
use kiss3d::color::Color;
use kiss3d::procedural::IndexBuffer;
use kiss3d::scene::{SceneNode2d, SceneNode3d};
use rapier::math::DIM;
use rapier::parry::shape::ShapeType;
use std::collections::HashMap;
use rapier::data::{Coarena, Index};
use rapier::prelude::{RigidBodyHandle, SharedShape};
#[cfg(feature = "dim2")]
use {
    glamx::{Mat2, Vec2},
    kiss3d::resource::GpuMesh2d,
    kiss3d::scene::InstanceData2d,
    std::cell::RefCell,
    std::rc::Rc,
};

#[cfg(feature = "dim3")]
use {glamx::Mat3, kiss3d::scene::InstanceData3d};
use nexus::prelude::NexusState;

pub struct InstancedNodeEntry {
    pub pose_index: u32,
    /// Environment (batch) this entry belongs to. Combined with `handle` it
    /// resolves the GPU pose slot through `state.rbd2gpu[env]`.
    pub env: u32,
    pub handle: Index,
    pub color: [f32; 4],
    pub scale: [f32; DIM],
    /// Local pose offset composed with the collider pose at render time. Defaults
    /// to identity; populated when a [`super::VisualShape`] override is registered
    /// in [`super::BatchEnvironment::visuals`] so that proxy-collider shapes (e.g.
    /// OBBs) can be replaced at the right local frame.
    pub local_pose: Pose,
}

/// Convert polygon vertices to a Vec<u32> key for exact matching in batching.
#[cfg(feature = "dim2")]
fn polygon_vertex_key(points: &[Vec2]) -> Vec<u32> {
    let mut key = Vec::with_capacity(points.len() * 2);
    for pt in points {
        key.push(pt.x.to_bits());
        key.push(pt.y.to_bits());
    }
    key
}

/// Convert polyhedron vertices to a Vec<u32> key for exact matching in batching.
#[cfg(feature = "dim3")]
fn polyhedron_vertex_key(points: &[Vec3]) -> Vec<u32> {
    let mut key = Vec::with_capacity(points.len() * 3);
    for pt in points {
        key.push(pt.x.to_bits());
        key.push(pt.y.to_bits());
        key.push(pt.z.to_bits());
    }
    key
}

pub struct InstancedNode {
    #[cfg(feature = "dim2")]
    pub node: SceneNode2d,
    #[cfg(feature = "dim3")]
    pub node: SceneNode3d,
    pub entries: Vec<InstancedNodeEntry>,
    #[cfg(feature = "dim2")]
    pub data: Vec<InstanceData2d>,
    #[cfg(feature = "dim3")]
    pub data: Vec<InstanceData3d>,
}

#[derive(Copy, Clone, Debug)]
pub struct InstanceEntry {
    pub instance: u32,
    pub entry: u32,
}

impl Default for InstanceEntry {
    fn default() -> Self {
        Self {instance: u32::MAX, entry: u32::MAX}
    }
}

pub struct RenderContext {
    pub shape2instance: HashMap<ShapeType, usize>,
    /// Instanced nodes for mesh-like shapes (convex hull / trimesh / polyline),
    /// keyed by an exact vertex hash so identical shapes share one node.
    pub mesh2instance: HashMap<Vec<u32>, usize>,
    pub body2instance: Coarena<InstanceEntry>,
    pub instances: Vec<InstancedNode>,
}

impl RenderContext {
    pub fn new() -> Self {
        Self {
            shape2instance: HashMap::new(),
            mesh2instance: HashMap::new(),
            body2instance: Coarena::new(),
            instances: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        for instance in &mut self.instances {
            instance.node.detach();
        }
        self.instances.clear();
        self.shape2instance.clear();
        self.mesh2instance.clear();
    }

    /// Pushes a render entry for `handle` (in environment `env`) into the
    /// instanced node `instance_id`, and records the body → instance mapping.
    fn push_entry(
        &mut self,
        instance_id: usize,
        env: u32,
        handle: RigidBodyHandle,
        color: Vec3,
        local_pose: Pose,
        scale: [f32; DIM],
    ) {
        let instanced_node = &mut self.instances[instance_id];
        instanced_node.entries.push(InstancedNodeEntry {
            pose_index: u32::MAX,
            env,
            handle: handle.0,
            color: [color.x, color.y, color.z, 1.0],
            local_pose,
            scale,
        });
        self.body2instance.insert(handle.0, InstanceEntry {
            instance: instance_id as u32,
            entry: instanced_node.entries.len() as u32 - 1,
        });
    }

    pub fn insert_shape(
        &mut self,
        scene_2d: &mut SceneNode2d,
        scene_3d: &mut SceneNode3d,
        env: u32,
        handle: RigidBodyHandle,
        shape: &SharedShape,
        local_pose: Pose,
    ) {
        #[cfg(feature = "dim2")]
        let scene = scene_2d;
        #[cfg(feature = "dim3")]
        let scene = scene_3d;

        let coeff = 1.0 / 255.0; // (1.0 - 0.15 * (i % 5) as f32) / 255.0;
        let color = match shape.shape_type() {
            ShapeType::Ball => Vec3::new(55.0, 126.0, 184.0) * coeff,
            ShapeType::Cuboid => Vec3::new(55.0, 126.0, 34.0) * coeff,
            #[cfg(feature = "dim3")]
            ShapeType::Cylinder => Vec3::new(140.0, 86.0, 75.0) * coeff,
            #[cfg(feature = "dim3")]
            ShapeType::Cone => Vec3::new(255.0, 217.0, 47.0) * coeff,
            ShapeType::Capsule => Vec3::new(204.0, 121.0, 167.0) * coeff,
            #[cfg(feature = "dim3")]
            ShapeType::ConvexPolyhedron => Vec3::new(228.0, 26.0, 28.0) * coeff,
            _ => Vec3::new(255.0, 127.0, 0.0) * coeff,
        };

        match shape.shape_type() {
            ShapeType::Ball => {
                let instance_id = *self.shape2instance.entry(ShapeType::Ball).or_insert_with(|| {
                    #[cfg(feature = "dim2")]
                    let node = scene.add_circle(0.5);
                    #[cfg(feature = "dim3")]
                    let node = {
                        let lowres_sphere = kiss3d::procedural::sphere(1.0, 10, 10, true);
                        scene.add_render_mesh(lowres_sphere, Vec3::ONE)
                    };
                    self.instances.push(InstancedNode {
                        node,
                        entries: vec![],
                        data: vec![],
                    });
                    self.instances.len() - 1
                });
                let ball = shape.as_ball().unwrap();
                self.push_entry(instance_id, env, handle, color, local_pose, [ball.radius * 2.0; DIM]);
            }
            ShapeType::Cuboid => {
                let instance_id = *self.shape2instance.entry(ShapeType::Cuboid).or_insert_with(|| {
                    #[cfg(feature = "dim2")]
                    let node = scene.add_rectangle(1.0, 1.0);
                    #[cfg(feature = "dim3")]
                    let node = scene.add_cube(1.0, 1.0, 1.0);
                    self.instances.push(InstancedNode {
                        node,
                        entries: vec![],
                        data: vec![],
                    });
                    self.instances.len() - 1
                });
                let cuboid = shape.as_cuboid().unwrap();
                let scale = (cuboid.half_extents * 2.0).into();
                self.push_entry(instance_id, env, handle, color, local_pose, scale);
            }
            #[cfg(feature = "dim3")]
            ShapeType::Cylinder => {
                let instance_id =
                    *self.shape2instance.entry(ShapeType::Cylinder).or_insert_with(|| {
                        let node = scene.add_cylinder(1.0, 1.0);
                        self.instances.push(InstancedNode {
                            node,
                            entries: vec![],
                            data: vec![],
                        });
                        self.instances.len() - 1
                    });
                let cyl = shape.as_cylinder().unwrap();
                self.push_entry(
                    instance_id,
                    env,
                    handle,
                    color,
                    local_pose,
                    [cyl.radius, cyl.half_height * 2.0, cyl.radius],
                );
            }
            #[cfg(feature = "dim3")]
            ShapeType::Cone => {
                let instance_id = *self.shape2instance.entry(ShapeType::Cone).or_insert_with(|| {
                    let node = scene.add_cone(1.0, 1.0);
                    self.instances.push(InstancedNode {
                        node,
                        entries: vec![],
                        data: vec![],
                    });
                    self.instances.len() - 1
                });
                let c = shape.as_cone().unwrap();
                self.push_entry(
                    instance_id,
                    env,
                    handle,
                    color,
                    local_pose,
                    [c.radius, c.half_height * 2.0, c.radius],
                );
            }
            ShapeType::Capsule => {
                let instance_id = *self.shape2instance.entry(ShapeType::Capsule).or_insert_with(|| {
                    #[cfg(feature = "dim2")]
                    let node = scene.add_capsule(0.5, 1.0);
                    #[cfg(feature = "dim3")]
                    let node = scene.add_capsule(0.5, 1.0);
                    self.instances.push(InstancedNode {
                        node,
                        entries: vec![],
                        data: vec![],
                    });
                    self.instances.len() - 1
                });
                let c = shape.as_capsule().unwrap();
                #[cfg(feature = "dim2")]
                let scale = [c.radius * 2.0, c.segment.length()];
                #[cfg(feature = "dim3")]
                let scale = [c.radius * 2.0, c.segment.length(), c.radius * 2.0];
                self.push_entry(instance_id, env, handle, color, local_pose, scale);
            }
            #[cfg(feature = "dim2")]
            ShapeType::ConvexPolygon => {
                let poly = shape.as_convex_polygon().unwrap();
                let points: Vec<_> = poly.points().to_vec();
                let vertex_key = polygon_vertex_key(&points);
                let instance_id = match self.mesh2instance.get(&vertex_key) {
                    Some(id) => *id,
                    None => {
                        let node = scene.add_convex_polygon(points, Vec2::ONE);
                        self.instances.push(InstancedNode {
                            node,
                            entries: vec![],
                            data: vec![],
                        });
                        let id = self.instances.len() - 1;
                        self.mesh2instance.insert(vertex_key, id);
                        id
                    }
                };
                self.push_entry(instance_id, env, handle, color, local_pose, [1.0; DIM]);
            }
            #[cfg(feature = "dim3")]
            ShapeType::ConvexPolyhedron => {
                let poly = shape.as_convex_polyhedron().unwrap();
                let points: Vec<_> = poly.points().to_vec();
                let vertex_key = polyhedron_vertex_key(&points);
                let instance_id = match self.mesh2instance.get(&vertex_key) {
                    Some(id) => *id,
                    None => {
                        let (vtx, idx) = poly.to_trimesh();
                        let mut render = kiss3d::procedural::RenderMesh::new(
                            vtx,
                            None,
                            None,
                            Some(IndexBuffer::Unified(idx)),
                        );
                        render.replicate_vertices();
                        render.recompute_normals();
                        let node = scene.add_render_mesh(render, Vec3::ONE);
                        self.instances.push(InstancedNode {
                            node,
                            entries: vec![],
                            data: vec![],
                        });
                        let id = self.instances.len() - 1;
                        self.mesh2instance.insert(vertex_key, id);
                        id
                    }
                };
                self.push_entry(instance_id, env, handle, color, local_pose, [1.0; DIM]);
            }
            #[cfg(feature = "dim3")]
            ShapeType::TriMesh => {
                let trimesh = shape.as_trimesh().unwrap();
                let vtx: Vec<_> = trimesh.vertices().to_vec();
                let idx: Vec<_> = trimesh.indices().to_vec();
                // Trimeshes are usually unique (a floor, a terrain); key by their
                // full vertex set so re-inserting the same mesh reuses the node.
                let mut vertex_key = Vec::with_capacity(vtx.len() * 3);
                for pt in &vtx {
                    vertex_key.push(pt.x.to_bits());
                    vertex_key.push(pt.y.to_bits());
                    vertex_key.push(pt.z.to_bits());
                }
                let instance_id = match self.mesh2instance.get(&vertex_key) {
                    Some(id) => *id,
                    None => {
                        let mut render = kiss3d::procedural::RenderMesh::new(
                            vtx,
                            None,
                            None,
                            Some(IndexBuffer::Unified(idx)),
                        );
                        render.recompute_normals();
                        let node = scene.add_render_mesh(render, Vec3::ONE);
                        self.instances.push(InstancedNode {
                            node,
                            entries: vec![],
                            data: vec![],
                        });
                        let id = self.instances.len() - 1;
                        self.mesh2instance.insert(vertex_key, id);
                        id
                    }
                };
                self.push_entry(instance_id, env, handle, color, local_pose, [1.0; DIM]);
            }
            #[cfg(feature = "dim2")]
            ShapeType::Polyline => {
                let polyline = shape.as_polyline().unwrap();
                let mut key = Vec::new();
                for v in polyline.vertices() {
                    key.push(v.x.to_bits());
                    key.push(v.y.to_bits());
                }
                let instance_id = match self.mesh2instance.get(&key) {
                    Some(id) => *id,
                    None => {
                        let mut vtx = vec![];
                        let mut idx = vec![];
                        for segment in polyline.segments() {
                            let thickness = 0.2;
                            let center = (segment.a + segment.b) * 0.5;
                            let length = (segment.b - segment.a).length();
                            let scaled_dir = segment.scaled_direction();
                            let angle =
                                scaled_dir.y.atan2(scaled_dir.x) - std::f32::consts::FRAC_PI_2;
                            let cos = angle.cos();
                            let sin = angle.sin();
                            let rot = Mat2::from_cols(Vec2::new(cos, sin), Vec2::new(-sin, cos));
                            let half_w = thickness;
                            let half_h = length / 2.0;
                            let local_vtx = [
                                Vec2::new(-half_w, -half_h),
                                Vec2::new(half_w, -half_h),
                                Vec2::new(half_w, half_h),
                                Vec2::new(-half_w, half_h),
                            ];
                            let base = vtx.len() as u32;
                            for lv in &local_vtx {
                                vtx.push(rot * *lv + center);
                            }
                            idx.push([base, base + 1, base + 2]);
                            idx.push([base, base + 2, base + 3]);
                        }
                        let mesh = GpuMesh2d::new(vtx, idx, None, false);
                        let node = scene.add_mesh(Rc::new(RefCell::new(mesh)), Vec2::ONE);
                        self.instances.push(InstancedNode {
                            node,
                            entries: vec![],
                            data: vec![],
                        });
                        let id = self.instances.len() - 1;
                        self.mesh2instance.insert(key, id);
                        id
                    }
                };
                self.push_entry(instance_id, env, handle, color, local_pose, [1.0; DIM]);
            }
            _ => todo!("unsupported render shape: {:?}", shape.shape_type()),
        }
    }
}

/// Convert a glamx Pose to position and deformation matrix for rendering
#[cfg(feature = "dim2")]
fn pose_to_render_data(pose: &Pose, scale: &[f32; 2]) -> (Vec2, Mat2) {
    let position = pose.translation;
    let cos = pose.rotation.cos();
    let sin = pose.rotation.sin();
    let mut deformation = Mat2::from_cols(Vec2::new(cos, sin), Vec2::new(-sin, cos));
    deformation.x_axis *= scale[0];
    deformation.y_axis *= scale[1];
    (position, deformation)
}

/// Convert a glamx Pose to position and deformation matrix for rendering
#[cfg(feature = "dim3")]
fn pose_to_render_data(pose: &Pose, scale: &[f32; 3]) -> (Vec3, Mat3) {
    let position = pose.translation;
    let deformation = Mat3::from_quat(pose.rotation);
    let deformation = Mat3::from_cols(
        deformation.x_axis * scale[0],
        deformation.y_axis * scale[1],
        deformation.z_axis * scale[2],
    );
    (position, deformation)
}


impl RenderContext {
    /// Update rendering instances from a slice of collider world poses, indexed by
    /// the collider slot stored in each [`InstancedNodeEntry::pose_index`].
    ///
    /// This is the backend-agnostic core of [`update_instances`]; it is used by the
    /// viewer-owned `NexusState` rendering path, which reads the poses straight from
    /// the GPU buffer instead of going through a [`PhysicsBackend`].
    pub fn update_instances_from_poses(&mut self, state: &NexusState, poses: &[Pose]) {
        for instanced_node in &mut self.instances {
            instanced_node.data.clear();

            for entry in &mut instanced_node.entries {
                if entry.pose_index == u32::MAX {
                    entry.pose_index = state
                        .rbd2gpu
                        .get(entry.env as usize)
                        .and_then(|env| env.get(entry.handle))
                        .map(|r| r.gpu_id)
                        .unwrap_or(u32::MAX);
                }

                if entry.pose_index == u32::MAX {
                    continue; // This entry isn’t active on the GPU yet.
                }

                let collider_pose = &poses[entry.pose_index as usize];
                let pose = *collider_pose * entry.local_pose;
                let (position, deformation) = pose_to_render_data(&pose, &entry.scale);

                #[cfg(feature = "dim2")]
                {
                    instanced_node.data.push(InstanceData2d {
                        position,
                        deformation,
                        color: entry.color,
                        lines_color: None,
                        lines_width: None,
                        points_color: None,
                        points_size: None,
                    });
                }

                #[cfg(feature = "dim3")]
                {
                    instanced_node.data.push(InstanceData3d {
                        position,
                        deformation,
                        color: Color::new(
                            entry.color[0],
                            entry.color[1],
                            entry.color[2],
                            entry.color[3],
                        ),
                        lines_color: None,
                        lines_width: None,
                        points_color: None,
                        points_size: None,
                    });
                }
            }

            instanced_node.node.set_instances(&instanced_node.data);
        }
    }
}
