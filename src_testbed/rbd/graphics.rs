use super::SimulationState;
use super::backend::PhysicsBackend;
use glamx::Vec3;
use nexus::rbd::math::Pose;

#[cfg(feature = "dim3")]
use kiss3d::color::Color;
use kiss3d::scene::{SceneNode2d, SceneNode3d};
use rapier::math::DIM;
use rapier::parry::shape::ShapeType;
use std::collections::HashMap;

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

pub struct InstancedNodeEntry {
    pub index: usize,
    pub color: [f32; 4],
    pub scale: [f32; DIM],
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

pub struct RenderContext {
    pub instances: Vec<InstancedNode>,
}

impl RenderContext {
    pub fn clear(&mut self) {
        for instance in &mut self.instances {
            instance.node.detach();
        }
        self.instances.clear();
    }
}

/// Set up a simple scene using instancing for efficient rendering
pub async fn setup_graphics(
    scene_2d: &mut SceneNode2d,
    scene_3d: &mut SceneNode3d,
    phys: &SimulationState,
) -> RenderContext {
    #[cfg(feature = "dim2")]
    let scene = scene_2d;
    #[cfg(feature = "dim3")]
    let scene = scene_3d;
    #[cfg(feature = "dim2")]
    let _ = scene_3d;
    #[cfg(feature = "dim3")]
    let _ = scene_2d;
    let fixed_color = Vec3::new(0.6, 0.6, 0.6);

    let mut instances = HashMap::new();
    #[cfg(feature = "dim2")]
    let mut polygon_instances: HashMap<Vec<u32>, InstancedNode> = HashMap::new();
    #[cfg(feature = "dim3")]
    let mut polyhedron_instances: HashMap<Vec<u32>, InstancedNode> = HashMap::new();
    let mut singletons = vec![];

    let max_colliders_per_batch = phys
        .environments
        .iter()
        .map(|e| e.colliders.len())
        .max()
        .unwrap_or(0);

    for (batch_idx, env) in phys.environments.iter().enumerate() {
        for (i, (_, co)) in env.colliders.iter().enumerate() {
            let shape = co.shape();
            let is_fixed = co.parent().map(|h| env.bodies[h].is_fixed()) != Some(false);
            let color = if is_fixed {
                fixed_color
            } else {
                let coeff = (1.0 - 0.15 * (i % 5) as f32) / 255.0;
                match shape.shape_type() {
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
                }
            };

            let index = batch_idx * max_colliders_per_batch + i;

            match shape.shape_type() {
                ShapeType::Ball => {
                    let instanced_node =
                        instances.entry(ShapeType::Ball).or_insert_with(|| {
                            #[cfg(feature = "dim2")]
                            let node = scene.add_circle(0.5);
                            #[cfg(feature = "dim3")]
                            let node = {
                                let lowres_sphere =
                                    kiss3d::procedural::sphere(1.0, 10, 10, true);
                                scene.add_render_mesh(lowres_sphere, Vec3::ONE)
                            };
                            InstancedNode {
                                node,
                                entries: vec![],
                                data: vec![],
                            }
                        });
                    let ball = shape.as_ball().unwrap();
                    instanced_node.entries.push(InstancedNodeEntry {
                        index,
                        color: [color.x, color.y, color.z, 1.0],
                        scale: [ball.radius * 2.0; DIM],
                    });
                }
                ShapeType::Cuboid => {
                    let instanced_node =
                        instances.entry(ShapeType::Cuboid).or_insert_with(|| {
                            #[cfg(feature = "dim2")]
                            let node = scene.add_rectangle(1.0, 1.0);
                            #[cfg(feature = "dim3")]
                            let node = scene.add_cube(1.0, 1.0, 1.0);
                            InstancedNode {
                                node,
                                entries: vec![],
                                data: vec![],
                            }
                        });
                    let cuboid = shape.as_cuboid().unwrap();
                    instanced_node.entries.push(InstancedNodeEntry {
                        index,
                        color: [color.x, color.y, color.z, 1.0],
                        scale: (cuboid.half_extents * 2.0).into(),
                    });
                }
                #[cfg(feature = "dim3")]
                ShapeType::Cylinder => {
                    let instanced_node =
                        instances.entry(ShapeType::Cylinder).or_insert_with(|| {
                            let node = scene.add_cylinder(1.0, 1.0);
                            InstancedNode {
                                node,
                                entries: vec![],
                                data: vec![],
                            }
                        });
                    let cyl = shape.as_cylinder().unwrap();
                    instanced_node.entries.push(InstancedNodeEntry {
                        index,
                        color: [color.x, color.y, color.z, 1.0],
                        scale: [cyl.radius, cyl.half_height * 2.0, cyl.radius],
                    });
                }
                #[cfg(feature = "dim3")]
                ShapeType::Cone => {
                    let instanced_node =
                        instances.entry(ShapeType::Cone).or_insert_with(|| {
                            let node = scene.add_cone(1.0, 1.0);
                            InstancedNode {
                                node,
                                entries: vec![],
                                data: vec![],
                            }
                        });
                    let c = shape.as_cone().unwrap();
                    instanced_node.entries.push(InstancedNodeEntry {
                        index,
                        color: [color.x, color.y, color.z, 1.0],
                        scale: [c.radius, c.half_height * 2.0, c.radius],
                    });
                }
                ShapeType::Capsule => {
                    let instanced_node =
                        instances.entry(ShapeType::Capsule).or_insert_with(|| {
                            #[cfg(feature = "dim2")]
                            let node = scene.add_capsule(0.5, 1.0);
                            #[cfg(feature = "dim3")]
                            let node = scene.add_capsule(0.5, 1.0);
                            InstancedNode {
                                node,
                                entries: vec![],
                                data: vec![],
                            }
                        });
                    let c = shape.as_capsule().unwrap();
                    instanced_node.entries.push(InstancedNodeEntry {
                        index,
                        color: [color.x, color.y, color.z, 1.0],
                        #[cfg(feature = "dim2")]
                        scale: [c.radius * 2.0, c.segment.length()],
                        #[cfg(feature = "dim3")]
                        scale: [c.radius * 2.0, c.segment.length(), c.radius * 2.0],
                    });
                }
                #[cfg(feature = "dim2")]
                ShapeType::ConvexPolygon => {
                    let poly = shape.as_convex_polygon().unwrap();
                    let points: Vec<_> = poly.points().to_vec();
                    let vertex_key = polygon_vertex_key(&points);

                    let instanced_node =
                        polygon_instances
                            .entry(vertex_key)
                            .or_insert_with(|| InstancedNode {
                                node: scene.add_convex_polygon(points, Vec2::ONE),
                                entries: vec![],
                                data: vec![],
                            });
                    instanced_node.entries.push(InstancedNodeEntry {
                        index,
                        color: [color.x, color.y, color.z, 1.0],
                        scale: [1.0, 1.0],
                    });
                }
                #[cfg(feature = "dim3")]
                ShapeType::ConvexPolyhedron => {
                    use kiss3d::procedural::RenderMesh;

                    let poly = shape.as_convex_polyhedron().unwrap();
                    let points: Vec<_> = poly.points().to_vec();
                    let vertex_key = polyhedron_vertex_key(&points);

                    let instanced_node =
                        polyhedron_instances.entry(vertex_key).or_insert_with(|| {
                            let (vtx, idx) = poly.to_trimesh();
                            let trimesh =
                                rapier::parry::shape::TriMesh::new(vtx, idx).unwrap();
                            let mut render = RenderMesh::from(trimesh);
                            render.replicate_vertices();
                            render.recompute_normals();
                            InstancedNode {
                                node: scene.add_render_mesh(render, Vec3::ONE),
                                entries: vec![],
                                data: vec![],
                            }
                        });
                    instanced_node.entries.push(InstancedNodeEntry {
                        index,
                        color: [color.x, color.y, color.z, 1.0],
                        scale: [1.0, 1.0, 1.0],
                    });
                }
                #[cfg(feature = "dim3")]
                ShapeType::TriMesh => {
                    use kiss3d::procedural::RenderMesh;

                    let trimesh = shape.as_trimesh().unwrap();
                    let mut render = RenderMesh::from(trimesh.clone());
                    render.replicate_vertices();
                    render.recompute_normals();
                    let node = scene.add_render_mesh(render, Vec3::ONE);
                    let mut singleton = InstancedNode {
                        node,
                        entries: vec![],
                        data: vec![],
                    };
                    singleton.entries.push(InstancedNodeEntry {
                        index,
                        color: [color.x, color.y, color.z, 1.0],
                        scale: [1.0; DIM],
                    });
                    singletons.push(singleton);
                }
                #[cfg(feature = "dim2")]
                ShapeType::Polyline => {
                    let polyline = shape.as_polyline().unwrap();
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
                        let rot =
                            glamx::Mat2::from_cols(Vec2::new(cos, sin), Vec2::new(-sin, cos));
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
                    let mut singleton = InstancedNode {
                        node,
                        entries: vec![],
                        data: vec![],
                    };
                    singleton.entries.push(InstancedNodeEntry {
                        index,
                        color: [color.x, color.y, color.z, 1.0],
                        scale: [1.0; DIM],
                    });
                    singletons.push(singleton);
                }
                _ => todo!(),
            }
        }
    }

    #[cfg(feature = "dim2")]
    let all_instances = instances
        .into_values()
        .chain(polygon_instances.into_values())
        .chain(singletons.into_iter())
        .collect();
    #[cfg(feature = "dim3")]
    let all_instances = instances
        .into_values()
        .chain(polyhedron_instances.into_values())
        .chain(singletons.into_iter())
        .collect();

    RenderContext {
        instances: all_instances,
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

/// Update rendering instances with current physics poses
pub fn update_instances(render_ctx: &mut RenderContext, physics_backend: &PhysicsBackend) {
    for instanced_node in &mut render_ctx.instances {
        instanced_node.data.clear();

        for entry in &instanced_node.entries {
            let pose = &physics_backend.poses()[entry.index];
            let (position, deformation) = pose_to_render_data(pose, &entry.scale);

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
