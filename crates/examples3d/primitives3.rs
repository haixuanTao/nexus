use nexus_testbed3d::{DemoBuilder, SimulationState};
use rapier3d::prelude::*;

pub fn builder() -> DemoBuilder {
    DemoBuilder::rbd("Primitives", build)
}

fn build() -> SimulationState {
    const NXZ: isize = 20;
    const NY: isize = 40;

    let mut bodies = RigidBodySet::default();
    let mut colliders = ColliderSet::default();
    let impulse_joints = ImpulseJointSet::default();

    /*
     * Falling dynamic objects.
     */
    // Create 5 predefined convex polyhedron shapes (so we can render
    // them efficiently with instancing).
    let polyhedron_shapes: Vec<SharedShape> = {
        let mut shapes = Vec::new();
        let mut rng = oorandom::Rand32::new(42);

        for _ in 0..5 {
            let mut points = Vec::new();
            let scale = 2.0;
            for _ in 0..10 {
                let pt = Vec3::new(
                    rng.rand_float() - 0.5,
                    rng.rand_float() - 0.5,
                    rng.rand_float() - 0.5,
                );
                points.push(pt * scale);
            }

            // Center the shape at its center-of-mass
            let shape = SharedShape::convex_hull(&points).unwrap();
            let mprops = shape.mass_properties(1.0);
            points.iter_mut().for_each(|pt| *pt -= mprops.local_com);
            shapes.push(SharedShape::convex_hull(&points).unwrap());
        }
        shapes
    };

    for j in 0..NY {
        let max_ik = NXZ / 2;
        for i in -max_ik..max_ik {
            for k in -max_ik..max_ik {
                let x = i as f32 * 1.1 + j as f32 * 0.01;
                let y = j as f32 * 1.6 + 1.0;
                let z = k as f32 * 1.1 + j as f32 * 0.01;
                let pos = Vec3::new(x, y, z);
                let body = bodies.insert(RigidBodyBuilder::dynamic().translation(pos));

                let collider = match j % 6 {
                    0 => ColliderBuilder::cylinder(0.5, 0.5),
                    1 => ColliderBuilder::cuboid(0.5, 0.5, 0.5),
                    2 => ColliderBuilder::cone(0.5, 0.5),
                    3 => ColliderBuilder::capsule_y(0.4, 0.4),
                    4 => ColliderBuilder::ball(0.5),
                    _ => {
                        if i % 2 == 0 || k % 2 == 0 {
                            continue;
                        }
                        // Reuse one of the 5 predefined polyhedron shapes
                        let shape_idx = ((i + max_ik) as usize + (k + max_ik) as usize)
                            % polyhedron_shapes.len();
                        ColliderBuilder::new(polyhedron_shapes[shape_idx].clone())
                    }
                };

                colliders.insert_with_parent(collider, body, &mut bodies);
            }
        }
    }

    /*
     * Floor made of large cuboids.
     */
    {
        let thick = NXZ as f32 * 1.3;
        let height = 8.0;
        let walls = [
            (Vec3::new(0.0, -0.5, 0.0), Vec3::new(thick, 0.5, thick)),
            (Vec3::new(thick, height, 0.0), Vec3::new(0.5, height, thick)),
            (
                Vec3::new(-thick, height, 0.0),
                Vec3::new(0.5, height, thick),
            ),
            (Vec3::new(0.0, height, thick), Vec3::new(thick, height, 0.5)),
            (
                Vec3::new(0.0, height, -thick),
                Vec3::new(thick, height, 0.5),
            ),
        ];

        for (wall_pos, wall_sz) in walls {
            colliders.insert(
                ColliderBuilder::cuboid(wall_sz.x, wall_sz.y, wall_sz.z).translation(wall_pos),
            );
        }
    }

    /*
     * Set up the testbed.
     */
    SimulationState {
        bodies,
        colliders,
        impulse_joints,
        num_batches: 1,
        batch_offsets: vec![],
    }
    // testbed.look_at(point![100.0, 100.0, 100.0], Point::origin());
}
