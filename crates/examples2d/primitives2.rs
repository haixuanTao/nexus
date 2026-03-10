use nexus_testbed2d::{DemoBuilder, SimulationState};
use rapier2d::prelude::*;

pub fn builder() -> DemoBuilder {
    DemoBuilder::rbd("Primitives", build)
}

fn build() -> SimulationState {
    /*
     * World
     */
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let impulse_joints = ImpulseJointSet::new();
    let _multibody_joints = MultibodyJointSet::new();

    /*
     * Ground
     */
    let ground_size = 150.0;

    let rigid_body = RigidBodyBuilder::fixed();
    let handle = bodies.insert(rigid_body);
    let collider = ColliderBuilder::cuboid(ground_size, 1.5);
    colliders.insert_with_parent(collider, handle, &mut bodies);

    let rigid_body = RigidBodyBuilder::fixed()
        .rotation(std::f32::consts::FRAC_PI_2)
        .translation(Vec2::new(ground_size, ground_size * 1.2));
    let handle = bodies.insert(rigid_body);
    let collider = ColliderBuilder::cuboid(ground_size * 1.2, 1.5);
    colliders.insert_with_parent(collider, handle, &mut bodies);

    let rigid_body = RigidBodyBuilder::fixed()
        .rotation(std::f32::consts::FRAC_PI_2)
        .translation(Vec2::new(-ground_size, ground_size * 1.2));
    let handle = bodies.insert(rigid_body);
    let collider = ColliderBuilder::cuboid(ground_size * 1.2, 1.5);
    colliders.insert_with_parent(collider, handle, &mut bodies);

    /*
     * Create the cubes
     */
    let num = 124;
    let rad = 0.5;

    let shift = rad * 2.0 + 0.4;
    let centerx = shift * (num as f32) / 2.0;
    let centery = shift / 2.0;

    // Create 5 predefined convex polygon shapes (so we can render
    // them efficiently with instancing).
    let polygon_shapes: Vec<SharedShape> = {
        let mut shapes = Vec::new();
        let mut rng = oorandom::Rand32::new(42);

        for _ in 0..5 {
            let mut points = Vec::new();
            let scale = 2.0;
            for _ in 0..10 {
                let pt = Vec2::new(rng.rand_float() - 0.5, rng.rand_float() - 0.5);
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

    for i in 0..num {
        for j in 0usize..num * 4 {
            let x = i as f32 * shift - centerx + (j % 2) as f32 * 0.2;
            let y = j as f32 * shift + centery + 20.0;

            // Build the rigid body.
            let rigid_body = RigidBodyBuilder::dynamic().translation(Vec2::new(x, y));
            let handle = bodies.insert(rigid_body);
            let collider = match j % 4 {
                0 => ColliderBuilder::cuboid(rad, rad),
                1 => ColliderBuilder::capsule_y(rad, rad),
                2 => {
                    // Reuse one of the 5 predefined polygon shapes
                    let shape_idx = (i + j) % polygon_shapes.len();
                    ColliderBuilder::new(polygon_shapes[shape_idx].clone())
                }
                _ => ColliderBuilder::ball(rad),
            };
            colliders.insert_with_parent(collider, handle, &mut bodies);
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
    // testbed.look_at(point![0.0, 50.0], 10.0);
}
