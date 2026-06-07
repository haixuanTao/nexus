use nexus_testbed2d::{DemoBuilder, SimulationState};
use rapier2d::prelude::*;

pub fn builder() -> DemoBuilder {
    DemoBuilder::rbd("Balls", build)
}

fn build() -> SimulationState {
    /*
     * World
     */
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let impulse_joints = ImpulseJointSet::new();

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

    let shift = rad * 2.0 + 0.2;
    let centerx = shift * (num as f32) / 2.0;
    let centery = shift / 2.0;

    for i in 0..num {
        for j in 0usize..num * 4 {
            let x = i as f32 * shift - centerx + (j % 2) as f32 * 0.2;
            let y = j as f32 * shift + centery + 20.0;

            // Build the rigid body.
            let rigid_body = RigidBodyBuilder::dynamic().translation(Vec2::new(x, y));
            let handle = bodies.insert(rigid_body);
            let collider = ColliderBuilder::ball(rad);
            colliders.insert_with_parent(collider, handle, &mut bodies);
        }
    }

    /*
     * Set up the testbed.
     */
    SimulationState::single(bodies, colliders, impulse_joints)
    // testbed.look_at(point![0.0, 50.0], 10.0);
}
