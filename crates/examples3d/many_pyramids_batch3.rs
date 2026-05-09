use nexus_testbed3d::{BatchEnvironment, DemoBuilder, SimulationState};
use rapier3d::prelude::*;

pub fn builder() -> DemoBuilder {
    DemoBuilder::rbd("Many pyramids (batched)", build)
}

fn create_pyramid(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    offset: Vector,
    stack_height: usize,
    rad: f32,
) {
    let shift = rad * 2.0;

    for i in 0usize..stack_height {
        for j in i..stack_height {
            let fj = j as f32;
            let fi = i as f32;
            let x = (fi * shift / 2.0) + (fj - fi) * shift;
            let y = fi * shift;

            // Build the rigid body.
            let rigid_body = RigidBodyBuilder::dynamic().translation(Vec3::new(x, y, 0.0) + offset);
            let handle = bodies.insert(rigid_body);
            let collider = ColliderBuilder::cuboid(rad, rad, rad);
            colliders.insert_with_parent(collider, handle, bodies);
        }
    }
}

fn build() -> SimulationState {
    let mut environments = vec![];
    let pyramid_count = 40;

    for pyramid_index in 0..pyramid_count {
        /*
         * World
         */
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let impulse_joints = ImpulseJointSet::new();

        let rad = 0.5;
        let spacing = 4.0;

        /*
         * Ground
         */
        let ground_size = 100.0;
        let ground_height = 0.1;

        let rigid_body = RigidBodyBuilder::fixed().translation(Vec3::new(0.0, -ground_height, 0.0));
        let ground_handle = bodies.insert(rigid_body);
        let collider = ColliderBuilder::cuboid(
            ground_size,
            ground_height,
            pyramid_count as f32 * spacing / 2.0 + ground_size,
        );
        colliders.insert_with_parent(collider, ground_handle, &mut bodies);

        /*
         * Create the cubes
         */
        let bottomy = rad;
        create_pyramid(
            &mut bodies,
            &mut colliders,
            Vec3::new(
                0.0,
                bottomy,
                (pyramid_index as f32 - pyramid_count as f32 / 2.0) * spacing,
            ),
            60,
            rad,
        );

        create_pyramid(
            &mut bodies,
            &mut colliders,
            Vec3::new(
                -75.0,
                bottomy,
                (pyramid_index as f32 - pyramid_count as f32 / 2.0) * spacing,
            ),
            60,
            rad,
        );

        environments.push(BatchEnvironment {
            bodies,
            colliders,
            impulse_joints,
            multibody_joints: rapier3d::prelude::MultibodyJointSet::new(),
            sim_params: Default::default(),
            visuals: Default::default(),
        });
    }

    /*
     * Set up the testbed.
     */
    SimulationState::from_environments(environments)
    // testbed.look_at(point![100.0, 100.0, 100.0], Point::origin());
}
