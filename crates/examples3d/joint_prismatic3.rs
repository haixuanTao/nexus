use nexus_testbed3d::{DemoBuilder, SimulationState};
use rapier3d::prelude::*;

pub fn builder() -> DemoBuilder {
    DemoBuilder::rbd("Joints (Prismatic)", build)
}

fn build() -> SimulationState {
    /*
     * World
     */
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut impulse_joints = ImpulseJointSet::new();

    let rad = 0.4;
    let num = 10;
    let shift = 1.0;

    for m in 0..20 {
        let z = m as f32 * shift * (num as f32 + 2.0);

        for l in 0..20 {
            let y = l as f32 * shift * (num as f32) * 2.0;

            for j in 0..30 {
                let x = j as f32 * shift * 4.0;

                let ground = RigidBodyBuilder::fixed().translation(Vec3::new(x, y, z));
                let mut curr_parent = bodies.insert(ground);
                let collider = ColliderBuilder::cuboid(rad, rad, rad);
                colliders.insert_with_parent(collider, curr_parent, &mut bodies);

                for i in 0..num {
                    let z = z + (i + 1) as f32 * shift;
                    let density = 1.0;
                    let rigid_body = RigidBodyBuilder::dynamic().translation(Vec3::new(x, y, z));
                    let curr_child = bodies.insert(rigid_body);
                    let collider = ColliderBuilder::cuboid(rad, rad, rad).density(density);
                    colliders.insert_with_parent(collider, curr_child, &mut bodies);

                    let axis = if i % 2 == 0 {
                        Vec3::new(1.0, 1.0, 0.0).normalize()
                    } else {
                        Vec3::new(-1.0, 1.0, 0.0).normalize()
                    };

                    let prism = PrismaticJointBuilder::new(axis)
                        .local_anchor2(Vec3::new(0.0, 0.0, -shift))
                        .limits([-2.0, 0.0]);
                    impulse_joints.insert(curr_parent, curr_child, prism, true);

                    curr_parent = curr_child;
                }
            }
        }
    }

    /*
     * Set up the testbed.
     */
    SimulationState {
        bodies,
        colliders,
        impulse_joints,
    }
    // testbed.look_at(point![262.0, 63.0, 124.0], point![101.0, 4.0, -3.0]);
}
