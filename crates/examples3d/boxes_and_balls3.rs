use nexus_testbed3d::{DemoBuilder, SimulationState};
use rapier3d::prelude::*;

pub fn builder() -> DemoBuilder {
    DemoBuilder::rbd("Boxes & balls", build)
}

fn build() -> SimulationState {
    const NXZ: isize = 30;
    const NY: isize = 70;

    let mut bodies = RigidBodySet::default();
    let mut colliders = ColliderSet::default();
    let impulse_joints = ImpulseJointSet::default();

    /*
     * Falling dynamic objects.
     */
    for j in 0..NY {
        let max_ik = NXZ / 2;
        for i in -max_ik..max_ik {
            for k in -max_ik..max_ik {
                let x = i as f32 * 1.1 + j as f32 * 0.01;
                let y = j as f32 * 1.1;
                let z = k as f32 * 1.1 + j as f32 * 0.01;
                let pos = Vec3::new(x, y, z);

                let body = bodies.insert(RigidBodyBuilder::dynamic().translation(pos));

                let collider = if j % 2 == 0 {
                    ColliderBuilder::cuboid(0.5, 0.5, 0.5)
                } else {
                    ColliderBuilder::ball(0.5)
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
    }
    // testbed.look_at(point![100.0, 100.0, 100.0], Point::origin());
}
