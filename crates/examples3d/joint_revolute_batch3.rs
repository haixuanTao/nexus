use nexus_testbed3d::{DemoBuilder, SimulationState};
use rapier3d::prelude::*;

pub fn builder() -> DemoBuilder {
    DemoBuilder::rbd("Joints (Revolute Batch)", build)
}

fn build() -> SimulationState {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut impulse_joints = ImpulseJointSet::new();

    let rad = 0.4;
    let num = 10;
    let shift = 2.0;

    // Build a single chain at the origin.
    let ground = RigidBodyBuilder::fixed().translation(Vec3::new(0.0, 0.0, 0.0));
    let mut curr_parent = bodies.insert(ground);
    let collider = ColliderBuilder::cuboid(rad, rad, rad);
    colliders.insert_with_parent(collider, curr_parent, &mut bodies);

    for i in 0..num {
        let z = i as f32 * shift * 2.0 + shift;
        let positions = [
            Pose3::translation(0.0, 0.0, z),
            Pose3::translation(shift, 0.0, z),
            Pose3::translation(shift, 0.0, z + shift),
            Pose3::translation(0.0, 0.0, z + shift),
        ];

        let mut handles = [curr_parent; 4];
        for k in 0..4 {
            let rigid_body = RigidBodyBuilder::dynamic().pose(positions[k]);
            handles[k] = bodies.insert(rigid_body);
            let collider = ColliderBuilder::cuboid(rad, rad, rad).density(1.0);
            colliders.insert_with_parent(collider, handles[k], &mut bodies);
        }

        let x = Vec3::X;
        let z = Vec3::Z;

        let revs = [
            RevoluteJointBuilder::new(z).local_anchor2(Vec3::new(0.0, 0.0, -shift)),
            RevoluteJointBuilder::new(x).local_anchor2(Vec3::new(-shift, 0.0, 0.0)),
            RevoluteJointBuilder::new(z).local_anchor2(Vec3::new(0.0, 0.0, -shift)),
            RevoluteJointBuilder::new(x).local_anchor2(Vec3::new(shift, 0.0, 0.0)),
        ];

        impulse_joints.insert(curr_parent, handles[0], revs[0], true);
        impulse_joints.insert(handles[0], handles[1], revs[1], true);
        impulse_joints.insert(handles[1], handles[2], revs[2], true);
        impulse_joints.insert(handles[2], handles[3], revs[3], true);

        curr_parent = handles[3];
    }

    // Replicate the chain across many batches with position offsets.
    let num_batches = 200u32;
    let batch_offsets: Vec<_> = (0..num_batches)
        .map(|i| {
            Vec3::new(
                (i as f32 - num_batches as f32 / 2.0) * 4.0,
                0.0,
                0.0,
            )
        })
        .collect();

    SimulationState {
        bodies,
        colliders,
        impulse_joints,
        num_batches,
        batch_offsets,
    }
}
