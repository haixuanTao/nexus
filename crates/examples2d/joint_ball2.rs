use nexus_testbed2d::{DemoBuilder, SimulationState};
use rapier2d::prelude::*;

pub fn builder() -> DemoBuilder {
    DemoBuilder::rbd("Joints (spherical)", build)
}

fn build() -> SimulationState {
    /*
     * World
     */
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut impulse_joints = ImpulseJointSet::new();

    /*
     * Create the balls
     */
    // Build the rigid body.
    let rad = 0.4;
    let numi = 350; // Num vertical nodes.
    let numk = 350; // Num horizontal nodes.
    let shift = 1.0;

    let mut body_handles = Vec::new();

    for k in 0..numk {
        for i in 0..numi {
            let fk = k as f32;
            let fi = i as f32;

            let status = if i == 0 && (k < numk / 5 || k >= (4 * numk) / 5) {
                RigidBodyType::Fixed
            } else {
                RigidBodyType::Dynamic
            };

            let rigid_body =
                RigidBodyBuilder::new(status).translation(Vec2::new(fk * shift, -fi * shift));
            let child_handle = bodies.insert(rigid_body);
            let collider = ColliderBuilder::ball(rad);
            colliders.insert_with_parent(collider, child_handle, &mut bodies);

            // Vertical joint.
            if i > 0 {
                let parent_handle = *body_handles.last().unwrap();
                let joint = RevoluteJointBuilder::new().local_anchor2(Vec2::new(0.0, shift));
                impulse_joints.insert(parent_handle, child_handle, joint, true);
            }

            // Horizontal joint.
            if k > 0 {
                let parent_index = body_handles.len() - numi;
                let parent_handle = body_handles[parent_index];
                let joint = RevoluteJointBuilder::new().local_anchor2(Vec2::new(-shift, 0.0));
                impulse_joints.insert(parent_handle, child_handle, joint, true);
            }

            body_handles.push(child_handle);
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
    // testbed.look_at(point![numk as f32 * rad, numi as f32 * -rad], 5.0);
}
