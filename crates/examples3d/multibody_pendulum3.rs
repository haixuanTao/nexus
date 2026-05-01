use nexus_testbed3d::{DemoBuilder, SimulationState};
use rapier3d::prelude::*;

pub fn builder() -> DemoBuilder {
    DemoBuilder::rbd("Multibody (Pendulum)", build)
}

fn build() -> SimulationState {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let impulse_joints = ImpulseJointSet::new();
    let mut multibody_joints = MultibodyJointSet::new();

    /*
     * The ground
     */
    // let ground_size = 100.0;
    // let ground_height = 0.1;
    //
    // let rigid_body = RigidBodyBuilder::fixed().translation(Vec3::new(0.0, -ground_height - 5.0, 0.0));
    // let ground_handle = bodies.insert(rigid_body);
    // let collider = ColliderBuilder::cuboid(
    //     ground_size,
    //     ground_height,
    //     ground_size,
    // );
    // colliders.insert_with_parent(collider, ground_handle, &mut bodies);

    /*
     * A 3-link pendulum modeled with rapier's MultibodyJointSet.
     *
     * - A fixed root body is anchored at the origin.
     * - Three dynamic links hang from revolute joints about the X axis.
     * - Under gravity alone, the chain should swing in the YZ plane.
     *
     * The GPU pipeline picks up the multibody set from `SimulationState::environments`
     * and runs `GpuMultibodySolver::step` each frame — no contacts or constraints
     * with multibodies are involved.
     */
    let rad = 0.4;
    let link_len = 2.0;
    let num_links = 20;

    // Fixed root at origin.
    let root_body = RigidBodyBuilder::fixed();
    let mut parent_handle = bodies.insert(root_body);
    let root_collider = ColliderBuilder::cuboid(rad, rad, rad);
    colliders.insert_with_parent(root_collider, parent_handle, &mut bodies);

    for i in 0..num_links {
        // Each link hangs `link_len` below its parent.
        let x = (i as f32 + 1.0) * link_len;
        let rigid_body = RigidBodyBuilder::dynamic().translation(Vec3::new(x, 0.0, 0.0));
        let handle = bodies.insert(rigid_body);
        let collider = ColliderBuilder::cuboid(link_len * 0.5, rad, rad);
        colliders.insert_with_parent(collider, handle, &mut bodies);

        // Revolute joint about X: anchor on parent is at its bottom
        // (or at origin for the root), anchor on child is at its top.
        let parent_anchor = if i == 0 {
            Vec3::ZERO
        } else {
            Vec3::new(link_len * 0.8, 0.0, 0.0)
        };
        let joint = RevoluteJointBuilder::new(Vec3::Z)
            .local_anchor1(parent_anchor)
            .local_anchor2(Vec3::new(-link_len * 0.8, 0.0, 0.0))
            // .limits([-0.1, 0.1])
            // .contacts_enabled(false)
            .build();
        multibody_joints.insert(parent_handle, handle, joint, true);

        parent_handle = handle;
    }

    SimulationState::single_with_multibody(bodies, colliders, impulse_joints, multibody_joints)
}
