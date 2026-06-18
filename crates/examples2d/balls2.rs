use khal::backend::GpuTimestamps;
use nexus_testbed2d::NexusViewer;
use nexus2d::prelude::{NexusState, RbdCoupling};
use rapier2d::prelude::*;

pub async fn run(viewer: &mut NexusViewer) -> anyhow::Result<NexusState> {
    let mut state = NexusState::default();
    let no_coupling = RbdCoupling::NONE;

    /*
     * Ground
     */
    let ground_size = 150.0;

    let body = RigidBodyBuilder::fixed().build();
    let collider = ColliderBuilder::cuboid(ground_size, 1.5).build();
    let shape = collider.shared_shape().clone();
    let handle = state.insert_rigid_body(body, collider, no_coupling);
    viewer.insert_shape(handle, &shape);

    let body = RigidBodyBuilder::fixed()
        .rotation(std::f32::consts::FRAC_PI_2)
        .translation(Vec2::new(ground_size, ground_size * 1.2))
        .build();
    let collider = ColliderBuilder::cuboid(ground_size * 1.2, 1.5).build();
    let shape = collider.shared_shape().clone();
    let handle = state.insert_rigid_body(body, collider, no_coupling);
    viewer.insert_shape(handle, &shape);

    let body = RigidBodyBuilder::fixed()
        .rotation(std::f32::consts::FRAC_PI_2)
        .translation(Vec2::new(-ground_size, ground_size * 1.2))
        .build();
    let collider = ColliderBuilder::cuboid(ground_size * 1.2, 1.5).build();
    let shape = collider.shared_shape().clone();
    let handle = state.insert_rigid_body(body, collider, no_coupling);
    viewer.insert_shape(handle, &shape);

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
            let body = RigidBodyBuilder::dynamic().translation(Vec2::new(x, y)).build();
            let collider = ColliderBuilder::ball(rad).build();
            let shape = collider.shared_shape().clone();
            let handle = state.insert_rigid_body(body, collider, no_coupling);
            viewer.insert_shape(handle, &shape);
        }
    }

    // Optional, useful so we can render even before starting the simulation.
    let mut timestamps = GpuTimestamps::new(viewer.backend(), 1024);
    state.finalize(viewer.backend()).await?;

    while viewer.render_frame().await {
        if viewer.simulating() {
            state.simulate(viewer.backend(), Some(&mut timestamps)).await;
        }
        viewer.sync(&mut state).await;
    }

    Ok(state)
}
