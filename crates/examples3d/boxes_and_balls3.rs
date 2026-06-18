use khal::backend::GpuTimestamps;
use nexus_testbed3d::NexusViewer;
use nexus3d::prelude::{NexusState, RbdCoupling};
use rapier3d::prelude::*;

pub async fn run(viewer: &mut NexusViewer) -> anyhow::Result<NexusState> {
    const NXZ: isize = 30;
    const NY: isize = 70;

    let mut state = NexusState::default();
    let no_coupling = RbdCoupling::NONE;

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

                let body = RigidBodyBuilder::dynamic().translation(pos).build();
                let collider = if j % 2 == 0 {
                    ColliderBuilder::cuboid(0.5, 0.5, 0.5)
                } else {
                    ColliderBuilder::ball(0.5)
                }
                .build();
                let shape = collider.shared_shape().clone();
                let handle = state.insert_rigid_body(body, collider, no_coupling);
                viewer.insert_shape(handle, &shape);
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
            let body = RigidBodyBuilder::fixed().build();
            let collider = ColliderBuilder::cuboid(wall_sz.x, wall_sz.y, wall_sz.z)
                .translation(wall_pos)
                .build();
            let shape = collider.shared_shape().clone();
            let handle = state.insert_rigid_body(body, collider, no_coupling);
            viewer.insert_shape(handle, &shape);
        }
    }

    let mut timestamps = GpuTimestamps::new(viewer.backend(), 1024);
    state.finalize(viewer.backend()).await?;

    while viewer.render_frame().await {
        if viewer.simulating() {
            state.simulate(viewer.backend(), Some(&mut timestamps)).await;
        }
        viewer.sync(&mut state).await;
    }

    Ok(state)
    // testbed.look_at(point![100.0, 100.0, 100.0], Point::origin());
}
