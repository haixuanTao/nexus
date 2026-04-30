use nexus_testbed3d::{DemoBuilder, SimulationState, VisualShape};
use rapier3d::prelude::*;
use rapier3d_urdf::{UrdfLoaderOptions, UrdfMultibodyOptions, UrdfRobot};
use std::collections::HashMap;
use std::path::PathBuf;

pub fn builder() -> DemoBuilder {
    DemoBuilder::rbd("URDF (multibody)", build)
}

fn urdf_path() -> PathBuf {
    // let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    // PathBuf::from(home).join("Downloads/robot.urdf")
    // PathBuf::from("/Users/sebcrozet/work/rapier/assets/3d/T12/urdf/T12.URDF")
    PathBuf::from("/Users/sebcrozet/work/nexus-demos/XoQ/js/examples/assets/openarm_v10.urdf")
}

fn build() -> SimulationState {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut impulse_joints = ImpulseJointSet::new();
    let mut multibody_joints = MultibodyJointSet::new();
    let mut visuals: HashMap<ColliderHandle, VisualShape> = HashMap::new();

    /*
     * Robot loaded from URDF.
     */
    let path = urdf_path();
    let scale = 40.0;
    let options = UrdfLoaderOptions {
        create_colliders_from_collision_shapes: true,
        // Many URDFs (including openarm_v10) only ship visual meshes and leave the
        // collision tags empty, so fall back to visuals when collisions are missing.
        create_colliders_from_visual_shapes: true,
        apply_imported_mass_props: true,
        make_roots_fixed: true,
        scale,
        // Use cheap oriented bounding boxes for physics. The loader keeps the
        // original triangle meshes attached to each collider as a `UrdfVisual`,
        // which we forward to the testbed's per-collider visual override map.
        mesh_converter: None, // Some(MeshConverter::Obb),
        // Lift the robot above the ground. URDF is Z-up but the testbed is Y-up,
        // so rotate -90° around X so the robot stands upright.
        shift: Pose::from_parts(
            Vec3::new(0.0, scale, 0.0),
            Rotation::from_rotation_x(-std::f32::consts::FRAC_PI_2),
        ),
        collider_blueprint: ColliderBuilder::ball(0.5).collision_groups(InteractionGroups::none()),
        ..UrdfLoaderOptions::default()
    };

    match UrdfRobot::from_file(&path, options, None) {
        Ok((robot, _)) => {
            // let handles = robot.insert_using_impulse_joints(
            //     &mut bodies,
            //     &mut colliders,
            //     &mut impulse_joints,
            // );
            let handles = robot.insert_using_multibody_joints(
                &mut bodies,
                &mut colliders,
                &mut multibody_joints,
                UrdfMultibodyOptions::DISABLE_SELF_CONTACTS
            );

            for link in &handles.links {
                for collider in &link.colliders {
                    if let Some(visual) = &collider.visual {
                        visuals.insert(
                            collider.handle,
                            VisualShape::with_local_pose(
                                visual.shape.clone(),
                                visual.local_pose,
                            ),
                        );
                    }
                }
            }
        }
        Err(e) => {
            eprintln!(
                "Failed to load URDF file at {}: {e}.",
                path.display()
            );
        }
    }

    SimulationState::single_with_multibody_and_visuals(
        bodies,
        colliders,
        impulse_joints,
        multibody_joints,
        visuals,
    )
}
