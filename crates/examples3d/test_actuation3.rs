//! End-to-end test of the MJCF actuation path (M4 verification).
//!
//! 2 envs × 3-link chains with AccelerationBased position motors (the
//! constraint path — independent of the M3 force-based PD). Checks:
//! 1. `control_multibody_motors` retargets env 0 ONLY (the per-env
//!    `sync_joint_data_from_rapier` under the batch-interleaved layout —
//!    an addressing bug would leak the retarget into env 1);
//! 2. `GpuMultibodySet::read_links` returns the de-SoA'd joint state:
//!    env 0's first joint coordinate tracks the new target, env 1's stays
//!    at the old one, and link world poses match the body-pose buffer.

use glamx::Vec3;
use khal::backend::{Backend, GpuBackend, WebGpu};
use khal::re_exports::wgpu;
use nexus3d::prelude::{NexusCapacities, NexusPipeline, NexusState};
use rapier3d::prelude::*;

const NB: usize = 2;
const LINKS: usize = 3;
const KP: f32 = 1.0e4;
const KD: f32 = 500.0;

fn build_chain_env(
    state: &mut NexusState,
    env: usize,
) -> Vec<rapier3d::dynamics::MultibodyJointHandle> {
    let rad = 0.4;
    let link_len = 2.0;
    let mut parent = state.insert_rigid_body_in(
        env,
        RigidBodyBuilder::fixed().build(),
        ColliderBuilder::cuboid(rad, rad, rad).build(),
    );
    let mut handles = Vec::new();
    for i in 0..LINKS {
        let x = (i as f32 + 1.0) * link_len;
        let handle = state.insert_rigid_body_in(
            env,
            RigidBodyBuilder::dynamic()
                .translation(Vec3::new(x, 0.0, 0.0))
                .build(),
            ColliderBuilder::cuboid(link_len * 0.45, rad, rad).build(),
        );
        let parent_anchor = if i == 0 {
            Vec3::ZERO
        } else {
            Vec3::new(link_len * 0.8, 0.0, 0.0)
        };
        let joint = RevoluteJointBuilder::new(Vec3::Z)
            .local_anchor1(parent_anchor)
            .local_anchor2(Vec3::new(-link_len * 0.8, 0.0, 0.0))
            .motor_model(MotorModel::AccelerationBased)
            .motor_position(0.0, KP, KD)
            .motor_max_force(1.0e6)
            .build();
        let jh = state
            .insert_multibody_joint_in(env, parent, handle, joint)
            .expect("invalid multibody chain");
        handles.push(jh);
        parent = handle;
    }
    // Robot-style: no self-collisions.
    let world = state.rbd_world_mut(env);
    let (mb, _) = world
        .multibody_joints
        .get_mut(handles[0])
        .expect("chain multibody missing");
    mb.set_self_contacts_enabled(false);
    handles
}

async fn link1_angle(backend: &GpuBackend, state: &NexusState, env: usize) -> f32 {
    let rbd = state.rbd.as_ref().unwrap();
    let poses: Vec<glamx::Pose3> = backend.slow_read_vec(rbd.body_poses().buffer()).await.unwrap();
    let stride = poses.len() / NB;
    let q = poses[env * stride + 1].rotation;
    2.0 * q.z.atan2(q.w)
}

async fn run() -> anyhow::Result<()> {
    let limits = wgpu::Limits {
        max_buffer_size: 1_000_000_000,
        max_storage_buffer_binding_size: 1_000_000_000,
        max_storage_buffers_per_shader_stage: 14,
        max_compute_workgroup_storage_size: 19_904,
        ..Default::default()
    };
    let mut webgpu = WebGpu::new(wgpu::Features::default(), limits).await.unwrap();
    webgpu.force_buffer_copy_src = true;
    let backend = GpuBackend::WebGpu(webgpu);

    let mut failures = 0usize;
    let mut check = |name: &str, ok: bool, detail: String| {
        println!("  [{}] {name}{detail}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            failures += 1;
        }
    };

    let mut state = NexusState::new(NexusCapacities::default().rbd_collisions(256));
    let mut joint_handles = Vec::new();
    for b in 0..NB {
        let env = if b == 0 { 0 } else { state.add_environment() };
        joint_handles.push(build_chain_env(&mut state, env));
    }
    let mut pipeline = NexusPipeline::default();
    state.finalize(&backend).await?;

    for _ in 0..150 {
        pipeline.simulate(&backend, &mut state, None).await?;
    }
    backend.synchronize()?;
    let a0_before = link1_angle(&backend, &state, 0).await;
    let a1_before = link1_angle(&backend, &state, 1).await;
    check(
        "both envs hold 0 before control",
        a0_before.abs() < 0.05 && a1_before.abs() < 0.05,
        format!(" ({a0_before:.4}, {a1_before:.4})"),
    );

    // Per-step control path: retarget ALL of env 0's joints to 0.5 through
    // the rapier world + one-write GPU sync. Env 1 untouched.
    let env0_handles = joint_handles[0].clone();
    state.control_multibody_motors(&backend, 0, |world| {
        for jh in &env0_handles {
            let (mb, lid) = world.multibody_joints.get_mut(*jh).expect("joint missing");
            let link = mb.link_mut(lid).expect("link missing");
            link.joint
                .data
                .set_motor_position(JointAxis::AngX, 0.5, KP, KD);
        }
    })?;

    for _ in 0..250 {
        pipeline.simulate(&backend, &mut state, None).await?;
    }
    backend.synchronize()?;
    let a0 = link1_angle(&backend, &state, 0).await;
    let a1 = link1_angle(&backend, &state, 1).await;
    check(
        "env0 tracks 0.5 after control_multibody_motors",
        (a0 - 0.5).abs() < 0.05,
        format!(" ({a0:.4})"),
    );
    check(
        "env1 unaffected (still holds 0)",
        a1.abs() < 0.05,
        format!(" ({a1:.4})"),
    );

    // read_links: de-SoA'd joint-state readback, per env.
    let rbd = state.rbd.as_ref().unwrap();
    let links0 = rbd.multibodies().read_links(&backend, 0).await;
    let links1 = rbd.multibodies().read_links(&backend, 1).await;
    check(
        "read_links returns per-batch link count",
        links0.len() == rbd.multibodies().links_per_batch() as usize && !links1.is_empty(),
        format!(" (len {})", links0.len()),
    );
    // Link entry 1 = first chain link (entry 0 is the fixed root). Its joint
    // coordinate for the revolute axis (AngX = coord slot 3) tracks 0.5 in
    // env 0 and 0 in env 1.
    let c0 = links0[1].coords[3];
    let c1 = links1[1].coords[3];
    check(
        "read_links coords: env0 ~0.5, env1 ~0",
        (c0 - 0.5).abs() < 0.05 && c1.abs() < 0.05,
        format!(" ({c0:.4}, {c1:.4})"),
    );
    // World pose in the workspace matches the body-pose buffer.
    let poses: Vec<glamx::Pose3> = backend.slow_read_vec(rbd.body_poses().buffer()).await.unwrap();
    let stride = poses.len() / NB;
    let dev = (links0[1].local_to_world.translation - poses[1].translation).length();
    let _ = stride;
    check(
        "read_links world pose matches body_poses",
        dev < 1.0e-5,
        format!(" (dev {dev:.2e})"),
    );

    if failures > 0 {
        anyhow::bail!("{failures} check(s) failed");
    }
    println!("all checks passed");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    pollster::block_on(run())
}
