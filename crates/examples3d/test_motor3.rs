//! End-to-end test of the FORCE_BASED PD-motor graft (M3 verification).
//!
//! 3 envs × 3-link chains under gravity, revolute joints about Z with
//! force-based PD motors. Checks:
//! 1. builder-configured PD holds the chain horizontal against gravity
//!    (vs. a motor-less chain that sags);
//! 2. `set_motor_position` retargets joints (tracks 0.5 rad);
//! 3. serial vs lane dynamics tier reach the same steady state;
//! 4. `scatter_motor_targets` (GPU action path) matches the setter path;
//! 5. actuator delay: with `k = ∞` and prev target 0, a new target is
//!    ignored (chain stays at 0), and two delay runs are bit-identical.

use glamx::Vec3;
use khal::backend::{Backend, GpuBackend, WebGpu};
use khal::re_exports::wgpu;
use nexus3d::prelude::{NexusCapacities, NexusPipeline, NexusState};
use rapier3d::prelude::*;

const NB: usize = 3;
const LINKS: usize = 3;
const KP: f32 = 2000.0;
const KD: f32 = 100.0;

fn build_chain_env(
    state: &mut NexusState,
    env: usize,
    motor_target: Option<f32>,
    kp: f32,
    kd: f32,
) {
    let rad = 0.4;
    let link_len = 2.0;
    let mut parent = state.insert_rigid_body_in(
        env,
        RigidBodyBuilder::fixed().build(),
        ColliderBuilder::cuboid(rad, rad, rad).build(),
    );
    let mut first_joint = None;
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
        let mut joint = RevoluteJointBuilder::new(Vec3::Z)
            .local_anchor1(parent_anchor)
            .local_anchor2(Vec3::new(-link_len * 0.8, 0.0, 0.0));
        if let Some(t) = motor_target {
            // NEXUS_TEST_ACCEL=1: A/B against the untouched constraint path.
            let model = if std::env::var("NEXUS_TEST_ACCEL").is_ok() {
                MotorModel::AccelerationBased
            } else {
                MotorModel::ForceBased
            };
            joint = joint
                .motor_model(model)
                .motor_position(t, kp, kd)
                .motor_max_force(1.0e6);
        }
        let jh = state
            .insert_multibody_joint_in(env, parent, handle, joint.build())
            .expect("invalid multibody chain");
        first_joint.get_or_insert(jh);
        parent = handle;
    }

    // Robot-style setup: links of the same multibody never collide with each
    // other. Cold-start transients otherwise bend the chain into self-contact
    // and the contact bounce + PD sustains a limit cycle, confounding the
    // steady-state motor checks.
    let jh = first_joint.expect("chain has at least one joint");
    let world = state.rbd_world_mut(env);
    let (mb, _) = world
        .multibody_joints
        .get_mut(jh)
        .expect("chain multibody missing");
    mb.set_self_contacts_enabled(false);
}

fn build(motor_target: Option<f32>) -> NexusState {
    build_g(motor_target, KP, KD)
}

fn build_g(motor_target: Option<f32>, kp: f32, kd: f32) -> NexusState {
    let mut state = NexusState::new(NexusCapacities::default().rbd_collisions(256));
    for b in 0..NB {
        let env = if b == 0 { 0 } else { state.add_environment() };
        build_chain_env(&mut state, env, motor_target, kp, kd);
    }
    state
}

/// First link's rotation angle about Z, per env.
async fn link1_angles(backend: &GpuBackend, state: &NexusState) -> Vec<f32> {
    let rbd = state.rbd.as_ref().unwrap();
    let poses: Vec<glamx::Pose3> = backend.slow_read_vec(rbd.body_poses().buffer()).await.unwrap();
    let stride = poses.len() / NB;
    if std::env::var("NEXUS_DBG_ALL").is_ok() {
        for l in 1..=LINKS {
            let q = poses[l].rotation;
            println!("  [dbg] link{l} env0 angle {}", 2.0 * q.z.atan2(q.w));
        }
        let mb = rbd.multibodies();
        let mut ds: Vec<f32> = vec![0.0; mb.dof_state().len() as usize];
        backend.slow_read_buffer(mb.dof_state().buffer(), &mut ds).await.unwrap();
        let mut dv: Vec<f32> = vec![0.0; mb.dof_values().len() as usize];
        backend.slow_read_buffer(mb.dof_values().buffer(), &mut dv).await.unwrap();
        let nb = mb.num_batches() as usize;
        println!(
            "  [dbg] env0 dof vels {:?} coords {:?}",
            (0..3).map(|d| ds[d * nb]).collect::<Vec<_>>(),
            (0..3).map(|d| dv[d * nb]).collect::<Vec<_>>(),
        );
    }
    (0..NB)
        .map(|e| {
            let q = poses[e * stride + 1].rotation;
            2.0 * q.z.atan2(q.w)
        })
        .collect()
}

async fn step_n(
    backend: &GpuBackend,
    pipeline: &mut NexusPipeline,
    state: &mut NexusState,
    n: usize,
) -> anyhow::Result<()> {
    for _ in 0..n {
        pipeline.simulate(backend, state, None).await?;
    }
    backend.synchronize()?;
    Ok(())
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

    let only: Option<usize> = std::env::args().nth(1).and_then(|a| a.parse().ok());
    let mut failures = 0usize;
    let mut check = |name: &str, ok: bool, detail: String| {
        println!("  [{}] {name}{detail}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            failures += 1;
        }
    };

    // 1. PD holds vs gravity; motor-less chain sags.
    if only.is_none() || only == Some(1) || only == Some(2) {
    let mut held = build(Some(0.0));
    let mut pipeline = NexusPipeline::default();
    held.finalize(&backend).await?;
    step_n(&backend, &mut pipeline, &mut held, 300).await?;
    let a_held = link1_angles(&backend, &held).await;

    let mut sag = build(None);
    let mut pipeline2 = NexusPipeline::default();
    sag.finalize(&backend).await?;
    step_n(&backend, &mut pipeline2, &mut sag, 300).await?;
    let a_sag = link1_angles(&backend, &sag).await;

    check(
        "PD holds chain horizontal vs gravity",
        a_held.iter().all(|a| a.abs() < 0.1),
        format!(" (angles {a_held:?})"),
    );
    check(
        "motor-less chain sags (control)",
        a_sag.iter().all(|a| a.abs() > 0.2),
        format!(" (angles {a_sag:?})"),
    );

    // 2. set_motor_position retargets to 0.5 rad.
    {
        let mb = held.rbd.as_mut().unwrap().multibodies_mut();
        for env in 0..NB as u32 {
            for link in 1..=LINKS as u32 {
                mb.set_motor_position(&backend, env, link, JointAxis::AngX, 0.5)?;
            }
        }
    }
    step_n(&backend, &mut pipeline, &mut held, 400).await?;
    let a_track = link1_angles(&backend, &held).await;
    // Droop-relative: the explicit PD carries a gravity droop of tau/kp on
    // both setpoints, so the DELTA isolates target tracking.
    check(
        "set_motor_position tracks +0.5 rad from hold",
        a_track
            .iter()
            .zip(a_held.iter())
            .all(|(t, h)| ((t - h) - 0.5).abs() < 0.06),
        format!(" (angles {a_track:?}, hold {a_held:?})"),
    );

    } // scenarios 1-2
    if only.is_none() || only == Some(3) {
    // 3. Serial vs lane tier: SHORT-horizon parity (50 steps). The serial
    // tier's baseline numerics diverge from the lane tier over long horizons
    // (documented in MIGRATION_GOLDENS.md; observable even with no motors),
    // so this checks only that the PD block in the serial kernel applies the
    // same torque while the trajectories are still close.
    let mut tier_angles: Vec<Vec<f32>> = Vec::new();
    for tier in ["0", "1"] {
        // Single-threaded test binary; no other thread reads the env.
        unsafe { std::env::set_var("NEXUS_SERIAL_MB", tier) };
        let mut st = build_g(Some(0.5), 500.0, 50.0);
        let mut pl = NexusPipeline::default();
        st.finalize(&backend).await?;
        step_n(&backend, &mut pl, &mut st, 50).await?;
        tier_angles.push(link1_angles(&backend, &st).await);
    }
    unsafe { std::env::remove_var("NEXUS_SERIAL_MB") };
    let (a_lane, a_serial) = (&tier_angles[0], &tier_angles[1]);
    let max_tier_dev = a_lane
        .iter()
        .zip(a_serial.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    check(
        "serial tier PD matches lane tier at short horizon",
        max_tier_dev < 0.05,
        format!(" (max dev {max_tier_dev:.2e}, lane {a_lane:?}, serial {a_serial:?})"),
    );

    } // scenario 3
    if only.is_none() || only == Some(4) {
    // 4. GPU scatter path drives joint 1 to 0.5 (links 2..: builder-held 0).
    let mut sc = build(Some(0.0));
    let mut pl = NexusPipeline::default();
    sc.finalize(&backend).await?;
    sc.rbd
        .as_mut()
        .unwrap()
        .multibodies_mut()
        .scatter_motor_targets(&backend, &[0.5; NB], &[1], JointAxis::AngX as u32)?;
    if std::env::var("NEXUS_DBG").is_ok() {
    backend.synchronize()?;
    {
        use nexus3d::rbd::shaders::dynamics::MultibodyLinkStatic;
        let mb = sc.rbd.as_ref().unwrap().multibodies();
        let mut ls: Vec<MultibodyLinkStatic> =
            bytemuck::zeroed_vec(mb.dbg_links_static().len() as usize);
        backend.slow_read_buffer(mb.dbg_links_static().buffer(), &mut ls).await.unwrap();
        let l1e0 = &ls[NB]; // link 1, env 0 (interleaved: 1*NB+0)
        println!(
            "  [dbg] link1/env0 after scatter: target_pos[3]={} motor_axes={:#x}",
            l1e0.data.motors[3].target_pos, l1e0.data.motor_axes
        );
    }
    }
    step_n(&backend, &mut pl, &mut sc, 400).await?;
    let a_scatter = link1_angles(&backend, &sc).await;
    check(
        "scatter_motor_targets tracks 0.5 rad (within droop band)",
        a_scatter.iter().all(|a: &f32| (a - 0.5).abs() < 0.15),
        format!(" (angles {a_scatter:?})"),
    );

    } // scenario 4
    if only.is_none() || only == Some(5) {
    // 5. Actuator delay: k = ∞, prev target 0 → new target 0.5 is ignored.
    let mut delay_angles: Vec<Vec<f32>> = Vec::new();
    for _rep in 0..2 {
        let mut st = build(Some(0.0));
        let mut pl = NexusPipeline::default();
        st.finalize(&backend).await?;
        {
            let mb = st.rbd.as_mut().unwrap().multibodies_mut();
            let stride = mb.motor_delay_stride() as usize;
            let mut delay = vec![0.0f32; stride * NB];
            for env in 0..NB {
                delay[env * stride + 1] = 1.0e9; // k = forever
                // prev targets stay 0.0
            }
            mb.write_motor_delay_state(&backend, &delay)?;
            for env in 0..NB as u32 {
                for link in 1..=LINKS as u32 {
                    mb.set_motor_position(&backend, env, link, JointAxis::AngX, 0.5)?;
                }
            }
        }
        step_n(&backend, &mut pl, &mut st, 200).await?;
        delay_angles.push(link1_angles(&backend, &st).await);
    }
    let (a_delay1, a_delay2) = (&delay_angles[0], &delay_angles[1]);
    check(
        "delayed PD ignores new target (holds prev 0)",
        a_delay1.iter().all(|a| a.abs() < 0.1 && (a - 0.5).abs() > 0.3),
        format!(" (angles {a_delay1:?})"),
    );
    check(
        "delay runs deterministic (bit-identical)",
        a_delay1
            .iter()
            .zip(a_delay2.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        format!(" ({a_delay1:?} vs {a_delay2:?})"),
    );

    } // scenario 5
    if failures > 0 {
        anyhow::bail!("{failures} check(s) failed");
    }
    println!("all checks passed");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    pollster::block_on(run())
}
