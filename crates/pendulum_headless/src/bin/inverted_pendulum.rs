//! Controlled inverted pendulum on the nexus GPU rbd pipeline.
//!
//! A single rod on a revolute joint (axis Z), starting ~30° off vertical. A PD
//! controller commands the joint *velocity* motor each step to drive the rod
//! toward upright and hold it there. We measure the fraction of steps the rod
//! spends "upside down" (balanced near vertical) — the classic inverted-pendulum
//! score — and contrast it with an unactuated baseline that just falls.
//!
//! Run: `cargo run -p pendulum_headless --bin inverted_pendulum --release`

use std::f32::consts::PI;

use khal::backend::{Backend, GpuBackend as KhalGpuBackend, WebGpu};
use khal::re_exports::wgpu;
use nexus3d::rbd::dynamics::GpuSimParams;
use nexus3d::rbd::math::Pose;
use nexus3d::rbd::pipeline::{GpuPhysicsPipeline, GpuPhysicsState};
use rapier3d::prelude::*;

// --- scene ---
const ROD_LEN: f32 = 2.0; // pivot-to-tip, pole lies along the body's local +X
const START_TILT: f32 = 30.0 * PI / 180.0; // initial angle away from vertical
const DT: f32 = 1.0 / 60.0;
const STEPS: usize = 300; // 5 s at 1/60

// --- controller (velocity-PD on the joint angle) ---
// Gravity torque on this rod is ~1.5 Nm, so the motor only needs a modest force
// budget; an over-large max_force flings the (light) rod and the loop diverges.
const KP: f32 = 5.0;
const KD: f32 = 0.5;
const VMAX: f32 = 8.0; // rad/s clamp on the commanded joint velocity
const MOTOR_DAMPING: f32 = 50.0; // stiff velocity tracking so the motor follows cmd
const MOTOR_MAX_FORCE: f32 = 50.0; // Nm the motor may exert (gravity torque ~1.5 Nm)
// LINK_ID and MOTOR_SIGN are taken from argv (defaults below) so we can probe
// the right link index / motor polarity without recompiling:
//   inverted_pendulum [link_id] [motor_sign]
const DEFAULT_LINK_ID: u32 = 1; // dynamic link (root = 0); commands land here
const DEFAULT_MOTOR_SIGN: f32 = 1.0; // +AngX velocity increases θ (toward upright)

const UPRIGHT: f32 = PI / 2.0; // pole pointing +Y, measured from +X axis
const BALANCED_TOL: f32 = 25.0 * PI / 180.0; // "upside down" = within this of vertical

/// Build the single-rod pendulum. With `motor`, the revolute joint carries a
/// velocity motor (damping + max force) so the runtime PD has authority.
fn build(motor: bool) -> (RigidBodySet, ColliderSet, MultibodyJointSet, GpuSimParams) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut multibody_joints = MultibodyJointSet::new();

    // Fixed pivot at the origin.
    let root = bodies.insert(RigidBodyBuilder::fixed());
    colliders.insert_with_parent(ColliderBuilder::cuboid(0.1, 0.1, 0.1), root, &mut bodies);

    // Rod: pole along local +X. Rotate by `theta0` about Z so it starts tilted,
    // and place the COM so the rod's inner end sits on the pivot.
    let theta0 = UPRIGHT - START_TILT; // 60° from +X axis
    let com = Vec3::new(ROD_LEN * theta0.cos(), ROD_LEN * theta0.sin(), 0.0);
    let rod = bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(com)
            .rotation(Vec3::new(0.0, 0.0, theta0)),
    );
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(ROD_LEN * 0.5, 0.1, 0.1).density(1.0),
        rod,
        &mut bodies,
    );

    let mut joint = RevoluteJointBuilder::new(Vec3::Z)
        .local_anchor1(Vec3::ZERO) // on the pivot
        .local_anchor2(Vec3::new(-ROD_LEN, 0.0, 0.0)); // rod's inner end
    if motor {
        joint = joint
            .motor_velocity(0.0, MOTOR_DAMPING)
            .motor_max_force(MOTOR_MAX_FORCE);
    }
    multibody_joints.insert(root, rod, joint.build(), true);

    let mut sim_params = GpuSimParams::default();
    sim_params.dt = DT;
    sim_params.num_solver_iterations = 8;

    (bodies, colliders, multibody_joints, sim_params)
}

async fn webgpu_backend() -> KhalGpuBackend {
    let limits = wgpu::Limits {
        max_buffer_size: 1_200_000_000,
        max_storage_buffer_binding_size: 1_200_000_000,
        max_storage_buffers_per_shader_stage: 14,
        max_compute_workgroup_storage_size: 19_904,
        ..Default::default()
    };
    let mut webgpu = WebGpu::new(wgpu::Features::default(), limits)
        .await
        .expect("init WebGPU");
    webgpu.force_buffer_copy_src = true;
    KhalGpuBackend::WebGpu(webgpu)
}

/// Pole angle from the +X axis (UPRIGHT = +90°), from the rod COM world pose.
fn pole_angle(poses: &[Pose]) -> f32 {
    let com = poses.last().expect("rod pose").translation;
    com.y.atan2(com.x)
}

async fn read_poses(gpu: &KhalGpuBackend, state: &GpuPhysicsState) -> Vec<Pose> {
    gpu.slow_read_vec(state.poses().buffer())
        .await
        .expect("read poses")
}

async fn run(label: &str, controlled: bool, link_id: u32, motor_sign: f32) -> f32 {
    let (bodies, colliders, multibody_joints, sim_params) = build(controlled);
    let impulse_joints = ImpulseJointSet::new();
    let gpu = webgpu_backend().await;
    let pipeline = GpuPhysicsPipeline::from_backend(&gpu);
    let envs = vec![(
        &bodies,
        &colliders,
        &impulse_joints,
        &multibody_joints,
        &sim_params,
    )];
    let mut state = GpuPhysicsState::from_rapier(&gpu, &envs);

    let poses0 = read_poses(&gpu, &state).await;
    println!("[{label}] {} poses:", poses0.len());
    for (i, p) in poses0.iter().enumerate() {
        let t = p.translation;
        println!(
            "    pose[{i}] = ({:>6.2}, {:>6.2}, {:>6.2})  θ={:>6.1}°",
            t.x,
            t.y,
            t.z,
            t.y.atan2(t.x).to_degrees()
        );
    }

    let mut theta = pole_angle(&read_poses(&gpu, &state).await);
    let mut prev_theta = theta;
    let mut balanced_steps = 0usize;
    println!(
        "[{label}] start θ = {:.1}° (target 90°), {STEPS} steps",
        theta.to_degrees()
    );

    for step in 1..=STEPS {
        let mut cmd = 0.0;
        if controlled {
            // Velocity-PD toward upright; θ̇ from the last step's finite difference.
            let err = UPRIGHT - theta;
            let theta_dot = (theta - prev_theta) / DT;
            cmd = (KP * err - KD * theta_dot).clamp(-VMAX, VMAX) * motor_sign;
            let _ = state
                .multibodies_mut()
                .set_motor_velocity(&gpu, 0, link_id, JointAxis::AngX, cmd);
        }

        let _ = pipeline.step(&gpu, &mut state, None).await;
        gpu.synchronize().expect("sync");
        pipeline.auto_resize_buffers(&gpu, &mut state).await;

        prev_theta = theta;
        theta = pole_angle(&read_poses(&gpu, &state).await);

        if (theta - UPRIGHT).abs() < BALANCED_TOL {
            balanced_steps += 1;
        }
        if controlled && step <= 12 {
            println!(
                "  step {step:>3}: θ = {:>6.1}°  cmd = {:>6.2} rad/s",
                theta.to_degrees(),
                cmd
            );
        } else if step % 50 == 0 || step == STEPS {
            println!("  step {step:>3}: θ = {:>6.1}°", theta.to_degrees());
        }
    }

    let frac = balanced_steps as f32 / STEPS as f32;
    println!(
        "[{label}] upside-down {balanced_steps}/{STEPS} steps = {:.0}%  (~{:.2}s of {:.2}s)",
        frac * 100.0,
        balanced_steps as f32 * DT,
        STEPS as f32 * DT,
    );
    frac
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let link_id = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_LINK_ID);
    let motor_sign = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MOTOR_SIGN);
    println!("(link_id = {link_id}, motor_sign = {motor_sign:+})");

    pollster::block_on(async {
        let ctrl = run("controlled", true, link_id, motor_sign).await;
        let base = run("baseline", false, link_id, motor_sign).await;
        println!(
            "\ninverted pendulum — controlled held upright {:.0}% vs baseline {:.0}%",
            ctrl * 100.0,
            base * 100.0
        );
    });
}
