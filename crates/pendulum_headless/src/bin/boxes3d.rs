//! Headless 3D rigid-body demo: a small pile of boxes dropped onto a floor,
//! simulated on the nexus GPU rbd pipeline (contacts + friction), recording each
//! box's full 6-DOF pose per step to CSV for 3D rendering.
//!
//! Run: `cargo run -p pendulum_headless --bin boxes3d --release [out.csv]`

use khal::backend::{Backend, GpuBackend as KhalGpuBackend, WebGpu};
use khal::re_exports::wgpu;
use nexus3d::rbd::math::Pose;
use nexus3d::rbd::pipeline::{GpuPhysicsPipeline, GpuPhysicsState};
use rapier3d::prelude::*;

const NX: i32 = 3;
const NY: i32 = 3; // layers
const NZ: i32 = 2;
const HALF: f32 = 0.5; // box half-extent (1.0 cube)
const SPACING: f32 = 1.15;
const DT: f32 = 1.0 / 60.0;
const STEPS: usize = 200;

struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn sym(&mut self, m: f32) -> f32 {
        (self.unit() * 2.0 - 1.0) * m
    }
}

fn num_boxes() -> usize {
    (NX * NY * NZ) as usize
}

/// Dynamic boxes (indices 0..N) above a fixed floor (index N). Slight random
/// position/orientation jitter so the stack topples instead of dropping cleanly.
fn build() -> (RigidBodySet, ColliderSet) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut rng = Lcg(0xC0FFEE);

    for iy in 0..NY {
        for ix in 0..NX {
            for iz in 0..NZ {
                let x = (ix as f32 - (NX - 1) as f32 * 0.5) * SPACING + rng.sym(0.05);
                let z = (iz as f32 - (NZ - 1) as f32 * 0.5) * SPACING + rng.sym(0.05);
                let y = 1.2 + iy as f32 * (2.0 * HALF + 0.25);
                let body = bodies.insert(
                    RigidBodyBuilder::dynamic()
                        .translation(Vec3::new(x, y, z))
                        .rotation(Vec3::new(rng.sym(0.15), rng.sym(0.15), rng.sym(0.15))),
                );
                colliders.insert_with_parent(
                    ColliderBuilder::cuboid(HALF, HALF, HALF).density(1.0),
                    body,
                    &mut bodies,
                );
            }
        }
    }

    // Fixed floor (top surface at y = 0), inserted last → pose index == num_boxes.
    let floor = bodies.insert(RigidBodyBuilder::fixed().translation(Vec3::new(0.0, -0.5, 0.0)));
    colliders.insert_with_parent(ColliderBuilder::cuboid(8.0, 0.5, 8.0), floor, &mut bodies);

    (bodies, colliders)
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

async fn read_poses(gpu: &KhalGpuBackend, state: &GpuPhysicsState) -> Vec<Pose> {
    gpu.slow_read_vec(state.poses().buffer())
        .await
        .expect("read poses")
}

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "/tmp/boxes3d.csv".to_string());
    let n = num_boxes();

    pollster::block_on(async {
        let (bodies, colliders) = build();
        let impulse_joints = ImpulseJointSet::new();
        let multibody_joints = MultibodyJointSet::new();
        let sim_params = nexus3d::rbd::dynamics::GpuSimParams::default();
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

        let p0 = read_poses(&gpu, &state).await;
        println!("{} poses total ({n} dynamic boxes + floor)", p0.len());

        // CSV (long format): step, box, x, y, z, qx, qy, qz, qw
        let mut csv = String::from("step,box,x,y,z,qx,qy,qz,qw\n");
        let mut dump = |step: usize, poses: &[Pose], csv: &mut String| {
            for b in 0..n {
                let p = poses[b];
                let t = p.translation;
                let q = p.rotation;
                csv.push_str(&format!(
                    "{step},{b},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
                    t.x, t.y, t.z, q.x, q.y, q.z, q.w
                ));
            }
        };
        dump(0, &p0, &mut csv);

        for step in 1..=STEPS {
            let _ = pipeline.step(&gpu, &mut state, None).await;
            gpu.synchronize().expect("sync");
            pipeline.auto_resize_buffers(&gpu, &mut state).await;
            let p = read_poses(&gpu, &state).await;
            dump(step, &p, &mut csv);
        }

        std::fs::write(&out, csv).expect("write csv");
        let last = read_poses(&gpu, &state).await;
        let ys: Vec<f32> = (0..n).map(|b| last[b].translation.y).collect();
        let miny = ys.iter().cloned().fold(f32::INFINITY, f32::min);
        let maxy = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("wrote {out}: {} steps × {n} boxes. settled y∈[{miny:.2},{maxy:.2}]", STEPS + 1);
    });
}
