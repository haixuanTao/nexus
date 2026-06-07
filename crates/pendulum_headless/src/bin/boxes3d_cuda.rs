//! CUDA-backend twin of `boxes3d` — identical simulation, but runs the nexus
//! rbd pipeline on the native-CUDA (cuda-oxide cubin) backend instead of WebGPU.
//! Dumps the same per-step pose CSV so the two backends can be diffed for
//! bit-exactness of the ported physics kernels.
//!
//! Run: `cargo run -p pendulum_headless --bin boxes3d_cuda --release [out.csv]`

use khal::backend::{Backend, Cuda, GpuBackend as KhalGpuBackend};
use nexus3d::rbd::math::Pose;
use nexus3d::rbd::pipeline::{GpuPhysicsPipeline, GpuPhysicsState};
use rapier3d::prelude::*;

const NX: i32 = 3;
const NY: i32 = 3;
const NZ: i32 = 2;
const HALF: f32 = 0.5;
const SPACING: f32 = 1.15;
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

    let floor = bodies.insert(RigidBodyBuilder::fixed().translation(Vec3::new(0.0, -0.5, 0.0)));
    colliders.insert_with_parent(ColliderBuilder::cuboid(8.0, 0.5, 8.0), floor, &mut bodies);

    (bodies, colliders)
}

async fn read_poses(gpu: &KhalGpuBackend, state: &GpuPhysicsState) -> Vec<Pose> {
    gpu.slow_read_vec(state.poses().buffer())
        .await
        .expect("read poses")
}

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "/tmp/boxes3d_cuda.csv".to_string());
    let n = num_boxes();

    pollster::block_on(async {
        let (bodies, colliders) = build();
        let impulse_joints = ImpulseJointSet::new();
        let multibody_joints = MultibodyJointSet::new();
        let sim_params = nexus3d::rbd::dynamics::GpuSimParams::default();
        let gpu = KhalGpuBackend::Cuda(Cuda::new(0).expect("init CUDA backend"));
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
        println!("[cuda] {} poses total ({n} dynamic boxes + floor)", p0.len());

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
            eprintln!("[STEP {step}]");
            let _ = pipeline.step(&gpu, &mut state, None).await;
            gpu.synchronize().expect("sync");
            if step == 23 {
                let cpl: Vec<u32> = gpu.slow_read_vec(state.dbg_collision_pairs_len().buffer()).await.unwrap();
                let cl: Vec<u32> = gpu.slow_read_vec(state.dbg_contacts_len().buffer()).await.unwrap();
                let pairs: Vec<[u32; 2]> = gpu.slow_read_vec(state.dbg_collision_pairs().buffer()).await.unwrap();
                let cap = state.dbg_contacts_capacity();
                eprintln!("DUMP cap={} nb={} collision_pairs_len={:?} contacts_len={:?} pairs_buf={}",
                    cap, state.dbg_num_batches(), &cpl[..cpl.len().min(2)], &cl[..cl.len().min(2)], pairs.len());
                let n = (cpl.get(0).copied().unwrap_or(0) as usize).min(60).min(pairs.len());
                for k in 0..n { eprintln!("PAIR {} = [{}, {}]", k, pairs[k][0], pairs[k][1]); }
                let contacts: Vec<nexus3d::rbd::queries::GpuIndexedContact> =
                    gpu.slow_read_vec(state.dbg_contacts().buffer()).await.unwrap();
                let stride = core::mem::size_of::<nexus3d::rbd::queries::GpuIndexedContact>() / 4;
                let raw: &[u32] = unsafe {
                    core::slice::from_raw_parts(contacts.as_ptr() as *const u32, contacts.len() * stride)
                };
                eprintln!("CONTACT struct stride(u32)={}", stride);
                for k in 0..contacts.len().min(cap as usize) {
                    let base = k * stride;
                    // dump the whole struct's u32 words for the first few non-empty slots
                    let words = &raw[base..base + stride];
                    if words.iter().any(|&w| w != 0) {
                        eprintln!("CONTACT {} words={:?}", k, words);
                    }
                }
            }

            pipeline.auto_resize_buffers(&gpu, &mut state).await;
            let p = read_poses(&gpu, &state).await;
            dump(step, &p, &mut csv);
        }

        std::fs::write(&out, csv).expect("write csv");
        let last = read_poses(&gpu, &state).await;
        let ys: Vec<f32> = (0..n).map(|b| last[b].translation.y).collect();
        let miny = ys.iter().cloned().fold(f32::INFINITY, f32::min);
        let maxy = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("[cuda] wrote {out}: {} steps × {n} boxes. settled y∈[{miny:.2},{maxy:.2}]", STEPS + 1);
    });
}
