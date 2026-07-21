//! End-to-end test of the RL per-env reset primitives (M2 verification).
//!
//! Builds 5 identical chain envs (deliberately non-power-of-two so an
//! interleaved-addressing off-by-one corrupts a detectable env), then:
//! 1. steps 10; snapshots (captures env 0 @ t10);
//! 2. steps 15 more (t25), records all poses;
//! 3. resets env 2 from the snapshot — env 2 must equal the t10 state
//!    BIT-EXACTLY, envs 0/1/3/4 must keep their t25 state BIT-EXACTLY;
//! 4. steps 15 more — env 2 must replay to the recorded t25 state (full
//!    multibody workspace/DOF restore, not just body poses; tolerance 1e-4
//!    for atomic-order nondeterminism in the contact path);
//! 5. offset-reset on env 3: the chain is FIXED-base, so `translated` must
//!    be a no-op — env 3 must equal the plain t10 state;
//! 6. `reserve_contacts` regrow + 5 more steps must stay finite.

use glamx::Vec3;
use khal::backend::{Backend, GpuBackend, WebGpu};
use khal::re_exports::wgpu;
use nexus3d::prelude::{NexusCapacities, NexusPipeline, NexusState};
use rapier3d::prelude::*;

fn build_chain_env(state: &mut NexusState, env: usize, num_links: usize) {
    let rad = 0.4;
    let link_len = 2.0;
    let mut parent = state.insert_rigid_body_in(
        env,
        RigidBodyBuilder::fixed().build(),
        ColliderBuilder::cuboid(rad, rad, rad).build(),
    );
    for i in 0..num_links {
        let x = (i as f32 + 1.0) * link_len;
        let handle = state.insert_rigid_body_in(
            env,
            RigidBodyBuilder::dynamic()
                .translation(Vec3::new(x, 0.0, 0.0))
                .build(),
            ColliderBuilder::cuboid(link_len * 0.55, rad, rad).build(),
        );
        let parent_anchor = if i == 0 {
            Vec3::ZERO
        } else {
            Vec3::new(link_len * 0.8, 0.0, 0.0)
        };
        let joint = RevoluteJointBuilder::new(Vec3::Z)
            .local_anchor1(parent_anchor)
            .local_anchor2(Vec3::new(-link_len * 0.8, 0.0, 0.0))
            .build();
        state
            .insert_multibody_joint_in(env, parent, handle, joint)
            .expect("invalid multibody chain");
        parent = handle;
    }
}

async fn read_poses(backend: &GpuBackend, state: &NexusState) -> Vec<glamx::Pose3> {
    let rbd = state.rbd.as_ref().expect("rbd state missing");
    backend
        .slow_read_vec(rbd.body_poses().buffer())
        .await
        .expect("pose readback failed")
}

fn env_slab(poses: &[glamx::Pose3], env: usize, nb: usize) -> &[glamx::Pose3] {
    let stride = poses.len() / nb;
    &poses[env * stride..(env + 1) * stride]
}

fn slabs_bit_equal(a: &[glamx::Pose3], b: &[glamx::Pose3]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.translation == y.translation && x.rotation == y.rotation)
}

fn slabs_close(a: &[glamx::Pose3], b: &[glamx::Pose3], tol: f32) -> (bool, f32) {
    let mut max = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        max = max.max((x.translation - y.translation).length());
    }
    (max <= tol, max)
}

async fn run() -> anyhow::Result<()> {
    const NB: usize = 5;
    const LINKS: usize = 6;
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

    let capacities = NexusCapacities::default().rbd_collisions(256);
    let mut state = NexusState::new(capacities);
    for b in 0..NB {
        let env = if b == 0 { 0 } else { state.add_environment() };
        build_chain_env(&mut state, env, LINKS);
    }
    let mut pipeline = NexusPipeline::default();
    state.finalize(&backend).await?;

    let mut failures = 0usize;
    let mut check = |name: &str, ok: bool, detail: String| {
        println!("  [{}] {name}{detail}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            failures += 1;
        }
    };

    for _ in 0..10 {
        pipeline.simulate(&backend, &mut state, None).await?;
    }
    backend.synchronize()?;
    let p10 = read_poses(&backend, &state).await;
    let snap = state.rbd.as_ref().unwrap().snapshot(&backend).await;

    for _ in 0..15 {
        pipeline.simulate(&backend, &mut state, None).await?;
    }
    backend.synchronize()?;
    let p25 = read_poses(&backend, &state).await;

    // 3. Reset env 2 to the snapshot.
    state
        .rbd
        .as_mut()
        .unwrap()
        .reset_env_from_snapshot(&backend, 2, &snap);
    backend.synchronize()?;
    let pr = read_poses(&backend, &state).await;

    check(
        "env2 == t10 after reset (bit-exact)",
        slabs_bit_equal(env_slab(&pr, 2, NB), env_slab(&p10, 0, NB)),
        String::new(),
    );
    for e in [0usize, 1, 3, 4] {
        check(
            &format!("env{e} untouched by env2 reset (bit-exact)"),
            slabs_bit_equal(env_slab(&pr, e, NB), env_slab(&p25, e, NB)),
            String::new(),
        );
    }

    // 4. Replay: env 2 must re-trace t10 -> t25.
    for _ in 0..15 {
        pipeline.simulate(&backend, &mut state, None).await?;
    }
    backend.synchronize()?;
    let p40 = read_poses(&backend, &state).await;
    let (ok, max) = slabs_close(env_slab(&p40, 2, NB), env_slab(&p25, 0, NB), 1e-4);
    check("env2 replays t10->t25 trajectory", ok, format!(" (max dev {max:.2e})"));

    // 5. Fixed-base offset reset must ignore the offset entirely.
    let rbd_snap = state.rbd.as_ref().unwrap().snapshot(&backend).await;
    state.rbd.as_mut().unwrap().reset_env_from_snapshot_offset(
        &backend,
        3,
        &rbd_snap,
        Vec3::new(100.0, 0.0, 0.0),
    );
    backend.synchronize()?;
    let po = read_poses(&backend, &state).await;
    let (ok, max) = slabs_close(env_slab(&po, 3, NB), env_slab(&p40, 0, NB), 1e-6);
    check(
        "fixed-base offset reset ignores offset",
        ok,
        format!(" (max dev {max:.2e})"),
    );

    // 6. reserve_contacts regrow + stability.
    state.rbd.as_mut().unwrap().reserve_contacts(&backend, 2048);
    for _ in 0..5 {
        pipeline.simulate(&backend, &mut state, None).await?;
    }
    backend.synchronize()?;
    let pf = read_poses(&backend, &state).await;
    let finite = pf.iter().all(|p| p.translation.is_finite());
    check("finite after reserve_contacts(2048) + 5 steps", finite, String::new());

    if failures > 0 {
        anyhow::bail!("{failures} check(s) failed");
    }
    println!("all checks passed");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    pollster::block_on(run())
}
