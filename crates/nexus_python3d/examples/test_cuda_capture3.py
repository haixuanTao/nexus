"""M6 CUDA-graph capture smoke: native CUDA backend, capture, replay, timing.

Headless, real robot MJCF per env (override with NEXUS_SMOKE_MJCF).
1. simulate N steps on the CUDA backend — physics advances, finite;
2. capture the rbd step into a CUDA graph;
3. replay it M times — physics continues advancing, finite;
4. replayed-step wall time vs simulate() wall time (the submission-floor
   payoff this migration was structured around).
"""

import os
import sys
import time

import numpy as np
from nexus3d import NexusPipeline, NexusState, NexusViewer

MJCF = os.environ.get(
    "NEXUS_SMOKE_MJCF",
    "/home/champagne/Documents/work/sim2sim/src/sim2sim/assets/quad12/quad12.xml",
)
N_ENVS = int(os.environ.get("NEXUS_SMOKE_ENVS", "1024"))

failures = 0


def check(name, ok, detail=""):
    global failures
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{detail}")
    if not ok:
        failures += 1


print("[stage] creating viewer...", flush=True)
viewer = NexusViewer(320, 240, headless=True).with_cuda()
viewer.init_backend()
print("[stage] cuda backend up", flush=True)
state = NexusState()
state.set_rbd_collisions_capacity(256)

for e in range(N_ENVS):
    env = 0 if e == 0 else state.add_environment()
    state.insert_mjcf(viewer, MJCF, render_colliders=False, env=env)

print("[stage] envs inserted", flush=True)
pipeline = NexusPipeline()
state.finalize(viewer)
print("[stage] finalized", flush=True)

# Warmup (buffer growth after capture invalidates the graph).
for i in range(20):
    pipeline.simulate(viewer, state)
    if i == 0:
        print("[stage] first step done", flush=True)
print("[stage] warmup done", flush=True)

coords0 = viewer.read_multibody_links(state, 0)[0].copy()

t0 = time.perf_counter()
for _ in range(50):
    pipeline.simulate(viewer, state)
t_simulate = (time.perf_counter() - t0) / 50

coords1 = viewer.read_multibody_links(state, 0)[0].copy()
check(
    "CUDA backend simulates (physics advances, finite)",
    np.isfinite(coords1).all() and np.abs(coords1 - coords0).max() > 1e-6,
    f" (max dcoord {np.abs(coords1 - coords0).max():.2e})",
)

ok = pipeline.capture_cuda_graph(viewer, state)
check("capture_cuda_graph", bool(ok))

t0 = time.perf_counter()
N_REPLAY = 500
for _ in range(N_REPLAY):
    pipeline.replay_cuda_graph()
# The readback forces completion of the async replays — sync-bounded timing.
coords2 = viewer.read_multibody_links(state, 0)[0].copy()
t_replay = (time.perf_counter() - t0) / N_REPLAY
check(
    "replayed graph advances physics (finite)",
    np.isfinite(coords2).all() and np.abs(coords2 - coords1).max() > 1e-6,
    f" (max dcoord {np.abs(coords2 - coords1).max():.2e})",
)

speedup = t_simulate / t_replay if t_replay > 0 else float("inf")
print(
    f"  [info] {N_ENVS} envs: simulate {t_simulate*1e6:.0f} us/step, "
    f"replay {t_replay*1e6:.0f} us/step ({speedup:.1f}x), "
    f"{N_ENVS/t_replay:,.0f} env-steps/s replayed"
)
check("replay not slower than simulate", t_replay <= t_simulate * 1.2)

print("all checks passed" if failures == 0 else f"{failures} check(s) FAILED")
sys.exit(1 if failures else 0)
