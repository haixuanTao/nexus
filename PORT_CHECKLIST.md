# Rebase onto `dimforge/rustgpu-clean` — port checklist

Goal: move our native-CUDA biped fork off the diverged `feat/per-env-parallelism`
branch (37 commits, merge-base `477854f`) and onto current upstream
`dimforge/rustgpu-clean` (3316a10), keeping our perf + physics features.
This is a **re-port, not a merge** — upstream rewrote the same solver/pipeline
incompatibly (see memory `nexus-upstream-divergence`).

Branch for the work: `rebase/onto-rustgpu-clean` (this branch).

---

## PROBE FINDINGS (2026-06-23, parallel to v58 training)

- **Deps fetch OK** here (crates.io + git resolve; only GitHub *API* is blocked).
- **Gate, structurally answered → cooperative re-port IS in scope.** Upstream's
  per-articulation solve is still `threads(1)` (contact + joint init/solve) vs our
  `threads(32)` lane-split. For our 1-biped-per-env config that serial LU+PGS is
  the bottleneck our rewrite removed (finalize 11.6×, joint-init 4.5×). A *measured*
  number needs a harness we'd port anyway (upstream has no headless CUDA bench —
  `pendulum_headless` is ours; upstream only has kiss3d GUI benches).
- **BACKEND FORK-IN-THE-ROAD (the big decision).** `rustgpu-clean` HAS a `cuda`
  feature but it compiles shaders via **khal's `cargo cuda` codegen** (khal-builder
  0.2 shells out to `cargo cuda build`). Probe build: Rust compiled fine
  (nexus_rbd_shaders3d 12.8s) but died at codegen — "Codegen backend not found, run
  `cargo cuda install`". `cargo-cuda` binary is present; its rustc codegen backend
  is NOT installed. Our cuda-oxide+libNVVM path exists precisely to bypass this
  (system ptxas = CUDA 12.0, NO sm_120; libNVVM works on Blackwell — see memory
  native-cuda-build-pinning).
  - **Option A**: `cargo cuda install` + use khal/cuda → drop our cuda-oxide/build_cuda
    burden. RISK: khal's codegen must clear the sm_120/Blackwell bar that drove us to
    cuda-oxide; and `cargo cuda install` mutates the pinned toolchain.
  - **Option B**: re-port our cuda-oxide path (1469f56 + build_cuda/) onto upstream.
    Known-good for sm_120; more porting; keeps the maintenance burden.
  - DECISION PENDING (user). Lean B for reliability unless khal/cuda is verified on sm_120.

## 0. THE GATE (decide scope first — needs GPU)

Our headline perf work is the cooperative-solver rewrite (`240874d`+`85246ac`:
+41% throughput @ N=8192, Isaac gap 4.8×→3.2×). Upstream *also* parallelized
since the fork (impulse-joint-solver parallelism + batch-sim rework).

**Benchmark upstream `rustgpu-clean` vs our fork at N=8192 before porting anything.**
- If upstream ≈ our throughput → **skip** re-porting `240874d`+`85246ac` (the
  multi-day part). Only re-apply physics features + host API. ~1 day.
- If upstream slower → re-port the cooperative lane-split onto their model. Multi-day.

Until this is answered, treat the cooperative rewrite as "maybe skip."

---

## 1. Commit inventory (merge-base `477854f` .. our HEAD `ef501d3`)

### A. Already upstreaming (drop from the port if merged)
- `2c2c31f` motor by-value → PR `upstream-pr/mb-motor-byvalue` (fe7d440). READY.
- `7368be7` per-env collider friction → PR `upstream-pr/mb-per-env-friction` (223f3c5). READY.
- `fa8b551` armature + frictionloss + ERP → NOT pushed (new bindings + known
  binding/layout fragility; needs cubin build + numerical validation; split armature out).

### B. Physics features to re-port (localized; do these first after the gate)
| Commit | Files | Adaptation onto upstream |
|---|---|---|
| `c647b65` force-based PD motor + contact-stiffness DR | gravity_and_lu (+53), contact_constraints, joint_constraints, multibody.rs/solver.rs/pipeline.rs host | motor model is host-side (kp×16/kd×4) — easy; contact-stiffness DR binding is the same fragile path as `fa8b551`'s ERP — validate. |
| `7214111` NEXUS_CONTACT_SWEEPS + diagnostics | multibody.rs host (+106) | host-only dispatch loop; re-apply around upstream's `RbdState` solve. Low risk. |
| `ef501d3` angfric (condim=6) + CFM | contact_constraints (+116), types.rs (CONTACT_CONSTRAINTS_PER_POINT 3→6) | LAST + behind a flag (gave no transfer gain). condim=6 changes constraint-slab sizing — validate. |

### C. Perf rewrite (THE GATE decides) — large, cross-cutting
| Commit | Footprint |
|---|---|
| `240874d` per-env parallelism + capture-safe dispatch | solver.rs, coloring.rs, joint.rs, radix_sort, lbvh, narrow_phase, joint_constraints (+210), scatter_motor (new), lib.rs (+60), multibody.rs |
| `85246ac` cooperative PGS solves + integrate | contact_constraints (+320), joint_constraints (+139), integrate, multibody.rs |
Re-port ONLY if the gate says upstream is slower. These fight upstream's own
parallelism rework head-on — most painful part of the whole effort.

### D. Host API for RL (zealot depends on these; re-apply with SAME signatures)
| Commit | API added (in `multibody.rs` / `pipeline.rs` host) |
|---|---|
| `2a247f6` | `stage_motor_position` + `flush_links_static` |
| `493cd07` | `set_motor_position` + diagnostic accessors |
| `394cee0` | `GpuPhysicsState::reset_env_from` (per-env reset) — touches pipeline.rs |
| `cc30966` | GPU-resident policy (vortx) — pendulum example, not core; port last/optional |
| `fa8b551` | `set_dof_armature` / `set_dof_frictionloss` setters (rides with B) |

### E. Build infra (keep, adapt to khal 0.2)
- `1469f56`, `2e8bf1e`, `5387b5e`, `240874d`'s build_cuda/* scripts → adapt env/deps.

### F. Obsolete / superseded (do NOT re-port)
- `7c5cd73` (merge of upstream v0.3.0) — superseded by basing on current upstream.
- `61258a6` (step-23 ILLEGAL_ADDRESS fix) — verify upstream's parallel solver
  doesn't have it; likely moot.
- pendulum demos (`b39afc8`..`74a9c65`, `ce0690a`, `3bf3f7f`, `458dad3`, …) —
  examples; rebuild from upstream or drop.

---

## 2. zealot `nexus3d` API delta

zealot's biped env call surface: `pipeline.step` ×15, `slurp_poses` ×12,
`GpuPhysicsState` ×9, `from_rapier` ×7, `reset_env_from` ×6, `flush_links_static`
×3, `stage_motor_position` ×2, `set_dof_armature` ×2, `slurp_state`,
`set_motor_position`, `set_dof_frictionloss`.

**Upstream renamed the core types:**
- `GpuPhysicsState` → `RbdState`
- `GpuPhysicsPipeline` → `RbdPipeline`

**Strategy: add aliases in nexus3d so zealot needs ZERO changes during the rebase:**
```rust
pub use RbdState as GpuPhysicsState;
pub use RbdPipeline as GpuPhysicsPipeline;
```
Everything else zealot uses (`slurp_poses`, `slurp_state`, `reset_env_from`,
`set_motor_position`, `stage_motor_position`, `flush_links_static`,
`set_dof_armature`, `set_dof_frictionloss`) are OUR additions — re-add them to
upstream's `RbdState`/`GpuMultibodySet` with identical signatures → zealot unchanged.
Only `from_rapier`'s internal `GpuMultibodySet` variant gained the `frictions`
param (PR #2); zealot calls the pipeline-level `from_rapier`, so confirm that
signature is unchanged after the rebase.

---

## 3. Execution order
1. Land PRs #1/#2 upstream (in progress); build+validate #3.
2. **GATE**: benchmark (needs GPU, after v58 training).
3. Re-apply Host API (D) + type aliases (§2) → get zealot compiling against upstream. Cubin build.
4. Re-apply physics features (B), each validated by a cubin build + bit-identical
   check vs a recorded rollout. angfric/CFM last, flagged.
5. Re-port cooperative rewrite (C) ONLY if the gate requires it.
6. Rebuild zealot's embedded cubin; **retrain** as the acceptance gate (semantic
   kernel ports can silently break dynamics — a clean curve + sim2sim check is the bar).

## 4. Validation gates (per step)
- Compiles (host) + cubin builds (sm_120, the `build_nexus_cubin_libnvvm.sh` toolchain).
- Bit-identical (or within tol) vs a recorded reference rollout where the change
  should be behavior-preserving (perf re-ports especially).
- Final: a training run that climbs like v58 + a sim2sim sanity check.
