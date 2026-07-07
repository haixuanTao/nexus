# Running nexus on macOS (Apple Silicon / Metal)

nexus runs on macOS through wgpu's Metal backend (the khal `webgpu` backend).
As of naga/wgpu 29 this is **broken out of the box** by a shader-compiler bug,
with a one-line workaround. This doc covers the bug, the fix, and how to
verify a working setup.

## TL;DR

```toml
# In the [patch.crates-io] table of the WORKSPACE THAT BUILDS YOUR BINARY
# (patches only apply from the workspace root — putting this in a dependency's
# manifest does nothing):
naga = { path = "../naga-fixed" }   # https://github.com/haixuanTao/naga-fixed
```

```bash
git clone https://github.com/haixuanTao/naga-fixed ../naga-fixed
cargo update -p naga    # REQUIRED: un-sticks the lockfile from registry naga
                        # (otherwise the patch is silently ignored and shows
                        # up as [[patch.unused]] in Cargo.lock)
```

Remove the patch once the upstream fix ships in a wgpu release —
track https://github.com/gfx-rs/wgpu/pull/9815.

## The bug (naga 29, MSL backend)

naga's Metal writer renders a loop's `continuing` block at the top of the
next `while` iteration (MSL has no `continuing` construct and `continue`
jumps past anything at the bottom of the body). The `break_if` condition was
re-evaluated at that hoisted position — **after** the continuing statements
had already advanced the loop variables the condition reads.

rust-gpu compiles every `while` loop to a conditional backedge whose
condition is computed in the loop *body* (pre-increment), so on Metal every
such loop exits **one body-execution early**, silently dropping the final
iteration's stores. CUDA (PTX) and Vulkan (SPIR-V passthrough) never go
through naga, so the same SPIR-V is correct there — which makes this look
like an engine bug when it isn't.

### How it manifests in nexus

The multibody solver's cooperative kernels split their `J·v` reductions
across 32 lanes: `while i < ndofs { acc += …; i += 32 }` — exactly **one
iteration per lane**. One-iteration loops run **zero** times under the
miscompile, so:

- contact and PD-joint sweeps compute `J·v = 0` → **zero impulses**;
- gravity still integrates every TGS iteration → bodies free-fall through
  the floor; accumulated penetration then produces huge bias impulses →
  bounce → NaN;
- **more solver iterations make it worse** (each iteration adds `g·dt` with
  nothing to cancel it) — a classic misleading symptom;
- broad-phase pair counts "flicker" — fallout of the wrong dynamics, not a
  broad-phase bug.

Upstream reports: bug + fix at [gfx-rs/wgpu#9815], long-standing symptom
report at [gfx-rs/wgpu#4558], nexus-side advisory at [dimforge/nexus#5].

[gfx-rs/wgpu#9815]: https://github.com/gfx-rs/wgpu/pull/9815
[gfx-rs/wgpu#4558]: https://github.com/gfx-rs/wgpu/issues/4558
[dimforge/nexus#5]: https://github.com/dimforge/nexus/issues/5

## Backend selection on macOS

There is no native-CUDA path on a Mac; khal auto-selects WebGPU (wgpu →
Metal). `KHAL_BACKEND=webgpu` forces it explicitly; `KHAL_BACKEND=cuda`
errors. The khal `metal` feature (a native Metal backend) exists but is not
wired into auto-selection and is not the supported path.

## Verifying a working setup

Any deterministic scene works; the reference used to validate the fix is the
zealot biped (`contact_probe`, seed `0xC0FFEE`, `BIPED_SPAWN_DR=0`):

| check | broken (unpatched naga) | fixed |
|---|---|---|
| resting contact impulse (step 0, `BIPED_DECIMATION=1`) | 0.000 | **0.1697** (CUDA golden: 0.1698) |
| dof velocities after 1 step | free-fall signature: all ≈0 except base-z = −n·g·dt | matches CUDA to ~1e-4 |
| torso height over steps | 0.72 → launch → NaN by ~step 60 | steady 0.718 / 0.716 / 0.715 |

A quick generic smoke test: any multibody resting on a ground collider must
hold its height; if it sinks ~`g·dt²` per step or launches upward, you are
on an unpatched naga.

## Performance expectations (Apple Silicon)

With the patch, Metal is a first-class training backend. Measured on the
zealot biped (1024 envs, ~24.6k samples/iter):

| | M-series Mac (Metal) | RTX 5060 (native CUDA) |
|---|---|---|
| wall clock / iter | 4.0–4.2 s | 4.0 s |
| physics GPU wait / step | ~96–105 ms | ~52 ms |
| CPU encode+launch / step | ~6.5 ms | ~48 ms |

The 5060's raw compute is ~2× faster but pays ~7× the per-dispatch CPU
overhead; at this batch size they tie. End-to-end PPO training (reward
curves, KL, fall rates) matches the CUDA reference run-for-run.

## Shader-edit gotcha (all platforms)

The shader crates path-include `src_rbd_shaders/*`; cargo does not track
path-included files, so after editing shader source run:

```bash
touch src_rbd_shaders/lib.rs   # force the SPIR-V rebuild
```
