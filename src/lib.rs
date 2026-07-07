//! # nexus — GPU-resident physics
//!
//! ## Running on macOS (Metal) — known issue with naga ≤ 29
//!
//! On macOS the engine runs through wgpu's Metal backend, and **naga 29's MSL
//! writer miscompiles rust-gpu loops**: a loop's `break if` condition is
//! re-evaluated after the `continuing` block has advanced the loop variables,
//! so every `while` loop whose condition is computed in the loop body exits
//! one iteration early ([gfx-rs/wgpu#4558], fixed by [gfx-rs/wgpu#9815]).
//!
//! In this engine the multibody solver's per-lane `J·v` reductions have
//! exactly one iteration per lane, so on an unpatched naga they run **zero**
//! times: contact and joint sweeps produce **zero impulses**, bodies
//! free-fall through the ground, and raising solver iterations makes the
//! blow-up *worse* (gravity integrates per iteration with nothing to cancel
//! it). CUDA and Vulkan are unaffected — the same SPIR-V never goes through
//! naga there.
//!
//! **Workaround until the wgpu fix ships** — patch naga in the workspace that
//! builds your final binary (patches only apply from the workspace root):
//!
//! ```toml
//! [patch.crates-io]
//! naga = { path = "../naga-fixed" } # github.com/haixuanTao/naga-fixed
//! ```
//!
//! then run `cargo update -p naga` — without it the lockfile keeps the
//! registry naga and the patch is silently ignored (`[[patch.unused]]`).
//!
//! Quick sanity check: a multibody resting on a ground collider must hold its
//! height; if it sinks ~`g·dt²` per step or launches upward, you are on an
//! unpatched naga. Full guide (verification values, Apple-Silicon
//! performance): `docs/RUNNING-ON-MACOS.md` in the repository.
//!
//! [gfx-rs/wgpu#4558]: https://github.com/gfx-rs/wgpu/issues/4558
//! [gfx-rs/wgpu#9815]: https://github.com/gfx-rs/wgpu/pull/9815

#[cfg(all(feature = "dim2", feature = "rbd"))]
pub use nexus_rbd2d as rbd;
#[cfg(all(feature = "dim3", feature = "rbd"))]
pub use nexus_rbd3d as rbd;

#[cfg(all(feature = "dim2", feature = "mpm"))]
pub use nexus_mpm2d as mpm;
#[cfg(all(feature = "dim3", feature = "mpm"))]
pub use nexus_mpm3d as mpm;

#[cfg(all(feature = "dim2", feature = "fem"))]
pub use nexus_fem2d as fem;
#[cfg(all(feature = "dim3", feature = "fem"))]
pub use nexus_fem3d as fem;

// Upstream (dimforge/nexus) additionally declares `pub mod pipeline;` /
// `pub mod state;` here — the high-level NexusState API. Those files are in
// the tree (src/pipeline.rs, src/state.rs) but not compiled in this fork:
// they target upstream's rewritten rbd API (RbdPipeline, RbdSimParams, …),
// which this fork's core predates. Re-enable with the dependency migration.
