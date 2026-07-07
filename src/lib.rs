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
