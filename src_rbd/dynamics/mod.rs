//! Rigid-body dynamics: forces, velocities, constraints, and solvers.

pub use crate::shaders::dynamics::SimParams as GpuSimParams;
pub use body::{BodyCoupling, BodyCouplingEntry, BodyDesc, GpuBodySet};
pub use coloring::{ColoringArgs, GpuColoring};
pub use joint::{GpuImpulseJointSet, GpuJointSolver, JointSolverArgs};
pub use mprops_update::GpuMpropsUpdate;
#[cfg(feature = "dim3")]
pub use multibody::{GpuMultibodySet, GpuMultibodySolver, MultibodySolverArgs};
pub use solver::{GpuSolver, SolverArgs};
pub use warmstart::{GpuWarmstart, WarmstartArgs};

pub mod body;
mod coloring;
mod joint;
mod mprops_update;
#[cfg(feature = "dim3")]
pub(crate) mod multibody;
mod solver;
pub mod warmstart;
