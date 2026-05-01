//! Reduced-coordinates articulated multibody (3D).
//!
//! GPU port of rapier's `Multibody`, restricted to:
//! - Forward kinematics (link world transforms from generalized coordinates).
//! - Body jacobians (one `6 × ndofs` per link).
//! - Augmented mass matrix via CRBA: `M = Σ Jᵢᵀ Mᵢ Jᵢ`.
//! - Gravity generalized force: `τ = Σ Jᵢᵀ (mᵢ g, 0)`.
//! - In-place LU solve of `M ẍ = τ`.
//!
//! No constraints, contacts, or Coriolis terms (not in scope).
//!
//! ### Memory layout
//!
//! Links and multibodies are stored flat across all simulation batches. Each batch
//! has a capacity; unused slots are padded out and skipped via per-batch length
//! counts (mirrors the impulse-joint infrastructure).
//!
//! - `links_static: Tensor<MultibodyLinkStatic>`: constant per-link config.
//! - `links_workspace: Tensor<MultibodyLinkWorkspace>`: per-step scratch (pose, shifts).
//! - `multibody_info: Tensor<MultibodyInfo>`: offsets/sizes per multibody.
//! - `dof_values: Tensor<f32>`: generalized coordinates (flat, ndofs per multibody).
//! - `dof_velocities: Tensor<f32>`: generalized velocities.
//! - `gen_forces: Tensor<f32>`: generalized forces (receives gravity).
//! - `body_jacobians: Tensor<f32>`: per-link `6 × ndofs` column-major.
//! - `mass_matrices: Tensor<f32>`: per-multibody `ndofs × ndofs` column-major.
//!
//! ### Kernel topology
//!
//! Forward kinematics, jacobian assembly, mass-matrix assembly, and LU are
//! inherently sequential within a single multibody (parent before child, or
//! i-th elimination step before (i+1)-th). They run as `threads(1)` with one
//! workgroup per multibody. Links are independent across multibodies so the
//! batch × multibody grid parallelises fine.

#![cfg(feature = "dim3")]

mod contact_constraints;
mod forward_kinematics;
mod gravity;
mod integrate;
mod jacobian;
mod joint_constraints;
mod lu;
mod mass_matrix;
mod types;
mod utils;
mod velocity;

pub use contact_constraints::*;
pub use forward_kinematics::*;
pub use gravity::*;
pub use integrate::*;
pub use jacobian::*;
pub use joint_constraints::*;
pub use lu::*;
pub use mass_matrix::*;
pub use types::*;
pub use utils::*;
pub use velocity::*;
