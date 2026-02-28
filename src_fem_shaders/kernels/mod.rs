//! Compute shader entry points for the FEM solver.
//!
//! Organized into explicit (symplectic Euler) and implicit (Newton-PCG) solver kernels.

pub mod explicit;
pub mod implicit;
