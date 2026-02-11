use crate::Matrix;

/// Result of a constitutive model update, containing the Kirchoff stress tensor.
#[derive(Clone, Copy)]
pub struct ModelUpdateResult {
    pub kirchoff_stress: Matrix,
}

impl ModelUpdateResult {
    #[inline]
    pub fn new(kirchoff_stress: Matrix) -> Self {
        Self { kirchoff_stress }
    }
}

/// Data passed to the particle model update function.
#[derive(Clone, Copy)]
pub struct ParticleUpdateData {
    pub dt: f32,
    pub cell_width: f32,
    pub particle_id: u32,
}

impl ParticleUpdateData {
    #[inline]
    pub fn new(dt: f32, cell_width: f32, particle_id: u32) -> Self {
        Self {
            dt,
            cell_width,
            particle_id,
        }
    }
}

/// Model behavior flags (bitflags stored as u32).
pub const MODEL_FLAGS_NONE: u32 = 0;
pub const MODEL_FLAGS_FLUID: u32 = 1;
