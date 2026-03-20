use crate::Vector;

/// Boundary condition type constants.
///
/// Represented as `u32` for GPU compatibility instead of a Rust enum.
pub const BOUNDARY_CONDITION_STICK: u32 = 0;
pub const BOUNDARY_CONDITION_SLIP: u32 = 1;
pub const BOUNDARY_CONDITION_SEPARATE: u32 = 2;
pub const BOUNDARY_CONDITION_NON_REFLECTING: u32 = 3;

/// A boundary condition applied to grid nodes at domain boundaries or collider surfaces.
///
/// The `ty` field should be one of the `BOUNDARY_CONDITION_*` constants.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(any(target_arch = "spirv", target_arch = "nvptx64")), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct BoundaryCondition {
    /// The type of boundary condition (see `BOUNDARY_CONDITION_*` constants).
    pub ty: u32,
    /// Friction coefficient (used by the `Separate` boundary condition).
    pub friction: f32,
}

impl BoundaryCondition {
    /// Creates a new boundary condition.
    pub fn new(ty: u32, friction: f32) -> Self {
        Self { ty, friction }
    }

    /// Projects a velocity according to this boundary condition.
    ///
    /// # Arguments
    /// * `vel` - The velocity to project.
    /// * `n` - The boundary normal (pointing inward).
    ///
    /// # Returns
    /// The projected velocity after applying the boundary condition.
    pub fn project_velocity(&self, vel: Vector, n: Vector) -> Vector {
        if self.ty == BOUNDARY_CONDITION_STICK {
            return Vector::ZERO;
        }

        if self.ty == BOUNDARY_CONDITION_SLIP {
            let normal_vel = vel.dot(n);
            let tangent_vel = vel - n * normal_vel;
            return tangent_vel;
        }

        if self.ty == BOUNDARY_CONDITION_SEPARATE {
            let normal_vel = vel.dot(n);

            if normal_vel < 0.0 {
                let tangent_vel = vel - n * normal_vel;
                let tangent_vel_len = tangent_vel.length();
                let tangent_vel_dir = if tangent_vel_len > 1.0e-8 {
                    tangent_vel / tangent_vel_len
                } else {
                    Vector::ZERO
                };
                let projected_len = tangent_vel_len + self.friction * normal_vel;
                let projected_len = if projected_len > 0.0 {
                    projected_len
                } else {
                    0.0
                };
                return tangent_vel_dir * projected_len;
            } else {
                return vel;
            }
        }

        // BOUNDARY_CONDITION_NON_REFLECTING or unknown: pass through.
        vel
    }
}
