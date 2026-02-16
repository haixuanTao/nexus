//! Rigid body particle transformation kernels.

use crate::cast_tensor;
use crate::cast_tensor_mut;
use crate::mpm_shaders::VectorWithPadding;
use crate::mpm_shaders::solver::rigid_particle_update::{
    GpuTransformSamplePoints, GpuTransformShapePoints,
};
use crate::solver::GpuRigidParticles;
use khal::Shader;
use khal::backend::{GpuBackendError, GpuPass};
use nexus::dynamics::GpuBodySet;
use nexus::math::Vector;

/// GPU kernels for updating rigid body particle positions.
///
/// Transforms surface-sampled particles from local to world coordinates
/// as rigid bodies move.
#[derive(Shader)]
pub struct WgRigidParticleUpdate {
    /// Kernel for transforming sample points.
    transform_sample_points: GpuTransformSamplePoints,
    /// Kernel for transforming shape vertex points.
    transform_shape_points: GpuTransformShapePoints,
}

impl WgRigidParticleUpdate {
    /// Transforms rigid body particles from local to world space.
    ///
    /// Updates surface particle positions based on current rigid body poses.
    /// Also transforms collision shape vertices for accurate collision detection.
    ///
    /// # Arguments
    ///
    /// * `pass` - Compute pass
    /// * `bodies` - Rigid bodies with current poses
    /// * `rigid_particles` - Particles to transform
    pub fn launch(
        &self,
        pass: &mut GpuPass,
        bodies: &mut GpuBodySet,
        rigid_particles: &mut GpuRigidParticles,
    ) -> Result<(), GpuBackendError> {
        if rigid_particles.is_empty() {
            return Ok(());
        }

        let sample_len = rigid_particles.local_sample_points.len() as u32;
        self.transform_sample_points.call(
            pass,
            [sample_len, 1, 1],
            &rigid_particles.sample_ids,
            &bodies.poses,
            cast_tensor::<_, VectorWithPadding>(&rigid_particles.local_sample_points),
            cast_tensor_mut::<_, VectorWithPadding>(&mut rigid_particles.sample_points),
        )?;

        // TODO: this is now incorrect since the vertex buffer also contains the BVH.
        // let vtx_len = bodies.shapes_vertex_buffers.len() as u32;
        // println!("Shape vbuf: {}", vtx_len);
        // self.transform_shape_points.call(
        //     pass,
        //     [vtx_len, 1, 1],
        //     &bodies.shapes_vertex_collider_id,
        //     &bodies.poses,
        //     &bodies.shapes_local_vertex_buffers,
        //     &mut bodies.shapes_vertex_buffers,
        // )

        Ok(())
    }
}
