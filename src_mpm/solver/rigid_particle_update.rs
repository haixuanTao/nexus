//! Rigid body particle transformation kernels.

use crate::cast_tensor;
use crate::cast_tensor_mut;
use crate::mpm_shaders::PaddedVector;
use crate::mpm_shaders::solver::rigid_particle_update::{
    GpuTransformSamplePoints, GpuTransformShapePoints,
};
use crate::solver::GpuRigidParticles;
use khal::Shader;
use khal::backend::{GpuBackendError, GpuPass};
use nexus_rbd::dynamics::GpuBodySet;
/// GPU kernels for updating rigid body particle positions.
///
/// Transforms surface-sampled particles from local to world coordinates
/// as rigid bodies move.
#[derive(Shader)]
pub struct WgRigidParticleUpdate {
    /// Kernel for transforming sample points.
    transform_sample_points: GpuTransformSamplePoints,
    /// Kernel for transforming shape vertex points.
    #[allow(dead_code)]
    transform_shape_points: GpuTransformShapePoints,
}

impl WgRigidParticleUpdate {
    /// Transforms rigid body particles from local to world space.
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
            cast_tensor::<_, PaddedVector>(&rigid_particles.local_sample_points),
            cast_tensor_mut::<_, PaddedVector>(&mut rigid_particles.sample_points),
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
