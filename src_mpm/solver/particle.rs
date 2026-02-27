use crate::mpm_shaders::solver::particle::{
    Kinematics, ParticleProperties, Position, RigidParticleIndices,
};
use crate::mpm_shaders::{PaddedMatrix, PaddingExt};
use crate::solver::particle_model::GpuParticleModelData;
use glamx::{Mat2, Mat3, Vec2, Vec3, Vec4};
use khal::BufferUsages;
use khal::backend::{Backend, GpuBackend, GpuBackendError};
use nexus_rbd::dynamics::GpuBodySet;
use nexus_rbd::math::{Matrix, Vector};
use std::ops::RangeBounds;
use vortx::tensor::Tensor;

#[cfg(feature = "from_rapier")]
use {
    crate::sampling::{self, SamplingBuffers, SamplingParams},
    nexus_rbd::dynamics::body::RapierBodyCouplingEntry,
    nexus_rbd::shapes::ShapeBuffers,
};

/// Particle position type used on the GPU.
///
/// In 2D: `Position` contains a Vec2.
/// In 3D: `Position` contains a Vec3.
pub type ParticlePosition = Position;

/// A single MPM particle with position, dynamics, and material model.
#[derive(Copy, Clone, Debug)]
pub struct Particle<Model> {
    /// Spatial position.
    pub position: Vector,
    /// Physical state (velocity, deformation, mass, etc.).
    pub dynamics: ParticleDynamics,
    /// Material model defining constitutive behavior.
    pub model: Model,
}

impl<Model> Particle<Model> {
    /// Creates a new particle with the given properties.
    pub fn new(position: Vector, radius: f32, density: f32, model: Model) -> Self {
        Particle {
            position,
            dynamics: ParticleDynamics::new(radius, density),
            model,
        }
    }
}

/// CPU-side particle dynamics for initialization.
///
/// Splits into GPU `Kinematics`, `Cdf`, deformation gradient, and `ParticleProperties` buffers on upload.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ParticleDynamics {
    /// Current velocity.
    pub velocity: Vector,
    /// Deformation gradient.
    pub def_grad: Matrix,
    /// APIC affine velocity matrix.
    pub affine: Matrix,
    /// Additional force * dt.
    pub force_dt: Vector,
    /// Determinant of velocity gradient.
    pub vel_grad_det: f32,
    /// Collision detection field.
    pub cdf: crate::mpm_shaders::solver::particle::Cdf,
    /// Initial particle volume.
    pub init_volume: f32,
    /// Initial particle radius.
    pub init_radius: f32,
    /// Particle mass.
    pub mass: f32,
    /// Damping coefficient.
    pub damping: f32,
    /// Particle phase.
    pub phase: f32,
    /// Whether this particle is active.
    pub enabled: u32,
    /// Whether this particle is fixed.
    pub fixed: u32,
}

impl ParticleDynamics {
    /// Creates new particle dynamics from radius and material density.
    pub fn new(radius: f32, density: f32) -> Self {
        let exponent = if cfg!(feature = "dim2") { 2 } else { 3 };
        let init_volume = (radius * 2.0).powi(exponent);
        Self {
            velocity: Vector::ZERO,
            def_grad: Matrix::IDENTITY,
            affine: Matrix::ZERO,
            force_dt: Vector::ZERO,
            vel_grad_det: 0.0,
            init_volume,
            init_radius: radius,
            mass: init_volume * density,
            damping: 0.0,
            cdf: crate::mpm_shaders::solver::particle::Cdf::zero(),
            phase: 1.0,
            enabled: 1,
            fixed: 0,
        }
    }

    /// Sets whether this particle is fixed.
    pub fn set_fixed(&mut self, fixed: bool) {
        self.fixed = fixed as u32;
    }

    /// Sets the damping coefficient.
    pub fn set_damping(&mut self, damping: f32) {
        self.damping = damping;
    }

    /// Updates the particle mass based on a new density.
    pub fn set_density(&mut self, density: f32) {
        self.mass = self.init_volume * density;
    }

    /// Converts to the GPU `Kinematics` struct.
    fn to_gpu_kinematics(&self) -> Kinematics {
        Kinematics {
            affine: PaddedMatrix::add_padding(self.affine),
            velocity: self.velocity,
            vel_grad_det: self.vel_grad_det,
            force_dt: self.force_dt,
            mass: self.mass,
            enabled: self.enabled,
            _padding: Default::default(),
            cdf: self.cdf,
            #[cfg(feature = "dim2")]
            _tail_padding: Default::default(),
        }
    }

    /// Converts the deformation gradient to a GPU `PaddedMatrix`.
    fn to_gpu_def_grad(&self) -> PaddedMatrix {
        PaddedMatrix::add_padding(self.def_grad)
    }

    /// Converts to the GPU `ParticleProperties` struct.
    fn to_gpu_properties(&self) -> ParticleProperties {
        ParticleProperties {
            init_volume: self.init_volume,
            init_radius: self.init_radius,
            damping: self.damping,
            phase: self.phase,
            fixed: self.fixed,
            padding: Default::default(),
        }
    }
}

struct SoAParticles<GpuModel: GpuParticleModelData> {
    positions: Vec<Position>,
    kinematics: Vec<Kinematics>,
    def_grad: Vec<PaddedMatrix>,
    properties: Vec<ParticleProperties>,
    models: Vec<GpuModel>,
}

impl<GpuModel: GpuParticleModelData> SoAParticles<GpuModel> {
    pub fn new(particles: &[Particle<GpuModel::Model>]) -> Self {
        let positions: Vec<_> = particles
            .iter()
            .map(|p| Position::new(p.position))
            .collect();
        let kinematics: Vec<_> = particles
            .iter()
            .map(|p| p.dynamics.to_gpu_kinematics())
            .collect();
        let def_grad: Vec<_> = particles
            .iter()
            .map(|p| p.dynamics.to_gpu_def_grad())
            .collect();
        let properties: Vec<_> = particles
            .iter()
            .map(|p| p.dynamics.to_gpu_properties())
            .collect();
        let models: Vec<_> = particles
            .iter()
            .map(|p| GpuModel::from_model(p.model))
            .collect();

        Self {
            positions,
            kinematics,
            def_grad,
            properties,
            models,
        }
    }
}

/// GPU buffers for particles sampled from rigid body surfaces.
pub struct GpuRigidParticles {
    /// Sample points in local (body-relative) coordinates.
    pub local_sample_points: Tensor<Position>,
    /// Sample points transformed to world coordinates.
    pub sample_points: Tensor<Position>,
    /// Bitmask indicating which rigid particles need grid cell blocking.
    pub rigid_particle_needs_block: Tensor<u32>,
    /// Linked list for spatially sorting rigid particles into grid cells.
    pub node_linked_lists: Tensor<u32>,
    /// Metadata associating each sample with its source collider and body.
    pub sample_ids: Tensor<RigidParticleIndices>,
}

impl GpuRigidParticles {
    /// Creates an empty set of rigid particles.
    pub fn new(backend: &GpuBackend) -> Result<Self, GpuBackendError> {
        let empty_positions: &[Position] = &[];
        let empty_ids: &[RigidParticleIndices] = &[];
        Ok(Self {
            local_sample_points: Tensor::vector(backend, empty_positions, BufferUsages::STORAGE)?,
            sample_points: Tensor::vector(backend, empty_positions, BufferUsages::STORAGE)?,
            node_linked_lists: Tensor::vector_uninit(backend, 0, BufferUsages::STORAGE)?,
            sample_ids: Tensor::vector(backend, empty_ids, BufferUsages::STORAGE)?,
            rigid_particle_needs_block: Tensor::vector_uninit(backend, 0, BufferUsages::STORAGE)?,
        })
    }

    #[cfg(feature = "from_rapier")]
    fn from_buffers(
        backend: &GpuBackend,
        sampling_buffers: &SamplingBuffers,
    ) -> Result<Self, GpuBackendError> {
        Ok(Self {
            local_sample_points: Tensor::vector(
                backend,
                &sampling_buffers.samples,
                BufferUsages::STORAGE,
            )?,
            sample_points: Tensor::vector(
                backend,
                &sampling_buffers.samples,
                BufferUsages::STORAGE,
            )?,
            node_linked_lists: Tensor::vector_uninit(
                backend,
                sampling_buffers.samples.len() as u32,
                BufferUsages::STORAGE,
            )?,
            sample_ids: Tensor::vector(
                backend,
                &sampling_buffers.samples_ids,
                BufferUsages::STORAGE,
            )?,
            rigid_particle_needs_block: Tensor::vector_uninit(
                backend,
                sampling_buffers.samples.len().div_ceil(32) as u32,
                BufferUsages::STORAGE,
            )?,
        })
    }

    /// Samples particles from collider surfaces for MPM coupling.
    #[cfg(feature = "from_rapier")]
    pub fn from_rapier(
        backend: &GpuBackend,
        colliders: &rapier::geometry::ColliderSet,
        gpu_bodies: &GpuBodySet,
        coupling: &[RapierBodyCouplingEntry],
        sampling_step: f32,
    ) -> Result<Self, GpuBackendError> {
        let mut sampling_buffers = SamplingBuffers::default();

        for (collider_id, (coupling, gpu_data)) in coupling
            .iter()
            .zip(gpu_bodies.shapes_data().iter())
            .enumerate()
        {
            let collider = &colliders[coupling.collider];

            #[cfg(feature = "dim2")]
            if let Some(polyline) = collider.shape().as_polyline() {
                // Use polyline_vertex_start() to get the correct base index,
                // which accounts for BVH AABB data preceding the actual vertices.
                let sampling_params = SamplingParams {
                    collider_id: collider_id as u32,
                    base_vid: gpu_data.polyline_vertex_start(),
                    sampling_step,
                };
                sampling::sample_polyline(polyline, &sampling_params, &mut sampling_buffers)
            }

            #[cfg(feature = "dim3")]
            if let Some(trimesh) = collider.shape().as_trimesh() {
                // Use trimesh_vertex_start() to get the correct base index,
                // which accounts for BVH AABB data preceding the actual vertices.
                let sampling_params = SamplingParams {
                    collider_id: collider_id as u32,
                    base_vid: gpu_data.trimesh_vertex_start(),
                    sampling_step,
                };
                sampling::sample_trimesh(trimesh, &sampling_params, &mut sampling_buffers)
            } else if let Some(heightfield) = collider.shape().as_heightfield() {
                let (vtx, idx) = heightfield.to_trimesh();
                let trimesh = rapier::geometry::TriMesh::new(vtx, idx).unwrap();
                // Use trimesh_vertex_start() to get the correct base index,
                // which accounts for BVH AABB data preceding the actual vertices.
                let sampling_params = SamplingParams {
                    collider_id: collider_id as u32,
                    base_vid: gpu_data.trimesh_vertex_start(),
                    sampling_step,
                };
                sampling::sample_trimesh(&trimesh, &sampling_params, &mut sampling_buffers)
            }
        }

        Self::from_buffers(backend, &sampling_buffers)
    }

    /// Returns the number of rigid body particles.
    pub fn len(&self) -> u64 {
        self.sample_points.len()
    }

    /// Returns true if there are no rigid body particles.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// GPU buffers storing all MPM particle data in Structure-of-Arrays layout.
pub struct GpuParticles<GpuModel: GpuParticleModelData> {
    len: usize,
    pub gpu_len: Tensor<u32>,
    pub positions: Tensor<Position>,
    pub kinematics: Tensor<Kinematics>,
    pub def_grad: Tensor<PaddedMatrix>,
    pub properties: Tensor<ParticleProperties>,
    pub models: Tensor<GpuModel>,
    pub sorted_ids: Tensor<u32>,
    pub node_linked_lists: Tensor<u32>,
}

impl<GpuModel: GpuParticleModelData> GpuParticles<GpuModel> {
    /// Returns true if there are no particles.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of particles.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns reference to GPU buffer containing particle count.
    pub fn gpu_len(&self) -> &Tensor<u32> {
        &self.gpu_len
    }

    /// Uploads CPU-side particles to GPU buffers.
    pub fn from_particles(
        backend: &GpuBackend,
        particles: &[Particle<GpuModel::Model>],
    ) -> Result<Self, GpuBackendError> {
        let data = SoAParticles::new(particles);
        let resizeable = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
        Ok(Self {
            len: particles.len(),
            gpu_len: Tensor::scalar(
                backend,
                particles.len() as u32,
                BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            )?,
            positions: Tensor::vector(backend, &data.positions, resizeable)?,
            kinematics: Tensor::vector(backend, &data.kinematics, resizeable)?,
            def_grad: Tensor::vector(backend, &data.def_grad, resizeable)?,
            properties: Tensor::vector(backend, &data.properties, resizeable)?,
            models: Tensor::vector(backend, &data.models, resizeable)?,
            sorted_ids: Tensor::vector_uninit(backend, particles.len() as u32, resizeable)?,
            node_linked_lists: Tensor::vector_uninit(backend, particles.len() as u32, resizeable)?,
        })
    }

    /// Returns reference to material model buffer.
    pub fn models(&self) -> &Tensor<GpuModel> {
        &self.models
    }

    /// Returns mutable reference to material model buffer.
    pub fn models_mut(&mut self) -> &mut Tensor<GpuModel> {
        &mut self.models
    }

    /// Returns reference to position buffer.
    pub fn positions(&self) -> &Tensor<Position> {
        &self.positions
    }

    /// Returns mutable reference to position buffer.
    pub fn positions_mut(&mut self) -> &mut Tensor<Position> {
        &mut self.positions
    }

    /// Returns reference to kinematics buffer.
    pub fn kinematics(&self) -> &Tensor<Kinematics> {
        &self.kinematics
    }

    /// Returns mutable reference to kinematics buffer.
    pub fn kinematics_mut(&mut self) -> &mut Tensor<Kinematics> {
        &mut self.kinematics
    }

    /// Returns reference to deformation gradient buffer.
    pub fn def_grad(&self) -> &Tensor<PaddedMatrix> {
        &self.def_grad
    }

    /// Returns mutable reference to deformation gradient buffer.
    pub fn def_grad_mut(&mut self) -> &mut Tensor<PaddedMatrix> {
        &mut self.def_grad
    }

    /// Returns reference to particle properties buffer (read-only on GPU).
    pub fn properties(&self) -> &Tensor<ParticleProperties> {
        &self.properties
    }

    /// Returns mutable reference to particle properties buffer.
    pub fn properties_mut(&mut self) -> &mut Tensor<ParticleProperties> {
        &mut self.properties
    }

    /// Returns reference to sorted particle ID buffer.
    pub fn sorted_ids(&self) -> &Tensor<u32> {
        &self.sorted_ids
    }

    /// Returns mutable reference to sorted particle ID buffer.
    pub fn sorted_ids_mut(&mut self) -> &mut Tensor<u32> {
        &mut self.sorted_ids
    }

    /// Returns reference to per-particle linked list buffer.
    pub fn node_linked_lists(&self) -> &Tensor<u32> {
        &self.node_linked_lists
    }

    /// Returns mutable reference to per-particle linked list buffer.
    pub fn node_linked_lists_mut(&mut self) -> &mut Tensor<u32> {
        &mut self.node_linked_lists
    }

    /// Removes a range of particles from the GPU buffers, shifting elements to fill the gap.
    ///
    /// Returns the number of removed particles on success.
    pub fn shift_remove(
        &mut self,
        backend: &GpuBackend,
        range: impl RangeBounds<usize> + Clone,
    ) -> Result<usize, GpuBackendError> {
        let Self {
            len,
            gpu_len,
            positions,
            kinematics,
            def_grad,
            properties,
            models,
            sorted_ids: _,
            node_linked_lists: _,
        } = self;

        let removed = positions.shift_remove(backend, range.clone())?;
        kinematics.shift_remove(backend, range.clone())?;
        def_grad.shift_remove(backend, range.clone())?;
        properties.shift_remove(backend, range.clone())?;
        models.shift_remove(backend, range)?;

        *len -= removed;
        backend.write_buffer(gpu_len.buffer_mut(), 0, &[*len as u32])?;
        Ok(removed)
    }

    /// Appends particles at the end of the GPU buffers.
    pub fn append(
        &mut self,
        backend: &GpuBackend,
        particles: &[Particle<GpuModel::Model>],
    ) -> Result<(), GpuBackendError> {
        let Self {
            len,
            gpu_len,
            positions,
            kinematics,
            def_grad,
            properties,
            models,
            sorted_ids,
            node_linked_lists,
        } = self;

        let data = SoAParticles::new(particles);
        let zeros = vec![0u32; particles.len()];

        positions.append(backend, &data.positions)?;
        kinematics.append(backend, &data.kinematics)?;
        def_grad.append(backend, &data.def_grad)?;
        properties.append(backend, &data.properties)?;
        models.append(backend, &data.models)?;
        sorted_ids.append(backend, &zeros)?;
        node_linked_lists.append(backend, &zeros)?;

        *len += particles.len();
        backend.write_buffer(gpu_len.buffer_mut(), 0, &[*len as u32])?;
        Ok(())
    }
}
