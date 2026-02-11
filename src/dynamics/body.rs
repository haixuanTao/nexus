//! Rigid-body definitions, mass properties, velocities, and GPU storage.
//!
//! This module provides the core data structures for representing rigid bodies on the GPU,
//! including their poses, velocities, forces, and mass properties. It also provides
//! [`GpuBodySet`] for managing collections of rigid bodies in GPU memory.

use crate::math::{Point, Pose, Vector};
use crate::shapes::ShapeBuffers;

use crate::shaders::dynamics::{LocalMassProperties, Velocity, WorldMassProperties};
use crate::shaders::shapes::Shape;
use khal::BufferUsages;
use khal::backend::GpuBackend;
use vortx::tensor::Tensor;

#[cfg(feature = "from_rapier")]
use {
    crate::rapier::dynamics::{RigidBodyHandle, RigidBodySet},
    crate::rapier::geometry::{ColliderHandle, ColliderSet},
    crate::rapier::prelude::MassProperties,
    crate::shapes::shape_from_parry,
    num_traits::Zero,
};
use crate::shaders::VectorWithPadding;
/// Re-export types from the shader crate for convenience.
pub use crate::shaders::dynamics::{
    Force, Impulse, LocalMassProperties as GpuLocalMassProperties, Velocity as GpuVelocity,
    WorldMassProperties as GpuWorldMassProperties,
};

/// A set of rigid-bodies stored on the gpu.
pub struct GpuBodySet {
    len: u32,
    shapes_data: Vec<Shape>,
    pub mprops: Tensor<WorldMassProperties>,
    pub local_mprops: Tensor<LocalMassProperties>,
    pub vels: Tensor<Velocity>,
    pub poses: Tensor<Pose>,
    pub shapes: Tensor<Shape>,
    pub shapes_local_vertex_buffers: Tensor<VectorWithPadding>,
    pub shapes_vertex_buffers: Tensor<VectorWithPadding>,
    pub shapes_vertex_collider_id: Tensor<u32>,
}

#[derive(Copy, Clone)]
/// Helper struct for defining a rigid-body to be added to a [`GpuBodySet`].
pub struct BodyDesc {
    /// The rigid-body's mass-properties in local-space.
    pub local_mprops: LocalMassProperties,
    /// The rigid-body's mass-properties in world-space.
    pub mprops: WorldMassProperties,
    /// The rigid-body's linear and angular velocities.
    pub vel: Velocity,
    /// The rigid-body's world-space pose.
    pub pose: Pose,
    /// The rigid-body's shape.
    pub shape: Shape,
}

impl Default for BodyDesc {
    fn default() -> Self {
        Self {
            local_mprops: Default::default(),
            mprops: Default::default(),
            vel: Default::default(),
            pose: Default::default(),
            shape: Shape::cuboid(Vector::splat(0.5)),
        }
    }
}

/// Coupling mode between a GPU body and the physics simulation.
///
/// This controls whether a body is affected by physics forces or acts as a kinematic body.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum BodyCoupling {
    /// One-way coupling: the body affects other bodies but is not affected by them.
    ///
    /// This is useful for kinematic bodies that move independently of physics forces.
    OneWay,
    /// Two-way coupling: the body both affects and is affected by other bodies.
    ///
    /// This is the standard mode for dynamic rigid bodies.
    #[default]
    TwoWays,
}

/// Associates a body/collider pair with a coupling mode.
///
/// Used when creating a [`GpuBodySet`] to specify which bodies should have
/// two-way vs one-way coupling.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BodyCouplingEntry {
    /// The rigid body index.
    pub body: usize,
    /// The collider index.
    pub collider: usize,
    /// The coupling mode for this body.
    pub mode: BodyCoupling,
}

#[cfg(feature = "from_rapier")]
/// Associates a Rapier body/collider pair with a coupling mode.
///
/// Used when creating a [`GpuBodySet`] from Rapier data structures to specify
/// which bodies should have two-way vs one-way coupling.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RapierBodyCouplingEntry {
    /// The Rapier rigid body handle.
    pub body: RigidBodyHandle,
    /// The Rapier collider handle.
    pub collider: ColliderHandle,
    /// The coupling mode for this body.
    pub mode: BodyCoupling,
}

impl GpuBodySet {
    /// Returns `true` if this set contains no rigid bodies.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of rigid bodies in this set.
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Creates a new GPU body set from Rapier rigid bodies and colliders.
    ///
    /// # Parameters
    ///
    /// - `backend`: The GPU backend for allocating buffers.
    /// - `bodies`: Rapier rigid body set.
    /// - `colliders`: Rapier collider set.
    /// - `coupling`: Body coupling entries specifying which bodies to include.
    #[cfg(feature = "from_rapier")]
    pub fn from_rapier(
        backend: &GpuBackend,
        bodies: &RigidBodySet,
        colliders: &ColliderSet,
        coupling: &[RapierBodyCouplingEntry],
    ) -> Self {
        let mut shape_buffers = ShapeBuffers::default();
        let mut gpu_bodies = vec![];
        let mut pt_collider_ids = vec![];

        for (co_id, coupling) in coupling.iter().enumerate() {
            let co = &colliders[coupling.collider];
            let rb = &bodies[coupling.body];

            let prev_len = shape_buffers.vertices.len();
            let shape =
                shape_from_parry(co.shape(), &mut shape_buffers).expect("Unsupported shape type");
            for _ in prev_len..shape_buffers.vertices.len() {
                pt_collider_ids.push(co_id as u32);
            }

            let zero_mprops = MassProperties::zero();
            let two_ways_coupling = rb.is_dynamic() && coupling.mode == BodyCoupling::TwoWays;
            let desc = BodyDesc {
                vel: Velocity::new(
                    rb.linvel(),
                    #[cfg(feature = "dim2")]
                    rb.angvel(),
                    #[cfg(feature = "dim3")]
                    rb.angvel(),
                ),
                pose: *rb.position(),
                shape,
                local_mprops: if two_ways_coupling {
                    convert_local_mprops(&rb.mass_properties().local_mprops)
                } else {
                    convert_local_mprops(&zero_mprops)
                },
                mprops: Default::default(),
            };
            gpu_bodies.push(desc);
        }

        Self::new(backend, &gpu_bodies, &pt_collider_ids, &shape_buffers)
    }

    /// Create a set of `bodies` on the gpu.
    pub fn new(
        backend: &GpuBackend,
        bodies: &[BodyDesc],
        pt_collider_ids: &[u32],
        shape_buffers: &ShapeBuffers,
    ) -> Self {
        #[allow(clippy::type_complexity)]
        let (local_mprops, (mprops, (vels, (poses, shapes_data)))): (
            Vec<_>,
            (Vec<_>, (Vec<_>, (Vec<_>, Vec<_>))),
        ) = bodies
            .iter()
            .copied()
            .map(|b| (b.local_mprops, (b.mprops, (b.vel, (b.pose, b.shape)))))
            .collect();

        Self {
            len: bodies.len() as u32,
            mprops: Tensor::vector(backend, &mprops, BufferUsages::STORAGE).unwrap(),
            local_mprops: Tensor::vector(backend, &local_mprops, BufferUsages::STORAGE).unwrap(),
            vels: Tensor::vector(
                backend,
                &vels,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            )
            .unwrap(),
            poses: Tensor::vector(
                backend,
                &poses,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            )
            .unwrap(),
            shapes: Tensor::vector(backend, &shapes_data, BufferUsages::STORAGE).unwrap(),
            shapes_local_vertex_buffers: Tensor::vector(
                backend,
                &shape_buffers.vertices,
                BufferUsages::STORAGE,
            )
            .unwrap(),
            shapes_vertex_buffers: Tensor::vector(
                backend,
                &shape_buffers.vertices,
                BufferUsages::STORAGE,
            )
            .unwrap(),
            shapes_vertex_collider_id: Tensor::vector(
                backend,
                pt_collider_ids,
                BufferUsages::STORAGE,
            )
            .unwrap(),
            shapes_data,
        }
    }

    /// GPU storage buffer containing the poses of every rigid-body.
    pub fn poses(&self) -> &Tensor<Pose> {
        &self.poses
    }

    /// GPU storage buffer containing the velocities of every rigid-body.
    pub fn vels(&self) -> &Tensor<Velocity> {
        &self.vels
    }

    /// GPU storage buffer containing the world-space mass-properties of every rigid-body.
    pub fn mprops(&self) -> &Tensor<WorldMassProperties> {
        &self.mprops
    }

    /// GPU storage buffer containing the local-space mass-properties of every rigid-body.
    pub fn local_mprops(&self) -> &Tensor<LocalMassProperties> {
        &self.local_mprops
    }

    /// GPU storage buffer containing the shape of every rigid-body.
    pub fn shapes(&self) -> &Tensor<Shape> {
        &self.shapes
    }

    /// Mutable reference to the GPU storage buffer containing the poses of every rigid-body.
    pub fn poses_mut(&mut self) -> &mut Tensor<Pose> {
        &mut self.poses
    }

    /// Mutable reference to the GPU storage buffer containing the velocities of every rigid-body.
    pub fn vels_mut(&mut self) -> &mut Tensor<Velocity> {
        &mut self.vels
    }

    /// Mutable reference to the GPU storage buffer containing the world-space mass-properties of every rigid-body.
    pub fn mprops_mut(&mut self) -> &mut Tensor<WorldMassProperties> {
        &mut self.mprops
    }

    /// Returns the GPU buffer containing shape vertices in world-space.
    ///
    /// This buffer is updated each frame as bodies move.
    pub fn shapes_vertex_buffers(&self) -> &Tensor<VectorWithPadding> {
        &self.shapes_vertex_buffers
    }

    /// Mutable reference to the GPU buffer containing shape vertices in world-space.
    pub fn shapes_vertex_buffers_mut(&mut self) -> &mut Tensor<VectorWithPadding> {
        &mut self.shapes_vertex_buffers
    }

    /// Returns the GPU buffer containing shape vertices in local-space.
    ///
    /// These are the original vertex positions before transformation.
    pub fn shapes_local_vertex_buffers(&self) -> &Tensor<VectorWithPadding> {
        &self.shapes_local_vertex_buffers
    }

    /// Returns the GPU buffer mapping each vertex to its collider ID.
    ///
    /// This is used by wgsparkl for particle-body interactions.
    pub fn shapes_vertex_collider_id(&self) -> &Tensor<u32> {
        &self.shapes_vertex_collider_id
    }

    /// Returns a CPU-side slice of the shape data.
    ///
    /// Useful for accessing shape information without GPU readback.
    pub fn shapes_data(&self) -> &[Shape] {
        &self.shapes_data
    }
}

#[cfg(feature = "from_rapier")]
fn convert_local_mprops(mprops: &MassProperties) -> LocalMassProperties {
    #[cfg(feature = "dim2")]
    {
        LocalMassProperties {
            inv_mass: glamx::Vec2::splat(mprops.inv_mass),
            com: mprops.local_com,
            padding2: 0,
            inv_inertia: mprops.inv_principal_inertia,
        }
    }
    #[cfg(feature = "dim3")]
    {
        LocalMassProperties {
            inertia_ref_frame: mprops.principal_inertia_local_frame,
            inv_principal_inertia: mprops.inv_principal_inertia,
            padding0: 0,
            inv_mass: glamx::Vec3::splat(mprops.inv_mass),
            padding1: 0,
            com: mprops.local_com,
            padding2: 0,
        }
    }
}
