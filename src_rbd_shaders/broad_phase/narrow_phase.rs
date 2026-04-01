//! Narrow phase contact generation kernels.
//!
//! Computes contact manifolds from collision pairs detected by the broad phase.

use crate::queries::{
    ContactManifold, IndexedManifold, ball_ball, ball_convex, convex_ball, cuboid_cuboid, pfm_pfm,
};
use crate::shapes::{
    Capsule, Polyline, SHAPE_TYPE_BALL, SHAPE_TYPE_CAPSULE, SHAPE_TYPE_CONE, SHAPE_TYPE_CUBOID,
    SHAPE_TYPE_CYLINDER, SHAPE_TYPE_POLYLINE, SHAPE_TYPE_TRIMESH, Shape, TriMesh,
};
use crate::{PaddedVector, Pose, Vector};
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::{sync::atomic_add_u32, iter::StepRng};

use crate::utils::{Slice, SliceMut};
use glamx::UVec2;

const WORKGROUP_SIZE: u32 = 64;

/// Resets the contacts counter.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_reset_narrow_phase(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] contacts_len: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] pfm_pairs_len: &mut [u32],
) {
    let batch_id = workgroup_id.y as usize;

    // NOTE: this `for` loop is silly. It doesn’t do anything
    //       more than a `*contacts_len = 0` in a convoluted
    //       way because otherwise rustgpu apparently does not generate
    //       the spirv for this kernel (seems to happen if the kernel is
    //       too trivial.
    for k in 0..1 {
        contacts_len.write(batch_id, k);
        pfm_pairs_len.write(batch_id, k);
    }
}

/// Initializes indirect dispatch arguments for constraint solver.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_narrow_phase_init_contacts_dispatch(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] contacts_len: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] indirect_args: &mut [u32; 3],
) {
    // For indirect dispatch, get the largest length along all batch dimensions.
    let mut highest_contacts_len = 0;
    for i in 0..contacts_len.len() {
        highest_contacts_len = highest_contacts_len.max(contacts_len.read(i));
    }

    *indirect_args.at_mut(0) = highest_contacts_len.div_ceil(WORKGROUP_SIZE);
    *indirect_args.at_mut(1) = contacts_len.len() as u32;
    *indirect_args.at_mut(2) = 1;
}

const PREDICTION: f32 = 2.0e-3; // TODO: make the prediction configurable.

/// Main narrow phase kernel.
///
/// Processes each collision pair, computes contacts, and filters by distance.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_narrow_phase_shape_shape(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] collision_pairs: &[[u32; 2]],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] collision_pairs_len: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] shapes: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] contacts: &mut [IndexedManifold],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] contacts_len: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)]
    pfm_pairs: &mut [NarrowPhasePfmPair],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] pfm_pairs_len: &mut [u32],
    // NOTE: we assume that max_pfm_pairs == contacts_batch_capacity
    //       And we assume all batch dimensions are given the same buffer allocation sizes
    //       (i.e. the same `contacts_batch_capacity`).
    #[spirv(uniform, descriptor_set = 0, binding = 8)] contacts_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 9)] colliders_batch_capacity: &u32,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] vertices: &[PaddedVector],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] indices: &[u32],
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y as usize;
    let contacts_batch_capacity = *contacts_batch_capacity as usize;
    let colliders_batch_capacity = *colliders_batch_capacity as usize;
    let contacts_start = batch_id * contacts_batch_capacity;
    let colliders_start = batch_id * colliders_batch_capacity;

    let collision_pairs = Slice(collision_pairs, contacts_start);
    let poses = Slice(poses, colliders_start);
    let shapes = Slice(shapes, colliders_start);
    let mut contacts = SliceMut(contacts, contacts_start);
    let mut pfm_pairs = SliceMut(pfm_pairs, contacts_start);
    let contacts_len = contacts_len.at_mut(batch_id);
    let pfm_pairs_len = pfm_pairs_len.at_mut(batch_id);
    let len = collision_pairs_len.read(batch_id);

    for i in StepRng::new(invocation_id.x..len, num_threads) {
        let pair = collision_pairs.read(i as usize);
        let pose1 = poses.read(pair.read(0) as usize);
        let pose2 = poses.read(pair.read(1) as usize);
        let shape1 = shapes.at(pair.read(0) as usize);
        let shape2 = shapes.at(pair.read(1) as usize);
        let shape_ty1 = shape1.shape_type();
        let shape_ty2 = shape2.shape_type();
        let mut manifold = ContactManifold::default();
        let pose12 = pose1.inverse() * pose2;
        let mut checked = false;

        // Ball - Convex
        if shape_ty1 == SHAPE_TYPE_BALL {
            if shape_ty2 == SHAPE_TYPE_BALL {
                let ball1 = shape1.to_ball();
                let ball2 = shape2.to_ball();
                manifold = ball_ball(pose12, &ball1, &ball2);
                checked = true;
            } else if shape_ty2 == SHAPE_TYPE_CUBOID
                || shape_ty2 == SHAPE_TYPE_CAPSULE
                || shape_ty2 == SHAPE_TYPE_CONE
                || shape_ty2 == SHAPE_TYPE_CYLINDER
            {
                let ball1 = shape1.to_ball();
                manifold = ball_convex(pose12, &ball1, shape2);
                checked = true;
            }
        }

        // Convex - Ball
        if !checked
            && shape_ty2 == SHAPE_TYPE_BALL
            && (shape_ty1 == SHAPE_TYPE_CUBOID
                || shape_ty1 == SHAPE_TYPE_CAPSULE
                || shape_ty1 == SHAPE_TYPE_CONE
                || shape_ty1 == SHAPE_TYPE_CYLINDER)
        {
            let ball2 = shape2.to_ball();
            manifold = convex_ball(pose12, shape1, &ball2);
            checked = true;
        }

        // Cuboid - Cuboid
        if !checked && shape_ty1 == SHAPE_TYPE_CUBOID && shape_ty2 == SHAPE_TYPE_CUBOID {
            let cuboid1 = shape1.to_cuboid();
            let cuboid2 = shape2.to_cuboid();
            manifold = cuboid_cuboid(pose12, &cuboid1, &cuboid2, PREDICTION);
            checked = true;
        }

        // PFM - PFM (generic convex shapes via GJK/EPA)
        // This is deferred to another kernel to reduce the spirv shader size
        // (otherwise it’s too big and hangs macos on loading).
        if !checked {
            let sub1 = shape1.pfm_subshape();
            let sub2 = shape2.pfm_subshape();

            if sub1.valid && sub2.valid {
                let pfm_pair = NarrowPhasePfmPair {
                    shape1: sub1.shape,
                    shape2: sub2.shape,
                    pose12,
                    thickness1: sub1.thickness,
                    thickness2: sub2.thickness,
                    colliders: UVec2::new(pair.read(0), pair.read(1)),
                };
                let pfm_index = atomic_add_u32(pfm_pairs_len, 1);
                pfm_pairs.write(pfm_index as usize, pfm_pair);

                // The actual calculations are deferred to another kernel.
                continue;
            }
        }

        // TriMesh - Convex
        // Note: trimesh collision writes contacts directly to the buffer and early-exits.
        if !checked && shape_ty1 == SHAPE_TYPE_TRIMESH {
            let mesh = shape1.to_trimesh();
            let convex = shape2;
            trimesh_convex(
                pose12,
                &mesh,
                convex,
                UVec2::new(pair.read(0), pair.read(1)),
                &mut pfm_pairs,
                pfm_pairs_len,
                vertices,
                indices,
            );
            continue;
        }

        if !checked && shape_ty2 == SHAPE_TYPE_TRIMESH {
            let convex = shape1;
            let mesh = shape2.to_trimesh();
            // NOTE: pair indices are flipped.
            trimesh_convex(
                pose12.inverse(),
                &mesh,
                convex,
                UVec2::new(pair.read(1), pair.read(0)),
                &mut pfm_pairs,
                pfm_pairs_len,
                vertices,
                indices,
            );
            continue;
        }

        // Polyline - Convex
        // Note: polyline collision writes contacts directly to the buffer and early-exits.
        if !checked && shape_ty1 == SHAPE_TYPE_POLYLINE {
            let pline = shape1.to_polyline();
            let convex = shape2;
            polyline_convex(
                pose12,
                &pline,
                convex,
                UVec2::new(pair.read(0), pair.read(1)),
                &mut pfm_pairs,
                pfm_pairs_len,
                vertices,
                indices,
            );
            continue;
        }

        if !checked && shape_ty2 == SHAPE_TYPE_POLYLINE {
            let convex = shape1;
            let pline = shape2.to_polyline();
            // NOTE: pair indices are flipped.
            polyline_convex(
                pose12.inverse(),
                &pline,
                convex,
                UVec2::new(pair.read(1), pair.read(0)),
                &mut pfm_pairs,
                pfm_pairs_len,
                vertices,
                indices,
            );
            continue;
        }

        if manifold.len > 0 && manifold.points_a.at(0).dist < PREDICTION {
            let target_contact_index = atomic_add_u32(contacts_len, 1) as usize;

            // NOTE: if we exceed the contacts allocation size, just skip
            //       the contact. It’s up to the caller to resize the buffer
            //       and re-run the narrow-phase.
            if target_contact_index < contacts_batch_capacity {
                contacts.write(
                    target_contact_index,
                    IndexedManifold {
                        contact: manifold,
                        colliders: UVec2::new(pair.read(0), pair.read(1)),
                        #[cfg(feature = "dim3")]
                        padding: [0; _],
                    },
                );
            }
        }
    }
}

/// Collision detection between a triangle mesh and a convex shape.
fn trimesh_convex(
    pose12: Pose,
    mesh: &TriMesh,
    convex: &Shape,
    pair: UVec2,
    pfm_pairs: &mut SliceMut<NarrowPhasePfmPair>,
    pfm_pairs_len: &mut u32,
    vertices: &[PaddedVector],
    indices: &[u32],
) {
    let sub2 = convex.pfm_subshape();
    if !sub2.valid {
        // Collisions with non-PFM shapes is not supported.
        return;
    }

    // Get the convex shape's AABB in the trimesh's local space, and enlarge with the PREDICTION.
    let mut test_aabb = convex.compute_aabb(pose12, vertices);
    test_aabb.mins -= Vector::splat(PREDICTION);
    test_aabb.maxs += Vector::splat(PREDICTION);

    if !test_aabb.intersects(&mesh.root_aabb) {
        // No collision possible.
        return;
    }

    let mut curr = 0u32;

    // NOTE: we use fixed-size for loops to avoid miscompilation issues of while loops on MacOs.
    for _ in 0..mesh.bvh_node_len {
        if curr >= mesh.bvh_node_len {
            break;
        }

        let idx = mesh.bvh_node_idx(indices, curr);
        if idx.entry_index == 0xffffffff {
            // This is a leaf.
            let tri = mesh.triangle(indices, vertices, idx.shape_index);
            let tri_shape = Shape::from_triangle(&tri);
            let sub1 = tri_shape.pfm_subshape();
            // TODO PERF: add special-cases for pairs that can be handled more efficiently than with GJK/EPA.
            let pfm_pair = NarrowPhasePfmPair {
                shape1: sub1.shape,
                shape2: sub2.shape,
                pose12,
                thickness1: sub1.thickness,
                thickness2: sub2.thickness,
                colliders: pair,
            };
            let pfm_index = atomic_add_u32(pfm_pairs_len, 1);
            pfm_pairs.write(pfm_index as usize, pfm_pair);

            // Continue traversal.
            curr = idx.exit_index;
        } else {
            let node_aabb = mesh.bvh_node_aabb(vertices, curr);
            if test_aabb.intersects(&node_aabb) {
                curr = idx.entry_index;
            } else {
                curr = idx.exit_index;
            }
        }
    }
}

/// Collision detection between a polyline and a convex shape.
///
/// This function is inlined because we need to write contacts directly to the output buffer,
/// and we can't pass storage buffers as function arguments.
fn polyline_convex(
    pose12: Pose,
    mesh: &Polyline,
    convex: &Shape,
    pair: UVec2,
    pfm_pairs: &mut SliceMut<NarrowPhasePfmPair>,
    pfm_pairs_len: &mut u32,
    vertices: &[PaddedVector],
    indices: &[u32],
) {
    let sub2 = convex.pfm_subshape();
    if !sub2.valid {
        // Collisions with non-PFM shapes is not supported.
        return;
    }

    // Get the convex shape's AABB in the polyline's local space, and enlarge with the PREDICTION.
    let thickness = 0.4; // TODO: make thickness configurable or part of the polyline struct
    let mut test_aabb = convex.compute_aabb(pose12, vertices);
    test_aabb.mins -= Vector::splat(PREDICTION + thickness);
    test_aabb.maxs += Vector::splat(PREDICTION + thickness);

    if !test_aabb.intersects(&mesh.root_aabb) {
        // No collision possible.
        return;
    }

    let mut curr = 0u32;

    // NOTE: we use fixed-size for loops to avoid miscompilation issues of while loops on MacOs.
    for _ in 0..mesh.bvh_node_len {
        if curr >= mesh.bvh_node_len {
            break;
        }

        let idx = mesh.bvh_node_idx(curr, indices);
        if idx.entry_index == 0xffffffff {
            // This is a leaf.
            let seg = mesh.segment(idx.shape_index, vertices, indices);
            // The segment is seen as a capsule with the given thickness.
            let capsule = Capsule::new(seg, thickness);
            let capsule_shape = Shape::from_capsule(&capsule);
            let sub1 = capsule_shape.pfm_subshape();
            // TODO PERF: add special-cases for pairs that can be handled more efficiently than with GJK/EPA.
            let pfm_pair = NarrowPhasePfmPair {
                shape1: sub1.shape,
                shape2: sub2.shape,
                pose12,
                thickness1: sub1.thickness,
                thickness2: sub2.thickness,
                colliders: pair,
            };
            let pfm_index = atomic_add_u32(pfm_pairs_len, 1);
            pfm_pairs.write(pfm_index as usize, pfm_pair);

            // Continue traversal.
            curr = idx.exit_index;
        } else {
            let node_aabb = mesh.bvh_node_aabb(curr, vertices);
            if test_aabb.intersects(&node_aabb) {
                curr = idx.entry_index;
            } else {
                curr = idx.exit_index;
            }
        }
    }
}

#[derive(Clone, Copy, Default)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
#[repr(C)]
pub struct NarrowPhasePfmPair {
    shape1: Shape,
    shape2: Shape,
    pose12: Pose,
    thickness1: f32,
    thickness2: f32,
    colliders: UVec2,
}

/// Initializes PFM-PFM dispatch arguments for constraint solver.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_init_pfm_pfm_dispatch(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] pfm_pairs_len: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] indirect_args: &mut [u32; 3],
) {
    let mut highest_pfm_pairs_len = 0;
    for batch_id in 0..pfm_pairs_len.len() {
        highest_pfm_pairs_len = highest_pfm_pairs_len.max(pfm_pairs_len.read(batch_id));
    }
    // TODO PERF: pfm_pfm is very divergent. Use a smaller workgroup size?
    *indirect_args.at_mut(0) = highest_pfm_pairs_len.div_ceil(WORKGROUP_SIZE);
    *indirect_args.at_mut(1) = pfm_pairs_len.len() as u32;
    *indirect_args.at_mut(2) = 1;
}

#[spirv_bindgen]
#[spirv(compute(threads(64)))] // TODO PERF: pfm_pfm is very divergent. Use a smaller workgroup size?
pub fn gpu_narrow_phase_pfm_pfm(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] contacts: &mut [IndexedManifold],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] contacts_len: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] pfm_pairs: &[NarrowPhasePfmPair],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] pfm_pairs_len: &[u32],
    // NOTE: we assume that max_pfm_pairs == contacts_batch_capacity
    //       And we assume all batch dimensions are given the same buffer allocation sizes
    //       (i.e. the same `contacts_batch_capacity`).
    #[spirv(uniform, descriptor_set = 0, binding = 4)] contacts_batch_capacity: &u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] vertices: &[PaddedVector],
    #[allow(unused_variables)]
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)]
    indices: &[u32],
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y as usize;
    let contacts_batch_capacity = *contacts_batch_capacity as usize;
    let start_id = batch_id * contacts_batch_capacity;

    let mut contacts = SliceMut(contacts, start_id);
    let pfm_pairs = Slice(pfm_pairs, start_id);
    let contacts_len = contacts_len.at_mut(batch_id);
    let pfm_pairs_len = pfm_pairs_len.read(batch_id);

    for i in StepRng::new(invocation_id.x..pfm_pairs_len, num_threads) {
        let pair = pfm_pairs.read(i as usize);
        let manifold = pfm_pfm(
            pair.pose12,
            &pair.shape1,
            pair.thickness1,
            &pair.shape2,
            pair.thickness2,
            PREDICTION,
            vertices,
            #[cfg(feature = "dim3")]
            indices,
        );

        if manifold.len > 0 && manifold.points_a.at(0).dist < PREDICTION {
            let target_contact_index = atomic_add_u32(contacts_len, 1) as usize;

            // NOTE: if we exceed the contacts allocation size, just skip
            //       the contact. It’s up to the caller to resize the buffer
            //       and re-run the narrow-phase.
            if target_contact_index < contacts_batch_capacity {
                contacts.write(
                    target_contact_index,
                    IndexedManifold {
                        contact: manifold,
                        colliders: pair.colliders,
                        #[cfg(feature = "dim3")]
                        padding: [0; _],
                    },
                );
            }
        }
    }
}
