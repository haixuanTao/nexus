//! Linear Bounding Volume Hierarchy (LBVH)
//!
//! This module implements a GPU-based LBVH construction and traversal. It is based on
//! the paper: https://research.nvidia.com/sites/default/files/publications/karras2012hpg_paper.pdf
//! The LBVH provides O(n log n) construction and O(n log n) query complexity.
//!
//! Pipeline stages:
//! 1. compute_domain: Parallel reduction to find AABB of all collider positions.
//! 2. compute_morton: Assigns 30-bit (3D) or 20-bit (2D) Morton codes to colliders.
//! 3. [External radix sort]: Sorts colliders by Morton code.
//! 4. build: Constructs binary tree topology in parallel (Karras algorithm).
//! 5. refit_leaves: Computes leaf AABBs from shapes.
//! 6. refit_internal: Bottom-up AABB propagation using atomic synchronization.
//! 7. find_collision_pairs: Tree traversal for each collider to find intersections.
//!
//! Data structures:
//! - LbvhNode: Binary tree node with AABB + left/right/parent pointers + refit counter
//! - Tree layout: Internal nodes [0..n-1], leaves [n..2n-1]
//! - Leaf nodes store collider index in 'left' field

use crate::bounding_volumes::Aabb;
use crate::shapes::Shape;
use crate::{atomic_add_u32, MaybeIndexUnchecked, Pose, Vector, VectorWithPadding};
use khal_derive::spirv_bindgen;
use spirv_std::arch::{control_barrier, workgroup_memory_barrier_with_group_sync};
use spirv_std::glam::UVec3;
use spirv_std::spirv;
use vortx_shaders::utils::step::StepRng;

const WORKGROUP_SIZE: u32 = 64;
const REDUCTION_WORKGROUP_SIZE: u32 = 128;

/// A node in the Linear BVH tree.
///
/// The tree has n-1 internal nodes and n leaf nodes. Internal nodes are stored
/// in indices [0..n-1], leaf nodes in [n..2n].
///
/// For internal nodes:
/// - left/right point to child nodes (may be internal or leaf)
/// - parent points to parent internal node (0 for root)
/// - refit_count tracks how many children have updated this node's AABB
///
/// For leaf nodes:
/// - left stores the collider index
/// - right is unused
/// - parent points to parent internal node
/// - refit_count is unused
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch = "spirv"), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct LbvhNode {
    /// Axis-aligned bounding box for this node's subtree.
    pub aabb: Aabb,
    /// Left child index (internal) or collider index (leaf).
    pub left: u32,
    /// Right child index (internal nodes only).
    pub right: u32,
    /// Parent node index.
    pub parent: u32,
    /// Counter for bottom-up refitting (0, 1, or 2).
    /// When a thread arrives at a node, it atomically increments this.
    /// If the old value was 0, the thread stops (sibling hasn't arrived yet).
    /// If the old value was 1, the thread continues upward (both children ready).
    pub refit_count: u32,
}

/// Resets the collision pairs counter.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_lbvh_reset_collision_pairs(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] collision_pairs_len: &mut u32,
) {
    // NOTE: this `for` loop is silly. It doesn’t do anything
    //       more than a `*collision_pairs_len = 0` in a convoluted
    //       way because otherwise rustgpu apparently does not generate
    //       the spirv for this kernel (seems to happen if the kernel is
    //       too trivial.
    for k in 0..1 {
        *collision_pairs_len = k;
    }
}

/// Initializes indirect dispatch arguments for narrow phase.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_lbvh_init_dispatch(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] collision_pairs_len: &u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] indirect_args: &mut [u32; 3],
) {
    *indirect_args.at_mut(0) = collision_pairs_len.div_ceil(WORKGROUP_SIZE);
    *indirect_args.at_mut(1) = 1;
    *indirect_args.at_mut(2) = 1;
}

/// Runs a reduction to compute the AABB of the collider positions.
/// Needs to be called with a single workgroup.
#[spirv_bindgen]
#[spirv(compute(threads(128)))]
pub fn gpu_lbvh_compute_domain(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] domain_aabb: &mut Aabb,
    #[spirv(uniform, descriptor_set = 0, binding = 2)] num_colliders: &u32,
    #[spirv(workgroup)] workspace_mins: &mut [Vector; 128],
    #[spirv(workgroup)] workspace_maxs: &mut [Vector; 128],
) {
    let thread_id = invocation_id.x;
    *workspace_mins.at_mut(thread_id as usize) = Vector::splat(1.0e20);
    *workspace_maxs.at_mut(thread_id as usize) = Vector::splat(-1.0e20);

    for i in StepRng::new(thread_id..*num_colliders, REDUCTION_WORKGROUP_SIZE) {
        let val_i = poses.at(i as usize).translation;
        *workspace_mins.at_mut(thread_id as usize) =
            workspace_mins.at(thread_id as usize).min(val_i);
        *workspace_maxs.at_mut(thread_id as usize) =
            workspace_maxs.at(thread_id as usize).max(val_i);
    }

    workgroup_memory_barrier_with_group_sync();

    // Reduction steps
    macro_rules! step_reduce(
        ($stride: expr) => {
            if thread_id < $stride {
                *workspace_mins.at_mut(thread_id as usize) = workspace_mins.at(thread_id as usize)
                    .min(*workspace_mins.at((thread_id + $stride) as usize));
                *workspace_maxs.at_mut(thread_id as usize) = workspace_maxs.at(thread_id as usize)
                    .max(*workspace_maxs.at((thread_id + $stride) as usize));
            }
            workgroup_memory_barrier_with_group_sync();
        }
    );
    step_reduce!(64);
    step_reduce!(32);
    step_reduce!(16);
    step_reduce!(8);
    step_reduce!(4);
    step_reduce!(2);
    step_reduce!(1);

    if thread_id == 0 {
        domain_aabb.mins = *workspace_mins.at(0);
        domain_aabb.maxs = *workspace_maxs.at(0);
    }
}

/// Computes Morton codes for all colliders.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_lbvh_compute_morton(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] domain_aabb: &Aabb,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] morton_keys: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] num_colliders: &u32,
) {
    // NOTE: for simplicity we compute the morton key of the collider position instead of
    //       the collider shape's AABB center. We might want to revisit that in the future
    //       once we start adding more complex shapes.
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;

    for i in StepRng::new(invocation_id.x..*num_colliders, num_threads) {
        let center = poses.at(i as usize).translation;
        let normalized = (center - domain_aabb.mins) / (domain_aabb.maxs - domain_aabb.mins);
        let morton_key = morton(normalized);
        morton_keys.write(i as usize, morton_key);
    }
}

/// Builds each node of the tree in parallel.
///
/// This only computes the tree topology (children and parent pointers).
/// This doesn't update the bounding boxes. Call `refit` for updating bounding boxes!
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_lbvh_build(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] morton_keys: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] tree: &mut [LbvhNode],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] num_colliders: &u32,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    let num_bodies = *num_colliders;
    let num_internal_nodes = num_bodies - 1;
    let first_leaf_id = num_internal_nodes;

    for i in StepRng::new(invocation_id.x..num_internal_nodes, num_threads) {
        // Determine the direction of the range (+1 or -1).
        let ii = i as i32;
        let curr_key = morton_keys.read(i as usize);
        let diff = prefix_len(curr_key, ii, ii + 1, num_bodies, morton_keys)
            - prefix_len(curr_key, ii, ii - 1, num_bodies, morton_keys);
        let d = if diff > 0 {
            1
        } else if diff < 0 {
            -1
        } else {
            0
        }; // Not using `diff.signum()` since it fail spirv compilation without `Int8` capabilities.

        // Compute upper bound for the length of the range.
        let delta_min = prefix_len(curr_key, ii, ii - d, num_bodies, morton_keys);
        let mut lmax = 2;

        for _ in 0..32u32 {
            if prefix_len(curr_key, ii, ii + lmax * d, num_bodies, morton_keys) > delta_min {
                lmax *= 2;
            } else {
                break;
            }
        }

        // Find the other end using binary search.
        let mut l = 0;
        let mut t = lmax / 2;

        // NOTE: we use fixed-size for loops to avoid miscompilation issues of while loops on MacOs.
        //       Running up to 32 loops is always correct since we can’t have more than 2^30 morton
        //       keys.
        for _ in 0..32u32 {
            if t < 1 {
                break;
            }
            if prefix_len(curr_key, ii, ii + (l + t) * d, num_bodies, morton_keys) > delta_min {
                l += t;
            }
            t /= 2;
        }
        let j = ii + l * d;

        // Find the split position using binary search.
        let delta_node = prefix_len(curr_key, ii, j, num_bodies, morton_keys);
        let mut s = 0;
        let mut t = div_ceil(l, 2);

        for _ in 0..32u32 {
            if t < 1 {
                break;
            }
            if prefix_len(curr_key, ii, ii + (s + t) * d, num_bodies, morton_keys) > delta_node {
                s += t;
            }
            t = div_ceil(t, 2);
        }

        let gamma = ii + s * d + d.min(0);

        // Output child and parent pointers.
        let left = if ii.min(j) == gamma {
            first_leaf_id as i32 + gamma
        } else {
            gamma
        };
        let right = if ii.max(j) == gamma + 1 {
            first_leaf_id as i32 + gamma + 1
        } else {
            gamma + 1
        };
        tree.at_mut(i as usize).left = left as u32;
        tree.at_mut(i as usize).right = right as u32;
        tree.at_mut(i as usize).refit_count = 0; // Might as well reset the refit count here.
        tree.at_mut(left as usize).parent = i;
        tree.at_mut(right as usize).parent = i;
    }
}

/// Computes leaf AABBs from shapes.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_lbvh_refit_leaves(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] shapes: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] sorted_colliders: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] tree: &mut [LbvhNode],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] num_colliders: &u32,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] vertices: &[VectorWithPadding],
) {
    // TODO PERF: we could use shared memory atomics between threads belonging to the same
    //            workgroup.
    // Bottom-up refit. Leaf index starts at `num_colliders`.
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    let first_leaf_id = *num_colliders - 1;

    for i in StepRng::new(invocation_id.x..*num_colliders, num_threads) {
        let curr_leaf_id = first_leaf_id + i;
        let leaf_collider = sorted_colliders.read(i as usize);
        let leaf_pose = poses.read(leaf_collider as usize);
        let leaf_shape = shapes.at(leaf_collider as usize);

        tree.at_mut(curr_leaf_id as usize).aabb = leaf_shape.compute_aabb(leaf_pose, vertices);
        tree.at_mut(curr_leaf_id as usize).left = leaf_collider;
    }
}

/// Bottom-up AABB propagation using atomic synchronization.
/// This version uses uniform control flow for web compatibility.
#[spirv_bindgen]
#[spirv(compute(threads(256)))]
pub fn gpu_lbvh_refit_internal(
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] tree: &mut [LbvhNode],
    #[spirv(uniform, descriptor_set = 0, binding = 1)] num_colliders: &u32,
) {
    // TODO PERF: we could use shared memory atomics between threads belonging to the same
    //            workgroup.
    // Bottom-up refit. Leaf index starts at `num_colliders`.
    let num_threads = 256u32;
    let num_bodies = *num_colliders;
    let first_leaf_id = num_bodies - 1;

    // All threads must execute the same number of outer loop iterations for uniform control flow.
    let num_iterations = num_bodies.div_ceil(num_threads);

    // NOTE: using unchecked indexing (via MaybeIndexUnchecked) because otherwise the bounds
    //        checking inserted by rustgpu breaks the shader when targeting some NVidia graphics
    //        (works fine on AMD).
    for iter in 0..num_iterations {
        let i = local_id.x + iter * num_threads;
        let mut thread_is_active = i < num_bodies;

        let mut curr_id = 0u32;
        if thread_is_active {
            let curr_leaf_id = first_leaf_id + i;
            curr_id = tree.at(curr_leaf_id as usize).parent;
        }

        // Process the tree level by level with uniform barriers.
        // Maximum tree depth is log2(num_colliders), but we use 32 as a safe upper bound.
        for _level in 0..32u32 {
            if thread_is_active {
                let refit_count = atomic_add_u32(&mut tree.at_mut(curr_id as usize).refit_count, 1);

                if refit_count == 0 {
                    // If `refit_count` was 0 then the other thread hasn't reached this node
                    // yet and the sibbling aabb might not be available yet.
                    // Stop the propagation to the parents here, the other thread will do it.
                    thread_is_active = false;
                } else {
                    // If `refit_count` was 1 then the other thread has already reached this node
                    // and we know the sibblings aabb is available. So we can continue the propagation.

                    // TODO PERF: instead of re-reading both aabbs, we could keep the aabb from the
                    //            previous loop so we don't have to re-fetch one of the two aabbs.
                    let left_idx = tree.at(curr_id as usize).left;
                    let right_idx = tree.at(curr_id as usize).right;
                    let left = tree.at(left_idx as usize).aabb;
                    let right = tree.at(right_idx as usize).aabb;
                    tree.at_mut(curr_id as usize).aabb = left.merged(&right);

                    if curr_id == 0 {
                        // We reached the root, can't go higher.
                        thread_is_active = false;
                    } else {
                        curr_id = tree.at(curr_id as usize).parent;
                    }
                }
            }

            // Barrier ensures all AABB writes (to device/storage buffer memory) are complete
            // before the next iteration's atomics. Uses QueueFamily scope (device-equivalent
            // under the Vulkan memory model) with UNIFORM_MEMORY to cover storage buffers.
            control_barrier::<
                { spirv_std::memory::Scope::Workgroup as u32 },
                { spirv_std::memory::Scope::QueueFamily as u32 },
                {
                    spirv_std::memory::Semantics::UNIFORM_MEMORY.bits()
                        | spirv_std::memory::Semantics::ACQUIRE_RELEASE.bits()
                },
            >();
        }
    }
}

/// Full refit: computes leaf AABBs and propagates to ancestors.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_lbvh_refit(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] shapes: &[Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] sorted_colliders: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] tree: &mut [LbvhNode],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] num_colliders: &u32,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] vertices: &[VectorWithPadding],
) {
    // TODO PERF: we could use shared memory atomics between threads belonging to the same
    //            workgroup.
    // Bottom-up refit. Leaf index starts at `num_colliders`.
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    let num_bodies = *num_colliders;
    let first_leaf_id = num_bodies - 1;

    for i in StepRng::new(invocation_id.x..num_bodies, num_threads) {
        let curr_leaf_id = first_leaf_id + i;
        let leaf_collider = sorted_colliders.read(i as usize);
        let leaf_pose = poses.read(leaf_collider as usize);
        let leaf_shape = shapes.at(leaf_collider as usize);

        tree.at_mut(curr_leaf_id as usize).aabb = leaf_shape.compute_aabb(leaf_pose, vertices);
        tree.at_mut(curr_leaf_id as usize).left = leaf_collider;

        // Propagate to ancestors.
        let mut curr_id = tree.at(curr_leaf_id as usize).parent;

        loop {
            let refit_count = atomic_add_u32(&mut tree.at_mut(curr_id as usize).refit_count, 1);

            if refit_count == 0 {
                // If `refit_count` was 0 then the other thread hasn't reached this node
                // yet and the sibbling aabb might not be available yet.
                // Stop the propagation to the parents here, the other thread will do it.
                break;
            }

            // If `refit_count` was 1 then the other thread has already reached this node
            // and we know the sibblings aabb is available. So we can continue the propagation.

            // TODO PERF: instead of re-reading both aabbs, we could keep the aabb from the
            //            previous loop so we don't have to re-fetch one of the two aabbs.
            let left_idx = tree.at(curr_id as usize).left;
            let right_idx = tree.at(curr_id as usize).right;
            let left = tree.at(left_idx as usize).aabb;
            let right = tree.at(right_idx as usize).aabb;
            tree.at_mut(curr_id as usize).aabb = left.merged(&right);

            if curr_id == 0 {
                // We reached the root, can't go higher.
                break;
            }

            curr_id = tree.at(curr_id as usize).parent;
        }
    }
}

/// Finds collision pairs by traversing the LBVH tree.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_lbvh_find_collision_pairs(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] tree: &[LbvhNode],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] collision_pairs: &mut [[u32; 2]],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] collision_pairs_len: &mut u32,
    #[spirv(uniform, descriptor_set = 0, binding = 3)] num_colliders: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 4)] max_collision_pairs: &u32,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    let num_bodies = *num_colliders;
    let first_leaf_id = num_bodies - 1;

    for leaf_i in StepRng::new(invocation_id.x..num_bodies, num_threads) {
        let i = tree.at((first_leaf_id + leaf_i) as usize).left;
        let mut aabb1 = tree.at((first_leaf_id + leaf_i) as usize).aabb;
        let prediction = 2.0e-3; // TODO: should be configurable.
        let dilation = Vector::splat(prediction);
        aabb1.mins -= dilation;
        aabb1.maxs += dilation;

        // Traverse the tree using a stack.
        let mut stack = [0u32; 64];
        let mut stack_len = 1u32;
        stack.write(0, 0);

        while stack_len != 0 {
            stack_len -= 1;
            let curr_id = stack.read(stack_len as usize);
            let node = tree.at(curr_id as usize);

            if curr_id >= first_leaf_id {
                // We reached a leaf, register a collision pair.
                let j = node.left;

                // NOTE: we don't have to compare i < j to avoid duplicates since that comparison already happened
                //       alongside the AABB check.
                let target_pair_index = atomic_add_u32(collision_pairs_len, 1);

                // NOTE: if the index is out-of-bounds (meaning the `collision_pairs` isn't
                //       big enough), don't write. But keep traversing so we get the exact count we need
                //       for reallocating the buffers.
                if target_pair_index < *max_collision_pairs {
                    collision_pairs.write(target_pair_index as usize, [i, j]);
                }
            } else {
                let left = node.left;
                let right = node.right;

                // Go on the child only if the AABB intersects and either the child isn't a leaf, or it is a leaf with associated collider
                // smaller than `i` (to avoid duplicate pairs).
                if (left < first_leaf_id || i < tree.at(left as usize).left)
                    && aabb1.intersects(&tree.at(left as usize).aabb)
                    && stack_len < 64
                {
                    stack.write(stack_len as usize, node.left);
                    stack_len += 1;
                }

                // NOTE: on leaves (including tree[right]), the collider id is stored as the left child index.
                if (right < first_leaf_id || i < tree.at(right as usize).left)
                    && aabb1.intersects(&tree.at(right as usize).aabb)
                    && stack_len < 64
                {
                    stack.write(stack_len as usize, node.right);
                    stack_len += 1;
                }
            }
        }
    }
}

/// Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit (3D).
#[cfg(feature = "dim3")]
pub fn expand_bits_3d(v: u32) -> u32 {
    let mut vv = (v * 0x00010001) & 0xFF0000FF;
    vv = (vv * 0x00000101) & 0x0F00F00F;
    vv = (vv * 0x00000011) & 0xC30C30C3;
    vv = (vv * 0x00000005) & 0x49249249;
    vv
}

/// Calculates a 30-bit Morton code for the given 3D point located within the unit cube [0,1].
#[cfg(feature = "dim3")]
pub fn morton_3d(v: Vector) -> u32 {
    let scaled_x = v.x.clamp(0.0, 1023.0 / 1024.0) * 1024.0;
    let scaled_y = v.y.clamp(0.0, 1023.0 / 1024.0) * 1024.0;
    let scaled_z = v.z.clamp(0.0, 1023.0 / 1024.0) * 1024.0;
    let xx = expand_bits_3d(scaled_x as u32);
    let yy = expand_bits_3d(scaled_y as u32);
    let zz = expand_bits_3d(scaled_z as u32);
    xx * 4 + yy * 2 + zz
}

/// Expands a 16-bit integer into 32 bits by inserting 1 zero after each bit (2D).
#[cfg(feature = "dim2")]
pub fn expand_bits_2d(v: u32) -> u32 {
    let mut x = v & 0x0000ffff;
    x = (x | (x << 8)) & 0x00ff00ff;
    x = (x | (x << 4)) & 0x0f0f0f0f;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    x
}

/// Calculates a 32-bit Morton code for the given 2D point located within the unit square [0,1].
#[cfg(feature = "dim2")]
pub fn morton_2d(v: Vector) -> u32 {
    let scaled_x = (v.x * 65536.0).clamp(0.0, 65535.0);
    let scaled_y = (v.y * 65536.0).clamp(0.0, 65535.0);
    let xx = expand_bits_2d(scaled_x as u32);
    let yy = expand_bits_2d(scaled_y as u32);
    xx | (yy << 1)
}

/// Calculates a Morton code for the given point located within the unit hypercube [0,1].
#[cfg(feature = "dim2")]
pub fn morton(v: Vector) -> u32 {
    morton_2d(v)
}

/// Calculates a Morton code for the given point located within the unit hypercube [0,1].
#[cfg(feature = "dim3")]
pub fn morton(v: Vector) -> u32 {
    morton_3d(v)
}

/// Computes the common prefix length between two Morton keys.
pub fn prefix_len(
    curr_key: u32,
    curr_index: i32,
    other_index: i32,
    num_colliders: u32,
    morton_keys: &[u32],
) -> i32 {
    if other_index < 0 || other_index > num_colliders as i32 - 1 {
        return -1;
    }

    let other_key = morton_keys.read(other_index as usize);
    let morton_prefix_len = ((curr_key as i32) ^ (other_key as i32)).leading_zeros() as i32;
    // Fallback to indices if the morton keys are equal.
    let fallback_prefix_len = 32 + (curr_index ^ other_index).leading_zeros() as i32;
    if curr_key != other_key {
        morton_prefix_len
    } else {
        fallback_prefix_len
    }
}

/// Division with ceiling.
pub fn div_ceil(x: i32, y: i32) -> i32 {
    (x + y - 1) / y
}
