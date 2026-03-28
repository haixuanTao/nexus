//! Graph coloring for parallel constraint solving
//!
//! This module implements graph coloring algorithms to enable parallel constraint
//! solving. The goal is to assign colors to constraints such that no two constraints
//! of the same color share a rigid body, allowing parallel solving within each color.
//!
//! Why Graph Coloring?
//! Sequential constraint solvers (Gauss-Seidel) are inherently serial because
//! solving one constraint affects bodies involved in other constraints. Graph
//! coloring breaks this dependency: constraints with the same color are independent
//! and can be solved in parallel.
//!
//! Algorithms Implemented:
//!
//! 1. Jones-Plassmann-Luby (Luby's Algorithm):
//!    - Randomized parallel algorithm.
//!    - Each iteration: nodes with locally maximal random weights are colored.
//!    - Better for sparse conflict graphs.
//!    - Reference: https://developer.nvidia.com/blog/graph-coloring-more-parallelism-for-incomplete-lu-factorization/
//!
//! 2. Topo-GC (Topological Graph Coloring):
//!    - Each node selects the smallest available color not used by neighbors.
//!    - Typically produces fewer colors than Luby.
//!    - Better for dense conflict graphs.
//!    - Reference: https://people.csail.mit.edu/xchen/docs/ipdpsw-2016.pdf
//!
//! Performance:
//! - Workgroup size: 64 threads (Topo-GC uses 1 for some kernels)
//! - Luby: O(log n) iterations (average, O(n) worst-case).
//! - Topo-GC: O(d) iterations where d is max degree, fewer colors
//!
//! Color Limits:
//! - Topo-GC: Up to 63 colors (using 2x u32 bitmasks with 1 reserved bit for bookkeeping)
//! - Luby: Unlimited colors

use khal_std::glamx::UVec3;
use khal_std::{iter::StepRng, arch::{atomic_add_u32, atomic_max_u32}};
use khal_std::macros::{spirv, spirv_bindgen};

use khal_std::index::MaybeIndexUnchecked;
use crate::utils::{Slice, SliceMut};

use super::constraint::TwoBodyConstraint;

const WORKGROUP_SIZE: u32 = 64;

/// Maximum u32 value (used to mark uncolored constraints in Luby algorithm).
pub const MAX_U32: u32 = u32::MAX;

/// Hash function for generating random weights.
///
/// Uses a variant of the Murmur3 hash function to generate pseudo-random
/// weights from constraint indices.
#[inline]
fn hash(packed_key: u32) -> u32 {
    let mut key = packed_key;
    key *= 0xcc9e2d51;
    key = key.rotate_left(15);
    key *= 0x1b873593;
    key
}

/*
 * Jones-Plassmann-Luby Graph Coloring Algorithm
 *
 * Randomized parallel graph coloring algorithm. In each iteration:
 * 1. Each uncolored node compares its random weight with neighbors
 * 2. Nodes with locally maximal weights are colored
 * 3. Repeat until all nodes are colored
 *
 * Expected iterations: O(log n) for bounded-degree graphs
 * Expected colors: O(Δ) where Δ is maximum degree
 *
 * This implementation uses a hash function instead of a true RNG for
 * simplicity.
 */

/// Initializes Luby algorithm state.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_reset_luby(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] constraints_colors: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] constraints_rands: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contacts_len: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] contacts_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let contacts_start = batch_id * *contacts_batch_capacity as usize;
    let mut constraints_colors = SliceMut(constraints_colors, contacts_start);
    let mut constraints_rands = SliceMut(constraints_rands, contacts_start);
    let len = contacts_len.read(batch_id);

    let i = invocation_id.x;

    if i < len {
        let idx = i as usize;
        // Mark as uncolored
        constraints_colors.write(idx, MAX_U32);
        // Assign random weight
        constraints_rands.write(idx, hash(i));
    }
}

/// Performs one iteration of Luby's graph coloring algorithm.
///
/// For each uncolored constraint:
/// 1. Compare random weight with all neighboring constraints
/// 2. If this constraint has the maximum weight among uncolored neighbors,
///    assign it the current color
/// 3. Otherwise, leave it uncolored for the next iteration
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_step_graph_coloring_luby(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] body_constraint_counts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] body_constraint_ids: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] constraints: &[TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] constraints_colors: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] constraints_rands: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] uncolored: &mut u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] curr_color: &u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] contacts_len: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 8)] contacts_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 9)] colliders_batch_capacity: &u32,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y as usize;
    let contacts_start = batch_id * *contacts_batch_capacity as usize;
    let colliders_start = batch_id * *colliders_batch_capacity as usize;
    let bci_start = batch_id * 2 * *contacts_batch_capacity as usize;

    let body_constraint_counts = Slice(body_constraint_counts, colliders_start);
    let body_constraint_ids = Slice(body_constraint_ids, bci_start);
    let constraints = Slice(constraints, contacts_start);
    let mut constraints_colors = SliceMut(constraints_colors, contacts_start);
    let constraints_rands = Slice(constraints_rands, contacts_start);

    let len = contacts_len.read(batch_id);
    let color = *curr_color;

    for constraint_i in StepRng::new(invocation_id.x..len, num_threads) {
        let i = constraint_i as usize;

        if constraints_colors.read(i) == MAX_U32 {
            // This constraint doesn't have a color yet.
            let rand_i = constraints_rands.read(i);
            let body_a = constraints.at(i).solver_body_a;
            let body_b = constraints.at(i).solver_body_b;

            let first_constraint_id_a = if body_a != 0 {
                body_constraint_counts.read(body_a as usize - 1) as usize
            } else {
                0
            };
            let last_constraint_id_a = body_constraint_counts.read(body_a as usize) as usize;

            let first_constraint_id_b = if body_b != 0 {
                body_constraint_counts.read(body_b as usize - 1) as usize
            } else {
                0
            };
            let last_constraint_id_b = body_constraint_counts.read(body_b as usize) as usize;

            let mut is_greatest = true;

            // Traverse all constraints from body A.
            for j in first_constraint_id_a..last_constraint_id_a {
                if !is_greatest {
                    break;
                }
                let constraint_j = body_constraint_ids.read(j);
                let rand_j = constraints_rands.read(constraint_j as usize);
                let color_j = constraints_colors.read(constraint_j as usize);
                // NOTE: there is a very rare case both constraints got assigned the same random number.
                //       in that case, we define the "greatest" comparison based on the constraint's array index.
                // NOTE: the equality in i >= j is important here to account for the fact we will iterate
                //       through the current constraint's index too.
                is_greatest = is_greatest
                    && (color_j != MAX_U32
                        || rand_i > rand_j
                        || (rand_i == rand_j && constraint_i >= constraint_j));
            }

            // Traverse all constraints from body B.
            for j in first_constraint_id_b..last_constraint_id_b {
                if !is_greatest {
                    break;
                }
                let cid = body_constraint_ids.read(j);
                let rand_j = constraints_rands.read(cid as usize);
                let color_j = constraints_colors.read(cid as usize);
                // NOTE: there is a very rare case both constraints got assigned the same random number.
                //       in that case, we define the "greatest" comparison based on the constraint's array index.
                // NOTE: the equality in i >= j is important here to account for the fact we will iterate
                //       through the current constraint's index too.
                is_greatest = is_greatest
                    && (color_j != MAX_U32
                        || rand_j < rand_i
                        || (rand_i == rand_j && constraint_i >= cid));
            }

            if is_greatest {
                constraints_colors.write(i, color);
            } else {
                // Still uncolored
                atomic_add_u32(uncolored, 1);
            }
        }
    }
}

/*
 * Topo-GC (Topological Graph Coloring) Algorithm
 * ==============================================================================
 *
 * Parallel graph coloring algorithm. Each iteration:
 * 1. Each uncolored node selects the smallest color not used by neighbors
 * 2. Conflicts are detected and resolved in the next iteration
 * 3. Repeat until convergence (no conflicts)
 *
 * Reference: https://people.csail.mit.edu/xchen/docs/ipdpsw-2016.pdf
 *
 * Advantages:
 * - Typically produces fewer colors than Luby (closer to optimal)
 * - Faster convergence for dense graphs
 *
 * Disadvantages:
 * - Limited to 63 colors (due to 2x u32 bitmask representation)
 * - May require more iterations than Luby in worst case
 *
 * Algorithm Steps:
 * 1. reset_topo_gc_kernel: Initialize all nodes as uncolored
 * 2. step_graph_coloring_topo_gc_kernel: Each node selects smallest available color
 * 3. fix_conflicts_topo_gc_kernel: Detect and uncolor conflicting nodes
 * 4. Repeat steps 2-3 until num_colors > 0 (no conflicts, algorithm finished)
 *
 * Color Representation:
 * Uses a 64-bit bitmask (2x u32) to track occupied colors for each node.
 * Bit i set means color i is used by a neighbor.
 * Color indices start at 1. (The index 0 is reserved as an implementation detail.)
 */

/// Initializes Topo-GC algorithm state.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_reset_topo_gc(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] constraints_colors: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] colored: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contacts_len: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] contacts_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let contacts_start = batch_id * *contacts_batch_capacity as usize;
    let mut constraints_colors = SliceMut(constraints_colors, contacts_start);
    let mut colored = SliceMut(colored, contacts_start);
    let len = contacts_len.read(batch_id);

    let i = invocation_id.x;

    if i < len {
        let idx = i as usize;
        // Color 0 is reserved for "uncolored" state
        constraints_colors.write(idx, 0);
        colored.write(idx, 0);
    }
}

/// Resets the convergence flag for Topo-GC.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_reset_completion_flag_topo_gc(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] num_colors: &mut u32,
) {
    if invocation_id.x == 0 {
        // NOTE: this `for` loop is silly. It doesn’t do anything
        //       more than a `*num_colors = 1` in a convoluted
        //       way because otherwise rustgpu apparently does not generate
        //       the spirv for this kernel (seems to happen if the kernel is
        //       too trivial.
        for k in 0..1 {
            // Non-zero value indicates algorithm should continue
            *num_colors = k + 1;
        }
    }
}

/// Performs one iteration of Topo-GC coloring.
///
/// For each uncolored constraint:
/// 1. Build a bitmask of colors used by neighboring constraints
/// 2. Select the smallest color (lowest bit) not in the bitmask
/// 3. Mark this constraint as colored
///
/// Uses a 64-bit bitmask (2x u32) to track up to 63 colors (color 0 = uncolored).
/// countTrailingZeros finds the position of the first unset bit, giving the
/// smallest available color.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_step_graph_coloring_topo_gc(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] body_constraint_counts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] body_constraint_ids: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] constraints: &[TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] constraints_colors: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] colored: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] num_colors: &mut u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] contacts_len: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 7)] contacts_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 8)] colliders_batch_capacity: &u32,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y as usize;
    let contacts_start = batch_id * *contacts_batch_capacity as usize;
    let colliders_start = batch_id * *colliders_batch_capacity as usize;
    let bci_start = batch_id * 2 * *contacts_batch_capacity as usize;

    let body_constraint_counts = Slice(body_constraint_counts, colliders_start);
    let body_constraint_ids = Slice(body_constraint_ids, bci_start);
    let constraints = Slice(constraints, contacts_start);
    let mut constraints_colors = SliceMut(constraints_colors, contacts_start);
    let mut colored = SliceMut(colored, contacts_start);

    let len = contacts_len.read(batch_id);

    for constraint_i in StepRng::new(invocation_id.x..len, num_threads) {
        let i = constraint_i as usize;

        if colored.read(i) == 0 {
            // This constraint doesn't have a color yet.
            // NOTE: generates up to 63 colors.
            // Note that we always mark the color 0 as occupied (cf. paper using i > 0).
            let mut color_mask = (1u32, 0u32);

            let body_a = constraints.at(i).solver_body_a;
            let body_b = constraints.at(i).solver_body_b;

            let first_constraint_id_a = if body_a != 0 {
                body_constraint_counts.read(body_a as usize - 1) as usize
            } else {
                0
            };
            let last_constraint_id_a = body_constraint_counts.read(body_a as usize) as usize;

            let first_constraint_id_b = if body_b != 0 {
                body_constraint_counts.read(body_b as usize - 1) as usize
            } else {
                0
            };
            let last_constraint_id_b = body_constraint_counts.read(body_b as usize) as usize;

            // Traverse all constraints from body A.
            for j in first_constraint_id_a..last_constraint_id_a {
                let constraint_j = body_constraint_ids.read(j);

                if constraint_j != constraint_i {
                    let color_j = constraints_colors.read(constraint_j as usize);
                    if color_j < 32 {
                        color_mask.0 |= 1u32 << color_j;
                    } else {
                        color_mask.1 |= 1u32 << (color_j - 32);
                    }
                }
            }

            // Traverse all constraints from body B.
            for j in first_constraint_id_b..last_constraint_id_b {
                let constraint_j = body_constraint_ids.read(j);

                if constraint_j != constraint_i {
                    let color_j = constraints_colors.read(constraint_j as usize);
                    if color_j < 32 {
                        color_mask.0 |= 1u32 << color_j;
                    } else {
                        color_mask.1 |= 1u32 << (color_j - 32);
                    }
                }
            }

            let my_color = (!color_mask.0).trailing_zeros() + (!color_mask.1).trailing_zeros();
            constraints_colors.write(i, my_color);
            colored.write(i, 1);
            // We are not finished coloring. 0 indicates the algorithm must continue.
            *num_colors = 0;
        }
    }
}

/// Fixes conflicts in Topo-GC coloring.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_fix_conflicts_topo_gc(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] body_constraint_counts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] body_constraint_ids: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] constraints: &[TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] constraints_colors: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] colored: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] num_colors: &mut u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] contacts_len: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 7)] contacts_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 8)] colliders_batch_capacity: &u32,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y as usize;
    let contacts_start = batch_id * *contacts_batch_capacity as usize;
    let colliders_start = batch_id * *colliders_batch_capacity as usize;
    let bci_start = batch_id * 2 * *contacts_batch_capacity as usize;

    let body_constraint_counts = Slice(body_constraint_counts, colliders_start);
    let body_constraint_ids = Slice(body_constraint_ids, bci_start);
    let constraints = Slice(constraints, contacts_start);
    let constraints_colors = Slice(constraints_colors, contacts_start);
    let mut colored = SliceMut(colored, contacts_start);

    let len = contacts_len.read(batch_id);

    for constraint_i in StepRng::new(invocation_id.x..len, num_threads) {
        let i = constraint_i as usize;
        let color_i = constraints_colors.read(i);

        // NOTE: this `num_colors` read doesn't need to be atomic. Any non-zero value is indicative of a finished
        //       algorithm.
        // If `num_colors > 0u` then we know that the coloring algorithm has converged. So we use this dispatch
        // as an opportunity to compute the colors count that will be ready back to the CPU side.
        if *num_colors > 0 {
            // TODO PERF: not sure if that would have a significant impact but we could keep track of
            //            whether the last iteration of the TOPO-GC algorithm already finished, in which
            //            case we can skip the atomic max entirely and just early-exist.

            atomic_max_u32(num_colors, color_i);
        } else {
            let body_a = constraints.at(i).solver_body_a;
            let body_b = constraints.at(i).solver_body_b;

            let first_constraint_id_a = if body_a != 0 {
                body_constraint_counts.read(body_a as usize - 1) as usize
            } else {
                0
            };
            let last_constraint_id_a = body_constraint_counts.read(body_a as usize) as usize;

            let first_constraint_id_b = if body_b != 0 {
                body_constraint_counts.read(body_b as usize - 1) as usize
            } else {
                0
            };
            let last_constraint_id_b = body_constraint_counts.read(body_b as usize) as usize;

            // Traverse all constraints from body A.
            for j in first_constraint_id_a..last_constraint_id_a {
                let constraint_j = body_constraint_ids.read(j);

                if constraint_j != constraint_i {
                    let color_j = constraints_colors.read(constraint_j as usize);
                    if color_i == color_j && constraint_i < constraint_j {
                        // Found a conflict, uncolor this node.
                        colored.write(i, 0);
                    }
                }
            }

            // Traverse all constraints from body B.
            for j in first_constraint_id_b..last_constraint_id_b {
                let constraint_j = body_constraint_ids.read(j);

                if constraint_j != constraint_i {
                    let color_j = constraints_colors.read(constraint_j as usize);
                    if color_i == color_j && constraint_i < constraint_j {
                        // Found a conflict, uncolor this node.
                        colored.write(i, 0);
                    }
                }
            }
        }
    }
}
