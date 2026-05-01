//! Forward kinematics: per-multibody walk that produces each link's
//! `local_to_parent` / `local_to_world` poses and the `shift02` / `shift23`
//! offsets that the jacobian / Coriolis kernels read.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

use crate::Pose;
use crate::dynamics::body::LocalMassProperties;
use crate::utils::{Slice, SliceMut};

use super::types::{MultibodyInfo, MultibodyLinkStatic, MultibodyLinkWorkspace};
use super::utils::body_to_parent;

/// Forward kinematics: one workgroup per multibody, links walked sequentially.
///
/// Writes `local_to_parent`, `local_to_world`, `shift02`, `shift23` into the workspace,
/// and publishes the link's world pose to the shared `poses` buffer for downstream
/// consumption (e.g. mprops update, collision).
#[spirv_bindgen]
// TODO(PERF): if we restricted all batches to have the same multibody topologies,
//             we could have multiple threads per workgroup working on these multibodies?
//             compute(threads(1, 64, 1)) ?
#[spirv(compute(threads(1)))]
pub fn gpu_mb_forward_kinematics(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    links_workspace: &mut [MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] poses: &mut [Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] links_local_mprops: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] num_multibodies: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 6)] multibodies_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 7)] links_batch_capacity: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 8)] colliders_batch_capacity: &u32,
) {
    let batch_id = invocation_id.y as usize;
    let mb_idx_in_batch = invocation_id.x;
    let num_mb = num_multibodies.read(batch_id);
    if mb_idx_in_batch >= num_mb {
        return;
    }

    let mb_start = batch_id * *multibodies_batch_capacity as usize;
    let links_start = batch_id * *links_batch_capacity as usize;
    let coll_start = batch_id * *colliders_batch_capacity as usize;

    let mb = multibody_info.read(mb_start + mb_idx_in_batch as usize);
    let num_links = mb.num_links;
    let first_link_global = links_start + mb.first_link as usize;

    let stat_slice = Slice(links_static, first_link_global);
    let mut ws_slice = SliceMut(links_workspace, first_link_global);
    let local_mprops_slice = Slice(links_local_mprops, first_link_global);
    let mut poses_slice = SliceMut(poses, coll_start);

    // Special case for the root, which has no parent.
    {
        let root_pose = poses_slice.read(stat_slice.at(0).rb_id as usize);
        let link = ws_slice.at_mut(0);
        link.local_to_parent = root_pose;
        link.local_to_world = root_pose;
    }

    for k in 1..num_links {
        let k_usize = k as usize;
        let stat = stat_slice.at(k_usize);
        // TODO(PERF): avoid this copy?
        let mut link = ws_slice.read(k_usize);
        let parent_link = ws_slice.at(stat.parent_link_id as usize);

        let local_to_parent = body_to_parent(&stat, &link);
        let local_to_world = parent_link.local_to_world * local_to_parent;
        link.local_to_parent = local_to_parent;
        link.local_to_world = local_to_world;

        let parent_lmp = local_mprops_slice.read(stat.parent_link_id as usize);
        let lmp = local_mprops_slice.read(k_usize);
        let world_com = local_to_world * lmp.com; // c3 in Rapier
        let parent_com_world = parent_link.local_to_world * parent_lmp.com; // c2 in Rapier
        let child_anchor_world = local_to_world * stat.data.local_frame_b.translation; // c0 in Rapier
        link.shift02 = child_anchor_world - parent_com_world;
        link.shift23 = world_com - child_anchor_world;

        ws_slice.write(k_usize, link);
        poses_slice.write(stat.rb_id as usize, local_to_world);
    }
}
