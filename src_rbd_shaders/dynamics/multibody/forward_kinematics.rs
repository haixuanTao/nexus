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

    // Root: a fixed root inherits its pose from the rigid-body pipeline (the
    // pose buffer is the source of truth). A free root reconstructs its pose
    // from the integrated joint coords / rotation and publishes it back to the
    // shared pose buffer for downstream consumers (FK of children, collision).
    {
        let stat = stat_slice.read(0);
        let root_pose = if mb.root_is_dynamic == 0 {
            poses_slice.read(stat.rb_id as usize)
        } else {
            // Pass the workspace fields by reference: rust-gpu's calling
            // convention copies arrays through Function storage when passed
            // by value, but `&[f32; N]` stays as a storage-buffer pointer.
            let ws_ref = ws_slice.at(0);
            let pose = body_to_parent(&stat, ws_ref.joint_rot, &ws_ref.coords);
            poses_slice.write(stat.rb_id as usize, pose);
            pose
        };
        let link_mut = ws_slice.at_mut(0);
        link_mut.local_to_parent = root_pose;
        link_mut.local_to_world = root_pose;
    }

    for k in 1..num_links {
        let k_usize = k as usize;
        let stat = stat_slice.at(k_usize);

        let local_to_parent;
        let parent_to_world;
        {
            let ws_ref = ws_slice.at(k_usize);
            let parent_ref = ws_slice.at(stat.parent_link_id as usize);
            parent_to_world = parent_ref.local_to_world;
            local_to_parent = body_to_parent(&stat, ws_ref.joint_rot, &ws_ref.coords);
        }
        let local_to_world = parent_to_world * local_to_parent;

        let parent_lmp = local_mprops_slice.read(stat.parent_link_id as usize);
        let lmp = local_mprops_slice.read(k_usize);
        let world_com = local_to_world * lmp.com; // c3 in Rapier
        let parent_com_world = parent_to_world * parent_lmp.com; // c2 in Rapier
        let child_anchor_world = local_to_world * stat.data.local_frame_b.translation; // c0 in Rapier
        let shift02 = child_anchor_world - parent_com_world;
        let shift23 = world_com - child_anchor_world;

        let link_mut = ws_slice.at_mut(k_usize);
        link_mut.local_to_parent = local_to_parent;
        link_mut.local_to_world = local_to_world;
        link_mut.shift02 = shift02;
        link_mut.shift23 = shift23;
        poses_slice.write(stat.rb_id as usize, local_to_world);
    }
}
