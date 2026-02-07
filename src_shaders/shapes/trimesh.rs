//! Triangle Mesh Shape Module
//!
//! This module provides geometric operations for triangle meshes.
//! A trimesh uses a BVH for efficient queries.
//!
//! The mesh's vertex and index buffer are organized so that the vertex
//! buffer contains the BVH first, and then the triangle vertices.
//! Similarly, its index buffer contains the BVH topology information first, and then
//! the triangle indices.
//!
//! The BVH topology follows https://docs.rs/bvh/0.12.0/bvh/flat_bvh/struct.FlatNode.html
//! So each BVH node implies 3 entries in the index buffer, and 2 entries in the vertex buffer.

use crate::bounding_volumes::Aabb;
use crate::shapes::triangle::Triangle;
use crate::{MaybeIndexUnchecked, VectorWithPadding};

/// A triangle mesh with BVH acceleration structure.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct TriMesh {
    /// Index of the root AABB in the vertex buffer.
    pub bvh_vtx_root_id: u32,
    /// The root AABB left-child index in the index buffer.
    pub bvh_idx_root_id: u32,
    /// The number of BVH nodes. Triangle indices are stored after the last BVH node.
    pub bvh_node_len: u32,
    /// Root AABB of the mesh.
    pub root_aabb: Aabb,
}

/// BVH node indices for tree traversal.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct BvhIdx {
    /// Index to enter (left child). If 0xffffffff, this is a leaf node.
    pub entry_index: u32,
    /// Index to exit (skip to sibling or parent's sibling).
    pub exit_index: u32,
    /// Index of the shape (triangle) in the mesh. Only valid for leaf nodes.
    pub shape_index: u32,
}

impl TriMesh {
    /// Creates a new triangle mesh.
    #[inline]
    pub fn new(
        bvh_vtx_root_id: u32,
        bvh_idx_root_id: u32,
        bvh_node_len: u32,
        root_aabb: Aabb,
    ) -> Self {
        Self {
            bvh_vtx_root_id,
            bvh_idx_root_id,
            bvh_node_len,
            root_aabb,
        }
    }

    /// Computes the AABB of a trimesh.
    pub fn aabb(&self) -> Aabb {
        self.root_aabb
    }

    /// Gets the AABB of a BVH node.
    #[inline]
    pub fn bvh_node_aabb(&self, node_id: u32, vertices: &[VectorWithPadding]) -> Aabb {
        // Multiply by 2 since there are two values per AABB (min/max).
        let vid = (self.bvh_vtx_root_id + node_id * 2) as usize;
        Aabb::new(*vertices.read(vid), *vertices.read(vid + 1))
    }

    /// Gets the BVH node indices for tree traversal.
    #[inline]
    pub fn bvh_node_idx(&self, node_id: u32, indices: &[u32]) -> BvhIdx {
        let base_id = (self.bvh_idx_root_id + node_id * 3) as usize;
        BvhIdx {
            entry_index: indices.read(base_id),
            exit_index: indices.read(base_id + 1),
            shape_index: indices.read(base_id + 2),
        }
    }

    /// Gets a triangle from the mesh by its index.
    #[inline]
    pub fn triangle(
        &self,
        tri_id: u32,
        vertices: &[VectorWithPadding],
        indices: &[u32],
    ) -> Triangle {
        let base_id = (self.bvh_idx_root_id + self.bvh_node_len * 3 + tri_id * 3) as usize;
        let base_vid = (self.bvh_vtx_root_id + self.bvh_node_len * 2) as usize;
        let a = *vertices.read(base_vid + indices.read(base_id) as usize);
        let b = *vertices.read(base_vid + indices.read(base_id + 1) as usize);
        let c = *vertices.read(base_vid + indices.read(base_id + 2) as usize);
        Triangle::new(a, b, c)
    }
}
