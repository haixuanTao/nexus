//! Mesh generation utilities for FEM simulation.
//!
//! Generates triangular (2D) or tetrahedral (3D) meshes from regular grids.

use crate::{Matrix, Vector, VERTS_PER_ELEM};

/// A single mesh element (triangle in 2D, tetrahedron in 3D).
#[derive(Clone)]
pub struct MeshElement {
    /// Vertex indices for this element.
    pub indices: [usize; VERTS_PER_ELEM],
    /// Inverse of the reference shape matrix.
    pub B_inv: Matrix,
    /// Rest area (2D) or volume (3D).
    pub vol: f32,
}

/// A finite element mesh.
#[derive(Clone)]
pub struct FemMesh {
    /// Vertex positions.
    pub positions: Vec<Vector>,
    /// Element connectivity with precomputed geometric data.
    pub elements: Vec<MeshElement>,
}

#[cfg(feature = "dim2")]
impl FemMesh {
    /// Generate a 2D triangular mesh from a regular grid in [lo, hi]^2.
    ///
    /// Each quad cell is subdivided into 2 triangles.
    pub fn generate_grid(res: [usize; 2], lo: Vector, hi: Vector) -> Self {
        let nx = res[0] + 1;
        let ny = res[1] + 1;
        let dx = (hi.x - lo.x) / res[0] as f32;
        let dy = (hi.y - lo.y) / res[1] as f32;

        let mut positions = Vec::with_capacity(nx * ny);
        for iy in 0..ny {
            for ix in 0..nx {
                positions.push(Vector::new(
                    lo.x + ix as f32 * dx,
                    lo.y + iy as f32 * dy,
                ));
            }
        }

        let vert = |ix: usize, iy: usize| -> usize { iy * nx + ix };

        let mut elements = Vec::new();
        for iy in 0..res[1] {
            for ix in 0..res[0] {
                let v00 = vert(ix, iy);
                let v10 = vert(ix + 1, iy);
                let v11 = vert(ix + 1, iy + 1);
                let v01 = vert(ix, iy + 1);

                // Two triangles per quad, alternating diagonal for conforming mesh.
                let parity = (ix + iy) % 2;
                let tris = if parity == 0 {
                    [[v00, v10, v01], [v10, v11, v01]]
                } else {
                    [[v00, v10, v11], [v00, v11, v01]]
                };

                for indices in &tris {
                    let p0 = positions[indices[0]];
                    let p1 = positions[indices[1]];
                    let p2 = positions[indices[2]];

                    let D_m = Matrix::from_cols(p1 - p0, p2 - p0);
                    let area = D_m.determinant().abs() / 2.0;
                    let B_inv = D_m.inverse();

                    elements.push(MeshElement {
                        indices: *indices,
                        B_inv,
                        vol: area,
                    });
                }
            }
        }

        Self {
            positions,
            elements,
        }
    }
}

#[cfg(feature = "dim3")]
impl FemMesh {
    /// Generate a 3D tetrahedral mesh from a regular hex grid in [lo, hi]^3.
    ///
    /// Each hex cell is subdivided into 5 tetrahedra with alternating diagonals
    /// to maintain a conforming mesh.
    pub fn generate_grid(res: [usize; 3], lo: Vector, hi: Vector) -> Self {
        let nx = res[0] + 1;
        let ny = res[1] + 1;
        let nz = res[2] + 1;
        let dx = (hi.x - lo.x) / res[0] as f32;
        let dy = (hi.y - lo.y) / res[1] as f32;
        let dz = (hi.z - lo.z) / res[2] as f32;

        let mut positions = Vec::with_capacity(nx * ny * nz);
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    positions.push(Vector::new(
                        lo.x + ix as f32 * dx,
                        lo.y + iy as f32 * dy,
                        lo.z + iz as f32 * dz,
                    ));
                }
            }
        }

        let vert = |ix: usize, iy: usize, iz: usize| -> usize { iz * ny * nx + iy * nx + ix };

        let mut elements = Vec::new();
        for iz in 0..res[2] {
            for iy in 0..res[1] {
                for ix in 0..res[0] {
                    let v = [
                        vert(ix, iy, iz),
                        vert(ix + 1, iy, iz),
                        vert(ix + 1, iy + 1, iz),
                        vert(ix, iy + 1, iz),
                        vert(ix, iy, iz + 1),
                        vert(ix + 1, iy, iz + 1),
                        vert(ix + 1, iy + 1, iz + 1),
                        vert(ix, iy + 1, iz + 1),
                    ];

                    // 5-tet decomposition with alternating diagonal for conforming mesh.
                    let parity = (ix + iy + iz) % 2;
                    let tet_indices = if parity == 0 {
                        [
                            [v[0], v[1], v[3], v[4]],
                            [v[1], v[2], v[3], v[6]],
                            [v[1], v[4], v[5], v[6]],
                            [v[3], v[4], v[6], v[7]],
                            [v[1], v[3], v[4], v[6]],
                        ]
                    } else {
                        [
                            [v[0], v[1], v[2], v[5]],
                            [v[0], v[2], v[3], v[7]],
                            [v[0], v[4], v[5], v[7]],
                            [v[2], v[5], v[6], v[7]],
                            [v[0], v[2], v[5], v[7]],
                        ]
                    };

                    for indices in &tet_indices {
                        let p0 = positions[indices[0]];
                        let p1 = positions[indices[1]];
                        let p2 = positions[indices[2]];
                        let p3 = positions[indices[3]];

                        let D_m = Matrix::from_cols(p1 - p0, p2 - p0, p3 - p0);
                        let vol = D_m.determinant().abs() / 6.0;
                        let B_inv = D_m.inverse();

                        elements.push(MeshElement {
                            indices: *indices,
                            B_inv,
                            vol,
                        });
                    }
                }
            }
        }

        Self {
            positions,
            elements,
        }
    }
}
