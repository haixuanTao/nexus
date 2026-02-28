//! FEM simulation pipeline: GPU buffer management and kernel dispatch.
//!
//! `FemPipeline` holds compiled GPU shaders, `FemData` holds all GPU state.
//! Call `FemPipeline::launch_step()` per substep to advance the simulation.

#![allow(non_snake_case)]

use crate::fem_shaders::kernels::explicit::AtomicForce;
use crate::fem_shaders::kernels::implicit::AtomicDiag;
use crate::fem_shaders::{pad_mat, pad_vec, unpad_vec};
use crate::mesh::FemMesh;
use crate::solver::kernels::{WgExplicitStep, WgImplicitStep};
use crate::solver::{FemConfig, FemMaterial, SolverMethod};
use crate::types::{
    ElementEnergyGrad, ElementHessian, ElementInfo, ElementPrecomputed, FemSimParams,
    LinesearchScalars, PcgScalars, PcgVertexState, VertexConstraint, VertexInfo, VertexState,
};
use crate::{Matrix, PaddedVector, Vector, VERTS_PER_ELEM};
use khal::backend::{Backend, Encoder, GpuBackend, GpuBackendError, GpuEncoder, GpuTimestamps};
use khal::{BufferUsages, Shader};
use vortx::tensor::Tensor;

// ── Entity Handle ──

/// Identifies a mesh entity's vertex/element ranges within the solver.
#[derive(Clone, Debug)]
pub struct EntityHandle {
    pub vertex_offset: usize,
    pub vertex_count: usize,
    pub element_offset: usize,
    pub element_count: usize,
}

// ── Pipeline ──

/// GPU compute pipeline for FEM simulation.
///
/// Holds compiled SPIR-V compute shaders for both explicit (symplectic Euler)
/// and implicit (Newton-PCG + line search) solvers.
pub struct FemPipeline {
    explicit: WgExplicitStep,
    implicit: WgImplicitStep,
}

impl FemPipeline {
    pub fn new(backend: &GpuBackend) -> Result<Self, GpuBackendError> {
        Ok(Self {
            explicit: WgExplicitStep::from_backend(backend)?,
            implicit: WgImplicitStep::from_backend(backend)?,
        })
    }

    /// Dispatch one substep of the FEM simulation.
    pub fn launch_step(
        &self,
        gpu: &mut GpuBackend,
        data: &mut FemData,
        timestamps: Option<&mut GpuTimestamps>,
    ) -> Result<(), GpuBackendError> {
        match data.method {
            SolverMethod::Explicit => {
                self.launch_explicit_step(gpu, data, timestamps)
            }
            SolverMethod::Implicit => {
                self.launch_implicit_step(gpu, data, timestamps)
            }
        }
    }

    fn launch_explicit_step(
        &self,
        gpu: &mut GpuBackend,
        data: &mut FemData,
        mut timestamps: Option<&mut GpuTimestamps>,
    ) -> Result<(), GpuBackendError> {
        let nv = data.num_vertices;
        let ne = data.num_elements;
        let mut encoder = gpu.begin_encoding();

        // 1. Compute elastic forces (per-element → scatter to vertices)
        {
            let mut pass = encoder.begin_pass("FEM: elastic forces", timestamps.as_deref_mut());
            self.explicit.compute_elastic_forces.call(
                &mut pass,
                ne,
                &data.sim_params,
                &data.elem_info,
                &data.vertex_state,
                &mut data.force_atomic,
            )?;
        }

        // 2. Apply accumulated dv, gravity, damping (per-vertex)
        {
            let mut pass =
                encoder.begin_pass("FEM: dv+gravity+damping", timestamps.as_deref_mut());
            self.explicit.apply_forces_gravity_damping.call(
                &mut pass,
                nv,
                &data.sim_params,
                &mut data.vertex_state,
                &mut data.force_atomic,
            )?;
        }

        // 3. Apply soft constraints (per-vertex)
        {
            let mut pass =
                encoder.begin_pass("FEM: soft constraints", timestamps.as_deref_mut());
            self.explicit.apply_soft_constraints.call(
                &mut pass,
                nv,
                &data.sim_params,
                &data.vertex_info,
                &mut data.vertex_state,
                &data.constraints,
            )?;
        }

        // 4. Integrate positions: x += dt * v (per-vertex)
        {
            let mut pass = encoder.begin_pass("FEM: integrate", timestamps.as_deref_mut());
            self.explicit.integrate_positions.call(
                &mut pass,
                nv,
                &data.sim_params,
                &mut data.vertex_state,
            )?;
        }

        // 5. Apply hard constraints (per-vertex)
        {
            let mut pass =
                encoder.begin_pass("FEM: hard constraints", timestamps.as_deref_mut());
            self.explicit.apply_hard_constraints.call(
                &mut pass,
                nv,
                &data.sim_params,
                &mut data.vertex_state,
                &data.constraints,
            )?;
        }

        // 6. Boundary conditions — floor collision (per-vertex)
        {
            let mut pass = encoder.begin_pass("FEM: boundary", timestamps.as_deref_mut());
            self.explicit.boundary_conditions.call(
                &mut pass,
                nv,
                &data.sim_params,
                &mut data.vertex_state,
            )?;
        }

        gpu.submit(encoder)?;

        Ok(())
    }

    fn launch_implicit_step(
        &self,
        gpu: &mut GpuBackend,
        data: &mut FemData,
        mut timestamps: Option<&mut GpuTimestamps>,
    ) -> Result<(), GpuBackendError> {
        let nv = data.num_vertices;
        let ne = data.num_elements;

        // ── Init: save x_prev, compute inertia target y ──
        let mut encoder = gpu.begin_encoding();
        {
            let mut pass =
                encoder.begin_pass("FEM: init implicit", timestamps.as_deref_mut());
            self.implicit.init_implicit_step.call(
                &mut pass,
                nv,
                &data.sim_params,
                &mut data.vertex_state,
                &data.constraints,
                &mut data.pcg_vertex,
            )?;
        }
        gpu.submit(encoder)?;

        // ── Newton iterations ──
        for _newton in 0..data.newton_iters {
            let mut encoder = gpu.begin_encoding();

            // Precompute material (SVD for corotated)
            {
                let mut pass =
                    encoder.begin_pass("FEM: precompute material", timestamps.as_deref_mut());
                self.implicit.precompute_material.call(
                    &mut pass,
                    ne,
                    &data.sim_params,
                    &data.elem_info,
                    &data.vertex_state,
                    &mut data.elem_precomp,
                )?;
            }

            // Compute energy, gradient, Hessian (per-element)
            {
                let mut pass = encoder.begin_pass("FEM: EGH", timestamps.as_deref_mut());
                self.implicit.compute_egh.call(
                    &mut pass,
                    ne,
                    &data.sim_params,
                    &data.elem_info,
                    &data.vertex_state,
                    &data.elem_precomp,
                    &mut data.elem_eg,
                    &mut data.elem_hessian,
                )?;
            }

            // Scatter elastic force + diagonal Hessian (per-element → vertices)
            {
                let mut pass =
                    encoder.begin_pass("FEM: scatter force+diag", timestamps.as_deref_mut());
                self.implicit.scatter_elastic_force_diag.call(
                    &mut pass,
                    ne,
                    &data.sim_params,
                    &data.elem_info,
                    &data.elem_eg,
                    &data.elem_hessian,
                    &mut data.force_atomic,
                    &mut data.diag_atomic,
                )?;
            }

            // Assemble total force, compute preconditioner, init PCG
            {
                let mut pass =
                    encoder.begin_pass("FEM: assemble+PCG init", timestamps.as_deref_mut());
                self.implicit.assemble_and_pcg_init.call(
                    &mut pass,
                    nv,
                    &data.sim_params,
                    &data.vertex_info,
                    &data.vertex_state,
                    &data.constraints,
                    &mut data.pcg_vertex,
                    &mut data.force_atomic,
                    &mut data.diag_atomic,
                    &mut data.scalar_atomic,
                )?;
            }

            // Finalize initial rTz
            {
                let mut pass =
                    encoder.begin_pass("FEM: PCG reduce init", timestamps.as_deref_mut());
                self.implicit.pcg_reduce_init.call(
                    &mut pass,
                    1u32,
                    &mut data.pcg_scalars,
                    &mut data.scalar_atomic,
                )?;
            }
            gpu.submit(encoder)?;

            // ── PCG iterations ──
            for _pcg in 0..data.pcg_iters {
                let mut encoder = gpu.begin_encoding();
                // Compute A*p via element scatter
                {
                    let mut pass =
                        encoder.begin_pass("FEM: PCG Ap", timestamps.as_deref_mut());
                    self.implicit.pcg_scatter_ap.call(
                        &mut pass,
                        ne,
                        &data.sim_params,
                        &data.elem_info,
                        &data.elem_hessian,
                        &data.pcg_vertex,
                        &mut data.ap_atomic,
                    )?;
                }

                // Finalize Ap + accumulate p·Ap
                {
                    let mut pass = encoder.begin_pass(
                        "FEM: PCG finalize Ap",
                        timestamps.as_deref_mut(),
                    );
                    self.implicit.pcg_finalize_ap_dot.call(
                        &mut pass,
                        nv,
                        &data.sim_params,
                        &data.vertex_info,
                        &data.constraints,
                        &mut data.pcg_vertex,
                        &mut data.ap_atomic,
                        &mut data.scalar_atomic,
                    )?;
                }

                // α = rTz / pTAp
                {
                    let mut pass =
                        encoder.begin_pass("FEM: PCG alpha", timestamps.as_deref_mut());
                    self.implicit.pcg_compute_alpha.call(
                        &mut pass,
                        1u32,
                        &mut data.pcg_scalars,
                        &mut data.scalar_atomic,
                    )?;
                }

                // x += α*p, r -= α*Ap, z = M⁻¹r
                {
                    let mut pass =
                        encoder.begin_pass("FEM: PCG update xrz", timestamps.as_deref_mut());
                    self.implicit.pcg_update_x_r_z.call(
                        &mut pass,
                        nv,
                        &data.sim_params,
                        &data.pcg_scalars,
                        &mut data.pcg_vertex,
                        &mut data.scalar_atomic,
                    )?;
                }

                // β = rTz_new / rTz
                {
                    let mut pass =
                        encoder.begin_pass("FEM: PCG beta", timestamps.as_deref_mut());
                    self.implicit.pcg_compute_beta.call(
                        &mut pass,
                        1u32,
                        &mut data.pcg_scalars,
                        &mut data.scalar_atomic,
                    )?;
                }

                // p = z + β*p
                {
                    let mut pass =
                        encoder.begin_pass("FEM: PCG update p", timestamps.as_deref_mut());
                    self.implicit.pcg_update_p.call(
                        &mut pass,
                        nv,
                        &data.sim_params,
                        &data.pcg_scalars,
                        &mut data.pcg_vertex,
                    )?;
                }
                gpu.submit(encoder)?;
            }

            // ── Line search ──

            let mut encoder = gpu.begin_encoding();
            // Init: save position, compute directional derivative m, vertex energy
            {
                let mut pass =
                    encoder.begin_pass("FEM: LS init", timestamps.as_deref_mut());
                self.implicit.ls_init.call(
                    &mut pass,
                    nv,
                    &data.sim_params,
                    &data.vertex_info,
                    &data.vertex_state,
                    &data.constraints,
                    &data.pcg_vertex,
                    &mut data.ls_prev_pos,
                    &mut data.scalar_atomic,
                )?;
            }

            // Element energy at initial position
            {
                let mut pass =
                    encoder.begin_pass("FEM: LS energy elem 0", timestamps.as_deref_mut());
                self.implicit.ls_energy_element.call(
                    &mut pass,
                    ne,
                    &data.sim_params,
                    &data.elem_info,
                    &data.vertex_state,
                    &data.elem_precomp,
                    &mut data.scalar_atomic,
                )?;
            }

            // Finalize: set prev_energy, step_size = 1
            {
                let mut pass =
                    encoder.begin_pass("FEM: LS finalize init", timestamps.as_deref_mut());
                self.implicit.ls_finalize_init.call(
                    &mut pass,
                    1u32,
                    &mut data.ls_scalars,
                    &mut data.scalar_atomic,
                )?;
            }
            gpu.submit(encoder)?;

            // Backtracking iterations
            for _ls in 0..data.ls_max_iters {
                let mut encoder = gpu.begin_encoding();
                // Trial position: x = prev + step * dx
                {
                    let mut pass =
                        encoder.begin_pass("FEM: LS update pos", timestamps.as_deref_mut());
                    self.implicit.ls_update_pos.call(
                        &mut pass,
                        nv,
                        &data.sim_params,
                        &mut data.vertex_state,
                        &data.constraints,
                        &data.pcg_vertex,
                        &data.ls_prev_pos,
                        &data.ls_scalars,
                    )?;
                }

                // Vertex energy at trial position
                {
                    let mut pass =
                        encoder.begin_pass("FEM: LS energy vtx", timestamps.as_deref_mut());
                    self.implicit.ls_energy_vertex.call(
                        &mut pass,
                        nv,
                        &data.sim_params,
                        &data.vertex_info,
                        &data.vertex_state,
                        &data.constraints,
                        &data.pcg_vertex,
                        &mut data.scalar_atomic,
                        &data.ls_scalars,
                    )?;
                }

                // Element energy at trial position
                {
                    let mut pass = encoder
                        .begin_pass("FEM: LS energy elem", timestamps.as_deref_mut());
                    self.implicit.ls_energy_element.call(
                        &mut pass,
                        ne,
                        &data.sim_params,
                        &data.elem_info,
                        &data.vertex_state,
                        &data.elem_precomp,
                        &mut data.scalar_atomic,
                    )?;
                }

                // Check Armijo condition, reduce step if needed
                {
                    let mut pass =
                        encoder.begin_pass("FEM: LS check", timestamps.as_deref_mut());
                    self.implicit.ls_check_armijo.call(
                        &mut pass,
                        1u32,
                        &mut data.ls_scalars,
                        &mut data.scalar_atomic,
                    )?;
                }

                gpu.submit(encoder)?;
            }
        }

        // ── Finalization ──
        let mut encoder = gpu.begin_encoding();
        // Compute velocity: v = (x - x_prev) / dt
        {
            let mut pass =
                encoder.begin_pass("FEM: compute velocity", timestamps.as_deref_mut());
            self.implicit.compute_velocity.call(
                &mut pass,
                nv,
                &data.sim_params,
                &mut data.vertex_state,
                &data.pcg_vertex,
            )?;
        }

        // Boundary conditions
        {
            let mut pass = encoder.begin_pass("FEM: boundary", timestamps.as_deref_mut());
            self.implicit.boundary_conditions.call(
                &mut pass,
                nv,
                &data.sim_params,
                &mut data.vertex_state,
            )?;
        }
        gpu.submit(encoder)?;

        Ok(())
    }
}

// ── FemData ──

/// GPU state for a FEM simulation.
///
/// Holds all GPU buffers (vertex state, element data, solver scratch space)
/// and simulation configuration.
pub struct FemData {
    // Configuration
    pub method: SolverMethod,
    pub num_substeps: u32,
    pub newton_iters: u32,
    pub pcg_iters: u32,
    pub ls_max_iters: u32,
    pub num_vertices: u32,
    pub num_elements: u32,
    pub entities: Vec<EntityHandle>,

    // Core GPU buffers
    pub sim_params: Tensor<FemSimParams>,
    pub vertex_state: Tensor<VertexState>,
    pub vertex_info: Tensor<VertexInfo>,
    pub constraints: Tensor<VertexConstraint>,
    pub elem_info: Tensor<ElementInfo>,
    pub force_atomic: Tensor<AtomicForce>,

    // Implicit solver buffers
    pub elem_precomp: Tensor<ElementPrecomputed>,
    pub elem_eg: Tensor<ElementEnergyGrad>,
    pub elem_hessian: Tensor<ElementHessian>,
    pub pcg_vertex: Tensor<PcgVertexState>,
    pub pcg_scalars: Tensor<PcgScalars>,
    pub diag_atomic: Tensor<AtomicDiag>,
    pub ap_atomic: Tensor<AtomicForce>,
    pub scalar_atomic: Tensor<u32>,
    pub ls_prev_pos: Tensor<PaddedVector>,
    pub ls_scalars: Tensor<LinesearchScalars>,

    // Readback
    pub vertex_staging: Tensor<VertexState>,
}

/// Compute S matrix rows from B_inv.
///
/// S[k] maps vertex k's position to the deformation gradient columns.
/// Convention: vertex 0 is the base vertex.
/// S[1..] = rows of B_inv, S[0] = -sum(S[1..]).
fn compute_S(B_inv: Matrix) -> [Vector; VERTS_PER_ELEM] {
    #[cfg(feature = "dim3")]
    {
        // Row i of B_inv = S[i+1]
        let s1 = Vector::new(B_inv.x_axis.x, B_inv.y_axis.x, B_inv.z_axis.x);
        let s2 = Vector::new(B_inv.x_axis.y, B_inv.y_axis.y, B_inv.z_axis.y);
        let s3 = Vector::new(B_inv.x_axis.z, B_inv.y_axis.z, B_inv.z_axis.z);
        let s0 = -(s1 + s2 + s3);
        [s0, s1, s2, s3]
    }
    #[cfg(feature = "dim2")]
    {
        let s1 = Vector::new(B_inv.x_axis.x, B_inv.y_axis.x);
        let s2 = Vector::new(B_inv.x_axis.y, B_inv.y_axis.y);
        let s0 = -(s1 + s2);
        [s0, s1, s2]
    }
}

impl FemData {
    /// Create FEM simulation data from meshes and materials.
    ///
    /// Each `(FemMesh, FemMaterial)` pair is an entity. Vertices and elements
    /// are merged into flat arrays. Returns entity handles via `self.entities`.
    pub fn new(
        backend: &GpuBackend,
        meshes: &[(FemMesh, FemMaterial)],
        config: &FemConfig,
    ) -> Result<Self, GpuBackendError> {
        let dt = config.dt / config.substeps as f32;

        // ── Merge all entities ──
        let mut positions: Vec<Vector> = Vec::new();
        let mut masses: Vec<f32> = Vec::new();
        let mut elem_infos: Vec<ElementInfo> = Vec::new();
        let mut entities: Vec<EntityHandle> = Vec::new();

        for (mesh, material) in meshes {
            let v_offset = positions.len();
            let e_offset = elem_infos.len();

            positions.extend_from_slice(&mesh.positions);
            masses.resize(positions.len(), 0.0);

            let mu = material.mu();
            let lam = material.lambda();
            let model_id = material.model.to_gpu_id();
            let rho = material.density;

            for elem in &mesh.elements {
                // Remap indices with vertex offset.
                let mut indices = [0u32; 4];
                for i in 0..VERTS_PER_ELEM {
                    indices[i] = (elem.indices[i] + v_offset) as u32;
                }

                // S matrix rows from B_inv.
                let S_vecs = compute_S(elem.B_inv);
                let mut S_padded = [pad_vec(Vector::ZERO); 4];
                for i in 0..VERTS_PER_ELEM {
                    S_padded[i] = pad_vec(S_vecs[i]);
                }

                // Mass distribution: equal share per vertex.
                let mass_per_vertex = elem.vol * rho / VERTS_PER_ELEM as f32;
                for i in 0..VERTS_PER_ELEM {
                    masses[indices[i] as usize] += mass_per_vertex;
                }

                elem_infos.push(ElementInfo {
                    indices,
                    B_inv: pad_mat(elem.B_inv),
                    S: S_padded,
                    vol: elem.vol,
                    mu,
                    lam,
                    model: model_id,
                    rho,
                    _pad0: 0.0,
                    _pad1: 0.0,
                    _pad2: 0.0,
                });
            }

            entities.push(EntityHandle {
                vertex_offset: v_offset,
                vertex_count: mesh.positions.len(),
                element_offset: e_offset,
                element_count: mesh.elements.len(),
            });
        }

        let num_vertices = positions.len() as u32;
        let num_elements = elem_infos.len() as u32;

        // ── Build per-vertex GPU data ──
        let vertex_states: Vec<VertexState> = positions
            .iter()
            .map(|&p| VertexState {
                pos: pad_vec(p),
                vel: pad_vec(Vector::ZERO),
            })
            .collect();

        let vertex_infos: Vec<VertexInfo> = masses
            .iter()
            .map(|&mass| VertexInfo {
                mass,
                mass_over_dt2: mass / (dt * dt),
                _pad0: 0.0,
                _pad1: 0.0,
            })
            .collect();

        let constraint_data: Vec<VertexConstraint> =
            vec![VertexConstraint::default(); num_vertices as usize];

        // ── Simulation parameters ──
        let params = FemSimParams {
            dt,
            damping: config.damping,
            alpha_rayleigh: config.alpha_rayleigh,
            beta_rayleigh: config.beta_rayleigh,
            gravity: pad_vec(config.gravity),
            floor_y: config.floor_y,
            num_vertices,
            num_elements,
            _pad: 0,
        };

        // ── Create GPU tensors ──
        let storage = BufferUsages::STORAGE;

        let sim_params = Tensor::scalar(
            backend,
            params,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        )?;
        let vertex_state = Tensor::vector(
            backend,
            &vertex_states,
            storage | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        )?;
        let vertex_info = Tensor::vector(backend, &vertex_infos, storage)?;
        let constraints = Tensor::vector(
            backend,
            &constraint_data,
            storage | BufferUsages::COPY_DST,
        )?;
        let elem_info = Tensor::vector(backend, &elem_infos, storage)?;

        // Atomic accumulators (zeroed).
        let force_atomic = Tensor::vector(
            backend,
            &vec![AtomicForce::default(); num_vertices as usize],
            storage,
        )?;

        // Implicit solver buffers.
        let elem_precomp = Tensor::vector(
            backend,
            &vec![ElementPrecomputed::default(); num_elements as usize],
            storage,
        )?;
        let elem_eg = Tensor::vector_uninit(backend, num_elements, storage)?;
        let elem_hessian = Tensor::vector_uninit(backend, num_elements, storage)?;
        let pcg_vertex = Tensor::vector(
            backend,
            &vec![PcgVertexState::default(); num_vertices as usize],
            storage,
        )?;
        let pcg_scalars = Tensor::scalar(backend, PcgScalars::default(), storage)?;
        let diag_atomic = Tensor::vector(
            backend,
            &vec![AtomicDiag::default(); num_vertices as usize],
            storage,
        )?;
        let ap_atomic = Tensor::vector(
            backend,
            &vec![AtomicForce::default(); num_vertices as usize],
            storage,
        )?;
        let scalar_atomic = Tensor::vector(backend, &vec![0u32; 8], storage)?;
        let ls_prev_pos = Tensor::vector_uninit(backend, num_vertices, storage)?;
        let ls_scalars = Tensor::scalar(backend, LinesearchScalars::default(), storage)?;

        // Readback staging.
        let vertex_staging = Tensor::vector_uninit(
            backend,
            num_vertices,
            BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        )?;

        Ok(Self {
            method: config.method,
            num_substeps: config.substeps,
            newton_iters: config.newton_iters,
            pcg_iters: config.pcg_iters,
            ls_max_iters: config.ls_max_iters,
            num_vertices,
            num_elements,
            entities,
            sim_params,
            vertex_state,
            vertex_info,
            constraints,
            elem_info,
            force_atomic,
            elem_precomp,
            elem_eg,
            elem_hessian,
            pcg_vertex,
            pcg_scalars,
            diag_atomic,
            ap_atomic,
            scalar_atomic,
            ls_prev_pos,
            ls_scalars,
            vertex_staging,
        })
    }

    /// Copy vertex state to staging buffer for CPU readback.
    ///
    /// Call this after encoding the step, before submitting the encoder.
    pub fn launch_readback(&mut self, encoder: &mut GpuEncoder) -> Result<(), GpuBackendError> {
        self.vertex_staging
            .copy_from_view(encoder, &self.vertex_state)
    }

    /// Read vertex positions from the staging buffer.
    ///
    /// Call after `backend.submit(encoder)` and `backend.synchronize()`.
    pub async fn read_vertex_states(
        &self,
        backend: &GpuBackend,
    ) -> Result<Vec<VertexState>, GpuBackendError> {
        let mut data = vec![VertexState::default(); self.num_vertices as usize];
        backend
            .read_buffer(self.vertex_staging.buffer(), &mut data)
            .await?;
        Ok(data)
    }

    /// Read vertex positions as Vector (unpacked from padded GPU layout).
    pub async fn read_positions(
        &self,
        backend: &GpuBackend,
    ) -> Result<Vec<Vector>, GpuBackendError> {
        let states = self.read_vertex_states(backend).await?;
        Ok(states.iter().map(|s| unpad_vec(s.pos)).collect())
    }
}
