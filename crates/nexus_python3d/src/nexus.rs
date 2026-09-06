//! Core simulation objects: `NexusState`, `NexusPipeline`, `GpuTimestamps`,
//! and the various entity handles.

use crate::loaders::{MjcfSceneInfo, UrdfLoaderOptions, UrdfRobotHandles};
use crate::math::{Pose, Vec3};
use crate::rbd::{
    Collider, ImpulseJointHandle, JointArg, JointAxis, MultibodyJointHandle, RigidBody,
    RigidBodyHandle, SharedShape,
};
use crate::viewer::NexusViewer;
use khal::backend::GpuTimestamps as RGpuTimestamps;
use nexus3d::prelude::{
    NexusPipeline as RNexusPipeline, NexusPipelineMask, NexusState as RNexusState,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use numpy::{PyArray2, PyArrayMethods};
use khal::backend::Backend as _;
use rapier3d::prelude as rp;

/// Maps a GPU backend error to a Python exception.
fn gpu_err<E: std::fmt::Debug>(e: E) -> PyErr {
    PyRuntimeError::new_err(format!("{e:?}"))
}

/// Entity counts for a `NexusState` (mirrors `NexusCounts`).
#[pyclass(name = "NexusCounts", from_py_object)]
#[derive(Clone, Copy)]
pub struct NexusCounts {
    #[pyo3(get)]
    pub num_environments: usize,
    #[pyo3(get)]
    pub rigid_bodies: usize,
    #[pyo3(get)]
    pub colliders: usize,
    #[pyo3(get)]
    pub impulse_joints: usize,
    #[pyo3(get)]
    pub multibodies: usize,
    #[pyo3(get)]
    pub multibody_dofs: usize,
}

/// Optional GPU timing-query buffer (`khal::backend::GpuTimestamps`).
#[pyclass(name = "GpuTimestamps", unsendable)]
pub struct GpuTimestamps(pub RGpuTimestamps);

#[pymethods]
impl GpuTimestamps {
    #[new]
    fn new(viewer: PyRef<NexusViewer>, capacity: u32) -> Self {
        GpuTimestamps(RGpuTimestamps::new(viewer.backend(), capacity))
    }

    /// Viewerless constructor for headless evaluation.
    #[staticmethod]
    fn headless(backend: PyRef<NexusBackend>, capacity: u32) -> Self {
        GpuTimestamps(RGpuTimestamps::new(&backend.0, capacity))
    }
}

/// dim3 SoA workspace layout (quad offsets per link). Mirrors
/// `src_rbd_shaders/dynamics/multibody/ws_soa.rs::layout`; the glob re-export
/// is not visible from this crate. `links_workspace_cuda()` checks WS_QUADS
/// against the real buffer length, and the field offsets are validated by the
/// Python test against the pose-derived joint angle.
mod ws_layout {
    pub const WS_JOINT_ROT: u32 = 0;
    pub const WS_COORDS: u32 = 1;
    pub const WS_LTP: u32 = 3;
    pub const WS_LTW: u32 = 5;
    pub const WS_JOINT_VEL: u32 = 9;
    pub const WS_RB_VELS: u32 = 11;
    pub const WS_KIN_ACC: u32 = 13;
    pub const WS_QUADS: u32 = 15;
}

/// A `__cuda_array_interface__` (v3) view over a Nexus GPU buffer.
///
/// Zero-copy: `torch.as_tensor(view, device="cuda")` aliases the simulator's
/// own memory, no staging buffer and no host round-trip. The view is valid
/// while the owning `NexusState` lives and is not re-finalized. Call
/// `NexusBackend.synchronize()` after a step before reading from it.
#[pyclass(name = "CudaArray", frozen)]
pub struct CudaArray {
    ptr: u64,
    byte_len: u64,
    shape: Vec<usize>,
    typestr: &'static str,
}

#[pymethods]
impl CudaArray {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("shape", PyTuple::new(py, &self.shape)?)?;
        d.set_item("typestr", self.typestr)?;
        d.set_item("data", (self.ptr, false))?;
        d.set_item("version", 3)?;
        d.set_item("strides", py.None())?;
        Ok(d)
    }
    #[getter]
    fn ptr(&self) -> u64 {
        self.ptr
    }
    #[getter]
    fn byte_len(&self) -> u64 {
        self.byte_len
    }
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }
}

/// Raw device pointer + byte length of a khal buffer on the CUDA backend.
fn cuda_ptr<T: khal::backend::DeviceValue>(
    buf: &khal::backend::GpuBuffer<T>,
) -> PyResult<(u64, u64)> {
    #[cfg(feature = "cuda")]
    if let khal::backend::GpuBuffer::Cuda(b) = buf {
        return Ok((b.device_ptr_raw(), b.byte_len()));
    }
    let _ = buf;
    Err(PyRuntimeError::new_err(
        "buffer is not on the CUDA backend; build NexusBackend(\"cuda\") and finalize_headless() with it",
    ))
}

fn cuda_array<T: khal::backend::DeviceValue>(
    buf: &khal::backend::GpuBuffer<T>,
    shape: Vec<usize>,
    elem_bytes: u64,
    what: &str,
) -> PyResult<CudaArray> {
    let (ptr, byte_len) = cuda_ptr(buf)?;
    let n: u64 = shape.iter().map(|&x| x as u64).product();
    if n * elem_bytes > byte_len {
        return Err(PyRuntimeError::new_err(format!(
            "{what}: shape {shape:?} needs {} bytes but buffer holds {byte_len}",
            n * elem_bytes
        )));
    }
    Ok(CudaArray { ptr, byte_len, shape, typestr: "<f4" })
}

/// A viewerless GPU backend for headless evaluation (no window, no
/// swapchain). `NexusBackend()` = headless WebGPU with the standard limits;
/// `NexusBackend("cuda")` = the native CUDA backend (cuda feature builds).
#[pyclass(unsendable)]
pub struct NexusBackend(pub khal::backend::GpuBackend);

#[pymethods]
impl NexusBackend {
    #[new]
    #[pyo3(signature = (kind = "webgpu"))]
    fn new(kind: &str) -> PyResult<Self> {
        use pyo3::exceptions::PyRuntimeError;
        match kind {
            #[cfg(feature = "cuda")]
            "cuda" => khal::backend::cuda::Cuda::new(0)
                .map(|c| NexusBackend(khal::backend::GpuBackend::Cuda(c)))
                .map_err(|e| PyRuntimeError::new_err(format!("CUDA init failed: {e:?}"))),
            "webgpu" => {
                let limits = khal::re_exports::wgpu::Limits {
                    max_buffer_size: 1_000_000_000,
                    max_storage_buffer_binding_size: 1_000_000_000,
                    max_storage_buffers_per_shader_stage: 14,
                    max_compute_workgroup_storage_size: 19_904,
                    ..Default::default()
                };
                let mut webgpu = pollster::block_on(khal::backend::WebGpu::new(
                    khal::re_exports::wgpu::Features::default(),
                    limits,
                ))
                .map_err(|e| PyRuntimeError::new_err(format!("WebGPU init failed: {e:?}")))?;
                webgpu.force_buffer_copy_src = true;
                Ok(NexusBackend(khal::backend::GpuBackend::WebGpu(webgpu)))
            }
            other => Err(PyRuntimeError::new_err(format!(
                "unknown backend kind {other:?} (use \"webgpu\" or \"cuda\")"
            ))),
        }
    }

    /// True if this backend is CUDA.
    fn is_cuda(&self) -> bool {
        self.0.is_cuda()
    }

    /// Block until all GPU work queued on this backend has completed.
    /// Required before reading a `CudaArray` view after `simulate_headless`.
    fn synchronize(&self) -> PyResult<()> {
        #[cfg(feature = "cuda")]
        if let khal::backend::GpuBackend::Cuda(c) = &self.0 {
            return c.stream().synchronize().map_err(gpu_err);
        }
        Ok(())
    }
}

/// The GPU-resident state of a multiphysics simulation
/// (`nexus3d::prelude::NexusState`). The second field keeps the
/// `rapier3d-mjcf` robot handles of the last `insert_mjcf`, so
/// `apply_actuator_controls` can drive the robot's actuators per step.
#[pyclass(name = "NexusState", unsendable)]
pub struct NexusState(
    pub RNexusState,
    pub Option<crate::loaders::MjcfHandles>,
    pub Vec<(Vec<u32>, u32, vortx::tensor::Tensor<f32>)>,
    pub Option<crate::loaders::MjcfNames>,
);


#[pymethods]
impl NexusState {
    #[new]
    fn new() -> Self {
        NexusState(RNexusState::default(), None, Vec::new(), None)
    }

    // --- rigid bodies -----------------------------------------------------

    fn insert_rigid_body(
        &mut self,
        body: PyRef<RigidBody>,
        collider: PyRef<Collider>,
    ) -> RigidBodyHandle {
        RigidBodyHandle(self.0.insert_rigid_body(body.0.clone(), collider.0.clone()))
    }

    fn insert_rigid_body_in(
        &mut self,
        env: usize,
        body: PyRef<RigidBody>,
        collider: PyRef<Collider>,
    ) -> RigidBodyHandle {
        RigidBodyHandle(
            self.0
                .insert_rigid_body_in(env, body.0.clone(), collider.0.clone()),
        )
    }

    fn insert_body(&mut self, body: PyRef<RigidBody>) -> RigidBodyHandle {
        RigidBodyHandle(self.0.insert_body(body.0.clone()))
    }

    /// Inserts a collider-less body into environment `env`; attach colliders to
    /// it afterwards with `insert_collider_in` (multiple colliders per body).
    fn insert_body_in(&mut self, env: usize, body: PyRef<RigidBody>) -> RigidBodyHandle {
        RigidBodyHandle(self.0.insert_body_in(env, body.0.clone()))
    }

    /// Attaches a collider to an existing body (`parent`), or inserts a
    /// parent-less one when `parent` is `None`, in environment `env`.
    #[pyo3(signature = (env, collider, parent=None))]
    fn insert_collider_in(
        &mut self,
        env: usize,
        collider: PyRef<Collider>,
        parent: Option<RigidBodyHandle>,
    ) {
        self.0
            .insert_collider_in(env, collider.0.clone(), parent.map(|h| h.0));
    }

    /// Reserves `capacity` spare GPU body slots (in environment 0) so later
    /// `add_rigid_bodies` calls append in place instead of forcing a full scene
    /// rebuild. Call this *before* the first `finalize`.
    fn reserve_rigid_bodies(&mut self, capacity: usize) {
        self.0.reserve_rigid_bodies(capacity);
    }

    /// Appends body+collider pairs to the *live* GPU scene (environment 0) in a
    /// single batch, without rebuilding — the fast path for spawning bodies
    /// mid-simulation. Unlike `insert_rigid_body` (whose bodies only reach the
    /// GPU on the next `finalize`), these are simulated immediately. Reserve
    /// capacity up-front with `reserve_rigid_bodies`; only primitive-shape
    /// colliders are supported on the fast path. Returns the new handles in
    /// input order.
    fn add_rigid_bodies(
        &mut self,
        viewer: PyRef<NexusViewer>,
        bodies: Vec<RigidBody>,
        colliders: Vec<Collider>,
    ) -> PyResult<Vec<RigidBodyHandle>> {
        if bodies.len() != colliders.len() {
            return Err(PyRuntimeError::new_err(
                "bodies and colliders must have the same length",
            ));
        }
        let pairs = bodies.into_iter().zip(colliders).map(|(b, c)| (b.0, c.0));
        self.0
            .add_rigid_bodies(viewer.backend(), pairs)
            .map(|hs| hs.into_iter().map(RigidBodyHandle).collect())
            .map_err(gpu_err)
    }

    // --- joints -----------------------------------------------------------

    fn insert_impulse_joint(
        &mut self,
        body1: RigidBodyHandle,
        body2: RigidBodyHandle,
        joint: JointArg,
    ) -> ImpulseJointHandle {
        ImpulseJointHandle(
            self.0
                .insert_impulse_joint(body1.0, body2.0, joint.into_generic()),
        )
    }

    fn insert_impulse_joint_in(
        &mut self,
        env: usize,
        body1: RigidBodyHandle,
        body2: RigidBodyHandle,
        joint: JointArg,
    ) -> ImpulseJointHandle {
        ImpulseJointHandle(self.0.insert_impulse_joint_in(
            env,
            body1.0,
            body2.0,
            joint.into_generic(),
        ))
    }

    fn insert_multibody_joint(
        &mut self,
        body1: RigidBodyHandle,
        body2: RigidBodyHandle,
        joint: JointArg,
    ) -> Option<MultibodyJointHandle> {
        self.0
            .insert_multibody_joint(body1.0, body2.0, joint.into_generic())
            .map(MultibodyJointHandle)
    }

    fn insert_multibody_joint_in(
        &mut self,
        env: usize,
        body1: RigidBodyHandle,
        body2: RigidBodyHandle,
        joint: JointArg,
    ) -> Option<MultibodyJointHandle> {
        self.0
            .insert_multibody_joint_in(env, body1.0, body2.0, joint.into_generic())
            .map(MultibodyJointHandle)
    }

    // --- batched environments ---------------------------------------------

    /// Allocates a new batched simulation environment, returning its index.
    fn add_environment(&mut self) -> usize {
        self.0.add_environment()
    }

    /// Number of GPU batches (== number of environments) once finalized.
    fn rbd_num_batches(&self) -> u32 {
        self.0.rbd_num_batches()
    }

    // --- zero-copy CUDA state views (Isaac Lab backend) --------------------

    /// Number of multibody links per batch (environment).
    fn links_per_batch(&self) -> PyResult<u32> {
        let rbd = self.0.rbd.as_ref().ok_or_else(|| PyRuntimeError::new_err("finalize first"))?;
        Ok(rbd.multibodies().links_per_batch())
    }

    /// Number of generalized DOFs per batch (environment).
    fn dofs_per_batch(&self) -> PyResult<u32> {
        let rbd = self.0.rbd.as_ref().ok_or_else(|| PyRuntimeError::new_err("finalize first"))?;
        Ok(rbd.multibodies().dofs_per_batch())
    }

    /// Layout of the per-link SoA workspace: quad offset of each field, quads
    /// per link, and batch geometry. The raw buffer index of (link k, quad q,
    /// batch b) is `(k * WS_QUADS + q) * num_batches + b` -- batch innermost --
    /// so `links_workspace_cuda()` viewed as (links, WS_QUADS, num_batches, 4)
    /// makes every field a strided view. Quaternions are stored x y z w; a
    /// pose field is (rotation quad, translation quad); a velocity field is
    /// (linear quad, angular quad).
    fn ws_layout<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        use crate::nexus::ws_layout::*;
        let rbd = self.0.rbd.as_ref().ok_or_else(|| PyRuntimeError::new_err("finalize first"))?;
        let d = PyDict::new(py);
        d.set_item("WS_JOINT_ROT", WS_JOINT_ROT)?;
        d.set_item("WS_COORDS", WS_COORDS)?;
        d.set_item("WS_LTP", WS_LTP)?;
        d.set_item("WS_LTW", WS_LTW)?;
        d.set_item("WS_JOINT_VEL", WS_JOINT_VEL)?;
        d.set_item("WS_RB_VELS", WS_RB_VELS)?;
        d.set_item("WS_KIN_ACC", WS_KIN_ACC)?;
        d.set_item("WS_QUADS", WS_QUADS)?;
        d.set_item("links_per_batch", rbd.multibodies().links_per_batch())?;
        d.set_item("dofs_per_batch", rbd.multibodies().dofs_per_batch())?;
        d.set_item("num_batches", self.0.rbd_num_batches())?;
        Ok(d)
    }

    /// Per-link static joint metadata for batch 0 (identical across batches),
    /// as a `(links_per_batch, 8)` uint32 array with columns
    /// `[rb_id, parent_link_id, multibody_id, assembly_id, ndofs, kinematic, locked_axes, motor_axes]`.
    /// `assembly_id` is the first flat DOF column of the link's joint and
    /// `ndofs` its width: together they map per-link `coords` slots onto
    /// Isaac Lab's flat `(num_envs, num_dofs)` joint vectors.
    fn links_static_host<'py>(&mut self, py: Python<'py>, backend: PyRef<NexusBackend>) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let nb = self.0.rbd_num_batches() as usize;
        let rbd = self.0.rbd.as_mut().ok_or_else(|| PyRuntimeError::new_err("finalize first"))?;
        let mb = rbd.multibodies_mut();
        let backend = &backend.0;
        let links = mb.links_per_batch() as usize;
        let t = mb.links_static_mut();
        let mut all = bytemuck::zeroed_vec(t.len() as usize);
        pollster::block_on(backend.slow_read_buffer(t.buffer(), &mut all)).map_err(gpu_err)?;
        if all.len() < links * nb {
            return Err(PyRuntimeError::new_err(format!("links_static len {} < links {} * batches {}", all.len(), links, nb)));
        }
        // batch-interleaved like the workspace: element (k, b) at k * nb + b; take b = 0
        let rows: Vec<Vec<u32>> = (0..links)
            .map(|k| {
                let l = &all[k * nb];
                vec![l.rb_id, l.parent_link_id, l.multibody_id, l.assembly_id, l.ndofs, l.kinematic, l.data.locked_axes, l.data.motor_axes]
            })
            .collect();
        PyArray2::from_vec2(py, &rows).map_err(gpu_err)
    }

    // --- write path (Isaac Lab backend) -----------------------------------

    /// Zero-copy CUDA view of the persistent external generalized forces (the
    /// RL torque input), shape `(dofs_per_batch, num_batches)` float32, layout
    /// `dof * num_batches + batch`. Write into it from torch; the gravity
    /// kernels add it every substep until it is overwritten. Zero it to stop.
    fn external_gen_forces_cuda(&mut self) -> PyResult<CudaArray> {
        let nb = self.0.rbd_num_batches() as usize;
        let rbd = self.0.rbd.as_mut().ok_or_else(|| PyRuntimeError::new_err("finalize first"))?;
        let mb = rbd.multibodies_mut();
        let dofs = mb.dofs_per_batch() as usize;
        let t = mb.external_gen_forces_mut();
        let have = t.len() as usize;
        if have < dofs * nb {
            return Err(PyRuntimeError::new_err(format!("external_gen_forces len {have} < dofs {dofs} * batches {nb}")));
        }
        cuda_array(t.buffer(), vec![dofs, nb], 4, "external_gen_forces")
    }

    /// Allocate a GPU target buffer for a group of actuated links sharing one
    /// joint axis, returning `(group_id, view)`. `view` is a zero-copy
    /// `(num_links, num_batches)` float32 array (element `(j, env)` at
    /// `j * num_batches + env`); write position targets into it from torch,
    /// then call `scatter_motor_targets(backend, group_id)` once per env step.
    fn motor_target_group(
        &mut self,
        backend: PyRef<NexusBackend>,
        link_ids: Vec<u32>,
        axis: u32,
    ) -> PyResult<(usize, CudaArray)> {
        use khal::BufferUsages;
        let nb = self.0.rbd_num_batches() as usize;
        let n = link_ids.len();
        let t = vortx::tensor::Tensor::vector(
            &backend.0,
            &vec![0.0f32; n * nb],
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        )
        .map_err(gpu_err)?;
        let view = cuda_array(t.buffer(), vec![n, nb], 4, "motor_targets")?;
        self.2.push((link_ids, axis, t));
        Ok((self.2.len() - 1, view))
    }

    /// Scatter a target group's positions into the GPU motor parameters
    /// (`motors[axis].target_pos` of each link, for every batch). One dispatch.
    fn scatter_motor_targets(&mut self, backend: PyRef<NexusBackend>, group_id: usize) -> PyResult<()> {
        let (ids, axis, t) = self
            .2
            .get(group_id)
            .ok_or_else(|| PyRuntimeError::new_err(format!("no motor target group {group_id}")))?;
        let rbd = self.0.rbd.as_mut().ok_or_else(|| PyRuntimeError::new_err("finalize first"))?;
        rbd.multibodies_mut()
            .scatter_motor_targets_gpu(&backend.0, t, ids, *axis)
            .map_err(gpu_err)
    }

    /// MJCF names resolved onto Nexus link indices (batch 0). Returns a dict:
    /// `link_body_names[k]`, `link_joint_names[k]` (joint driving link k, or
    /// "" for the root), `joint_names`, `joint_link_ids` (Nexus link per MJCF
    /// joint, -1 if unresolved), `actuator_names`, `actuator_joint_idx`.
    /// Call after `finalize_headless`.
    fn mjcf_names<'py>(&mut self, py: Python<'py>, backend: PyRef<NexusBackend>) -> PyResult<Bound<'py, PyDict>> {
        let names = self.3.clone().ok_or_else(|| PyRuntimeError::new_err("no MJCF loaded"))?;
        // gpu slot of each rapier body handle (env 0)
        let (body_gpu, joint_gpu): (Vec<Option<u32>>, Vec<Option<u32>>) = {
            let rb_gpu = |h: Option<rapier3d::prelude::RigidBodyHandle>| -> Option<u32> {
                let h = h?;
                let r = self.0.rbd2gpu.first()?.get(h.0)?;
                (r.gpu_id != u32::MAX).then_some(r.gpu_id)
            };
            (
                names.body_handles.iter().map(|h| rb_gpu(*h)).collect(),
                names.joint_body_handles.iter().map(|h| rb_gpu(*h)).collect(),
            )
        };
        // per-link rb_id from links_static
        let stat = self.links_static_host(py, backend)?;
        let stat = stat.readonly();
        let stat = stat.as_array();
        let links = stat.shape()[0];
        let mut link_body = vec![String::new(); links];
        let mut link_joint = vec![String::new(); links];
        let mut joint_link_ids = vec![-1i64; names.joint_names.len()];
        for k in 0..links {
            let rb = stat[[k, 0]];
            if let Some(i) = body_gpu.iter().position(|g| *g == Some(rb)) {
                link_body[k] = names.body_names[i].clone();
            }
            if let Some(j) = joint_gpu.iter().position(|g| *g == Some(rb)) {
                link_joint[k] = names.joint_names[j].clone();
                joint_link_ids[j] = k as i64;
            }
        }
        let d = PyDict::new(py);
        d.set_item("link_body_names", link_body)?;
        d.set_item("link_joint_names", link_joint)?;
        d.set_item("joint_names", names.joint_names.clone())?;
        d.set_item("joint_link_ids", joint_link_ids)?;
        d.set_item("actuator_names", names.actuator_names.clone())?;
        d.set_item("actuator_joint_idx", names.actuator_joint_idx.iter().map(|x| x.map(|v| v as i64).unwrap_or(-1)).collect::<Vec<i64>>())?;
        Ok(d)
    }

    /// Capture the current physics state (call right after `finalize_headless`,
    /// i.e. the initial pose) and publish it to the GPU as reset template 0.
    /// Required once before `reset_envs`.
    fn publish_reset_template(&mut self, backend: PyRef<NexusBackend>) -> PyResult<()> {
        let rbd = self.0.rbd.as_mut().ok_or_else(|| PyRuntimeError::new_err("finalize first"))?;
        let snap = pollster::block_on(rbd.snapshot(&backend.0));
        rbd.publish_reset_templates(&backend.0, &[&snap]);
        Ok(())
    }

    /// Batched reset: restore every env in `env_ids` from template 0, translated
    /// by `offsets[i]` (x y z), with `dof_vels` (`dofs_per_batch` floats per env,
    /// flattened) written into the generalized velocities. One upload, two
    /// dispatches, one submit for the whole batch.
    /// Reset `env_ids` to their published templates. `offsets` and `dof_vels`
    /// default to zeros when omitted — the common RL case, and passing them
    /// explicitly means marshalling `len(env_ids) * dofs` Python floats every
    /// reset (143k of them at 4096 envs x 35 dofs).
    #[pyo3(signature = (backend, env_ids, offsets=None, dof_vels=None))]
    fn reset_envs(
        &mut self,
        backend: PyRef<NexusBackend>,
        env_ids: Vec<u32>,
        offsets: Option<Vec<[f32; 3]>>,
        dof_vels: Option<Vec<f32>>,
    ) -> PyResult<()> {
        let rbd = self.0.rbd.as_mut().ok_or_else(|| PyRuntimeError::new_err("finalize first"))?;
        let dofs = rbd.multibodies().dofs_per_batch() as usize;
        if let Some(o) = offsets.as_ref() {
            if o.len() != env_ids.len() {
                return Err(PyRuntimeError::new_err(format!(
                    "reset_envs: {} envs, {} offsets", env_ids.len(), o.len()
                )));
            }
        }
        if let Some(v) = dof_vels.as_ref() {
            if v.len() != env_ids.len() * dofs {
                return Err(PyRuntimeError::new_err(format!(
                    "reset_envs: {} envs, {} dof_vels (need {} = envs * dofs {})",
                    env_ids.len(), v.len(), env_ids.len() * dofs, dofs
                )));
            }
        }
        let resets: Vec<(u32, u32)> = env_ids.iter().map(|&e| (e, 0u32)).collect();
        let offs: Vec<glamx::Vec3> = match offsets {
            Some(o) => o.iter().map(|o| glamx::Vec3::new(o[0], o[1], o[2])).collect(),
            None => vec![glamx::Vec3::ZERO; env_ids.len()],
        };
        let vels: Vec<f32> = dof_vels.unwrap_or_else(|| vec![0.0; env_ids.len() * dofs]);
        rbd.reset_envs_from_templates(&backend.0, &resets, &offs, &vels);
        Ok(())
    }

    /// Set PD motor gains (force-based model) on `link_ids` for `axis`, for every
    /// batch, and upload once. Call BEFORE the first `scatter_motor_targets`.
    #[pyo3(signature = (backend, link_ids, axis, stiffness, damping, max_force, max_velocity = f32::INFINITY))]
    fn set_motor_gains(
        &mut self,
        backend: PyRef<NexusBackend>,
        link_ids: Vec<u32>,
        axis: u32,
        stiffness: f32,
        damping: f32,
        max_force: f32,
        max_velocity: f32,
    ) -> PyResult<()> {
        use nexus3d::rbd::rapier::prelude::JointAxis;
        let nb = self.0.rbd_num_batches();
        let axis = match axis {
            0 => JointAxis::LinX,
            1 => JointAxis::LinY,
            2 => JointAxis::LinZ,
            3 => JointAxis::AngX,
            4 => JointAxis::AngY,
            5 => JointAxis::AngZ,
            _ => return Err(PyRuntimeError::new_err(format!("bad joint axis {axis} (0..5)"))),
        };
        let rbd = self.0.rbd.as_mut().ok_or_else(|| PyRuntimeError::new_err("finalize first"))?;
        let mb = rbd.multibodies_mut();
        for &k in &link_ids {
            for b in 0..nb {
                mb.stage_motor_gains(b, k, axis, stiffness, damping, max_force, 1 /* FORCE_BASED */, max_velocity);
            }
        }
        mb.flush_links_static(&backend.0).map_err(gpu_err)
    }

    /// Zero-copy CUDA view of the multibody contact-constraint slab as raw
    /// float32 words, shape `(total_slots, stride)`; u32 fields must be
    /// reinterpreted (`.view(torch.int32)`). See `contact_layout()`.
    #[pyo3(signature = (solved = true))]
    fn contact_constraints_cuda(&mut self, solved: bool) -> PyResult<CudaArray> {
        let rbd = self.0.rbd.as_mut().ok_or_else(|| PyRuntimeError::new_err("finalize first"))?;
        let mbs = rbd.multibodies();
        // NOTE: multibody-vs-static contacts are solved on the rigid-body path
        // (`RbdState::old_constraints`); this slab only carries multibody/free-body
        // coupling and shows impulse == 0 for floor contacts. `solved` is kept for API stability.
        let _ = solved;
        let t = mbs.contact_constraints();
        let n = t.len() as usize;
        let stride = (t.bytes_len() / t.len().max(1)) as usize / 4;
        if stride != 28 {
            return Err(PyRuntimeError::new_err(format!(
                "MultibodyContactConstraint stride is {stride} floats, binding assumes 28; update contact_layout"
            )));
        }
        cuda_array(t.buffer(), vec![n, stride], 4, "contact_constraints")
    }

    /// Layout of the contact-constraint slab: slots per batch, per-multibody
    /// stride, multibodies per batch, and float-word offsets of the fields
    /// needed for per-link net contact force.
    fn contact_layout<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let rbd = self.0.rbd.as_ref().ok_or_else(|| PyRuntimeError::new_err("finalize first"))?;
        let mb = rbd.multibodies();
        let nb = self.0.rbd_num_batches().max(1) as usize;
        let d = PyDict::new(py);
        d.set_item("per_batch", mb.contact_constraints_per_batch())?;
        d.set_item("multibodies_per_batch", (mb.multibody_info().len() as usize / nb) as u32)?;
        d.set_item("stride", 28u32)?;
        d.set_item("kind_normal", 1u32)?;
        d.set_item("kind_tangent", 2u32)?;
        d.set_item("off_multibody_id", 0u32)?;
        d.set_item("off_link_id", 1u32)?;
        d.set_item("off_kind", 2u32)?;
        d.set_item("off_free_body_id", 3u32)?;
        d.set_item("off_lin_jac", 8u32)?;
        d.set_item("off_impulse", 23u32)?;
        Ok(d)
    }

    /// Zero-copy view of the SOLVED rigid-body contact constraints (last step),
    /// raw float32 words `(total_slots, stride)`; see `rigid_contact_layout()`.
    /// Slot layout is `batch * per_batch + i`; `contacts_len_cuda()` gives the
    /// live count per batch.
    fn rigid_contacts_cuda(&mut self) -> PyResult<CudaArray> {
        let rbd = self.0.rbd.as_mut().ok_or_else(|| PyRuntimeError::new_err("finalize first"))?;
        let t = rbd.old_constraints();
        let n = t.len() as usize;
        let stride = (t.bytes_len() / t.len().max(1)) as usize / 4;
        cuda_array(t.buffer(), vec![n, stride], 4, "rigid_contacts")
    }

    /// Zero-copy view of per-batch contact counts, shape `(num_batches,)` (uint32 bits as f32 view; reinterpret).
    fn contacts_len_cuda(&mut self) -> PyResult<CudaArray> {
        let rbd = self.0.rbd.as_mut().ok_or_else(|| PyRuntimeError::new_err("finalize first"))?;
        let t = rbd.contacts_len();
        cuda_array(t.buffer(), vec![t.len() as usize], 4, "contacts_len")
    }

    /// Layout of `rigid_contacts_cuda()` in float words, computed from the Rust
    /// struct with `offset_of!` (no guessing): stride, dir_a, solver_body_a/b,
    /// len, elements, elem_stride, elem_normal_impulse, plus per_batch slots.
    fn rigid_contact_layout<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        use nexus3d::rbd::pipeline::RbdState;
        let rbd = self.0.rbd.as_ref().ok_or_else(|| PyRuntimeError::new_err("finalize first"))?;
        let l = RbdState::two_body_constraint_layout();
        let nb = self.0.rbd_num_batches().max(1) as usize;
        let d = PyDict::new(py);
        for (k, v) in ["stride", "off_dir_a", "off_solver_body_a", "off_solver_body_b", "off_len", "off_elements", "elem_stride", "off_elem_normal_impulse"].iter().zip(l) {
            d.set_item(*k, v)?;
        }
        d.set_item("per_batch", (rbd.old_constraints().len() as usize / nb) as u32)?;
        d.set_item("max_elements", 4u32)?;
        Ok(d)
    }

    /// Zero-copy view of ALL rigid-body world poses (multibody links and free
    /// bodies alike), raw float32 `(num_bodies_total, stride)`; a Pose3 is a
    /// quaternion (x y z w) followed by a translation. Index = gpu body id.
    fn body_poses_cuda(&mut self) -> PyResult<CudaArray> {
        let rbd = self.0.rbd.as_mut().ok_or_else(|| PyRuntimeError::new_err("finalize first"))?;
        let t = rbd.body_poses();
        let n = t.len() as usize;
        let stride = (t.bytes_len() / t.len().max(1)) as usize / 4;
        cuda_array(t.buffer(), vec![n, stride], 4, "body_poses")
    }

    /// Configure the engine's contact force sensors: up to `MAX_CONTACT_SENSORS`
    /// (4) multibody link ids, shared by every multibody in every batch. Returns
    /// the number of sensors accepted.
    fn set_contact_sensor_links(&mut self, backend: PyRef<NexusBackend>, links: Vec<u32>) -> PyResult<u32> {
        let rbd = self.0.rbd.as_mut().ok_or_else(|| PyRuntimeError::new_err("finalize first"))?;
        let mb = rbd.multibodies_mut();
        mb.set_contact_sensor_links(&backend.0, &links);
        Ok(mb.num_contact_sensors())
    }

    /// Zero-copy view of the contact sensor readout, shape
    /// `(multibodies_per_batch, num_batches, MAX_CONTACT_SENSORS)` float32:
    /// accumulated normal impulse per sensed link over the last step.
    fn contact_sensor_out_cuda(&mut self) -> PyResult<CudaArray> {
        let nb = self.0.rbd_num_batches().max(1) as usize;
        let rbd = self.0.rbd.as_mut().ok_or_else(|| PyRuntimeError::new_err("finalize first"))?;
        let mb = rbd.multibodies_mut();
        let mbs = mb.multibody_info().len() as usize / nb;
        let t = mb.contact_sensor_out();
        let n = t.len() as usize;
        let maxs = if mbs * nb > 0 { n / (mbs * nb) } else { 0 };
        if maxs == 0 || mbs * nb * maxs != n {
            return Err(PyRuntimeError::new_err(format!("contact_sensor_out len {n} vs mbs {mbs} * batches {nb}")));
        }
        cuda_array(t.buffer(), vec![mbs, nb, maxs], 4, "contact_sensor_out")
    }

    /// Zero-copy CUDA view of the links workspace, shape
    /// `(links_per_batch, WS_QUADS, num_batches, 4)` float32. See `ws_layout`.
    fn links_workspace_cuda(&self) -> PyResult<CudaArray> {
        use crate::nexus::ws_layout::WS_QUADS;
        let rbd = self.0.rbd.as_ref().ok_or_else(|| PyRuntimeError::new_err("finalize first"))?;
        let mb = rbd.multibodies();
        let t = mb.links_workspace_buffer();
        let nb = self.0.rbd_num_batches() as usize;
        let quads = WS_QUADS as usize;
        let links = mb.links_per_batch() as usize;
        let expect = links * quads * nb;
        let have = t.len() as usize;
        if have != expect {
            return Err(PyRuntimeError::new_err(format!(
                "links_workspace length {have} != links({links}) * WS_QUADS({quads}) * batches({nb}) = {expect}"
            )));
        }
        cuda_array(t.buffer(), vec![links, quads, nb, 4], 4, "links_workspace")
    }

    /// Zero-copy CUDA view of the DOF state, shape
    /// `(sections, dofs_per_batch, num_batches)` float32; section 0 is the
    /// generalized velocities, later sections are per-DOF parameters.
    fn dof_state_cuda(&self) -> PyResult<CudaArray> {
        let rbd = self.0.rbd.as_ref().ok_or_else(|| PyRuntimeError::new_err("finalize first"))?;
        let mb = rbd.multibodies();
        let t = mb.dof_state();
        let nb = self.0.rbd_num_batches() as usize;
        let dofs = mb.dofs_per_batch() as usize;
        let per = dofs * nb;
        let have = t.len() as usize;
        if per == 0 || have % per != 0 {
            return Err(PyRuntimeError::new_err(format!(
                "dof_state length {have} not a multiple of dofs({dofs}) * batches({nb})"
            )));
        }
        cuda_array(t.buffer(), vec![have / per, dofs, nb], 4, "dof_state")
    }

    // --- robot loaders ----------------------------------------------------

    /// Loads a URDF robot into environment 0 as a multibody and returns the
    /// per-collider render shapes plus the link count. Register the shapes with
    /// `viewer.insert_visual_shape(0, body, shape, pose)`.
    ///
    /// With `actuate_angx_motors=True` every joint's `AngX` motor is switched to
    /// acceleration-based mode (initial target velocity 0), ready for per-frame
    /// `set_multibody_motor_velocity` control.
    #[pyo3(signature = (path, options, actuate_angx_motors=false))]
    fn insert_urdf(
        &mut self,
        path: std::path::PathBuf,
        options: PyRef<UrdfLoaderOptions>,
        actuate_angx_motors: bool,
    ) -> PyResult<UrdfRobotHandles> {
        use rapier3d_urdf::{UrdfMultibodyOptions, UrdfRobot};
        let opts = options.to_rapier();
        let (mut robot, _) = UrdfRobot::from_file(&path, opts, None).map_err(|e| {
            PyRuntimeError::new_err(format!("failed to load URDF {}: {e}", path.display()))
        })?;
        if actuate_angx_motors {
            for j in &mut robot.joints {
                j.joint
                    .set_motor_model(rp::JointAxis::AngX, rp::MotorModel::AccelerationBased);
                j.joint.set_motor_velocity(rp::JointAxis::AngX, 0.0, 1.0);
            }
        }
        let world = self.0.rbd_world_mut(0);
        let handles = robot.insert_using_multibody_joints(
            &mut world.bodies,
            &mut world.colliders,
            &mut world.multibody_joints,
            UrdfMultibodyOptions::DISABLE_SELF_CONTACTS,
        );
        let num_links = handles.links.len() as u32;
        let mut render_shapes = Vec::new();
        for link in &handles.links {
            for collider in &link.colliders {
                let (shape, local_pose) = match &collider.visual {
                    Some(v) => (v.shape.clone(), v.local_pose),
                    None => (
                        world.colliders[collider.handle].shared_shape().clone(),
                        rp::Pose::IDENTITY,
                    ),
                };
                render_shapes.push((
                    RigidBodyHandle(link.body),
                    SharedShape(shape),
                    Pose(local_pose),
                ));
            }
        }
        Ok(UrdfRobotHandles {
            render_shapes,
            num_links,
        })
    }

    /// Loads a MuJoCo MJCF scene into environment 0 as multibodies, registering
    /// its render shapes (and a sized floor) with `viewer`. Returns scene info
    /// (suggested camera + whether the scene is Z-up). Call `finalize` after.
    /// Per-environment collision-pair capacity (default 4096). Lower this
    /// before `finalize` when batching many small environments — pair-keyed
    /// GPU workspaces scale with `capacity x num_envs`.
    /// Selects the multibody integration mode: `False` = MuJoCo/Genesis-style
    /// explicit coriolis — the mass matrix / LU / gravity solve runs ONCE per
    /// step instead of once per substep (~4x less dynamics work at 4
    /// substeps; slightly different integration semantics). Call after
    /// `finalize`.
    fn set_implicit_coriolis(&mut self, enabled: bool) {
        if let Some(rbd) = self.0.rbd.as_mut() {
            rbd.multibodies_mut().set_implicit_coriolis(enabled);
        }
    }

    /// Sets every environment's physics timestep (call before `finalize`;
    /// triggers a rebuild). Headless-eval surface.
    fn set_rbd_dt(&mut self, dt: f32) {
        self.0.set_rbd_dt(dt);
    }

    /// Sets every environment's solver substep count (call before `finalize`;
    /// triggers a rebuild). Headless-eval surface — match an external engine's
    /// integration cadence (e.g. MuJoCo Euler = 1; nexus default = 4).
    fn set_rbd_solver_iterations(&mut self, iterations: usize) {
        self.0.set_rbd_solver_iterations(iterations);
    }

    /// Physics-only MJCF load (robot + auto floor, no renderer).
    /// Load an MJCF robot into env `env`. `translation` shifts every robot body
    /// at spawn (e.g. to start above terrain); `auto_floor` adds the loader's
    /// flat floor under the robot (disable when the scene has its own ground).
    #[pyo3(signature = (scene_path, env, translation = None, auto_floor = true))]
    fn insert_mjcf_headless(
        &mut self,
        scene_path: std::path::PathBuf,
        env: usize,
        translation: Option<[f32; 3]>,
        auto_floor: bool,
    ) -> PyResult<MjcfSceneInfo> {
        let (info, handles, names) = crate::loaders::insert_mjcf_headless(&mut self.0, &scene_path, env, translation, auto_floor)?;
        if names.is_some() {
            self.3 = names;
        }
        if env == 0 {
            self.1 = handles;
        }
        Ok(info)
    }

    /// Insert the same MJCF into environments `env_start..env_end`, parsing the
    /// file and building its convex hulls ONCE (spawning 4096 copies one call at
    /// a time re-parses the XML and its meshes 4096 times).
    #[pyo3(signature = (scene_path, env_start, env_end, translation=None, auto_floor=true))]
    fn insert_mjcf_headless_range(
        &mut self,
        scene_path: std::path::PathBuf,
        env_start: usize,
        env_end: usize,
        translation: Option<[f32; 3]>,
        auto_floor: bool,
    ) -> PyResult<MjcfSceneInfo> {
        let (info, handles, names) = crate::loaders::insert_mjcf_headless_range(
            &mut self.0, &scene_path, env_start, env_end, translation, auto_floor,
        )?;
        if names.is_some() {
            self.3 = names;
        }
        if env_start == 0 {
            self.1 = handles;
        }
        Ok(info)
    }

    /// Windowless `finalize`: uploads the scene to the GPU.
    fn finalize_headless(&mut self, backend: PyRef<NexusBackend>) -> PyResult<()> {
        pollster::block_on(self.0.finalize(&backend.0)).map_err(gpu_err)
    }

    /// Windowless gravity setter (call after `finalize_headless`).
    fn set_rbd_gravity_headless(&mut self, backend: PyRef<NexusBackend>, gravity: Vec3) {
        self.0
            .set_rbd_gravity(&backend.0, [gravity.0.x, gravity.0.y, gravity.0.z]);
    }

    /// Sets environment `env`'s persistent external generalized forces (RL
    /// torque input: free-base DOFs first, then joints in link order).
    /// Applied every substep until the next call.
    fn set_multibody_gen_forces_headless(
        &mut self,
        backend: PyRef<NexusBackend>,
        env: u32,
        forces: Vec<f32>,
    ) -> PyResult<()> {
        let Some(rbd) = self.0.rbd.as_mut() else {
            return Err(PyRuntimeError::new_err("state not finalized"));
        };
        rbd.multibodies_mut()
            .set_external_gen_forces(&backend.0, env, &forces)
            .map_err(gpu_err)
    }

    /// Environment 0's per-link generalized joint coordinates as an
    /// `(n_links, 6)` float32 array (GPU link traversal order).
    fn link_coords<'py>(
        &self,
        py: Python<'py>,
        backend: PyRef<NexusBackend>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        let Some(rbd) = self.0.rbd.as_ref() else {
            return Err(PyRuntimeError::new_err("state not finalized"));
        };
        let links = pollster::block_on(rbd.multibodies().read_links(&backend.0, 0));
        let rows: Vec<Vec<f32>> = links.iter().map(|w| w.coords.to_vec()).collect();
        Ok(numpy::PyArray2::from_vec2(py, &rows).map_err(|e| PyRuntimeError::new_err(e.to_string()))?)
    }

    /// Environment 0's generalized velocities (free-base spatial velocity
    /// first — linear 0:3, angular 3:6, world frame — then joint rates in
    /// link order) as a float32 array.
    fn dof_velocities<'py>(
        &self,
        py: Python<'py>,
        backend: PyRef<NexusBackend>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f32>>> {
        use khal::backend::Backend as _;
        let Some(rbd) = self.0.rbd.as_ref() else {
            return Err(PyRuntimeError::new_err("state not finalized"));
        };
        let mb = rbd.multibodies();
        let nb = mb.num_batches() as usize;
        let dpb = mb.dofs_per_batch() as usize;
        let mut all = vec![0.0f32; mb.dof_state().len() as usize];
        pollster::block_on(backend.0.slow_read_buffer(mb.dof_state().buffer(), &mut all))
            .map_err(gpu_err)?;
        // Velocity section, batch-interleaved: DOF d of env 0 at d*nb.
        let vels: Vec<f32> = (0..dpb).map(|d| all[d * nb]).collect();
        Ok(numpy::PyArray1::from_vec(py, vels))
    }

    /// Environment 0's rigid-body poses as an `(n_bodies, 7)` float32 array,
    /// rows `[tx, ty, tz, qx, qy, qz, qw]`.
    fn body_poses<'py>(
        &self,
        py: Python<'py>,
        backend: PyRef<NexusBackend>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        use khal::backend::Backend as _;
        let Some(rbd) = self.0.rbd.as_ref() else {
            return Err(PyRuntimeError::new_err("state not finalized"));
        };
        let mut all: Vec<glamx::Pose3> = vec![Default::default(); rbd.body_poses().len() as usize];
        pollster::block_on(backend.0.slow_read_buffer(rbd.body_poses().buffer(), &mut all))
            .map_err(gpu_err)?;
        let nb = self.0.num_environments().max(1);
        let stride = all.len() / nb;
        let rows: Vec<Vec<f32>> = all[..stride]
            .iter()
            .map(|p| {
                vec![
                    p.translation.x,
                    p.translation.y,
                    p.translation.z,
                    p.rotation.x,
                    p.rotation.y,
                    p.rotation.z,
                    p.rotation.w,
                ]
            })
            .collect();
        Ok(numpy::PyArray2::from_vec2(py, &rows).map_err(|e| PyRuntimeError::new_err(e.to_string()))?)
    }

    /// Collision-buffer resize policy: "grow" (engine default: grow on any
    /// batch's spike, never shrink), "fit" (grow and shrink), or "fixed"
    /// (never resize; overflowing pairs are DROPPED). Applies to a finalized
    /// state immediately and is also recorded for the next finalize.
    fn set_rbd_resize_policy(&mut self, policy: &str) -> PyResult<()> {
        use nexus3d::prelude::RbdResizePolicy;
        let p = match policy {
            "grow" => RbdResizePolicy::Grow,
            "fit" => RbdResizePolicy::Fit,
            "fixed" => RbdResizePolicy::Fixed,
            other => return Err(PyRuntimeError::new_err(format!("unknown resize policy {other:?} (grow|fit|fixed)"))),
        };
        self.0.set_rbd_resize_policy(p);
        Ok(())
    }

    /// Resize bookkeeping of the live rigid-body state, as a dict:
    /// `pairs_len` (last read-back max collision pairs across batches),
    /// `capacity_per_batch` (allocated), `capacity_min` (configured floor),
    /// `max_colors`, `colors_high_water`, `rb_contacts_inert`.
    fn rbd_resize_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("capacity_min", self.0.rbd_collisions_capacity())?;
        if let Some(rbd) = self.0.rbd.as_ref() {
            d.set_item("pairs_len", rbd.collision_pairs_len_cpu())?;
            d.set_item("capacity_per_batch", rbd.collisions_capacity_per_batch())?;
            d.set_item("max_colors", rbd.max_colors())?;
            d.set_item("colors_high_water", rbd.colors_high_water())?;
            d.set_item("rb_contacts_inert", rbd.rb_contacts_inert())?;
        }
        Ok(d)
    }

    fn set_rbd_collisions_capacity(&mut self, capacity: u32) {
        self.0.set_rbd_collisions_capacity(capacity);
    }

    #[pyo3(signature = (viewer, scene_path, render_colliders=false, env=0))]
    fn insert_mjcf(
        &mut self,
        viewer: PyRefMut<NexusViewer>,
        scene_path: std::path::PathBuf,
        render_colliders: bool,
        env: usize,
    ) -> PyResult<MjcfSceneInfo> {
        let (info, handles) =
            crate::loaders::insert_mjcf(&mut self.0, viewer, &scene_path, render_colliders, env)?;
        if env == 0 {
            self.1 = handles;
        }
        Ok(info)
    }

    // --- MJCF actuation -----------------------------------------------------

    /// Names of the MJCF `<actuator>`s of the robot loaded by `insert_mjcf`, in
    /// actuator (control-vector) order. Unnamed actuators fall back to the name
    /// of the joint they drive. Empty before `insert_mjcf`.
    fn actuator_names(&self) -> Vec<String> {
        self.1
            .as_ref()
            .map(|h| {
                h.actuators
                    .iter()
                    .map(|a| {
                        a.actuator
                            .name
                            .clone()
                            .or_else(|| a.actuator.joint.clone())
                            .unwrap_or_default()
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Applies one MJCF control vector (one entry per actuator, in
    /// `actuator_names` order) to the robot loaded by `insert_mjcf`, with full
    /// MJCF actuator semantics (`<position>` servos with kp/kv, `<motor>`
    /// force/gear, force limits), and pushes the resulting joint-motor state to
    /// the GPU in one buffer write.
    ///
    /// Call once per control step, after `finalize`; the next
    /// `NexusPipeline.simulate` steps the solver against the new targets. This
    /// is the GPU counterpart of stepping rapier natively with actuators.
    #[pyo3(signature = (viewer, ctrl, env=0))]
    fn apply_actuator_controls(
        &mut self,
        viewer: PyRef<NexusViewer>,
        ctrl: Vec<f32>,
        env: usize,
    ) -> PyResult<()> {
        let Some(handles) = self.1.as_ref() else {
            return Err(PyRuntimeError::new_err(
                "no MJCF robot loaded (call insert_mjcf first)",
            ));
        };
        if ctrl.len() != handles.actuators.len() {
            return Err(PyRuntimeError::new_err(format!(
                "ctrl has {} entries but the robot has {} actuators",
                ctrl.len(),
                handles.actuators.len()
            )));
        }
        let handles = handles.clone();
        self.0
            .control_multibody_motors(viewer.backend(), env, |world| {
                handles.apply_controls_multibody(
                    &mut world.bodies,
                    &mut world.multibody_joints,
                    &ctrl,
                );
            })
            .map_err(gpu_err)
    }

    // --- rbd config -------------------------------------------------------

    fn set_rbd_steps_per_frame(&mut self, steps: u32) {
        self.0.set_rbd_steps_per_frame(steps);
    }

    fn set_rbd_gravity(&mut self, viewer: PyRef<NexusViewer>, gravity: Vec3) {
        self.0
            .set_rbd_gravity(viewer.backend(), [gravity.0.x, gravity.0.y, gravity.0.z]);
    }

    /// Sets the gravity of the CPU-side rapier world used by `step_rapier`
    /// (independent of the GPU state's gravity set by `set_rbd_gravity`).
    #[pyo3(signature = (gravity, env=0))]
    fn set_rapier_gravity(&mut self, gravity: Vec3, env: usize) {
        self.0.rbd_world_mut(env).gravity = gravity.0;
    }

    /// Advances the CPU-side rapier world natively (no GPU physics at all) by
    /// `steps` timesteps of its `integration_parameters.dt` (default 1/60 s).
    ///
    /// This steps the same rapier world the scene was built into — including
    /// multibody joints and the position servos imported from MJCF actuators —
    /// so robots hold their actuated stance. Pair with
    /// `NexusViewer.sync_rapier(state)` to push the resulting poses into the
    /// renderer. Orders of magnitude faster than the GPU pipeline for a single
    /// environment (no per-kernel dispatch overhead).
    /// Debug: (bodies, colliders, contact_pairs, active_contact_points, min_dynamic_mass).
    #[pyo3(signature = (env=0))]
    fn rapier_debug(&mut self, env: usize) -> (usize, usize, usize, usize, f32) {
        let world = self.0.rbd_world_mut(env);
        let mut pairs = 0usize;
        let mut points = 0usize;
        for c in world.narrow_phase.contact_pairs() {
            pairs += 1;
            points += c.manifolds.iter().map(|m| m.points.len()).sum::<usize>();
        }
        let min_mass = world
            .bodies
            .iter()
            .filter(|(_, b)| b.is_dynamic())
            .map(|(_, b)| b.mass())
            .fold(f32::INFINITY, f32::min);
        (world.bodies.len(), world.colliders.len(), pairs, points, min_mass)
    }

    /// Debug: raises every dynamic body's mass (and inertia, proportionally)
    /// to at least `min_mass` kg. Workaround for near-massless connector links
    /// destabilizing multibody contact resolution. Returns how many bodies
    /// were boosted.
    #[pyo3(signature = (min_mass, env=0))]
    fn boost_light_rapier_links(&mut self, min_mass: f32, env: usize) -> u32 {
        use rp::MassProperties;
        let world = self.0.rbd_world_mut(env);
        let mut boosted = 0;
        for (_, body) in world.bodies.iter_mut() {
            if !body.is_dynamic() {
                continue;
            }
            let mass = body.mass();
            if mass <= 0.0 || mass >= min_mass {
                continue;
            }
            let f = min_mass / mass;
            let local = body.mass_properties().local_mprops;
            let props = MassProperties::new(
                local.local_com,
                mass * f,
                local.principal_inertia() * f,
            );
            body.set_additional_mass_properties(props, true);
            boosted += 1;
        }
        boosted
    }

    /// Debug: scales every multibody motor's stiffness/damping (0 = disable).
    #[pyo3(signature = (scale, env=0))]
    fn scale_rapier_motors(&mut self, scale: f32, env: usize) {
        let world = self.0.rbd_world_mut(env);
        let links: Vec<_> = {
            let joints = &world.multibody_joints;
            world
                .bodies
                .iter()
                .filter_map(|(h, _)| joints.rigid_body_link(h).copied())
                .collect()
        };
        for lid in links {
            let Some(mb) = world.multibody_joints.get_multibody_mut(lid.multibody) else {
                continue;
            };
            let Some(link) = mb.link_mut(lid.id) else { continue };
            for motor in link.joint.data.motors.iter_mut() {
                motor.stiffness *= scale;
                motor.damping *= scale;
            }
        }
    }

    /// Debug: per multibody link, motors with any nonzero parameter:
    /// (link_id, axis, stiffness, damping, target_pos, max_force).
    #[pyo3(signature = (env=0))]
    fn rapier_debug_motors(&mut self, env: usize) -> Vec<(usize, usize, f32, f32, f32, f32)> {
        let world = self.0.rbd_world_mut(env);
        let mut out = Vec::new();
        for mb in world.multibody_joints.multibodies() {
            for (i, link) in mb.links().enumerate() {
                for (axis, m) in link.joint().data.motors.iter().enumerate() {
                    if m.stiffness != 0.0 || m.damping != 0.0 || m.max_force != 0.0 {
                        out.push((i, axis, m.stiffness, m.damping, m.target_pos, m.max_force));
                    }
                }
            }
        }
        out
    }

    /// Debug: per-body (is_dynamic, z, vz, mass).
    #[pyo3(signature = (env=0))]
    fn rapier_debug_bodies(&mut self, env: usize) -> Vec<(bool, f32, f32, f32)> {
        let world = self.0.rbd_world_mut(env);
        world
            .bodies
            .iter()
            .map(|(_, b)| {
                (
                    b.is_dynamic(),
                    b.position().translation.z,
                    b.linvel().z,
                    b.mass(),
                )
            })
            .collect()
    }

    /// Debug: per-collider (has_parent, world_z, groups_bits).
    #[pyo3(signature = (env=0))]
    fn rapier_debug_colliders(&mut self, env: usize) -> Vec<(bool, f32, u32)> {
        let world = self.0.rbd_world_mut(env);
        world
            .colliders
            .iter()
            .map(|(_, c)| {
                (
                    c.parent().is_some(),
                    c.position().translation.z,
                    c.collision_groups().memberships.bits(),
                )
            })
            .collect()
    }

    #[pyo3(signature = (steps=1, env=0, dt=None))]
    fn step_rapier(&mut self, steps: u32, env: usize, dt: Option<f32>) {
        let world = self.0.rbd_world_mut(env);
        if let Some(dt) = dt {
            world.integration_parameters.dt = dt;
        }
        for _ in 0..steps {
            world.step();
        }
    }

    fn set_multibody_motor_velocity(
        &mut self,
        viewer: PyRef<NexusViewer>,
        batch: u32,
        link_id: u32,
        axis: JointAxis,
        target_vel: f32,
    ) -> PyResult<()> {
        self.0
            .set_multibody_motor_velocity(
                viewer.backend(),
                batch,
                link_id,
                axis.to_rapier(),
                target_vel,
            )
            .map_err(gpu_err)
    }

    // --- lifecycle --------------------------------------------------------

    fn counts(&self) -> NexusCounts {
        let c = self.0.counts();
        NexusCounts {
            num_environments: c.num_environments,
            rigid_bodies: c.rigid_bodies,
            colliders: c.colliders,
            impulse_joints: c.impulse_joints,
            multibodies: c.multibodies,
            multibody_dofs: c.multibody_dofs,
        }
    }

    /// Uploads the scene to the GPU. Must be called before the first
    /// `simulate`. Blocks on the underlying async GPU work.
    fn finalize(&mut self, viewer: PyRef<NexusViewer>) -> PyResult<()> {
        pollster::block_on(self.0.finalize(viewer.backend())).map_err(gpu_err)
    }
}

/// The GPU compute pipelines (`nexus3d::prelude::NexusPipeline`).
#[pyclass(name = "NexusPipeline", unsendable)]
pub struct NexusPipeline(pub RNexusPipeline);

#[pymethods]
impl NexusPipeline {
    #[new]
    fn new() -> Self {
        NexusPipeline(RNexusPipeline::default())
    }

    /// Compiles all GPU pipelines up-front.
    fn preload_pipelines(&mut self, viewer: PyRef<NexusViewer>) -> PyResult<()> {
        self.0
            .preload_pipelines(viewer.backend(), NexusPipelineMask::all())
            .map_err(gpu_err)
    }

    /// Captures one frame's rigid-body step sequence
    /// (`rbd_steps_per_frame` solver steps) into a CUDA graph, executing it
    /// once. Subsequent `replay_cuda_graph` calls replay the whole sequence
    /// with a single `cuGraphLaunch` — the fast path for capture/eval loops.
    ///
    /// Returns `False` when the backend is not CUDA. Call after the scene is
    /// finalized and a few warmup `simulate` calls (the graph records raw
    /// buffer addresses; buffer growth after capture invalidates it).
    #[cfg(feature = "cuda")]
    fn capture_cuda_graph(
        &mut self,
        viewer: PyRef<NexusViewer>,
        mut state: PyRefMut<NexusState>,
    ) -> PyResult<bool> {
        pollster::block_on(self.0.capture_rbd_graph(viewer.backend(), &mut state.0))
            .map_err(gpu_err)
    }

    /// Windowless `capture_cuda_graph`: same contract, on a viewerless backend.
    /// Call after `finalize_headless` and a few warmup `simulate_headless`
    /// calls; the graph freezes buffer addresses and the coloring loop, and
    /// replay skips `auto_resize_buffers`, so the scene must have settled.
    #[cfg(feature = "cuda")]
    fn capture_cuda_graph_headless(
        &mut self,
        backend: PyRef<NexusBackend>,
        mut state: PyRefMut<NexusState>,
    ) -> PyResult<bool> {
        pollster::block_on(self.0.capture_rbd_graph(&backend.0, &mut state.0)).map_err(gpu_err)
    }

    /// Replays the captured rigid-body CUDA graph (see `capture_cuda_graph`).
    /// Returns `False` when no graph has been captured.
    #[cfg(feature = "cuda")]
    fn replay_cuda_graph(&mut self) -> PyResult<bool> {
        self.0.replay_rbd_graph().map_err(gpu_err)
    }

    /// Enables per-collider-pair contact reduction: all manifolds a pair emits
    /// (a trimesh emits one per touched triangle) are merged into one manifold
    /// of the deepest `MAX_MANIFOLD_POINTS` points before the solvers run.
    /// Off upstream; without it a foot over a fine terrain mesh keeps whatever
    /// 4 points the BVH traversal emitted first and sinks through the surface.
    fn set_contact_reduction(&mut self, backend: PyRef<NexusBackend>, enabled: bool) -> PyResult<()> {
        self.0
            .preload_pipelines(&backend.0, nexus3d::pipeline::NexusPipelineMask::RBD)
            .map_err(gpu_err)?;
        if let Some(p) = self.0.rbd_pipeline.as_mut() {
            p.contact_reduction = enabled;
        }
        Ok(())
    }

    /// Compiles all GPU pipelines up-front on a viewerless backend.
    fn preload_pipelines_headless(&mut self, backend: PyRef<NexusBackend>) -> PyResult<()> {
        self.0
            .preload_pipelines(&backend.0, nexus3d::pipeline::NexusPipelineMask::all())
            .map_err(gpu_err)
    }

    /// Advances the simulation by one frame on a viewerless backend.
    #[pyo3(signature = (backend, state, timestamps=None))]
    fn simulate_headless(
        &mut self,
        backend: PyRef<NexusBackend>,
        mut state: PyRefMut<NexusState>,
        mut timestamps: Option<PyRefMut<GpuTimestamps>>,
    ) -> PyResult<()> {
        let ts = timestamps.as_deref_mut().map(|t| &mut t.0);
        pollster::block_on(self.0.simulate(&backend.0, &mut state.0, ts)).map_err(gpu_err)
    }

    /// Advances the simulation by one frame. Blocks on the async GPU work.
    #[pyo3(signature = (viewer, state, timestamps=None))]
    fn simulate(
        &mut self,
        viewer: PyRef<NexusViewer>,
        mut state: PyRefMut<NexusState>,
        mut timestamps: Option<PyRefMut<GpuTimestamps>>,
    ) -> PyResult<()> {
        let backend = viewer.backend();
        let ts = timestamps.as_deref_mut().map(|t| &mut t.0);
        pollster::block_on(self.0.simulate(backend, &mut state.0, ts)).map_err(gpu_err)
    }
}
