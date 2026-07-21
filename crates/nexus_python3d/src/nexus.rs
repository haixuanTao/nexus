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
}

/// The GPU-resident state of a multiphysics simulation
/// (`nexus3d::prelude::NexusState`). The second field keeps the
/// `rapier3d-mjcf` robot handles of the last `insert_mjcf`, so
/// `apply_actuator_controls` can drive the robot's actuators per step.
#[pyclass(name = "NexusState", unsendable)]
pub struct NexusState(pub RNexusState, pub Option<crate::loaders::MjcfHandles>);

#[pymethods]
impl NexusState {
    #[new]
    fn new() -> Self {
        NexusState(RNexusState::default(), None)
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
    #[pyo3(signature = (scene_path, env=0))]
    fn insert_mjcf_headless(
        &mut self,
        scene_path: std::path::PathBuf,
        env: usize,
    ) -> PyResult<MjcfSceneInfo> {
        let (info, handles) = crate::loaders::insert_mjcf_headless(&mut self.0, &scene_path, env)?;
        if env == 0 {
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

    /// Replays the captured rigid-body CUDA graph (see `capture_cuda_graph`).
    /// Returns `False` when no graph has been captured.
    #[cfg(feature = "cuda")]
    fn replay_cuda_graph(&mut self) -> PyResult<bool> {
        self.0.replay_rbd_graph().map_err(gpu_err)
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
