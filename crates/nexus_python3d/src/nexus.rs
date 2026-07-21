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
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
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

    /// Loads a MuJoCo MJCF scene into environment `env` as multibodies. For
    /// `env == 0` its render shapes (and a sized floor) are registered with
    /// `viewer`; batch environments are physics-only (the viewer draws env 0).
    /// Returns scene info (suggested camera + whether the scene is Z-up).
    /// Call `finalize` after.
    /// Reads the current simulation parameters (environment 0) as a dict.
    ///
    /// Keys match the `set_sim_params` keywords. `dt` is the outer timestep, not
    /// the per-substep one the GPU consumes.
    fn sim_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let Some(sp) = self.0.rbd_sim_params().first() else {
            return Ok(d);
        };
        d.set_item("dt", sp.dt)?;
        d.set_item("num_solver_iterations", sp.num_solver_iterations)?;
        d.set_item("gravity", (sp.gravity_x, sp.gravity_y, sp.gravity_z))?;
        d.set_item("contact_natural_frequency", sp.contact_natural_frequency)?;
        d.set_item("contact_damping_ratio", sp.contact_damping_ratio)?;
        d.set_item("joint_natural_frequency", sp.joint_natural_frequency)?;
        d.set_item("joint_damping_ratio", sp.joint_damping_ratio)?;
        d.set_item("warmstart_coefficient", sp.warmstart_coefficient)?;
        d.set_item("length_unit", sp.length_unit)?;
        d.set_item(
            "normalized_allowed_linear_error",
            sp.normalized_allowed_linear_error,
        )?;
        d.set_item(
            "normalized_max_corrective_velocity",
            sp.normalized_max_corrective_velocity,
        )?;
        d.set_item(
            "normalized_prediction_distance",
            sp.normalized_prediction_distance,
        )?;
        Ok(d)
    }

    /// Overrides simulation parameters for every environment. Every argument is
    /// optional; `None` (the default) leaves that parameter untouched, so this
    /// never changes behaviour unless a value is passed.
    ///
    /// Defaults, from rapier's TGS-soft configuration:
    ///   dt=1/60, num_solver_iterations=4, gravity=(0, -9.81, 0),
    ///   contact_natural_frequency=30, contact_damping_ratio=5,
    ///   joint_natural_frequency=1e6, joint_damping_ratio=1,
    ///   warmstart_coefficient=1, length_unit=1,
    ///   normalized_allowed_linear_error=0.001,
    ///   normalized_max_corrective_velocity=10,
    ///   normalized_prediction_distance=0.002
    ///
    /// `dt` matters for benchmarks: `simulate` advances one outer `dt`, which is
    /// 1/60 by default no matter what timestep the scene's MJCF declares — the
    /// importer does not read it. Set it explicitly to compare against engines
    /// stepping at a different rate.
    ///
    /// `gravity` also updates the multibody gravity, so it stays equivalent to
    /// `set_rbd_gravity`. Safe before or after `finalize`; `num_solver_iterations`
    /// is the exception — it fixes a dispatch count at `finalize`, so changing it
    /// afterwards rescales the substep dt without changing iterations actually run.
    #[pyo3(signature = (viewer, *, dt=None, num_solver_iterations=None, gravity=None,
                        contact_natural_frequency=None, contact_damping_ratio=None,
                        joint_natural_frequency=None, joint_damping_ratio=None,
                        warmstart_coefficient=None, length_unit=None,
                        normalized_allowed_linear_error=None,
                        normalized_max_corrective_velocity=None,
                        normalized_prediction_distance=None))]
    #[allow(clippy::too_many_arguments)]
    fn set_sim_params(
        &mut self,
        viewer: PyRef<NexusViewer>,
        dt: Option<f32>,
        num_solver_iterations: Option<u32>,
        gravity: Option<(f32, f32, f32)>,
        contact_natural_frequency: Option<f32>,
        contact_damping_ratio: Option<f32>,
        joint_natural_frequency: Option<f32>,
        joint_damping_ratio: Option<f32>,
        warmstart_coefficient: Option<f32>,
        length_unit: Option<f32>,
        normalized_allowed_linear_error: Option<f32>,
        normalized_max_corrective_velocity: Option<f32>,
        normalized_prediction_distance: Option<f32>,
    ) -> PyResult<()> {
        if let Some(dt) = dt {
            if !(dt > 0.0) {
                return Err(PyValueError::new_err(format!("dt must be > 0 (got {dt})")));
            }
        }
        if let Some(n) = num_solver_iterations {
            if n == 0 {
                return Err(PyValueError::new_err("num_solver_iterations must be >= 1"));
            }
        }
        let backend = viewer.backend();
        self.0.update_rbd_sim_params(backend, |sp| {
            if let Some(v) = dt {
                sp.dt = v;
            }
            if let Some(v) = num_solver_iterations {
                sp.num_solver_iterations = v;
            }
            if let Some((x, y, z)) = gravity {
                sp.gravity_x = x;
                sp.gravity_y = y;
                sp.gravity_z = z;
            }
            if let Some(v) = contact_natural_frequency {
                sp.contact_natural_frequency = v;
            }
            if let Some(v) = contact_damping_ratio {
                sp.contact_damping_ratio = v;
            }
            if let Some(v) = joint_natural_frequency {
                sp.joint_natural_frequency = v;
            }
            if let Some(v) = joint_damping_ratio {
                sp.joint_damping_ratio = v;
            }
            if let Some(v) = warmstart_coefficient {
                sp.warmstart_coefficient = v;
            }
            if let Some(v) = length_unit {
                sp.length_unit = v;
            }
            if let Some(v) = normalized_allowed_linear_error {
                sp.normalized_allowed_linear_error = v;
            }
            if let Some(v) = normalized_max_corrective_velocity {
                sp.normalized_max_corrective_velocity = v;
            }
            if let Some(v) = normalized_prediction_distance {
                sp.normalized_prediction_distance = v;
            }
        });
        // Keep the multibody gravity in step with the free-body one.
        if let Some((x, y, z)) = gravity {
            self.0.set_rbd_gravity(backend, [x, y, z]);
        }
        Ok(())
    }

    /// Blocking readback of GPU body poses as `(x, y, z, qx, qy, qz, qw)`,
    /// indexed by `gpu_id` (bodies of every environment, concatenated).
    ///
    /// Unlike `rapier_debug_bodies`, which reads the CPU-side staging world and
    /// therefore always reports the *initial* pose, this observes what the GPU
    /// solver actually computed. Stalls the pipeline — use it to check results,
    /// not inside a timed loop.
    fn rbd_body_poses(&mut self, viewer: PyRef<NexusViewer>) -> Vec<(f32, f32, f32, f32, f32, f32, f32)> {
        self.0
            .rbd_read_body_poses(viewer.backend())
            .into_iter()
            .map(|p| {
                let t = p.translation;
                let r = p.rotation;
                (t.x, t.y, t.z, r.x, r.y, r.z, r.w)
            })
            .collect()
    }

    /// Solver color passes per stage, and the coloring iteration count.
    ///
    /// The first value is `max_colors + 1` — the configured *capacity*, not a
    /// count of colors the scene actually needed. The solver walks it
    /// sequentially (`for _ in 0..num_colors`, per stage, per solver iteration),
    /// so it is a direct multiplier on dispatches per step regardless of how
    /// many colors the contact graph really uses. Pair with `set_max_colors`.
    ///
    /// The second value is always 0: `coloring_iterations` is declared on
    /// `RunStats` but never assigned by the pipeline.
    /// Per-pass GPU times from the last `simulate` as `[(label, ms), ...]`,
    /// plus the total, e.g. `state.gpu_pass_times()`. Empty unless that
    /// `simulate` call was given a `GpuTimestamps` — timing queries are only
    /// encoded when one is passed.
    fn gpu_pass_times(&self) -> (Vec<(String, f64)>, f64) {
        (
            self.0.run_stats.gpu_pass_times.clone(),
            self.0.run_stats.gpu_total_time_ms,
        )
    }

    fn solver_color_passes(&self) -> (u32, u32) {
        (
            self.0.run_stats.num_colors,
            self.0.run_stats.coloring_iterations,
        )
    }

    /// Cap the solver's sequential color passes (default 8, i.e. 9 passes).
    ///
    /// The solver walks `0..max_colors + 1` per stage per substep, over this
    /// *capacity* rather than the colors the contact graph actually needs — so
    /// a scene simpler than the default pays for passes it never uses. A single
    /// free cube needs 2 and runs ~2x faster set to 2; an articulated robot
    /// needs ~7, where the default is already about right.
    ///
    /// This is a performance knob, not a correctness one. Setting it too low
    /// makes the bounded coloring fail to converge, and the default `Grow`
    /// policy then adds 5 until it does (rbd_step.rs), converging on identical
    /// physics — measured bit-identical across 1..=8 on a 12-DOF robot. The
    /// costs of under-provisioning are the overshoot (asking for 4 can settle
    /// at 15 passes, worse than asking for 8) and a transient window during the
    /// first frames, before the readback-driven growth catches up.
    ///
    /// Call before `finalize`.
    fn set_max_colors(&mut self, max_colors: u32) {
        self.0.set_rbd_solver_colors(max_colors);
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

    /// Overrides the per-environment collision-pair capacity (default 4096)
    /// used when the GPU state is allocated at `finalize`. Lower it for many
    /// small batched envs; pair-keyed buffers scale as capacity × envs.
    /// TEMPORARY: dof_state raw readback.
    fn dbg_dof_state(&self, viewer: PyRef<NexusViewer>) -> Vec<f32> {
        self.0.rbd_dbg_dof_state(viewer.backend())
    }

    /// TEMPORARY: links_static raw readback.
    fn dbg_links_static(&self, viewer: PyRef<NexusViewer>) -> Vec<f32> {
        self.0.rbd_dbg_links_static(viewer.backend())
    }

    /// TEMPORARY inert-motor diagnostic (see NexusState::rbd_dbg_joint_constraints).
    fn dbg_joint_constraints(&self, viewer: PyRef<NexusViewer>) -> Vec<f32> {
        self.0.rbd_dbg_joint_constraints(viewer.backend())
    }

    fn set_rbd_collisions_capacity(&mut self, capacity: u32) {
        self.0.set_rbd_collisions_capacity(capacity);
    }

    fn set_rbd_gravity(&mut self, viewer: PyRef<NexusViewer>, gravity: Vec3) {
        self.0
            .set_rbd_gravity(viewer.backend(), [gravity.0.x, gravity.0.y, gravity.0.z]);
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
