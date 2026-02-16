use crate::{PhysicsState, RunState, Stage};
use khal::backend::Backend;
use nexus_mpm::solver::{GpuParticleModelData, SimulationParams};

pub use nexus_mpm::solver::prep_readback::{
    GpuReadbackData, ReadbackData, RenderConfig, WgPrepReadback,
};

#[derive(Default)]
pub struct SimulationTimes {
    pub total_step_time: f32,
    pub encoding_time: f32,
    pub readback_time: f32,
    /// Per-pass GPU timings aggregated by label (ordered).
    pub gpu_pass_times: Vec<(String, f64)>,
    /// Total GPU time across all passes.
    pub gpu_total_time: f64,
}

#[derive(Default)]
pub struct SimulationStepResult {
    pub instances: Vec<ReadbackData>,
    pub rigid_instances: Vec<ReadbackData>,
    pub timings: SimulationTimes,
}

impl<GpuModel: GpuParticleModelData> Stage<GpuModel> {
    pub async fn step_simulation(&mut self) -> bool {
        if self.app_state.run_state == RunState::Paused {
            return false;
        }

        let physics = &mut self.physics;
        let prev_particle_count = physics.data.particles.len();

        // Run callbacks.
        for callback in &mut physics.callbacks {
            let mut phx = PhysicsState {
                backend: &self.gpu,
                data: &mut physics.data,
                results: &self.step_result,
                step_id: self.step_id,
            };
            callback.update(&mut phx);
        }

        // Check if particle count changed.
        let new_particle_count = physics.data.particles.len();
        let num_rigid_particles = physics.data.rigid_particles.len() as usize;
        if prev_particle_count != new_particle_count {
            self.readback
                .resize(
                    &self.gpu,
                    new_particle_count,
                    num_rigid_particles,
                    self.app_state.render_mode as u32,
                )
                .unwrap();
            self.step_result
                .instances
                .resize(new_particle_count, ReadbackData::default());
        }

        let t_total = web_time::Instant::now();
        let base_dt = physics.data.base_dt;
        let prev_num_substeps = self.app_state.num_substeps;

        if self.app_state.min_num_substeps < self.app_state.max_num_substeps {
            // Adaptive stepping.
            let bounds = self
                .app_state
                .pipeline
                .timestep_bounds
                .compute_bounds(
                    &self.gpu,
                    Some(&mut self.timestamps),
                    &physics.data.grid,
                    &physics.data.particles,
                    &mut physics.data.timestep_bounds,
                    &mut physics.data.timestep_bounds_staging,
                )
                .await
                .unwrap();

            let num_substeps_estimated = (base_dt / bounds).ceil() as u32;
            let num_substeps = num_substeps_estimated.clamp(
                self.app_state.min_num_substeps,
                self.app_state.max_num_substeps,
            );
            self.app_state.num_substeps = num_substeps;
        } else if self.app_state.num_substeps != self.app_state.max_num_substeps {
            self.app_state.num_substeps = self.app_state.max_num_substeps;
        }

        if prev_num_substeps != self.app_state.num_substeps {
            let gravity = physics.data.gravity;
            let params = SimulationParams {
                gravity,
                dt: base_dt / self.app_state.num_substeps as f32,
                #[cfg(feature = "dim2")]
                padding: 0.0,
            };
            self.gpu
                .write_buffer(physics.data.sim_params.params.buffer_mut(), 0, &[params])
                .unwrap();
        }

        let t_encoding = web_time::Instant::now();
        self.timestamps.reset();

        // Run substeps.
        let mut no_state = Box::new(());
        let hooks_state = physics.hooks_state.as_deref_mut().unwrap_or(&mut no_state);
        for _ in 0..self.app_state.num_substeps {
            let mut encoder = self.gpu.begin_encoding();
            self.app_state
                .pipeline
                .launch_step(
                    &self.gpu,
                    &mut encoder,
                    &mut physics.data,
                    Some(&mut self.timestamps),
                    &mut *self.hooks,
                    hooks_state,
                )
                .await
                .unwrap();
            self.gpu.submit(encoder).unwrap();
        }

        let mut encoder = self.gpu.begin_encoding();
        // Prepare readback data on GPU and copy to staging.
        let mut encoder = self.gpu.begin_encoding();
        self.readback_shader
            .launch(
                &mut encoder,
                Some(&mut self.timestamps),
                &mut self.readback,
                &physics.data.sim_params,
                &physics.data.grid,
                &physics.data.particles,
                &physics.data.rigid_particles,
            )
            .unwrap();

        // Resolve timestamps before submitting.
        self.timestamps.resolve(&mut encoder);

        self.gpu.submit(encoder).unwrap();
        let t_encoding = t_encoding.elapsed().as_secs_f32() * 1000.0;

        self.gpu.synchronize().unwrap();
        let t_total_step = t_total.elapsed().as_secs_f32() * 1000.0;

        // Read back timestamp results.
        let (gpu_pass_times, gpu_total_time) = if let Ok(results) =
            self.timestamps.read(&self.gpu).await
        {
            // Aggregate by label.
            let mut aggregated: Vec<(String, f64)> = vec![];
            for r in &results {
                if let Some(existing) = aggregated.iter_mut().find(|(label, _)| label == &r.label) {
                    existing.1 += r.duration_ms;
                } else {
                    aggregated.push((r.label.clone(), r.duration_ms));
                }
            }
            let total = aggregated.iter().map(|e| e.1).sum();
            (aggregated, total)
        } else {
            (Vec::new(), 0.0)
        };

        // Read back readback data.
        let t_readback = web_time::Instant::now();
        self.gpu
            .read_buffer(
                self.readback.instances_staging.buffer(),
                self.step_result.instances.as_mut_slice(),
            )
            .await
            .unwrap();

        // Read back rigid particle readback data.
        if num_rigid_particles > 0 {
            self.gpu
                .read_buffer(
                    self.readback.rigid_instances_staging.buffer(),
                    self.step_result.rigid_instances.as_mut_slice(),
                )
                .await
                .unwrap();
        }
        let t_readback = t_readback.elapsed().as_secs_f32() * 1000.0;

        // Step rapier to update kinematic bodies.
        let rapier = &mut self.physics.rapier_data;
        rapier.physics_pipeline.step(
            nexus::math::Vector::ZERO,
            &rapier.params,
            &mut rapier.islands,
            &mut rapier.broad_phase,
            &mut rapier.narrow_phase,
            &mut rapier.bodies,
            &mut rapier.colliders,
            &mut rapier.impulse_joints,
            &mut rapier.multibody_joints,
            &mut rapier.ccd_solver,
            &(),
            &(),
        );

        if self.app_state.run_state == RunState::Step {
            self.app_state.run_state = RunState::Paused;
        }

        self.step_result.timings = SimulationTimes {
            total_step_time: t_total_step,
            encoding_time: t_encoding,
            readback_time: t_readback,
            gpu_pass_times,
            gpu_total_time,
        };
        self.step_id += 1;

        true
    }
}
