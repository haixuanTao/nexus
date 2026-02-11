use crate::{PhysicsState, RenderMode, RunState, Stage};
use glamx::Vec4;
use khal::backend::{Backend, GpuBackend, GpuBackendError};
use khal::BufferUsages;
use nexus_mpm::mpm_shaders::solver::particle::{Dynamics, Position};
use nexus_mpm::solver::{GpuParticleModelData, SimulationParams};
use vortx::tensor::Tensor;


#[cfg(feature = "dim2")]
#[derive(Default, Copy, Clone, Debug)]
#[repr(C)]
pub struct ReadbackData {
    pub color: Vec4,
    pub deformation: glamx::Mat2,
    pub position: glamx::Vec2,
}

#[cfg(feature = "dim3")]
#[derive(Default, Copy, Clone, Debug)]
#[repr(C)]
pub struct ReadbackData {
    pub color: Vec4,
    pub deformation: glamx::Mat3,
    pub position: glamx::Vec3,
}

#[derive(Default)]
pub struct SimulationTimes {
    pub total_step_time: f32,
    pub encoding_time: f32,
    pub readback_time: f32,
}

#[derive(Default)]
pub struct SimulationStepResult {
    pub instances: Vec<ReadbackData>,
    pub timings: SimulationTimes,
}

pub struct ReadbackBuffers {
    pub positions_staging: Tensor<Position>,
    pub dynamics_staging: Tensor<Dynamics>,
    base_colors: Vec<Vec4>,
}

impl ReadbackBuffers {
    pub fn new(backend: &GpuBackend, num_particles: usize) -> Result<Self, GpuBackendError> {
        let palette = [
            Vec4::new(124.0 / 255.0, 144.0 / 255.0, 1.0, 1.0),
            Vec4::new(8.0 / 255.0, 144.0 / 255.0, 1.0, 1.0),
            Vec4::new(124.0 / 255.0, 7.0 / 255.0, 1.0, 1.0),
            Vec4::new(124.0 / 255.0, 144.0 / 255.0, 7.0 / 255.0, 1.0),
            Vec4::new(200.0 / 255.0, 37.0 / 255.0, 1.0, 1.0),
            Vec4::new(124.0 / 255.0, 230.0 / 255.0, 25.0 / 255.0, 1.0),
        ];
        let base_colors: Vec<_> = (0..num_particles)
            .map(|i| palette[i % palette.len()])
            .collect();

        Ok(Self {
            positions_staging: Tensor::vector_uninit(
                backend,
                num_particles as u32,
                BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            )?,
            dynamics_staging: Tensor::vector_uninit(
                backend,
                num_particles as u32,
                BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            )?,
            base_colors,
        })
    }

    pub fn resize(
        &mut self,
        backend: &GpuBackend,
        num_particles: usize,
    ) -> Result<(), GpuBackendError> {
        let palette = [
            Vec4::new(124.0 / 255.0, 144.0 / 255.0, 1.0, 1.0),
            Vec4::new(8.0 / 255.0, 144.0 / 255.0, 1.0, 1.0),
            Vec4::new(124.0 / 255.0, 7.0 / 255.0, 1.0, 1.0),
            Vec4::new(124.0 / 255.0, 144.0 / 255.0, 7.0 / 255.0, 1.0),
            Vec4::new(200.0 / 255.0, 37.0 / 255.0, 1.0, 1.0),
            Vec4::new(124.0 / 255.0, 230.0 / 255.0, 25.0 / 255.0, 1.0),
        ];
        self.base_colors = (0..num_particles)
            .map(|i| palette[i % palette.len()])
            .collect();
        self.positions_staging = Tensor::vector_uninit(
            backend,
            num_particles as u32,
            BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        )?;
        self.dynamics_staging = Tensor::vector_uninit(
            backend,
            num_particles as u32,
            BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        )?;
        Ok(())
    }
}

/// Compute singular values of a 2x2 matrix.
#[cfg(feature = "dim2")]
fn singular_values_2x2(m: glamx::Mat2) -> glamx::Vec2 {
    // Singular values from eigenvalues of M^T * M.
    let mtm = m.transpose() * m;
    let a = mtm.col(0).x;
    let b = mtm.col(1).x; // = mtm.col(0).y
    let c = mtm.col(1).y;
    let avg = (a + c) * 0.5;
    let diff = ((a - c) * 0.5).hypot(b);
    let s1 = (avg + diff).max(0.0).sqrt();
    let s2 = (avg - diff).max(0.0).sqrt();
    glamx::Vec2::new(s1, s2)
}

/// Compute singular values of a 3x3 matrix.
#[cfg(feature = "dim3")]
fn singular_values_3x3(m: glamx::Mat3) -> glamx::Vec3 {
    // Eigenvalues of M^T * M via characteristic polynomial.
    let mtm = m.transpose() * m;
    let a11 = mtm.col(0).x;
    let a12 = mtm.col(1).x;
    let a13 = mtm.col(2).x;
    let a22 = mtm.col(1).y;
    let a23 = mtm.col(2).y;
    let a33 = mtm.col(2).z;

    // Coefficients of characteristic polynomial: -lambda^3 + c2*lambda^2 + c1*lambda + c0 = 0
    let c2 = a11 + a22 + a33; // trace
    let c1 = a12 * a12 + a13 * a13 + a23 * a23 - a11 * a22 - a11 * a33 - a22 * a33;
    let c0 = a11 * a22 * a33 + 2.0 * a12 * a13 * a23 - a11 * a23 * a23 - a22 * a13 * a13
        - a33 * a12 * a12; // determinant

    // Solve using Cardano's method for symmetric positive semi-definite matrix.
    let p = c2 * c2 + 3.0 * c1;
    let q = -(2.0 * c2 * c2 * c2 + 9.0 * c1 * c2 - 27.0 * c0);

    let p = (p / 9.0).max(0.0);
    let disc = (q * q - 4.0 * p * p * p).max(0.0);

    let (e1, e2, e3) = if p < 1e-10 {
        let v = c2 / 3.0;
        (v, v, v)
    } else {
        let phi = (q / (2.0 * p * p.sqrt())).clamp(-1.0, 1.0).acos();
        let sqrt_p = p.sqrt();
        let base = c2 / 3.0;
        (
            base + 2.0 * sqrt_p * (phi / 3.0).cos(),
            base + 2.0 * sqrt_p * ((phi - std::f32::consts::TAU) / 3.0).cos(),
            base + 2.0 * sqrt_p * ((phi + std::f32::consts::TAU) / 3.0).cos(),
        )
    };

    glamx::Vec3::new(e1.max(0.0).sqrt(), e2.max(0.0).sqrt(), e3.max(0.0).sqrt())
}

fn compute_readback_data(
    positions: &[Position],
    dynamics: &[Dynamics],
    base_colors: &[Vec4],
    mode: RenderMode,
    cell_width: f32,
    dt: f32,
) -> Vec<ReadbackData> {
    positions
        .iter()
        .zip(dynamics.iter())
        .zip(base_colors.iter())
        .map(|((pos, dyn_), base_color)| {
            let r = dyn_.init_radius;
            let def_grad = dyn_.def_grad;

            // Clamp deformation gradient to [-4, 4] and scale by init diameter.
            #[cfg(feature = "dim2")]
            let deformation = {
                let init_def = glamx::Mat2::from_diagonal(glamx::Vec2::splat(r * 2.0));
                let clamped = glamx::Mat2::from_cols(
                    def_grad.col(0).clamp(glamx::Vec2::splat(-4.0), glamx::Vec2::splat(4.0)),
                    def_grad.col(1).clamp(glamx::Vec2::splat(-4.0), glamx::Vec2::splat(4.0)),
                );
                init_def * clamped
            };
            #[cfg(feature = "dim3")]
            let deformation = {
                use glamx::Vec4Swizzles;
                let init_def = glamx::Mat3::from_diagonal(glamx::Vec3::splat(r * 2.0));
                let clamped = glamx::Mat3::from_cols(
                    def_grad.col(0).xyz().clamp(glamx::Vec3::splat(-4.0), glamx::Vec3::splat(4.0)),
                    def_grad.col(1).xyz().clamp(glamx::Vec3::splat(-4.0), glamx::Vec3::splat(4.0)),
                    def_grad.col(2).xyz().clamp(glamx::Vec3::splat(-4.0), glamx::Vec3::splat(4.0)),
                );
                init_def * clamped
            };

            let color = match mode {
                RenderMode::Default => *base_color,
                RenderMode::Velocity => {
                    let vel = dyn_.velocity;
                    #[cfg(feature = "dim2")]
                    {
                        let c = glamx::Vec2::new(vel.x.abs(), vel.y.abs()) * dt * 100.0
                            + glamx::Vec2::splat(0.2);
                        Vec4::new(c.x, c.y, base_color.z, base_color.w)
                    }
                    #[cfg(feature = "dim3")]
                    {
                        let c = glamx::Vec3::new(vel.x.abs(), vel.y.abs(), vel.z.abs()) * dt
                            * 100.0
                            + glamx::Vec3::splat(0.2);
                        Vec4::new(c.x, c.y, c.z, base_color.w)
                    }
                }
                RenderMode::Volume => {
                    #[cfg(feature = "dim2")]
                    {
                        let sv = singular_values_2x2(def_grad);
                        let c = (glamx::Vec2::ONE - sv) / 0.005 + glamx::Vec2::splat(0.2);
                        Vec4::new(c.x, c.y, base_color.z, base_color.w)
                    }
                    #[cfg(feature = "dim3")]
                    {
                        use crate::nexus_mpm::mpm_shaders::PaddingExt;
                        let sv = singular_values_3x3(def_grad.remove_padding());
                        let c = (glamx::Vec3::ONE - sv) / 0.005 + glamx::Vec3::splat(0.2);
                        Vec4::new(c.x, c.y, c.z, base_color.w)
                    }
                }
                RenderMode::Phase => {
                    let phase = dyn_.phase;
                    Vec4::new(0.0, 0.4 * phase, 0.4 * (1.0 - phase), base_color.w)
                }
                RenderMode::CdfNormals => {
                    let normal = dyn_.cdf.normal;
                    #[cfg(feature = "dim2")]
                    {
                        if normal == glamx::Vec2::ZERO {
                            Vec4::new(0.0, 0.0, 0.0, base_color.w)
                        } else {
                            let n = (normal + glamx::Vec2::ONE) * 0.5;
                            Vec4::new(n.x, n.y, 0.0, base_color.w)
                        }
                    }
                    #[cfg(feature = "dim3")]
                    {
                        if normal == glamx::Vec3::ZERO {
                            Vec4::new(0.0, 0.0, 0.0, base_color.w)
                        } else {
                            let n = (normal + glamx::Vec3::ONE) * 0.5;
                            Vec4::new(n.x, n.y, n.z, base_color.w)
                        }
                    }
                }
                RenderMode::CdfDistances => {
                    let d = dyn_.cdf.signed_distance / (cell_width * 1.5);
                    if d > 0.0 {
                        Vec4::new(0.0, d.abs(), 0.0, base_color.w)
                    } else {
                        Vec4::new(d.abs(), 0.0, 0.0, base_color.w)
                    }
                }
                RenderMode::CdfSigns => {
                    let d = dyn_.cdf.affinity;
                    let a = (d >> 16) & (d & 0x0000ffff);
                    if d == 0 {
                        Vec4::new(0.0, 0.0, 0.0, base_color.w)
                    } else if a == 0 {
                        Vec4::new(0.0, 1.0, 0.0, base_color.w)
                    } else {
                        Vec4::new(1.0, 0.0, 0.0, base_color.w)
                    }
                }
            };

            // In 3D, mark disabled (failed) particles red.
            #[cfg(feature = "dim3")]
            let color = if dyn_.enabled == 0 {
                Vec4::new(1.0, 0.0, 0.0, 1.0)
            } else {
                color
            };

            ReadbackData {
                position: pos.pt,
                color,
                deformation,
            }
        })
        .collect()
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
        if prev_particle_count != new_particle_count {
            self.readback
                .resize(&self.gpu, new_particle_count)
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
        let mut encoder = self.gpu.begin_encoding();

        // Run substeps.
        let mut no_state = Box::new(());
        let hooks_state = physics.hooks_state.as_deref_mut().unwrap_or(&mut no_state);
        for _ in 0..self.app_state.num_substeps {
            self.app_state
                .pipeline
                .launch_step(
                    &self.gpu,
                    &mut encoder,
                    &mut physics.data,
                    &mut *self.hooks,
                    hooks_state,
                )
                .await
                .unwrap();
        }

        // Copy particle data to staging buffers for readback.
        self.readback
            .positions_staging
            .copy_from_view(&mut encoder, physics.data.particles.positions())
            .unwrap();
        self.readback
            .dynamics_staging
            .copy_from_view(&mut encoder, physics.data.particles.dynamics())
            .unwrap();

        self.gpu.submit(encoder).unwrap();
        let t_encoding = t_encoding.elapsed().as_secs_f32() * 1000.0;

        self.gpu.synchronize().unwrap();
        let t_total_step = t_total.elapsed().as_secs_f32() * 1000.0;

        // Read back particle data.
        let t_readback = web_time::Instant::now();
        let mut positions_cpu = vec![Position::default(); new_particle_count];
        let mut dynamics_cpu = vec![Dynamics::default(); new_particle_count];
        self.gpu
            .read_buffer(
                self.readback.positions_staging.buffer(),
                positions_cpu.as_mut_slice(),
            )
            .await
            .unwrap();
        self.gpu
            .read_buffer(
                self.readback.dynamics_staging.buffer(),
                dynamics_cpu.as_mut_slice(),
            )
            .await
            .unwrap();

        // Compute render data on CPU.
        let cell_width = physics.data.grid.cpu_meta.cell_width;
        let dt = base_dt / self.app_state.num_substeps.max(1) as f32;
        self.step_result.instances = compute_readback_data(
            &positions_cpu,
            &dynamics_cpu,
            &self.readback.base_colors,
            self.app_state.render_mode,
            cell_width,
            dt,
        );
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
        };
        self.step_id += 1;

        true
    }
}
