use crate::mpm::step::RenderConfig;
use crate::mpm::RenderMode;
use crate::rbd::BackendType;
use crate::{ActiveDemo, DemoBuilder, RunState, UiSections};
use khal::backend::{Backend, GpuBackend as KhalGpuBackend};
use kiss3d::egui;
use kiss3d::window::Window;
use nexus::rbd::pipeline::RunStats;

use egui::{CollapsingHeader, Color32, ComboBox, CornerRadius, RichText, Stroke};

#[derive(Default, Copy, Clone)]
pub struct UiInteractions {
    pub new_selected_demo: Option<usize>,
}

/// Sets up a custom warm theme that complements the app's off-white background.
fn setup_custom_theme(ctx: &egui::Context) {
    let bg_fill = Color32::from_rgb(250, 250, 245);
    let window_fill = Color32::from_rgb(252, 252, 248);
    let faint_bg = Color32::from_rgb(240, 240, 232);
    let extreme_bg = Color32::from_rgb(255, 255, 252);

    let text_color = Color32::from_rgb(60, 58, 52);

    let accent = Color32::from_rgb(82, 130, 150);
    let accent_active = Color32::from_rgb(70, 115, 135);

    let widget_bg = Color32::from_rgb(235, 235, 225);
    let widget_bg_hover = Color32::from_rgb(225, 225, 215);
    let widget_bg_active = Color32::from_rgb(215, 215, 205);

    let stroke_color = Color32::from_rgb(200, 198, 190);
    let stroke_hover = Color32::from_rgb(180, 178, 170);

    let rounding = CornerRadius::same(6);
    let small_rounding = CornerRadius::same(4);

    ctx.style_mut(|style| {
        let v = &mut style.visuals;
        v.dark_mode = false;

        v.widgets.noninteractive.bg_fill = faint_bg;
        v.widgets.noninteractive.weak_bg_fill = faint_bg;
        v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, stroke_color);
        v.widgets.noninteractive.corner_radius = rounding;
        v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, text_color);

        v.widgets.inactive.bg_fill = widget_bg;
        v.widgets.inactive.weak_bg_fill = widget_bg;
        v.widgets.inactive.bg_stroke = Stroke::new(1.0, stroke_color);
        v.widgets.inactive.corner_radius = small_rounding;
        v.widgets.inactive.fg_stroke = Stroke::new(1.0, text_color);

        v.widgets.hovered.bg_fill = widget_bg_hover;
        v.widgets.hovered.weak_bg_fill = widget_bg_hover;
        v.widgets.hovered.bg_stroke = Stroke::new(1.0, stroke_hover);
        v.widgets.hovered.corner_radius = small_rounding;
        v.widgets.hovered.fg_stroke = Stroke::new(1.5, text_color);

        v.widgets.active.bg_fill = widget_bg_active;
        v.widgets.active.weak_bg_fill = widget_bg_active;
        v.widgets.active.bg_stroke = Stroke::new(1.0, accent);
        v.widgets.active.corner_radius = small_rounding;
        v.widgets.active.fg_stroke = Stroke::new(2.0, accent_active);

        v.widgets.open.bg_fill = widget_bg;
        v.widgets.open.weak_bg_fill = widget_bg;
        v.widgets.open.bg_stroke = Stroke::new(1.0, stroke_color);
        v.widgets.open.corner_radius = small_rounding;
        v.widgets.open.fg_stroke = Stroke::new(1.0, text_color);

        v.selection.bg_fill = accent.gamma_multiply(0.25);
        v.selection.stroke = Stroke::new(1.0, accent);

        v.hyperlink_color = accent;
        v.faint_bg_color = faint_bg;
        v.extreme_bg_color = extreme_bg;
        v.code_bg_color = Color32::from_rgb(230, 230, 220);
        v.warn_fg_color = Color32::from_rgb(180, 120, 60);
        v.error_fg_color = Color32::from_rgb(180, 70, 70);

        v.window_corner_radius = CornerRadius::same(8);
        v.window_fill = window_fill;
        v.window_stroke = Stroke::new(1.0, stroke_color);

        v.panel_fill = bg_fill;

        v.slider_trailing_fill = true;
        v.handle_shape = egui::style::HandleShape::Circle;

        style.spacing.item_spacing = egui::vec2(6.0, 3.0);
        style.spacing.window_margin = egui::Margin::same(10);
        style.spacing.button_padding = egui::vec2(6.0, 3.0);
        style.spacing.slider_width = 130.0;
        style.spacing.indent = 14.0;
        style.spacing.interact_size = egui::vec2(32.0, 18.0);
        style.spacing.combo_width = 100.0;
    });
}

pub fn render_compiling_message(window: &mut Window) {
    window.draw_ui(|ctx| {
        setup_custom_theme(ctx);
        egui::Window::new("Nexus Testbed").show(ctx, |ui| {
            ui.colored_label(
                Color32::from_rgb(82, 130, 150),
                "Compiling shaders...\nThe app will freeze for a few seconds.\n\nIf nothing happens after a minute or two, check the dev console for an error.",
            );
        });
    });
}

#[allow(clippy::too_many_arguments)]
pub fn render_ui(
    window: &mut Window,
    builders: &[DemoBuilder],
    selected_demo: &mut usize,
    ui_sections: &mut UiSections,
    backend_type: &mut BackendType,
    use_cpu: &mut bool,
    run_state: &mut RunState,
    run_stats: &RunStats,
    active_demo: &mut ActiveDemo,
    gpu: Option<&KhalGpuBackend>,
    gpu_init_error: &Option<String>,
) -> UiInteractions {
    let mut result = UiInteractions::default();

    window.draw_ui(|ctx| {
        setup_custom_theme(ctx);

        egui::Window::new("Nexus Testbed")
            .default_width(300.0)
            .show(ctx, |ui| {
                // GPU error banner.
                if let Some(error_msg) = gpu_init_error {
                    ui.colored_label(
                        Color32::from_rgb(180, 70, 70),
                        format!("GPU: {}", error_msg),
                    );
                    ui.separator();
                }

                // Section toggles.
                ui.horizontal(|ui| {
                    ui.toggle_value(&mut ui_sections.show_performance, "Performance");
                    ui.toggle_value(&mut ui_sections.show_settings, "Settings");
                    ui.toggle_value(&mut ui_sections.show_examples, "Examples");
                });

                // Section contents.
                egui::ScrollArea::vertical()
                    .max_height(500.0)
                    .show(ui, |ui| {
                        if ui_sections.show_settings {
                            ui.separator();
                            settings_section(
                                ui,
                                builders,
                                active_demo,
                                backend_type,
                                use_cpu,
                                selected_demo,
                                gpu,
                                &mut result,
                            );
                        }

                        if ui_sections.show_performance {
                            ui.separator();
                            performance_section(ui, run_stats, backend_type, active_demo);
                        }

                        if ui_sections.show_examples {
                            ui.separator();
                            examples_section(ui, builders, selected_demo, &mut result);
                        }
                    });

                ui.separator();

                // Bottom controls.
                ui.horizontal(|ui| {
                    let (play_label, play_hover) = if *run_state == RunState::Running {
                        ("Pause", "Pause simulation (T)")
                    } else {
                        ("Play", "Start simulation (T)")
                    };

                    if ui
                        .button(play_label)
                        .on_hover_text(play_hover)
                        .clicked()
                    {
                        *run_state = if *run_state == RunState::Running {
                            RunState::Paused
                        } else {
                            RunState::Running
                        };
                    }

                    if ui
                        .button("Step")
                        .on_hover_text("Single step (S)")
                        .clicked()
                    {
                        *run_state = RunState::Step;
                    }

                    if ui
                        .button("Restart")
                        .on_hover_text("Restart example (R)")
                        .clicked()
                    {
                        result.new_selected_demo = Some(*selected_demo);
                    }
                });
            });
    });

    result
}

fn examples_section(
    ui: &mut egui::Ui,
    builders: &[DemoBuilder],
    selected_demo: &mut usize,
    result: &mut UiInteractions,
) {
    // Previous/Next navigation + current demo name.
    ui.horizontal(|ui| {
        if ui
            .add_enabled(*selected_demo > 0, egui::Button::new("<"))
            .on_hover_text("Previous example")
            .clicked()
        {
            *selected_demo -= 1;
            result.new_selected_demo = Some(*selected_demo);
        }

        if ui
            .add_enabled(*selected_demo + 1 < builders.len(), egui::Button::new(">"))
            .on_hover_text("Next example")
            .clicked()
        {
            *selected_demo += 1;
            result.new_selected_demo = Some(*selected_demo);
        }

        ui.label(RichText::new(builders[*selected_demo].name()).strong().italics());
    });

    ui.add_space(4.0);
    ui.separator();
    ui.add_space(4.0);

    // Group demos by type.
    let rbd_demos: Vec<(usize, &str)> = builders
        .iter()
        .enumerate()
        .filter_map(|(i, b)| match b {
            DemoBuilder::Rbd(name, ..) => Some((i, *name)),
            _ => None,
        })
        .collect();

    let mpm_demos: Vec<(usize, &str)> = builders
        .iter()
        .enumerate()
        .filter_map(|(i, b)| match b {
            DemoBuilder::Mpm(name, ..) => Some((i, name.as_str())),
            _ => None,
        })
        .collect();

    if !rbd_demos.is_empty() {
        CollapsingHeader::new(format!("Rigid Bodies ({})", rbd_demos.len()))
            .default_open(true)
            .show(ui, |ui| {
                for (idx, name) in &rbd_demos {
                    let is_selected = *selected_demo == *idx;
                    let text = if is_selected {
                        RichText::new(*name).strong()
                    } else {
                        RichText::new(*name)
                    };
                    if ui
                        .selectable_label(is_selected, text)
                        .on_hover_text("Click to run this example")
                        .clicked()
                        && !is_selected
                    {
                        *selected_demo = *idx;
                        result.new_selected_demo = Some(*idx);
                    }
                }
            });
    }

    if !mpm_demos.is_empty() {
        CollapsingHeader::new(format!("MPM ({})", mpm_demos.len()))
            .default_open(true)
            .show(ui, |ui| {
                for (idx, name) in &mpm_demos {
                    let is_selected = *selected_demo == *idx;
                    let text = if is_selected {
                        RichText::new(*name).strong()
                    } else {
                        RichText::new(*name)
                    };
                    if ui
                        .selectable_label(is_selected, text)
                        .on_hover_text("Click to run this example")
                        .clicked()
                        && !is_selected
                    {
                        *selected_demo = *idx;
                        result.new_selected_demo = Some(*idx);
                    }
                }
            });
    }

    let fem_demos: Vec<(usize, &str)> = builders
        .iter()
        .enumerate()
        .filter_map(|(i, b)| match b {
            DemoBuilder::Fem(name, ..) => Some((i, name.as_str())),
            _ => None,
        })
        .collect();

    if !fem_demos.is_empty() {
        CollapsingHeader::new(format!("FEM ({})", fem_demos.len()))
            .default_open(true)
            .show(ui, |ui| {
                for (idx, name) in &fem_demos {
                    let is_selected = *selected_demo == *idx;
                    let text = if is_selected {
                        RichText::new(*name).strong()
                    } else {
                        RichText::new(*name)
                    };
                    if ui
                        .selectable_label(is_selected, text)
                        .on_hover_text("Click to run this example")
                        .clicked()
                        && !is_selected
                    {
                        *selected_demo = *idx;
                        result.new_selected_demo = Some(*idx);
                    }
                }
            });
    }
}

fn settings_section(
    ui: &mut egui::Ui,
    builders: &[DemoBuilder],
    active_demo: &mut ActiveDemo,
    backend_type: &mut BackendType,
    #[cfg_attr(not(feature = "cpu"), allow(unused_variables))]
    use_cpu: &mut bool,
    selected_demo: &mut usize,
    gpu: Option<&KhalGpuBackend>,
    result: &mut UiInteractions,
) {
    match active_demo {
        ActiveDemo::Rbd { .. } => {
            rbd_settings(ui, backend_type, selected_demo, gpu, result);
        }
        ActiveDemo::Mpm {
            stage,
            colliders_gfx,
            ..
        } => {
            gpu_backend_selector(ui, backend_type, use_cpu, selected_demo, gpu, result);
            ui.add_space(4.0);
            mpm_settings(ui, builders, stage, colliders_gfx, result);
        }
        ActiveDemo::Fem { stage, .. } => {
            gpu_backend_selector(ui, backend_type, use_cpu, selected_demo, gpu, result);
            ui.add_space(4.0);
            fem_settings(ui, builders, stage, result);
        }
    }
}

fn rbd_settings(
    ui: &mut egui::Ui,
    backend_type: &mut BackendType,
    selected_demo: &mut usize,
    gpu: Option<&KhalGpuBackend>,
    result: &mut UiInteractions,
) {
    ui.label(RichText::new("Physics Backend").strong());
    ui.add_space(2.0);

    let mut backend_changed = false;

    if gpu.is_some()
        && ui
            .radio(
                matches!(*backend_type, BackendType::Gpu { .. }),
                "GPU (nexus)",
            )
            .on_hover_text("GPU-accelerated physics with nexus")
            .clicked()
        && !matches!(*backend_type, BackendType::Gpu { .. })
    {
        *backend_type = BackendType::Gpu;
        backend_changed = true;
    }

    #[cfg(feature = "cuda")]
    if ui
        .radio(*backend_type == BackendType::Cuda, "CUDA (nexus)")
        .on_hover_text("GPU-accelerated physics with nexus via CUDA")
        .clicked()
        && *backend_type != BackendType::Cuda
    {
        *backend_type = BackendType::Cuda;
        backend_changed = true;
    }

    #[cfg(feature = "cpu")]
    if ui
        .radio(*backend_type == BackendType::Cpu, "CPU (nexus)")
        .on_hover_text("CPU physics using the nexus GPU pipeline executed on CPU")
        .clicked()
        && *backend_type != BackendType::Cpu
    {
        *backend_type = BackendType::Cpu;
        backend_changed = true;
    }

    if ui
        .radio(*backend_type == BackendType::Rapier, "CPU (rapier)")
        .on_hover_text("CPU physics with rapier")
        .clicked()
        && *backend_type != BackendType::Rapier
    {
        *backend_type = BackendType::Rapier;
        backend_changed = true;
    }

    if backend_changed {
        result.new_selected_demo = Some(*selected_demo);
    }
}

/// Backend selector for MPM/FEM demos: GPU, CUDA, and optionally CPU.
fn gpu_backend_selector(
    ui: &mut egui::Ui,
    backend_type: &mut BackendType,
    #[cfg_attr(not(feature = "cpu"), allow(unused_variables))]
    use_cpu: &mut bool,
    selected_demo: &mut usize,
    gpu: Option<&KhalGpuBackend>,
    result: &mut UiInteractions,
) {
    ui.label(RichText::new("Execution Backend").strong());
    ui.add_space(2.0);

    let mut backend_changed = false;

    if gpu.is_some()
        && ui
            .radio(
                matches!(*backend_type, BackendType::Gpu { .. }) && !*use_cpu,
                "GPU",
            )
            .clicked()
        && !matches!(*backend_type, BackendType::Gpu { .. })
    {
        *backend_type = BackendType::Gpu;
        *use_cpu = false;
        backend_changed = true;
    }

    #[cfg(feature = "cuda")]
    if ui
        .radio(*backend_type == BackendType::Cuda, "CUDA")
        .clicked()
        && *backend_type != BackendType::Cuda
    {
        *backend_type = BackendType::Cuda;
        *use_cpu = false;
        backend_changed = true;
    }

    #[cfg(feature = "cpu")]
    if ui.radio(*use_cpu, "CPU").clicked() && !*use_cpu {
        *use_cpu = true;
        backend_changed = true;
    }

    if backend_changed {
        result.new_selected_demo = Some(*selected_demo);
    }
}

fn mpm_settings(
    ui: &mut egui::Ui,
    builders: &[DemoBuilder],
    stage: &mut crate::mpm::MpmStage<nexus::mpm::solver::GpuParticleModel>,
    colliders_gfx: &mut std::collections::HashMap<
        rapier::geometry::ColliderHandle,
        crate::RenderNode,
    >,
    result: &mut UiInteractions,
) {
    ui.label(RichText::new("Rendering").strong());
    ui.add_space(2.0);

    // Render mode selector.
    let prev_render_mode = stage.app_state.render_mode;
    ComboBox::from_label("Render mode")
        .selected_text(stage.app_state.render_mode.text())
        .show_ui(ui, |ui| {
            for mode in RenderMode::ALL {
                ui.selectable_value(&mut stage.app_state.render_mode, *mode, mode.text());
            }
        });

    if stage.app_state.render_mode != prev_render_mode {
        stage
            .gpu
            .write_buffer(
                stage.readback.mode.buffer_mut(),
                0,
                &[RenderConfig {
                    mode: stage.app_state.render_mode as u32,
                }],
            )
            .unwrap();
    }

    ui.checkbox(&mut stage.app_state.show_rigid_particles, "Show rigid particles")
        .on_hover_text("Display particles belonging to rigid bodies");

    ui.add_space(8.0);

    ui.label(RichText::new("Solver").strong());
    ui.add_space(2.0);

    ui.checkbox(&mut stage.app_state.use_cpic, "Use CPIC")
        .on_hover_text("Compatible Particle-In-Cell transfer");

    ui.add_space(8.0);

    // Handle MPM demo switching within MPM demos.
    if let Some(new_demo_idx) = result.new_selected_demo {
        if matches!(builders[new_demo_idx], DemoBuilder::Mpm(..)) {
            let new_name = builders[new_demo_idx].name();
            if let Some(mpm_idx) = stage
                .builders
                .iter()
                .position(|(name, _)| name == new_name)
            {
                stage.set_demo(mpm_idx);
                for (_, mut node) in colliders_gfx.drain() {
                    node.detach();
                }
            }
        }
    }
}

fn performance_section(
    ui: &mut egui::Ui,
    run_stats: &RunStats,
    backend_type: &BackendType,
    active_demo: &mut ActiveDemo,
) {
    match active_demo {
        ActiveDemo::Rbd { physics, .. } => {
            rbd_performance(ui, run_stats, backend_type, physics);
        }
        ActiveDemo::Mpm { stage, .. } => {
            mpm_performance(ui, stage);
        }
        ActiveDemo::Fem { stage, .. } => {
            fem_performance(ui, stage);
        }
    }
}

fn rbd_performance(
    ui: &mut egui::Ui,
    run_stats: &RunStats,
    backend_type: &BackendType,
    physics: &crate::rbd::PhysicsContext,
) {
    // Scene info.
    ui.label(RichText::new("Scene").strong());
    ui.add_space(2.0);

    egui::Grid::new("rbd_scene_grid")
        .num_columns(2)
        .spacing([20.0, 2.0])
        .show(ui, |ui| {
            ui.label("Bodies:");
            ui.label(format!("{}", physics.backend.num_bodies()));
            ui.end_row();

            ui.label("Joints:");
            ui.label(format!("{}", physics.backend.num_joints()));
            ui.end_row();

            ui.label("Batches:");
            ui.label(format!("{}", physics.backend.num_batches()));
            ui.end_row();
        });

    ui.add_space(8.0);
    ui.separator();
    ui.add_space(4.0);

    // Timing.
    let total_ms = run_stats.total_simulation_time_ms();
    let fps = if total_ms > 0.0 {
        (1000.0f32 / total_ms).round()
    } else {
        0.0
    };

    ui.label(RichText::new(format!("Total: {:.2}ms - {:.0} FPS", total_ms, fps)).strong());
    ui.add_space(4.0);

    if !matches!(backend_type, BackendType::Rapier) {
        CollapsingHeader::new("Simulation details")
            .id_salt("rbd_sim_details")
            .default_open(false)
            .show(ui, |ui| {
                ui.label(format!("Colors: {}", run_stats.num_colors));
                ui.label(format!(
                    "Coloring: {:.2}ms",
                    run_stats.coloring_time.as_secs_f32() * 1000.0
                ));
                ui.label(format!(
                    "Coloring iterations: {} x 10",
                    run_stats.coloring_iterations
                ));
                ui.label(format!(
                    "Start to pairs count: {:.2}ms",
                    run_stats.start_to_pairs_count_time.as_secs_f32() * 1000.0
                ));
                ui.label(format!(
                    "Coloring fallback: {:.2}ms",
                    run_stats.coloring_fallback_time.as_secs_f32() * 1000.0
                ));
            });

        if !run_stats.gpu_pass_times.is_empty() {
            CollapsingHeader::new(format!(
                "GPU passes: {:.2}ms",
                run_stats.gpu_total_time
            ))
            .id_salt("rbd_gpu_passes")
            .default_open(false)
            .show(ui, |ui| {
                egui::Grid::new("rbd_timestamp_grid")
                    .num_columns(2)
                    .spacing([20.0, 2.0])
                    .show(ui, |ui| {
                        for (label, ms) in &run_stats.gpu_pass_times {
                            ui.label(format!("{}:", label));
                            ui.label(format!("{:.2}ms", ms));
                            ui.end_row();
                        }
                    });
            });
        }

        // Slow performance warning.
        if run_stats
            .total_simulation_time_with_readback
            .as_secs_f32()
            > 0.1
        {
            ui.add_space(4.0);
            ui.colored_label(
                Color32::from_rgb(180, 120, 60),
                #[cfg(not(target_arch = "wasm32"))]
                "Running slow? If you have both an integrated and discrete GPU, ensure the discrete GPU is in use.",
                #[cfg(target_arch = "wasm32")]
                "Running slow? If you have both an integrated and discrete GPU, ensure your browser runs exclusively on the discrete GPU.",
            );
        }
    }
}

fn mpm_performance(
    ui: &mut egui::Ui,
    stage: &crate::mpm::MpmStage<nexus::mpm::solver::GpuParticleModel>,
) {
    // Scene info.
    ui.label(RichText::new("Scene").strong());
    ui.add_space(2.0);

    egui::Grid::new("mpm_scene_grid")
        .num_columns(2)
        .spacing([20.0, 2.0])
        .show(ui, |ui| {
            ui.label("Particles:");
            ui.label(format!("{}", stage.physics.data.particles.len()));
            ui.end_row();

            ui.label("Substeps:");
            ui.label(format!("{}", stage.app_state.num_substeps));
            ui.end_row();
        });

    ui.add_space(8.0);
    ui.separator();
    ui.add_space(4.0);

    // Timing.
    let timings = &stage.step_result.timings;

    let total_ms = timings.total_step_time;
    let fps = if total_ms > 0.0 {
        (1000.0 / total_ms).round()
    } else {
        0.0
    };

    ui.label(RichText::new(format!("Total: {:.2}ms - {:.0} FPS", total_ms, fps)).strong());
    ui.add_space(4.0);

    egui::Grid::new("mpm_timing_grid")
        .num_columns(2)
        .spacing([20.0, 2.0])
        .show(ui, |ui| {
            ui.label("Encoding:");
            ui.label(format!("{:.1}ms", timings.encoding_time));
            ui.end_row();

            ui.label("Readback:");
            ui.label(format!("{:.1}ms", timings.readback_time));
            ui.end_row();
        });

    if !timings.gpu_pass_times.is_empty() {
        ui.add_space(4.0);

        CollapsingHeader::new(format!("GPU passes: {:.2}ms", timings.gpu_total_time))
            .id_salt("mpm_gpu_passes")
            .default_open(false)
            .show(ui, |ui| {
                egui::Grid::new("mpm_gpu_grid")
                    .num_columns(2)
                    .spacing([20.0, 2.0])
                    .show(ui, |ui| {
                        for (label, ms) in &timings.gpu_pass_times {
                            ui.label(format!("{}:", label));
                            ui.label(format!("{:.2}ms", ms));
                            ui.end_row();
                        }
                    });
            });
    }
}

fn fem_settings(
    ui: &mut egui::Ui,
    builders: &[DemoBuilder],
    stage: &mut crate::fem::FemStage,
    result: &mut UiInteractions,
) {
    ui.label(RichText::new("Scene").strong());
    ui.add_space(2.0);

    egui::Grid::new("fem_scene_info")
        .num_columns(2)
        .spacing([20.0, 2.0])
        .show(ui, |ui| {
            ui.label("Vertices:");
            ui.label(format!("{}", stage.data.num_vertices));
            ui.end_row();

            ui.label("Elements:");
            ui.label(format!("{}", stage.data.num_elements));
            ui.end_row();

            ui.label("Substeps:");
            ui.label(format!("{}", stage.data.num_substeps));
            ui.end_row();
        });

    // Handle FEM demo switching within FEM demos.
    if let Some(new_demo_idx) = result.new_selected_demo {
        if matches!(builders[new_demo_idx], DemoBuilder::Fem(..)) {
            let new_name = builders[new_demo_idx].name();
            if let Some(fem_idx) = stage
                .builders
                .iter()
                .position(|(name, _)| name == new_name)
            {
                stage.set_demo(fem_idx);
            }
        }
    }
}

fn fem_performance(
    ui: &mut egui::Ui,
    stage: &crate::fem::FemStage,
) {
    ui.label(RichText::new("Scene").strong());
    ui.add_space(2.0);

    egui::Grid::new("fem_perf_scene_grid")
        .num_columns(2)
        .spacing([20.0, 2.0])
        .show(ui, |ui| {
            ui.label("Vertices:");
            ui.label(format!("{}", stage.data.num_vertices));
            ui.end_row();

            ui.label("Elements:");
            ui.label(format!("{}", stage.data.num_elements));
            ui.end_row();

            ui.label("Substeps:");
            ui.label(format!("{}", stage.data.num_substeps));
            ui.end_row();
        });

    ui.add_space(8.0);
    ui.separator();
    ui.add_space(4.0);

    let timings = &stage.timings;
    let total_ms = timings.total_step_time;
    let fps = if total_ms > 0.0 {
        (1000.0 / total_ms).round()
    } else {
        0.0
    };

    ui.label(RichText::new(format!("Total: {:.2}ms - {:.0} FPS", total_ms, fps)).strong());
    ui.add_space(4.0);

    egui::Grid::new("fem_timing_grid")
        .num_columns(2)
        .spacing([20.0, 2.0])
        .show(ui, |ui| {
            ui.label("Encoding:");
            ui.label(format!("{:.1}ms", timings.encoding_time));
            ui.end_row();

            ui.label("Readback:");
            ui.label(format!("{:.1}ms", timings.readback_time));
            ui.end_row();
        });

    if !timings.gpu_pass_times.is_empty() {
        ui.add_space(4.0);

        CollapsingHeader::new(format!("GPU passes: {:.2}ms", timings.gpu_total_time))
            .id_salt("fem_gpu_passes")
            .default_open(false)
            .show(ui, |ui| {
                egui::Grid::new("fem_gpu_grid")
                    .num_columns(2)
                    .spacing([20.0, 2.0])
                    .show(ui, |ui| {
                        for (label, ms) in &timings.gpu_pass_times {
                            ui.label(format!("{}:", label));
                            ui.label(format!("{:.2}ms", ms));
                            ui.end_row();
                        }
                    });
            });
    }
}
