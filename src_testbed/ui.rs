use crate::mpm::RenderMode;
use crate::mpm::step::RenderConfig;
use crate::rbd::BackendType;
use crate::viewer::UiState;
use crate::{DemoKind, RunState, Scene, Transition};
use khal::backend::Backend;
use kiss3d::egui;
use kiss3d::window::Window;
use nexus::rbd::pipeline::RbdStats;

use egui::{Button, CollapsingHeader, Color32, ComboBox, CornerRadius, RichText, Stroke};

/// Sets up a custom warm theme that complements the app's off-white background.
pub fn setup_custom_theme(ctx: &egui::Context) {
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

/// Builds the testbed control panel. Mutates `state` in place (run state, demo
/// selection, backend choice) and queries the scene for scene-specific widgets.
pub fn main_panel<S: Scene>(
    ctx: &egui::Context,
    state: &mut UiState,
    gpu_available: bool,
    scene: &mut S,
) {
    egui::Window::new("Nexus Testbed")
        .default_width(300.0)
        .show(ctx, |ui| {
            // GPU error banner.
            if let Some(error_msg) = &state.gpu_init_error {
                ui.colored_label(Color32::from_rgb(180, 70, 70), format!("GPU: {}", error_msg));
                ui.separator();
            }

            // Section toggles.
            ui.horizontal(|ui| {
                ui.toggle_value(&mut state.ui_sections.show_performance, "Performance");
                ui.toggle_value(&mut state.ui_sections.show_settings, "Settings");
                ui.toggle_value(&mut state.ui_sections.show_examples, "Examples");
            });

            egui::ScrollArea::vertical()
                .max_height(500.0)
                .show(ui, |ui| {
                    if state.ui_sections.show_settings {
                        ui.separator();
                        backend_selector(ui, state, gpu_available, scene.is_rbd());
                        ui.add_space(4.0);
                        scene.settings_ui(ui);
                    }

                    if state.ui_sections.show_performance {
                        ui.separator();
                        performance_ui(ui, &state.run_stats, state.backend_type);
                    }

                    if state.ui_sections.show_examples && !state.demos.is_empty() {
                        ui.separator();
                        examples_section(ui, state);
                    }
                });

            ui.separator();

            // Bottom controls.
            ui.horizontal(|ui| {
                let (play_label, play_hover) = if state.run_state == RunState::Running {
                    ("Pause", "Pause simulation (T)")
                } else {
                    ("Play", "Start simulation (T)")
                };

                if ui.button(play_label).on_hover_text(play_hover).clicked() {
                    state.run_state = if state.run_state == RunState::Running {
                        RunState::Paused
                    } else {
                        RunState::Running
                    };
                }

                if ui.button("Step").on_hover_text("Single step (S)").clicked() {
                    state.run_state = RunState::Step;
                }

                if ui
                    .button("Restart")
                    .on_hover_text("Restart example (R)")
                    .clicked()
                {
                    state.transition = Some(Transition::Switch);
                }
            });
        });
}

fn performance_ui(ui: &mut egui::Ui, rbd_stats: &RbdStats, backend: BackendType) {
    // // Scene info.
    // ui.label(RichText::new("Scene").strong());
    // ui.add_space(2.0);
    //
    // egui::Grid::new("rbd_scene_grid")
    //     .num_columns(2)
    //     .spacing([20.0, 2.0])
    //     .show(ui, |ui| {
    //         ui.label("Bodies:");
    //         ui.label(format!("{}", physics.backend.num_bodies()));
    //         ui.end_row();
    //
    //         ui.label("Joints:");
    //         ui.label(format!("{}", physics.backend.num_joints()));
    //         ui.end_row();
    //
    //         ui.label("Batches:");
    //         ui.label(format!("{}", physics.backend.num_batches()));
    //         ui.end_row();
    //     });

    ui.add_space(8.0);
    ui.separator();
    ui.add_space(4.0);

    // Timing.
    let total_ms_with_readback = rbd_stats.total_simulation_time_with_readback_ms();
    let total_ms_without_readback = rbd_stats.total_simulation_time_without_readback_ms();
    let total_readback_time = total_ms_with_readback - total_ms_without_readback;
    let fps = if total_ms_with_readback > 0.0 {
        (1000.0f32 / total_ms_with_readback).round()
    } else {
        0.0
    };

    ui.label(
        RichText::new(format!(
            "Total: {:.2}ms (+ readback: {:.2}ms) - {:.0} FPS",
            total_ms_without_readback, total_readback_time, fps
        ))
            .strong(),
    );
    ui.add_space(4.0);

    CollapsingHeader::new("Simulation details")
        .id_salt("rbd_sim_details")
        .default_open(false)
        .show(ui, |ui| {
            ui.label(format!("Colors: {}", rbd_stats.num_colors));
            ui.label(format!(
                "Coloring: {:.2}ms",
                rbd_stats.coloring_time.as_secs_f32() * 1000.0
            ));
            ui.label(format!(
                "Coloring iterations: {} x 10",
                rbd_stats.coloring_iterations
            ));
            ui.label(format!(
                "Start to pairs count: {:.2}ms",
                rbd_stats.start_to_pairs_count_time.as_secs_f32() * 1000.0
            ));
            ui.label(format!(
                "Coloring fallback: {:.2}ms",
                rbd_stats.coloring_fallback_time.as_secs_f32() * 1000.0
            ));
        });

    if !rbd_stats.gpu_pass_times.is_empty() {
        CollapsingHeader::new(format!("GPU passes: {:.2}ms", rbd_stats.gpu_total_time))
            .id_salt("rbd_gpu_passes")
            .default_open(false)
            .show(ui, |ui| {
                egui::Grid::new("rbd_timestamp_grid")
                    .num_columns(2)
                    .spacing([20.0, 2.0])
                    .show(ui, |ui| {
                        for (label, ms) in &rbd_stats.gpu_pass_times {
                            ui.label(format!("{}:", label));
                            ui.label(format!("{:.2}ms", ms));
                            ui.end_row();
                        }
                    });
            });
    }

    // Slow performance warning.
    if rbd_stats.total_simulation_time_with_readback.as_secs_f32() > 0.1 {
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

fn examples_section(ui: &mut egui::Ui, state: &mut UiState) {
    // Previous/Next navigation + current demo name.
    ui.horizontal(|ui| {
        if ui
            .add_enabled(state.selected_demo > 0, Button::new("<"))
            .on_hover_text("Previous example")
            .clicked()
        {
            state.selected_demo -= 1;
            state.transition = Some(Transition::Switch);
        }

        if ui
            .add_enabled(
                state.selected_demo + 1 < state.demos.len(),
                Button::new(">"),
            )
            .on_hover_text("Next example")
            .clicked()
        {
            state.selected_demo += 1;
            state.transition = Some(Transition::Switch);
        }

        ui.label(
            RichText::new(state.demos[state.selected_demo].0.clone())
                .strong()
                .italics(),
        );
    });

    ui.add_space(4.0);
    ui.separator();
    ui.add_space(4.0);

    demo_group(ui, state, DemoKind::Rbd, "Rigid Bodies");
    demo_group(ui, state, DemoKind::Mpm, "MPM");
    demo_group(ui, state, DemoKind::Fem, "FEM");
}

fn demo_group(ui: &mut egui::Ui, state: &mut UiState, kind: DemoKind, label: &str) {
    // Collect owned (index, name) so the closure can freely mutate `state`.
    let demos: Vec<(usize, String)> = state
        .demos
        .iter()
        .enumerate()
        .filter(|(_, (_, k))| *k == kind)
        .map(|(i, (name, _))| (i, name.clone()))
        .collect();

    if demos.is_empty() {
        return;
    }

    CollapsingHeader::new(format!("{} ({})", label, demos.len()))
        .default_open(true)
        .show(ui, |ui| {
            for (idx, name) in &demos {
                let is_selected = state.selected_demo == *idx;
                let text = if is_selected {
                    RichText::new(name).strong()
                } else {
                    RichText::new(name)
                };
                if ui
                    .selectable_label(is_selected, text)
                    .on_hover_text("Click to run this example")
                    .clicked()
                    && !is_selected
                {
                    state.selected_demo = *idx;
                    state.transition = Some(Transition::Switch);
                }
            }
        });
}

/// Unified backend selector. The "CPU (rapier)" option is only shown for RBD
/// scenes; for MPM/FEM scenes a current Rapier selection is shown as CPU (nexus).
fn backend_selector(ui: &mut egui::Ui, state: &mut UiState, gpu_available: bool, is_rbd: bool) {
    ui.label(RichText::new("Physics Backend").strong());
    ui.add_space(2.0);

    let mut new_backend: Option<BackendType> = None;
    let effective = if !is_rbd && state.backend_type == BackendType::Rapier {
        BackendType::Cpu
    } else {
        state.backend_type
    };

    if gpu_available
        && ui
            .radio(effective == BackendType::Gpu, "GPU (nexus)")
            .on_hover_text("GPU-accelerated physics with nexus")
            .clicked()
        && effective != BackendType::Gpu
    {
        new_backend = Some(BackendType::Gpu);
    }

    #[cfg(feature = "cuda")]
    if ui
        .radio(effective == BackendType::Cuda, "CUDA (nexus)")
        .on_hover_text("GPU-accelerated physics with nexus via CUDA")
        .clicked()
        && effective != BackendType::Cuda
    {
        new_backend = Some(BackendType::Cuda);
    }

    #[cfg(feature = "metal")]
    if ui
        .radio(effective == BackendType::Metal, "Metal (nexus)")
        .on_hover_text("GPU-accelerated physics with nexus via native Metal")
        .clicked()
        && effective != BackendType::Metal
    {
        new_backend = Some(BackendType::Metal);
    }

    #[cfg(feature = "cpu")]
    if ui
        .radio(effective == BackendType::Cpu, "CPU (nexus)")
        .on_hover_text("CPU physics using the nexus GPU pipeline executed on CPU")
        .clicked()
        && effective != BackendType::Cpu
    {
        new_backend = Some(BackendType::Cpu);
    }

    if is_rbd
        && ui
            .radio(state.backend_type == BackendType::Rapier, "CPU (rapier)")
            .on_hover_text("CPU physics with rapier")
            .clicked()
        && state.backend_type != BackendType::Rapier
    {
        new_backend = Some(BackendType::Rapier);
    }

    if let Some(bt) = new_backend {
        state.backend_type = bt;
        state.transition = Some(Transition::Switch);
    }
}

// ===========================================================================
// Per-scene UI, implemented through the `Scene` trait.
// ===========================================================================

impl Scene for crate::rbd::RbdScene {
    fn is_rbd(&self) -> bool {
        true
    }

    fn performance_ui(&mut self, ui: &mut egui::Ui, run_stats: &RbdStats, backend_type: BackendType) {
        let physics = &self.physics;

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
        let total_ms_with_readback = run_stats.total_simulation_time_with_readback_ms();
        let total_ms_without_readback = run_stats.total_simulation_time_without_readback_ms();
        let total_readback_time = total_ms_with_readback - total_ms_without_readback;
        let fps = if total_ms_with_readback > 0.0 {
            (1000.0f32 / total_ms_with_readback).round()
        } else {
            0.0
        };

        ui.label(
            RichText::new(format!(
                "Total: {:.2}ms (+ readback: {:.2}ms) - {:.0} FPS",
                total_ms_without_readback, total_readback_time, fps
            ))
            .strong(),
        );
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
                CollapsingHeader::new(format!("GPU passes: {:.2}ms", run_stats.gpu_total_time))
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
            if run_stats.total_simulation_time_with_readback.as_secs_f32() > 0.1 {
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
}

impl Scene for crate::mpm::MpmScene {
    fn settings_ui(&mut self, ui: &mut egui::Ui) {
        let stage = &mut self.stage;

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

        ui.checkbox(
            &mut stage.app_state.show_rigid_particles,
            "Show rigid particles",
        )
        .on_hover_text("Display particles belonging to rigid bodies");

        ui.add_space(8.0);

        ui.label(RichText::new("Solver").strong());
        ui.add_space(2.0);

        ui.checkbox(&mut stage.app_state.use_cpic, "Use CPIC")
            .on_hover_text("Compatible Particle-In-Cell transfer");
    }

    fn performance_ui(
        &mut self,
        ui: &mut egui::Ui,
        _run_stats: &RbdStats,
        _backend_type: BackendType,
    ) {
        let stage = &self.stage;

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
}

impl Scene for crate::fem::FemScene {
    fn settings_ui(&mut self, ui: &mut egui::Ui) {
        let stage = &self.stage;
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
    }

    fn performance_ui(
        &mut self,
        ui: &mut egui::Ui,
        _run_stats: &RbdStats,
        _backend_type: BackendType,
    ) {
        let stage = &self.stage;
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
}
