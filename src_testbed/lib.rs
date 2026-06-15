#![allow(clippy::result_large_err)]
#![allow(clippy::too_many_arguments)]

#[cfg(feature = "dim2")]
pub extern crate nexus2d as nexus;
#[cfg(feature = "dim3")]
pub extern crate nexus3d as nexus;
#[cfg(feature = "dim2")]
pub extern crate rapier2d as rapier;
#[cfg(feature = "dim3")]
pub extern crate rapier3d as rapier;

pub mod fem;
pub mod mpm;
pub mod rbd;
pub mod viewer;
mod ui;

use nexus::rbd::pipeline::RunStats;

pub use rbd::BackendType;
pub use rbd::{BatchEnvironment, PhysicsBackend, RbdScene, SimulationState, VisualShape};

pub use fem::FemScene;
pub use mpm::MpmScene;
pub use viewer::{UiState, Viewer};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum RunState {
    Running,
    Paused,
    Step,
}

#[derive(Copy, Clone)]
pub struct UiSections {
    pub show_examples: bool,
    pub show_settings: bool,
    pub show_performance: bool,
}

/// The kind of solver a registered demo uses. Used only to group demos in the
/// picker UI.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum DemoKind {
    Rbd,
    Mpm,
    Fem,
}

/// A loop transition requested from the UI: stop entirely, or switch to another
/// registered demo. The target index is carried in [`UiState::selected_demo`];
/// this only signals the example-owned `while viewer.render()` loop to exit so
/// the browser can run the next demo.
pub(crate) enum Transition {
    Quit,
    Switch,
}

/// Implemented by every scene type ([`RbdScene`], [`MpmScene`], [`FemScene`]) so
/// the generic [`Viewer::render`] can draw scene-specific settings/performance
/// widgets without knowing the concrete type.
pub trait Scene {
    /// Whether this is a rigid-body scene (controls whether the backend selector
    /// offers the rapier CPU backend).
    fn is_rbd(&self) -> bool {
        false
    }

    /// Scene-specific settings widgets (e.g. MPM render mode / CPIC).
    fn settings_ui(&mut self, _ui: &mut kiss3d::egui::Ui) {}

    /// Scene-specific performance/scene-info widgets.
    fn performance_ui(
        &mut self,
        _ui: &mut kiss3d::egui::Ui,
        _run_stats: &RunStats,
        _backend_type: BackendType,
    ) {
    }
}
