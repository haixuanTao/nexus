#![allow(dead_code)]

use inflector::Inflector;
use nexus_testbed2d::{DemoKind, NexusViewer};

mod balls2;
mod boxes2;
mod boxes_and_balls2;
mod joint_ball2;
mod joint_fixed2;
mod joint_prismatic2;
mod polyline2;
mod primitives2;
mod pyramid2;

// MPM examples.
mod centilever_beam2;
mod elastic_cut2;
mod elasticity2;
mod sand2;

/// Declares the demo registry: a `(name, kind)` list for the picker UI and a
/// name -> `run()` dispatcher. Keeping both in one macro keeps them in sync.
macro_rules! demos {
    ( $( $name:literal => $kind:ident : $module:ident ),* $(,)? ) => {
        fn demo_list() -> Vec<(String, DemoKind)> {
            let mut demos: Vec<(String, DemoKind)> =
                vec![ $( ($name.to_string(), DemoKind::$kind) ),* ];
            demos.sort_by(|a, b| match (a.0.starts_with('('), b.0.starts_with('(')) {
                (true, true) | (false, false) => a.0.cmp(&b.0),
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
            });
            demos
        }

        async fn dispatch(name: &str, viewer: &mut NexusViewer) {
            match name {
                // `run` may return `()` (legacy demos) or a `Result` (demos
                // migrated to the `NexusState` API); discard whatever it yields
                // so every arm has the same `()` type.
                $( $name => { let _ = $module::run(viewer).await; }, )*
                _ => eprintln!("Unknown demo: '{name}'"),
            }
        }
    };
}

demos! {
    "Balls" => Rbd : balls2,
    "Boxes" => Rbd : boxes2,
    "Boxes & balls" => Rbd : boxes_and_balls2,
    "Pyramid" => Rbd : pyramid2,
    "Primitives" => Rbd : primitives2,
    "Polyline" => Rbd : polyline2,
    "Joints (spherical)" => Rbd : joint_ball2,
    "Joints (prismatic)" => Rbd : joint_prismatic2,
    "Joints (fixed)" => Rbd : joint_fixed2,
    // MPM demos.
    "Cantilever beam" => Mpm : centilever_beam2,
    "Sand" => Mpm : sand2,
    "Elasticity" => Mpm : elasticity2,
    "Elastic cut" => Mpm : elastic_cut2,
}

struct CliOptions {
    example: Option<String>,
    list: bool,
    cpu: bool,
    metal: bool,
    run: bool,
}

fn parse_command_line() -> CliOptions {
    let mut args = std::env::args();
    let mut opts = CliOptions {
        example: None,
        list: false,
        cpu: false,
        metal: false,
        run: false,
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--example" => opts.example = args.next(),
            "--list" => opts.list = true,
            "--cpu" => opts.cpu = true,
            "--metal" => opts.metal = true,
            "--run" => opts.run = true,
            _ => {}
        }
    }

    opts
}

#[kiss3d::main]
pub async fn main() {
    let opts = parse_command_line();
    let demos = demo_list();

    if opts.list {
        for (name, _) in &demos {
            println!("{}", name.to_camel_case());
        }
        return;
    }

    let mut selected = 0;
    if let Some(ref demo) = opts.example {
        match demos
            .iter()
            .position(|(name, _)| name.to_camel_case().as_str() == demo.as_str())
        {
            Some(i) => selected = i,
            None => {
                eprintln!("Invalid example to run provided: '{demo}'");
                return;
            }
        }
    }

    let mut viewer = NexusViewer::new(demos.clone()).await;
    viewer = viewer.with_selected_demo(selected);
    if opts.cpu {
        viewer = viewer.with_cpu();
    }
    #[cfg(feature = "metal")]
    if opts.metal {
        viewer = viewer.with_backend(nexus_testbed2d::BackendType::Metal);
    }
    if opts.run {
        viewer = viewer.with_running();
    }

    loop {
        // Initialize the currently-selected backend (it may have just changed
        // via the UI backend selector). Idempotent for already-created backends.
        viewer.init_backend();
        let sel = viewer.selected_demo();
        dispatch(&demos[sel].0, &mut viewer).await;
        if viewer.quitting() {
            break;
        }
        viewer.clear_scene();
        viewer.clear_transition();
    }
}
