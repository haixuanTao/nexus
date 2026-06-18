#![allow(dead_code)]

use inflector::Inflector;
use nexus_testbed3d::{DemoKind, NexusViewer};

mod balls3;
mod boxes3;
mod boxes_and_balls3;
mod joint_ball3;
mod joint_fixed3;
mod joint_prismatic3;
mod joint_revolute3;
mod joint_revolute_batch3;
mod joints3;
mod keva3;
mod many_pyramids3;
mod many_pyramids_batch3;
mod multibody_pendulum3;
mod primitives3;
mod pyramid3;
mod trimesh3;
mod urdf3;

// MPM examples.
mod centilever_beam3;
mod elastic_cut3;
mod heightfield3;
mod sand3;

// FEM examples.
mod fem_cube3;

/// Declares the demo registry: a `(name, kind)` list for the picker UI and a
/// name -> `run()` dispatcher. Keeping both in one macro keeps them in sync.
macro_rules! demos {
    ( $( $name:literal => $kind:ident : $module:ident ),* $(,)? ) => {
        fn demo_list() -> Vec<(String, DemoKind)> {
            let mut demos: Vec<(String, DemoKind)> =
                vec![ $( ($name.to_string(), DemoKind::$kind) ),* ];
            // Lexicographic sort, with stress tests (names starting with '(')
            // moved to the end of the list.
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
    "Balls" => Rbd : balls3,
    "Boxes" => Rbd : boxes3,
    "Boxes & balls" => Rbd : boxes_and_balls3,
    "Primitives" => Rbd : primitives3,
    "Pyramid" => Rbd : pyramid3,
    "Many pyramids" => Rbd : many_pyramids3,
    "Many pyramids (batched)" => Rbd : many_pyramids_batch3,
    "Keva tower" => Rbd : keva3,
    "Joints (multibody)" => Rbd : joints3,
    "Joints (Spherical)" => Rbd : joint_ball3,
    "Joints (Fixed)" => Rbd : joint_fixed3,
    "Joints (Prismatic)" => Rbd : joint_prismatic3,
    "Joints (Revolute)" => Rbd : joint_revolute3,
    "Joints (Revolute - Batched)" => Rbd : joint_revolute_batch3,
    "Multibody (Pendulum)" => Rbd : multibody_pendulum3,
    "Trimesh" => Rbd : trimesh3,
    "URDF (multibody)" => Rbd : urdf3,
    // MPM demos.
    "Cantilever beam" => Mpm : centilever_beam3,
    "Sand" => Mpm : sand3,
    "Heightfield" => Mpm : heightfield3,
    "Elastic cut" => Mpm : elastic_cut3,
    // FEM demos.
    "FEM cube" => Fem : fem_cube3,
}

struct CliOptions {
    example: Option<String>,
    list: bool,
    cpu: bool,
    cuda: bool,
    metal: bool,
    run: bool,
}

fn parse_command_line() -> CliOptions {
    let mut args = std::env::args();
    let mut opts = CliOptions {
        example: None,
        list: false,
        cpu: false,
        cuda: false,
        metal: false,
        run: false,
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--example" => opts.example = args.next(),
            "--list" => opts.list = true,
            "--cpu" => opts.cpu = true,
            "--cuda" => opts.cuda = true,
            "--metal" => opts.metal = true,
            "--run" => opts.run = true,
            _ => {}
        }
    }

    opts
}

#[kiss3d::main]
pub async fn main() {
    env_logger::init();
    let opts = parse_command_line();
    let demos = demo_list();

    if opts.list {
        for (name, _) in &demos {
            println!("{}", name.to_camel_case());
        }
        return;
    }

    // Resolve `--example NAME` to a starting demo index.
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
    #[cfg(feature = "cuda")]
    if opts.cuda {
        viewer = viewer.with_backend(nexus_testbed3d::BackendType::Cuda);
    }
    #[cfg(feature = "metal")]
    if opts.metal {
        viewer = viewer.with_backend(nexus_testbed3d::BackendType::Metal);
    }
    if opts.run {
        viewer = viewer.with_running();
    }

    viewer.init_backend();

    // Each selected demo owns its own loop (`run`); it returns when the user
    // closes the window or picks another demo (via the picker, which makes
    // `viewer.render()` return false).
    loop {
        let sel = viewer.selected_demo();
        dispatch(&demos[sel].0, &mut viewer).await;
        if viewer.quitting() {
            break;
        }
        viewer.clear_scene();
        viewer.clear_transition();
    }
}
