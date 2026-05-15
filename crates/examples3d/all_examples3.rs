#![allow(dead_code)]

use inflector::Inflector;

use nexus_testbed3d::{DemoBuilder, Testbed};
use std::cmp::Ordering;

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

struct CliOptions {
    command: Command,
    cpu: bool,
    cuda: bool,
    metal: bool,
    run: bool,
}

enum Command {
    Run(String),
    List,
    RunAll,
}

fn parse_command_line() -> CliOptions {
    let mut args = std::env::args();
    let mut command = Command::RunAll;
    let mut cpu = false;
    let mut cuda = false;
    let mut metal = false;
    let mut run = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--example" => command = Command::Run(args.next().unwrap_or_default()),
            "--list" => command = Command::List,
            "--cpu" => cpu = true,
            "--cuda" => cuda = true,
            "--metal" => metal = true,
            "--run" => run = true,
            _ => {}
        }
    }

    CliOptions {
        command,
        cpu,
        cuda,
        metal,
        run,
    }
}

#[allow(clippy::type_complexity)]
pub fn demo_builders() -> Vec<DemoBuilder> {
    let mut builders: Vec<DemoBuilder> = vec![
        balls3::builder(),
        boxes3::builder(),
        boxes_and_balls3::builder(),
        primitives3::builder(),
        pyramid3::builder(),
        many_pyramids3::builder(),
        many_pyramids_batch3::builder(),
        keva3::builder(),
        joints3::builder(),
        joint_ball3::builder(),
        joint_fixed3::builder(),
        joint_prismatic3::builder(),
        joint_revolute3::builder(),
        joint_revolute_batch3::builder(),
        multibody_pendulum3::builder(),
        trimesh3::builder(),
        urdf3::builder(),
        // MPM demos.
        centilever_beam3::builder(),
        sand3::builder(),
        heightfield3::builder(),
        elastic_cut3::builder(),
        // FEM demos.
        fem_cube3::builder(),
    ];

    // Lexicographic sort, with stress tests moved at the end of the list.
    builders.sort_by(
        |a, b| match (a.name().starts_with('('), b.name().starts_with('(')) {
            (true, true) | (false, false) => a.name().cmp(b.name()),
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
        },
    );
    builders
}

#[kiss3d::main]
pub async fn main() {
    env_logger::init();
    let opts = parse_command_line();
    let builders = demo_builders();

    let apply_opts = |testbed: Testbed| {
        let mut t = testbed;
        if opts.cpu {
            t = t.with_cpu();
        }
        #[cfg(feature = "cuda")]
        if opts.cuda {
            t = t.with_backend(nexus_testbed3d::BackendType::Cuda);
        }
        #[cfg(feature = "metal")]
        if opts.metal {
            t = t.with_backend(nexus_testbed3d::BackendType::Metal);
        }
        if opts.run {
            t = t.with_running();
        }
        t
    };

    match opts.command {
        Command::Run(demo) => {
            if let Some(i) = builders
                .iter()
                .position(|builder| builder.name().to_camel_case().as_str() == demo.as_str())
            {
                let single = vec![builders.into_iter().nth(i).unwrap()];
                apply_opts(Testbed::from_builders(single)).run().await
            } else {
                eprintln!("Invalid example to run provided: '{demo}'");
            }
        }
        Command::RunAll => apply_opts(Testbed::from_builders(builders)).run().await,
        Command::List => {
            for builder in &builders {
                println!("{}", builder.name().to_camel_case())
            }
        }
    }
}
