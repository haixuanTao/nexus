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
mod keva3;
mod many_pyramids3;
mod primitives3;
mod pyramid3;
mod trimesh3;

// MPM examples.
mod centilever_beam3;
mod elastic_cut3;
mod heightfield3;
mod sand3;

enum Command {
    Run(String),
    List,
    RunAll,
}

fn parse_command_line() -> Command {
    let mut args = std::env::args();

    while let Some(arg) = args.next() {
        if &arg[..] == "--example" {
            return Command::Run(args.next().unwrap_or_default());
        } else if &arg[..] == "--list" {
            return Command::List;
        }
    }

    Command::RunAll
}

#[allow(clippy::type_complexity)]
pub fn demo_builders() -> Vec<DemoBuilder> {
    let mut builders: Vec<DemoBuilder> = vec![
        DemoBuilder::Rbd("Balls", balls3::init_world),
        DemoBuilder::Rbd("Boxes", boxes3::init_world),
        DemoBuilder::Rbd("Boxes & balls", boxes_and_balls3::init_world),
        DemoBuilder::Rbd("Primitives", primitives3::init_world),
        DemoBuilder::Rbd("Pyramid", pyramid3::init_world),
        DemoBuilder::Rbd("Many pyramids", many_pyramids3::init_world),
        DemoBuilder::Rbd("Keva tower", keva3::init_world),
        DemoBuilder::Rbd("Joints (Spherical)", joint_ball3::init_world),
        DemoBuilder::Rbd("Joints (Fixed)", joint_fixed3::init_world),
        DemoBuilder::Rbd("Joints (Prismatic)", joint_prismatic3::init_world),
        DemoBuilder::Rbd("Joints (Revolute)", joint_revolute3::init_world),
        DemoBuilder::Rbd("Trimesh", trimesh3::init_world),
        // MPM demos.
        DemoBuilder::Mpm("Cantilever beam".to_string(), centilever_beam3::beam_demo),
        DemoBuilder::Mpm("Sand".to_string(), sand3::sand_demo),
        DemoBuilder::Mpm("Heightfield".to_string(), heightfield3::heightfield_demo),
        DemoBuilder::Mpm("Elastic cut".to_string(), elastic_cut3::elastic_cut_demo),
    ];

    // Lexicographic sort, with stress tests moved at the end of the list.
    builders.sort_by(|a, b| match (a.name().starts_with('('), b.name().starts_with('(')) {
        (true, true) | (false, false) => a.name().cmp(b.name()),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
    });
    builders
}

#[kiss3d::main]
pub async fn main() {
    let command = parse_command_line();
    let builders = demo_builders();

    match command {
        Command::Run(demo) => {
            if let Some(i) = builders
                .iter()
                .position(|builder| builder.name().to_camel_case().as_str() == demo.as_str())
            {
                // Extract the single builder for the specific demo.
                let single = vec![builders.into_iter().nth(i).unwrap()];
                Testbed::from_builders(single).run().await
            } else {
                eprintln!("Invalid example to run provided: '{demo}'");
            }
        }
        Command::RunAll => Testbed::from_builders(builders).run().await,
        Command::List => {
            for builder in &builders {
                println!("{}", builder.name().to_camel_case())
            }
        }
    }
}
