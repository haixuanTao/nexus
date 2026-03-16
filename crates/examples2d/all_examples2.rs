#![allow(dead_code)]

use inflector::Inflector;
use nexus_testbed2d::{DemoBuilder, Testbed};
use std::cmp::Ordering;

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

struct CliOptions {
    example: Option<String>,
    list: bool,
    cpu: bool,
    run: bool,
}

fn parse_command_line() -> CliOptions {
    let mut args = std::env::args();
    let mut opts = CliOptions {
        example: None,
        list: false,
        cpu: false,
        run: false,
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--example" => opts.example = args.next(),
            "--list" => opts.list = true,
            "--cpu" => opts.cpu = true,
            "--run" => opts.run = true,
            _ => {}
        }
    }

    opts
}

#[allow(clippy::type_complexity)]
pub fn demo_builders() -> Vec<DemoBuilder> {
    let mut builders: Vec<DemoBuilder> = vec![
        balls2::builder(),
        boxes2::builder(),
        boxes_and_balls2::builder(),
        pyramid2::builder(),
        primitives2::builder(),
        polyline2::builder(),
        joint_ball2::builder(),
        joint_prismatic2::builder(),
        joint_fixed2::builder(),
        // MPM demos.
        centilever_beam2::builder(),
        sand2::builder(),
        elasticity2::builder(),
        elastic_cut2::builder(),
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
    let opts = parse_command_line();
    let mut builders = demo_builders();

    if opts.list {
        for builder in &builders {
            println!("{}", builder.name().to_camel_case());
        }
        return;
    }

    if let Some(ref demo) = opts.example {
        if let Some(i) = builders
            .iter()
            .position(|b| b.name().to_camel_case().as_str() == demo.as_str())
        {
            let single = vec![builders.into_iter().nth(i).unwrap()];
            builders = single;
        } else {
            eprintln!("Invalid example to run provided: '{demo}'");
            return;
        }
    }

    let mut testbed = Testbed::from_builders(builders);
    if opts.cpu {
        testbed = testbed.with_cpu();
    }
    if opts.run {
        testbed = testbed.with_running();
    }
    testbed.run().await
}
