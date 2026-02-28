#![allow(dead_code)]

use nexus_testbed2d::Testbed;
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

fn demo_name_from_command_line() -> Option<String> {
    let mut args = std::env::args();

    while let Some(arg) = args.next() {
        if &arg[..] == "--example" {
            return args.next();
        }
    }

    None
}

#[cfg(target_arch = "wasm32")]
fn demo_name_from_url() -> Option<String> {
    None
}

#[cfg(not(target_arch = "wasm32"))]
fn demo_name_from_url() -> Option<String> {
    None
}

#[kiss3d::main]
pub async fn main() {
    let mut builders = vec![
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

    let testbed = Testbed::from_builders(builders);

    testbed.run().await
}
