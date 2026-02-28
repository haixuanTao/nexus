#![allow(dead_code)]

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
    let mut builders: Vec<DemoBuilder> = vec![
        DemoBuilder::Rbd("Balls", balls2::init_world),
        DemoBuilder::Rbd("Boxes", boxes2::init_world),
        DemoBuilder::Rbd("Boxes & balls", boxes_and_balls2::init_world),
        DemoBuilder::Rbd("Pyramid", pyramid2::init_world),
        DemoBuilder::Rbd("Primitives", primitives2::init_world),
        DemoBuilder::Rbd("Polyline", polyline2::init_world),
        DemoBuilder::Rbd("Joints (spherical)", joint_ball2::init_world),
        DemoBuilder::Rbd("Joints (prismatic)", joint_prismatic2::init_world),
        DemoBuilder::Rbd("Joints (fixed)", joint_fixed2::init_world),
        // MPM demos.
        DemoBuilder::Mpm("Cantilever beam".to_string(), centilever_beam2::beam_demo),
        DemoBuilder::Mpm("Sand".to_string(), sand2::sand_demo),
        DemoBuilder::Mpm("Elasticity".to_string(), elasticity2::elasticity_demo),
        DemoBuilder::Mpm("Elastic cut".to_string(), elastic_cut2::elastic_cut_demo),
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
