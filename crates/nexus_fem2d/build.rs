//! Build script for nexus_fem2d - compiles 2D rust-gpu shaders to SPIR-V.

use std::process::Command;

fn main() {
    let shader_crate = "../nexus_fem_shaders2d";
    let output_dir = "../../crates/nexus_fem2d/shaders-spirv";

    println!("cargo:rerun-if-changed={}", shader_crate);
    for entry in walkdir::WalkDir::new("../../src_fem_shaders")
        .into_iter()
        .filter_map(|e| e.ok())
    {
        println!("cargo:rerun-if-changed={}", entry.path().display());
    }

    let mut features = vec!["dim2"];

    // #[cfg(target_arch = "wasm32")]
    // features.push("web-compat");

    features.push("unsafe_remove_boundchecks");

    let features = features.join(",");

    let mut args = vec![
        "gpu",
        "build",
        "--shader-crate",
        shader_crate,
        "--output-dir",
        output_dir,
        "--multimodule",
    ];

    if !features.is_empty() {
        args.push("--features");
        args.push(&features);
    }

    let status = Command::new("cargo")
        .args(args)
        .status()
        .expect("failed to run cargo gpu");

    if !status.success() {
        panic!("cargo gpu build failed for 2D FEM shaders");
    }
}
