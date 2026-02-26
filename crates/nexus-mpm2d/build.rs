//! Build script for nexus2d - compiles 2D rust-gpu shaders to SPIR-V.

use std::process::Command;

fn main() {
    let shader_crate = "../nexus-mpm-shaders2d";
    let output_dir = "../../crates/nexus-mpm2d/shaders-spirv";

    println!("cargo:rerun-if-changed={}", shader_crate);
    // Watch all files in src_shaders recursively
    for entry in walkdir::WalkDir::new("../../src_mpm_shaders")
        .into_iter()
        .filter_map(|e| e.ok())
    {
        println!("cargo:rerun-if-changed={}", entry.path().display());
    }
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_PUSH_CONSTANTS");

    let mut features = vec!["dim2"];

    #[cfg(target_arch = "wasm32")]
    features.push("web-compat");
    
    // #[cfg(feature = "unsafe_remove_boundchecks")]
    features.push("unsafe_remove_boundchecks");

    // NOTE: push_constants is not currently a defined feature in Cargo.toml
    // #[cfg(feature = "push_constants")]
    // features.push("push_constants");

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
        panic!("cargo gpu build failed for 2D shaders");
    }
}
