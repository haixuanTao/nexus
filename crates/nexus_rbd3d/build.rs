//! Build script for nexus_rbd3d - compiles 3D rust-gpu shaders to SPIR-V.

use std::process::Command;

fn main() {
    let shader_crate = "../nexus_rbd_shaders3d";
    let output_dir = "../../crates/nexus_rbd3d/shaders-spirv";

    println!("cargo:rerun-if-changed={}", shader_crate);
    // Watch all files in src_rbd_shaders recursively
    for entry in walkdir::WalkDir::new("../../src_rbd_shaders")
        .into_iter()
        .filter_map(|e| e.ok())
    {
        println!("cargo:rerun-if-changed={}", entry.path().display());
    }
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_PUSH_CONSTANTS");

    // // Clean the output directory before building
    // let output_path = Path::new(output_dir);
    // if output_path.exists() {
    //     if let Ok(entries) = fs::read_dir(output_path) {
    //         for entry in entries.filter_map(|e| e.ok()) {
    //             let path = entry.path();
    //             if path.extension().is_some_and(|ext| ext == "spv")
    //                 || path.file_name().is_some_and(|name| name == "manifest.json")
    //             {
    //                 let _ = fs::remove_file(&path);
    //             }
    //         }
    //     }
    // }

    let mut features = vec!["dim3"];

    // #[cfg(feature = "unsafe_remove_boundchecks")]
    features.push("unsafe_remove_boundchecks");

    // NOTE: push_constants feature is not currently defined in Cargo.toml
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
        panic!("cargo gpu build failed for 3D shaders");
    }
}
