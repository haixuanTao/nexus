use khal_builder::KhalBuilder;
use std::path::PathBuf;

fn main() {
    let output_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR not set by cargo"))
        .join("shaders-spirv");

    KhalBuilder::from_dependency("nexus_fem_shaders3d", true)
        .feature("dim3")
        .build(output_dir);
}
