use khal_builder::KhalBuilder;
use std::path::PathBuf;

fn main() {
    let output_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR not set by cargo"))
        .join("shaders-spirv");

    let mut builder = KhalBuilder::from_dependency("nexus_rbd_shaders3d", true)
        .feature("dim3")
        // Feature enabled unconditionally for the radix-sort device lost issue (see comment in the radix sort shader code).
        .feature("unsafe_remove_boundchecks");
    // Browsers run strict WGSL uniformity analysis that rejects the fast
    // barrier-loop bounds; wasm builds get the strict variants (native SPIR-V
    // is unchanged).
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("wasm32") {
        builder = builder.feature("strict_uniformity");
    }
    builder.build(output_dir);
}
