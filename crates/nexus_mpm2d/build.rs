use khal_builder::KhalBuilder;

fn main() {
    let shader_crate = "../nexus_mpm_shaders2d";
    let output_dir = "../../crates/nexus_mpm2d/shaders-spirv";
    let src_dir = "../../src_mpm_shaders";
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    let mut builder = KhalBuilder::new(shader_crate, true)
        .shader_src(src_dir)
        .feature("dim2");

    if target_arch == "wasm32" {
        builder = builder.feature("web-compat");
    }

    builder.build(output_dir);
}
