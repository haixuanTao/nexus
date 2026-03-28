use khal_builder::KhalBuilder;

fn main() {
    let shader_crate = "../nexus_mpm_shaders2d";
    let output_dir = "../../crates/nexus_mpm2d/shaders-spirv";
    let src_dir = "../../src_mpm_shaders";

    KhalBuilder::new(shader_crate, true)
        .shader_src(src_dir)
        .feature("dim2")
        .build(output_dir);
}
