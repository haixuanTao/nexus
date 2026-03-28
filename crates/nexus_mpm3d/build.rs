use khal_builder::KhalBuilder;

fn main() {
    let shader_crate = "../nexus_mpm_shaders3d";
    let output_dir = "../../crates/nexus_mpm3d/shaders-spirv";
    let src_dir = "../../src_mpm_shaders";

    KhalBuilder::new(shader_crate, true)
        .shader_src(src_dir)
        .feature("dim3")
        .build(output_dir);
}
