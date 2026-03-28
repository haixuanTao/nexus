use khal_builder::KhalBuilder;

fn main() {
    let shader_crate = "../nexus_fem_shaders3d";
    let output_dir = "../../crates/nexus_fem3d/shaders-spirv";
    let src_dir = "../../src_fem_shaders";

    KhalBuilder::new(shader_crate, true)
        .shader_src(src_dir)
        .feature("dim3")
        .build(output_dir);
}