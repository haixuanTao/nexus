use khal_builder::KhalBuilder;

fn main() {
    let shader_crate = "../nexus_rbd_shaders3d";
    let output_dir = "../../crates/nexus_rbd3d/shaders-spirv";
    let src_dir = "../../src_rbd_shaders";

    KhalBuilder::new(shader_crate, true)
        .shader_src(src_dir)
        .feature("dim3")
        // Feature enabled unconditionally for the radix-sort device lost issue (see comment in the radix sort shader code).
        .feature("unsafe_remove_boundchecks")
        .build(output_dir);
}
