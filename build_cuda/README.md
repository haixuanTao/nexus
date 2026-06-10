# Native-CUDA (cuda-oxide) build for nexus_rbd_shaders3d

Builds the rapier-derived rigid-body shaders to a sm_120 cubin via cuda-oxide
(Rust->PTX) + libNVVM, and embeds it into the host (boxes3d_cuda) through
nexus_rbd3d build.rs (CUDA_OXIDE_SHADERS_PTX).

## CRITICAL build flags
- `-Zmir-enable-passes=-JumpThreading` (REQUIRED): rustc JumpThreading
  correlates repeated `if lane==0 {..}` conditions and DUPLICATES the
  intervening `workgroup_barrier()` across the if-arms -> asymmetric per-lane
  bar.sync arrivals -> CTA DEADLOCK at the first such barrier (step-1 hang).
  Upstream cuda-oxide already sets this in cargo-oxide build_rustflags; the
  manual nexus build must set it too. Minimal repro: barrier_div_test.rs
  (two sequential `if lane==0 {..}; barrier`).
- `-Zalways-encode-mir` (REQUIRED): makes external-crate (parry3d/rapier3d) MIR collectable.
- clean build (`cargo clean -p nexus_rbd_shaders3d`) before each build.
- features: `cuda-oxide dim3 unsafe_remove_boundchecks`, target nvptx64, -Zbuild-std=core.

## Scripts
- build_nexus_cubin_libnvvm.sh : cuda-oxide .ll -> make_cubin(libNVVM) -> embed host (hash-asserted)
- build_nexus_cubin.sh         : same but llc+ptxas cubin path

## Status
- Builds + runs on the 5090; bit-exact vs WebGPU.
- FIXED (29fac1e): the step-23 ILLEGAL_ADDRESS in gpu_solver_init_constraints
  (contact path) is resolved. The contact-heavy biped (zealot iter_e2e_bench)
  now runs end-to-end on native CUDA, gather bit-exact (err 0.0).
