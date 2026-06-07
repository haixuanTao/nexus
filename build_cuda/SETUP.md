# Native-CUDA (cuda-oxide) build environment — bootstrap for a new box

The native-CUDA nexus port needs a heavy, mostly-non-git environment. Repos are
on GitHub (haixuanTao forks); the toolchain/wheels are downloaded.

## 1. Repos (clone at these branches, as sibling dirs under ~/Documents/work/ + ~/)
- haixuanTao/nexus-cuda      (master)                         -> ~/Documents/work/nexus-cuda
- haixuanTao/khal            (feat/cuda-oxide-backend)        -> ~/Documents/work/khal
- haixuanTao/vortx           (feat/gpu-policy-shaders)        -> ~/Documents/work/vortx
- haixuanTao/cuda-oxide      (feat/nexus3d-vortx-native-cuda) -> ~/cuda-oxide-src
(zealot lives separately; it drives training and syncs to champagne.)

## 2. Toolchain
- rustup nightly-2026-04-03 + components rust-src, rustc-dev, llvm-tools
  (build-std=core needs rust-src; the backend links rustc internals).
- LLVM 21.1.0 at ~/llvm21 (cuda-oxide backend links it; CUDA_OXIDE_LLC=~/llvm21/bin/llc).

## 3. CUDA 12.9 wheels (the box system CUDA 12.0 is too old; download userspace)
- pip download nvidia-cuda-nvcc-cu12==12.9.86  -> extract -> ~/nvvm-wheel/extracted/.../cuda_nvcc/
    libnvvm    = .../nvvm/lib64/libnvvm.so
    libdevice  = .../nvvm/libdevice/libdevice.10.bc
    ptxas      = .../bin/ptxas
- pip download nvidia-nvjitlink-cu12==12.9.86  -> ~/nvjit-wheel/extracted/.../libnvJitLink.so.12
- 12.9 cuobjdump/nvdisasm (SASS for sm_120): redist tarballs cuda_cuobjdump / cuda_nvdisasm 12.9.

## 4. Build the cuda-oxide backend .so
  cd ~/cuda-oxide-src/crates/rustc-codegen-cuda && cargo +nightly-2026-04-03 build
  -> target/debug/librustc_codegen_cuda.so   (NOT a workspace member; build from its own dir)

## 5. make_cubin (libNVVM .ll -> cubin)
  ~/make_cubin : cargo +nightly-2026-04-03 build --release
  env it needs: CUDA_OXIDE_LIBDEVICE, LIBNVVM_PATH, LIBNVJITLINK_PATH (point at the wheels above).

## 6. Build + run nexus native-CUDA
  cd ~/Documents/work/nexus-cuda && bash build_cuda/build_nexus_cubin_libnvvm.sh
  (EDIT the absolute /home/baguette/... paths in the scripts for the new box.)
  Then: CUDA_OXIDE_SHADERS_PTX is set by the script; ./target/release/boxes3d_cuda runs the sim.

## CRITICAL FLAGS (see build_cuda/README.md)
- -Zmir-enable-passes=-JumpThreading  (step-1 barrier-deadlock fix; REQUIRED)
- -Zalways-encode-mir, clean build, features "cuda-oxide dim3 unsafe_remove_boundchecks".

## Shortcut: rsync from baguette (fastest, copies the built env)
  rsync -a baguette:~/{cuda-oxide-src,make_cubin,nvvm-wheel,nvjit-wheel,llvm21} ~/
  rsync -a baguette:~/Documents/work/nexus-cuda ~/Documents/work/
