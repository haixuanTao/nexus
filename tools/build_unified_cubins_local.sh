#!/bin/bash
# Local (champagne) adaptation of build_unified_cubins.sh: cuda-oxide cubins
# for the migration stack (nexus-migrate + vortx-unified + khal-unified).
# Outputs to ~/rt_build/nexus_ptx. Consumed by khal-builder via
#   CUDA_OXIDE_SHADERS_PTX_NEXUS_RBD_SHADERS3D=$HOME/rt_build/nexus_ptx/nexus_rbd_shaders3d.cubin
#   CUDA_OXIDE_SHADERS_PTX_VORTX_SHADERS=$HOME/rt_build/nexus_ptx/vortx_shaders.cubin
set -eo pipefail
TOOL=$HOME/.rustup/toolchains/nightly-2026-04-03-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/bin
LIBDEV=$HOME/miniconda3/nvvm/libdevice/libdevice.10.bc
PTXAS=$HOME/miniconda3/bin/ptxas
BACKEND=$HOME/rt_build/cuda-oxide/crates/rustc-codegen-cuda/target/debug/librustc_codegen_cuda.so
export CUDA_OXIDE_PTX_DIR=$HOME/rt_build/nexus_ptx
mkdir -p "$CUDA_OXIDE_PTX_DIR"
export PATH=$HOME/.cargo/bin:$PATH

build_one () { # $1 workspace dir, $2 package, $3 features, $4 ll name
  cd "$1"
  cargo clean -p "$2" 2>/dev/null || true
  CARGO_INCREMENTAL=0 RUSTFLAGS="-Z codegen-backend=$BACKEND -Zalways-encode-mir -Zmir-enable-passes=-JumpThreading" \
    cargo +nightly-2026-04-03 build -p "$2" --release \
    --no-default-features --features "$3" \
    --target nvptx64-nvidia-cuda -Z build-std=core
  LL=$CUDA_OXIDE_PTX_DIR/$4.ll
  echo "ll: $(grep -c '^define' "$LL") defines"
  "$TOOL"/llvm-as "$LL" -o /tmp/u.bc
  "$TOOL"/llvm-link /tmp/u.bc "$LIBDEV" -o /tmp/u_linked.bc
  "$TOOL"/opt -passes="internalize,globaldce" /tmp/u_linked.bc -o /tmp/u_pruned.bc
  "$TOOL"/llc -mcpu=sm_120 -O3 /tmp/u_pruned.bc -o /tmp/u.ptx
  "$PTXAS" -arch=sm_120 -O3 /tmp/u.ptx -o "$CUDA_OXIDE_PTX_DIR/$4.cubin"
  ls -la "$CUDA_OXIDE_PTX_DIR/$4.cubin"
}

build_one "$HOME/rt_build/nexus-migrate" nexus_rbd_shaders3d "cuda-oxide dim3 unsafe_remove_boundchecks" nexus_rbd_shaders3d
build_one "$HOME/rt_build/vortx-unified" vortx-shaders "cuda-oxide" vortx_shaders
echo DONE
