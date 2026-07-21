#!/bin/bash
# Cubins for the UNIFIED stack (nexus-merge + vortx-unified + khal-unified).
# Outputs to ~/nexus_ptx_unified — never clobbers the production ~/nexus_ptx.
set -eo pipefail
TOOL=/home/baguette/.rustup/toolchains/nightly-2026-04-03-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/bin
LIBDEV=/home/baguette/nvvm-wheel/extracted/nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc
PTXAS=/home/baguette/cuda-13.3-tile/bin/ptxas
BACKEND=/home/baguette/cuda-oxide-src/crates/rustc-codegen-cuda/target/debug/librustc_codegen_cuda.so
export CUDA_OXIDE_PTX_DIR=$HOME/nexus_ptx_unified
mkdir -p $CUDA_OXIDE_PTX_DIR
export PATH=$HOME/.cargo/bin:$PATH

build_one () { # $1 workspace dir, $2 package, $3 features, $4 ll name
  cd "$1"
  cargo clean -p "$2" 2>/dev/null || true
  CARGO_INCREMENTAL=0 RUSTFLAGS="-Z codegen-backend=$BACKEND -Zalways-encode-mir -Zmir-enable-passes=-JumpThreading" \
    cargo +nightly-2026-04-03 build -p "$2" --release \
    --no-default-features --features "$3" \
    --target nvptx64-nvidia-cuda -Z build-std=core
  LL=$CUDA_OXIDE_PTX_DIR/$4.ll
  echo "ll: $(grep -c '^define' $LL) defines"
  $TOOL/llvm-as $LL -o /tmp/u.bc
  $TOOL/llvm-link /tmp/u.bc $LIBDEV -o /tmp/u_linked.bc
  $TOOL/opt -passes="internalize,globaldce" /tmp/u_linked.bc -o /tmp/u_pruned.bc
  $TOOL/llc -mcpu=sm_120 -O3 /tmp/u_pruned.bc -o /tmp/u.ptx
  $PTXAS -arch=sm_120 -O3 /tmp/u.ptx -o $CUDA_OXIDE_PTX_DIR/$4.cubin
  ls -la $CUDA_OXIDE_PTX_DIR/$4.cubin
}

build_one ~/Documents/work/nexus-merge nexus_rbd_shaders3d "cuda-oxide dim3 unsafe_remove_boundchecks" nexus_rbd_shaders3d
build_one ~/Documents/work/vortx-unified vortx-shaders "cuda-oxide" vortx_shaders
echo DONE
