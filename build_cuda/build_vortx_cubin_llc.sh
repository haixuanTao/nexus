#!/bin/bash
# Build the vortx-shaders cubin via the llc+ptxas path (sm_89 / Ada, 4090).
# libNVVM rejects cuda-oxide's opaque-pointer IR on pre-Blackwell, so we use
# the llc+ptxas path here instead of build_vortx_cubin_only.sh (libNVVM).
set -eo pipefail
TOOL=$HOME/.rustup/toolchains/nightly-2026-04-03-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/bin
LIBDEV=/usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc
PTXAS=/usr/bin/ptxas
BACKEND=$HOME/cuda-oxide-src/crates/rustc-codegen-cuda/target/debug/librustc_codegen_cuda.so
export CUDA_OXIDE_PTX_DIR=$HOME/nexus_ptx
export PATH=$HOME/.cargo/bin:$PATH
export CUDA_OXIDE_UNROLL_LOOPS=1
cd /home/peter/Documents/work/nex/vortx

echo "=== [1/5] cuda-oxide build vortx-shaders -> .ll ==="
cargo clean -p vortx-shaders 2>/dev/null || true
touch vortx-shaders/src/lib.rs
CARGO_INCREMENTAL=0 RUSTFLAGS="-Z codegen-backend=$BACKEND -Zalways-encode-mir -Zmir-enable-passes=-JumpThreading" \
  cargo +nightly-2026-04-03 build -p vortx-shaders --release \
  --no-default-features --features "cuda-oxide unsafe_remove_boundchecks" \
  --target nvptx64-nvidia-cuda -Z build-std=core
VLL=$CUDA_OXIDE_PTX_DIR/vortx_shaders.ll
echo "vortx ll: $(grep -c '^define' $VLL) defines; unroll-md $(grep -cE '!llvm.loop' $VLL)"

echo "=== [2/5] assemble + link libdevice ==="
$TOOL/llvm-as $VLL -o /tmp/vx.bc
$TOOL/llvm-link /tmp/vx.bc $LIBDEV -o /tmp/vx_linked.bc
echo "=== [3/5] internalize + globaldce ==="
$TOOL/opt -passes="internalize,globaldce" /tmp/vx_linked.bc -o /tmp/vx_pruned.bc
echo "=== [4/5] llc -> ptx ==="
$TOOL/llc -mcpu=sm_89 -O3 /tmp/vx_pruned.bc -o /tmp/vx.ptx
echo "=== [5/5] ptxas -> cubin ==="
rm -f $CUDA_OXIDE_PTX_DIR/vortx_shaders.cubin
$PTXAS -arch=sm_89 -O3 /tmp/vx.ptx -o $CUDA_OXIDE_PTX_DIR/vortx_shaders.cubin
ls -la $CUDA_OXIDE_PTX_DIR/vortx_shaders.cubin
echo "VORTX_CUBIN_DONE"
