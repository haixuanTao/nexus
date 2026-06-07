#!/bin/bash
set -eo pipefail
BACKEND=/home/baguette/cuda-oxide-src/crates/rustc-codegen-cuda/target/debug/librustc_codegen_cuda.so
export CUDA_OXIDE_PTX_DIR=$HOME/nexus_ptx
export PATH=$HOME/.cargo/bin:$PATH
export CUDA_OXIDE_LIBDEVICE=/home/baguette/nvvm-wheel/extracted/nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc
export LIBNVVM_PATH=/home/baguette/nvvm-wheel/extracted/nvidia/cuda_nvcc/nvvm/lib64/libnvvm.so
export LIBNVJITLINK_PATH=/home/baguette/nvjit-wheel/extracted/nvidia/nvjitlink/lib/libnvJitLink.so.12
cd ~/Documents/work/nexus-cuda

echo "=== [1/4] cuda-oxide build nexus_rbd_shaders3d -> .ll ==="
cargo clean -p nexus_rbd_shaders3d 2>/dev/null || true
CARGO_INCREMENTAL=0 RUSTFLAGS="-Z codegen-backend=$BACKEND -Zalways-encode-mir -Zmir-enable-passes=-JumpThreading" \
  cargo +nightly-2026-04-03 build -p nexus_rbd_shaders3d --release \
  --no-default-features --features "cuda-oxide dim3 unsafe_remove_boundchecks" \
  --target nvptx64-nvidia-cuda -Z build-std=core
LL=$CUDA_OXIDE_PTX_DIR/nexus_rbd_shaders3d.ll
DEV_HASH=$(grep -oE "gpu_solver_init_constraints_cuda_entry_[0-9a-f]+" $LL | sort -u)
echo "ll: $(grep -c "^define" $LL) defines; device init_constraints = $DEV_HASH"

echo "=== [2/4] make_cubin (libNVVM + nvJitLink) ==="
rm -f $CUDA_OXIDE_PTX_DIR/nexus_rbd_shaders3d.cubin
cd ~/make_cubin
OUT=$(cargo +nightly-2026-04-03 run --release -- $LL sm_120 2>/tmp/mkcubin.err | grep "^CUBIN=" | cut -d= -f2)
cat /tmp/mkcubin.err | tail -3
echo "produced: $OUT"
[ -s "$CUDA_OXIDE_PTX_DIR/nexus_rbd_shaders3d.cubin" ] || { echo "!!! make_cubin produced NO cubin"; cat /tmp/mkcubin.err; exit 1; }
[ "$OUT" = "$CUDA_OXIDE_PTX_DIR/nexus_rbd_shaders3d.cubin" ] || cp "$OUT" $CUDA_OXIDE_PTX_DIR/nexus_rbd_shaders3d.cubin
ls -la $CUDA_OXIDE_PTX_DIR/nexus_rbd_shaders3d.cubin
cd ~/Documents/work/nexus-cuda

echo "=== [3/4] rebuild HOST, embedding cubin ==="
export CUDA_OXIDE_SHADERS_PTX=$CUDA_OXIDE_PTX_DIR/nexus_rbd_shaders3d.cubin
touch crates/nexus_rbd3d/build.rs
cargo build --release -p pendulum_headless --bin boxes3d_cuda

echo "=== [4/4] verify embed ==="
EMB=$(find target/release/build -path "*nexus_rbd3d*/out/shaders-spirv/shaders.ptx" | head -1)
EMB_HASH=$(strings "$EMB" | grep -oE "gpu_solver_init_constraints_cuda_entry_[0-9a-f]+" | sort -u)
echo "embedded $EMB ($(stat -c%s "$EMB") bytes); init_constraints = $EMB_HASH"
[ "$DEV_HASH" = "$EMB_HASH" ] && echo "EMBED HASH MATCH OK" || { echo "MISMATCH dev=$DEV_HASH emb=$EMB_HASH"; exit 1; }
echo "DONE"
