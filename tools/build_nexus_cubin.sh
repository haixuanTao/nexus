#!/bin/bash
set -eo pipefail
TOOL=/home/baguette/.rustup/toolchains/nightly-2025-08-04-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/bin
LIBDEV=/home/baguette/nvvm-wheel/extracted/nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc
PTXAS=/home/baguette/.local/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas
BACKEND=/home/baguette/cuda-oxide-src/crates/rustc-codegen-cuda/target/debug/librustc_codegen_cuda.so
export CUDA_OXIDE_PTX_DIR=$HOME/nexus_ptx
export PATH=$HOME/.cargo/bin:$PATH
cd ~/Documents/work/nexus-cuda

echo "=== [1/6] cuda-oxide build nexus_rbd_shaders3d -> .ll ==="
cargo clean -p nexus_rbd_shaders3d 2>/dev/null || true
set -o pipefail
CARGO_INCREMENTAL=0 RUSTFLAGS="-Z codegen-backend=$BACKEND -Zalways-encode-mir -Zmir-enable-passes=-JumpThreading" \
  cargo +nightly-2026-04-03 build -p nexus_rbd_shaders3d --release \
  --no-default-features --features "cuda-oxide dim3 unsafe_remove_boundchecks" \
  --target nvptx64-nvidia-cuda -Z build-std=core
LL=$CUDA_OXIDE_PTX_DIR/nexus_rbd_shaders3d.ll
DEV_HASH=$(grep -oE "gpu_solver_init_constraints_cuda_entry_[0-9a-f]+" $LL | sort -u)
echo "ll: $(grep -c "^define" $LL) defines; device init_constraints = $DEV_HASH"

echo "=== [2/6] assemble + link libdevice ==="
$TOOL/llvm-as $LL -o /tmp/nx.bc
$TOOL/llvm-link /tmp/nx.bc $LIBDEV -o /tmp/nx_linked.bc
echo "=== [3/6] internalize + globaldce ==="
$TOOL/opt -passes="internalize,globaldce" /tmp/nx_linked.bc -o /tmp/nx_pruned.bc
echo "=== [4/6] llc -> ptx ==="
$TOOL/llc -mcpu=sm_120 -O3 /tmp/nx_pruned.bc -o /tmp/nx.ptx
echo "=== [5/6] ptxas -> cubin ==="
rm -f $CUDA_OXIDE_PTX_DIR/nexus_rbd_shaders3d.cubin
$PTXAS -arch=sm_120 -O3 /tmp/nx.ptx -o $CUDA_OXIDE_PTX_DIR/nexus_rbd_shaders3d.cubin
ls -la $CUDA_OXIDE_PTX_DIR/nexus_rbd_shaders3d.cubin

echo "=== [6/6] rebuild HOST binary (boxes3d_cuda), embedding the cubin via build.rs ==="
export CUDA_OXIDE_SHADERS_PTX=$CUDA_OXIDE_PTX_DIR/nexus_rbd_shaders3d.cubin
touch crates/nexus_rbd3d/build.rs   # force build.rs rerun so the fresh cubin is re-copied/embedded
cargo build --release -p pendulum_headless --bin boxes3d_cuda
HOST_HASH=$(strings target/release/boxes3d_cuda | grep -oE "gpu_solver_init_constraints_cuda_entry_[0-9a-f]+" | sort -u)
echo "host init_constraints = $HOST_HASH"
EMB=$(find target/release/build -path "*nexus_rbd3d*/out/shaders-spirv/shaders.ptx" | head -1)
EMB_HASH=$(strings "$EMB" | grep -oE "gpu_solver_init_constraints_cuda_entry_[0-9a-f]+" | sort -u)
echo "embedded shaders.ptx = $EMB ($(stat -c%s "$EMB") bytes); init_constraints = $EMB_HASH"
if [ "$DEV_HASH" = "$EMB_HASH" ]; then echo "EMBED HASH MATCH OK"; else echo "!!! EMBED MISMATCH: dev=$DEV_HASH emb=$EMB_HASH"; exit 1; fi
echo "DONE"
