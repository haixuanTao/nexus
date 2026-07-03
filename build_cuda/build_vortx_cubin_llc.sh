#!/usr/bin/env bash
# vortx-shaders -> cubin via the llc+ptxas path (arch = $SM, default sm_120).
# libNVVM rejects cuda-oxide's opaque-pointer IR on pre-Blackwell, so the
# llc+ptxas path is used here instead of build_vortx_cubin_only.sh (libNVVM).
# Paths auto-detect (see detect_env.sh); override any var before running.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/detect_env.sh"
export CUDA_OXIDE_UNROLL_LOOPS=1
cd "$VORTX_DIR"

echo "=== [1/5] cuda-oxide build vortx-shaders -> .ll ==="
cargo clean -p vortx-shaders 2>/dev/null || true
touch vortx-shaders/src/lib.rs
CARGO_INCREMENTAL=0 RUSTFLAGS="-Z codegen-backend=$BACKEND -Zalways-encode-mir -Zmir-enable-passes=-JumpThreading" \
  cargo +"$NIGHTLY" build -p vortx-shaders --release \
  --no-default-features --features "cuda-oxide unsafe_remove_boundchecks" \
  --target nvptx64-nvidia-cuda -Z build-std=core
VLL=$CUDA_OXIDE_PTX_DIR/vortx_shaders.ll
echo "vortx ll: $(grep -c '^define' "$VLL") defines; unroll-md $(grep -cE '!llvm.loop' "$VLL")"

echo "=== [2/5] assemble + link libdevice ==="
"$TOOL"/llvm-as "$VLL" -o /tmp/vx.bc
"$TOOL"/llvm-link /tmp/vx.bc "$LIBDEV" -o /tmp/vx_linked.bc
echo "=== [3/5] internalize + globaldce ==="
"$TOOL"/opt -passes="internalize,globaldce" /tmp/vx_linked.bc -o /tmp/vx_pruned.bc
echo "=== [4/5] llc -> ptx ($SM) ==="
"$TOOL"/llc -mcpu="$SM" -O3 /tmp/vx_pruned.bc -o /tmp/vx.ptx
echo "=== [5/5] ptxas -> cubin ($SM) ==="
rm -f "$CUDA_OXIDE_PTX_DIR/vortx_shaders.cubin"
"$PTXAS" -arch="$SM" -O3 /tmp/vx.ptx -o "$CUDA_OXIDE_PTX_DIR/vortx_shaders.cubin"
ls -la "$CUDA_OXIDE_PTX_DIR/vortx_shaders.cubin"
echo "VORTX_CUBIN_DONE"
