#!/bin/bash
# Rebuild ONLY the vortx-shaders cubin via the libNVVM path (the production-best
# path per native-cuda-loop-unroll memory). Used after adding a vortx kernel
# (e.g. gpu_sample_targets) so the native-CUDA backend embeds it. nexus cubin is
# untouched.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/detect_env.sh"
export CUDA_OXIDE_UNROLL_LOOPS=1

echo "########## VORTX (libNVVM + unroll) ##########"
cd "$VORTX_DIR"
cargo clean -p vortx-shaders 2>/dev/null || true
touch vortx-shaders/src/lib.rs
CARGO_INCREMENTAL=0 RUSTFLAGS="-Z codegen-backend=$BACKEND -Zalways-encode-mir -Zmir-enable-passes=-JumpThreading" \
  cargo +nightly-2026-04-03 build -p vortx-shaders --release \
  --no-default-features --features "cuda-oxide unsafe_remove_boundchecks" \
  --target nvptx64-nvidia-cuda -Z build-std=core
VLL=$CUDA_OXIDE_PTX_DIR/vortx_shaders.ll
echo "vortx ll: $(grep -c '^define' $VLL) defines; unroll-md $(grep -cE '!llvm.loop' $VLL)"
echo "sample kernel present in .ll: $(grep -c 'gpu_sample_targets' $VLL)"
rm -f $CUDA_OXIDE_PTX_DIR/vortx_shaders.cubin
cd "$MAKE_CUBIN_DIR"
OUT=$(cargo +"$NIGHTLY" run --release -- "$VLL" "$SM" 2>/tmp/mkcubin_v.err | grep "^CUBIN=" | cut -d= -f2)
tail -3 /tmp/mkcubin_v.err
[ -s "$CUDA_OXIDE_PTX_DIR/vortx_shaders.cubin" ] || { [ -n "$OUT" ] && cp "$OUT" $CUDA_OXIDE_PTX_DIR/vortx_shaders.cubin; }
ls -la $CUDA_OXIDE_PTX_DIR/vortx_shaders.cubin

echo "sample kernel symbol in cubin:"
"$CUOBJ" -sass $CUDA_OXIDE_PTX_DIR/vortx_shaders.cubin 2>/dev/null | grep -oE 'gpu_sample_targets_cuda_entry_[0-9a-f]+' | head -1
echo "VORTX_CUBIN_DONE"
