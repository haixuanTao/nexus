#!/bin/bash
# libNVVM + unroll for BOTH cubins (the synthesis): cuda-oxide .ll (unroll hint) -> make_cubin (libNVVM).
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/detect_env.sh"
export CUDA_OXIDE_UNROLL_LOOPS=1   # emit the hint; libNVVM decides whether to honor it

# --- nexus: reuse the authored libNVVM script (inherits CUDA_OXIDE_UNROLL_LOOPS) ---
touch "$NEXUS_DIR/src_rbd_shaders/lib.rs"   # force nvptx64 recompile (clean -p won't)
echo "########## NEXUS (libNVVM + unroll) ##########"
bash "$NEXUS_DIR/build_cuda/build_nexus_cubin_libnvvm.sh"

# --- vortx: cuda-oxide build -> .ll -> make_cubin ---
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
rm -f $CUDA_OXIDE_PTX_DIR/vortx_shaders.cubin
cd "$MAKE_CUBIN_DIR"
OUT=$(cargo +"$NIGHTLY" run --release -- "$VLL" "$SM" 2>/tmp/mkcubin_v.err | grep "^CUBIN=" | cut -d= -f2)
tail -3 /tmp/mkcubin_v.err
[ -s "$CUDA_OXIDE_PTX_DIR/vortx_shaders.cubin" ] || { [ -n "$OUT" ] && cp "$OUT" $CUDA_OXIDE_PTX_DIR/vortx_shaders.cubin; }
ls -la $CUDA_OXIDE_PTX_DIR/vortx_shaders.cubin

NAME=$("$CUOBJ" -sass $CUDA_OXIDE_PTX_DIR/vortx_shaders.cubin 2>/dev/null | grep -oE 'gemm_tiled_cuda_entry_[0-9a-f]+' | head -1)
echo "libNVVM vortx gemm_tiled FP mix:"; "$CUOBJ" -sass -fun "$NAME" $CUDA_OXIDE_PTX_DIR/vortx_shaders.cubin 2>/dev/null | grep -E '/\*[0-9a-f]+\*/' | grep -oE '\b(FFMA|FMUL|FADD)\b' | sort | uniq -c
echo "LIBNVVM_BOTH_DONE"
