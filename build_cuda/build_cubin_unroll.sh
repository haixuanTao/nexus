#!/bin/bash
# Build a shader-crate cubin with the cuda-oxide loop-unroll win (HANDOFF_5090).
# Usage: build_cubin_unroll.sh <crate> <features> <crate_dir> <src_touch> <ll_name> <verify_sym>
set -eo pipefail
CRATE=$1; FEATURES=$2; CRATE_DIR=$3; SRC_TOUCH=$4; LL_NAME=$5; VSYM=$6

# LLVM 21 tools — MUST match the LLVM the cuda-oxide backend links (LLVM 21).
# (The old build_nexus_cubin.sh used the rust-nightly LLVM-20 tools, which silently
# ignore the LLVM-21-emitted !llvm.loop.unroll.full hint → unroll no-op.)
TOOL=${LLVM21_BIN:-/home/baguette/llvm21/bin}
LIBDEV=/home/baguette/nvvm-wheel/extracted/nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc
PTXAS=/home/baguette/.local/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas
BACKEND=/home/baguette/cuda-oxide-src/crates/rustc-codegen-cuda/target/debug/librustc_codegen_cuda.so
export CUDA_OXIDE_PTX_DIR=${CUDA_OXIDE_PTX_DIR:-$HOME/nexus_ptx}
export PATH=$HOME/.cargo/bin:$PATH
DO_UNROLL=${DO_UNROLL:-1}                 # 0 = same toolchain, unroll pass OFF (control)
[ "$DO_UNROLL" = "1" ] && export CUDA_OXIDE_UNROLL_LOOPS=1   # emit !llvm.loop.unroll.full hints
mkdir -p $CUDA_OXIDE_PTX_DIR
cd "$CRATE_DIR"

echo "=== [$CRATE 1/6] cuda-oxide build -> .ll (unroll ON) ==="
cargo clean -p "$CRATE" 2>/dev/null || true
touch "$SRC_TOUCH"        # GOTCHA: clean -p doesn't clear nvptx64 artifact; force recompile
CARGO_INCREMENTAL=0 RUSTFLAGS="-Z codegen-backend=$BACKEND -Zalways-encode-mir -Zmir-enable-passes=-JumpThreading" \
  cargo +nightly-2026-04-03 build -p "$CRATE" --release \
  --no-default-features --features "$FEATURES" \
  --target nvptx64-nvidia-cuda -Z build-std=core
LL=$CUDA_OXIDE_PTX_DIR/$LL_NAME.ll
UNROLL_MD=$(grep -cE 'llvm\.loop\.unroll\.full|!llvm\.loop' "$LL" || true)
echo "ll: $(grep -c '^define' "$LL") defines; unroll-metadata lines = $UNROLL_MD (DO_UNROLL=$DO_UNROLL)"
if [ "$DO_UNROLL" = "1" ] && [ "$UNROLL_MD" -eq 0 ]; then echo "!!! no unroll metadata in .ll — backend not emitting hint"; exit 1; fi

echo "=== [$CRATE 2/6] assemble + link libdevice ==="
$TOOL/llvm-as "$LL" -o /tmp/u.bc
$TOOL/llvm-link /tmp/u.bc "$LIBDEV" -o /tmp/u_linked.bc

echo "=== [$CRATE 3/6] value-only prune + inline (NO -O3: avoids barrier-dup deadlock) ==="
$TOOL/opt -passes="internalize,globaldce,inline,sroa,early-cse,instcombine<no-verify-fixpoint>,gvn" \
  /tmp/u_linked.bc -o /tmp/u1.bc

echo "=== [$CRATE 4/6] canonicalize loops + UNROLL ==="
$TOOL/opt -passes="loop-simplify,loop(loop-rotate,indvars),loop-unroll,sroa,instcombine<no-verify-fixpoint>,gvn,adce" \
  /tmp/u1.bc -o /tmp/u2.bc
$TOOL/opt -passes="globaldce" /tmp/u2.bc -o /tmp/u3.bc

echo "=== [$CRATE 5/6] llc -> ptx -> cubin ==="
$TOOL/llc -mcpu=sm_120 -O3 --fp-contract=fast /tmp/u3.bc -o /tmp/u.ptx
OUT=$CUDA_OXIDE_PTX_DIR/$LL_NAME.cubin
rm -f "$OUT"
$PTXAS -arch=sm_120 -O3 /tmp/u.ptx -o "$OUT"
ls -la "$OUT"

echo "=== [$CRATE 6/6] SASS density (FFMA share) — non-fatal ==="
set +e
CUOBJDUMP=$(dirname "$PTXAS")/cuobjdump
SASS=$("$CUOBJDUMP" -sass "$OUT" 2>/dev/null)
FFMA=$(echo "$SASS" | grep -cE '\bFFMA\b')
IMAD=$(echo "$SASS" | grep -cE '\bIMAD\b')
TOTAL=$(echo "$SASS" | grep -cE '/\*[0-9a-f]+\*/')
echo "$CRATE: FFMA=$FFMA  IMAD=$IMAD  total-insns≈$TOTAL"
echo "DONE $CRATE -> $OUT"
exit 0
