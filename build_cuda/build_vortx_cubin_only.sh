#!/bin/bash
# Rebuild ONLY the vortx-shaders cubin via the libNVVM path (the production-best
# path per native-cuda-loop-unroll memory). Used after adding a vortx kernel
# (e.g. gpu_sample_targets) so the native-CUDA backend embeds it. nexus cubin is
# untouched.
set -eo pipefail
BACKEND=/home/baguette/cuda-oxide-src/crates/rustc-codegen-cuda/target/debug/librustc_codegen_cuda.so
export CUDA_OXIDE_PTX_DIR=$HOME/nexus_ptx
export PATH=$HOME/.cargo/bin:$PATH
export CUDA_OXIDE_LIBDEVICE=/home/baguette/nvvm-wheel/extracted/nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc
export LIBNVVM_PATH=/home/baguette/nvvm-wheel/extracted/nvidia/cuda_nvcc/nvvm/lib64/libnvvm.so
export LIBNVJITLINK_PATH=/home/baguette/nvjit-wheel/extracted/nvidia/nvjitlink/lib/libnvJitLink.so.12
export CUDA_OXIDE_UNROLL_LOOPS=1

echo "########## VORTX (libNVVM + unroll) ##########"
cd ~/Documents/work/vortx
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
cd ~/make_cubin
OUT=$(cargo +nightly-2026-04-03 run --release -- $VLL sm_120 2>/tmp/mkcubin_v.err | grep "^CUBIN=" | cut -d= -f2)
tail -3 /tmp/mkcubin_v.err
[ -s "$CUDA_OXIDE_PTX_DIR/vortx_shaders.cubin" ] || { [ -n "$OUT" ] && cp "$OUT" $CUDA_OXIDE_PTX_DIR/vortx_shaders.cubin; }
ls -la $CUDA_OXIDE_PTX_DIR/vortx_shaders.cubin

CUOBJ=/home/baguette/.local/lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump
echo "sample kernel symbol in cubin:"
"$CUOBJ" -sass $CUDA_OXIDE_PTX_DIR/vortx_shaders.cubin 2>/dev/null | grep -oE 'gpu_sample_targets_cuda_entry_[0-9a-f]+' | head -1
echo "VORTX_CUBIN_DONE"
