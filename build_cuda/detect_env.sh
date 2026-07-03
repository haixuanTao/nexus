#!/usr/bin/env bash
# Shared environment detection for the cuda-oxide cubin builds.
#
# Auto-detects every toolchain path and artifact so the build scripts need NO
# hand-editing per machine. Every value is overridable: export the variable
# before running a build script to force it. Source this from a build script:
#
#   source "$(dirname "${BASH_SOURCE[0]}")/detect_env.sh"
#
# Common overrides:  SM=sm_89 (4090)  NIGHTLY=nightly-YYYY-MM-DD  PTXAS=/path
#                    CUDA_OXIDE_PTX_DIR=/path  VORTX_DIR=/path  BACKEND=/path.so

# first existing path among the args (echoes nothing if none; never fails under set -e)
_first() { local p; for p in "$@"; do [ -n "$p" ] && [ -e "$p" ] && { printf '%s\n' "$p"; return 0; }; done; return 0; }

# --- repo roots (relative to this file: build_cuda/ sits at the nexus repo root) ---
_DETECT_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${NEXUS_DIR:=$(cd "$_DETECT_SELF/.." && pwd)}"     # nexus-cuda repo root
: "${WORK_DIR:=$(cd "$NEXUS_DIR/.." && pwd)}"         # parent holding the sibling repos
: "${VORTX_DIR:=$WORK_DIR/vortx}"
: "${CUDA_OXIDE_SRC:=$HOME/cuda-oxide-src}"
: "${MAKE_CUBIN_DIR:=$HOME/make_cubin}"               # libNVVM make_cubin tool (libnvvm scripts)

# --- artifact output dir (.ll / .ptx / .cubin) ---
: "${CUDA_OXIDE_PTX_DIR:=$HOME/nexus_ptx}"
export CUDA_OXIDE_PTX_DIR
mkdir -p "$CUDA_OXIDE_PTX_DIR"

# --- target GPU arch: Blackwell/5090 by default; SM=sm_89 for Ada/4090, etc. ---
: "${SM:=sm_120}"

# --- rust: cargo on PATH + nightly toolchain (the codegen backend needs this exact nightly) ---
export PATH="$HOME/.cargo/bin:$PATH"
: "${NIGHTLY:=nightly-2026-04-03}"

# --- cuda-oxide codegen backend .so ---
: "${BACKEND:=$CUDA_OXIDE_SRC/crates/rustc-codegen-cuda/target/debug/librustc_codegen_cuda.so}"

# --- LLVM-21 tools (llvm-as/llvm-link/opt/llc). Prefer ~/llvm21, else the nightly's
#     rustlib bin. The .ll is LLVM-21 IR, so these MUST be LLVM 21. ---
: "${TOOL:=$(_first "$HOME/llvm21/bin" "$HOME/.rustup/toolchains/$NIGHTLY-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/bin")}"

# --- CUDA 12.9 wheel artifacts (glob the extraction dirs; fall back to system CUDA) ---
: "${LIBDEV:=$(_first "$HOME"/nvvm-wheel/extracted/*/cuda_nvcc/nvvm/libdevice/libdevice.10.bc /usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc)}"
: "${LIBNVVM_PATH:=$(_first "$HOME"/nvvm-wheel/extracted/*/cuda_nvcc/nvvm/lib64/libnvvm.so)}"
: "${LIBNVJITLINK_PATH:=$(_first "$HOME"/nvjit-wheel/extracted/*/nvjitlink/lib/libnvJitLink.so.12)}"
export CUDA_OXIDE_LIBDEVICE="${CUDA_OXIDE_LIBDEVICE:-$LIBDEV}"
export LIBNVVM_PATH LIBNVJITLINK_PATH

# --- ptxas / cuobjdump: prefer the 12.9 wheel, then triton's bundled copy, then $PATH ---
: "${PTXAS:=$(_first "$HOME"/nvvm-wheel/extracted/*/cuda_nvcc/bin/ptxas "$HOME"/.local/lib/python*/site-packages/triton/backends/nvidia/bin/ptxas "$(command -v ptxas 2>/dev/null || true)" /usr/bin/ptxas)}"
: "${CUOBJ:=$(_first "$HOME"/.local/lib/python*/site-packages/triton/backends/nvidia/bin/cuobjdump "$(command -v cuobjdump 2>/dev/null || true)")}"

# --- surface anything unresolved (warn, don't hard-fail: not every script needs all) ---
for _v in BACKEND TOOL PTXAS LIBDEV; do
  eval "_p=\${$_v}"
  [ -n "$_p" ] && [ -e "$_p" ] || echo "[detect_env] WARN: $_v unresolved (${_p:-<empty>}) — set it explicitly if this build needs it" >&2
done
echo "[detect_env] SM=$SM NIGHTLY=$NIGHTLY NEXUS_DIR=$NEXUS_DIR PTX_DIR=$CUDA_OXIDE_PTX_DIR"
