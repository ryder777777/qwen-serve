#!/usr/bin/env bash
# Launch llama-server (llama.cpp) for Qwen3.6-35B-A3B Q4_K_M on Intel Arc B70
# via the SYCL backend.
#
# Why this path exists:
#   sglang XPU FP8 (~95 tok/s) is our primary B70 backend, but it requires an
#   env-var workaround for the `!!!` bug (fp32 SSM state + GDN ESIMD disabled).
#   llama.cpp's ggml SYCL backend implements GDN (GGML_OP_GATED_DELTA_NET) with
#   fp32 state enforced at three levels (cache allocation, op assertion, kernel),
#   so it is structurally immune to the fp16 drift bug. It also supports EAGLE3
#   speculative decoding on SYCL (PR #24593), which is blocked in sglang XPU.
#
# Quant: Q4_K_M (~18-20 GB) fits a single B70 (32 GB). For 2× B70 use -sm row
#   to split. Q4_K_S is smaller/faster; Q8_0 is higher quality.
#
# Build: see llamacpp/build_b70_sycl.sh for the oneAPI 2026 SYCL build.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-35B-A3B-Q4_K_M.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8040}"

# --- device selection ---
# Pin to B70 device 0 only by default. For 2× B70, set ZE_AFFINITY_MASK=0,1
# and add -sm row to split the model across both cards.
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"

# --- GDN / recurrent state ---
# llama.cpp enforces fp32 GDN state at the op + cache level — no env vars
# needed, no !!! bug possible. The -ctk f16 / -ctv f16 below set the KV cache
# dtype for the 10 GQA full-attention layers only (NOT the GDN recurrent
# state, which is always fp32 for qwen35moe).

# --- Flash attention via XMX/oneDNN SDPA (PR #25222) ---
# Enabled by default in recent llama.cpp builds. Accelerates the 10 GQA layers.

# --- MTP / EAGLE3 speculative decoding (PR #24593) ---
# Uncomment the -draft-* lines below to enable speculative decoding using the
# MTP module built into the Qwen3.6 GGUF (n_layer_nextn).
# Requires a GGUF that includes the MTP draft layers (converted with
# convert_hf_to_gguf.py which writes n_layer_nextn from the HF config).

LLAMA_BIN="${LLAMA_BIN:-./llama-server}"

exec "${LLAMA_BIN}" \
    --model "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    -ngl 99 \
    -t 8 \
    -c 8192 \
    -b 2048 \
    -ub 2048 \
    -fa 1 \
    -ctk f16 \
    -ctv f16 \
    --no-warmup 0
#    --draft-max 16 \
#    --draft-min 4 \
#    --draft-p-min 0.9
