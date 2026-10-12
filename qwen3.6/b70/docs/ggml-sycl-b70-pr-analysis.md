# ggml-org SYCL PR Analysis — B70 Speedup & Applicability to Qwen3.6-35B-A3B

> Research date: 2026-08-18
> Trigger: User report of a ggml-org PR "yesterday" (≈Aug 17, 2026) with a large
> SYCL speedup on Intel Arc B70.

## TL;DR

The PR is **llama.cpp #27160** (`sycl: fix thread/block count in quantized cpy
kernel launches`), merged 2026-08-17. It fixes a work-group under-subscription
bug in the SYCL quantized tensor-copy kernels — **7.83× speedup** on B70 for
`q4_0 → f32` dequantizing copies.

**Direct applicability to our sglang XPU path: limited.** sglang uses
`sgl-kernel-xpu`, not ggml's SYCL backend, so the PR doesn't touch our kernel
stack. An audit of `sgl-kernel-xpu` found no equivalent under-subscription bug
in its copy/quantize kernels.

**Indirect applicability: high.** llama.cpp has a complete, first-class
`qwen35moe` architecture implementation with a fused `GGML_OP_GATED_DELTA_NET`
op on SYCL that is **fp32 end-to-end** — structurally immune to the `!!!` bug we
fight in sgl-kernel-xpu. A cluster of sibling SYCL PRs (Jun–Aug 2026) optimizes
every component of Qwen3.6-35B-A3B on Battlemage. The llama.cpp Q4_K path is an
actionable alternative B70 backend that sidesteps our fp16 drift problem entirely.

---

## PR #27160 — Technical Breakdown

| Field | Value |
|---|---|
| **PR** | [#27160](https://github.com/ggml-org/llama.cpp/pull/27160) |
| **Title** | `sycl: fix thread/block count in quantized cpy kernel launches` |
| **Repo** | ggml-org/llama.cpp |
| **Author** | Titaniumtown |
| **Merged** | 2026-08-17 04:32 UTC |
| **File** | `ggml/src/ggml-sycl/cpy.cpp` (single file, +102/−56) |
| **AI disclosure** | Claude Opus 5 found the thread-spawn discrepancy |

### The bug

Every quantized `cpy` (tensor copy / dtype-convert) kernel in `cpy.cpp`
launched its SYCL `parallel_for` with a work-group size of **1**
(`sycl::range<3>(1,1,1)`). The block count was set to either `ne` (total
element count) for `q→f32`/`q→q` paths, or `ne / QK` for `f32→q` paths.

This means:
- Each quant block was processed by a **single thread** — severe SIMD
  under-utilization on Xe2 vector engines.
- The `q→f32` paths were additionally **over-subscribed by QK×** (e.g. 32×
  for Q4_0), launching 32× more blocks than needed.

### The fix

- Work-group size is now `SYCL_CPY_BLOCK_SIZE` (a proper sub-group-aligned block).
- `num_blocks = ceil_div(ne / QK, SYCL_CPY_BLOCK_SIZE)` — proportional to the
  quant block size, eliminating both the single-thread under-subscription and the
  `QK`× over-subscription.
- Added missing `GGML_ASSERT(ne % QK == 0)` checks on the `q→f32` paths.

### Affected kernels

All quantized cpy variants: `f32↔q8_0`, `q2_0_f32`, `f32↔q4_0`, `f32↔q4_1`,
`f32↔q5_0`, `f32↔q5_1`, `mxfp4↔f32`, `iq4_nl`, `f16↔q4_0/q4_1/q5_0`, and
same-type `q8_0_q8_0`, `q5_0_q5_0`, `q5_1_q5_1`, `q4_0_q4_0` copies.

### Benchmark

| Hardware | Path | Before | After | Speedup |
|---|---|---|---|---|
| **Intel Arc B70** (BMG-G31) | q4_0 → f32 | 20.21 GB/s | 158.19 GB/s | **7.83×** |
| Intel Arc B60 | Q4_0 | 9.08 GB/s | 104.73 GB/s | ~11.5× |

**All other quant types are flat** (no impact). The headline 7.83× win is
specific to the `q4_0 → f32` dequantizing-copy path, which had the worst
under-subscription. K-quants (Q4_K, Q8_0, Q6_K) use different cpy code paths
not touched by this PR.

---

## Applicability to Our sglang XPU Path

### Does the PR touch our kernel stack? No.

Our B70 inference uses sglang with `sgl-kernel-xpu` (the `sgl-kernel-xpu` SYCL
kernel package), not ggml's SYCL backend. The `cpy.cpp` fix is in llama.cpp's
own ggml backend and has no effect on sglang's kernel stack.

### Does sgl-kernel-xpu have the same bug? No.

Audited the full `sgl-kernel-xpu` source tree (`sgl-project/sgl-kernel-xpu`,
cloned to `/tmp/sgl-kernel-xpu-audit`). Results:

- **Zero** instances of `sycl::range<3>(1,1,1)` or the equivalent
  `sycl_kernel_submit(1, 1, ...)`.
- The AWQ dequantize kernel (the closest analog to ggml's q4→f32 cpy) uses a
  healthy 2D work-group `(GroupN=16, GroupK=16)` = 256 work-items.
- All other quantize/copy/KV-cache/MoE kernels use explicit group sizes of
  16–1024.
- **One** low-severity finding: `MoEPrepareInputs.cpp:169` launches
  `compute_expert_blockscale_offsets_sycl_K_T` as a single work-item — but this
  is a tiny prefix-scan over `num_experts` (O(8–128) ops), sub-microsecond,
  dominated by launch overhead. A cleanup candidate, not a perf fix.

**Conclusion: the under-subscription bug class is not present in sgl-kernel-xpu.
There is no equivalent speedup to be had by porting PR #27160's fix.**

---

## The Real Finding: llama.cpp as an Alternative B70 Backend

While investigating PR #27160, the research surfaced that **llama.cpp has
complete, first-class support for Qwen3.6-35B-A3B** — and its GDN
implementation is structurally immune to the `!!!` bug.

### Architecture support

- Registered as `qwen35moe` (GGUF arch string), maps to HF class
  `Qwen3_5MoeForCausalLM`.
- Dedicated loader: `src/models/qwen35moe.cpp` — the 40-layer model is
  explicitly handled (`case 40: LLM_TYPE_35B_A3B`).
- 30 GDN + 10 GQA interleaving from `full_attn_interval = 4` — matches our model
  exactly.
- 256 experts / 8 routed + 1 shared: read from GGUF hparams, not hardcoded.
- MTP/NextN supported (`n_layer_nextn`, EAGLE3 spec decode via PR #24593).
- HF→GGUF converter: `conversion/qwen.py` registers
  `Qwen3_5MoeForCausalLM` → `QWEN35MOE`.
- Arch tests: `tests/test-llama-archs.cpp` exercises `LLM_ARCH_QWEN35MOE`.

### GDN implementation: fp32 end-to-end (no `!!!` bug)

GDN is a single fused custom op: `GGML_OP_GATED_DELTA_NET`. The fp32 invariant
is enforced at **three independent levels**:

1. **Cache allocation hardcoded fp32.** `src/llama-model.cpp` lines ~2316-2318:
   `recurrent_type_r = GGML_TYPE_F32`, `recurrent_type_s = GGML_TYPE_F32` for
   `LLM_ARCH_QWEN35MOE`. Not user-overridable — not wired to `-ctk/-ctv/-ctr`.
2. **The op refuses non-fp32.** `ggml_gated_delta_net()` in `ggml.c` asserts
   `state->type == GGML_TYPE_F32` (and q/k/v/g/beta/output all fp32). A non-fp32
   state would `GGML_ABORT`, not silently drift.
3. **SYCL kernel is fp32 end-to-end.** `ggml-sycl/gated_delta_net.cpp` takes
   `const float *` for all inputs and `float *` for state/output. All register
   state, delta-rule updates, and warp reductions are `float`. No `sycl::half`
   path exists.

**This means the fp16 SSM state drift that causes `!!!` in sgl-kernel-xpu
cannot occur in llama.cpp for this model.** The GDN recurrent state is allocated
fp32, the op mandates fp32, and the SYCL kernel computes in fp32. The KV cache
for the 10 GQA layers *can* be f16 (standard attention KV, not GDN state), but
that's a different buffer.

### Sibling SYCL PRs that benefit Qwen3.6 on B70

PR #27160 is the cpy fix, but the real per-component speedups for our model come
from a cluster of sibling SYCL PRs (same author/hardware, Jun–Aug 2026):

| PR | Date | What it optimizes | B70 benchmark | Qwen3.6 component |
|---|---|---|---|---|
| [#20455](https://github.com/ggml-org/llama.cpp/pull/20455) | 2026-03-12 | Original SYCL `GATED_DELTA_NET` kernel | +9× PP on Arc770 (9B distill) | GDN layers (foundational) |
| [#23142](https://github.com/ggml-org/llama.cpp/pull/23142) | mid-Jul | MoE prefill: counting-sort expert row gather | +70% PP (Arc B60, Q4_K_S, 308→548 t/s) | 8 routed experts |
| [#25222](https://github.com/ggml-org/llama.cpp/pull/25222) | 2026-07-02 | Flash attention via XMX/oneDNN SDPA | ×1.21–4.26 PP (Xe2, p=512–80k) | 10 GQA full-attention layers |
| [#26643](https://github.com/ggml-org/llama.cpp/pull/26643) | 2026-08-05 | Fused GDN state writeback cpy | +1.2% tg (B70, Q4_K) | GDN state snapshots (MTP) |
| [#26779](https://github.com/ggml-org/llama.cpp/pull/26779) | 2026-08-08 | Fused mul_mat(gate)+mul_mat(up)+GLU for q4_K FFN | +2.8% tg, +12.4% at B=8 | Shared expert / dense FFN |
| [#25217](https://github.com/ggml-org/llama.cpp/pull/25217) | 2026-07-02 | Fused top-k MoE routing | — | MoE router |
| [#24452](https://github.com/ggml-org/llama.cpp/pull/24452) | 2026-06-11 | Reordered Q4_K/Q5_K/Q6_K MoE `MUL_MAT_ID` | — | Routed expert matmuls |
| [#23174](https://github.com/ggml-org/llama.cpp/pull/23174) | — | SYCL GDN K>1 (multi-snapshot) | — | MTP draft writeback |
| [#26612](https://github.com/ggml-org/llama.cpp/pull/26612) | — | Coalesce ssm_conv window loads | — | GDN conv1d stage |
| [#26015](https://github.com/ggml-org/llama.cpp/pull/26015) | — | Fuse RMS_NORM+MUL | — | GDN sub-ops |
| [#26411](https://github.com/ggml-org/lllama.cpp/pull/26411) | — | Fuse UNARY silu/sigmoid/softplus + MUL | — | GDN gate/alpha/beta |

The CUDA counterpart of #26643 (PR #23940) was benchmarked on
**Qwen3.6-35B-A3B-UD-Q4_K_M**: +3% decode, +4% with MTP.

### Quantization

Q4_K / Q4_K_M / Q4_K_S / Q8_0 all available for `qwen35moe`. The 35B-A3B at
Q4_K_M is ~18-20 GB, fits comfortably in one B70's 32 GB. Multiple SYCL PRs
benchmark exactly this quant on B70/B60.

---

## Actionable Paths

### Path A: Stay on sglang XPU FP8 (current, ~95 tok/s)

Our current path works and is fast. The `!!!` bug is worked around via env vars
(fp32 SSM state, GDN ESIMD disabled). The permanent fix (rebuild sgl-kernel-xpu
with fp32 SSM state in the ESIMD kernel) remains the path to re-enabling the
ESIMD GDN prefill speedup.

**PR #27160 does not help this path** — different kernel stack, no equivalent bug.

### Path B: llama.cpp SYCL Q4_K on B70 (new alternative)

llama.cpp offers a structurally different approach:
- **No `!!!` bug possible** — GDN state is fp32 at three enforcement levels.
- **Q4_K_M** instead of FP8 — different quality/speed tradeoff.
- **MTP via EAGLE3** (PR #24593) — speculative decoding works on SYCL, unlike
  our blocked sglang XPU MTP path.
- **Single-card** — 18-20 GB Q4_K fits one B70, no TP needed. Or TP=2 for larger
  context / batch.
- Benefits from all the sibling SYCL PRs above.

**Tradeoff**: llama.cpp server (llama-server) has a different API than sglang
(OpenAI-compatible but fewer features — no radix cache, no chunked prefill
scheduler, no continuous batching at the same sophistication). For raw throughput
on a single model it's competitive; for production serving, sglang's scheduler
is more mature.

### Path C: Benchmark both, pick the winner

The honest answer is we don't know the llama.cpp Q4_K throughput on B70 for
this exact model without running it. Given:
- sglang XPU FP8: ~95 tok/s (known)
- llama.cpp SYCL Q4_K: unknown, but the PR benchmarks suggest it's competitive
  (Q4_K_S on B60 hits 548 t/s prefill for 35B-A3B; decode numbers not published
  for B70 specifically)

A benchmark is the right next step. The `llamacpp/` directory in this repo
contains a launch script for the B70 path.

---

## What does NOT apply

1. **PR #27160's specific fix** (cpy work-group sizing) — not applicable to
   sgl-kernel-xpu, which doesn't have the bug.
2. **The q4_0 → f32 7.83× speedup** — only affects Q4_0 quant, which neither
   our sglang FP8 path nor a Q4_K llama.cpp path would use.
3. **Porting the fix to sgl-kernel-xpu** — no equivalent kernel to port to.

## What DOES apply

1. **The llama.cpp SYCL backend as a whole** — complete Qwen3.6-35B-A3B support
   with fp32-safe GDN, MTP, and active SYCL optimization for every model
   component.
2. **The sibling PR cluster** — GDN writeback fusion (#26643), MoE prefill
   (#23142), GQA flash attention (#25222), fused FFN GLU (#26779) all benefit
   Qwen3.6 on B70 directly.
3. **The fp32 state enforcement pattern** — ggml's three-level fp32 invariant is
   the correct architectural pattern that sgl-kernel-xpu should adopt. Our
   identified fix (patch sgl-kernel-xpu ESIMD kernel to read fp32 state) is
   exactly what ggml already does structurally.
4. **MTP on XPU via llama.cpp** — EAGLE3 spec decode works on SYCL in
   llama.cpp (#24593), unblocking the speculative decoding path that's blocked
   in sglang XPU.

---

## References

- [PR #27160 — sycl: fix thread/block count in quantized cpy kernel launches](https://github.com/ggml-org/llama.cpp/pull/27160)
- [PR #26643 — sycl: fuse the gated-delta-net state writeback cpy](https://github.com/ggml-org/llama.cpp/pull/26643)
- [PR #23940 — CUDA: Remove redundant CUDA copies after gated_delta_net](https://github.com/ggml-org/llama.cpp/pull/23940) (CUDA original of #26643, benchmarked on 35B-A3B)
- [PR #23142 — sycl: improve MoE prefill throughput (+70% with Qwen3.6-35B)](https://github.com/ggml-org/llama.cpp/pull/23142)
- [PR #25222 — sycl: Flash Attention with XMX via oneDNN SDPA](https://github.com/ggml-org/llama.cpp/pull/25222)
- [PR #26779 — sycl: fuse mul_mat(gate)+mul_mat(up)+GLU for q4_K dense FFN](https://github.com/ggml-org/llama.cpp/pull/26779)
- [PR #24593 — spec: support eagle3 for qwen3.5 & 3.6](https://github.com/ggml-org/llama.cpp/pull/24593)
- [sgl-kernel-xpu repo](https://github.com/sgl-project/sgl-kernel-xpu) (audited, no equivalent bug found)
- [vllm-xpu-kernels PR #437](https://github.com/vllm-project/vllm-xpu-kernels/pull/437) (our identified permanent fix for sgl-kernel-xpu)
