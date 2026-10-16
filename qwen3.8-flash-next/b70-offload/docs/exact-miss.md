# Exact decode miss path (no masking)

Status (2026-10-16, branch `exact-miss`): design plus a first implementation. The host service compiles and passes
its protocol test (Linux container, g++ 12). The SYCL kernel changes and the Python wiring have **not been compiled or
run**: there was no oneAPI toolchain or GPU available. The budget numbers come from replaying the real K02 decode
routing traces through a model of the tier. They are not measured server numbers.

## 1. Answer

- **What changes.** A decode pick whose expert is in neither VRAM nor RAM is no longer masked. The ensure kernel
  publishes the missed keys as a job in host-mapped memory. A native host thread reads them from NVMe with O_DIRECT
  Linux AIO into stage pages. A one-work-item wait kernel spins, with a time limit, until the reads have landed, and
  then points the GEMV at the stage pages. The GEMV writes them through into the VRAM slots ensure assigned, as it
  does today for RAM hits. All of these are graph nodes, so this works under `--cuda-graph-backend-decode full`. A
  call that hits the time limit falls back to today's masking for that call only. The fallback is counted, logged,
  and fatal under `EXL3_EXACT_STRICT=1`. It never hangs and never commits a slot whose data was not read.
- **Miss rates, from real routing (K02 traces, 75.8k decode tokens).** These count NVMe misses per token, with the
  exact path on:
  - **12 GB total**: RAM tier ≈ 0.25 GB. 40 misses/token with 8,000 VRAM slots; 55 with the 6,600 slots that 262k
    context plus pinned MTP experts leave.
  - **28 GB total**: RAM tier ≈ 14 GB. 7.5 misses/token at 8,000 slots; 10.7 at 6,600.
  - **Simulator check:** at today's 2 GB and 7 GB tiers it gives 32.6 and 18.0 per token. N130 measured 17-55
    (median ~40) and 9-28 (median ~20) on natural panel prompts.
- **Is ≥ 25 tok/s plausible?** The range in each row is pessimistic → optimistic core step (section 4.2), with an
  NVMe first-byte latency of 0.2 ms.
  - **12 GB, 262k, no MTP: no.** 20.6-22.9 tok/s, or 22.9-25.9 with 70%-recall next-layer prediction.
  - **12 GB with MTP (NEXTN 3 steps, verify M=4): yes, with a thin margin.** 26.7-29.7 tok/s without prediction,
    30.1-33.8 with it. If first-byte latency is 0.35 ms: 25.2-27.8.
  - **28 GB, no MTP: borderline.** 24.6-28.0 tok/s. Prediction barely helps here.
  - **28 GB with MTP: yes.** 31.7-35.9 tok/s.
  - **Prefill:** stays at ~1.1k tok/s, far above 500. The staged prefill path is already exact and is unchanged.
- **The CPU alternative is not worth it** (section 5). A missed expert still has to come from NVMe, the CPU saves at
  most ~35 µs of PCIe time per miss, and the expert never reaches VRAM, so it misses again next time.
- **Exact decode alone may not pass the quality band.** The teacher-forced reference panel goes through the prefill
  path, which never masks, and it already misses the band (top-1 0.985, KL 0.003). It gets worse on a second pass
  (KL 0.0076). The likely causes are the victim write-back race (N111: bad RAM pages) and the fp8 KV cache. The exact
  path must run with the victim ring or with no write-back (section 7). Test T6 includes the A/B that isolates fp8 KV.

## 2. Today (masked decode) in one paragraph

`launch_ensure` (kernels/moe-a8/exl3_moe.a8.sycl, patch a6) rewrites every pick with `slot_of[key] < 0 && !ram_res[key]`
to (safe expert, weight 0) and logs the key to the miss ring. `NvTierAsync.step` / the N130 admission step
(plugin/exl3xpu/nvtier.py) reads the ring one step later and fills the key from NVMe into its RAM-tier page. The next
pick of that expert is a RAM hit: the GEMV reads the page over xe SVM and writes it through into an LRU VRAM slot. The
output is approximate: masked/step was 18-135 at 16 GB and 5-60 at 32 GB, decode-vs-prefill KL 0.055 / 0.021, and
thinking-mode loops occurred.

## 3. Design

### 3.1 Data flow of one decode MoE call (all graph nodes, in-order queue)

```text
 router + top-k (SGLang)                                                         GPU
 [hint_next: router of layer L+a on this layer's input -> top-K -> moe_exact_hint]  (optional, EXL3_EXACT_PREDICT=1)
 ensure (1 WG of 1024)   hits re-stamped; picks in neither VRAM nor RAM -> xsel[e];          ┐
                         work-item 0 writes job j = seq % J: keys (ascending e), layer, n,   │ publishes to
                         release fence (system), job.seq = seq; then LRU slot assignment     │ USM host ctl
                         (fill_all[key] = slot) as before                                    ┘
 victim copy             (only with direct write-back; use the victim ring or NOVICT instead)
 WAIT (1 work-item)      n == 0: return. Else poll done[j].seq == seq (<= SPIN_ITERS PCIe reads, ~20 ms);
                         ok: ptrs_all[key] = addr[j][i] (stage / prefetch page);
                         timeout or status != 0: mask those picks, log to miss ring, roll back their slot
 gate/up GEMV, down GEMV  read ptrs[e] (VRAM slot, RAM page or stage page), write misses through to fill[e]
 combine                 commit ptrs[e] = slot for every filled expert
 RELEASE (1 work-item)   keys with no slot: ptrs back to the RAM-tier address; rel[j] = seq (system scope)

 ExactSvc (host thread, native, no GIL)                                                CPU
   poll hint count -> unseen keys -> LRU prefetch page -> AIO O_DIRECT read            (prefetch pool)
   poll job[j].seq -> per key: ready pool page | loading pool page (wait) | AIO read into stage page (j, i)
                   -> addr[j][i]; when the last read lands: done[j] = {status, seq} (release)
```

The control region is `exact::CTL_BYTES` = 32 KiB of `sycl::malloc_host` memory (`host_alloc`). It holds the job
slots, the done words, the release words, `addr[J][KMAX]`, the hint ring (1,024 keys) and the host statistics. The
device scalars are an XPU int64 tensor `xdev`: seq, current job slot, n, layer, the keys and addresses of the current
call, and the device statistics. The layout is defined once, in `kernels/moe-a8/exact_miss.h`.

### 3.2 Why this is safe

- **Stage pages.** Every job slot owns `KMAX` pages, page (j, i) at `stage + (j·KMAX + i)·2 MiB`.
  - The host starts a new job on slot j only after all reads of the previous job on that slot have completed.
  - The device publishes job seq+J only after job seq's GEMVs and release, because the queue is in order.
  - So a stage page is never written while a kernel reads it, and never read before its read completes.
- **Prefetch pages.** An address reaches the device only through a job answer, and the host is the only party that
  manages the pool.
  - A page is pinned by the last job (j, seq) that was answered with it.
  - It can be recycled only when `rel[j] >= seq`. Release order is global, so a later release covers earlier pins.
- **Timeout.** The wait kernel masks the picks of the job's keys (exactly the a6 fallback). If ensure gave such a key
  a slot this call, the kernel releases the slot (`slot_key = -1`, `slot_last = -1`, `slot_of = -1`) and sets
  `fill_all[key]` to the RAM-tier address. Combine then commits the host address, and no slot is left holding data
  that was not read. The key also goes to the miss ring, so the old background fill still happens.
- **Late answers.** A host answer that arrives after a timeout writes `done[j] = old seq`. The device waits for an
  exact seq match, so the stale answer is ignored.
- **Uncommitted keys.** A key that got no VRAM slot would keep pointing at a stage page. The release kernel restores
  its RAM-tier address. With `max_fill = E` and 8,000 slots this does not happen in practice; it is a guard.

### 3.3 Next-layer prediction (optional)

In layer L's MoE call, `ExactMiss.hint_next` applies the router of layer `ckpt[L] + a` to layer L's MoE input. The
router is found by `register_gates` as `layers.N.mlp.gate`, 512 outputs. Hints are issued as follows:

- **Selection.** Top-`EXL3_EXACT_PREDICT_K` (16) per row.
- **Filter.** `moe_exact_hint`, one work-item, drops the keys that are VRAM- or RAM-resident and appends the rest to
  the hint ring.
- **Timing.** The service reads them into the prefetch pool while layer L's MoE and layer L+1's attention/GDN run:
  about 0.6 ms per layer at C1, against about 0.3 ms for a 1.86 MB read.

When the demand job arrives, keys found in the pool are answered without NVMe; a miss the predictor did not cover still
pays the full read. The predictor's graph nodes cost about 25 µs per layer, about 1.2 ms per token, and the budget
includes that. Recall for this model (512 experts, top-10, hyper-connections) is unknown: test T5 measures it, and the
budget shows r = 0 and r = 0.7.

MTP draft tokens do not give main-model routing ahead of time, because the draft runs its own MTP layer. MTP helps in a
different way: one verify call (M = 4) pays the first-byte latency once for up to 40 picks, and the fixed per-token
cost is amortised over the accepted tokens.

### 3.4 RAM tier, victims and memory

| | 12 GB total | 28 GB total |
|---|---|---|
| process anon (scheduler 6.6, launcher 1.3, detok 1.0; N130 at 131k) | 9.5 GiB | 9.5 GiB |
| staged-prefill host ring (2 × 0.91) | 1.8 GiB | 1.8 GiB |
| exact stage pages (`JOBS 2 × KMAX 40 × 2 MiB`) | 0.16 GiB | 0.16 GiB |
| prefetch pool (`PF_PAGES` × 2 MiB) | 64 → 0.125 GiB | 256 → 0.5 GiB |
| pinned safe experts (48 × 2 MiB) | 0.09 GiB | 0.09 GiB |
| RAM expert tier (`EXL3_NVTIER_RAM_GB`) | **0.25 GiB (128 experts)** | **14 GiB (7,000 experts)** |
| slack (page cache, peaks) | ~0.1-0.3 GiB | ~1.5 GiB |

- **12 GB.** The tier cannot hold victims. Run with `EXL3_NVTIER_NO_VICTIM=1`: victims are dropped, which removes the
  write-back race and the ~0.9 ms per victim. Almost every VRAM miss is then an exact NVMe miss.
- **28 GB.** The tier is only useful as a victim cache. Use the victim ring (N111 `a9`/`a9b`, or N140 on branch
  `b70-speed`): the VRAM victim is parked in a VRAM ring and the host drains it into the RAM page, so only the host
  sets the resident flag. Without the ring, direct write-back costs ~0.9 ms per victim and has the race in section 7.

## 4. Budget

### 4.1 Misses per token (trace-driven, `tools/exact_miss_budget.py`)

Method:

- **Routing.** Real decode routing from the K02 traces: 22 requests (prose, code and long-context answers, thinking on
  and off, natural completion length), 75,811 counted tokens after 3,000 warm-up tokens.
- **Cache model.** Back-to-back requests; one 8,000- or 6,600-slot VRAM arena with exact LRU across layers (as
  `launch_ensure` does). Victims go to an exclusive LRU RAM tier of R experts.
- **Prefill.** Not replayed: staged prefill does not touch slots or the tier.
- **MTP.** M = 4 verify rows per call, advancing 2.6 tokens per step on average. Wasted rows use the true next tokens'
  routing, so the cost of rejected drafts is somewhat underestimated.

| VRAM slots | RAM tier (experts) | NVMe misses / token | VRAM misses / token (PCIe reads) | MoE calls with ≥ 1 NVMe miss / token | with MTP: misses / verify step, calls with a miss / token |
|---|---|---|---|---|---|
| 8,000 | 1,024 (today's 16 GB) | 32.6 | 41.2 | 19.0 | 84.9, 12.2 |
| 8,000 | 3,584 (today's 32 GB) | 18.0 | 41.2 | 12.1 | 46.8, 8.8 |
| **8,000** | **128 (12 GB)** | **40.0** | 41.2 | 21.8 | 104.1, 13.3 |
| **8,000** | **7,000 (28 GB)** | **7.5** | 41.2 | 5.9 | 19.6, 4.9 |
| 6,600 | 1,024 | 44.9 | 56.9 | 23.5 | 116.8, 14.0 |
| 6,600 | 3,584 | 25.0 | 56.9 | 15.7 | 65.1, 10.7 |
| **6,600** | **128 (12 GB, 262k + MTP)** | **55.2** | 56.9 | 26.7 | 143.8, 15.0 |
| **6,600** | **7,000 (28 GB, 262k + MTP)** | **10.7** | 56.9 | 8.0 | 27.9, 6.3 |

Validation against N130, masked mode, where `nvme/step` counts the distinct new keys filled per step:

| Config | Simulated | Measured, natural panel prompts | Measured, synthetic bench prompts |
|---|---|---|---|
| 2 GB tier | 32.6 | 17-55, median ~40 (`results/n130/p16/tier_log.txt`) | 15.7 (8k C1) |
| 7 GB tier | 18.0 | 9-28, median ~20 (`p32`) | 9.1 |

The bench prompts use a 10-word vocabulary and understate natural routing.

The VRAM hit rate in decode is 91% at 8,000 slots (41 of 480 picks miss). The "64% VRAM hit rate" in
`research/18-b70-ingest-ceiling.md` is the static top-8,000 share of the frequency prior (`serve/qwen_expert_prior.json`
gives 0.642), not a dynamic hit rate.

Distinct misses per call never exceed 40 (M = 4 × top-10), so `KMAX = 40` never overflows.

### 4.2 Cost per token

ms/token = core + 0.085·V + 0.093·(1−r)·N + t0·L_unhidden(r) + 0.02·(L − L_unhidden) + 2·0.008·48/tps (+ 0.025·48/tps
with prediction)

| term | value | source |
|---|---|---|
| core (no expert PCIe reads, no NVMe waits) | 27.5 ms (optimistic) / 32.5 ms (pessimistic) at C1 | srv26 (no write-back) 8k C1 34.5 tok/s, p50 27 ms; N130 16 GB C1 34.2 ms including ~0.9 ms per direct write-back |
| core with MTP | × 0.743 | masked-mode MTP gain measured at 262k: 30.9 → 41.6 tok/s (commit a115bd7) |
| PCIe read of a VRAM-missed pick | 0.085 ms | 1.86 MB at ~22 GB/s (SVM reads, N106/N109: 25.5-26.7 GB/s link) |
| NVMe bandwidth per miss | 0.093 ms | 1.86 MB at ~20 GB/s with parallel reads (prefill fills reach 21-24 GB/s, `tier_log.txt`) |
| t0: first byte + AIO + 2 PCIe polls | 0.20 ms (0.35 pessimistic) | estimate: md RAID0 splits a record into ~113 KB requests over 4 PCIe 5 drives; **T4 measures it** |
| wait + release nodes | 8 µs each, 48 calls per step | estimate |

Resulting decode tok/s at C1, pessimistic → optimistic core (`tools/exact_miss_budget.py report`):

| budget | slots | MTP | t0 0.2 ms, no prediction | t0 0.2 ms, recall 0.7 | t0 0.35 ms, no prediction | t0 0.35 ms, recall 0.7 |
|---|---|---|---|---|---|---|
| 12 GB | 8,000 | off | 22.3-25.1 | 24.2-27.6 | 20.8-23.2 | 23.4-26.5 |
| 12 GB | 8,000 | on | 29.1-32.7 | 32.1-36.4 | 27.5-30.7 | 31.0-35.0 |
| **12 GB** | **6,600** | **off** | **20.6-22.9** | **22.9-25.9** | 19.0-21.0 | 22.0-24.7 |
| **12 GB** | **6,600** | **on** | **26.7-29.7** | **30.1-33.8** | 25.2-27.8 | 28.8-32.3 |
| 28 GB | 8,000 | off | 25.9-29.7 | 25.9-29.7 | 25.3-29.0 | 25.7-29.4 |
| 28 GB | 8,000 | on | 33.8-38.6 | 34.4-39.5 | 33.0-37.5 | 34.1-39.0 |
| **28 GB** | **6,600** | **off** | **24.6-28.0** | **24.8-28.3** | 23.9-27.1 | 24.6-28.0 |
| **28 GB** | **6,600** | **on** | **31.7-35.9** | **32.6-37.1** | 30.8-34.8 | 32.2-36.6 |

- **12 GB.** Needs MTP to reach 25 tok/s. Without MTP, 262k context and exact misses cost 6-8 ms per token, and
  prediction wins back only ~2-3 ms.
- **28 GB.** Reaches 25 tok/s without MTP only with the optimistic core. With MTP the margin is 6-10 tok/s.
- **NVMe load.** It stays small: 55 misses/token at 30 tok/s is 3.1 GB/s, 12% of the array. Prediction over-reads
  add up to ~1-2× that.
- **C2/C4.** Each step carries 2-4 sequences with less sharing, so expect 2-4× the misses per step. Per-stream speed
  falls the way it does today; the aggregate does not collapse, because t0 is paid once per call.
- **Prefill.** Unchanged and exact today: 1,139-1,332 tok/s at 8k/32k in N130 (≥ 500).

## 5. Alternative: compute missed experts on the CPU

- **Method.** Read the record from NVMe (the same read), then run the trellis GEMV on the EPYC 7443P.
- **Speed.** The mul1 CPU decode is memory-bound at ~105 GB/s on 24 pinned cores. The container has 8 CPUs, so
  expect ~35 GB/s: ~50 µs per expert at M = 1, against the 85 µs PCIe read the GPU path pays.
- **What it would need.** The job protocol would carry x (5 KB per row) to the host and y back, and the GEMV would
  run the missed pair with weight 0 and add the CPU result. That is a second numerics path, since the CPU fp32
  formula is not the GPU fp16 FMA.
- **What it loses.** The expert is not written into VRAM. 91% of the time the next use of a decode expert hits VRAM,
  so skipping the write-through turns one miss into several.
- **Cost on CPUs.** It takes 4-8 of the container's 8 CPUs away from the scheduler. The exact GPU path needs one
  spinning core.
- **Verdict.** Roughly neutral on latency and worse on miss count. Revisit only if the PCIe link saturates at C4
  (V/step × 1.86 MB ≈ 300 MB per step at C4).

## 6. Changes on this branch

| file | function / symbol | change |
|---|---|---|
| `kernels/moe-a8/exact_miss.h` (new) | `exact::` layout constants, `XDev`, `HStat`, `SvcCfg`, `ExactSvc` | protocol; host service: one thread, Linux AIO (raw syscalls), O_DIRECT reads into stage and prefetch pages, LRU prefetch pool with release-based pinning, bounded drain on stop |
| `kernels/moe-a8/exl3_moe.a8.sycl` | `moe_exact_setup`, `moe_exact_start`, `moe_exact_stop`, `moe_exact_pause`, `moe_exact_hint` (new ops); `launch_ensure` (exact block replaces the a6 mask when set); `moe_forward_cached` (wait + release kernels); `TORCH_LIBRARY` | everything under `#ifdef EXL3_EXACT_MISS`. With the macro off, `unifdef` gives the a8 source plus one brace pair (the a6 block got its own scope) |
| `kernels/moe-a8/build.sh` | `MOE_EXACT=1` | adds `-DEXL3_EXACT_MISS`, output `_moe_a8x.so`. The default build is unchanged |
| `kernels/moe-a8/tests/exact_svc_test.cpp` (new) | | CPU protocol test of `ExactSvc` (T1) |
| `plugin/exl3xpu/exact_miss.py` (new) | `ExactMiss` (`__init__`, `register_gates`, `hint_next`, `host_stats`, `dev_stats`, `log_stats`, `close`), `_Pages` | allocation (2 MiB-aligned, populated, `mlock`, GPU-side `moe_svm_prefetch`), setup before capture, stats, strict mode |
| `plugin/sglang_plugin.tier.py` | `_nvtier_init`, `Exl3XpuMoEMethod.apply`, `_patch_nvtier_step`, `_patch_exact_gates` (new), `activate` | create `ExactMiss` after the tier warm-up and before capture; prediction hint before each decode MoE call; stats every `EXL3_NVTIER_LOG_EVERY` steps; router registration after `load_model` (target runner only) |
| `tools/exact_miss_budget.py` (new) | `sim`, `table`, `report`, `cost` | section 4 |
| `tests/exact_miss_gpu_test.py` (new, not run) | | T2-T4 |

`nvtier.py` is unchanged. In exact mode the miss ring only receives fallbacks, which the existing step fills as before.
Porting to the `a9`/N140 kernels (victim ring, branch `b70-speed`) means applying the same three insertion points:
the include and globals, the `launch_ensure` exact block, and the `moe_forward_cached` wait and release.

### Env switches

| variable | default | meaning |
|---|---|---|
| `EXL3_MOE_LIB` | – | must point at `_moe_a8x.so` (`MOE_EXACT=1 kernels/moe-a8/build.sh`) |
| `EXL3_EXACT_MISS` | 0 | 1 = exact decode misses (needs `EXL3_NVTIER=1`, `EXL3_MOE_CACHE=1`) |
| `EXL3_EXACT_JOBS` | 2 | job slots (1-4) |
| `EXL3_EXACT_KMAX` | 40 | distinct missed experts per call (≤ 64); more are masked and counted as overflow |
| `EXL3_EXACT_SPIN_ITERS` | 20000 | wait-kernel polls (~1 µs each) before the masked fallback |
| `EXL3_EXACT_PF_PAGES` | 64 | prefetch pool pages (2 MiB each); 0 = none |
| `EXL3_EXACT_SVC_SPIN_US` / `_SLEEP_US` | 2000 / 50 | service busy-polls this long after activity, then sleeps in steps |
| `EXL3_EXACT_PREDICT` | 0 | next-layer prediction hints |
| `EXL3_EXACT_PREDICT_K` / `_AHEAD` | 16 / 1 | predicted experts per row / layers ahead |
| `EXL3_EXACT_STRICT` | 0 | 1 = raise at the next stats check if any call fell back to masking or overflowed |
| `EXL3_EXACT_POLL` | 3 | how the wait kernel reads the host-written words (done seq, status, page addresses; all in `host_alloc` USM host memory): 3 = uncached read hint through `annotated_ptr` (L1/L2/L3 uncached, the GLM `ft_xpu.sycl` production setting), 1 = system atomic RMW (`fetch_add(0)`), 0 = acquire fence + acquire load, 2 = relaxed load. Modes 0 and 2 read stale values on B70 (runs 1-2). Kept in `xdev`, read every call, switchable after capture with `ExactMiss.set_poll_mode` |

Suggested settings:

- **12 GB:** `--memory 12g --memory-swap 12g`, `EXL3_NVTIER_RAM_GB=0.25`, `EXL3_NVTIER_NO_VICTIM=1`,
  `EXL3_EXACT_PF_PAGES=64`, MTP on.
- **28 GB:** `--memory 28g`, `EXL3_NVTIER_RAM_GB=14`, victim ring on, `EXL3_EXACT_PF_PAGES=256`.

## 7. Failure and timeout semantics

| event | what happens | visible as |
|---|---|---|
| answer later than `SPIN_ITERS` polls (~20 ms) | that call's missed picks are masked (the a6 fallback) and their slot assignment is rolled back; keys go to the miss ring and the tier step fills them in the background | `fallbacks (masked) N calls / K keys` in the stats line; `EXL3_EXACT_STRICT=1` raises |
| NVMe short read / EIO | status 1 in `done[j]`: same fallback; a failed prefetch page is dropped | `read errors` |
| service thread dead or not started | every call with a miss waits ~20 ms and then masks: slow, never hangs | heartbeat age logged as an error when > 1 s |
| > `KMAX` distinct misses in one call | only the overflow keys are masked (impossible for graph decode with M ≤ 4) | `overflow` |
| late answer after a fallback | ignored (seq mismatch); the slot's pages are reused only after its reads finished | – |
| stage page GPU mapping lost (driver invalidation) | still correct; the first GPU access refaults (~5-8 ms per page), so the latency spike can turn into a fallback | answer max µs, fallbacks |
| eager batches with M ≥ `EXL3_NVTIER_PREFILL_M` | unchanged: `ensure_for_prefill` / staged prefill, already exact | – |
| shutdown | `moe_exact_stop` drains in-flight AIO (each wait bounded to 1 s) before the pages go away | `stopped` = 1 |
| setup change after graph capture | refused while the service runs (captured kernels bake in the pointers) | `TORCH_CHECK` |

Two fixes are needed alongside exactness:

- **Write-back race.** With the a8 direct write-back, the GPU marks a written-back expert resident while the host
  still has its page queued for punching (N111: 50-52 bad rows, 8-10 bad pages). Exact decode would read those bad
  pages exactly. Use `EXL3_NVTIER_NO_VICTIM=1` (12 GB) or the victim ring (28 GB).
- **Prefill-path deviation.** It already misses the band without any masking: 0.985 / 0.003, and 0.979 / 0.0076 on
  a second pass. T6 attributes it.

## 8. Test plan

| # | test | where | pass |
|---|---|---|---|
| T0 | build | oneAPI image | `MOE_EXACT=0` gives `_moe_a8.so` the same size as today's (±8 B, as in N111); `MOE_EXACT=1` builds `_moe_a8x.so` |
| T1 | host service protocol (`kernels/moe-a8/tests/exact_svc_test.cpp`) | any Linux, no GPU | **done, PASS** (section 9) |
| T2 | exactness (`tests/exact_miss_gpu_test.py`) | B70, 4 real layers, 640 slots, 0-byte RAM tier, K02 routing | 2,400 calls bit-identical to the all-VRAM golden `moe_forward`; 0 fallbacks; slot audit 0 bad |
| T3 | fallback | same, service paused | 1 fallback; output == golden on the post-mask ids; masked ids == safe expert; 0 cold keys left holding a slot; exact again after resume |
| T4 | latency per call vs misses (1, 2, 4, 10, 20, 40 cold) | same | fit t0 and per-expert cost. Within the budget if t0 ≤ 0.35 ms and per expert ≤ 0.2 ms; otherwise rerun `report` with the measured values |
| T5 | prediction | server, `EXL3_EXACT_PREDICT=1` vs 0, natural prompts | pool answers ≥ 60% of keys; prefetch reads ≤ 2× misses; decode tok/s gain > 0 after the ~1.2 ms/token predictor cost |
| T6 | quality | server, exact, `EXL3_EXACT_STRICT=1`, NO_VICTIM (12 GB) / victim ring (28 GB) | teacher-forced ref panel top-1 ≥ 0.987 and mean KL ≤ 0.0013, on a fresh server **and** on a second pass; decode-vs-prefill (`serve/n130/decode_kl_n130.py`) KL ≤ 0.002, top-1 ≥ 0.99 (masked: 0.021-0.055); greedy panel compared with exllamav3 text; lab chat gate (thinking, "Name three primary colors") ends naturally in 5/5 runs, including right after start; 0 fallbacks. If the panel still misses: rerun with a bf16 KV cache to isolate fp8 KV |
| T7 | speed (perf table: prefill size, prefill, decode, concurrency, kv cache, gpu count) | 12 GB and 28 GB caps; 8k/32k C1/C2/C4 and 262k C1 with MTP | decode C1 ≥ 25 tok/s with MTP at 12 GB and 28 GB (and without MTP at 28 GB); prefill ≥ 500 tok/s; log misses/step and mean/max answer µs next to the table |
| T8 | memory | `--memory 12g` / `28g`, swap = memory | 0 OOM kills while serving; peak anon + shmem ≤ cap − 0.3 GiB |
| T9 | soak | 2 h mixed prose/code/long traffic | 0 fallbacks, heartbeat < 100 ms, no runaway generations |

## 9. What was verified here

- **T1, the host service, built with `-Wall -Wextra` (no warnings) and run in `node:22-bookworm` (g++ 12.2, aarch64,
  O_DIRECT on overlayfs) under Docker on macOS** (`results/exact-miss/exact_svc_test.txt`):
  - 3,631 jobs and 22,809 answered keys; every answered page holds exactly the requested record (0 bad).
  - The store-layer remap is exercised (reversed on purpose).
  - 20 of 20 jobs timed out while the service was paused, and none at any other time.
  - The prefetch pool answered 6,162 keys (5,000 ready, 1,162 in flight).
  - Shutdown drained cleanly.
  - The latencies in that run (mean 92 µs, max 4.5 ms) come from a VM disk and say nothing about the B70 box.
- **Simulation:** 16 configurations (section 4), 30 s on the Mac. Raw output in `results/exact-miss/sim_table.jsonl`;
  the full tok/s grid is in `report.md`.
- **Python:** all files parse and pyflakes is clean on the new files.
- **SYCL:** braces and parentheses balance; the macro-off source matches a8 (unifdef). **Not compiled.**

## 9b. GPU run 1 (2026-10-16, card 83, image d30df691, `_moe_a8x.so`) and fixes

- **Result.** T2 had 0 mismatches and a slot audit of 0, but 884 of 1,365 jobs fell back. T4 hit the ~20 ms wait limit
  on every call with ≥ 4 misses, even though the host answered every job within 9.5 ms (0.19 ms per key).
- **Cause.** The wait kernel polled `done[j]` with a *relaxed* system-scope load. Once the line was cached on the GPU,
  later polls kept returning the stale value. Answers that landed before the first poll were seen (1-2 misses, ~0.5 ms
  of reads, median 0.8 ms); answers that landed after it never were.
- **Fix.**
  - Poll modes 0 (acquire fence + acquire load per poll, default) and 1 (atomic RMW).
  - The answered addresses are read the same way: a stale cached address would have been silent corruption once
    prefetch pages vary.
  - Per-job debug logs:
    - device side: outcome, polls, last done value seen;
    - host side: t_seen and t_done.
  - `ExactMiss.debug_jobs()` joins the two logs and counts "host answered but GPU never saw it".
- **T3.** The test chose "cold" experts from the host residency view. The GPU's direct victim write-back had flagged
  some of them RAM-resident on the device only, so 8 of the 10 were legitimate RAM hits: they kept their slots and were
  not masked. The test now uses the device view and runs with `EXL3_NVTIER_NO_VICTIM=1`. The masked-ids and
  slot-rollback checks are part of PASS.

- **Run 2 (ca9a157).** T3 passes. T2 still had 543 of 1,125 jobs fall back. Every failed job had host_seen and
  host_answered (host median ~660 µs, max ~10 ms), while the GPU's last_seen stayed at an older seq for all 20,000
  polls. So mode 0 (acquire fence + load) is just as stale on Xe2. The default is now mode 3, an uncached read hint
  through `annotated_ptr`: the GLM pattern on this card, with 0 spin timeouts in ~80k layer calls. It is used for the
  done word, the status and every page address.

## 10. Open risks

1. **SYCL not compiled or run.**
   - Device stores of job keys to USM host memory with a system-scope fence, then a seq store, are taken as visible
     in order to the host. That is the same pattern as the GLM host-flag polling and the copy-engine prefetch `host_plan`, but not this
     exact kernel.
   - Single-work-item spin kernels under graph replay are assumed to be fine; the GLM campaign proved that on B70.
2. **Stage and pool pages are anonymous THP memory read by the GPU over xe SVM.** The design assumes that
   `moe_svm_prefetch` at init plus `mlock` keeps the GPU mapping across O_DIRECT DMA writes. If the driver invalidates
   it (compaction, migration), each first access refaults for ~5-8 ms. That costs latency, not correctness. The fallback
   design is USM host stage pages with an anonymous bounce buffer and a host memcpy (~0.1-0.2 ms more per miss).
3. **t0 is unmeasured.** Decode misses queue behind staged-prefill reads if a prefill chunk's read-ahead is in flight:
   32 readers at 22-24 GB/s. That could push single calls past the limit and into the fallback. Watch the fallbacks
   in T7 at C2/C4 with mixed prefill. The mitigation is a separate AIO priority (`IOCB_FLAG_IOPRIO`) or a smaller
   prefill reader pool during decode.
4. **Miss rates come from single-stream K02 traces** (22 requests, one machine). Real multi-user traffic shares fewer
   experts per step.
5. **Prediction recall is unknown** (hyper-connections, 512 experts). The 12 GB no-MTP target depends on it, and even
   with it stays below 26 tok/s.
6. **The 12 GB fixed memory is tight.** The 9.5 GiB anon was measured at 131k context without MTP. If 262k + MTP adds
   more than ~0.3 GiB, the RAM tier goes to zero (N rises from 55 toward ~57) or the prefill ring has to shrink.
7. **The quality band may still fail** for reasons unrelated to masking: the fp8 KV cache, the write-back race, and
   kernel numerics against exllamav3. See T6.
8. **The MTP gain (×0.743 per token) is one measurement**, in masked mode, from the a115bd7 commit message. The
   acceptance rate under exact routing can differ.
9. **One CPU core spins during decode** (the service), out of the container's 8.
