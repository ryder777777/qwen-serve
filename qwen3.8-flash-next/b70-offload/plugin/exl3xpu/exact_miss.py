"""Exact decode miss path for the NVMe expert tier (docs/exact-miss.md).

Default tier decode masks a pick whose expert is in neither VRAM nor RAM (weight 0) and fills it in the background.
With EXL3_EXACT_MISS=1 (and the library built with MOE_EXACT=1, `_moe_a8x.so`) nothing is masked:

  ensure kernel   -> job (missed keys) in a USM host control region
  ExactSvc thread -> O_DIRECT Linux AIO reads into stage pages (or answers from the prefetch pool), done[j] = seq
  wait kernel     -> bounded spin on done[j], ptrs_all[key] = page; the GEMV reads it and writes it through to VRAM
  release kernel  -> uncommitted keys point back to RAM, rel[j] = seq (pages reusable)

Everything on the device side is a captured graph node; the host side is a native thread (no GIL). A timeout or a
read error falls back to the old masking for that call only (counted, logged; EXL3_EXACT_STRICT=1 turns it into an
error at the next stats check).

Optional next-layer prediction (EXL3_EXACT_PREDICT=1): in layer L's MoE call, layer L+a's router is applied to layer
L's MoE input (a = EXL3_EXACT_PREDICT_AHEAD) and its top-K picks that are in neither VRAM nor RAM are queued as hints;
the service prefetches them into an LRU pool so that the later demand job is answered without waiting for NVMe.

Env (read once at tier init, before graph capture):
  EXL3_EXACT_MISS=1            on (needs EXL3_NVTIER=1)
  EXL3_EXACT_JOBS=2            job slots (1..4); stage pages = JOBS * KMAX
  EXL3_EXACT_KMAX=40           distinct missed experts per call (graph decode M<=4 x top-10 = 40; max 64)
  EXL3_EXACT_SPIN_ITERS=20000  wait-kernel polls before the masked fallback (one PCIe read each, ~1 us: ~20 ms)
  EXL3_EXACT_PF_PAGES=64       prefetch pool pages (2 MiB each; 0 = no prefetch pool)
  EXL3_EXACT_SVC_SPIN_US=2000  service busy-polls this long after the last activity, then sleeps
  EXL3_EXACT_SVC_SLEEP_US=50   sleep step when idle (worst added latency of the first miss after an idle period)
  EXL3_EXACT_PREDICT=0         next-layer prediction hints
  EXL3_EXACT_PREDICT_K=16      predicted experts per row (over-predict: only non-resident ones are read)
  EXL3_EXACT_PREDICT_AHEAD=1   layers ahead (1 or 2)
  EXL3_EXACT_STRICT=0          1: raise at the next stats check if any call fell back to masking
  EXL3_EXACT_POLL=3            how the wait kernel reads the host-written done/status/addr words (host_alloc USM host):
                               3 = uncached read hint, annotated_ptr (GLM FTX_POLL=2 pattern), 1 = system atomic RMW
                               (fetch_add 0), 0 = acquire fence + acquire load, 2 = relaxed load (0 and 2 read stale
                               values on B70: runs 1-2). Runtime value (xdev), switchable with set_poll_mode().
"""
from __future__ import annotations

import ctypes
import logging
import mmap
import os
import re
import time

import numpy as np
import torch

from .moe_offload import s64

log = logging.getLogger("exl3xpu")
MiB2 = 2 << 20
libc = ctypes.CDLL(None, use_errno=True)

# keep in sync with kernels/moe-a8/exact_miss.h
CTL_BYTES, HSTAT_OFF, HSTAT_N, KMAX, JMAX = 32768, 24576, 24, 64, 4
HLOG_OFF, LOG_N = 26624, 128
X_POLL_MODE, X_LOG = 12, 16 + 2 * KMAX
X_SIZE = X_LOG + LOG_N * 4
XDEV = ("seq", "j", "n", "layer", "jobs", "keys", "timeouts", "timeout_keys", "overflow_keys", "hint_calls", "hints",
        "hint_cnt", "poll_mode", "polls", "polls_max")
OUTCOME = {1: "ok", 2: "timeout", 3: "read_error"}
HSTAT = ("heartbeat_ns", "jobs", "keys", "demand_reads", "pf_hit_ready", "pf_hit_inflight", "pf_reads", "hints",
         "hints_dropped", "read_errors", "svc_ns", "svc_ns_max", "bytes", "eagain", "hints_known", "job_deferred",
         "stopped")


def _env(name, default, cast=int):
    return cast(os.environ.get(name, default))


class _Pages:
    """n 2 MiB-aligned pages of anonymous memory: O_DIRECT target (USM host memory is not: EFAULT) and read by the GPU
    through xe SVM. Populated and locked once, never punched, so the GPU-side mapping made at init stays valid."""

    def __init__(self, n):
        self.n = max(1, n)
        size = self.n * MiB2
        self.m = mmap.mmap(-1, size + MiB2)
        p = ctypes.addressof(ctypes.c_char.from_buffer(self.m))
        self.base = (p + MiB2 - 1) // MiB2 * MiB2
        libc.madvise(ctypes.c_void_p(self.base), ctypes.c_size_t(size), 14)          # MADV_HUGEPAGE
        ctypes.memset(self.base, 0, size)                                              # populate
        self.locked = libc.mlock(ctypes.c_void_p(self.base), ctypes.c_size_t(size)) == 0
        self.size = size

    def addr(self, i):
        return self.base + i * MiB2


class ExactMiss:
    def __init__(self, tier, store):
        X = store.X
        if not hasattr(X, "moe_exact_setup"):
            raise RuntimeError("EXL3_EXACT_MISS=1 needs the MoE library built with MOE_EXACT=1 (kernels/moe-a8/build.sh "
                               "-> _moe_a8x.so, EXL3_MOE_LIB=...)")
        if tier.rec > MiB2 or tier.rec % 4096:
            raise RuntimeError(f"exact: store record {tier.rec} B must be a 4 KiB multiple <= 2 MiB")
        self.X, self.tier, self.store, self.E = X, tier, store, tier.E
        self.jobs = _env("EXL3_EXACT_JOBS", "2")
        self.kmax = _env("EXL3_EXACT_KMAX", "40")
        self.spin = _env("EXL3_EXACT_SPIN_ITERS", "20000")
        self.pf_pages = _env("EXL3_EXACT_PF_PAGES", "64")
        self.strict = os.environ.get("EXL3_EXACT_STRICT", "0") == "1"
        assert 1 <= self.jobs <= JMAX and 1 <= self.kmax <= KMAX, (self.jobs, self.kmax)
        dev = tier.dev
        self.ctl = X.host_alloc(CTL_BYTES)
        self.ctl.zero_()
        self.hstat = self.ctl[HSTAT_OFF:HSTAT_OFF + HSTAT_N * 8].view(torch.int64)     # shares the USM host memory
        self.xdev = torch.zeros(X_SIZE, dtype=torch.int64, device=dev)
        self.xdev[X_POLL_MODE] = _env("EXL3_EXACT_POLL", "3")
        self.hlog = self.ctl[HLOG_OFF:HLOG_OFF + LOG_N * 32].view(torch.int64).view(LOG_N, 4)
        t = time.perf_counter()
        self.stage = _Pages(self.jobs * self.kmax)
        self.pf = _Pages(self.pf_pages)
        # map every page on the GPU side once (a8 moe_svm_prefetch): first touch inside a kernel costs ~5-8 ms per page
        addrs = [self.stage.addr(i) for i in range(self.stage.n)] + [self.pf.addr(i) for i in range(self.pf.n)]
        X.moe_svm_prefetch(torch.tensor(addrs, dtype=torch.int64), MiB2, tier.res_dev)
        torch.xpu.synchronize()
        X.moe_exact_setup(self.ctl, self.xdev, self.jobs, self.kmax, self.spin)
        X.moe_exact_start(tier.nfd, tier.rec, torch.as_tensor(np.asarray(tier.ckpt, dtype=np.int64)), self.E,
                          s64(self.stage.base), s64(self.pf.base), self.pf_pages, MiB2,
                          _env("EXL3_EXACT_SVC_SPIN_US", "2000"), _env("EXL3_EXACT_SVC_SLEEP_US", "50"))
        self.running = True
        # prediction
        self.predict = os.environ.get("EXL3_EXACT_PREDICT", "0") == "1"
        self.pred_k = _env("EXL3_EXACT_PREDICT_K", "16")
        self.ahead = _env("EXL3_EXACT_PREDICT_AHEAD", "1")
        self.gates: dict = {}                            # checkpoint layer -> router module
        self.li_of_ckpt = {int(c): li for li, c in enumerate(tier.ckpt)}
        self._last_dev = None
        self._last_host = None
        log.warning("exl3xpu: exact decode miss path on: %d job slots x %d keys, %d prefetch pages, stage+pool %.2f GB "
                    "(locked %s/%s), wait limit %d polls, prediction %s (K=%d, ahead %d), setup %.1f s", self.jobs,
                    self.kmax, self.pf_pages, (self.stage.size + self.pf.size) / 2 ** 30, self.stage.locked,
                    self.pf.locked, self.spin, self.predict, self.pred_k, self.ahead, time.perf_counter() - t)

    # ---- prediction
    def register_gates(self, model) -> int:
        """Find each decoder layer's router (`layers.N.mlp.gate`, out_features = E) for next-layer prediction."""
        pat = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.gate$")
        for name, mod in model.named_modules():
            m = pat.search(name)
            if m is None or name.startswith("mtp") or ".mtp." in name:
                continue
            w = getattr(mod, "weight", None)
            out = getattr(mod, "output_size", None) or getattr(mod, "out_features", None) or (w.shape[0] if w is not None and w.dim() == 2 else None)
            if out == self.E:
                self.gates[int(m.group(1))] = mod
        log.warning("exl3xpu: exact prediction: %d router gates registered", len(self.gates))
        return len(self.gates)

    def hint_next(self, li: int, x: torch.Tensor) -> None:
        """Layer li's MoE input x [M, H] -> router of layer ckpt[li] + ahead -> hints. Graph-capturable (torch ops +
        one single-work-item kernel)."""
        if not self.predict or not self.gates:
            return
        target = int(self.tier.ckpt[li]) + self.ahead
        g = self.gates.get(target)
        tl = self.li_of_ckpt.get(target)
        if g is None or tl is None:
            return
        logits = g(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        ids = torch.topk(logits.float(), min(self.pred_k, self.E), dim=-1).indices.to(torch.int32).contiguous()
        self.X.moe_exact_hint(ids, self.store.slot_of_dev, tl, self.E)

    def set_poll_mode(self, m: int) -> None:
        """Stream-ordered: calls queued after this use mode m (graphs included: the wait kernel reads it each call)."""
        self.xdev[X_POLL_MODE].fill_(int(m))

    # ---- debug: per-job join of the device log (outcome, polls, last done value seen) and the host log
    def debug_jobs(self) -> dict:
        d = self.xdev[X_LOG:X_LOG + LOG_N * 4].view(LOG_N, 4).cpu().numpy()      # one sync
        h = self.hlog.numpy().copy()
        host = {int(r[0]): r for r in h if r[0] > 0}
        jobs = []
        for r in d:
            seq = int(r[0])
            if seq <= 0:
                continue
            hr = host.get(seq)
            t_seen, t_done = (int(hr[2]), int(hr[3])) if hr is not None else (0, 0)
            jobs.append(dict(seq=seq, n=int(r[1]) & 0xFFFF, outcome=OUTCOME.get(int(r[1]) >> 16, "?"), polls=int(r[2]),
                             last_seen=int(r[3]), host_seen=hr is not None, host_answered=t_done > 0,
                             host_us=(t_done - t_seen) / 1e3 if t_done else None))
        jobs.sort(key=lambda x: x["seq"])
        to = [x for x in jobs if x["outcome"] != "ok"]
        okj = [x for x in jobs if x["outcome"] == "ok"]
        hus = [x["host_us"] for x in jobs if x["host_us"] is not None]
        summ = dict(
            poll_mode=int(self.xdev[X_POLL_MODE].item()), jobs_logged=len(jobs), ok=len(okj), not_ok=len(to),
            # the key diagnostic: the host answered (done[j] = seq written) but the GPU never read that value
            not_ok_host_answered=sum(x["host_answered"] for x in to),
            not_ok_host_never_saw_job=sum(not x["host_seen"] for x in to),
            not_ok_last_seen_older_seq=sum(0 < x["last_seen"] < x["seq"] for x in to),
            not_ok_by_n={n: sum(x["n"] == n for x in to) for n in sorted({x["n"] for x in to})},
            ok_polls_median=float(np.median([x["polls"] for x in okj])) if okj else None,
            host_us_median=float(np.median(hus)) if hus else None, host_us_max=max(hus) if hus else None)
        return dict(summary=summ, last=jobs[-8:])

    # ---- stats
    def host_stats(self) -> dict:
        h = self.hstat.numpy().copy()
        return {k: int(h[i]) for i, k in enumerate(HSTAT)}

    def dev_stats(self) -> dict:
        d = self.xdev[:len(XDEV)].cpu().numpy()                 # one sync; called every log interval only
        return {k: int(d[i]) for i, k in enumerate(XDEV)}

    def log_stats(self, logger, steps: int) -> None:
        d, h = self.dev_stats(), self.host_stats()
        bd = self._last_dev or {k: 0 for k in d}
        bh = self._last_host or {k: 0 for k in h}
        dd = {k: d[k] - bd[k] for k in d}
        dh = {k: h[k] - bh[k] for k in h}
        n = max(1, steps)
        jobs = max(1, dh["jobs"])
        age = (time.clock_gettime_ns(time.CLOCK_MONOTONIC) - h["heartbeat_ns"]) / 1e6
        logger.info("exl3xpu: exact last %d steps: misses/step %.2f jobs/step %.2f | answered from pool %.0f%% "
                    "(ready %d, in flight %d), NVMe demand reads/step %.2f, prefetch reads/step %.2f | mean answer "
                    "%.0f us, max %.0f us | fallbacks (masked) %d calls / %d keys, overflow %d, read errors %d | "
                    "hints/step %.1f dropped %d | polls/job %.0f (max %d, mode %d) | service heartbeat %.1f ms ago",
                    n, dd["keys"] / n, dd["jobs"] / n,
                    100.0 * (dh["pf_hit_ready"] + dh["pf_hit_inflight"]) / max(1, dh["keys"]), dh["pf_hit_ready"],
                    dh["pf_hit_inflight"], dh["demand_reads"] / n, dh["pf_reads"] / n, dh["svc_ns"] / jobs / 1e3,
                    h["svc_ns_max"] / 1e3, dd["timeouts"], dd["timeout_keys"], dd["overflow_keys"], dh["read_errors"],
                    dd["hints"] / n, dh["hints_dropped"], dd["polls"] / max(1, dd["jobs"]), d["polls_max"], d["poll_mode"],
                    age)
        if age > 1000 and self.running:
            logger.error("exl3xpu: exact service heartbeat is %.0f ms old: every miss will time out and be masked", age)
        self._last_dev, self._last_host = d, h
        if self.strict and (d["timeouts"] or d["overflow_keys"]):
            raise RuntimeError(f"EXL3_EXACT_STRICT: {d['timeouts']} masked fallbacks ({d['timeout_keys']} keys), "
                               f"{d['overflow_keys']} overflow keys since start")

    def close(self) -> list:
        if not self.running:
            return []
        torch.xpu.synchronize()
        r = list(self.X.moe_exact_stop())
        self.X.moe_exact_setup(torch.zeros(0, dtype=torch.uint8), torch.zeros(0, dtype=torch.int64), 1, 1, 1)
        self.running = False
        return r
