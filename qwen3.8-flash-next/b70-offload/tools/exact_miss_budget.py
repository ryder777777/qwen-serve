#!/usr/bin/env python3
"""Exact decode miss path: trace-driven miss simulation + per-token cost model (docs/exact-miss.md, section 4).

sim:    replays real decode routing (K02 traces: decode_ids [T, 48, 10], natural prose/code/long-context answers,
        thinking on and off) through a model of the tier in EXACT mode: one VRAM arena of S slots shared by all
        layers, exact LRU (ensure kernel), every VRAM miss written through; VRAM victims go to a RAM tier of R experts
        (LRU, exclusive: a RAM hit moves the expert to VRAM) when write-back / the victim ring is on; a pick in neither
        is an NVMe miss (read exactly, not masked). Requests run back to back, state carried over, the first
        --warm tokens are not counted. MTP verify: M consecutive tokens per call, advancing by the accepted count.
        Prefill rows are not replayed: staged prefill (STAGE=1) streams experts through its own buffers and does not
        touch the VRAM slots or the RAM tier.
cost:   ms/token = core + t_pcie * V + t_nvme * (1 - r) * N + t0 * L_unhidden(r) + t_rt * (L - L_unhidden(r))
                       + 2 * t_node * 48 / tokens_per_step
        V = VRAM-missed picks per token (each is read over PCIe by the GEMV: RAM tier or stage page), N = NVMe
        misses per token, L = MoE calls per token with >= 1 NVMe miss. Prediction recall r hides a predicted miss's
        NVMe time; a call still waits t0 if any of its misses was not predicted (1 - r^n for n misses).

usage: exact_miss_budget.py sim --traces DIR [--slots 8000 --ram 1024 --mtp 0] ...   (prints JSON lines)
       exact_miss_budget.py table --traces DIR > t.jsonl                             (all configs of the doc)
       exact_miss_budget.py report --jsonl t.jsonl                                   (tok/s grid)
"""
import argparse, glob, json, os
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

E, L, TOPK = 512, 48, 10
BLOB = 1_862_400


def load(traces):
    out = []
    for f in sorted(glob.glob(os.path.join(traces, "*.npz"))):
        z = np.load(f)
        if "decode_ids" in z.files and z["decode_ids"].shape[0] > 0:
            out.append((os.path.basename(f)[:-4], z["decode_ids"].astype(np.int32)))
    return out


def sim(traces, slots, ram, victims=True, mtp_m=1, accept=(1,), warm=3000, recalls=(0.0, 0.5, 0.7, 0.85)):
    data = load(traces)
    vram, rt = OrderedDict(), OrderedDict()
    lay = (np.arange(L, dtype=np.int64) * E)[None, :, None]
    tok = steps = 0
    V = N = 0
    calls_hit = 0
    l_unhid = {r: 0.0 for r in recalls}
    hist = np.zeros(64, dtype=np.int64)
    ai = 0
    for name, ids in data:
        keys = ids.astype(np.int64) + lay                     # [T, 48, 10]
        T, p = keys.shape[0], 0
        while p + mtp_m <= T:
            count = tok >= warm
            for l in range(L):
                ks = set(keys[p:p + mtp_m, l, :].ravel().tolist())
                miss = []
                for k in ks:
                    if k in vram:
                        vram.move_to_end(k)
                    else:
                        miss.append(k)
                n_nv = 0
                for k in miss:
                    if k in rt:
                        del rt[k]
                    else:
                        n_nv += 1
                    if len(vram) >= slots:
                        old, _ = vram.popitem(last=False)
                        if victims and ram > 0:
                            rt[old] = None
                            if len(rt) > ram:
                                rt.popitem(last=False)
                    vram[k] = None
                if count:
                    V += len(miss); N += n_nv
                    hist[min(n_nv, 63)] += 1
                    if n_nv:
                        calls_hit += 1
                        for r in recalls:
                            l_unhid[r] += 1.0 - r ** n_nv  # P(at least one of the call's misses was not predicted)
            adv = accept[ai % len(accept)] if mtp_m > 1 else 1
            ai += 1
            p += adv
            if count:
                tok += adv; steps += 1
            else:
                tok += adv
    counted_tokens = max(1, tok - warm)
    return dict(slots=slots, ram=ram, victims=victims, mtp_m=mtp_m, accept=list(accept),
                tokens=int(counted_tokens), steps=int(steps),
                V_tok=V / counted_tokens, N_tok=N / counted_tokens, L_tok=calls_hit / counted_tokens,
                L_unhidden_tok={str(r): l_unhid[r] / counted_tokens for r in recalls},
                N_step=N / max(1, steps), V_step=V / max(1, steps), L_step=calls_hit / max(1, steps),
                calls_with_miss_hist=hist[:12].tolist())


def cost(t, core_ms, r=0.0, t_pcie=0.085, t_nvme=0.093, t0=0.20, t_rt=0.02, t_node=0.008, t_pred=0.025):
    """ms per token for sim result t in exact mode.
    core_ms: per-token time without expert PCIe reads and without NVMe waits (attention/GDN/HC/dense, resident MoE,
             host bubble). t_pcie: one VRAM-missed pick read by the GEMV over PCIe (1.86 MB at ~22 GB/s).
    t_nvme:  bandwidth share of one 1.86 MB read on the array (~20 GB/s with parallel reads). t0: first-byte latency of
             a call's reads + AIO submit + two PCIe polls. t_rt: host round trip of a call whose misses were all
             prefetched. t_node: the wait + release graph nodes of every MoE call (48 per step). t_pred: the
             predictor's graph nodes per MoE call when prediction is on (router GEMV 2.6 MB + top-k + hint kernel)."""
    lu = t["L_unhidden_tok"][str(r)]
    tps = t["tokens"] / max(1, t["steps"])
    return (core_ms + t_pcie * t["V_tok"] + t_nvme * (1 - r) * t["N_tok"] + t0 * lu + t_rt * (t["L_tok"] - lu)
            + 2 * t_node * L / tps + (t_pred * L / tps if r > 0 else 0.0))


def report(path, core=(27.5, 32.5), mtp_factor=0.743, t0s=(0.20, 0.35)):
    """Cost grid from `table` output. core: per-token core ms without MTP (srv26-like no-write-back step minus its
    expert reads, and a pessimistic +5 ms); with MTP the core is scaled by the measured masked-mode MTP gain
    (30.9 -> 41.6 tok/s at 262k: x0.743 per token)."""
    rows = [json.loads(l) for l in open(path)]
    print("| slots | RAM tier | MTP | NVMe misses/token | calls with a miss/token | recall | t0 ms | tok/s (core lo) | tok/s (core hi) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for t in rows:
        m = t["mtp_m"] > 1
        for r in (0.0, 0.7):
            for t0 in t0s:
                c = [cost(t, cm * (mtp_factor if m else 1.0), r=r, t0=t0) for cm in core]
                print(f"| {t['slots']} | {t['ram']} | {'M=4, 2.6/step' if m else 'off'} | {t['N_tok']:.1f} | "
                      f"{t['L_tok']:.1f} | {r} | {t0} | {1000 / c[0]:.1f} | {1000 / c[1]:.1f} |")


def _run(kw):
    return sim(**kw)


def table(traces, jobs):
    # RAM-tier sizes: 2 GB (N130 nvme16) = 1024, 7 GB (nvme32) = 3584 experts; targets 12 GB -> ~0.25 GB tier (128),
    # 28 GB -> ~14 GB tier (7000). Slots: 8000 today (131k ctx); 6600 with 262k context (+1.5 GiB fp8 KV) and the MTP
    # layer's 512 experts pinned in VRAM (n113).
    cfgs = []
    for s in (8000, 6600):
        for r in (1024, 3584, 128, 7000):
            cfgs.append(dict(traces=traces, slots=s, ram=r, victims=True, mtp_m=1))
            cfgs.append(dict(traces=traces, slots=s, ram=r, victims=True, mtp_m=4, accept=(3, 2, 3, 3, 2)))  # mean 2.6
    with ProcessPoolExecutor(jobs) as ex:
        res = list(ex.map(_run, cfgs))
    for r in res:
        print(json.dumps(r), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sim", "table", "report"])
    ap.add_argument("--jsonl", help="report: output of `table`")
    ap.add_argument("--traces")
    ap.add_argument("--slots", type=int, default=8000)
    ap.add_argument("--ram", type=int, default=1024)
    ap.add_argument("--no-victims", action="store_true")
    ap.add_argument("--mtp", type=int, default=1, help="verify rows per step (1 = no MTP)")
    ap.add_argument("--accept", default="3,2,3,3,2", help="accepted tokens per verify step, cycled")
    ap.add_argument("--warm", type=int, default=3000)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    if a.mode == "sim":
        print(json.dumps(sim(a.traces, a.slots, a.ram, not a.no_victims, a.mtp,
                             tuple(int(x) for x in a.accept.split(",")), a.warm)))
    elif a.mode == "table":
        table(a.traces, a.jobs)
    else:
        report(a.jsonl)


if __name__ == "__main__":
    main()
