# Exact decode miss path: standalone GPU test (docs/exact-miss.md, test plan T2-T4).
# Run 1 (2026-10-16, card 83, poll mode = relaxed load): T2 0 mismatches but 884/1365 jobs timed out; T4 >= 4 misses
# always hit the 20 ms limit. Cause: the wait kernel's relaxed load kept returning a cached done word (fix: poll modes
# 0/1, EXL3_EXACT_POLL); T3 "failures" came from picking "cold" experts with the stale host residency view while the
# GPU's victim write-back had flagged some RAM-resident (fix: device view, no write-back in this test).
# Each phase now prints ex.debug_jobs(): per job outcome, GPU polls, last done value the GPU saw, host answer time.
# Run 2 (ca9a157): T3 PASS; mode 0 still stale (GPU last_seen stuck at an older seq while the host had answered).
# Now the default is mode 3 (uncached read hint, annotated_ptr), as in the GLM ft_xpu.sycl production setting.
# Needs: B70, the MoE library built with MOE_EXACT=1 (EXL3_MOE_LIB=.../_moe_a8x.so), the packed NVMe store + meta,
# K02 routing traces. Mounts like experimental/n111-victim-ring/run.sh: /pkg/exl3xpu (plugin/exl3xpu), /s store, /t traces.
#   T2 exact   : small VRAM arena (SLOTS) and a 0-byte RAM tier, trace routing, every call vs a golden all-VRAM
#                moe_forward: outputs must be bit-identical, 0 fallbacks; slot audit: every VRAM-resident key's slot
#                bytes == the store record.
#   T3 timeout : service paused -> the call falls back to masking (output == golden on the post-mask ids, the miss
#                ring logged the keys, no slot left assigned to a key that was not read); unpaused -> exact again.
#   T4 latency : cold calls with n = 1, 2, 4, 10, 20, 40 NVMe misses: wall per call (events) -> t0 and per-expert cost.
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, "/pkg")
from exl3xpu.moe_offload import ops, s64
from exl3xpu.nvtier import TierStore, NvTierAsync
from exl3xpu.exact_miss import ExactMiss

X = ops(); dev = torch.device("xpu")
assert hasattr(X, "moe_exact_setup"), "build with MOE_EXACT=1"
H, I, K, E, TOPK = 2560, 640, 3, 512, 10
STORE = os.environ.get("STORE", "/s/qwen_experts.bin")
META = json.load(open(os.environ.get("META", STORE.rsplit(".", 1)[0] + ".json")))
LAYERS = [int(x) for x in os.environ.get("LAYERS", "6,18,30,42").split(",")]
SLOTS = int(os.environ.get("SLOTS", "640"))
CALLS = int(os.environ.get("CALLS", "600"))
OUT = os.environ.get("OUT", "/o/exact_gpu_test.json")
res = dict(meta=dict(layers=LAYERS, slots=SLOTS, lib=os.environ.get("EXL3_MOE_LIB")), t2=None, t3=None, t4=[])
sync = torch.xpu.synchronize


def dump():
    json.dump(res, open(OUT + ".tmp", "w"), indent=1); os.replace(OUT + ".tmp", OUT)


st = TierStore(H, I, K, E, SLOTS, n_layers=len(LAYERS))
for li in range(len(LAYERS)):
    st.add_layer(f"layers.{LAYERS[li]}.mlp.experts")
os.environ.setdefault("EXL3_NVTIER_NO_VICTIM", "1")            # no GPU victim write-back: the RAM tier stays empty
tier = NvTierAsync(st, STORE, META, 0, ckpt_layer=LAYERS)        # 0-byte RAM budget: only the pinned safe experts
os.environ.setdefault("EXL3_EXACT_PF_PAGES", "0")
ex = ExactMiss(tier, st)
REC, BLOB = tier.rec, st.blob

# golden: every expert of the test layers in VRAM (REC stride)
gold_h = np.empty((len(LAYERS) * E, REC), dtype=np.uint8)
fd = os.open(STORE, os.O_RDONLY)
for li, L in enumerate(LAYERS):
    gold_h[li * E:(li + 1) * E] = np.frombuffer(os.pread(fd, E * REC, L * E * REC), dtype=np.uint8).reshape(E, REC)
gold = torch.from_numpy(gold_h).to(dev)
GB = s64(gold.data_ptr())
gtab = [torch.tensor([GB + (li * E + e) * REC for e in range(E)], dtype=torch.int64, device=dev) for li in range(len(LAYERS))]

tr = [np.load(os.path.join("/t", f))["decode_ids"] for f in sorted(os.listdir("/t")) if f.endswith(".npz")]
rng = np.random.default_rng(0)
xs = [torch.randn(1, H, dtype=torch.bfloat16, device=dev) for _ in range(32)]


def call(li, ids_np, x, w=None):
    ids = torch.from_numpy(np.ascontiguousarray(ids_np, dtype=np.int32)).reshape(x.shape[0], TOPK).to(dev)
    w = torch.full((x.shape[0], TOPK), 0.1, dtype=torch.float32, device=dev) if w is None else w
    y = st.forward_cached(f"layers.{LAYERS[li]}.mlp.experts", x, ids, w)
    return y, ids, w            # ids / w as the kernel left them (masked picks rewritten on fallback only)


def golden(li, ids, w, x):
    return X.moe_forward(x, ids, w, gtab[li], I, K, E)


def cold_experts(li, k):
    """Experts of layer li in neither VRAM nor RAM, from the DEVICE state (the host view tier.res is not updated
    without tier.step(), and GPU victim write-back flags pages resident on the device only)."""
    so = st.slot_of_dev[li * E:(li + 1) * E].cpu().numpy()
    rr = tier.res_dev[li * E:(li + 1) * E].cpu().numpy()
    return [e for e in rng.permutation(E).tolist() if so[e] < 0 and not rr[e]][:k]


def dbg(tag):
    d = ex.debug_jobs()
    print(json.dumps(dict(debug=tag, **d["summary"])), flush=True)
    res.setdefault("debug", {})[tag] = d
    return d


def slot_audit():
    so = st.slot_of_dev[:len(LAYERS) * E].cpu().numpy()
    sk = st.slot_key.cpu().numpy()
    bad = 0
    for key in np.nonzero(so >= 0)[0][:400]:
        s = so[key]
        if sk[s] != key:
            bad += 1; continue
        a = st.slots[s].cpu().numpy()
        bad += int(not np.array_equal(a[:BLOB], gold_h[key, :BLOB]))
    return bad


# ---- T2 exactness
t = time.perf_counter(); mism = 0; n = 0
d0 = ex.dev_stats()
for c in range(CALLS):
    trace = tr[c % len(tr)]; pos = (c * 7) % trace.shape[0]
    for li, L in enumerate(LAYERS):
        x = xs[(c + li) % len(xs)]
        y, ids, w = call(li, trace[pos, L], x)
        g = golden(li, ids, w, x)
        mism += int(not torch.equal(y, g)); n += 1
sync()
d1 = ex.dev_stats()
res["t2"] = dict(calls=n, mismatches=mism, jobs=d1["jobs"] - d0["jobs"], keys=d1["keys"] - d0["keys"],
                 fallbacks=d1["timeouts"] - d0["timeouts"], slot_audit_bad=slot_audit(), host=ex.host_stats(),
                 s=round(time.perf_counter() - t, 1))
print(json.dumps(res["t2"]), flush=True); dbg("t2"); dump()

# ---- T3 timeout fallback and recovery
X.moe_exact_pause(1)
li = 0; L = LAYERS[li]
cold = cold_experts(li, TOPK)
x = xs[0]
d0 = ex.dev_stats()
y, ids, w = call(li, np.array(cold), x)
sync()
d1 = ex.dev_stats()
masked_ok = torch.equal(y, golden(li, ids, w, x))
so2 = st.slot_of_dev[li * E:(li + 1) * E].cpu().numpy()
X.moe_exact_pause(0)
y2, ids2, w2 = call(li, np.array(cold), x)
sync()
res["t3"] = dict(cold=len(cold), fallbacks=d1["timeouts"] - d0["timeouts"],
                 fallback_keys=d1["timeout_keys"] - d0["timeout_keys"], masked_output_ok=bool(masked_ok),
                 masked_ids_all_safe=bool((ids.cpu().numpy() == tier.safe[li].item()).all()),
                 cold_keys_left_with_slot=int((so2[cold] >= 0).sum()),
                 after_resume_exact=bool(torch.equal(y2, golden(li, ids2, w2, x))), slot_audit_bad=slot_audit())
print(json.dumps(res["t3"]), flush=True); dbg("t3"); dump()

# ---- T4 latency per call vs number of NVMe misses, for each poll mode (2 = relaxed load and 0 = acquire fence + load:
# both stale on B70 in runs 1-2; 3 = uncached read hint, the default; 1 = atomic RMW, last: needs PCIe AtomicOps)
for mode in [int(m) for m in os.environ.get("POLL_SWEEP", "2,0,3,1").split(",")]:
  ex.set_poll_mode(mode); sync()
  d0 = ex.dev_stats()
  for nmiss in (1, 2, 4, 10, 20, 40):
    walls = []
    for rep in range(10):
        li = rep % len(LAYERS)
        cold = cold_experts(li, nmiss)
        if len(cold) < nmiss: continue
        rows = (nmiss + TOPK - 1) // TOPK
        ids_np = np.array((cold * TOPK)[:rows * TOPK])         # rows <= 4: pad with repeats of cold experts
        x = torch.randn(rows, H, dtype=torch.bfloat16, device=dev)
        sync(); a = torch.xpu.Event(enable_timing=True); b = torch.xpu.Event(enable_timing=True)
        a.record(); call(li, ids_np, x); b.record(); sync()
        walls.append(a.elapsed_time(b))
    res["t4"].append(dict(poll_mode=mode, nmiss=nmiss, ms_median=float(np.median(walls)) if walls else None,
                          ms_p90=float(np.percentile(walls, 90)) if walls else None, n=len(walls)))
    print(json.dumps(res["t4"][-1]), flush=True); dump()
  d1 = ex.dev_stats()
  res.setdefault("t4_modes", []).append(dict(poll_mode=mode, jobs=d1["jobs"] - d0["jobs"], fallbacks=d1["timeouts"] - d0["timeouts"],
                                             polls_per_job=(d1["polls"] - d0["polls"]) / max(1, d1["jobs"] - d0["jobs"])))
  print(json.dumps(res["t4_modes"][-1]), flush=True); dbg(f"t4_mode{mode}"); dump()
res["host_final"] = ex.close()
dump()
ok = res["t2"]["mismatches"] == 0 and res["t2"]["fallbacks"] == 0 and res["t2"]["slot_audit_bad"] == 0 and \
    res["t3"]["fallbacks"] == 1 and res["t3"]["masked_output_ok"] and res["t3"]["masked_ids_all_safe"] and \
    res["t3"]["cold_keys_left_with_slot"] == 0 and res["t3"]["after_resume_exact"] and \
    all(m["fallbacks"] == 0 for m in res["t4_modes"] if m["poll_mode"] != 2)
print("PASS" if ok else "FAIL", flush=True)
sys.exit(0 if ok else 1)
