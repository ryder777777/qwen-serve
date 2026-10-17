#!/usr/bin/env python3
"""Packed NVMe expert store for Qwen3.8-Flash-Next EXL3 3.05bpw_h5_ng5 (the N104 layout).

One 4K-aligned record per (layer, expert): record r = layer*512 + e at offset r*REC, holding the exl3xpu blob
(pack_expert of gate/up/down trellis + suh/svh, K=3); 48 x 512 x 1,863,680 B = 45.8 GB, plus qwen_experts.json (meta)
and qwen_experts.bin.done.

Store layers 0-47 are the target's MoE layers. When the checkpoint has the MTP draft's MoE layer
(mtp.layers.0.mlp.experts.*), it is packed as store layer 48 (records 48*512 .. 48*512+511, +0.95 GB) and the meta says
L=49, target_layers=48, mtp_layers=[48]: the tier plugin then serves it as an ordinary NVMe layer. A 48-layer store
(meta L=48, or --no-mtp) keeps today's behaviour (MTP layer packed into RAM and pinned at load).
`pack` on a complete 48-layer store only appends the MTP layer (~1 GB), it does not rewrite layers 0-47.

  pack_store.py pack   [--model DIR] [--out DIR] [--no-mtp]   build / extend, then verify
  pack_store.py verify [--model DIR] [--out DIR]   size + .done + O_DIRECT re-read of 64 random records (+ 16 of
                                                   every MTP layer) vs a fresh pack
  pack_store.py probe  [--out DIR]                 the store exists and O_DIRECT reads work on its filesystem
"""
import argparse, json, mmap, os, random, re, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["pack", "verify", "probe"])
ap.add_argument("--model", default=os.environ.get("EXL3_MODEL_PATH", "/models/turboderp-Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5"))
ap.add_argument("--out", default=os.path.dirname(os.environ.get("EXL3_NVTIER_STORE", "/nvx/qwen_experts.bin")))
ap.add_argument("--no-mtp", action="store_true", help="pack only the 48 target layers (MTP layer stays RAM-pinned)")
ap.add_argument("--verify-n", type=int, default=64)
a = ap.parse_args()
LT, E, K = 48, 512, 3                      # target MoE layers, experts, trellis bits
BIN, META = os.path.join(a.out, "qwen_experts.bin"), os.path.join(a.out, "qwen_experts.json")


def odirect_probe(path, nbytes=1 << 20):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    except OSError as e:
        sys.exit(f"O_DIRECT open failed on {path}: {e} (the store must live on a local filesystem that supports O_DIRECT, e.g. xfs or ext4)")
    buf = mmap.mmap(-1, nbytes)
    try:
        n = os.preadv(fd, [buf], 0)
    except OSError as e:
        sys.exit(f"O_DIRECT read failed on {path}: {e}")
    finally:
        os.close(fd)
    return n


def packer():
    import torch  # noqa: F401
    from safetensors import safe_open
    from exl3xpu.moe_offload import pack_expert
    idx = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]; fh = {}
    mtp = [] if a.no_mtp else mtp_prefixes(idx)
    def get(n):
        f = idx[n]
        if f not in fh: fh[f] = safe_open(os.path.join(a.model, f), "pt")
        return fh[f].get_tensor(n)
    def pack(l, e):
        pre = f"model.language_model.layers.{l}.mlp.experts.{e}." if l < LT else f"{mtp[l - LT]}{e}."
        d = {n: {k: get(pre + n + "." + k) for k in ("trellis", "suh", "svh")} for n in ("gate_proj", "up_proj", "down_proj")}
        return pack_expert(d["gate_proj"], d["up_proj"], d["down_proj"], K).numpy().tobytes()
    return pack, len(mtp)


def mtp_prefixes(idx):
    """Checkpoint prefixes of the MTP draft's routed experts, by MTP layer: '<...>mtp.layers.<i>.mlp.experts.'. Only
    complete layers (all E experts with gate/up/down trellis + suh + svh) count."""
    pat = re.compile(r"^(.*?mtp\.layers\.(\d+)\.mlp\.experts\.)(\d+)\.(gate_proj|up_proj|down_proj)\.(trellis|suh|svh)$")
    found = {}
    for n in idx:
        m = pat.match(n)
        if m:
            found.setdefault((int(m.group(2)), m.group(1)), set()).add((int(m.group(3)), m.group(4), m.group(5)))
    out = []
    for (i, pre), have in sorted(found.items()):
        if len(have) == E * 9:
            out.append(pre)
        else:
            print(json.dumps({"warning": f"MTP layer {i} ({pre}) incomplete: {len(have)} of {E * 9} tensors, not packed"}), flush=True)
    return out


def verify(pack, n_mtp_ckpt, n=64, n_per_mtp=16):
    meta = json.load(open(META)); blob, rec = meta["blob"], meta["rec"]
    L = int(meta["L"])
    assert L >= LT and os.path.getsize(BIN) == L * E * rec, f"store size {os.path.getsize(BIN)} != meta L={L} x {E} x {rec}"
    mtp = [int(c) for c in meta.get("mtp_layers", [])]
    assert mtp == list(range(LT, L)), f"meta mtp_layers {mtp} vs L={L}"
    assert len(mtp) <= n_mtp_ckpt, f"store has {len(mtp)} MTP layer(s), checkpoint {n_mtp_ckpt}"
    fd = os.open(BIN, os.O_RDONLY | os.O_DIRECT); buf = mmap.mmap(-1, rec); bad = 0
    random.seed(0)
    picks = [(random.randrange(L), random.randrange(E)) for _ in range(n)]
    picks += [(l, random.randrange(E)) for l in mtp for _ in range(n_per_mtp)] + [(l, e) for l in mtp for e in (0, E - 1)]
    for l, e in picks:
        assert os.preadv(fd, [buf], (l * E + e) * rec) == rec
        bad += bytes(buf[:blob]) != pack(l, e)
    os.close(fd)
    return bad, len(picks), L, mtp


if a.cmd == "probe":
    if not (os.path.exists(BIN) and os.path.exists(BIN + ".done") and os.path.exists(META)):
        sys.exit(f"expert store missing in {a.out} (qwen_experts.bin + .done + qwen_experts.json): run the image with `pack-store` first")
    odirect_probe(BIN); print(json.dumps({"store": BIN, "o_direct": True})); sys.exit(0)

def write_meta(L, blob, rec):
    """Atomic meta update (tmp + rename): readers never see a half-written meta."""
    m = {"L": L, "E": E, "K": K, "blob": blob, "rec": rec, "H": 2560, "I": 640, "target_layers": LT,
         "mtp_layers": list(range(LT, L))}
    with open(META + ".tmp", "w") as f:
        json.dump(m, f)
    os.replace(META + ".tmp", META)


os.makedirs(a.out, exist_ok=True)
pack, n_mtp = packer()
if a.cmd == "pack":
    t0 = time.time()
    blob0 = pack(0, 0); BLOB = len(blob0); REC = (BLOB + 4095) // 4096 * 4096
    L = LT + n_mtp
    # layers already complete: a finished store (.done + meta) of L0 layers with the same record size. Layers L0..L-1
    # are appended (an existing 48-layer store gets only the MTP layer); anything else is rebuilt from layer 0.
    L0 = 0
    if os.path.exists(BIN + ".done") and os.path.exists(META):
        m0 = json.load(open(META))
        if int(m0["L"]) > L:
            sys.exit(f"store already has {m0['L']} layers (MTP included); packing {L} would rebuild it. Drop --no-mtp, "
                     f"or delete {BIN} first")
        if m0.get("rec") == REC and m0.get("blob") == BLOB and LT <= int(m0["L"]) <= L \
                and os.path.getsize(BIN) >= int(m0["L"]) * E * REC:
            L0 = int(m0["L"])
    if L0 < L:
        fd = os.open(BIN, os.O_RDWR | os.O_CREAT, 0o644); os.ftruncate(fd, L * E * REC)
        pad = bytes(REC - BLOB)
        print(json.dumps({"packing_layers": [L0, L - 1], "mtp_layers": list(range(LT, L))}), flush=True)
        for l in range(L0, L):
            for e in range(E):
                os.pwrite(fd, pack(l, e) + pad, (l * E + e) * REC)
            if l % 8 == 7 or l >= LT: print(json.dumps({"layer": l, "elapsed_s": round(time.time() - t0)}), flush=True)
        os.fsync(fd); os.close(fd)
        write_meta(L, BLOB, REC)               # an interrupted extension leaves meta at L0: the next pack resumes from L0
        open(BIN + ".done", "w").close()
    elif os.path.getsize(BIN) != L * E * REC:  # meta says L, file longer (an older interrupted run): trim
        os.truncate(BIN, L * E * REC)
    odirect_probe(BIN)
bad, nv, L, mtp = verify(pack, n_mtp, a.verify_n)
note = None
if not mtp and n_mtp:
    note = "store has no MTP layer: the MTP draft's MoE layer stays RAM-pinned (~1 GB); run pack-store to append it"
print(json.dumps({"store": BIN, "L": L, "mtp_layers": mtp, "verify_bad": bad, "verify_n": nv, "note": note}), flush=True)
sys.exit(1 if bad else 0)
