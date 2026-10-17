#!/usr/bin/env python3
"""CPU dry run of serve/pack_store.py against a tiny synthetic checkpoint (48 target layers + the MTP draft layer, 512
experts each, H = I = 64 so a record is 8 KiB). No GPU, no real model; needs torch + safetensors.

  python3 tests/pack_store_dryrun.py [WORKDIR]

Checks: a 48-layer store (--no-mtp), then `pack` appends only store layer 48 (layers 0-47 untouched), verify passes and
covers layer 48, a corrupted layer-48 record is caught, an old-format 48-layer meta (no mtp_layers) still verifies.
On macOS (no O_DIRECT) os.O_DIRECT is shimmed to 0.
"""
import hashlib, json, os, subprocess, sys, tempfile

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "plugin")            # exl3xpu.moe_offload.pack_expert (pure torch)
PACK = os.path.join(HERE, "..", "serve", "pack_store.py")
LT, E, H, I, K = 48, 512, 64, 64, 3


def make_ckpt(d):
    g = torch.Generator().manual_seed(0)
    t = {}
    def expert(pre, seed):
        g.manual_seed(seed)
        for n, (k, m) in (("gate_proj", (H, I)), ("up_proj", (H, I)), ("down_proj", (I, H))):
            t[f"{pre}{n}.trellis"] = torch.randint(-32768, 32767, (k // 16, m // 16, 16 * K), generator=g, dtype=torch.int16)
            t[f"{pre}{n}.suh"] = torch.randn(k, generator=g).half()
            t[f"{pre}{n}.svh"] = torch.randn(m, generator=g).half()
    for l in range(LT):
        for e in range(E):
            expert(f"model.language_model.layers.{l}.mlp.experts.{e}.", l * E + e)
    for e in range(E):
        expert(f"mtp.layers.0.mlp.experts.{e}.", 10_000_000 + e)
    save_file(t, os.path.join(d, "model.safetensors"))
    json.dump({"weight_map": {k: "model.safetensors" for k in t}}, open(os.path.join(d, "model.safetensors.index.json"), "w"))


def run(*args):
    code = ("import os, sys\n"
            "if not hasattr(os, 'O_DIRECT'): os.O_DIRECT = 0\n"
            f"sys.path.insert(0, {PKG!r}); sys.argv = ['pack_store.py'] + {list(args)!r}\n"
            f"import runpy; runpy.run_path({PACK!r}, run_name='__main__')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=1800)
    last = [l for l in r.stdout.splitlines() if l.startswith("{")]
    return r.returncode, json.loads(last[-1]) if last else None, r.stdout + r.stderr


def sha(path, a, b):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(a); h.update(f.read(b - a))
    return h.hexdigest()


def main():
    wd = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="pack_dryrun_")
    model, out = os.path.join(wd, "model"), os.path.join(wd, "nvx")
    os.makedirs(model, exist_ok=True)
    if not os.path.exists(os.path.join(model, "model.safetensors.index.json")):
        make_ckpt(model)
    for f in ("qwen_experts.bin", "qwen_experts.bin.done", "qwen_experts.json"):
        if os.path.exists(os.path.join(out, f)): os.remove(os.path.join(out, f))
    res = {}
    rc, j, log = run("pack", "--model", model, "--out", out, "--no-mtp")
    res["pack_48"] = dict(rc=rc, L=j and j["L"], bad=j and j["verify_bad"])
    assert rc == 0 and j["L"] == 48 and j["mtp_layers"] == [], log
    meta = json.load(open(os.path.join(out, "qwen_experts.json"))); rec = meta["rec"]
    binp = os.path.join(out, "qwen_experts.bin")
    h48 = sha(binp, 0, LT * E * rec)
    # old-format meta (before this change): no target_layers / mtp_layers
    json.dump({k: meta[k] for k in ("L", "E", "K", "blob", "rec", "H", "I")}, open(os.path.join(out, "qwen_experts.json"), "w"))
    rc, j, log = run("verify", "--model", model, "--out", out)
    res["verify_old_meta"] = dict(rc=rc, L=j and j["L"], note=j and j["note"])
    assert rc == 0 and j["L"] == 48 and j["note"], log
    rc, j, log = run("pack", "--model", model, "--out", out)
    res["pack_extend"] = dict(rc=rc, L=j and j["L"], mtp=j and j["mtp_layers"], verify_n=j and j["verify_n"],
                              packed=[json.loads(l)["packing_layers"] for l in log.splitlines() if '"packing_layers"' in l])
    assert rc == 0 and j["L"] == 49 and j["mtp_layers"] == [48] and j["verify_bad"] == 0, log
    assert res["pack_extend"]["packed"] == [[48, 48]], log
    assert sha(binp, 0, LT * E * rec) == h48, "layers 0-47 changed by the extension"
    assert os.path.getsize(binp) == 49 * E * rec
    meta = json.load(open(os.path.join(out, "qwen_experts.json")))
    assert meta["L"] == 49 and meta["mtp_layers"] == [48] and meta["target_layers"] == 48, meta
    rc, j, log = run("pack", "--model", model, "--out", out)            # idempotent: nothing to pack
    assert rc == 0 and '"packing_layers"' not in log, log
    with open(binp, "r+b") as f:                                        # corrupt every layer-48 record's first byte
        for e in range(E):
            f.seek((48 * E + e) * rec); b = f.read(1); f.seek((48 * E + e) * rec); f.write(bytes([b[0] ^ 0xFF]))
    rc, j, log = run("verify", "--model", model, "--out", out)
    res["verify_corrupt_mtp"] = dict(rc=rc, bad=j and j["verify_bad"])
    assert rc == 1 and j["verify_bad"] > 0, log
    res["record_bytes"] = rec
    print(json.dumps(res))
    print("PASS")


if __name__ == "__main__":
    main()
