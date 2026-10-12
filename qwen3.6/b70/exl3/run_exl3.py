#!/usr/bin/env python3
"""EXL3 inference script for Qwen3.6-35B-A3B on RTX 3090 (Ampere).

Requires the exl3_ampere_patch.py in the same directory or PYTHONPATH.
The patch forces Triton conv1d fallback and disables the BC GDN path
that crashes on Ampere GPUs.

Usage:
  # In the exl3-test Docker container:
  export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib
  python3 run_exl3.py --model /models/Qwen3.6-35B-A3B-EXL3 --prompt "Hello"
"""
import argparse
import os
import sys
import time

# Ensure the Ampere patch is loaded before any EXL3 imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exl3_ampere_patch  # noqa: F401

import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/models/Qwen3.6-35B-A3B-EXL3")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("LD_LIBRARY_PATH", "/usr/local/lib/python3.12/dist-packages/torch/lib")

    config = Config.from_directory(args.model)
    print(f"Architecture: {config.architecture}, layers: {config.num_hidden_layers}")

    model = Model.from_config(config)
    model.load(device=args.device, max_batch_size=1)
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded! VRAM: {vram:.2f} GB")

    tokenizer = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=4096, max_batch_size=1)
    generator = Generator(model, cache, tokenizer, max_batch_size=1, max_chunk_size=512)

    print(f"Prompt: {args.prompt}")
    t0 = time.time()
    output = generator.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        add_bos=True,
    )
    torch.cuda.synchronize()
    t1 = time.time()

    in_ids = tokenizer.encode(args.prompt, add_bos=True)
    out_ids = tokenizer.encode(output, add_bos=True)
    gen_tokens = out_ids.shape[1] - in_ids.shape[1] if out_ids.dim() > 1 else len(out_ids) - len(in_ids)
    elapsed = t1 - t0

    print(f"Output: {output}")
    print(f"Generated: {gen_tokens} tokens in {elapsed:.2f}s ({gen_tokens/elapsed:.1f} tok/s)")
    print(f"VRAM: {vram:.2f} GB on {args.device}")

if __name__ == "__main__":
    main()
