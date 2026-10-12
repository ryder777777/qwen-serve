#!/usr/bin/env python3

import argparse
import json
import math
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def make_prompt(base_url, target):
    marker = uuid.uuid4().hex
    text = f"Run {marker}. Read the following benchmark context, then answer DONE. "
    text += "Omarchy serves private local agents on measured hardware. " * max(1, target // 9)
    tokens = requests.post(f"{base_url}/tokenize", json={"prompt": text}, timeout=30).json()["tokens"]
    if len(tokens) > target:
        tokens = tokens[:target]
        text = requests.post(f"{base_url}/detokenize", json={"tokens": tokens}, timeout=30).json()["text"]
    return text


def generate(base_url, prompt, output_tokens):
    started = time.perf_counter()
    response = requests.post(
        f"{base_url}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": output_tokens,
                "ignore_eos": True,
            },
            "stream": True,
        },
        stream=True,
        timeout=600,
    )
    response.raise_for_status()
    first_token = None
    result = None
    for line in response.iter_lines(chunk_size=1, decode_unicode=True):
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            continue
        event = json.loads(data)
        if first_token is None and event.get("meta_info", {}).get("completion_tokens"):
            first_token = time.perf_counter()
        result = event
    finished = time.perf_counter()
    meta = result["meta_info"]
    ids = meta.get("output_ids", [])
    text = result.get("text", "")
    decode_seconds = finished - first_token
    return {
        "prompt_tokens": meta["prompt_tokens"],
        "completion_tokens": meta["completion_tokens"],
        "ttft_s": first_token - started,
        "latency_s": finished - started,
        "client_decode_tok_s": max(0, meta["completion_tokens"] - 1) / decode_seconds,
        "server_decode_tok_s": meta.get("decode_throughput"),
        "valid": bool(text.strip()) and "!!!" not in text and 0 not in ids,
        "sample": text[:160],
    }


def run_point(base_url, target, concurrency, output_tokens):
    prompts = [make_prompt(base_url, target) for _ in range(concurrency)]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda prompt: generate(base_url, prompt, output_tokens), prompts))
    elapsed = time.perf_counter() - started
    completions = sum(result["completion_tokens"] for result in results)
    ttfts = [result["ttft_s"] for result in results]
    return {
        "target_context": target,
        "concurrency": concurrency,
        "prompt_tokens": [result["prompt_tokens"] for result in results],
        "completion_tokens": completions,
        "wall_s": elapsed,
        "aggregate_tok_s": completions / elapsed,
        "decode_tok_s_p50": statistics.median(result["client_decode_tok_s"] for result in results),
        "ttft_p50_s": statistics.median(ttfts),
        "ttft_p95_s": percentile(ttfts, 0.95),
        "valid": all(result["valid"] for result in results),
        "requests": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--results")
    args = parser.parse_args()

    requests.get(f"{args.base_url}/health", timeout=30).raise_for_status()
    server = requests.get(f"{args.base_url}/server_info", timeout=30).json()
    generate(args.base_url, make_prompt(args.base_url, 128), 16)

    matrix = [(context, 1) for context in (128, 512, 2048, 8192)]
    matrix += [(512, concurrency) for concurrency in (2, 4, 8)]
    points = []
    for context, concurrency in matrix:
        point = run_point(args.base_url, context, concurrency, args.output_tokens)
        points.append(point)
        print(
            f"context={context:5} concurrency={concurrency:2} "
            f"ttft={point['ttft_p50_s']:.3f}s decode={point['decode_tok_s_p50']:.1f} tok/s "
            f"aggregate={point['aggregate_tok_s']:.1f} tok/s valid={point['valid']}",
            flush=True,
        )

    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": args.base_url,
        "server": {
            key: server.get(key)
            for key in (
                "model_path",
                "load_format",
                "tp_size",
                "dtype",
                "quantization",
                "attention_backend",
                "mem_fraction_static",
                "disable_cuda_graph",
            )
        },
        "output_tokens": args.output_tokens,
        "points": points,
    }
    destination = Path(args.results or f"benchmarks/results/{datetime.now():%Y%m%d-%H%M%S}-b70-sglang.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
