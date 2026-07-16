#!/usr/bin/env python
"""vLLM NVFP4 服务冒烟 + tps 实测 — 与 bench_nemotron_bf16.py 同工况对照。

文本 / 图文各 3 轮(r0 = 预热);tok/s 按 completion_tokens/总耗时(含 prefill,
口径偏保守,仍可与 BF16 的 decode 数字比量级)。图用 ingest 真实证据 crop。
用法(节点): python scripts/vllm_smoke.py [base_url]
"""

from __future__ import annotations

import base64
import glob
import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
ROUNDS = 3


def api(path: str, payload: dict | None = None) -> dict:
    url = BASE + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def bench(model: str, messages: list, label: str) -> None:
    print(f"== {label} ==", flush=True)
    for r in range(ROUNDS):
        payload = {"model": model, "messages": messages, "max_tokens": 128, "temperature": 0}
        t0 = time.perf_counter()
        out = api("/v1/chat/completions", payload)
        dt = time.perf_counter() - t0
        u = out.get("usage", {})
        ct = u.get("completion_tokens", 0) or 1
        print(
            f"  r{r}: {dt:.2f}s | prompt {u.get('prompt_tokens')} tok | completion {ct} tok "
            f"| ~{ct / dt:.1f} tok/s (含 prefill)",
            flush=True,
        )
    text = out["choices"][0]["message"]["content"]
    print(f"  sample: {text[:180]!r}", flush=True)


def main() -> int:
    model = api("/v1/models")["data"][0]["id"]
    print(f"served model: {model}", flush=True)

    bench(model, [
        {"role": "user", "content": "List 40 common bedroom objects, one per line, with a color adjective."},
    ], "text-only")

    hits = sorted(glob.glob(str(Path.home() / "proj/local-data/ingest_a_v5/v1/evidence/*.jpg")))
    if not hits:
        print("no evidence crop found — skip image test", flush=True)
        return 0
    b64 = base64.b64encode(Path(hits[0]).read_bytes()).decode()
    print(f"image: {hits[0]}", flush=True)
    bench(model, [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": "List every object visible in this image. For each: "
                                     "category, color, material, and any visible text or markings."},
        ]},
    ], "image+text")
    print("SMOKE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
