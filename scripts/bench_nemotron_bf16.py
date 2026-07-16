#!/usr/bin/env python
"""Nemotron-Nano-12B-v2-VL BF16 decode tps 基准 — vLLM/NVFP4 迁移决策依据。

判据(Sean 2026-07-15 拍板):BF16 decode ≥20 tok/s 则不迁移 vLLM/NVFP4;
否则迁移(评分叙事 + 带宽收益)。口径:文本 / 图文各测 3 轮,prefill 与
decode 分开计时(先 1 token 测 prefill,再全长减去);图用 ingest 真实证据
crop,prompt 用 S5 属性抽取工况。
"""

from __future__ import annotations

import glob
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

MODEL = str(Path.home() / "models" / "nv-community__NVIDIA-Nemotron-Nano-12B-v2-VL-BF16")
MAX_NEW = 128
ROUNDS = 3


def pick_image() -> Image.Image:
    for pattern in ("~/proj/local-data/ingest_a_v5/v1/evidence/*.jpg",
                    "~/proj/local-data/ingest_a_v5/v1/keyframes/*.jpg"):
        hits = sorted(glob.glob(str(Path(pattern).expanduser())))
        if hits:
            print(f"image: {hits[0]}", flush=True)
            return Image.open(hits[0]).convert("RGB")
    print("image: synthetic fallback", flush=True)
    return Image.new("RGB", (512, 512), (128, 100, 80))


def bench(model, tokenizer, inputs, label: str) -> None:
    in_len = inputs["input_ids"].shape[1]
    print(f"== {label} (prompt {in_len} tok) ==", flush=True)
    gen_kw = dict(do_sample=False, eos_token_id=tokenizer.eos_token_id)
    for r in range(ROUNDS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(**inputs, max_new_tokens=1, **gen_kw)
        torch.cuda.synchronize()
        t_pre = time.perf_counter() - t0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=MAX_NEW, **gen_kw)
        torch.cuda.synchronize()
        t_tot = time.perf_counter() - t0
        n_new = out.shape[1] - in_len if out.shape[1] > in_len else out.shape[1]
        decode_tps = (n_new - 1) / max(t_tot - t_pre, 1e-9)
        print(
            f"  r{r}: prefill {t_pre:.2f}s | {n_new} new tok in {t_tot:.2f}s "
            f"| decode {decode_tps:.1f} tok/s",
            flush=True,
        )
    if label.startswith("image"):
        text = tokenizer.decode(out[0][in_len:] if out.shape[1] > in_len else out[0],
                                skip_special_tokens=True)
        print(f"  sample output: {text[:200]!r}", flush=True)


def main() -> None:
    print(f"loading {MODEL}", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, device_map="cuda:0", torch_dtype=torch.bfloat16
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    print(f"loaded in {time.perf_counter() - t0:.0f}s", flush=True)

    sys_msg = {"role": "system", "content": "/no_think"}

    # 文本-only:S5 文案工况
    messages = [sys_msg, {"role": "user", "content": [{"type": "text",
        "text": "List 40 common bedroom objects, one per line, with a color adjective."}]}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        inputs = processor(text=[prompt], return_tensors="pt").to("cuda:0")
    except Exception:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    bench(model, tokenizer, inputs, "text-only")

    # 图文:S5 属性抽取真实工况(证据 crop)
    image = pick_image()
    messages = [sys_msg, {"role": "user", "content": [{"type": "image", "image": ""},
        {"type": "text", "text": "List every object visible in this image. For each: "
         "category, color, material, and any visible text or markings."}]}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda:0")
    # README 官方示例只传三个键;processor 额外返回的 num_patches 会被
    # generate 的 kwargs 校验拒绝
    inputs = {k: inputs[k] for k in ("input_ids", "attention_mask", "pixel_values") if k in inputs}
    bench(model, tokenizer, inputs, "image+text")

    print(f"peak cuda mem: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
