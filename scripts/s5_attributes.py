#!/usr/bin/env python
"""S5 属性抽取客户端:hero crop → 本地 vLLM(Nemotron NVFP4)→ 结构化属性 + 命名。

设计(codex 裁决 #2/#3/#5,docs/NEMOTRON_SPARK_SERVING.md):
- hero 单图首抽;JSON 解析失败 / confidence=low / 颜色材质双 unknown 时,
  才对 prototype_refs[1:3] 逐张补抽(服务端 limit image:1,不发多图请求),
  按 hero→p1→p2 顺序每键取第一个非 unknown 值合并(确定性)。
- 输出枚举约束 + "unknown" 合法 + temperature 0;首选 vLLM guided_json,
  服务端不支持时自动降级为纯 prompt + 解析重试一次。
- 持久缓存键 = sha256(crop 字节) + schema 版本 + 模型 id;重跑零成本。
- 产物:attributes JSONL(matcher 只认可比键;label_en/label_zh 是展示命名
  权威候选,见 docs/家居视觉校准工厂_设计.md §5)+ manifest(计数/未知率/
  token 用量)。本脚本只在节点上运行,打 127.0.0.1,不出网。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from PIL import Image  # noqa: E402

SCHEMA_VERSION = "s5-attr-v1"
COLORS = ["white", "black", "gray", "brown", "beige", "red", "orange", "yellow",
          "green", "blue", "purple", "pink", "transparent", "multicolor", "unknown"]
MATERIALS = ["plastic", "wood", "metal", "fabric", "leather", "glass", "paper",
             "ceramic", "rubber", "unknown"]
PATTERNS = ["solid", "striped", "dotted", "cartoon", "floral", "plaid",
            "text_print", "unknown"]
SHAPES = ["box", "cylinder", "sphere", "flat_panel", "bag_soft", "irregular", "unknown"]

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "color_primary": {"type": "string", "enum": COLORS},
        "color_secondary": {"type": "string", "enum": COLORS},
        "material": {"type": "string", "enum": MATERIALS},
        "pattern": {"type": "string", "enum": PATTERNS},
        "shape": {"type": "string", "enum": SHAPES},
        "text_marks": {"type": "string", "maxLength": 40},
        "label_en": {"type": "string", "maxLength": 40},
        "label_zh": {"type": "string", "maxLength": 20},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["color_primary", "color_secondary", "material", "pattern",
                 "shape", "text_marks", "label_en", "label_zh", "confidence"],
    "additionalProperties": False,
}

PROMPT = (
    "You see one household object cropped from a room video. Identify the object "
    "independently (do not guess from crop framing). Output ONLY a JSON object: "
    '{"color_primary": main color, "color_secondary": second color or "unknown", '
    '"material": dominant material, "pattern": surface pattern, "shape": overall shape, '
    '"text_marks": readable printed text/brand or "", '
    '"label_en": object name in 1-3 English words, "label_zh": 中文物品名, '
    '"confidence": "high"/"medium"/"low"}. '
    f"color enums: {COLORS}. material enums: {MATERIALS}. "
    f"pattern enums: {PATTERNS}. shape enums: {SHAPES}. "
    'Use "unknown" when unsure. No explanations.'
)

COMPARABLE = ("color_primary", "color_secondary", "material", "pattern", "shape", "text_marks")
UNKNOWN = {"", "unknown", "none", "null"}

_print_lock = threading.Lock()


def letterbox_b64(path: Path, size: int = 512) -> tuple[str, str]:
    """裁剪图 letterbox 到 size²(防长条 crop 直接 resize 变形),返回 (b64, sha256)。"""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    with Image.open(io.BytesIO(raw)) as image:
        image = image.convert("RGB")
        image.thumbnail((size, size), Image.LANCZOS)
        canvas = Image.new("RGB", (size, size), (127, 127, 127))
        canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
        buffer = io.BytesIO()
        canvas.save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode(), digest


class Client:
    def __init__(self, endpoint: str, model: str, guided: bool):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.guided = guided
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "errors": 0}
        self._usage_lock = threading.Lock()

    def chat(self, image_b64: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 160,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }],
        }
        if self.guided:
            payload["guided_json"] = JSON_SCHEMA
        request = urllib.request.Request(
            self.endpoint + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            data = json.load(response)
        usage = data.get("usage", {})
        with self._usage_lock:
            self.usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            self.usage["completion_tokens"] += usage.get("completion_tokens", 0)
            self.usage["calls"] += 1
        return data["choices"][0]["message"]["content"]


def parse_attrs(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        raw = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    out = {}
    for key in JSON_SCHEMA["properties"]:
        value = str(raw.get(key, "unknown")).strip()
        enum = JSON_SCHEMA["properties"][key].get("enum")
        if enum and value.lower() not in enum:
            value = "unknown"
        out[key] = value.lower() if enum else value
    return out


def needs_escalation(attrs: dict | None) -> bool:
    if attrs is None:
        return True
    if attrs.get("confidence") == "low":
        return True
    return attrs.get("color_primary") == "unknown" and attrs.get("material") == "unknown"


def merge_attrs(results: list[dict]) -> dict:
    """按 hero→p1→p2 顺序,每键取第一个非 unknown 值;confidence 取最好一档。"""
    merged = dict(results[0])
    for later in results[1:]:
        for key in COMPARABLE + ("label_en", "label_zh"):
            if str(merged.get(key, "")).strip().lower() in UNKNOWN:
                merged[key] = later.get(key, merged.get(key, "unknown"))
    order = {"high": 0, "medium": 1, "low": 2}
    merged["confidence"] = min(
        (r.get("confidence", "low") for r in results), key=lambda c: order.get(c, 2)
    )
    return merged


def extract_one(client: Client, tracklet: dict,
                cache: dict, cache_path: Path, cache_lock: threading.Lock) -> dict:
    tid = tracklet["tracklet_id"]
    refs = tracklet.get("prototype_refs") or []
    hero_ref = (tracklet.get("attributes") or {}).get("hero_ref") or (refs[0] if refs else None)
    if not hero_ref:
        return {"tracklet_id": tid, "status": "NO_HERO", "attributes": {}}
    ordered = [hero_ref] + [r for r in refs if r != hero_ref][:2]

    results, sources, calls = [], [], 0
    for ref in ordered:
        path = PROJ / ref
        if not path.exists():
            continue
        image_b64, digest = letterbox_b64(path)
        cache_key = f"{digest}:{SCHEMA_VERSION}:{client.model}"
        with cache_lock:
            hit = cache.get(cache_key)
        if hit is not None:
            attrs = hit
        else:
            attrs = None
            for _ in range(2):  # 失败原样重试一次(网络/解析;temperature 0)
                try:
                    attrs = parse_attrs(client.chat(image_b64))
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
                    with client._usage_lock:
                        client.usage["errors"] += 1
                    with _print_lock:
                        print(f"[warn] {tid} {ref}: {error}", file=sys.stderr)
                    attrs = None
                calls += 1
                if attrs is not None:
                    break
            if attrs is not None:
                with cache_lock:
                    cache[cache_key] = attrs
                    with cache_path.open("a") as handle:
                        handle.write(json.dumps({"key": cache_key, "attrs": attrs},
                                                ensure_ascii=False) + "\n")
        if attrs is not None:
            results.append(attrs)
            sources.append(ref)
        if results and not needs_escalation(merge_attrs(results)):
            break

    if not results:
        return {"tracklet_id": tid, "status": "EXTRACTION_FAILED", "attributes": {}, "calls": calls}
    merged = merge_attrs(results)
    return {
        "tracklet_id": tid,
        "status": "OK",
        "attributes": merged,
        "sources": sources,
        "escalated": len(sources) > 1,
        "calls": calls,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest-root", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="/models/nv-community__NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD")
    parser.add_argument("--out", required=True, help="attributes JSONL 输出路径")
    parser.add_argument("--cache", default=None, help="缓存 JSONL(默认 <out>.cache)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="冒烟用:只处理前 N 轨")
    parser.add_argument(
        "--max-failed-rate", type=float, default=0.01,
        help="失败轨占比容忍上限:失败样本已如实落盘(EXTRACTION_FAILED),仅超限才阶段致命",
    )
    parser.add_argument("--no-guided", action="store_true", help="禁用 guided_json")
    args = parser.parse_args()

    root = Path(args.ingest_root)
    tracklets = []
    for path in sorted(root.glob("*/tracklets.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                tracklets.append(json.loads(line))
    tracklets.sort(key=lambda t: t["tracklet_id"])
    if args.limit:
        tracklets = tracklets[: args.limit]

    cache_path = Path(args.cache or (args.out + ".cache"))
    cache: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["key"]] = row["attrs"]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_lock = threading.Lock()

    client = Client(args.endpoint, args.model, guided=not args.no_guided)
    # guided_json 探活:失败一次即全局降级,避免 945 次各自 400
    probe = tracklets[0] if tracklets else None
    if probe and client.guided:
        ref = (probe.get("attributes") or {}).get("hero_ref")
        if ref and (PROJ / ref).exists():
            image_b64, _ = letterbox_b64(PROJ / ref)
            try:
                client.chat(image_b64)
            except urllib.error.HTTPError as error:
                if error.code == 400:
                    client.guided = False
                    print("[info] guided_json 不被服务端接受,降级纯 prompt", file=sys.stderr)
                else:
                    raise

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(
            lambda t: extract_one(client, t, cache, cache_path, cache_lock),
            tracklets,
        ))
    rows.sort(key=lambda r: r["tracklet_id"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    ok = [r for r in rows if r["status"] == "OK"]
    unknown_rate = {
        key: round(sum(str(r["attributes"].get(key, "unknown")).lower() in UNKNOWN
                       for r in ok) / max(1, len(ok)), 4)
        for key in COMPARABLE
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "guided_json": client.guided,
        "tracklets": len(tracklets),
        "ok": len(ok),
        "failed": len(rows) - len(ok),
        "escalated": sum(1 for r in ok if r.get("escalated")),
        "unknown_rate": unknown_rate,
        "usage": client.usage,
        "wall_seconds": round(time.time() - started, 1),
        "concurrency": args.concurrency,
    }
    Path(str(out) + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    failed_ids = [r["tracklet_id"] for r in rows if r["status"] != "OK"]
    if not failed_ids:
        return 0
    # 个别坏输出=失败样本(下游按 missing 语义剔除),不让整个批次阶段致命;
    # 超过容忍率才判失败(A1 formal 同款契约:严格记错,继续完成)。
    rate = len(failed_ids) / max(1, len(rows))
    print(
        f"failed {len(failed_ids)}/{len(rows)} ({rate:.4%}), "
        f"门限 {args.max_failed_rate:.2%}: {failed_ids[:10]}",
        file=sys.stderr,
    )
    return 0 if rate <= args.max_failed_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())
