#!/usr/bin/env python
"""AutoTune-v1 云端调优代理通道(step-3.7-flash,只在本地 Mac 运行)。

契约(验收门修订记录 2026-07-17):
- 送云内容仅限脱敏物品 crop(不含整帧/位置元数据);GT 锚点归属**不送云**——
  attribution 展示给云端的分组是 reid 运行自己的实体输出,云端独立判断同物性。
- GT 在本地仅用于"选哪些锚点做错误归因"(错误归因用途,每次动用记
  `results/autotune/GT_USAGE.md`);tutor 输出只是"候选/评审意见",经
  autotune_pseudo_labels.py 构造伪标签,绝不直接写入真值或契约对象。
- 每次 API 调用记台账(model/prompt_sha/图片数/tokens)于
  results/autotune/tutor_calls.log.jsonl;内容寻址缓存避免重复计费。

用法:
  python scripts/autotune_tutor.py attribution \
      --reid-dir results/hero_s1/reid-w2 --ingest-root results/hero_s1/ingest \
      --gt fixtures/hero_s1/annotations/anchor_review.confirmed.json \
      --anchors 壁纸刀,玩具娃娃,咖啡罐,剪刀,护手霜,饼干,梳子,跳跳糖 \
      --out results/autotune/attribution
  python scripts/autotune_tutor.py pairs \
      --reid-dir results/hero_s1/reid-w2 --ingest-root results/hero_s1/ingest \
      --extra results/autotune/attribution/merge_pairs.jsonl \
      --max-pairs 400 --out results/autotune/tutor_pairs.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "scripts"))

from stepfun_api import StepfunAPIError, chat_completion, image_part  # noqa: E402

MODEL = "step-3.7-flash"
AUTOTUNE_ROOT = PROJ / "results" / "autotune"
CACHE_DIR = AUTOTUNE_ROOT / "cache"
CALL_LOG = AUTOTUNE_ROOT / "tutor_calls.log.jsonl"


# ---------------------------------------------------------------- shared infra


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _log_call(channel: str, prompt_sha: str, n_images: int, usage: dict, cached: bool):
    CALL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with CALL_LOG.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "channel": channel,
                    "model": MODEL,
                    "prompt_sha": prompt_sha,
                    "images": n_images,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "cached": cached,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _cached_chat(channel: str, cache_key: str, messages: list[dict], n_images: int) -> dict:
    """内容寻址缓存 + 429/网络退避重试;返回 {content, usage, cached}。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        hit = json.loads(cache_file.read_text(encoding="utf-8"))
        _log_call(channel, cache_key[:16], n_images, hit.get("usage", {}), cached=True)
        return {**hit, "cached": True}
    last_err: Exception | None = None
    for attempt, delay in enumerate((0, 5, 20, 60)):
        if delay:
            time.sleep(delay)
        try:
            data = chat_completion(model=MODEL, messages=messages, temperature=0.1)
            result = {
                "content": data["choices"][0]["message"]["content"],
                "usage": data.get("usage", {}),
            }
            cache_file.write_text(
                json.dumps(result, ensure_ascii=False), encoding="utf-8"
            )
            _log_call(channel, cache_key[:16], n_images, result["usage"], cached=False)
            return {**result, "cached": False}
        except StepfunAPIError as exc:
            last_err = exc
            transient = "429" in str(exc) or "network error" in str(exc) or "5" == str(exc)[5:6]
            if not transient and attempt >= 1:
                break
    raise StepfunAPIError(f"tutor call failed after retries: {last_err}")


def _parse_json_block(text: str) -> dict | None:
    """从可能带说明文字的回复中抽取最后一个合法 JSON 对象。

    兜底:tutor 偶发在 JSON 字符串里嵌未转义引号(如 reason 里引用商品名),
    整体解析失败时用正则直接提取 same/confidence 结构字段(位于病灶之前)。
    """
    for candidate in re.findall(r"\{.*\}", text, flags=re.DOTALL) or []:
        # 从最长匹配逐步收缩右边界尝试解析
        for end in range(len(candidate), 1, -1):
            chunk = candidate[:end]
            if chunk.count("{") != chunk.count("}"):
                continue
            try:
                obj = json.loads(chunk)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    same = re.search(r'"same"\s*:\s*(true|false)', text)
    conf = re.search(r'"confidence"\s*:\s*([01](?:\.\d+)?)', text)
    if same and conf:
        return {
            "same": same.group(1) == "true",
            "confidence": float(conf.group(1)),
            "_recovered": "regex fallback(JSON 内嵌未转义引号)",
        }
    return None


def load_tracklet_index(ingest_root: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for path in sorted(ingest_root.glob("*/tracklets.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            attrs = row.get("attributes") or {}
            index[row["tracklet_id"]] = {
                "video_id": row["video_id"],
                "hero_ref": attrs.get("hero_ref"),
                "label": attrs.get("label"),
                "hero_score": float(attrs.get("hero_score") or 0.0),
            }
    if not index:
        raise SystemExit(f"no tracklets under {ingest_root}")
    return index


# ------------------------------------------------------------- attribution 通道


ATTRIBUTION_PROMPT = """以下多组图片截取自同一房间的多段视频,每组是自动跟踪系统判定的"同一物品实例"。
请独立判断并只输出一个 JSON 对象(不要输出其他文字):
{
  "same_item_group_pairs": [["A","B"], ...],  // 判为同一物理物品实例的组对(两两列出;没有则空数组)
  "invariant_features": ["...", "..."],        // 该类物品跨视角/跨光照最稳定的 2-4 条可辨识特征(中文)
  "confusables": "...",                        // 房间里最容易与之混淆的物品与区分要点(中文一句)
  "hard_crops": ["A2", "C1"],                  // 最不典型、最难与同组其他图对上的图(组字母+图序号)
  "detection_phrases_en": ["...", "..."],      // 供开放词汇检测器用的 2-3 条英文短语(含颜色/材质/图案)
  "notes": "..."                               // 其他有助于跨视频匹配调优的观察(中文,可空)
}
判断"同一实例"须基于颜色/图案/磨损/配件等实物证据,同类但不同件不算同一实例。"""


def cmd_attribution(args: argparse.Namespace) -> int:
    tracklets = load_tracklet_index(args.ingest_root)
    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    by_label = {e["display_label_zh"]: e for e in gt["entities"]}
    by_id = {e["anchor_id"]: e for e in gt["entities"]}
    wanted = []
    for token in args.anchors.split(","):
        token = token.strip()
        ent = by_id.get(token) or by_label.get(token)
        if ent is None:
            match = [e for z, e in by_label.items() if token in z]
            if len(match) != 1:
                raise SystemExit(f"anchor not found or ambiguous: {token}")
            ent = match[0]
        wanted.append(ent)

    tid_to_entity: dict[str, str] = {}
    for line in (args.reid_dir / "entities.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for tid in row["tracklet_ids"]:
            tid_to_entity[tid] = row["entity_id"]

    args.out.mkdir(parents=True, exist_ok=True)
    merge_pairs_path = args.out / "merge_pairs.jsonl"
    merge_rows: list[dict] = []
    summary: list[dict] = []

    for ent in wanted:
        anchor_id = ent["anchor_id"]
        confirmed = [
            tid
            for tids in ent.get("confirmed_tracklet_ids_by_video", {}).values()
            for tid in tids
        ]
        fragments: dict[str, list[str]] = {}
        loose: list[str] = []
        for tid in confirmed:
            eid = tid_to_entity.get(tid)
            (fragments.setdefault(eid, []) if eid else loose).append(tid)
        ordered = sorted(fragments.items(), key=lambda kv: -len(kv[1]))[: args.max_groups]
        groups: dict[str, dict] = {}
        letters = "ABCDEFGH"
        for i, (eid, tids) in enumerate(ordered):
            sel = sorted(tids, key=lambda t: -tracklets[t]["hero_score"])[: args.max_crops]
            groups[letters[i]] = {"entity_id": eid, "shown": sel, "total": len(tids)}
        if loose:
            sel = sorted(loose, key=lambda t: -tracklets[t]["hero_score"])[: args.max_crops]
            groups["Z"] = {"entity_id": None, "shown": sel, "total": len(loose)}

        content: list[dict] = [{"type": "text", "text": ATTRIBUTION_PROMPT}]
        img_shas: list[str] = []
        for letter, g in groups.items():
            head = "散轨(未成实体)" if letter == "Z" else "系统判为同一实体"
            content.append(
                {"type": "text", "text": f"\n组{letter}({head},共{g['total']}轨):"}
            )
            for j, tid in enumerate(g["shown"], 1):
                ref = tracklets[tid]["hero_ref"]
                content.append({"type": "text", "text": f"组{letter}-图{j}:"})
                content.append(image_part(ref))
                img_shas.append(_sha(Path(ref).read_bytes()))

        cache_key = _sha(
            ("attr-v1|" + MODEL + "|" + ATTRIBUTION_PROMPT + "|" + "|".join(img_shas)).encode()
        )
        n_images = len(img_shas)
        print(f"[{anchor_id}] {ent['display_label_zh']}: {len(groups)} 组 {n_images} 图 …", flush=True)
        reply = _cached_chat(
            "attribution",
            cache_key,
            [{"role": "user", "content": content}],
            n_images,
        )
        parsed = _parse_json_block(reply["content"])
        record = {
            "anchor_id": anchor_id,
            "display_label_zh": ent["display_label_zh"],
            "groups": groups,
            "tutor": parsed,
            "tutor_raw": None if parsed else reply["content"],
            "usage": reply["usage"],
            "cached": reply["cached"],
        }
        (args.out / f"{anchor_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if parsed:
            for a, b in parsed.get("same_item_group_pairs", []):
                ga, gb = groups.get(str(a).strip()), groups.get(str(b).strip())
                if not ga or not gb or not ga["shown"] or not gb["shown"]:
                    continue
                merge_rows.append(
                    {
                        "tracklet_a": ga["shown"][0],
                        "tracklet_b": gb["shown"][0],
                        "source": f"attribution:{anchor_id}:{a}-{b}",
                    }
                )
        summary.append(
            {
                "anchor_id": anchor_id,
                "label": ent["display_label_zh"],
                "n_fragments": len([g for g in groups.values() if g["entity_id"]]),
                "n_loose": groups.get("Z", {}).get("total", 0),
                "parsed": bool(parsed),
            }
        )

    merge_pairs_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merge_rows),
        encoding="utf-8",
    )
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"anchors": len(summary), "merge_pairs": len(merge_rows)}, ensure_ascii=False))
    return 0


# ------------------------------------------------------------------ pairs 通道


PAIR_PROMPT = """两张图片来自同一房间不同视频的自动跟踪截图。请判断它们是否为**同一个物理物品实例**
(同一件实物;同类别但不同件不算)。依据颜色、图案、文字、磨损、配件、尺寸比例等实物证据。
只输出一个 JSON 对象:{"same": true 或 false, "confidence": 0.0到1.0, "reason": "不超过30字"}"""


def _judge_pair(tracklets: dict, pair: dict) -> dict:
    a, b = pair["tracklet_a"], pair["tracklet_b"]
    ref_a, ref_b = tracklets[a]["hero_ref"], tracklets[b]["hero_ref"]
    sha_a, sha_b = _sha(Path(ref_a).read_bytes()), _sha(Path(ref_b).read_bytes())
    cache_key = _sha(f"pair-v1|{MODEL}|{PAIR_PROMPT}|{sha_a}|{sha_b}".encode())
    content = [
        {"type": "text", "text": PAIR_PROMPT},
        {"type": "text", "text": "图1:"},
        image_part(ref_a),
        {"type": "text", "text": "图2:"},
        image_part(ref_b),
    ]
    reply = _cached_chat("pairs", cache_key, [{"role": "user", "content": content}], 2)
    parsed = _parse_json_block(reply["content"])
    return {
        **pair,
        "tutor": parsed,
        "tutor_raw": None if parsed else reply["content"],
        "cached": reply["cached"],
    }


def cmd_pairs(args: argparse.Namespace) -> int:
    tracklets = load_tracklet_index(args.ingest_root)
    lo, hi = (float(x) for x in args.band.split(":"))
    near_miss, low_margin = [], []
    for line in (args.reid_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        a, b = row["tracklet_a"], row["tracklet_b"]
        if a not in tracklets or b not in tracklets:
            continue
        item = {
            "tracklet_a": a,
            "tracklet_b": b,
            "score": row["score"],
            "margin": row.get("margin"),
            "decision": row["decision"],
            "source": "candidates",
        }
        if row.get("assigned"):
            if row["score"] < hi or (row.get("margin") or 1.0) < args.low_margin:
                low_margin.append(item)
        elif lo <= row["score"] < 0.84:
            near_miss.append(item)

    rng = random.Random(args.seed)
    near_miss.sort(key=lambda r: (r["tracklet_a"], r["tracklet_b"]))
    low_margin.sort(key=lambda r: (r["tracklet_a"], r["tracklet_b"]))
    half = args.max_pairs // 2
    chosen = rng.sample(near_miss, min(half, len(near_miss))) + rng.sample(
        low_margin, min(args.max_pairs - half, len(low_margin))
    )
    if args.extra and args.extra.exists():
        seen = {(r["tracklet_a"], r["tracklet_b"]) for r in chosen}
        for line in args.extra.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (row["tracklet_a"], row["tracklet_b"])
            if key not in seen and all(t in tracklets for t in key):
                chosen.append({**row, "score": None, "margin": None, "decision": None})
                seen.add(key)
    print(
        f"near_miss 池 {len(near_miss)} / low_margin 池 {len(low_margin)} → 送判 {len(chosen)} 对",
        flush=True,
    )

    results: list[dict] = []
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_judge_pair, tracklets, p): p for p in chosen}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                results.append(fut.result())
            except StepfunAPIError as exc:
                failed += 1
                results.append({**futures[fut], "tutor": None, "error": str(exc)[:200]})
            if i % 25 == 0 or i == len(chosen):
                print(f"  {i}/{len(chosen)} (失败 {failed})", flush=True)

    results.sort(key=lambda r: (r["tracklet_a"], r["tracklet_b"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results),
        encoding="utf-8",
    )
    parsed_n = sum(1 for r in results if r.get("tutor"))
    same_n = sum(1 for r in results if (r.get("tutor") or {}).get("same") is True)
    print(
        json.dumps(
            {"pairs": len(results), "parsed": parsed_n, "tutor_same": same_n, "failed": failed},
            ensure_ascii=False,
        )
    )
    return 0 if failed / max(1, len(chosen)) <= 0.05 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    at = sub.add_parser("attribution")
    at.add_argument("--reid-dir", required=True, type=Path)
    at.add_argument("--ingest-root", required=True, type=Path)
    at.add_argument("--gt", required=True, type=Path)
    at.add_argument("--anchors", required=True, help="anchor_id 或中文名,逗号分隔")
    at.add_argument("--out", required=True, type=Path)
    at.add_argument("--max-crops", type=int, default=6)
    at.add_argument("--max-groups", type=int, default=6)
    at.set_defaults(func=cmd_attribution)

    pr = sub.add_parser("pairs")
    pr.add_argument("--reid-dir", required=True, type=Path)
    pr.add_argument("--ingest-root", required=True, type=Path)
    pr.add_argument("--out", required=True, type=Path)
    pr.add_argument("--extra", type=Path, default=None, help="attribution merge_pairs.jsonl")
    pr.add_argument("--band", default="0.76:0.92", help="不确定带 score 范围 lo:hi")
    pr.add_argument("--low-margin", type=float, default=0.04)
    pr.add_argument("--max-pairs", type=int, default=400)
    pr.add_argument("--seed", type=int, default=20260718)
    pr.add_argument("--workers", type=int, default=6)
    pr.set_defaults(func=cmd_pairs)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
