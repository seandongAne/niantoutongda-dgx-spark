#!/usr/bin/env python
"""StepFun LLM 生成 GDINO 词表候选 — 纯文本进出,喂本地打分回路。

定位(docs/STEPFUN_API_PLAYBOOK.md 红线内的 dev-time 工具):
- 输入只有物品清单文本(中文名+外观描述),不含任何家庭影像/音频;
- 云端输出**只是候选**,一律经 word_candidate_scan.py(GDINO 真机扫描)+
  word_candidate_rank.py(冻结公式判卷)本地裁决后才可能进词表;
- 只在本地 Mac 运行(key 在 .env);演示主链不调云。

产物直接对接扫描脚本的 --candidates 输入格式:{category: [phrase, ...]}。

用法:
  .venv/bin/python scripts/vocab_candidates_gen.py \\
      --items fixtures/hero_s1/items.json \\
      --out results/wordgen/hero_s1_candidates.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stepfun_api import StepfunAPIError, chat_completion  # noqa: E402

# 词表工程原则 = D3 实拍教训 + 48 词扫描判卷(results/wordscan/ranking.json)的胜出模式
SYSTEM_PROMPT = """\
你是开放词表检测(GroundingDINO)的检测短语工程师。给定家居物品清单,为每件物品生成英文检测短语候选。

硬规则:
1. 每条短语 1-4 个英文单词,全小写,只含字母和空格。
2. 短语是"召回网"不是分类器:宁可宽泛可召回,不追求排他;身份精度由下游重识别负责。
3. 只用具体可见的视觉词:颜色、材质、形状、功能部件(rolling/handle/plush);禁止品牌名、抽象词、比喻。
4. 每件物品给 6 条候选,覆盖三种策略:
   a. 属性+物体(如 pink rolling kids luggage / teal play kitchen fridge)——实测最强模式;
   b. 上位词或近义物体(玩具冰箱→toy refrigerator / play kitchen);
   c. 按外观描述的"看起来像什么"(白色柱体夜灯→white cylinder lamp)——外观词优先于功能词。
5. 同一物品的 6 条候选彼此要有真差异(换属性、换词根、换粒度),不要只调换语序。

只输出一个 JSON 对象:{"<category>": ["phrase1", ...], ...},key 用输入给出的 category,不要任何解释。"""


def build_user_prompt(items: list[dict]) -> str:
    lines = ["物品清单:"]
    for it in items:
        desc = it.get("description_zh", "")
        lines.append(
            f"- category={it['canonical_id']}  名称:{it['name_zh']}"
            + (f"  外观:{desc}" if desc else "")
        )
    return "\n".join(lines)


def parse_candidates(text: str, expected: set[str]) -> dict[str, list[str]]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"输出中未找到 JSON 对象: {text[:200]!r}")
    data = json.loads(m.group(0))
    bad_phrase = re.compile(r"[^a-z ]")
    out: dict[str, list[str]] = {}
    problems: list[str] = []
    for cat in sorted(expected):
        phrases = data.get(cat)
        if not isinstance(phrases, list) or not phrases:
            problems.append(f"{cat}: 缺失或为空")
            continue
        cleaned = []
        for p in phrases:
            p = str(p).strip().lower()
            if not p or bad_phrase.search(p) or len(p.split()) > 4:
                problems.append(f"{cat}: 非法短语 {p!r}")
                continue
            if p not in cleaned:
                cleaned.append(p)
        if cleaned:
            out[cat] = cleaned
    if problems:
        print("[warn] " + "; ".join(problems), file=sys.stderr)
    missing = expected - set(out)
    if missing:
        raise ValueError(f"以下 category 无合法候选: {sorted(missing)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", required=True, type=Path,
                    help='JSON: {"items": [{"canonical_id","name_zh","description_zh"?}]}')
    ap.add_argument("--model", default="step-3.5-flash")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    items = json.loads(args.items.read_text(encoding="utf-8"))["items"]
    expected = {it["canonical_id"] for it in items}
    if len(expected) != len(items):
        raise SystemExit("canonical_id 重复")

    try:
        data = chat_completion(
            model=args.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(items)},
            ],
            temperature=args.temperature,
        )
    except StepfunAPIError as exc:
        raise SystemExit(str(exc))

    text = data["choices"][0]["message"]["content"]
    candidates = parse_candidates(text, expected)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    usage = data.get("usage", {})
    manifest = {
        "model": args.model,
        "temperature": args.temperature,
        "items": len(items),
        "phrases": sum(len(v) for v in candidates.values()),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "note": "云端候选,未判卷;入词表前必须过 word_candidate_scan + rank 本地裁决",
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
