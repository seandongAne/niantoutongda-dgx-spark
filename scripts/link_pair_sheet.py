#!/usr/bin/env python
"""渲染跨视频链接对的 hero 图对照表,供人工目检误链接。

输入 pairs JSON: [{"a": "v1_t001", "b": "v2_t002", "score": 0.87}, ...]
每行一对:左右 hero 图 + id + 分数。阈值调整前的固定目检动作。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from PIL import Image, ImageDraw, ImageFont, ImageOps  # noqa: E402

from backend.tools.reid.model import Vocabulary, load_features  # noqa: E402

CELL_W, CELL_H = 220, 190
IMAGE_SIZE = (200, 145)
COLUMNS = 2  # 每行一对


def _resolve(ref: str, ingest_root: Path) -> Path:
    path = Path(ref)
    if path.is_absolute() and path.exists():
        return path
    for base in (Path.cwd(), ingest_root, *ingest_root.parents):
        candidate = base / path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(ref)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest-root", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--stitch-map", help="可选:成员经代表 id 映射到 hero")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ingest_root = Path(args.ingest_root)
    vocab = Vocabulary.from_json(args.vocab)
    pairs = json.loads(Path(args.pairs).read_text())
    features = {f.tracklet_id: f for f in load_features(ingest_root, vocab=vocab, embedding_dim=768)}

    rep_hero: dict[str, str] = {}
    if args.stitch_map:
        groups = json.loads(Path(args.stitch_map).read_text()).get("groups", {})
        for rep, members in groups.items():
            rep_hero[rep] = rep  # 代表 id 本身就是合法原始 id,直接用其 hero

    rows = len(pairs)
    canvas = Image.new("RGB", (COLUMNS * CELL_W + 140, rows * (CELL_H + 8) + 10), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row, pair in enumerate(pairs):
        y = row * (CELL_H + 8) + 5
        draw.text((COLUMNS * CELL_W + 10, y + 70), f"score={pair.get('score', '?')}", fill="black", font=font)
        for col, tid in enumerate((pair["a"], pair["b"])):
            x = col * CELL_W + 5
            feature = features.get(rep_hero.get(tid, tid))
            if feature and feature.tracklet.prototype_refs:
                source = _resolve(feature.tracklet.prototype_refs[0], ingest_root)
                with Image.open(source).convert("RGB") as image:
                    thumb = ImageOps.contain(image, IMAGE_SIZE)
                    slot = Image.new("RGB", IMAGE_SIZE, "#eeeeee")
                    slot.paste(thumb, ((IMAGE_SIZE[0] - thumb.width) // 2, (IMAGE_SIZE[1] - thumb.height) // 2))
                    canvas.paste(slot, (x, y))
            draw.text((x, y + 148), tid, fill="black", font=font)
            label = feature.raw_label[:28] if feature else "?"
            draw.text((x, y + 163), label, fill="#444444", font=font)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=85, optimize=True)
    print(json.dumps({"pairs": rows, "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
