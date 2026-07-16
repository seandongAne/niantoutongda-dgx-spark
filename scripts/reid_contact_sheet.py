#!/usr/bin/env python
"""为 S3 人工真值标注生成小体积 tracklet contact sheet。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

CELL_W, CELL_H = 220, 190
IMAGE_SIZE = (200, 145)


def _resolve(ref: str, ingest_root: Path) -> Path:
    path = Path(ref)
    if path.is_absolute() and path.exists():
        return path
    for base in (Path.cwd(), ingest_root, *ingest_root.parents):
        candidate = base / path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(ref)


def _sheet(category: str, features, ingest_root: Path, out_path: Path, columns: int) -> None:
    rows = max(1, math.ceil(len(features) / columns))
    canvas = Image.new("RGB", (columns * CELL_W, rows * CELL_H), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, feature in enumerate(features):
        x, y = (index % columns) * CELL_W, (index // columns) * CELL_H
        if feature.tracklet.prototype_refs:
            source = _resolve(feature.tracklet.prototype_refs[0], ingest_root)
            with Image.open(source).convert("RGB") as image:
                thumb = ImageOps.contain(image, IMAGE_SIZE)
                slot = Image.new("RGB", IMAGE_SIZE, "#eeeeee")
                slot.paste(thumb, ((IMAGE_SIZE[0] - thumb.width) // 2, (IMAGE_SIZE[1] - thumb.height) // 2))
                canvas.paste(slot, (x + 10, y + 5))
        draw.text((x + 10, y + 153), feature.tracklet_id, fill="black", font=font)
        draw.text((x + 10, y + 168), feature.raw_label[:30], fill="#444444", font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=88, optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest-root", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--categories", default="bookshelf,suitcase,cabinet,water_bottle", help="comma-separated category_id"
    )
    parser.add_argument("--columns", type=int, default=6)
    args = parser.parse_args()

    from backend.tools.reid.model import Vocabulary, load_features

    ingest_root = Path(args.ingest_root)
    vocab = Vocabulary.from_json(args.vocab)
    # Contact sheet 不使用向量数值，但 loader 同时完成 v5 引用/维度完整性检查。
    features = load_features(ingest_root, vocab=vocab, embedding_dim=768)
    categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    counts = {}
    for category in categories:
        selected = sorted(
            (feature for feature in features if feature.category_id == category),
            key=lambda feature: (feature.video_id, feature.tracklet_id),
        )
        counts[category] = len(selected)
        _sheet(category, selected, ingest_root, Path(args.out) / f"hard-negative-{category}.jpg", args.columns)
    (Path(args.out) / "contact-sheet-counts.json").write_text(
        json.dumps(counts, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
