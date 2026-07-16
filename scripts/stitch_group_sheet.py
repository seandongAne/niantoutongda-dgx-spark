#!/usr/bin/env python
"""渲染 stitch-map.json 里每个合并组的成员 hero 图,供人工目检误合并。

每组一行:代表轨在最左,成员按 id 排;标题带余弦与视频。输出单张 jpg,
配套 json 记录组信息。合并组目检是 stitch 调参后的固定动作。
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
    parser.add_argument("--stitch-map", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ingest_root = Path(args.ingest_root)
    vocab = Vocabulary.from_json(args.vocab)
    stitch = json.loads(Path(args.stitch_map).read_text())
    groups = stitch.get("groups", {})
    if not groups:
        print(json.dumps({"groups": 0, "note": "stitch map has no merged groups"}))
        return 0

    features = {f.tracklet_id: f for f in load_features(ingest_root, vocab=vocab, embedding_dim=768)}
    cosine_by_pair = {
        tuple(sorted((m["a"], m["b"]))): m.get("cosine")
        for m in stitch.get("merges", [])
    }

    max_members = max(len(ms) for ms in groups.values())
    rows = len(groups)
    canvas = Image.new("RGB", (max_members * CELL_W + 10, rows * (CELL_H + 22) + 10), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row, (rep, members) in enumerate(sorted(groups.items())):
        y = row * (CELL_H + 22) + 8
        cosines = [
            cosine_by_pair.get(tuple(sorted((a, b))))
            for a in members
            for b in members
            if a < b and tuple(sorted((a, b))) in cosine_by_pair
        ]
        cos_text = ",".join(f"{c:.3f}" for c in cosines if c is not None)
        labels = {features[m].raw_label for m in members if m in features}
        draw.text(
            (8, y),
            f"group {rep} · members={len(members)} · cos=[{cos_text}] · labels={'/'.join(sorted(labels))[:60]}",
            fill="black",
            font=font,
        )
        for col, member in enumerate(members):
            x = col * CELL_W + 8
            feature = features.get(member)
            if feature and feature.tracklet.prototype_refs:
                source = _resolve(feature.tracklet.prototype_refs[0], ingest_root)
                with Image.open(source).convert("RGB") as image:
                    thumb = ImageOps.contain(image, IMAGE_SIZE)
                    slot = Image.new("RGB", IMAGE_SIZE, "#eeeeee")
                    slot.paste(thumb, ((IMAGE_SIZE[0] - thumb.width) // 2, (IMAGE_SIZE[1] - thumb.height) // 2))
                    canvas.paste(slot, (x, y + 16))
            draw.text((x, y + 16 + 148), member, fill="black", font=font)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=85, optimize=True)
    print(json.dumps({"groups": rows, "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
