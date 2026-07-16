#!/usr/bin/env python
"""17 锚点 → tracklet 人工对账物料生成器。

对 entities.template.json 里的每个锚点,按 category 聚出 v6 候选 tracklet,
渲染 per-anchor contact sheet(hero 图 + id + 视频分组),并产出预填候选的
review JSON。候选只是"同类别全集",成员归属必须由数据所有者目视裁定;
本脚本不做任何相似度筛选,避免把真值采集变成模型自证。
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
COLUMNS = 6


def _resolve(ref: str, ingest_root: Path) -> Path:
    path = Path(ref)
    if path.is_absolute() and path.exists():
        return path
    for base in (Path.cwd(), ingest_root, *ingest_root.parents):
        candidate = base / path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(ref)


def _sheet(title: str, features, ingest_root: Path, out_path: Path) -> None:
    import math

    rows = max(1, math.ceil(len(features) / COLUMNS)) if features else 1
    canvas = Image.new("RGB", (COLUMNS * CELL_W, rows * CELL_H + 30), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((10, 8), title, fill="black", font=font)
    for index, feature in enumerate(features):
        x, y = (index % COLUMNS) * CELL_W, (index // COLUMNS) * CELL_H + 30
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
    canvas.save(out_path, quality=85, optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest-root", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--template", required=True, help="entities.template.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ingest_root = Path(args.ingest_root)
    vocab = Vocabulary.from_json(args.vocab)
    template = json.loads(Path(args.template).read_text())
    features = load_features(ingest_root, vocab=vocab, embedding_dim=768)
    out = Path(args.out)

    review_entities = []
    counts = {}
    for anchor in template["entities"]:
        anchor_id = anchor["anchor_id"]
        category = anchor["category_id"]
        selected = sorted(
            (f for f in features if f.category_id == category),
            key=lambda f: (f.video_id, f.tracklet_id),
        )
        counts[anchor_id] = len(selected)
        _sheet(
            f"{anchor_id} {anchor['display_label_zh']} (category={category}, candidates={len(selected)})",
            selected,
            ingest_root,
            out / f"{anchor_id}.jpg",
        )
        by_video: dict[str, list[str]] = {}
        for feature in selected:
            by_video.setdefault(feature.video_id, []).append(feature.tracklet_id)
        review_entities.append(
            {
                "anchor_id": anchor_id,
                "display_label_zh": anchor["display_label_zh"],
                "category_id": category,
                "candidate_tracklets_by_video": by_video,
                # 数据所有者裁定后填以下两个字段;candidates 只是同类别全集
                "confirmed_tracklet_ids_by_video": {},
                "visible_in": [],
            }
        )

    review = {
        "schema_version": "1.0",
        "dataset_version": template.get("dataset_version", "unknown"),
        "status": "prefilled_candidates_pending_data_owner_confirmation",
        "note": (
            "candidate_tracklets_by_video 为同 category 全集(未做任何相似度筛选);"
            "请对照 per-anchor jpg 把确认成员填入 confirmed_tracklet_ids_by_video。"
            "四组硬负真值 (A-D) 由 entities.template.json 的 hard_negative_pairs "
            "在锚点确认后自动导出,无需单独标注。"
        ),
        "hard_negative_pairs": template.get("hard_negative_pairs", []),
        "entities": review_entities,
    }
    (out / "anchor_review.v6.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"anchors": len(review_entities), "candidate_counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
