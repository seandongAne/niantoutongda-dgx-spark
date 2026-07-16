#!/usr/bin/env python
"""stitch 参数探针:在真实 ingest 产物上输出决定 stitch 阈值所需的分布。

只读诊断,不写任何 S3 产物;输出为单个 JSON,供人工定参后写入 reid v2 配置。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.tools.reid.model import ReIDConfig, StitchConfig, Vocabulary, load_features  # noqa: E402
from backend.tools.reid.stitch import _co_occurs, _cosine, _label_compatible, stitch_features  # noqa: E402

THRESHOLDS = (0.80, 0.85, 0.88, 0.90, 0.92, 0.95)
FOCUS_CATEGORIES = ("water_bottle", "cabinet", "bookshelf", "suitcase", "security_camera", "lamp")


def _deciles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        f"p{p}": round(ordered[min(len(ordered) - 1, int(len(ordered) * p / 100))], 6)
        for p in (10, 25, 50, 75, 90, 95, 99)
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest-root", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = ReIDConfig.from_yaml(args.config)
    vocab = Vocabulary.from_json(args.vocab)
    features = load_features(args.ingest_root, vocab=vocab, embedding_dim=config.embedding_dim)

    obs_hist = Counter(min(feature.observation_count, 5) for feature in features)
    category_counts = Counter(feature.category_id or "uncategorised" for feature in features)

    by_video = defaultdict(list)
    for feature in features:
        by_video[feature.video_id].append(feature)

    pair_cosines: list[float] = []
    cooccur_pairs = 0
    compatible_pairs = 0
    for video_features in by_video.values():
        ordered = sorted(video_features, key=lambda f: f.tracklet_id)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                if not _label_compatible(a, b):
                    continue
                compatible_pairs += 1
                pair_cosines.append(_cosine(a.vector, b.vector))
                if _co_occurs(a, b):
                    cooccur_pairs += 1

    sweeps = {}
    for threshold in THRESHOLDS:
        probe_config = replace(config, stitch=StitchConfig(enabled=True, min_cosine=threshold))
        result = stitch_features(features, probe_config)
        after_categories = Counter(f.category_id or "uncategorised" for f in result.features)
        sweeps[f"{threshold:.2f}"] = {
            "tracklets_after": len(result.features),
            "merge_count": result.report["merge_count"],
            "cooccurrence_vetoes": result.report["vetoes"]["co_occurrence"],
            "per_video": result.report["per_video"],
            "focus_categories": {c: after_categories.get(c, 0) for c in FOCUS_CATEGORIES},
        }

    report = {
        "ingest_root": str(args.ingest_root),
        "config_version": config.version,
        "tracklet_count": len(features),
        "observation_count_histogram": {
            ("5+" if k == 5 else str(k)): obs_hist[k] for k in sorted(obs_hist)
        },
        "category_counts_top": dict(category_counts.most_common(20)),
        "same_video_same_label_pairs": compatible_pairs,
        "cooccurring_pairs": cooccur_pairs,
        "pair_cosine_deciles": _deciles(pair_cosines),
        "stitch_threshold_sweep": sweeps,
        "focus_categories_before": {c: category_counts.get(c, 0) for c in FOCUS_CATEGORIES},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("tracklet_count", "same_video_same_label_pairs", "cooccurring_pairs")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
