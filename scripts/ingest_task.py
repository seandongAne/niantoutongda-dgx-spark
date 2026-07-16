#!/usr/bin/env python
"""批量 ingest — 多段视频 → S1+S2 正式产物(Observation/Tracklet/证据裁剪/嵌入)。

用法(节点主环境):
  python scripts/ingest_task.py --out local-data/ingest_a \
      --config-version dev-a-vocab6 \
      --vocab fixtures/dev_a/vocab.json \
      --videos v1=local-data/g0_a_old1_v2.mp4 v2=local-data/g0_a_old2.mp4 v3=local-data/g0_a_old3.mp4

旧的 ``--prompts "a,b,..."`` 入口仍保留，二者择一。

产物: <out>/<video_id>/{keyframes,evidence,observations.jsonl,tracklets.jsonl,audit-events.jsonl}
结尾打印逐视频标签×轨迹统计,供锚点对账。
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config-version", required=True)
    prompt_source = ap.add_mutually_exclusive_group(required=True)
    prompt_source.add_argument("--vocab", help="canonical detection vocabulary JSON")
    prompt_source.add_argument("--prompts", help="legacy comma-separated prompt list")
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--tile-box-threshold", type=float, default=0.22)
    ap.add_argument("--frame-batch-size", type=int, default=2)
    ap.add_argument("--tile-overlap", type=float, default=0.20)
    ap.add_argument("--clutter-tile-count", type=int, default=2)
    ap.add_argument("--tile-max-area-ratio", type=float, default=0.12)
    ap.add_argument("--tile-edge-margin-ratio", type=float, default=0.03)
    ap.add_argument("--tile-max-per-canonical", type=int, default=3)
    ap.add_argument("--stationary-min-ms", type=int, default=2000)
    ap.add_argument("--adaptive-tile-quantile", type=float, default=0.10)
    ap.add_argument("--adaptive-tile-max", type=int, default=12)
    ap.add_argument("--adaptive-tile-min-gap-ms", type=int, default=2000)
    ap.add_argument(
        "--no-stationary-tiles",
        action="store_true",
        help="disable 2x2 + selective 3x3 tiles for >=stationary-min-ms views",
    )
    ap.add_argument("--videos", nargs="+", required=True, help="video_id=path ...")
    args = ap.parse_args()

    from backend.pipeline.detect import GroundingDinoDetector
    from backend.pipeline.embed import Dinov2Embedder
    from backend.pipeline.ingest import ingest_video
    from backend.pipeline.vocab import load_vocabulary

    if args.vocab:
        prompts = load_vocabulary(args.vocab).compile()
        print(
            f"[vocab] {len(prompts)} prompts in {len(prompts.batches)} batches: "
            + " | ".join(", ".join(batch) for batch in prompts.batches),
            flush=True,
        )
    else:
        prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
        if not prompts:
            ap.error("--prompts must contain at least one non-empty prompt")

    detector = GroundingDinoDetector(
        str(Path.home() / "models" / "IDEA-Research__grounding-dino-base"),
        box_threshold=args.box_threshold,
        tile_box_threshold=args.tile_box_threshold,
        image_batch_size=args.frame_batch_size,
        tile_overlap=args.tile_overlap,
        clutter_tile_count=args.clutter_tile_count,
        tile_max_area_ratio=args.tile_max_area_ratio,
        tile_edge_margin_ratio=args.tile_edge_margin_ratio,
        tile_max_per_canonical=args.tile_max_per_canonical,
    )
    embedder = Dinov2Embedder(str(Path.home() / "models" / "facebook__dinov2-base"))

    summaries = {}
    for spec in args.videos:
        video_id, _, path = spec.partition("=")
        t0 = time.perf_counter()
        result = ingest_video(
            video_id=video_id,
            video_path=path,
            prompts=prompts,
            workdir=Path(args.out) / video_id,
            detector=detector,
            embedder=embedder,
            config_version=args.config_version,
            stationary_min_ms=args.stationary_min_ms,
            enable_stationary_tiles=not args.no_stationary_tiles,
            adaptive_tile_quantile=args.adaptive_tile_quantile,
            adaptive_tile_max_count=args.adaptive_tile_max,
            adaptive_tile_min_gap_ms=args.adaptive_tile_min_gap_ms,
        )
        labels = Counter(t.attributes["label"] for t in result.tracklets)
        summaries[video_id] = labels
        print(
            f"[{video_id}] {len(result.keyframes)} kf, {len(result.observations)} obs, "
            f"{len(result.tracklets)} tracklets, {time.perf_counter() - t0:.0f}s "
            f"(detect={result.detection_elapsed_s:.1f}s, batch={args.frame_batch_size if result.frame_batching_used else 1}, "
            f"tiled_kf={result.tiled_keyframe_count}, tile_mode={result.tile_selection_mode})",
            flush=True,
        )

    all_labels = sorted({label for c in summaries.values() for label in c})
    print("\n==== 标签 × 视频 轨迹数 ====")
    header = "label".ljust(36) + "".join(v.ljust(6) for v in summaries)
    print(header)
    for label in all_labels:
        row = label.ljust(36) + "".join(str(summaries[v].get(label, 0)).ljust(6) for v in summaries)
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
