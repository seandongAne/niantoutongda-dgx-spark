#!/usr/bin/env python
"""无冻结 GT 的静态邻域拓扑 ReID 代理实验。

输入只允许 ingest 检测框、词表、逐视角向量、tutor 判断与 pseudo identities。
脚本先识别在三段视频中稳定呈单实例的类别，再为每条轨迹构建最多 K 个邻居的
局部图；邻域分只在证据覆盖时重排现有视觉分，不扩大候选召回集合。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.pipeline.vocab import Vocabulary  # noqa: E402
from backend.tools.reid.multiview import quantile_calibrate  # noqa: E402
from backend.tools.reid.neighborhood import (  # noqa: E402
    CONTEXT_FORMAT_VERSION,
    ContextEvidence,
    NeighborhoodSignature,
    build_frame_index,
    build_signatures,
    collapse_stitched_geometries,
    compare_signatures,
    load_stitch_groups,
    load_track_geometries,
    select_anchor_categories,
)
from scripts.reid_multiview_proxy_sweep import (  # noqa: E402
    _split_tutor_rows,
    blend_prepared_scores,
    build_leave_last_video_pairs,
    build_view_spaces,
    load_pseudo_identities,
    load_tutor_pairs,
    load_views,
    prepare_pair_scores,
    reject_frozen_gt_paths,
    retrieval_metrics,
    sha256_file,
    tutor_metrics,
)


Pair = tuple[str, str]


def _video_id(tracklet_id: str) -> str:
    video, separator, suffix = tracklet_id.rpartition("_t")
    if not separator or not video or not suffix:
        raise ValueError(f"cannot infer video id from tracklet: {tracklet_id}")
    return video


def _contextualize(
    baseline: Mapping[Pair, float],
    signatures: Mapping[str, NeighborhoodSignature],
    *,
    min_shared_anchors: int,
    blend: float,
    score_min: float = 0.70,
    score_max: float = 0.82,
) -> tuple[dict[Pair, float], dict[Pair, ContextEvidence | None], dict[str, int]]:
    """按视频对校准邻域排序，使原有分数分布与阈值语义保持可比。"""

    if not 0 <= blend <= 1:
        raise ValueError("blend must be in [0, 1]")
    evidence: dict[Pair, ContextEvidence | None] = {}
    if not 0 <= score_min < score_max <= 1:
        raise ValueError("context score band must satisfy 0 <= min < max <= 1")
    score_band_pairs = {
        pair for pair, score in baseline.items() if score_min < score < score_max
    }
    for pair in baseline:
        left, right = signatures.get(pair[0]), signatures.get(pair[1])
        evidence[pair] = (
            compare_signatures(
                left, right, min_shared_anchors=min_shared_anchors
            )
            if left is not None and right is not None
            else None
        )

    # 没有足够共同锚点的 pair 必须逐字保持 baseline，不能把 0.5 中性占位符
    # 当成真实邻域排序依据。分位数映射只在“分数带内且有证据”的子集内部进行。
    eligible_pairs = {
        pair for pair in score_band_pairs if evidence[pair] is not None
    }
    grouped: dict[tuple[str, str], list[Pair]] = defaultdict(list)
    for pair in eligible_pairs:
        grouped[tuple(sorted((_video_id(pair[0]), _video_id(pair[1]))))].append(pair)

    calibrated: dict[Pair, float] = {}
    for video_pair, pairs in sorted(grouped.items()):
        raw_context = {pair: evidence[pair].score for pair in pairs}
        calibrated.update(
            quantile_calibrate(
                {pair: baseline[pair] for pair in pairs}, raw_context
            )
        )
    scores = dict(baseline)
    scores.update(
        {
            pair: (1.0 - blend) * baseline[pair] + blend * calibrated[pair]
            for pair in calibrated
        }
    )
    counts = {
        "pairs": len(baseline),
        "score_band_pairs": len(score_band_pairs),
        "covered": len(eligible_pairs),
        "uncovered": len(score_band_pairs - eligible_pairs),
        "applied": len(calibrated),
    }
    return scores, evidence, counts


def _write_context_sidecar(
    path: Path,
    *,
    candidate_path: Path,
    signatures: Mapping[str, NeighborhoodSignature],
    min_shared_anchors: int,
) -> dict[str, int]:
    reject_frozen_gt_paths((candidate_path, path))
    pairs: dict[Pair, dict] = {}
    for line in candidate_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        a, b = str(row["tracklet_a"]), str(row["tracklet_b"])
        pair = tuple(sorted((a, b)))
        pairs.setdefault(pair, {"a": a, "b": b})

    path.parent.mkdir(parents=True, exist_ok=True)
    covered = 0
    with path.open("w", encoding="utf-8") as handle:
        for pair in sorted(pairs):
            left, right = signatures.get(pair[0]), signatures.get(pair[1])
            item = (
                compare_signatures(
                    left, right, min_shared_anchors=min_shared_anchors
                )
                if left is not None and right is not None
                else None
            )
            covered += item is not None
            row = {
                "schema_version": CONTEXT_FORMAT_VERSION,
                "tracklet_a": pair[0],
                "tracklet_b": pair[1],
                "score": round(item.score, 8) if item is not None else None,
                "shared_anchors": list(item.shared_anchors) if item is not None else [],
                "overlap": round(item.overlap, 8) if item is not None else None,
                "relation_agreement": (
                    round(item.relation_agreement, 8) if item is not None else None
                ),
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"candidate_pairs": len(pairs), "covered": covered, "uncovered": len(pairs) - covered}


def run_proxy(
    *,
    ingest_root: str | Path,
    vocab_path: str | Path,
    stitch_map_path: str | Path,
    candidates_path: str | Path,
    views_path: str | Path,
    tutor_pairs_path: str | Path,
    pseudo_labels_path: str | Path,
    projection_path: str | Path | None,
    projection_sha256: str | None,
    output_dir: str | Path,
    frame_size: tuple[int, int] = (1920, 1080),
    max_neighbors_grid: Sequence[int] = (3, 5, 7),
    min_shared_grid: Sequence[int] = (3,),
    blends: Sequence[float] = (0.10, 0.20, 0.30),
    min_anchor_frames: int = 5,
    min_anchor_single_fraction: float = 0.65,
    anchor_nms_iou: float = 0.30,
    min_tutor_confidence: float = 0.90,
    max_diff_p95_delta: float = 0.005,
    holdout_modulus: int = 5,
    holdout_bucket: int = 0,
    context_score_min: float = 0.70,
    context_score_max: float = 0.82,
) -> dict:
    input_paths = [
        ingest_root,
        vocab_path,
        stitch_map_path,
        candidates_path,
        views_path,
        tutor_pairs_path,
        pseudo_labels_path,
    ]
    if projection_path is not None:
        input_paths.append(projection_path)
    reject_frozen_gt_paths(input_paths)

    vocab = Vocabulary.from_json(vocab_path)
    original_geometries = load_track_geometries(ingest_root, vocab=vocab)
    frame_index = build_frame_index(original_geometries)
    videos = sorted({geometry.video_id for geometry in original_geometries.values()})
    frame_sizes = {video: frame_size for video in videos}
    anchor_categories, anchor_diagnostics = select_anchor_categories(
        frame_index,
        videos=videos,
        min_visible_frames=min_anchor_frames,
        min_single_fraction=min_anchor_single_fraction,
        nms_iou=anchor_nms_iou,
    )
    if not anchor_categories:
        raise ValueError("no stable single-instance anchor categories")

    raw_views = load_views(views_path)
    view_spaces, baseline_vectors, projection_metadata = build_view_spaces(
        raw_views,
        projection_path=projection_path,
        projection_sha256=projection_sha256,
    )
    raw_space = view_spaces["raw"]
    tutor_rows, tutor_counts = load_tutor_pairs(
        tutor_pairs_path, raw_views, min_confidence=min_tutor_confidence
    )
    pseudo_samples, pseudo_counts = load_pseudo_identities(
        pseudo_labels_path, raw_views
    )
    query, gallery, retrieval_pairs, split_counts = build_leave_last_video_pairs(
        pseudo_samples
    )
    selection_rows, holdout_rows = _split_tutor_rows(
        tutor_rows,
        holdout_modulus=holdout_modulus,
        holdout_bucket=holdout_bucket,
    )
    tutor_pairs = [row["pair"] for row in tutor_rows]

    # 当前生产主工作点：projected mean baseline + raw max_pair 0.50。
    tutor_prepared = prepare_pair_scores(
        tutor_pairs,
        raw_space,
        max_views=6,
        baseline_vectors=baseline_vectors,
    )
    retrieval_prepared = prepare_pair_scores(
        retrieval_pairs,
        raw_space,
        max_views=6,
        baseline_vectors=baseline_vectors,
    )
    baseline_tutor = blend_prepared_scores(
        tutor_prepared, method="max_pair", blend=0.50
    )
    baseline_retrieval = blend_prepared_scores(
        retrieval_prepared, method="max_pair", blend=0.50
    )
    baseline_metrics = {
        "selection_tutor": tutor_metrics(selection_rows, baseline_tutor),
        "holdout_tutor": tutor_metrics(holdout_rows, baseline_tutor),
        "pseudo_retrieval": retrieval_metrics(
            query, gallery, pseudo_samples, baseline_retrieval
        ),
    }

    signature_cache: dict[int, dict[str, NeighborhoodSignature]] = {}
    grid = []
    for max_neighbors in max_neighbors_grid:
        signatures = build_signatures(
            original_geometries,
            frame_index,
            anchor_categories=anchor_categories,
            frame_sizes=frame_sizes,
            max_neighbors=int(max_neighbors),
            nms_iou=anchor_nms_iou,
        )
        signature_cache[int(max_neighbors)] = signatures
        for min_shared in min_shared_grid:
            for blend in blends:
                tutor_scores, tutor_evidence, tutor_coverage = _contextualize(
                    baseline_tutor,
                    signatures,
                    min_shared_anchors=int(min_shared),
                    blend=float(blend),
                    score_min=context_score_min,
                    score_max=context_score_max,
                )
                retrieval_scores, retrieval_evidence, retrieval_coverage = _contextualize(
                    baseline_retrieval,
                    signatures,
                    min_shared_anchors=int(min_shared),
                    blend=float(blend),
                    score_min=context_score_min,
                    score_max=context_score_max,
                )
                selection = tutor_metrics(selection_rows, tutor_scores)
                holdout = tutor_metrics(holdout_rows, tutor_scores)
                retrieval = retrieval_metrics(
                    query, gallery, pseudo_samples, retrieval_scores
                )
                base_selection = baseline_metrics["selection_tutor"]
                diff_p95_delta = (
                    round(selection["different_p95"] - base_selection["different_p95"], 6)
                    if selection["different_p95"] is not None
                    and base_selection["different_p95"] is not None
                    else None
                )
                eligible = (
                    diff_p95_delta is not None
                    and diff_p95_delta <= max_diff_p95_delta
                    and retrieval["recall_at_5"]
                    >= baseline_metrics["pseudo_retrieval"]["recall_at_5"]
                    and retrieval_coverage["covered"] > 0
                )
                grid.append(
                    {
                        "max_neighbors": int(max_neighbors),
                        "min_shared_anchors": int(min_shared),
                        "blend": float(blend),
                        "eligible": eligible,
                        "guards": {
                            "different_p95_delta": diff_p95_delta,
                            "different_p95_delta_max": max_diff_p95_delta,
                            "pseudo_recall_at_5_non_degrading": (
                                retrieval["recall_at_5"]
                                >= baseline_metrics["pseudo_retrieval"]["recall_at_5"]
                            ),
                        },
                        "coverage": {
                            "tutor": tutor_coverage,
                            "pseudo_retrieval": retrieval_coverage,
                            "tutor_shared_anchor_histogram": dict(
                                sorted(
                                    defaultdict(int, {
                                        str(count): sum(
                                            item is not None
                                            and context_score_min < baseline_tutor[pair] < context_score_max
                                            and len(item.shared_anchors) == count
                                            for pair, item in tutor_evidence.items()
                                        )
                                        for count in range(1, max_neighbors + 1)
                                    }).items()
                                )
                            ),
                            "pseudo_evidence_count": retrieval_coverage["covered"],
                        },
                        "selection_tutor": selection,
                        "holdout_tutor": holdout,
                        "pseudo_retrieval": retrieval,
                    }
                )

    eligible = [row for row in grid if row["eligible"]]
    pool = eligible or grid

    def _candidate_key(row: dict) -> tuple:
        retrieval = row["pseudo_retrieval"]
        tutor = row["selection_tutor"]
        return (
            -retrieval["recall_at_1"],
            -(tutor["auc"] if tutor["auc"] is not None else -1.0),
            -retrieval["mean_reciprocal_rank"],
            row["blend"],
            row["min_shared_anchors"],
            row["max_neighbors"],
        )

    winner = min(pool, key=_candidate_key)
    production_geometries = collapse_stitched_geometries(
        original_geometries, load_stitch_groups(stitch_map_path)
    )
    production_signatures = build_signatures(
        production_geometries,
        frame_index,
        anchor_categories=anchor_categories,
        frame_sizes=frame_sizes,
        max_neighbors=winner["max_neighbors"],
        nms_iou=anchor_nms_iou,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sidecar = output / "context-scores.jsonl"
    sidecar_counts = _write_context_sidecar(
        sidecar,
        candidate_path=Path(candidates_path),
        signatures=production_signatures,
        min_shared_anchors=winner["min_shared_anchors"],
    )

    report = {
        "schema_version": "1.0",
        "scope": "STATIC_NEIGHBORHOOD_PROXY_ONLY_NO_FROZEN_GT",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selection_status": "eligible_winner" if eligible else "no_candidate_passed_guards",
        "selection_policy": (
            "guard tutor different-p95 and pseudo R@5; maximize pseudo R@1, "
            "tutor AUC and pseudo MRR; prefer smaller blend"
        ),
        "frozen_gt_policy": {
            "read": False,
            "note": "selection consumes tutor and pseudo identities only",
        },
        "inputs": {
            "ingest_root": str(ingest_root),
            "vocab": {"path": str(vocab_path), "sha256": sha256_file(vocab_path)},
            "stitch_map": {"path": str(stitch_map_path), "sha256": sha256_file(stitch_map_path)},
            "candidates": {"path": str(candidates_path), "sha256": sha256_file(candidates_path)},
            "views": {"path": str(views_path), "sha256": sha256_file(views_path)},
            "tutor_pairs": {"path": str(tutor_pairs_path), "sha256": sha256_file(tutor_pairs_path)},
            "pseudo_labels": {"path": str(pseudo_labels_path), "sha256": sha256_file(pseudo_labels_path)},
            "projection": projection_metadata,
        },
        "counts": {
            "videos": videos,
            "original_tracklets": len(original_geometries),
            "anchor_categories": sorted(anchor_categories),
            "anchor_category_count": len(anchor_categories),
            "tutor": tutor_counts,
            "pseudo": pseudo_counts,
            "pseudo_split": split_counts,
            "production_sidecar": sidecar_counts,
        },
        "parameters": {
            "frame_size": list(frame_size),
            "min_anchor_frames": min_anchor_frames,
            "min_anchor_single_fraction": min_anchor_single_fraction,
            "anchor_nms_iou": anchor_nms_iou,
            "max_neighbors_grid": list(max_neighbors_grid),
            "min_shared_grid": list(min_shared_grid),
            "blends": list(blends),
            "baseline": "projected-mean + raw-max-pair blend-0.50",
            "context_scope": "uncertain_only",
            "context_score_band": [context_score_min, context_score_max],
        },
        "baseline": baseline_metrics,
        "winner": winner,
        "grid": grid,
        "anchor_diagnostics": anchor_diagnostics,
        "artifacts": {
            "context_scores": {
                "path": str(sidecar),
                "sha256": sha256_file(sidecar),
                "format": CONTEXT_FORMAT_VERSION,
            }
        },
    }
    report_path = output / "proxy-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest-root", default="results/hero_s1/ingest")
    parser.add_argument("--vocab", default="fixtures/hero_s1/vocab.json")
    parser.add_argument(
        "--stitch-map", default="results/hero_s1/reid-closure-multiview-v2/stitch-map.json"
    )
    parser.add_argument(
        "--candidates", default="results/hero_s1/reid-closure-multiview-v2/candidates.jsonl"
    )
    parser.add_argument("--views", default="results/hero_s1/multiview-dinov2-v1/views.npz")
    parser.add_argument("--tutor-pairs", default="results/autotune/tutor_pairs.jsonl")
    parser.add_argument(
        "--pseudo-labels", default="fixtures/autotune/pseudo_labels.tutor-v1.json"
    )
    parser.add_argument(
        "--projection", default="results/sf1/hero_s1_autotune_v1/projection.npz"
    )
    parser.add_argument(
        "--projection-sha256",
        default="79a74f135acad1f43ac25d391a01becdbac1dc8b219430a4a14c6ddaf6da02e4",
    )
    parser.add_argument(
        "--out", default="results/acceptance/HERO_S1/neighborhood-context-v4"
    )
    parser.add_argument("--frame-width", type=int, default=1920)
    parser.add_argument("--frame-height", type=int, default=1080)
    args = parser.parse_args()
    report = run_proxy(
        ingest_root=args.ingest_root,
        vocab_path=args.vocab,
        stitch_map_path=args.stitch_map,
        candidates_path=args.candidates,
        views_path=args.views,
        tutor_pairs_path=args.tutor_pairs,
        pseudo_labels_path=args.pseudo_labels,
        projection_path=args.projection,
        projection_sha256=args.projection_sha256,
        output_dir=args.out,
        frame_size=(args.frame_width, args.frame_height),
    )
    print(
        json.dumps(
            {
                "selection_status": report["selection_status"],
                "anchor_categories": report["counts"]["anchor_category_count"],
                "baseline": report["baseline"],
                "winner": report["winner"],
                "sidecar": report["counts"]["production_sidecar"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["selection_status"] == "eligible_winner" else 2


if __name__ == "__main__":
    raise SystemExit(main())
