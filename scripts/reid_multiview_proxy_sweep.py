#!/usr/bin/env python
"""Select a multi-view ReID reranker without reading frozen hero ground truth.

The sweep consumes only the DINO per-view sidecar, AutoTune tutor judgements,
and optional pseudo identities.  It evaluates the production methods
``max_pair``, ``mean_chamfer`` and ``symmetric_top2`` at fixed blend values.
The data-owner file ``fixtures/hero_s1/annotations/anchor_review.confirmed.json``
is forbidden both by interface validation and by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.tools.reid.multiview import quantile_calibrate  # noqa: E402
from backend.tools.sf1.projection import NumpyProjectionHead  # noqa: E402


METHODS = ("max_pair", "mean_chamfer", "symmetric_top2")
BLENDS = (0.25, 0.50, 0.75)
ACCEPTED_VIEW_FORMATS = frozenset(
    {"reid-multiview-v1", "reid-multiview-embeddings-v1"}
)
FORBIDDEN_GT = (
    PROJ / "fixtures" / "hero_s1" / "annotations" / "anchor_review.confirmed.json"
).resolve()

Vector = tuple[float, ...]
Views = dict[str, tuple[Vector, ...]]
Pair = tuple[str, str]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pair_key(a: str, b: str) -> Pair:
    return tuple(sorted((a, b)))


def _video_id(tracklet_id: str) -> str:
    video, separator, suffix = tracklet_id.rpartition("_t")
    if not separator or not video or not suffix:
        raise ValueError(f"cannot infer video id from tracklet: {tracklet_id}")
    return video


def reject_frozen_gt_paths(paths: Iterable[str | Path]) -> None:
    """Fail before opening a user-supplied path that could be the frozen GT."""

    for raw in paths:
        path = Path(raw).resolve()
        if path == FORBIDDEN_GT or path.name == FORBIDDEN_GT.name:
            raise ValueError(f"frozen hero GT is forbidden in proxy selection: {path}")


def load_views(path: str | Path) -> Views:
    """Load either sidecar spelling currently present in the working tree."""

    artifact = Path(path)
    reject_frozen_gt_paths((artifact,))
    grouped: dict[str, list[tuple[int, Vector]]] = defaultdict(list)
    with np.load(artifact, allow_pickle=False) as data:
        version = str(data["format_version"].item())
        if version not in ACCEPTED_VIEW_FORMATS:
            raise ValueError(f"unsupported multiview format: {version}")
        ids = np.asarray(data["tracklet_ids"]).astype(str)
        if "view_indices" in data:
            indices = np.asarray(data["view_indices"], dtype=np.int32)
        elif "view_index" in data:
            indices = np.asarray(data["view_index"], dtype=np.int32)
        else:
            raise ValueError("multiview artifact lacks view_index/view_indices")
        vectors = np.asarray(data["vectors"], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] <= 0:
        raise ValueError(f"multiview vectors must be [N,D], got {vectors.shape}")
    if len(ids) != len(indices) or len(ids) != len(vectors):
        raise ValueError("multiview arrays have inconsistent row counts")
    if not np.isfinite(vectors).all():
        raise ValueError("multiview vectors contain non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if (norms < 1e-12).any():
        raise ValueError("multiview vectors contain zero-norm rows")
    vectors = vectors / norms[:, None]
    for tracklet_id, index, vector in zip(ids, indices, vectors):
        grouped[str(tracklet_id)].append(
            (int(index), tuple(float(value) for value in vector))
        )
    result: Views = {}
    for tracklet_id, rows in sorted(grouped.items()):
        ordered = sorted(rows)
        indices_for_track = [index for index, _ in ordered]
        if indices_for_track != list(range(len(ordered))):
            raise ValueError(f"non-contiguous view indices for {tracklet_id}")
        result[tracklet_id] = tuple(vector for _, vector in ordered)
    if not result:
        raise ValueError("multiview artifact contains no tracklets")
    return result


def load_tutor_pairs(
    path: str | Path,
    views: Mapping[str, Sequence[Vector]],
    *,
    min_confidence: float,
) -> tuple[list[dict], dict[str, int]]:
    source = Path(path)
    reject_frozen_gt_paths((source,))
    by_pair: dict[Pair, dict] = {}
    counts = defaultdict(int)
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        tutor = row.get("tutor") or {}
        confidence = tutor.get("confidence")
        same = tutor.get("same")
        if not isinstance(confidence, (int, float)) or confidence < min_confidence:
            counts["below_confidence"] += 1
            continue
        if not isinstance(same, bool):
            counts["missing_boolean_label"] += 1
            continue
        a, b = str(row["tracklet_a"]), str(row["tracklet_b"])
        if _video_id(a) == _video_id(b):
            counts["same_video"] += 1
            continue
        if a not in views or b not in views:
            counts["missing_views"] += 1
            continue
        key = _pair_key(a, b)
        normalized = {
            "pair": key,
            "same": same,
            "confidence": float(confidence),
            "source_line": line_number,
        }
        previous = by_pair.get(key)
        if previous is not None and previous["same"] != same:
            raise ValueError(f"conflicting tutor labels for pair {key}")
        if previous is None or normalized["confidence"] > previous["confidence"]:
            by_pair[key] = normalized
    rows = [by_pair[key] for key in sorted(by_pair)]
    counts["kept"] = len(rows)
    counts["same"] = sum(row["same"] for row in rows)
    counts["different"] = len(rows) - counts["same"]
    if not rows:
        raise ValueError("no eligible cross-video tutor pairs")
    return rows, dict(sorted(counts.items()))


def load_pseudo_identities(
    path: str | Path,
    views: Mapping[str, Sequence[Vector]],
) -> tuple[dict[str, tuple[str, str]], dict[str, int]]:
    source = Path(path)
    reject_frozen_gt_paths((source,))
    raw = json.loads(source.read_text(encoding="utf-8"))
    samples: dict[str, tuple[str, str]] = {}
    counts = defaultdict(int)
    for entity in raw.get("entities", []):
        identity = str(entity["anchor_id"])
        for video_id, tracklets in entity.get(
            "confirmed_tracklet_ids_by_video", {}
        ).items():
            for tracklet_id in tracklets:
                tracklet_id = str(tracklet_id)
                if tracklet_id not in views:
                    counts["missing_views"] += 1
                    continue
                if tracklet_id in samples:
                    raise ValueError(f"duplicate pseudo-labeled tracklet: {tracklet_id}")
                actual_video = _video_id(tracklet_id)
                if actual_video != str(video_id):
                    raise ValueError(
                        f"pseudo video mismatch for {tracklet_id}: {video_id} != {actual_video}"
                    )
                samples[tracklet_id] = (identity, actual_video)
    counts["kept"] = len(samples)
    counts["identities"] = len({identity for identity, _ in samples.values()})
    if not samples:
        raise ValueError("no pseudo samples have multiview vectors")
    return samples, dict(sorted(counts.items()))


def build_leave_last_video_pairs(
    samples: Mapping[str, tuple[str, str]],
) -> tuple[list[str], list[str], list[Pair], dict[str, int]]:
    """Mirror the SF1 pseudo split, then remove same-video query/gallery pairs."""

    by_identity: dict[str, list[str]] = defaultdict(list)
    for tracklet_id, (identity, _) in samples.items():
        by_identity[identity].append(tracklet_id)
    query: list[str] = []
    gallery: list[str] = []
    excluded_identities = 0
    for identity, tracklets in sorted(by_identity.items()):
        videos = sorted({samples[tracklet_id][1] for tracklet_id in tracklets})
        if len(videos) < 2:
            excluded_identities += 1
            continue
        held_out = videos[-1]
        query.extend(
            sorted(tracklet_id for tracklet_id in tracklets if samples[tracklet_id][1] == held_out)
        )
        gallery.extend(
            sorted(tracklet_id for tracklet_id in tracklets if samples[tracklet_id][1] != held_out)
        )
    query = sorted(set(query))
    gallery = sorted(set(gallery))
    pairs = [
        (query_id, gallery_id)
        for query_id in query
        for gallery_id in gallery
        if samples[query_id][1] != samples[gallery_id][1]
    ]
    if not pairs:
        raise ValueError("pseudo split has no cross-video retrieval pairs")
    return query, gallery, pairs, {
        "queries": len(query),
        "gallery": len(gallery),
        "pairs": len(pairs),
        "excluded_single_video_identities": excluded_identities,
    }


def _mean_vector(view_set: Sequence[Vector]) -> Vector:
    values = np.asarray(view_set, dtype=np.float32)
    mean = values.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("view-set mean is not normalizable")
    return tuple(float(value) for value in mean / norm)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b))))


def _all_set_similarities(
    a: Sequence[Vector], b: Sequence[Vector], *, max_views: int
) -> dict[str, float]:
    """Compute all production formulas from one NumPy cosine matrix."""

    left = np.asarray(a[:max_views], dtype=np.float32)
    right = np.asarray(b[:max_views], dtype=np.float32)
    if left.ndim != 2 or right.ndim != 2 or not len(left) or not len(right):
        raise ValueError("multiview score requires non-empty 2-D view sets")
    if left.shape[1] != right.shape[1]:
        raise ValueError("multiview view dimensions differ")
    matrix = np.clip(left @ right.T, 0.0, 1.0)
    max_pair = float(matrix.max())
    chamfer = 0.5 * (
        float(matrix.max(axis=1).mean()) + float(matrix.max(axis=0).mean())
    )
    candidates = sorted(
        (
            (float(matrix[row, column]), row, column)
            for row in range(matrix.shape[0])
            for column in range(matrix.shape[1])
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    used_rows, used_columns, selected = set(), set(), []
    for score, row, column in candidates:
        if row in used_rows or column in used_columns:
            continue
        used_rows.add(row)
        used_columns.add(column)
        selected.append(score)
        if len(selected) == 2:
            break
    top2 = sum(selected) / len(selected)
    return {
        "max_pair": max(0.0, min(1.0, max_pair)),
        "mean_chamfer": max(0.0, min(1.0, chamfer)),
        "symmetric_top2": max(0.0, min(1.0, 0.5 * chamfer + 0.5 * top2)),
    }


def prepare_pair_scores(
    pairs: Sequence[Pair],
    views: Mapping[str, Sequence[Vector]],
    *,
    max_views: int,
    baseline_vectors: Mapping[str, Vector] | None = None,
) -> dict[str, dict[Pair, float]]:
    """Cache baseline and every calibrated formula once for all blend values."""

    used_tracklets = sorted({tracklet_id for pair in pairs for tracklet_id in pair})
    means = (
        {tracklet_id: baseline_vectors[tracklet_id] for tracklet_id in used_tracklets}
        if baseline_vectors is not None
        else {
            tracklet_id: _mean_vector(views[tracklet_id])
            for tracklet_id in used_tracklets
        }
    )
    baseline = {
        pair: _cosine(means[pair[0]], means[pair[1]])
        for pair in pairs
    }
    raw_by_method: dict[str, dict[Pair, float]] = {
        method: {} for method in METHODS
    }
    for pair in pairs:
        values = _all_set_similarities(
            views[pair[0]], views[pair[1]], max_views=max_views
        )
        for method in METHODS:
            raw_by_method[method][pair] = values[method]
    grouped: dict[tuple[str, str], list[Pair]] = defaultdict(list)
    for pair in pairs:
        grouped[tuple(sorted((_video_id(pair[0]), _video_id(pair[1]))))].append(pair)
    prepared = {"baseline": baseline}
    for method in METHODS:
        calibrated: dict[Pair, float] = {}
        for video_pair in sorted(grouped):
            keys = grouped[video_pair]
            calibrated.update(
                quantile_calibrate(
                    {key: baseline[key] for key in keys},
                    {key: raw_by_method[method][key] for key in keys},
                )
            )
        prepared[method] = calibrated
    return prepared


def blend_prepared_scores(
    prepared: Mapping[str, Mapping[Pair, float]],
    *,
    method: str | None,
    blend: float,
) -> dict[Pair, float]:
    if method is not None and method not in METHODS:
        raise ValueError(f"unsupported method: {method}")
    if not 0 <= blend <= 1:
        raise ValueError("blend must be in [0, 1]")
    baseline = prepared["baseline"]
    if method is None or blend == 0:
        return dict(baseline)
    calibrated = prepared[method]
    return {
        pair: (1.0 - blend) * baseline[pair] + blend * calibrated[pair]
        for pair in baseline
    }


def score_pairs(
    pairs: Sequence[Pair],
    views: Mapping[str, Sequence[Vector]],
    *,
    method: str | None,
    blend: float,
    max_views: int,
) -> dict[Pair, float]:
    """Convenience entrypoint used by focused tests and one-off diagnostics."""

    return blend_prepared_scores(
        prepare_pair_scores(pairs, views, max_views=max_views),
        method=method,
        blend=blend,
    )


def build_view_spaces(
    raw_views: Mapping[str, Sequence[Vector]],
    *,
    projection_path: str | Path | None,
    projection_sha256: str | None,
) -> tuple[dict[str, Views], dict[str, Vector], dict | None]:
    """Return raw/projected view sets and the deployed mean-vector baseline."""

    raw = {
        tracklet_id: tuple(tuple(float(value) for value in vector) for vector in view_set)
        for tracklet_id, view_set in raw_views.items()
    }
    raw_means = {
        tracklet_id: _mean_vector(view_set) for tracklet_id, view_set in raw.items()
    }
    if projection_path is None:
        if projection_sha256:
            raise ValueError("projection_sha256 requires projection_path")
        return {"raw": raw}, raw_means, None
    reject_frozen_gt_paths((projection_path,))
    head = NumpyProjectionHead.load(
        projection_path, expected_sha256=projection_sha256
    )
    if any(len(vector) != head.input_dim for vector in raw_means.values()):
        raise ValueError("projection input dimension does not match multiview vectors")
    projected: Views = {}
    for tracklet_id, view_set in raw.items():
        values = head.apply(np.asarray(view_set, dtype=np.float32))
        projected[tracklet_id] = tuple(
            tuple(float(value) for value in vector) for vector in values
        )
    ids = sorted(raw_means)
    projected_means_matrix = head.apply(
        np.asarray([raw_means[tracklet_id] for tracklet_id in ids], dtype=np.float32)
    )
    projected_baseline = {
        tracklet_id: tuple(float(value) for value in vector)
        for tracklet_id, vector in zip(ids, projected_means_matrix)
    }
    return (
        {"raw": raw, "projected": projected},
        projected_baseline,
        {
            "path": str(projection_path),
            "sha256": sha256_file(projection_path),
            "expected_sha256": projection_sha256,
            "input_dim": head.input_dim,
            "output_dim": head.output_dim,
            "mode": head.mode,
            "residual_scale": head.residual_scale,
        },
    )


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values), percentile)), 6)


def _auc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += positive > negative
            wins += 0.5 * (positive == negative)
    return round(wins / (len(positives) * len(negatives)), 6)


def tutor_metrics(rows: Sequence[dict], scores: Mapping[Pair, float]) -> dict:
    labels = [bool(row["same"]) for row in rows]
    values = [float(scores[row["pair"]]) for row in rows]
    same_scores = [score for label, score in zip(labels, values) if label]
    diff_scores = [score for label, score in zip(labels, values) if not label]
    return {
        "pairs": len(rows),
        "same": len(same_scores),
        "different": len(diff_scores),
        "auc": _auc(labels, values),
        "same_mean": round(float(np.mean(same_scores)), 6) if same_scores else None,
        "same_p10": _percentile(same_scores, 10),
        "different_mean": round(float(np.mean(diff_scores)), 6) if diff_scores else None,
        "different_p95": _percentile(diff_scores, 95),
        "mean_separation": (
            round(float(np.mean(same_scores) - np.mean(diff_scores)), 6)
            if same_scores and diff_scores
            else None
        ),
    }


def retrieval_metrics(
    query: Sequence[str],
    gallery: Sequence[str],
    samples: Mapping[str, tuple[str, str]],
    scores: Mapping[Pair, float],
) -> dict:
    ranks: list[int] = []
    margins: list[float] = []
    skipped = 0
    for query_id in query:
        candidates = []
        for gallery_id in gallery:
            if samples[query_id][1] == samples[gallery_id][1]:
                continue
            pair = (query_id, gallery_id)
            if pair not in scores:
                pair = (gallery_id, query_id)
            value = scores[pair]
            is_positive = samples[query_id][0] == samples[gallery_id][0]
            candidates.append((value, gallery_id, is_positive))
        positives = [item for item in candidates if item[2]]
        negatives = [item for item in candidates if not item[2]]
        if not positives or not negatives:
            skipped += 1
            continue
        ordered = sorted(candidates, key=lambda item: (-item[0], item[1]))
        rank = next(index for index, item in enumerate(ordered, 1) if item[2])
        ranks.append(rank)
        margins.append(max(item[0] for item in positives) - max(item[0] for item in negatives))
    if not ranks:
        raise ValueError("no pseudo queries have both positives and negatives")
    return {
        "queries": len(ranks),
        "skipped_queries": skipped,
        "recall_at_1": round(sum(rank == 1 for rank in ranks) / len(ranks), 6),
        "recall_at_5": round(sum(rank <= 5 for rank in ranks) / len(ranks), 6),
        "mean_reciprocal_rank": round(sum(1.0 / rank for rank in ranks) / len(ranks), 6),
        "top1_margin_mean": round(float(np.mean(margins)), 6),
    }


def _split_tutor_rows(
    rows: Sequence[dict], *, holdout_modulus: int, holdout_bucket: int
) -> tuple[list[dict], list[dict]]:
    if holdout_modulus < 2:
        raise ValueError("holdout_modulus must be >= 2")
    if not 0 <= holdout_bucket < holdout_modulus:
        raise ValueError("holdout_bucket is out of range")
    selection, holdout = [], []
    for row in rows:
        key = "\0".join(row["pair"]).encode("utf-8")
        bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % holdout_modulus
        (holdout if bucket == holdout_bucket else selection).append(row)
    if not selection:
        raise ValueError("tutor selection split is empty")
    return selection, holdout


def run_proxy_sweep(
    *,
    views_path: str | Path,
    tutor_pairs_path: str | Path,
    pseudo_labels_path: str | Path,
    min_tutor_confidence: float = 0.90,
    methods: Sequence[str] = METHODS,
    blends: Sequence[float] = BLENDS,
    max_views: int = 6,
    max_diff_p95_delta: float = 0.005,
    holdout_modulus: int = 5,
    holdout_bucket: int = 0,
    projection_path: str | Path | None = None,
    projection_sha256: str | None = None,
    spaces: Sequence[str] | None = None,
) -> dict:
    input_paths = [views_path, tutor_pairs_path, pseudo_labels_path]
    if projection_path is not None:
        input_paths.append(projection_path)
    reject_frozen_gt_paths(input_paths)
    raw_views = load_views(views_path)
    view_spaces, baseline_vectors, projection_metadata = build_view_spaces(
        raw_views,
        projection_path=projection_path,
        projection_sha256=projection_sha256,
    )
    selected_spaces = tuple(spaces) if spaces is not None else tuple(view_spaces)
    if not selected_spaces:
        raise ValueError("at least one view space is required")
    if len(set(selected_spaces)) != len(selected_spaces):
        raise ValueError("view spaces must be unique")
    unsupported_spaces = set(selected_spaces) - set(view_spaces)
    if unsupported_spaces:
        raise ValueError(
            f"view spaces unavailable without a compatible projection: {sorted(unsupported_spaces)}"
        )
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

    baseline_by_space = {}
    grid = []
    for space in selected_spaces:
        space_views = view_spaces[space]
        tutor_prepared = prepare_pair_scores(
            tutor_pairs,
            space_views,
            max_views=max_views,
            baseline_vectors=baseline_vectors,
        )
        retrieval_prepared = prepare_pair_scores(
            retrieval_pairs,
            space_views,
            max_views=max_views,
            baseline_vectors=baseline_vectors,
        )
        baseline_tutor_scores = blend_prepared_scores(
            tutor_prepared, method=None, blend=0.0
        )
        baseline_retrieval_scores = blend_prepared_scores(
            retrieval_prepared, method=None, blend=0.0
        )
        baseline_selection = tutor_metrics(selection_rows, baseline_tutor_scores)
        baseline_holdout = tutor_metrics(holdout_rows, baseline_tutor_scores)
        baseline_retrieval = retrieval_metrics(
            query, gallery, pseudo_samples, baseline_retrieval_scores
        )
        baseline_by_space[space] = {
            "selection_tutor": baseline_selection,
            "holdout_tutor": baseline_holdout,
            "pseudo_retrieval": baseline_retrieval,
        }

        for method in methods:
            if method not in METHODS:
                raise ValueError(f"unsupported sweep method: {method}")
            for blend in blends:
                tutor_scores = blend_prepared_scores(
                    tutor_prepared,
                    method=method,
                    blend=float(blend),
                )
                retrieval_scores = blend_prepared_scores(
                    retrieval_prepared,
                    method=method,
                    blend=float(blend),
                )
                selection = tutor_metrics(selection_rows, tutor_scores)
                holdout = tutor_metrics(holdout_rows, tutor_scores)
                retrieval = retrieval_metrics(
                    query, gallery, pseudo_samples, retrieval_scores
                )
                diff_p95_delta = None
                if (
                    selection["different_p95"] is not None
                    and baseline_selection["different_p95"] is not None
                ):
                    diff_p95_delta = round(
                        selection["different_p95"]
                        - baseline_selection["different_p95"],
                        6,
                    )
                eligible = (
                    diff_p95_delta is not None
                    and diff_p95_delta <= max_diff_p95_delta
                    and retrieval["recall_at_5"]
                    >= baseline_retrieval["recall_at_5"]
                )
                grid.append(
                    {
                        "space": space,
                        "method": method,
                        "blend": float(blend),
                        "eligible": eligible,
                        "guards": {
                            "different_p95_delta": diff_p95_delta,
                            "different_p95_delta_max": max_diff_p95_delta,
                            "pseudo_recall_at_5_non_degrading": (
                                retrieval["recall_at_5"]
                                >= baseline_retrieval["recall_at_5"]
                            ),
                        },
                        "selection_tutor": selection,
                        "holdout_tutor": holdout,
                        "pseudo_retrieval": retrieval,
                    }
                )

    eligible = [candidate for candidate in grid if candidate["eligible"]]
    pool = eligible or grid

    def candidate_key(candidate: dict) -> tuple:
        retrieval = candidate["pseudo_retrieval"]
        tutor = candidate["selection_tutor"]
        return (
            -retrieval["recall_at_1"],
            -(tutor["auc"] if tutor["auc"] is not None else -1.0),
            -retrieval["mean_reciprocal_rank"],
            0 if candidate["space"] == "projected" else 1,
            candidate["blend"],
            candidate["method"],
        )

    winner = min(pool, key=candidate_key)
    return {
        "schema_version": "1.0",
        "scope": "AUTOTUNE_MULTIVIEW_PROXY_ONLY_NO_FROZEN_GT",
        "selection_status": "eligible_winner" if eligible else "no_candidate_passed_guards",
        "selection_policy": (
            "guard tutor selection diff_p95 delta and pseudo R@5; then maximize "
            "pseudo R@1, tutor AUC, pseudo MRR; prefer smaller blend"
        ),
        "frozen_gt_policy": {
            "read": False,
            "forbidden_path": str(FORBIDDEN_GT.relative_to(PROJ)),
            "note": "winner selection consumes tutor and pseudo labels only",
        },
        "inputs": {
            "views": {"path": str(views_path), "sha256": sha256_file(views_path)},
            "tutor_pairs": {
                "path": str(tutor_pairs_path),
                "sha256": sha256_file(tutor_pairs_path),
            },
            "pseudo_labels": {
                "path": str(pseudo_labels_path),
                "sha256": sha256_file(pseudo_labels_path),
            },
        },
        "counts": {
            "view_tracklets": len(raw_views),
            "view_vectors": sum(len(value) for value in raw_views.values()),
            "tutor": tutor_counts,
            "pseudo": pseudo_counts,
            "pseudo_split": split_counts,
            "tutor_selection_pairs": len(selection_rows),
            "tutor_holdout_pairs": len(holdout_rows),
        },
        "parameters": {
            "methods": list(methods),
            "blends": [float(value) for value in blends],
            "spaces": list(selected_spaces),
            "max_views": max_views,
            "min_tutor_confidence": min_tutor_confidence,
            "max_diff_p95_delta": max_diff_p95_delta,
            "holdout": {
                "hash": "sha256(sorted_pair_with_nul)",
                "modulus": holdout_modulus,
                "bucket": holdout_bucket,
                "used_for_selection": False,
            },
        },
        "projection": projection_metadata,
        "baseline": baseline_by_space,
        "winner": winner,
        "grid": grid,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", required=True, help="multiview NPZ sidecar")
    parser.add_argument(
        "--tutor-pairs", default="results/autotune/tutor_pairs.jsonl"
    )
    parser.add_argument(
        "--pseudo-labels", default="fixtures/autotune/pseudo_labels.tutor-v1.json"
    )
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--blends", default=",".join(str(value) for value in BLENDS))
    parser.add_argument("--min-tutor-confidence", type=float, default=0.90)
    parser.add_argument("--max-views", type=int, default=6)
    parser.add_argument("--max-diff-p95-delta", type=float, default=0.005)
    parser.add_argument("--holdout-modulus", type=int, default=5)
    parser.add_argument("--holdout-bucket", type=int, default=0)
    parser.add_argument(
        "--projection",
        help="optional SF1 projection.npz; enables raw and projected view-space comparison",
    )
    parser.add_argument("--projection-sha256")
    parser.add_argument(
        "--spaces",
        help="comma-separated raw/projected; default is raw,projected with --projection, else raw",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = run_proxy_sweep(
        views_path=args.views,
        tutor_pairs_path=args.tutor_pairs,
        pseudo_labels_path=args.pseudo_labels,
        min_tutor_confidence=args.min_tutor_confidence,
        methods=tuple(value.strip() for value in args.methods.split(",") if value.strip()),
        blends=tuple(float(value) for value in args.blends.split(",") if value.strip()),
        max_views=args.max_views,
        max_diff_p95_delta=args.max_diff_p95_delta,
        holdout_modulus=args.holdout_modulus,
        holdout_bucket=args.holdout_bucket,
        projection_path=args.projection,
        projection_sha256=args.projection_sha256,
        spaces=(
            tuple(value.strip() for value in args.spaces.split(",") if value.strip())
            if args.spaces
            else None
        ),
    )
    output = Path(args.out)
    reject_frozen_gt_paths((output,))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    winner = report["winner"]
    print(
        json.dumps(
            {
                "selection_status": report["selection_status"],
                "space": winner["space"],
                "method": winner["method"],
                "blend": winner["blend"],
                "eligible": winner["eligible"],
                "pseudo_recall_at_1": winner["pseudo_retrieval"]["recall_at_1"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
