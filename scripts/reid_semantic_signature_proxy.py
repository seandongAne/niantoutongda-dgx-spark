#!/usr/bin/env python
"""Leakage-aware proxy A/B for an instance semantic/visual reranker.

The script never reads frozen human GT.  ``select`` chooses one semantic weight
on pseudo identities assigned to the development partition.  ``holdout`` then
replays only that frozen choice on identity-disjoint pseudo identities.
Results are diagnostic proxy evidence, not a claim about hero Recall@1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

ALGORITHM_VERSION = "semantic-signature-reranker-v2"
EVALUATOR_CODE_PATHS = (
    Path("scripts/reid_semantic_signature_proxy.py"),
    Path("scripts/reid_multiview_proxy_sweep.py"),
    Path("backend/pipeline/vocab.py"),
    Path("backend/tools/reid/model.py"),
)

from backend.pipeline.vocab import Vocabulary, VocabTranscription  # noqa: E402
from backend.tools.reid.model import (  # noqa: E402
    COMPARABLE_ATTRIBUTE_KEYS,
    UNKNOWN_ATTRIBUTE_VALUES,
)
from scripts.reid_multiview_proxy_sweep import (  # noqa: E402
    Pair,
    Vector,
    blend_prepared_scores,
    build_leave_last_video_pairs,
    build_view_spaces,
    load_pseudo_identities,
    load_views,
    prepare_pair_scores,
    reject_frozen_gt_paths,
    sha256_file,
)


def _identity_digest(members: Sequence[str]) -> str:
    return hashlib.sha256("\0".join(sorted(members)).encode("utf-8")).hexdigest()


def identity_disjoint_split(
    samples: Mapping[str, tuple[str, str]], *, modulus: int, holdout_bucket: int
) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, str]], dict]:
    if modulus < 2:
        raise ValueError("identity holdout modulus must be >= 2")
    if not 0 <= holdout_bucket < modulus:
        raise ValueError("identity holdout bucket is out of range")
    by_identity: dict[str, list[str]] = defaultdict(list)
    for tracklet_id, (identity, _) in samples.items():
        by_identity[identity].append(tracklet_id)
    development_ids, holdout_ids = set(), set()
    commitments = {}
    for identity, members in sorted(by_identity.items()):
        digest = _identity_digest(members)
        bucket = int(digest[:16], 16) % modulus
        commitments[identity] = {"digest": digest, "bucket": bucket, "members": len(members)}
        (holdout_ids if bucket == holdout_bucket else development_ids).add(identity)
    if not development_ids or not holdout_ids:
        raise ValueError("identity split produced an empty partition")
    development = {
        tracklet_id: value
        for tracklet_id, value in samples.items()
        if value[0] in development_ids
    }
    holdout = {
        tracklet_id: value
        for tracklet_id, value in samples.items()
        if value[0] in holdout_ids
    }
    if set(development) & set(holdout):
        raise AssertionError("identity-disjoint split leaked tracklets")
    return development, holdout, {
        "hash": "sha256(sorted_identity_members_with_nul)",
        "modulus": modulus,
        "holdout_bucket": holdout_bucket,
        "development_identities": len(development_ids),
        "holdout_identities": len(holdout_ids),
        "development_tracklets": len(development),
        "holdout_tracklets": len(holdout),
        "tracklet_overlap": 0,
        "commitments": commitments,
    }


def _load_raw_labels(ingest_root: Path) -> dict[str, str]:
    labels = {}
    for path in sorted(ingest_root.glob("*/tracklets.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            labels[str(row["tracklet_id"])] = str((row.get("attributes") or {}).get("label", ""))
    if not labels:
        raise ValueError(f"no tracklets under {ingest_root}")
    return labels


def _ingest_input_manifest(ingest_root: Path) -> dict:
    paths = sorted(ingest_root.glob("*/tracklets.jsonl"))
    if not paths:
        raise ValueError(f"no tracklet inputs under {ingest_root}")
    return {
        "root": str(ingest_root),
        "files": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in paths
        ],
    }


def _evaluator_manifest() -> dict:
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "files": [
            {"path": str(path), "sha256": sha256_file(PROJ / path)}
            for path in EVALUATOR_CODE_PATHS
        ],
    }


def _load_attributes(path: Path) -> dict[str, dict[str, str]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row["tracklet_id"])] = {
            str(key): str(value)
            for key, value in (row.get("attributes") or {}).items()
        }
    return rows


def _known(value: str | None) -> str | None:
    normalized = " ".join(str(value or "").casefold().split())
    return None if normalized in UNKNOWN_ATTRIBUTE_VALUES else normalized


def build_signatures(
    tracklet_ids: Sequence[str],
    *,
    vocab: Vocabulary,
    raw_labels: Mapping[str, str],
    attributes: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, dict], dict]:
    signatures = {}
    status_counts: dict[str, int] = defaultdict(int)
    for tracklet_id in sorted(set(tracklet_ids)):
        attrs = attributes.get(tracklet_id, {})
        transcription = vocab.transcribe(
            raw_labels.get(tracklet_id),
            attrs.get("label_en"),
            attrs.get("label_zh"),
        )
        status_counts[transcription.status] += 1
        signatures[tracklet_id] = {
            "transcription": transcription,
            "attributes": {
                key: value
                for key in COMPARABLE_ATTRIBUTE_KEYS
                if (value := _known(attrs.get(key))) is not None
            },
        }
    mapped = status_counts.get("mapped", 0)
    total = len(signatures)
    return signatures, {
        "tracklets": total,
        "mapped": mapped,
        "mapped_rate": round(mapped / total, 6) if total else None,
        "status": dict(sorted(status_counts.items())),
    }


def _label_similarity(left: VocabTranscription, right: VocabTranscription) -> float | None:
    if left.category_id is None or right.category_id is None:
        return None
    confidence = min(left.confidence, right.confidence)
    if left.canonical_id == right.canonical_id:
        return confidence
    if left.category_id == right.category_id:
        return 0.75 * confidence
    return 0.0


def _attribute_similarity(left: Mapping[str, str], right: Mapping[str, str]) -> float | None:
    shared = sorted(set(left) & set(right))
    if not shared:
        return None
    return sum(left[key] == right[key] for key in shared) / len(shared)


def signature_scores(pairs: Sequence[Pair], signatures: Mapping[str, dict]) -> dict[Pair, float | None]:
    scores = {}
    for pair in pairs:
        left, right = signatures[pair[0]], signatures[pair[1]]
        label = _label_similarity(left["transcription"], right["transcription"])
        attribute = _attribute_similarity(left["attributes"], right["attributes"])
        components = []
        if label is not None:
            components.append((0.40, label))
        if attribute is not None:
            components.append((0.60, attribute))
        denominator = sum(weight for weight, _ in components)
        scores[pair] = (
            sum(weight * value for weight, value in components) / denominator
            if denominator
            else None
        )
    return scores


def _blend_signature(
    visual: Mapping[Pair, float],
    semantic: Mapping[Pair, float | None],
    *,
    weight: float,
) -> dict[Pair, float]:
    if not 0 <= weight <= 1:
        raise ValueError("semantic weight must be in [0,1]")
    return {
        pair: (
            visual[pair]
            if semantic[pair] is None
            else (1.0 - weight) * visual[pair] + weight * float(semantic[pair])
        )
        for pair in visual
    }


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [round(max(0.0, center - half_width), 6), round(min(1.0, center + half_width), 6)]


def _mcnemar_exact_p(gains: int, losses: int) -> float | None:
    discordant = gains + losses
    if discordant == 0:
        return None
    tail = sum(
        math.comb(discordant, index) * (0.5**discordant)
        for index in range(min(gains, losses) + 1)
    )
    return round(min(1.0, 2.0 * tail), 8)


def topk_retrieval_metrics(
    query: Sequence[str],
    gallery: Sequence[str],
    samples: Mapping[str, tuple[str, str]],
    baseline_scores: Mapping[Pair, float],
    candidate_scores: Mapping[Pair, float],
    *,
    top_k: int,
) -> dict:
    ranks: list[int | None] = []
    baseline_ranks: list[int | None] = []
    margins: list[float] = []
    query_outcomes = []
    positive_recalled = 0
    for query_id in query:
        rows = []
        for gallery_id in gallery:
            if samples[query_id][1] == samples[gallery_id][1]:
                continue
            pair = (query_id, gallery_id)
            rows.append(
                (
                    baseline_scores[pair],
                    candidate_scores[pair],
                    gallery_id,
                    samples[query_id][0] == samples[gallery_id][0],
                )
            )
        positives = [row for row in rows if row[3]]
        negatives = [row for row in rows if not row[3]]
        if not positives or not negatives:
            continue
        recalled = sorted(rows, key=lambda row: (-row[0], row[2]))[:top_k]
        positive_recalled += any(row[3] for row in recalled)
        baseline_rank = next(
            (index for index, row in enumerate(recalled, 1) if row[3]), None
        )
        ordered = sorted(recalled, key=lambda row: (-row[1], row[2]))
        rank = next((index for index, row in enumerate(ordered, 1) if row[3]), None)
        baseline_ranks.append(baseline_rank)
        ranks.append(rank)
        query_outcomes.append(
            {
                "query_id": query_id,
                "baseline_rank": baseline_rank,
                "candidate_rank": rank,
                "positive_in_baseline_top_k": baseline_rank is not None,
            }
        )
        best_positive = max((row[1] for row in recalled if row[3]), default=0.0)
        best_negative = max((row[1] for row in recalled if not row[3]), default=0.0)
        margins.append(best_positive - best_negative)
    if not ranks:
        raise ValueError("partition has no evaluable pseudo queries")
    baseline_top1 = [rank == 1 for rank in baseline_ranks]
    candidate_top1 = [rank == 1 for rank in ranks]
    gains = sum(not before and after for before, after in zip(baseline_top1, candidate_top1))
    losses = sum(before and not after for before, after in zip(baseline_top1, candidate_top1))
    baseline_successes = sum(baseline_top1)
    candidate_successes = sum(candidate_top1)
    return {
        "queries": len(ranks),
        "candidate_top_k": top_k,
        "positive_recalled": positive_recalled,
        "baseline_top_k_positive_coverage": round(positive_recalled / len(ranks), 6),
        "positive_recall_rate": round(positive_recalled / len(ranks), 6),
        "recall_at_1": round(candidate_successes / len(ranks), 6),
        "recall_at_1_wilson_95": _wilson_interval(candidate_successes, len(ranks)),
        "recall_at_5": round(sum(rank is not None and rank <= 5 for rank in ranks) / len(ranks), 6),
        "mean_reciprocal_rank": round(
            sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / len(ranks), 6
        ),
        "top1_margin_mean": round(sum(margins) / len(margins), 6),
        "paired_top1": {
            "baseline_correct": baseline_successes,
            "candidate_correct": candidate_successes,
            "gains": gains,
            "losses": losses,
            "unchanged_correct": sum(before and after for before, after in zip(baseline_top1, candidate_top1)),
            "unchanged_incorrect": sum(
                not before and not after
                for before, after in zip(baseline_top1, candidate_top1)
            ),
            "mcnemar_exact_two_sided_p": _mcnemar_exact_p(gains, losses),
        },
        "query_outcomes": query_outcomes,
    }


def _partition_scores(
    samples: Mapping[str, tuple[str, str]],
    *,
    views,
    baseline_vectors: Mapping[str, Vector],
    signatures: Mapping[str, dict],
    visual_method: str,
    visual_blend: float,
    max_views: int,
    top_k: int,
    semantic_weights: Sequence[float],
) -> tuple[dict, list[dict]]:
    query, gallery, pairs, counts = build_leave_last_video_pairs(samples)
    prepared = prepare_pair_scores(
        pairs,
        views,
        max_views=max_views,
        baseline_vectors=baseline_vectors,
    )
    visual = blend_prepared_scores(
        prepared,
        method=visual_method,
        blend=visual_blend,
    )
    semantic = signature_scores(pairs, signatures)
    baseline = topk_retrieval_metrics(
        query, gallery, samples, visual, visual, top_k=top_k
    )
    grid = []
    for weight in semantic_weights:
        candidate = _blend_signature(visual, semantic, weight=weight)
        metrics = topk_retrieval_metrics(
            query, gallery, samples, visual, candidate, top_k=top_k
        )
        grid.append(
            {
                "semantic_weight": float(weight),
                "candidate_universe_fixed": True,
                "metrics": metrics,
            }
        )
    return {"split": counts, "baseline": baseline}, grid


def _load_common(args):
    paths = [
        args.views,
        args.pseudo_labels,
        args.pseudo_manifest,
        args.attributes,
        args.ingest_root,
        args.vocab,
    ]
    if args.projection:
        paths.append(args.projection)
    reject_frozen_gt_paths(paths)
    if args.projection and not args.allow_transductive_projection:
        raise ValueError(
            "learned projection may contain holdout identities; omit --projection for "
            "an identity-independent proxy or pass --allow-transductive-projection to "
            "label the run as diagnostic"
        )
    if args.allow_transductive_projection and not args.projection:
        raise ValueError("--allow-transductive-projection requires --projection")
    pseudo_manifest = json.loads(Path(args.pseudo_manifest).read_text(encoding="utf-8"))
    if pseudo_manifest.get("gt_overlap", {}).get("status") != "NOT_COMPUTED_NO_GT_READ":
        raise ValueError("pseudo labels must be rebuilt without reading frozen GT")
    prefixes = pseudo_manifest.get("sources", {}).get("tutor_source_prefixes")
    if prefixes != ["candidates"]:
        raise ValueError("pseudo labels must use candidate-only tutor rows")
    raw_views = load_views(args.views)
    samples, pseudo_counts = load_pseudo_identities(args.pseudo_labels, raw_views)
    development, holdout, split = identity_disjoint_split(
        samples,
        modulus=args.identity_modulus,
        holdout_bucket=args.holdout_bucket,
    )
    view_spaces, baseline_vectors, projection_metadata = build_view_spaces(
        raw_views,
        projection_path=args.projection,
        projection_sha256=args.projection_sha256,
    )
    if args.space not in view_spaces:
        raise ValueError(f"view space is unavailable: {args.space}")
    inputs = {
        "views": {"path": str(args.views), "sha256": sha256_file(args.views)},
        "pseudo_labels": {"path": str(args.pseudo_labels), "sha256": sha256_file(args.pseudo_labels)},
        "pseudo_manifest": {"path": str(args.pseudo_manifest), "sha256": sha256_file(args.pseudo_manifest)},
        "attributes": {"path": str(args.attributes), "sha256": sha256_file(args.attributes)},
        "ingest_tracklets": _ingest_input_manifest(Path(args.ingest_root)),
        "vocab": {"path": str(args.vocab), "sha256": sha256_file(args.vocab)},
        "projection": projection_metadata,
        "evaluator": _evaluator_manifest(),
    }
    return {
        "raw_views": raw_views,
        "views": view_spaces[args.space],
        "baseline_vectors": baseline_vectors,
        "development": development,
        "holdout": holdout,
        "split": split,
        "vocab": Vocabulary.from_json(args.vocab),
        "raw_labels": _load_raw_labels(Path(args.ingest_root)),
        "attributes": _load_attributes(Path(args.attributes)),
        "inputs": inputs,
        "pseudo_counts": pseudo_counts,
        "projection": projection_metadata,
        "evaluation_regime": (
            "transductive_fixed_projection"
            if projection_metadata is not None
            else "identity_disjoint_raw_no_learned_projection"
        ),
    }


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--views", required=True)
    parser.add_argument("--pseudo-labels", required=True)
    parser.add_argument("--pseudo-manifest", required=True)
    parser.add_argument("--attributes", required=True)
    parser.add_argument("--ingest-root", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--projection")
    parser.add_argument("--projection-sha256")
    parser.add_argument(
        "--allow-transductive-projection",
        action="store_true",
        help="explicitly allow a pre-trained projection that may contain holdout identities",
    )
    parser.add_argument("--space", default="raw")
    parser.add_argument("--visual-method", default="symmetric_top2")
    parser.add_argument("--visual-blend", type=float, default=0.5)
    parser.add_argument("--max-views", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--identity-modulus", type=int, default=3)
    parser.add_argument("--holdout-bucket", type=int, default=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    select = subparsers.add_parser("select")
    _common_parser(select)
    select.add_argument("--semantic-weights", default="0.00,0.05,0.10,0.15,0.20")
    select.add_argument("--out", required=True)
    holdout = subparsers.add_parser("holdout")
    _common_parser(holdout)
    holdout.add_argument("--selection", required=True)
    holdout.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite sealed proxy result: {output}")
    common = _load_common(args)
    parameters = {
        "visual_method": args.visual_method,
        "visual_blend": args.visual_blend,
        "space": args.space,
        "max_views": args.max_views,
        "top_k": args.top_k,
        "identity_modulus": args.identity_modulus,
        "holdout_bucket": args.holdout_bucket,
        "evaluation_regime": common["evaluation_regime"],
    }
    if args.phase == "select":
        signatures, coverage = build_signatures(
            sorted(common["development"]),
            vocab=common["vocab"],
            raw_labels=common["raw_labels"],
            attributes=common["attributes"],
        )
        weights = tuple(float(value) for value in args.semantic_weights.split(",") if value)
        development, grid = _partition_scores(
            common["development"],
            views=common["views"],
            baseline_vectors=common["baseline_vectors"],
            signatures=signatures,
            visual_method=args.visual_method,
            visual_blend=args.visual_blend,
            max_views=args.max_views,
            top_k=args.top_k,
            semantic_weights=weights,
        )
        winner = min(
            grid,
            key=lambda row: (
                -row["metrics"]["recall_at_1"],
                -row["metrics"]["mean_reciprocal_rank"],
                row["semantic_weight"],
            ),
        )
        report = {
            "schema_version": "1.0",
            "scope": "DEV_PROXY_SEMANTIC_SIGNATURE_SELECTION_NO_FROZEN_GT",
            "frozen_gt_read": False,
            "inputs": common["inputs"],
            "pseudo_counts": common["pseudo_counts"],
            "identity_split": common["split"],
            "signature_coverage": coverage,
            "projection": common["projection"],
            "parameters": parameters | {"semantic_weights": list(weights)},
            "development": development,
            "winner": winner,
            "grid": grid,
            "holdout_metrics": "SEALED_NOT_OPENED",
        }
    else:
        selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
        if selection.get("scope") != "DEV_PROXY_SEMANTIC_SIGNATURE_SELECTION_NO_FROZEN_GT":
            raise ValueError("selection artifact has the wrong scope")
        if selection.get("inputs") != common["inputs"]:
            raise ValueError("selection input hashes do not match holdout inputs")
        selected_parameters = {
            key: selection["parameters"][key] for key in parameters
        }
        if selected_parameters != parameters:
            raise ValueError("holdout runtime parameters differ from frozen selection")
        winner_weight = float(selection["winner"]["semantic_weight"])
        signatures, coverage = build_signatures(
            sorted(common["holdout"]),
            vocab=common["vocab"],
            raw_labels=common["raw_labels"],
            attributes=common["attributes"],
        )
        holdout_summary, grid = _partition_scores(
            common["holdout"],
            views=common["views"],
            baseline_vectors=common["baseline_vectors"],
            signatures=signatures,
            visual_method=args.visual_method,
            visual_blend=args.visual_blend,
            max_views=args.max_views,
            top_k=args.top_k,
            semantic_weights=(winner_weight,),
        )
        candidate = grid[0]["metrics"]
        baseline = holdout_summary["baseline"]
        report = {
            "schema_version": "1.0",
            "scope": (
                "SEALED_IDENTITY_DISJOINT_RAW_PROXY_HOLDOUT_NO_FROZEN_GT"
                if common["evaluation_regime"] == "identity_disjoint_raw_no_learned_projection"
                else "METRIC_SEALED_TRANSDUCTIVE_PROXY_HOLDOUT_NO_FROZEN_GT"
            ),
            "frozen_gt_read": False,
            "selection": {"path": str(args.selection), "sha256": sha256_file(args.selection)},
            "inputs": common["inputs"],
            "identity_split": common["split"],
            "signature_coverage": coverage,
            "parameters": parameters | {"semantic_weight": winner_weight},
            "baseline": baseline,
            "candidate": candidate,
            "delta": {
                "recall_at_1": round(candidate["recall_at_1"] - baseline["recall_at_1"], 6),
                "recall_at_5": round(candidate["recall_at_5"] - baseline["recall_at_5"], 6),
                "mean_reciprocal_rank": round(
                    candidate["mean_reciprocal_rank"] - baseline["mean_reciprocal_rank"], 6
                ),
            },
            "interpretation": "diagnostic pseudo-label proxy only; not hero R@1",
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"scope": report["scope"], "out": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
