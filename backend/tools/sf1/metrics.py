"""冻结的跨视频检索指标：验证视频 query 对训练视频 gallery。"""

from __future__ import annotations

import math

import numpy as np

from backend.tools.sf1.dataset import SF1Sample


def _matrix(samples: tuple[SF1Sample, ...]) -> np.ndarray:
    return np.stack([sample.vector for sample in samples]).astype(np.float32)


def retrieval_metrics(
    train: tuple[SF1Sample, ...],
    validation: tuple[SF1Sample, ...],
    *,
    projected_train: np.ndarray | None = None,
    projected_validation: np.ndarray | None = None,
) -> dict[str, float | int]:
    if not train or not validation:
        raise ValueError("retrieval metrics require non-empty train and validation")
    gallery = projected_train if projected_train is not None else _matrix(train)
    queries = projected_validation if projected_validation is not None else _matrix(validation)
    gallery = np.asarray(gallery, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    if gallery.shape[0] != len(train) or queries.shape[0] != len(validation):
        raise ValueError("projected vector count does not match samples")
    gallery = gallery / np.clip(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-12, None)
    queries = queries / np.clip(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12, None)
    similarities = queries @ gallery.T
    gallery_labels = np.asarray([item.identity_id for item in train])
    query_labels = np.asarray([item.identity_id for item in validation])

    hits1 = hits5 = 0
    reciprocal_ranks: list[float] = []
    positive_scores: list[float] = []
    hardest_negatives: list[float] = []
    margins: list[float] = []
    for index, label in enumerate(query_labels):
        order = np.argsort(-similarities[index], kind="stable")
        ranked_labels = gallery_labels[order]
        matching_ranks = np.flatnonzero(ranked_labels == label)
        if matching_ranks.size == 0:
            raise ValueError(f"validation identity {label} absent from gallery")
        rank = int(matching_ranks[0]) + 1
        hits1 += int(rank == 1)
        hits5 += int(rank <= 5)
        reciprocal_ranks.append(1.0 / rank)
        positive = similarities[index][gallery_labels == label]
        negative = similarities[index][gallery_labels != label]
        positive_scores.extend(float(value) for value in positive)
        hardest = float(np.max(negative))
        hardest_negatives.append(hardest)
        margins.append(float(np.max(positive)) - hardest)

    def rounded(value: float) -> float:
        return round(float(value), 6)

    return {
        "queries": len(validation),
        "gallery": len(train),
        "identities": len(set(query_labels)),
        "recall_at_1": rounded(hits1 / len(validation)),
        "recall_at_5": rounded(hits5 / len(validation)),
        "mean_reciprocal_rank": rounded(np.mean(reciprocal_ranks)),
        "positive_cosine_mean": rounded(np.mean(positive_scores)),
        "hardest_negative_cosine_mean": rounded(np.mean(hardest_negatives)),
        "hardest_negative_cosine_p95": rounded(
            np.percentile(hardest_negatives, 95)
        ),
        "top1_margin_mean": rounded(np.mean(margins)),
        "top1_margin_min": rounded(np.min(margins)),
        "finite": bool(
            all(math.isfinite(value) for value in positive_scores + hardest_negatives + margins)
        ),
    }
