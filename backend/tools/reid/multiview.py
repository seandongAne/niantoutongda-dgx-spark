"""候选集内的逐视角 ReID 重排。

全库召回仍使用已冻结的轨迹均值向量；本模块只消费双向 Top-K
候选并集，因此不会在重排时偷偷扩大召回集。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


Vector = tuple[float, ...]
PairKey = tuple[str, str]
ARTIFACT_FORMAT_VERSION = "reid-multiview-embeddings-v1"


def _cosine(a: Vector, b: Vector) -> float:
    if not a or len(a) != len(b):
        raise ValueError("multiview vectors must have the same positive dimension")
    value = sum(x * y for x, y in zip(a, b))
    if not math.isfinite(value):
        raise ValueError("multiview cosine is non-finite")
    return max(0.0, min(1.0, value))


def _matrix(a: Sequence[Vector], b: Sequence[Vector]) -> list[list[float]]:
    if not a or not b:
        raise ValueError("multiview score requires non-empty view sets")
    return [[_cosine(left, right) for right in b] for left in a]


def _mean_chamfer(matrix: list[list[float]]) -> float:
    left = sum(max(row) for row in matrix) / len(matrix)
    columns = zip(*matrix)
    right = sum(max(column) for column in columns) / len(matrix[0])
    return 0.5 * (left + right)


def _greedy_top2(matrix: list[list[float]]) -> float:
    """确定性的至多两条不共用端点视角边。"""

    candidates = sorted(
        (
            (score, row_index, column_index)
            for row_index, row in enumerate(matrix)
            for column_index, score in enumerate(row)
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    used_rows: set[int] = set()
    used_columns: set[int] = set()
    selected: list[float] = []
    for score, row_index, column_index in candidates:
        if row_index in used_rows or column_index in used_columns:
            continue
        used_rows.add(row_index)
        used_columns.add(column_index)
        selected.append(score)
        if len(selected) == 2:
            break
    return sum(selected) / len(selected)


def set_similarity(
    a: Sequence[Vector],
    b: Sequence[Vector],
    *,
    method: str,
    max_views: int,
) -> float:
    """计算对称、有界的逐视角相似度。"""

    if max_views <= 0:
        raise ValueError("max_views must be positive")
    matrix = _matrix(tuple(a[:max_views]), tuple(b[:max_views]))
    if method == "max_pair":
        value = max(max(row) for row in matrix)
    elif method == "mean_chamfer":
        value = _mean_chamfer(matrix)
    elif method == "symmetric_top2":
        value = 0.5 * _mean_chamfer(matrix) + 0.5 * _greedy_top2(matrix)
    else:
        raise ValueError(f"unsupported multiview method: {method}")
    return max(0.0, min(1.0, value))


def quantile_calibrate(
    baseline: Mapping[PairKey, float],
    multiview: Mapping[PairKey, float],
) -> dict[PairKey, float]:
    """把多视角排序映射到 baseline 分数的原经验分布。

    这使现有 match/new 阈值仍有可比语义；排序改变，分数 multiset
    不变。多视角同分时保留 baseline 相对顺序，避免无新增证据时按 pair id
    任意洗牌；最后才以 pair id 确定性解决双重同分。
    """

    if set(baseline) != set(multiview):
        raise ValueError("baseline and multiview pair sets differ")
    ordered_pairs = sorted(
        multiview,
        key=lambda key: (-multiview[key], -baseline[key], key),
    )
    ordered_baseline = sorted(baseline.values(), reverse=True)
    return {key: ordered_baseline[index] for index, key in enumerate(ordered_pairs)}
