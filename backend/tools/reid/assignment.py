"""无额外依赖的确定性矩形匈牙利分配。"""

from __future__ import annotations

import math


def maximise_assignment(weights: list[list[float]], unmatched_weight: float) -> list[int | None]:
    """为每行返回唯一列或 ``None``。

    每行添加一个私有 dummy 列，允许不匹配；真实列全局一对一。实现采用
    O(n^3) Hungarian potentials，输入与 tie-break 顺序决定输出，重复运行稳定。
    """

    if not weights:
        return []
    real_columns = len(weights[0])
    if any(len(row) != real_columns for row in weights):
        raise ValueError("assignment matrix must be rectangular")

    rows = len(weights)
    columns = real_columns + rows
    maximum = max(
        [unmatched_weight]
        + [value for row in weights for value in row if math.isfinite(value)]
    )
    forbidden = maximum + 10.0
    costs: list[list[float]] = []
    for row_index, row in enumerate(weights):
        converted = [maximum - value if math.isfinite(value) else forbidden for value in row]
        for dummy_index in range(rows):
            # 私有 dummy 最便宜；其他 dummy 仍可用，避免退化矩阵无解。
            penalty = 0.0 if dummy_index == row_index else 1e-9 * (dummy_index + 1)
            converted.append(maximum - unmatched_weight + penalty)
        costs.append(converted)

    # 标准最小费用 Hungarian，rows <= columns（因为已加 rows 个 dummy）。
    u = [0.0] * (rows + 1)
    v = [0.0] * (columns + 1)
    p = [0] * (columns + 1)
    way = [0] * (columns + 1)
    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        minv = [math.inf] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0
            for j in range(1, columns + 1):
                if used[j]:
                    continue
                current = costs[i0 - 1][j - 1] - u[i0] - v[j]
                if current < minv[j] - 1e-15:
                    minv[j] = current
                    way[j] = j0
                if minv[j] < delta - 1e-15 or (
                    abs(minv[j] - delta) <= 1e-15 and (j1 == 0 or j < j1)
                ):
                    delta = minv[j]
                    j1 = j
            if not math.isfinite(delta):
                raise ValueError("assignment matrix has no finite solution")
            for j in range(columns + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment: list[int | None] = [None] * rows
    for column in range(1, columns + 1):
        row = p[column]
        if row and column <= real_columns:
            assignment[row - 1] = column - 1
    return assignment
