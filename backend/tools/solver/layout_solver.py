"""S6 约束求解式自动布局 — OR-Tools CP-SAT。

硬约束(违反 = 不可行): 支撑类型兼容、容量、电源、同放、禁放、互斥。
软目标: 整数权重得分矩阵(角色相似/关系保留/可达性/证据质量在上游算好)。
状态映射(设计文档 §6.7): OPTIMAL/FEASIBLE→PLAN_READY, INFEASIBLE→NEW_SPACE_INCOMPATIBLE,
UNKNOWN/超时→PLANNER_ERROR。求解固定 seed + 单 worker,保证可复现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ortools.sat.python import cp_model


@dataclass(frozen=True)
class PlacementUnit:
    group_id: str
    size_units: int  # 粗容量占用(small=1, medium=2, large=3)
    requires_power: bool = False
    allowed_support: frozenset[str] = frozenset({"surface", "drawer", "shelf", "floor", "wall"})


@dataclass(frozen=True)
class CandidateRegion:
    region_id: str
    support_type: str
    capacity_units: int
    near_power: bool = False


@dataclass
class LayoutProblem:
    units: list[PlacementUnit]
    regions: list[CandidateRegion]
    # (group_id, region_id) -> 整数得分;缺省 0
    scores: dict[tuple[str, str], int] = field(default_factory=dict)
    co_locate: list[tuple[str, str]] = field(default_factory=list)
    mutex: list[tuple[str, str]] = field(default_factory=list)
    forbidden: list[tuple[str, str]] = field(default_factory=list)  # (group_id, region_id)


@dataclass
class LayoutResult:
    status: str  # PLAN_READY / NEW_SPACE_INCOMPATIBLE / PLANNER_ERROR
    assignments: dict[str, str] = field(default_factory=dict)  # group_id -> region_id
    alternatives: dict[str, Optional[str]] = field(default_factory=dict)
    objective: int = 0
    conflicts: list[str] = field(default_factory=list)


def _unary_feasible(u: PlacementUnit, r: CandidateRegion, problem: LayoutProblem) -> bool:
    if r.support_type not in u.allowed_support:
        return False
    if u.requires_power and not r.near_power:
        return False
    if (u.group_id, r.region_id) in problem.forbidden:
        return False
    if u.size_units > r.capacity_units:
        return False
    return True


def _explain_infeasible(problem: LayoutProblem) -> list[str]:
    """单元级冲突解释:哪个布局单元被哪类硬约束排除到零候选。"""
    conflicts: list[str] = []
    for u in problem.units:
        reasons: list[str] = []
        candidates = 0
        for r in problem.regions:
            if _unary_feasible(u, r, problem):
                candidates += 1
        if candidates == 0:
            for r in problem.regions:
                if r.support_type not in u.allowed_support:
                    reasons.append(f"{r.region_id}:支撑类型不兼容")
                elif u.requires_power and not r.near_power:
                    reasons.append(f"{r.region_id}:无电源证据")
                elif (u.group_id, r.region_id) in problem.forbidden:
                    reasons.append(f"{r.region_id}:用户禁放")
                elif u.size_units > r.capacity_units:
                    reasons.append(f"{r.region_id}:容量不足")
            conflicts.append(f"{u.group_id} 无可行区域({'; '.join(reasons)})")
    if not conflicts:
        conflicts.append("组合级冲突:容量/同放/互斥约束联合不可行")
    return conflicts


def _solve_once(
    problem: LayoutProblem,
    time_limit_s: float,
    extra_forbidden: Optional[set[tuple[str, str]]] = None,
) -> tuple[str, dict[str, str], int]:
    extra = extra_forbidden or set()
    model = cp_model.CpModel()
    x: dict[tuple[str, str], cp_model.IntVar] = {}
    for u in problem.units:
        for r in problem.regions:
            v = model.new_bool_var(f"x_{u.group_id}_{r.region_id}")
            x[(u.group_id, r.region_id)] = v
            if not _unary_feasible(u, r, problem) or (u.group_id, r.region_id) in extra:
                model.add(v == 0)
    for u in problem.units:
        model.add_exactly_one(x[(u.group_id, r.region_id)] for r in problem.regions)
    for r in problem.regions:
        model.add(
            sum(u.size_units * x[(u.group_id, r.region_id)] for u in problem.units)
            <= r.capacity_units
        )
    for g1, g2 in problem.co_locate:
        for r in problem.regions:
            model.add(x[(g1, r.region_id)] == x[(g2, r.region_id)])
    for g1, g2 in problem.mutex:
        for r in problem.regions:
            model.add(x[(g1, r.region_id)] + x[(g2, r.region_id)] <= 1)

    model.maximize(
        sum(
            problem.scores.get((u.group_id, r.region_id), 0) * x[(u.group_id, r.region_id)]
            for u in problem.units
            for r in problem.regions
        )
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.random_seed = 42
    solver.parameters.num_workers = 1
    status = solver.solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        assignments = {
            u.group_id: r.region_id
            for u in problem.units
            for r in problem.regions
            if solver.value(x[(u.group_id, r.region_id)]) == 1
        }
        return "OK", assignments, int(solver.objective_value)
    if status == cp_model.INFEASIBLE:
        return "INFEASIBLE", {}, 0
    return "UNKNOWN", {}, 0


def solve_layout(problem: LayoutProblem, time_limit_s: float = 2.0) -> LayoutResult:
    status, assignments, objective = _solve_once(problem, time_limit_s)
    if status == "INFEASIBLE":
        return LayoutResult(
            status="NEW_SPACE_INCOMPATIBLE", conflicts=_explain_infeasible(problem)
        )
    if status == "UNKNOWN":
        return LayoutResult(status="PLANNER_ERROR", conflicts=["求解器超时或未知状态"])

    # 替代区域:对每个单元禁掉首选后重解一次
    alternatives: dict[str, Optional[str]] = {}
    for u in problem.units:
        alt_status, alt_assign, _ = _solve_once(
            problem, time_limit_s, extra_forbidden={(u.group_id, assignments[u.group_id])}
        )
        alternatives[u.group_id] = alt_assign.get(u.group_id) if alt_status == "OK" else None

    return LayoutResult(
        status="PLAN_READY",
        assignments=assignments,
        alternatives=alternatives,
        objective=objective,
    )
