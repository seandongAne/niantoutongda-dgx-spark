#!/usr/bin/env python
"""S6 布局阶段 — HeroGroup + 区域 manifest → CP-SAT → PlacementPlan。

PLAN_READY 之外的状态照实写盘并以退出码 3 让主链停下:
NEW_SPACE_INCOMPATIBLE 是需要人看的结论,不是可以静默跳过的小事。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.schemas.core import (  # noqa: E402
    AgentHandoff,
    AgentRole,
    Assignment,
    PlacementPlan,
)
from backend.schemas.hero_bundle import HeroGroup, RegionManifest  # noqa: E402
from backend.tools.solver.assemble import build_layout_problem  # noqa: E402
from backend.tools.solver.layout_solver import solve_layout  # noqa: E402
from backend.tools.trace import (  # noqa: E402
    finalize_message,
    require_handoff,
    write_fragment,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--groups", required=True, type=Path)
    ap.add_argument("--regions", required=True, type=Path)
    ap.add_argument("--requires-power-groups", default="")
    ap.add_argument("--plan-id", default="plan-hero-v1")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--trace-id")
    ap.add_argument("--trace-parent", type=Path)
    ap.add_argument("--trace-out", type=Path)
    args = ap.parse_args()
    trace_args = (args.trace_id, args.trace_parent, args.trace_out)
    if any(trace_args) and not all(trace_args):
        ap.error("--trace-id, --trace-parent and --trace-out must be provided together")

    groups = []
    with args.groups.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                groups.append(HeroGroup.model_validate(json.loads(line)))
    manifest = RegionManifest.model_validate(
        json.loads(args.regions.read_text(encoding="utf-8"))
    )
    requires_power = frozenset(
        g for g in args.requires_power_groups.split(",") if g
    )

    problem = build_layout_problem(
        groups, manifest, requires_power_group_ids=requires_power
    )
    result = solve_layout(problem)

    plan = PlacementPlan(
        plan_id=args.plan_id,
        assignments=[
            Assignment(
                group_id=gid,
                region_id=region_id,
                score_breakdown={
                    "narration_hint": problem.scores.get((gid, region_id), 0)
                },
                alternative_region_id=result.alternatives.get(gid),
            )
            for gid, region_id in sorted(result.assignments.items())
        ],
        hard_constraints=[
            "支撑类型兼容",
            "区域容量",
            "电源证据",
            "同放/互斥/禁放",
        ],
        soft_scores={"objective": result.objective},
        solver_status=result.status,
        conflicts=result.conflicts,
    )

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(
        plan.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (out / "layout.json").write_text(
        json.dumps(
            {
                "status": result.status,
                "assignments": result.assignments,
                "alternatives": result.alternatives,
                "objective": result.objective,
                "conflicts": result.conflicts,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": result.status, "objective": result.objective,
             "assignments": len(result.assignments)},
            ensure_ascii=False,
        )
    )
    if result.status != "PLAN_READY":
        print(f"布局未就绪: {result.conflicts}", file=sys.stderr)
        return 3
    if args.trace_out:
        parent = require_handoff(args.trace_parent, "GROUPS_READY")
        message = finalize_message(
            AgentHandoff(
                message_id=f"{args.trace_id}-placement-ready",
                correlation_id=parent.correlation_id,
                causation_id=parent.message_id,
                producer=AgentRole.SPACE,
                target=AgentRole.EXEC,
                action="PLACEMENT_READY",
                item_ids=[assignment.group_id for assignment in plan.assignments],
                artifact_refs=[str(out / "plan.json"), str(out / "layout.json")],
                summary={
                    "solver_status": result.status,
                    "assignments": len(result.assignments),
                    "objective": result.objective,
                },
            )
        )
        write_fragment(args.trace_out, [message])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
