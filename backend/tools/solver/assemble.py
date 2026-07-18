"""布局问题组装 — HeroGroup + RegionManifest → LayoutProblem。

得分来源同样遵循证据优先级:组的 target_region_hint(旁白"搬过去放床头")
命中区域锚点/展示名 → 高分;模板先验 → 低分。共现不进得分。
"""

from __future__ import annotations

from backend.schemas.hero_bundle import HeroGroup, RegionManifest
from backend.tools.solver.layout_solver import LayoutProblem, PlacementUnit
from backend.tools.solver.region_adapter import (
    DEFAULT_CAPACITY_UNITS,
    to_candidate_regions,
)

# 旁白去向关键词 → 区域锚点(Region.anchor 词域)
DEFAULT_ANCHOR_HINTS: dict[str, tuple[str, ...]] = {
    "床头": ("bed",),
    "床": ("bed",),
    # 同一生活语义兼容旧人工 manifest 与自动空间生产器的词域。
    "书桌": ("desk", "study_desk"),
    "梳妆台": ("vanity",),
    "墙上搁板": ("wall_shelf",),
    "置物架": ("shelf", "wall_shelf"),
    "展示柜": ("display_cabinet",),
    "斗柜": ("chest_of_drawers",),
    "桌": ("desk", "study_desk"),
    "柜": ("closet", "chest_of_drawers", "display_cabinet"),
    "架": ("shelf", "wall_shelf"),
    "角落": ("corner",),
}

NARRATION_HINT_SCORE = 10
DEFAULT_ALLOWED_SUPPORT = frozenset({"surface", "shelf", "drawer", "floor"})


def group_size_units(member_count: int) -> int:
    """组的粗容量占用:1~2 件=1,3~4 件=2,更多=3。"""
    if member_count <= 2:
        return 1
    if member_count <= 4:
        return 2
    return 3


def build_layout_problem(
    groups: list[HeroGroup],
    manifest: RegionManifest,
    *,
    requires_power_group_ids: frozenset[str] = frozenset(),
    allowed_support: frozenset[str] = DEFAULT_ALLOWED_SUPPORT,
    anchor_hints: dict[str, str | tuple[str, ...]] | None = None,
    capacity_units: dict[str, int] | None = None,
    template_scores: dict[tuple[str, str], int] | None = None,
) -> LayoutProblem:
    hints = anchor_hints or DEFAULT_ANCHOR_HINTS
    units = [
        PlacementUnit(
            group_id=g.group_id,
            size_units=group_size_units(len(g.entity_ids)),
            requires_power=g.group_id in requires_power_group_ids,
            allowed_support=allowed_support,
        )
        for g in sorted(groups, key=lambda g: g.group_id)
    ]
    regions = to_candidate_regions(manifest, capacity_units or DEFAULT_CAPACITY_UNITS)

    scores: dict[tuple[str, str], int] = dict(template_scores or {})
    entries = {e.region_id: e for e in manifest.entries}
    for g in sorted(groups, key=lambda g: g.group_id):
        if not g.target_region_hint:
            continue
        wanted_anchors = {
            anchor
            for keyword, configured in sorted(hints.items())
            if keyword in g.target_region_hint
            for anchor in (
                (configured,) if isinstance(configured, str) else configured
            )
        }
        for region_id in sorted(entries):
            e = entries[region_id]
            if e.anchor in wanted_anchors or any(
                keyword in e.display_name_zh
                for keyword in sorted(hints)
                if keyword in g.target_region_hint
            ):
                scores[(g.group_id, region_id)] = max(
                    scores.get((g.group_id, region_id), 0), NARRATION_HINT_SCORE
                )
    return LayoutProblem(units=units, regions=regions, scores=scores)
