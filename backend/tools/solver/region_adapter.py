"""新家区域 manifest → 求解器输入的适配层。

new_1.mp4 本身不会自动变成 CandidateRegion:初赛口径下,候选区域由人工
对照新家视频逐段登记为 RegionManifest(每个区域必须带画面证据引用),
本模块负责校验并转换。若赛后接入自动区域检测,替换 manifest 生产端即可。
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.schemas.core import Region
from backend.schemas.hero_bundle import RegionManifest
from backend.tools.solver.layout_solver import CandidateRegion

# 区域容量粗估(单位与 PlacementUnit.size_units 同域):
# 一个组合(3~4 件)按 2 个单位计,small 区域恰好放一个组合。
DEFAULT_CAPACITY_UNITS: dict[str, int] = {"small": 2, "medium": 4, "large": 6}


def load_region_manifest(path: str | Path) -> RegionManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return RegionManifest.model_validate(data)


def to_candidate_regions(
    manifest: RegionManifest,
    capacity_units: dict[str, int] | None = None,
) -> list[CandidateRegion]:
    units = capacity_units or DEFAULT_CAPACITY_UNITS
    return [
        CandidateRegion(
            region_id=e.region_id,
            support_type=e.support_type.value,
            capacity_units=units[e.capacity_class.value],
            near_power=e.near_power,
        )
        for e in sorted(manifest.entries, key=lambda e: e.region_id)
    ]


def to_core_regions(manifest: RegionManifest) -> list[Region]:
    return [
        e.to_core_region()
        for e in sorted(manifest.entries, key=lambda e: e.region_id)
    ]
