import json

import pytest

from backend.schemas.hero_bundle import (
    EvidenceSource,
    GroupEvidence,
    HeroGroup,
    RegionManifest,
)
from backend.tools.solver.assemble import build_layout_problem, group_size_units
from backend.tools.solver.layout_solver import LayoutResult, solve_layout
from backend.tools.solver.region_adapter import (
    load_region_manifest,
    to_candidate_regions,
    to_core_regions,
)
from backend.tools.taskcards import build_task_cards, task_card_markdown

MANIFEST = {
    "video_id": "new_1",
    "entries": [
        {
            "region_id": "bedside",
            "anchor": "bed",
            "display_name_zh": "床头柜面",
            "support_type": "surface",
            "capacity_class": "small",
            "near_power": True,
            "evidence_refs": ["new_1.mp4@00:12"],
        },
        {
            "region_id": "desk_top",
            "anchor": "desk",
            "display_name_zh": "书桌面",
            "support_type": "surface",
            "capacity_class": "medium",
            "near_power": True,
            "evidence_refs": ["new_1.mp4@00:41"],
        },
        {
            "region_id": "closet_shelf",
            "anchor": "closet",
            "display_name_zh": "柜子隔层",
            "support_type": "shelf",
            "capacity_class": "medium",
            "evidence_refs": ["new_1.mp4@01:05"],
        },
    ],
}


def _groups():
    def g(gid, name, members, hint):
        return HeroGroup(
            group_id=gid,
            name_zh=name,
            entity_ids=members,
            dominant_source=EvidenceSource.NARRATION,
            member_evidence=[
                GroupEvidence(entity_id=m, source=EvidenceSource.NARRATION, detail="旁白")
                for m in members
            ],
            target_region_hint=hint,
        )

    return [
        g("g01", "睡前组合", ["e1", "e3"], "床头"),
        g("g02", "学习组合", ["e4", "e5", "e7"], "书桌"),
        g("g03", "收纳组合", ["e2", "e6"], "柜子"),
    ]


DISPLAY = {
    "e1": {"display_name_zh": "水壶", "hero_crop_ref": "crops/e1.jpg"},
    "e3": {"display_name_zh": "台灯", "hero_crop_ref": "crops/e3.jpg"},
}


def test_manifest_roundtrip_and_adapters(tmp_path):
    path = tmp_path / "regions.json"
    path.write_text(json.dumps(MANIFEST, ensure_ascii=False), encoding="utf-8")
    manifest = load_region_manifest(path)
    regions = to_candidate_regions(manifest)
    by_id = {r.region_id: r for r in regions}
    assert by_id["bedside"].capacity_units == 2
    assert by_id["bedside"].near_power is True
    assert by_id["closet_shelf"].support_type == "shelf"
    core = to_core_regions(manifest)
    assert [r.region_id for r in core] == ["bedside", "closet_shelf", "desk_top"]


def test_group_size_units_mapping():
    assert group_size_units(2) == 1
    assert group_size_units(4) == 2
    assert group_size_units(5) == 3


def test_assemble_and_solve_hero_fixture():
    manifest = RegionManifest.model_validate(MANIFEST)
    problem = build_layout_problem(_groups(), manifest)
    # 旁白去向提示进得分:睡前组合→床头区域
    assert problem.scores[("g01", "bedside")] == 10
    assert problem.scores[("g02", "desk_top")] == 10
    result = solve_layout(problem)
    assert result.status == "PLAN_READY"
    assert result.assignments["g01"] == "bedside"
    assert result.assignments["g02"] == "desk_top"
    assert result.assignments["g03"] == "closet_shelf"


def test_auto_spatial_anchor_aliases_score_all_five_frozen_targets():
    manifest = RegionManifest.model_validate(
        {
            "video_id": "new_1",
            "entries": [
                {
                    "region_id": f"auto_{anchor}",
                    "anchor": anchor,
                    "display_name_zh": name,
                    "support_type": support,
                    "capacity_class": "large",
                    "evidence_refs": [f"new_1@{index}ms"],
                }
                for index, (anchor, name, support) in enumerate(
                    [
                        ("study_desk", "学习桌面", "surface"),
                        ("display_cabinet", "展示柜层板", "shelf"),
                        ("vanity", "梳妆台面", "surface"),
                        ("wall_shelf", "墙面置物架", "shelf"),
                        ("chest_of_drawers", "斗柜台面", "surface"),
                    ]
                )
            ],
        }
    )
    groups = [
        HeroGroup(
            group_id=f"auto-g{index}",
            name_zh=f"自动组合{index}",
            entity_ids=[f"hero-{index}"],
            dominant_source=EvidenceSource.CONFIRMATION,
            member_evidence=[
                GroupEvidence(
                    entity_id=f"hero-{index}",
                    source=EvidenceSource.CONFIRMATION,
                    detail="technical closure",
                )
            ],
            target_region_hint=hint,
        )
        for index, hint in enumerate(
            ["书桌", "展示柜", "梳妆台", "墙上搁板", "斗柜"], start=1
        )
    ]

    problem = build_layout_problem(groups, manifest)

    expected = [
        "auto_study_desk",
        "auto_display_cabinet",
        "auto_vanity",
        "auto_wall_shelf",
        "auto_chest_of_drawers",
    ]
    assert [problem.scores[(group.group_id, region_id)] for group, region_id in zip(groups, expected, strict=True)] == [10] * 5


def test_task_cards_from_layout():
    manifest = RegionManifest.model_validate(MANIFEST)
    groups = _groups()
    result = solve_layout(build_layout_problem(groups, manifest))
    cards = build_task_cards(groups, result, manifest, DISPLAY)
    assert [c.group_id for c in cards] == ["g01", "g02", "g03"]
    card = cards[0]
    assert card.box_label_zh == "睡前组合箱"
    assert card.target_region_name_zh == "床头柜面"
    assert card.items[0].display_name_zh == "水壶"
    assert card.items[0].hero_crop_ref == "crops/e1.jpg"
    assert any("台灯 出现在「床头柜面」" in c for c in card.verification_checklist)
    md = task_card_markdown(card)
    assert "睡前组合箱" in md and "- [ ]" in md


def test_task_cards_refuse_unready_layout():
    manifest = RegionManifest.model_validate(MANIFEST)
    with pytest.raises(ValueError, match="不生成任务卡"):
        build_task_cards(
            _groups(),
            LayoutResult(status="NEW_SPACE_INCOMPATIBLE"),
            manifest,
            DISPLAY,
        )
