from backend.schemas.hero_bundle import EvidenceSource, GroupConfirmation
from backend.tools.grouping.groups import build_groups
from backend.tools.grouping.narration import (
    match_name_to_entities,
    parse_narration_line,
    resolve_narration,
)

# 合成夹具:英雄房间式实体表(含一对同款不同色水壶)
ENTITY_DISPLAY = {
    "e1": {"display_name_zh": "水壶", "color_primary": "blue"},
    "e2": {"display_name_zh": "水壶", "color_primary": "pink"},
    "e3": {"display_name_zh": "台灯", "color_primary": "white"},
    "e4": {"display_name_zh": "绘本", "color_primary": "green"},
    "e5": {"display_name_zh": "耳机", "color_primary": "black"},
    "e6": {"display_name_zh": "收纳盒", "color_primary": "gray"},
}


def test_parse_canonical_narration_line():
    item = parse_narration_line(
        "n1", "蓝色水壶,妹妹的,现在在书桌上,搬过去放床头,和台灯一组。"
    )
    assert item.label_zh == "蓝色水壶"
    assert item.owner == "妹妹"
    assert item.source_location == "书桌上"
    assert item.target_location == "床头"
    assert item.group_partners == ["台灯"]
    assert item.color_words == ["blue"]


def test_parse_multiple_partners():
    item = parse_narration_line("n2", "台灯,放到床头柜,和蓝色水壶、绘本一组")
    assert item.group_partners == ["蓝色水壶", "绘本"]
    assert item.target_location == "床头柜"


def test_parse_real_a1_transcript_line():
    # 真实 A1 誊写句式:全角逗号+"这是一个X"+"是谁的"(s1 首跑实锤回归)
    item = parse_narration_line(
        "n01", "这是一个水壶，是孩子的，现在在客厅，搬到新家想放在客厅，和杯子打包在一起。"
    )
    assert item.label_zh == "水壶"
    assert item.owner == "孩子"
    assert item.source_location == "客厅"
    assert item.target_location == "客厅"
    assert item.group_partners == ["杯子"]


def test_parse_target_and_partner_in_same_segment():
    item = parse_narration_line(
        "n02", "这是一个防晒霜，是孩子的，现在在客厅搬到新家，想放在洗手间和梳子打包在一起。"
    )
    assert item.source_location == "客厅"
    assert item.target_location == "洗手间"
    assert item.group_partners == ["梳子"]


def test_parse_classifier_color_and_set_square():
    item = parse_narration_line(
        "n11", "这是一个红色圆珠笔，是爸爸的。现在在客厅，搬到新家，想放在客厅学习区，和文具打包在一起。"
    )
    assert item.label_zh == "红色圆珠笔"
    assert item.owner == "爸爸"
    assert item.target_location == "客厅学习区"
    assert item.group_partners == ["文具"]
    assert item.color_words == ["red"]
    item = parse_narration_line(
        "n16", "这是一个三角尺，是孩子的。现在在客厅，搬到新家，想放在客厅学习区，和文具打包在一起。"
    )
    assert item.label_zh == "三角尺"  # 数词剥离不得误伤"三角尺"
    item = parse_narration_line(
        "n22", "这是一袋跳跳糖，是孩子的，现在在客厅，搬到新家，想放在客厅和零食打包在一起。"
    )
    assert item.label_zh == "跳跳糖"
    assert item.target_location == "客厅"
    assert item.group_partners == ["零食"]


def test_color_disambiguates_lookalike_pair():
    candidates, method = match_name_to_entities("蓝色水壶", [], ENTITY_DISPLAY)
    assert (candidates, method) == (["e1"], "name_color")
    candidates, method = match_name_to_entities("水壶", [], ENTITY_DISPLAY)
    assert method == "unresolved_ambiguous"
    assert candidates == ["e1", "e2"]


def test_resolve_narration_unresolved_goes_to_none():
    items = [parse_narration_line("n1", "望远镜,放柜子上")]
    (res,) = resolve_narration(items, ENTITY_DISPLAY)
    assert res.entity_id is None
    assert res.method == "unresolved_no_match"


def _hero_items():
    return [
        parse_narration_line(
            "n1", "蓝色水壶,妹妹的,现在在书桌上,搬过去放床头,和台灯一组。"
        ),
        parse_narration_line("n2", "台灯,现在在床头柜,搬过去放床头,和蓝色水壶一组"),
        parse_narration_line("n3", "粉色水壶,搬过去放柜子,和收纳盒一组"),
    ]


def _build(**kw):
    items = _hero_items()
    resolutions = resolve_narration(items, ENTITY_DISPLAY)
    return build_groups(
        ENTITY_DISPLAY,
        items,
        resolutions,
        created_at="2026-07-16T12:00:00+00:00",
        **kw,
    )


def test_narration_edges_form_groups_with_names():
    build = _build()
    by_name = {g.name_zh: g for g in build.groups}
    assert by_name["睡前组合"].entity_ids == ["e1", "e3"]
    assert by_name["收纳组合"].entity_ids == ["e2", "e6"]
    assert by_name["睡前组合"].dominant_source == EvidenceSource.NARRATION
    assert by_name["睡前组合"].target_region_hint == "床头"


def test_confirmation_fills_gap_but_narration_wins_conflict():
    build = _build(
        confirmations=[
            GroupConfirmation(entity_id="e5", group_name_zh="学习组合"),
            GroupConfirmation(entity_id="e1", group_name_zh="学习组合"),  # 与旁白冲突
        ]
    )
    by_name = {g.name_zh: g for g in build.groups}
    assert by_name["学习组合"].entity_ids == ["e5"]
    assert by_name["学习组合"].dominant_source == EvidenceSource.CONFIRMATION
    # 旁白胜出:e1 仍在睡前组合,冲突被记录
    assert "e1" in by_name["睡前组合"].entity_ids
    assert any("e1" in c for c in build.conflicts)


def test_template_assigns_leftover_by_display_name():
    build = _build(template_rules={"绘本": "学习组合", "书": "学习组合"})
    by_name = {g.name_zh: g for g in build.groups}
    assert "e4" in by_name["学习组合"].entity_ids
    evidence = [
        e for e in by_name["学习组合"].member_evidence if e.entity_id == "e4"
    ]
    assert evidence[0].source == EvidenceSource.TEMPLATE


def test_cooccurrence_only_corroborates_never_assigns():
    build = _build(cooccurrence={("e1", "e3"): 5, ("e4", "e5"): 9})
    by_name = {g.name_zh: g for g in build.groups}
    corroborations = [
        e
        for e in by_name["睡前组合"].member_evidence
        if e.source == EvidenceSource.COOCCURRENCE
    ]
    assert corroborations and "共现 5 次" in corroborations[0].detail
    # e4/e5 共现再多也不成组:仍进轻确认队列
    assert {"e4", "e5"} <= set(build.unassigned_entity_ids)
    reasons = {c.entity_id: c.reason for c in build.clarifications}
    assert reasons["e4"] == "unassigned_entity"


def test_unresolved_narration_becomes_clarification():
    items = _hero_items() + [parse_narration_line("n9", "望远镜,放柜子上")]
    resolutions = resolve_narration(items, ENTITY_DISPLAY)
    build = build_groups(
        ENTITY_DISPLAY, items, resolutions, created_at="2026-07-16T12:00:00+00:00"
    )
    assert any(c.reason == "unresolved_narration" for c in build.clarifications)


def test_build_is_deterministic():
    kw = dict(
        confirmations=[GroupConfirmation(entity_id="e5", group_name_zh="学习组合")],
        template_rules={"绘本": "学习组合"},
        cooccurrence={("e1", "e3"): 5},
    )
    a, b = _build(**kw), _build(**kw)
    assert [g.model_dump() for g in a.groups] == [g.model_dump() for g in b.groups]
    assert [c.model_dump() for c in a.clarifications] == [
        c.model_dump() for c in b.clarifications
    ]
    assert [e.model_dump() for e in a.audit_events] == [
        e.model_dump() for e in b.audit_events
    ]
