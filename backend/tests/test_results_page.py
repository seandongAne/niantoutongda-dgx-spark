import json
from pathlib import Path

from scripts.results_page import build_page


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _write_completion_summaries(run_dir: Path) -> None:
    _write_jsonl(
        run_dir / "inventory/display.jsonl",
        [
            {
                "entity_id": "trusted-entity",
                "display_name_zh": "可信展示名",
                "hero_crop_ref": "",
            }
        ],
    )
    _write_jsonl(
        run_dir / "naming/display.jsonl",
        [
            {
                "entity_id": "legacy-entity",
                "display_name_zh": "不应出现的旧展示名",
                "hero_crop_ref": "",
            }
        ],
    )
    _write_json(
        run_dir / "inventory/metrics.json",
        {
            "raw_entity_count": 3306,
            "trusted_inventory_count": 20,
            "downstream_eligible_count": 20,
            "raw_link_unresolved_count": 9,
            "clarification_cap": 4,
            "giant_audit": "DO_NOT_RENDER_INVENTORY_AUDIT" * 100,
        },
    )
    _write_jsonl(
        run_dir / "inventory/clarifications.jsonl",
        [
            {
                "clarification_id": f"q{index}",
                "projected_entity_id": f"trusted-{index}",
                "question_zh": "<script>question</script>",
                "status": "PARTIAL",
            }
            for index in range(4)
        ],
    )
    _write_jsonl(
        run_dir / "group/clarifications.jsonl",
        [
            {
                "entity_id": "legacy-entity",
                "question_zh": "不应出现的旧澄清",
                "reason": "legacy",
            }
        ],
    )
    _write_json(
        run_dir / "group/boxlist.json",
        {
            "canonical_item_count": 20,
            "box_count": 5,
            "boxes": [
                {
                    "box_id": f"box-{index}",
                    "box_type": "life_group" if index <= 3 else "technical_pack_unit",
                    "box_label_zh": (
                        "<img src=x onerror=alert(1)>" if index == 1 else f"箱 {index}"
                    ),
                    "items": [{"canonical_id": f"item-{index}-{item}"} for item in range(4)],
                }
                for index in range(1, 6)
            ],
            "giant_audit": "DO_NOT_RENDER_BOXLIST_AUDIT" * 100,
        },
    )
    _write_jsonl(
        run_dir / "group/placement_groups.jsonl",
        [
            {
                "group_id": "trusted-placement",
                "name_zh": "可信 placement 组合",
                "entity_ids": ["trusted-entity"],
                "dominant_source": "template",
                "member_evidence": [],
                "target_region_hint": "",
            }
        ],
    )
    _write_jsonl(
        run_dir / "group/groups.jsonl",
        [
            {
                "group_id": "legacy-group",
                "name_zh": "不应出现的旧组合",
                "entity_ids": ["legacy-entity"],
                "dominant_source": "template",
                "member_evidence": [],
                "target_region_hint": "",
            }
        ],
    )
    _write_json(
        run_dir / "group/metrics.json",
        {
            "group_count": 3,
            "placement_group_count": 5,
            "covered_canonical_item_count": 20,
            "trusted_inventory_count": 20,
            "box_count": 5,
        },
    )
    _write_json(
        run_dir / "spatial/metrics.json",
        {
            "gate_status": "PASS",
            "candidate_count": 7,
            "auto_accepted_count": 5,
            "projected_region_count": 5,
            "needs_user_count": 2,
            "not_observed_count": 0,
            "gate_reasons": [],
            "giant_audit": "DO_NOT_RENDER_SPATIAL_AUDIT" * 100,
        },
    )
    _write_json(
        run_dir / "risk/assessments.json",
        {
            "assessments": [
                {
                    "rule_id": "CHILD_SHARP_TOOL_REACH",
                    "status": "TRIGGERED",
                    "confidence": 0.91,
                    "reason_codes": ["RULE_TRIGGERED:CHILD_SHARP_TOOL_REACH"],
                    "evidence": {"large": "DO_NOT_RENDER_RISK_EVIDENCE" * 100},
                },
                {
                    "rule_id": "TRIP_HAZARD_IN_PATH",
                    "status": "NOT_APPLICABLE",
                    "confidence": 0.92,
                    "reason_codes": ["NEGATED_TRIGGER_FACT:trip_hazard_present"],
                },
                {
                    "rule_id": "POWER_IN_WET_ZONE",
                    "status": "NEEDS_USER",
                    "confidence": 0.0,
                    "reason_codes": ["MISSING_EVIDENCE:powered_item_present"],
                },
            ]
        },
    )
    _write_json(
        run_dir / "risk/metrics.json",
        {
            "rule_count": 3,
            "status_counts": {
                "TRIGGERED": 1,
                "NEEDS_USER": 1,
                "NOT_APPLICABLE": 1,
            },
            "disclaimer_zh": (
                "仅为辅助风险提醒，不构成安全认证。"
                "<script>alert('disclaimer')</script>"
            ),
        },
    )


def test_results_page_renders_completion_summaries_and_escapes_all_text(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_completion_summaries(run_dir)

    page = build_page(run_dir)

    assert '<h2 id="trusted-inventory">可信库存' in page
    assert '<h2 id="boxlist">箱单' in page
    assert '<h2 id="automatic-space">自动空间' in page
    assert '<h2 id="risk-reminders">风险提醒' in page
    assert "3306" in page and "→" in page and "20" in page
    assert "3306 → 20" in page
    assert "上限 4" in page
    assert "placement 单元" in page and "物品覆盖" in page and "20/20" in page
    assert "候选区域" in page and "已投影" in page and "PASS" in page
    assert "儿童可触及锐器" in page
    assert "通道绊倒风险" in page
    assert "潮湿区域用电" in page
    assert "已触发 1" in page and "待人工确认 1" in page
    assert "当前条件不成立 1" in page
    assert "不构成安全认证" in page
    assert "可信展示名" in page and "可信 placement 组合" in page
    assert '<h2 id="entities">可信库存实体' in page
    assert (
        '<div class="stat-n">1</div><div class="stat-l">placement 单元</div>'
        in page
    )
    assert '<div class="stat-n">1</div><div class="stat-l">生活组合</div>' not in page
    assert "raw ReID 仅保留为审计证据" in page
    assert "展示名 = 本地 VLM" not in page
    assert "不应出现的旧展示名" not in page
    assert "不应出现的旧组合" not in page
    assert "不应出现的旧澄清" not in page

    assert "<script>" not in page
    assert "<img src=x onerror=alert(1)>" not in page
    assert "&lt;img src=x onerror=alert(1)&gt;" in page
    assert "&lt;script&gt;alert(&#x27;disclaimer&#x27;)&lt;/script&gt;" in page
    for marker in (
        "DO_NOT_RENDER_INVENTORY_AUDIT",
        "DO_NOT_RENDER_BOXLIST_AUDIT",
        "DO_NOT_RENDER_SPATIAL_AUDIT",
        "DO_NOT_RENDER_RISK_EVIDENCE",
    ):
        assert marker not in page


def test_results_page_without_new_artifacts_keeps_legacy_sections(tmp_path):
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "naming/display.jsonl",
        [
            {
                "entity_id": "legacy-entity",
                "display_name_zh": "旧链展示名",
                "hero_crop_ref": "",
            }
        ],
    )
    _write_jsonl(
        run_dir / "group/groups.jsonl",
        [
            {
                "group_id": "legacy-group",
                "name_zh": "旧链生活组",
                "entity_ids": ["legacy-entity"],
                "dominant_source": "template",
                "member_evidence": [],
                "target_region_hint": "",
            }
        ],
    )
    page = build_page(run_dir)

    assert "房间成果总览" in page
    assert '<h2 id="entities">实体卡' in page
    assert '<h2 id="groups">生活组合' in page
    assert '<h2 id="layout">新家布局' in page
    assert '<h2 id="cards">任务卡' in page
    assert "旧链展示名" in page and "旧链生活组" in page
    assert "展示名 = 本地 VLM" in page
    assert 'id="trusted-inventory"' not in page
    assert 'id="boxlist"' not in page
    assert 'id="automatic-space"' not in page
    assert 'id="risk-reminders"' not in page


def test_incomplete_optional_artifact_pairs_do_not_render_partial_blocks(tmp_path):
    run_dir = tmp_path / "partial"
    _write_json(run_dir / "inventory/metrics.json", {"raw_entity_count": 3306})
    _write_json(run_dir / "group/boxlist.json", {"boxes": []})
    _write_json(run_dir / "risk/assessments.json", {"assessments": []})

    page = build_page(run_dir)

    assert 'id="trusted-inventory"' not in page
    assert 'id="boxlist"' not in page
    assert 'id="risk-reminders"' not in page
