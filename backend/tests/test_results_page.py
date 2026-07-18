import hashlib
import json
from pathlib import Path

from scripts.results_page import build_page, select_demo_space_frames


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
        run_dir / "spatial_score/metrics.json",
        {
            "acceptance_passed": True,
            "score": "5/5",
            "matched_anchor_count": 5,
            "exact_semantic_match_count": 5,
            "extra_prediction_count": 0,
            "support_type_mismatch_count": 0,
            "capacity_class_mismatch_count": 0,
            "informational_power_mismatch_count": 2,
            "gate_reasons": [],
            "giant_audit": "DO_NOT_RENDER_SPATIAL_SCORE_AUDIT" * 100,
        },
    )
    _write_json(
        run_dir / "spatial_review/metrics.json",
        {
            "gate_status": "PASS",
            "decision_count": 5,
            "visually_adjudicated_count": 5,
            "projected_region_count": 5,
            "needs_user_count": 0,
            "power_state_counts": {"NEAR": 2, "UNKNOWN": 3},
            "gate_reasons": [],
            "giant_audit": "DO_NOT_RENDER_SPATIAL_REVIEW_AUDIT" * 100,
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


def _write_deferred_risk_scope(run_dir: Path, defer_reason: str) -> dict:
    rule_reasons = {
        "CHILD_SHARP_TOOL_REACH": "MISSING_EVIDENCE:child_present",
        "TRIP_HAZARD_IN_PATH": "MISSING_EVIDENCE:trip_hazard_present",
        "POWER_IN_WET_ZONE": "MISSING_EVIDENCE:powered_item_present",
    }
    assessments = {
        "scope_status": "DEFERRED",
        "blocking": False,
        "defer_reason_zh": defer_reason,
        "assessments": [
            {
                "rule_id": rule_id,
                "status": "NEEDS_USER",
                "confidence": 0.0,
                "reason_codes": [reason],
            }
            for rule_id, reason in rule_reasons.items()
        ],
    }
    _write_json(run_dir / "risk/assessments.json", assessments)
    _write_json(
        run_dir / "risk/metrics.json",
        {
            "scope_status": "DEFERRED",
            "blocking": False,
            "defer_reason_zh": defer_reason,
            "rule_count": 3,
            "status_counts": {
                "TRIGGERED": 0,
                "NEEDS_USER": 3,
                "NOT_APPLICABLE": 0,
            },
            "disclaimer_zh": "仅为辅助风险提醒，不构成安全认证。",
        },
    )
    return assessments


def test_results_page_renders_completion_summaries_and_escapes_all_text(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_completion_summaries(run_dir)

    page = build_page(run_dir)

    assert '<h2 id="trusted-inventory">可信库存' in page
    assert '<h2 id="boxlist">箱单' in page
    assert '<h2 id="automatic-space">自动空间' in page
    assert '<h2 id="automatic-space-score">独立空间评分' in page
    assert '<h2 id="visual-space-review">视觉代理裁定' in page
    assert '<h2 id="risk-reminders">风险提醒' in page
    assert "3306" in page and "→" in page and "20" in page
    assert "3306 → 20" in page
    assert "上限 4" in page
    assert "placement 单元" in page and "物品覆盖" in page and "20/20" in page
    assert "候选实例" in page and "已投影" in page and "PASS" in page
    assert "精确语义分" in page and "5/5" in page and "零额外预测" in page
    assert "视觉接受" in page and "来源不会计入 AUTO_ACCEPTED" in page
    assert "NEAR 2" in page and "UNKNOWN 3" in page
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
        "DO_NOT_RENDER_SPATIAL_SCORE_AUDIT",
        "DO_NOT_RENDER_SPATIAL_REVIEW_AUDIT",
        "DO_NOT_RENDER_RISK_EVIDENCE",
    ):
        assert marker not in page


def test_results_page_promotes_judge_story_only_after_every_hard_gate_passes(tmp_path):
    run_dir = tmp_path / "demo-ready"
    run_dir.mkdir()
    _write_completion_summaries(run_dir)
    _write_json(
        run_dir / "layout/layout.json",
        {
            "status": "PLAN_READY",
            "assignments": {"trusted-placement": "region-desk"},
            "alternatives": {},
            "conflicts": [],
        },
    )
    _write_json(
        run_dir / "regions/regions.json",
        {
            "entries": [
                {
                    "region_id": "region-desk",
                    "display_name_zh": "书桌",
                }
            ]
        },
    )
    _write_jsonl(
        run_dir / "taskcards/taskcards.jsonl",
        [
            {
                "card_id": "card-demo",
                "box_label_zh": "学习文具箱",
                "target_region_name_zh": "书桌",
                "alternative_region_id": None,
                "items": [
                    {
                        "display_name_zh": "可信展示名",
                        "hero_crop_ref": "results/item.jpg",
                    }
                ],
                "verification_checklist": ["可信展示名出现在书桌"],
            }
        ],
    )
    _write_json(
        run_dir / "audit/replay-report.json",
        {
            "message_count": 4,
            "main_chain": {
                "complete": 1,
                "actions": [
                    "ENTITIES_READY",
                    "GROUPS_READY",
                    "PLACEMENT_READY",
                    "TASKS_READY",
                ],
            },
        },
    )

    page = build_page(run_dir)

    assert "把旧家的生活组合，" in page and "带到新家" in page
    assert "比赛技术闭环已通过" in page
    assert 'id="demo-story"' in page
    assert "3306→20" in page
    assert "候选实例" in page and "独立语义评分 5/5" in page
    assert "代表任务卡" in page and "学习文具箱" in page
    assert 'alt="可信展示名物品图"' in page
    assert "Agent 消息" not in page

    _write_json(
        run_dir / "spatial_score/metrics.json",
        {
            "acceptance_passed": False,
            "score": "4/5",
            "matched_anchor_count": 4,
            "exact_semantic_match_count": 4,
            "extra_prediction_count": 0,
            "support_type_mismatch_count": 0,
            "capacity_class_mismatch_count": 0,
            "gate_reasons": ["missing_anchor"],
        },
    )
    failed_page = build_page(run_dir)
    assert "比赛技术闭环已通过" not in failed_page
    assert 'id="demo-story"' not in failed_page
    assert "房间成果总览" in failed_page


def test_select_demo_space_frames_uses_only_final_assignment_evidence(tmp_path):
    asset_root = tmp_path / "frames"
    asset_root.mkdir()
    for name in ("frame-a.jpg", "frame-b.jpg", "unused.jpg"):
        (asset_root / name).write_bytes(b"jpeg")
    assignments = [
        {
            "anchor": "desk",
            "evidence_refs": ["frame:input/frame-a.jpg#observation=1"],
        },
        {
            "anchor": "shelf",
            "evidence_refs": [
                "frame:input/frame-a.jpg#observation=2",
                "frame:input/unused-missing.jpg#observation=3",
            ],
        },
        {
            "anchor": "cabinet",
            "evidence_refs": ["frame:input/frame-b.jpg#observation=4"],
        },
    ]

    selected = select_demo_space_frames(
        tmp_path / "run",
        assignments,
        asset_roots=[asset_root],
    )

    assert [item["basename"] for item in selected] == [
        "frame-a.jpg",
        "frame-b.jpg",
    ]
    assert selected[0]["anchors"] == ["desk", "shelf"]
    assert selected[1]["anchors"] == ["cabinet"]


def test_results_page_marks_deferred_risk_diagnostics_non_blocking(tmp_path):
    run_dir = tmp_path / "deferred-risk"
    run_dir.mkdir()
    _write_completion_summaries(run_dir)
    defer_reason = "现场关系难以从空房视频确认；<script>not-current</script>"
    assessments = _write_deferred_risk_scope(run_dir, defer_reason)

    page = build_page(run_dir)

    assert "已延期" in page
    assert "非阻塞" in page
    assert "诊断缺证据（已延期） 3" in page
    assert "待人工确认" not in page
    assert "MISSING_EVIDENCE:child_present" in page
    assert "MISSING_EVIDENCE:trip_hazard_present" in page
    assert "MISSING_EVIDENCE:powered_item_present" in page
    assert defer_reason not in page
    assert "现场关系难以从空房视频确认；&lt;script&gt;not-current&lt;/script&gt;" in page
    assert [item["status"] for item in assessments["assessments"]] == [
        "NEEDS_USER",
        "NEEDS_USER",
        "NEEDS_USER",
    ]


def test_results_page_renders_machine_readable_competition_scope(tmp_path):
    run_dir = tmp_path / "scope"
    run_dir.mkdir()
    closure = tmp_path / "closure.json"
    _write_json(
        closure,
        {
            "geometry_policy": {
                "reference_assumption": {
                    "value_cm": "<120>",
                    "status": "ASSUMED_PRIOR",
                },
                "non_required_metrics": [
                    "exact_surface_area",
                    "clear_height",
                    "load_capacity",
                    "doorway_clear_width",
                    "walk_path_clear_width",
                ],
            },
            "post_placement_verification_contract": {
                "purpose_zh": "只证明物理执行<script>bad</script>",
            },
        },
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        json.dumps(
            {
                "stages": {
                    "group": {"closure": str(closure)},
                    "risk": {"closure": str(closure)},
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    page = build_page(run_dir, config)

    assert 'id="competition-scope"' in page
    assert "相对容量" in page and "ASSUMED_PRIOR" in page
    assert "精确面积、净高、承重、门洞净宽、通道净宽" in page
    assert "可选延期" in page and "只证明物理执行" in page
    assert "&lt;120&gt;" in page
    assert "<script>bad</script>" not in page


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


def test_results_page_prefers_current_stage_state_and_explicit_config_hash(tmp_path):
    run_dir = tmp_path / "state-backed"
    _write_json(
        run_dir / "bundle.json",
        {
            "bundle_id": "state-backed",
            "config_refs": {"stale.yaml": "1" * 64},
            "artifacts": [
                {
                    "stage": "regions",
                    "path": "/stale/old-regions.json",
                    "sha256": "2" * 64,
                }
            ],
        },
    )
    _write_json(
        run_dir / "state/regions.json",
        {
            "stage": "regions",
            "status": "done",
            "outputs": {"/current/regions.json": "3" * 64},
        },
    )
    _write_json(
        run_dir / "state/report.json",
        {
            "stage": "report",
            "status": "done",
            "outputs": {"/stale/index.html": "4" * 64},
        },
    )
    config = tmp_path / "current.yaml"
    config.write_text("run_dir: current\n", encoding="utf-8")

    page = build_page(run_dir, config)

    assert "current.yaml" in page
    assert hashlib.sha256(config.read_bytes()).hexdigest()[:16] in page
    assert "regions.json" in page and ("3" * 16) in page
    assert "stale.yaml" not in page
    assert "old-regions.json" not in page
    assert "index.html" not in page
