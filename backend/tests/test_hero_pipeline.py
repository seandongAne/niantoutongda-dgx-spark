"""hero_pipeline 端到端:合成夹具全链路 + 断点续跑语义。

真子进程执行(阶段隔离是被测行为的一部分),使用 --run-dir 隔离产物。
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.hero_pipeline import STAGE_ORDER, build_stages

PROJ = Path(__file__).resolve().parent.parent.parent
CONFIG = PROJ / "configs/hero_pipeline_dev.yaml"


def run_pipeline(run_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PROJ / "scripts/hero_pipeline.py"),
         "--config", str(CONFIG), "--run-dir", str(run_dir), *extra],
        capture_output=True, text=True, cwd=PROJ,
    )


def test_full_chain_then_resume_then_from_stage(tmp_path):
    run_dir = tmp_path / "run"

    first = run_pipeline(run_dir)
    assert first.returncode == 0, first.stderr
    assert "[run ] naming" in first.stdout
    assert "✅ 主链完成" in first.stdout
    bundle = json.loads((run_dir / "bundle.json").read_text(encoding="utf-8"))
    stages_in_bundle = {a["stage"] for a in bundle["artifacts"]}
    assert {"naming", "narration", "regions", "group", "layout", "taskcards",
            "verify", "trace"} <= stages_in_bundle
    cards = (run_dir / "taskcards/taskcards.md").read_text(encoding="utf-8")
    assert "水壶(蓝色)" in cards and "水壶(粉色)" in cards

    verdicts = json.loads(
        (run_dir / "verify/verdicts.json").read_text(encoding="utf-8")
    )
    assert {v["verdict"] for v in verdicts.values()} == {
        "VERIFIED", "FAILED", "NEEDS_USER"
    }
    assert (run_dir / "index.html").exists()
    replay = json.loads(
        (run_dir / "audit/replay-report.json").read_text(encoding="utf-8")
    )
    assert replay["status"] == "PASS"
    assert replay["main_chain"]["complete"] == 1
    assert replay["producer_counts"] == {
        "EXEC": 7,
        "GROUP": 1,
        "MEM": 5,
        "SPACE": 4,
        "USER": 3,
    }
    assert replay["clarifications"] == {"requests": 1, "closed": 1, "open": 0}
    assert replay["verification"]["requests"] == 3
    assert replay["verification"]["adjudication_closed"] == 2

    second = run_pipeline(run_dir)
    assert second.returncode == 0, second.stderr
    assert second.stdout.count("[skip]") == 9
    assert "[run ] naming" not in second.stdout

    third = run_pipeline(run_dir, "--from-stage", "layout")
    assert third.returncode == 0, third.stderr
    assert "[skip] group" in third.stdout
    assert "[run ] layout" in third.stdout
    assert "[run ] taskcards" in third.stdout


def test_dry_run_lists_plan_without_side_effects(tmp_path):
    run_dir = tmp_path / "dry"
    proc = run_pipeline(run_dir, "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert "[run ] naming" in proc.stdout
    assert not run_dir.exists()


def test_stale_stage_reruns_but_unchanged_output_spares_downstream(tmp_path):
    """内容寻址续跑:上游被判 stale 后重跑,但产物字节不变时下游不陪跑。"""
    run_dir = tmp_path / "run"
    assert run_pipeline(run_dir).returncode == 0
    state_path = run_dir / "state/narration.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    transcript = next(iter(state["inputs"]))
    state["inputs"][transcript] = "0" * 64  # 伪造输入哈希 → narration 判 stale
    state_path.write_text(json.dumps(state), encoding="utf-8")
    proc = run_pipeline(run_dir)
    assert proc.returncode == 0, proc.stderr
    assert "[skip] naming" in proc.stdout
    assert "[run ] narration" in proc.stdout
    # 重跑产物与上次字节一致 → group 及以下按内容寻址跳过
    assert "[skip] group" in proc.stdout
    assert "[skip] taskcards" in proc.stdout


def test_build_stages_wires_trusted_inventory_auto_space_and_risk(tmp_path):
    run = tmp_path / "run"
    entities = tmp_path / "entities.jsonl"
    items = tmp_path / "items.json"
    review = tmp_path / "anchor_review.json"
    observations = tmp_path / "space_observations.jsonl"
    closure = tmp_path / "technical_closure.json"
    facts = tmp_path / "risk_facts.json"
    manual_regions = tmp_path / "manual_regions_must_not_be_used.json"
    observed_local = tmp_path / "space-observe-pulled.jsonl"
    cfg = {
        "run_dir": str(run),
        "trace_id": "trusted-hero",
        "stages": {
            "space_observe": {
                "enabled": True,
                "spark_cmd": "python scripts/observe_new_home.py --video new_1.mp4",
                "local_outputs": [str(observed_local)],
            },
            "pull": {"enabled": True, "cmd": ["/bin/true"]},
            "inventory": {
                "enabled": True,
                "entities": str(entities),
                "items": str(items),
                "anchor_review": str(review),
                "max_clarifications": 4,
            },
            "space": {
                "enabled": True,
                "observations": str(observations),
                "video_id": "new_1",
                "min": 5,
                "min_observations": 2,
                "min_confidence": 0.8,
                "min_hard_field_confidence": 0.75,
                "min_power_confidence": 0.72,
                "min_field_consensus": 0.7,
                "dedupe_iou": 0.3,
                "expected_anchor": ["bed", "desk", "closet"],
            },
            "regions": {
                "enabled": True,
                "source": "auto",
            },
            "group": {
                "enabled": True,
                "trusted_inventory": True,
                "closure": str(closure),
            },
            "layout": {"enabled": True},
            "taskcards": {"enabled": True},
            "risk": {
                "enabled": True,
                "closure": str(closure),
                "facts": str(facts),
            },
            "trace": {"enabled": True},
            "report": {"enabled": True},
            "bundle": {"enabled": False},
        },
    }

    stages = build_stages(cfg, "PYTHON")
    enabled_order = [name for name in STAGE_ORDER if name in stages]
    assert enabled_order == [
        "space_observe",
        "pull",
        "inventory",
        "space",
        "regions",
        "group",
        "layout",
        "taskcards",
        "risk",
        "trace",
        "report",
    ]

    space_observe = stages["space_observe"]
    assert space_observe.kind == "spark"
    assert space_observe.spark_cmd == (
        "python scripts/observe_new_home.py --video new_1.mp4"
    )
    assert space_observe.outputs == [observed_local]

    inventory = stages["inventory"]
    inventory_dir = run / "inventory"
    assert inventory.argv == [
        "PYTHON",
        str(PROJ / "scripts/inventory_task.py"),
        "--entities",
        str(entities),
        "--items",
        str(items),
        "--anchor-review",
        str(review),
        "--out-dir",
        str(inventory_dir),
        "--max-clarifications",
        "4",
        "--trace-id",
        "trusted-hero",
        "--trace-out",
        str(inventory_dir / "trace.jsonl"),
    ]
    assert inventory.inputs == [entities, items, review]
    assert inventory.outputs == [
        inventory_dir / "inventory.jsonl",
        inventory_dir / "trusted_entities.jsonl",
        inventory_dir / "display.jsonl",
        inventory_dir / "clarifications.jsonl",
        inventory_dir / "metrics.json",
        inventory_dir / "manifest.json",
        inventory_dir / "hashes.json",
        inventory_dir / "trace.jsonl",
    ]

    space = stages["space"]
    assert space.inputs == [observations]
    assert space.argv[:8] == [
        "PYTHON",
        str(PROJ / "scripts/space_task.py"),
        "--video-id",
        "new_1",
        "--observations",
        str(observations),
        "--out-dir",
        str(run / "spatial"),
    ]
    for flag, value in (
        ("--min-regions", "5"),
        ("--min-observations", "2"),
        ("--min-confidence", "0.8"),
        ("--min-hard-field-confidence", "0.75"),
        ("--min-power-confidence", "0.72"),
        ("--min-field-consensus", "0.7"),
        ("--dedupe-iou", "0.3"),
    ):
        index = space.argv.index(flag)
        assert space.argv[index + 1] == value
    assert [
        space.argv[index + 1]
        for index, value in enumerate(space.argv)
        if value == "--expected-anchor"
    ] == ["bed", "desk", "closet"]
    assert space.outputs == [
        run / "spatial/candidate_manifest.json",
        run / "spatial/regions.json",
        run / "spatial/metrics.json",
        run / "spatial/normalized.sha256",
    ]

    regions = stages["regions"]
    assert regions.inputs == [run / "spatial/regions.json"]
    assert str(run / "spatial/regions.json") in regions.argv
    assert str(manual_regions) not in regions.argv
    assert manual_regions not in regions.inputs

    cfg["stages"]["space"]["shadow_only"] = True
    with pytest.raises(
        ValueError, match="regions.source=auto requires non-shadow automatic space"
    ):
        build_stages(cfg, "PYTHON")

    cfg["stages"]["regions"]["source"] = "fixture"
    cfg["stages"]["regions"]["manifest"] = str(manual_regions)
    shadow_stages = build_stages(cfg, "PYTHON")
    shadow_space = shadow_stages["space"]
    assert "--shadow-only" in shadow_space.argv
    assert shadow_space.outputs == [
        run / "spatial/candidate_manifest.json",
        run / "spatial/metrics.json",
        run / "spatial/normalized.sha256",
    ]
    shadow_regions = shadow_stages["regions"]
    assert shadow_regions.inputs == [manual_regions]
    assert str(manual_regions) in shadow_regions.argv
    assert str(run / "spatial/regions.json") not in shadow_regions.argv

    group = stages["group"]
    assert group.argv[1] == str(PROJ / "scripts/trusted_group_task.py")
    assert group.inputs == [
        closure,
        inventory_dir / "inventory.jsonl",
        inventory_dir / "display.jsonl",
        inventory_dir / "trace.jsonl",
    ]
    assert group.outputs == [
        run / "group/groups.jsonl",
        run / "group/life_groups.jsonl",
        run / "group/placement_groups.jsonl",
        run / "group/independent_pack_items.jsonl",
        run / "group/boxlist.json",
        run / "group/metrics.json",
        run / "group/trace.jsonl",
    ]
    assert group.argv[-6:] == [
        "--trace-id",
        "trusted-hero",
        "--trace-parent",
        str(inventory_dir / "trace.jsonl"),
        "--trace-out",
        str(run / "group/trace.jsonl"),
    ]

    placement_groups = run / "group/placement_groups.jsonl"
    assert stages["layout"].inputs[0] == placement_groups
    assert stages["taskcards"].inputs[0] == placement_groups
    assert stages["taskcards"].inputs[3] == inventory_dir / "display.jsonl"

    risk = stages["risk"]
    assert risk.inputs == [closure, facts]
    assert risk.argv == [
        "PYTHON",
        str(PROJ / "scripts/risk_task.py"),
        "--closure",
        str(closure),
        "--facts",
        str(facts),
        "--out-dir",
        str(run / "risk"),
    ]
    assert risk.outputs == [
        run / "risk/assessments.json",
        run / "risk/metrics.json",
    ]

    trace = stages["trace"]
    assert trace.inputs[0] == inventory_dir / "trace.jsonl"
    assert run / "naming/trace.jsonl" not in trace.inputs
    report_inputs = stages["report"].inputs
    for expected in (
        inventory_dir / "display.jsonl",
        run / "group/placement_groups.jsonl",
        inventory_dir / "metrics.json",
        run / "group/boxlist.json",
        closure,
        run / "spatial/metrics.json",
        run / "risk/metrics.json",
    ):
        assert expected in report_inputs


def test_build_stages_wires_visual_space_adjudication_as_explicit_source(tmp_path):
    run = tmp_path / "run"
    observations = tmp_path / "observations.jsonl"
    review = tmp_path / "visual-review.json"
    frame = tmp_path / "frame.jpg"
    cfg = {
        "run_dir": str(run),
        "stages": {
            "space": {
                "enabled": True,
                "shadow_only": True,
                "video_id": "new_1",
                "observations": str(observations),
            },
            "space_review": {
                "enabled": True,
                "review": str(review),
                "evidence_frames": [str(frame)],
            },
            "regions": {
                "enabled": True,
                "source": "visual_adjudication",
            },
            "report": {"enabled": True},
            "bundle": {"enabled": False},
        },
    }

    stages = build_stages(cfg, "PYTHON")

    assert [name for name in STAGE_ORDER if name in stages] == [
        "space", "space_review", "regions", "report"
    ]
    adjudication = stages["space_review"]
    assert adjudication.inputs == [
        run / "spatial/candidate_manifest.json",
        run / "spatial/normalized.sha256",
        review,
        frame,
    ]
    assert adjudication.outputs == [
        run / "spatial_review/adjudication_manifest.json",
        run / "spatial_review/regions.json",
        run / "spatial_review/metrics.json",
        run / "spatial_review/normalized.sha256",
    ]
    assert stages["regions"].inputs == [run / "spatial_review/regions.json"]
    assert run / "spatial_review/metrics.json" in stages["report"].inputs


def test_visual_space_source_requires_review_stage(tmp_path):
    cfg = {
        "run_dir": str(tmp_path / "run"),
        "stages": {
            "regions": {
                "enabled": True,
                "source": "visual_adjudication",
            },
            "bundle": {"enabled": False},
        },
    }

    with pytest.raises(ValueError, match="requires space_review"):
        build_stages(cfg, "PYTHON")


def test_space_score_is_independent_sidecar_and_never_produces_regions(tmp_path):
    run = tmp_path / "run"
    observations = tmp_path / "observations.jsonl"
    truth = tmp_path / "semantic-truth.json"
    cfg = {
        "run_dir": str(run),
        "stages": {
            "space": {
                "enabled": True,
                "video_id": "new_1",
                "observations": str(observations),
                "expected_anchor": ["a", "b", "c", "d", "e"],
            },
            "space_score": {
                "enabled": True,
                "truth": str(truth),
                "required_expected_anchor_count": 5,
            },
            "regions": {"enabled": True, "source": "auto"},
            "bundle": {"enabled": False},
        },
    }

    stages = build_stages(cfg, "PYTHON")

    assert [name for name in STAGE_ORDER if name in stages] == [
        "space",
        "space_score",
        "regions",
    ]
    score = stages["space_score"]
    assert score.inputs == [run / "spatial/regions.json", truth]
    assert score.outputs == [
        run / "spatial_score/score_manifest.json",
        run / "spatial_score/metrics.json",
        run / "spatial_score/normalized.sha256",
    ]
    assert all(path.name != "regions.json" for path in score.outputs)
    assert stages["regions"].inputs == [run / "spatial/regions.json"]
    assert not any("spatial_score" in str(path) for path in stages["regions"].inputs)


def test_auto_region_source_forbids_parallel_manual_manifest(tmp_path):
    cfg = {
        "run_dir": str(tmp_path / "run"),
        "stages": {
            "space": {
                "enabled": True,
                "video_id": "new_1",
                "observations": str(tmp_path / "observations.jsonl"),
            },
            "regions": {
                "enabled": True,
                "source": "auto",
                "manifest": str(tmp_path / "manual.json"),
            },
            "bundle": {"enabled": False},
        },
    }

    with pytest.raises(ValueError, match="source=auto forbids manifest"):
        build_stages(cfg, "PYTHON")


def test_final_config_is_strict_auto_with_fresh_pull_and_semantic_score():
    config_path = PROJ / "configs/hero_pipeline_s1_final.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stages = build_stages(cfg, ".venv/bin/python", config_path=config_path)

    space = stages["space"]
    regions = stages["regions"]
    score = stages["space_score"]
    assert "--shadow-only" not in space.argv
    assert "--anchor-candidates" in space.argv
    assert "--anchor-hashes" in space.argv
    assert "--observation-hashes" in space.argv
    run_path = Path(cfg["run_dir"])
    expected_spatial = PROJ / run_path / "spatial/regions.json"
    assert regions.inputs == [expected_spatial]
    assert score.inputs[0] == expected_spatial
    assert all(path.name != "regions.json" for path in score.outputs)
    assert stages["pull"].always_run is True
    assert "space_review" not in stages
    forbidden = (
        "space_visual_adjudication",
        "regions.target5",
        "spatial_review/regions.json",
    )
    production_text = "\n".join(
        [*space.argv, *map(str, regions.inputs), *regions.argv]
    )
    assert not any(value in production_text for value in forbidden)
