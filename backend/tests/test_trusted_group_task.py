from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from backend.schemas.core import AgentHandoff, AgentRole
from backend.schemas.hero_bundle import HeroGroup
from backend.tools.trace import (
    finalize_message,
    load_trace,
    require_handoff,
    validate_trace,
    write_fragment,
)


PROJ = Path(__file__).resolve().parent.parent.parent
SCRIPT = PROJ / "scripts/trusted_group_task.py"
CLOSURE = PROJ / "fixtures/hero_s1/technical_closure.json"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture_inputs(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    closure = json.loads(CLOSURE.read_text(encoding="utf-8"))
    canonical_ids = closure["canonical_item_ids"]
    inventory = []
    display = []
    for index, canonical_id in enumerate(canonical_ids):
        entity_id = f"trusted-{index:02d}-{canonical_id}"
        record = {
            "canonical": {
                "canonical_id": canonical_id,
                "name_zh": f"库存名{index:02d}",
            },
            "downstream_eligible": True,
            "quantity": 3 if canonical_id == "book" else 1,
        }
        if index % 2 == 0:
            record.update(
                status="data-owner-confirmed",
                projected_entity_id=entity_id,
            )
        else:
            record.update(
                status="TRUSTED",
                entity={"entity_id": entity_id},
                evidence={"anchor_review_status": "data_owner_confirmed"},
            )
        inventory.append(record)
        # 半数由 display 覆盖，半数验证 inventory.canonical 名称回退。
        if index % 2 == 0:
            display.append(
                {
                    "entity_id": entity_id,
                    "display_name_zh": f"展示名{index:02d}",
                }
            )

    inventory_path = tmp_path / "inventory.jsonl"
    display_path = tmp_path / "display.jsonl"
    _write_jsonl(inventory_path, inventory)
    _write_jsonl(display_path, display)
    return inventory_path, display_path, canonical_ids


def _run(
    inventory: Path,
    display: Path,
    out: Path,
    *,
    trace_id: str | None = None,
    trace_parent: Path | None = None,
    trace_out: Path | None = None,
):
    argv = [
            sys.executable,
            str(SCRIPT),
            "--closure",
            str(CLOSURE),
            "--inventory",
            str(inventory),
            "--display",
            str(display),
            "--out-dir",
            str(out),
        ]
    if trace_id is not None:
        argv += ["--trace-id", trace_id]
    if trace_parent is not None:
        argv += ["--trace-parent", str(trace_parent)]
    if trace_out is not None:
        argv += ["--trace-out", str(trace_out)]
    return subprocess.run(
        argv,
        cwd=PROJ,
        capture_output=True,
        text=True,
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_trusted_cli_builds_exact_closure_groups_independents_and_boxlist(tmp_path):
    inventory, display, canonical_ids = _fixture_inputs(tmp_path)
    out = tmp_path / "out"
    proc = _run(inventory, display, out)
    assert proc.returncode == 0, proc.stderr

    closure = json.loads(CLOSURE.read_text(encoding="utf-8"))
    groups = [HeroGroup.model_validate(row) for row in _read_jsonl(out / "groups.jsonl")]
    assert len(groups) == 3
    projected_by_canonical = {
        row["canonical"]["canonical_id"]: (
            row.get("projected_entity_id") or row["entity"]["entity_id"]
        )
        for row in _read_jsonl(inventory)
    }
    for group, frozen in zip(groups, closure["life_groups"], strict=True):
        assert group.group_id == frozen["group_id"]
        assert group.name_zh == frozen["name_zh"]
        assert group.entity_ids == [
            projected_by_canonical[canonical_id]
            for canonical_id in frozen["canonical_item_ids"]
        ]
        assert {
            evidence.detail for evidence in group.member_evidence
        } == {f"已确认与「{group.name_zh}」物品一起打包"}
        assert all("closure" not in evidence.detail for evidence in group.member_evidence)
    assert sum(len(group.entity_ids) for group in groups) == 15

    life_groups = _read_jsonl(out / "life_groups.jsonl")
    assert len(life_groups) == 3
    assert all(group["source"] == "user" for group in life_groups)

    independent = _read_jsonl(out / "independent_pack_items.jsonl")
    assert len(independent) == 5
    assert [row["canonical_id"] for row in independent] == closure[
        "independent_pack_item_ids"
    ]
    assert all(row["is_life_group"] is False for row in independent)
    assert all(row["group_id"] is None for row in independent)
    assert {row["placement_unit_id"] for row in independent} == {
        "technical_toys_pack",
        "technical_snacks_pack",
    }

    placement_groups = [
        HeroGroup.model_validate(row)
        for row in _read_jsonl(out / "placement_groups.jsonl")
    ]
    assert len(placement_groups) == 5
    assert [group.group_id for group in placement_groups[:3]] == [
        group.group_id for group in groups
    ]
    assert [group.group_id for group in placement_groups[3:]] == [
        "technical_toys_pack",
        "technical_snacks_pack",
    ]
    assert [group.name_zh for group in placement_groups[3:]] == [
        "玩具收纳",
        "零食收纳",
    ]
    assert all(
        evidence.detail == f"按用途统一放入「{group.name_zh}箱」"
        for group in placement_groups[3:]
        for evidence in group.member_evidence
    )
    assert [group.target_region_hint for group in placement_groups] == [
        "书桌",
        "展示柜",
        "梳妆台",
        "墙上搁板",
        "斗柜",
    ]
    assert sum(len(group.entity_ids) for group in placement_groups) == 20

    boxlist = json.loads((out / "boxlist.json").read_text(encoding="utf-8"))
    assert boxlist["canonical_item_order"] == canonical_ids
    assert boxlist["canonical_item_count"] == 20
    assert boxlist["box_count"] == 5
    covered = [
        item["canonical_id"]
        for box in boxlist["boxes"]
        for item in box["items"]
    ]
    assert len(covered) == len(set(covered)) == 20
    assert set(covered) == set(canonical_ids)
    assert [box["box_type"] for box in boxlist["boxes"]].count("life_group") == 3
    assert [box["box_type"] for box in boxlist["boxes"]].count(
        "technical_pack_unit"
    ) == 2
    assert [box["box_label_zh"] for box in boxlist["boxes"][3:]] == [
        "玩具收纳箱",
        "零食收纳箱",
    ]
    # 名称只取自 display 或 inventory，且保留来源；数量也来自可信库存。
    all_box_items = [item for box in boxlist["boxes"] for item in box["items"]]
    assert {item["display_name_source"] for item in all_box_items} == {
        "display",
        "inventory.canonical",
    }
    assert next(item for item in all_box_items if item["canonical_id"] == "book")[
        "quantity"
    ] == 3


def test_outputs_and_hashes_are_byte_deterministic(tmp_path):
    inventory, display, _ = _fixture_inputs(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert _run(inventory, display, first).returncode == 0
    assert _run(inventory, display, second).returncode == 0

    names = (
        "groups.jsonl",
        "life_groups.jsonl",
        "placement_groups.jsonl",
        "independent_pack_items.jsonl",
        "boxlist.json",
        "metrics.json",
    )
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes()
        assert _sha256(first / name) == _sha256(second / name)
    metrics = json.loads((first / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["input_sha256"] == {
        "closure": _sha256(CLOSURE),
        "inventory": _sha256(inventory),
        "display": _sha256(display),
    }
    assert metrics["output_sha256"] == {
        name: _sha256(first / name) for name in names if name != "metrics.json"
    }
    assert metrics["group_count"] == 3
    assert metrics["placement_group_count"] == 5
    assert metrics["technical_pack_unit_count"] == 2
    assert metrics["grouped_item_count"] == 15
    assert metrics["placement_grouped_item_count"] == 20
    assert metrics["independent_item_count"] == 5
    assert metrics["covered_canonical_item_count"] == 20


def test_inventory_to_group_trace_validates_and_business_bytes_stay_identical(tmp_path):
    inventory, display, _ = _fixture_inputs(tmp_path)
    plain_out = tmp_path / "plain"
    traced_out = tmp_path / "traced"
    assert _run(inventory, display, plain_out).returncode == 0

    inventory_trace = tmp_path / "inventory-trace.jsonl"
    trusted_entity_ids = sorted(
        row.get("projected_entity_id") or row["entity"]["entity_id"]
        for row in _read_jsonl(inventory)
    )
    parent = finalize_message(
        AgentHandoff(
            message_id="hero-trusted-entities-ready",
            correlation_id="hero-trusted",
            producer=AgentRole.MEM,
            target=AgentRole.GROUP,
            action="ENTITIES_READY",
            item_ids=trusted_entity_ids,
            artifact_refs=[str(inventory)],
            summary={"trusted_inventory": 20},
        )
    )
    write_fragment(inventory_trace, [parent])
    group_trace = tmp_path / "group-trace.jsonl"
    proc = _run(
        inventory,
        display,
        traced_out,
        trace_id="hero-trusted",
        trace_parent=inventory_trace,
        trace_out=group_trace,
    )
    assert proc.returncode == 0, proc.stderr

    handoff = require_handoff(group_trace, "GROUPS_READY")
    assert handoff.producer == AgentRole.GROUP
    assert handoff.target == AgentRole.SPACE
    assert handoff.causation_id == parent.message_id
    assert handoff.correlation_id == parent.correlation_id
    assert handoff.item_ids == [
        "study_stationery",
        "cups_and_drinks",
        "toiletries_and_care",
        "technical_toys_pack",
        "technical_snacks_pack",
    ]
    assert handoff.artifact_refs == [
        str(traced_out / "groups.jsonl"),
        str(traced_out / "life_groups.jsonl"),
        str(traced_out / "placement_groups.jsonl"),
        str(traced_out / "independent_pack_items.jsonl"),
        str(traced_out / "boxlist.json"),
        str(traced_out / "metrics.json"),
    ]
    assert handoff.summary == {
        "life_groups": 3,
        "placement_groups": 5,
        "trusted_entities": 20,
    }
    report = validate_trace(load_trace(inventory_trace) + load_trace(group_trace))
    assert report["status"] == "PASS"
    assert report["producer_counts"] == {"GROUP": 1, "MEM": 1}

    # trace 参数只增加旁路 fragment，原五项业务产物及 metrics 均逐字节不变。
    for name in (
        "groups.jsonl",
        "life_groups.jsonl",
        "placement_groups.jsonl",
        "independent_pack_items.jsonl",
        "boxlist.json",
        "metrics.json",
    ):
        assert (plain_out / name).read_bytes() == (traced_out / name).read_bytes()


def test_trace_arguments_are_all_or_none(tmp_path):
    inventory, display, _ = _fixture_inputs(tmp_path)
    proc = _run(inventory, display, tmp_path / "out", trace_id="incomplete")
    assert proc.returncode != 0
    assert (
        "--trace-id, --trace-parent and --trace-out must be provided together"
        in proc.stderr
    )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "extra", "ineligible"])
def test_bad_inventory_fails_closed_without_outputs(tmp_path, mutation):
    inventory, display, _ = _fixture_inputs(tmp_path)
    rows = _read_jsonl(inventory)
    if mutation == "duplicate":
        rows[-1]["canonical"] = dict(rows[0]["canonical"])
    elif mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        extra = dict(rows[-1])
        extra["canonical"] = {"canonical_id": "unexpected", "name_zh": "额外物品"}
        extra["projected_entity_id"] = "trusted-extra"
        extra.pop("entity", None)
        rows.append(extra)
    else:
        rows[0]["downstream_eligible"] = False
    _write_jsonl(inventory, rows)
    out = tmp_path / "out"

    proc = _run(inventory, display, out)

    assert proc.returncode != 0
    assert not out.exists()


def test_projected_entity_ids_never_come_from_untrusted_display(tmp_path):
    inventory, display, _ = _fixture_inputs(tmp_path)
    display_rows = _read_jsonl(display)
    display_rows.append(
        {
            "entity_id": "model-only-entity",
            "display_name_zh": "不可信模型物品",
        }
    )
    _write_jsonl(display, display_rows)
    out = tmp_path / "out"
    assert _run(inventory, display, out).returncode == 0

    emitted_ids = {
        item["entity_id"]
        for box in json.loads((out / "boxlist.json").read_text(encoding="utf-8"))["boxes"]
        for item in box["items"]
    }
    assert "model-only-entity" not in emitted_ids
    trusted_ids = {
        row.get("projected_entity_id") or row["entity"]["entity_id"]
        for row in _read_jsonl(inventory)
    }
    assert emitted_ids == trusted_ids
