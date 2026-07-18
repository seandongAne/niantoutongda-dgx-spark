from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from backend.tools.risks import RISK_DISCLAIMER_ZH, RULE_FACT_KEYS


PROJ = Path(__file__).resolve().parent.parent.parent
SCRIPT = PROJ / "scripts/risk_task.py"
CLOSURE = PROJ / "fixtures/hero_s1/technical_closure.json"


def _fact(value: bool, confidence: float = 0.95, ref: str = "new_1.mp4@00:01"):
    return {
        "value": value,
        "confidence": confidence,
        "evidence_refs": [ref],
    }


def _all_true(rule_id: str):
    return {
        key: _fact(True, ref=f"new_1.mp4@{key}")
        for key in RULE_FACT_KEYS[rule_id]
    }


def _run(facts_path: Path, out_dir: Path):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--closure",
            str(CLOSURE),
            "--facts",
            str(facts_path),
            "--out-dir",
            str(out_dir),
        ],
        cwd=PROJ,
        capture_output=True,
        text=True,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def mixed_facts(tmp_path: Path) -> Path:
    # 第三条故意缺失：CLI 必须交给规则评估器产出 NEEDS_USER。
    payload = {
        "CHILD_SHARP_TOOL_REACH": {
            "subject_ids": ["utility_knife", "scissors"],
            "facts": _all_true("CHILD_SHARP_TOOL_REACH"),
        },
        "TRIP_HAZARD_IN_PATH": {
            "subject_ids": ["floor_mat"],
            "facts": {
                "trip_hazard_present": _fact(False, ref="new_1.mp4@no-hazard"),
            },
        },
    }
    path = tmp_path / "risk-facts.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_cli_uses_closure_order_and_missing_rule_needs_user(mixed_facts, tmp_path):
    out = tmp_path / "out"
    proc = _run(mixed_facts, out)
    assert proc.returncode == 0, proc.stderr

    closure = json.loads(CLOSURE.read_text(encoding="utf-8"))
    expected_order = [rule["rule_id"] for rule in closure["risk_contract"]["rules"]]
    assessments = json.loads((out / "assessments.json").read_text(encoding="utf-8"))
    assert assessments["rule_order"] == expected_order
    assert [item["rule_id"] for item in assessments["assessments"]] == expected_order
    assert [item["status"] for item in assessments["assessments"]] == [
        "TRIGGERED",
        "NOT_APPLICABLE",
        "NEEDS_USER",
    ]
    missing = assessments["assessments"][2]
    assert missing["reason_codes"] == ["MISSING_EVIDENCE:powered_item_present"]
    assert missing["confidence"] == 0.0
    assert all(item["disclaimer_zh"] == RISK_DISCLAIMER_ZH
               for item in assessments["assessments"])

    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status_counts"] == {
        "TRIGGERED": 1,
        "NEEDS_USER": 1,
        "NOT_APPLICABLE": 1,
    }
    assert metrics["disclaimer_zh"] == RISK_DISCLAIMER_ZH
    assert metrics["input_sha256"] == {
        "closure": _sha256(CLOSURE),
        "facts": _sha256(mixed_facts),
    }
    assert metrics["output_sha256"] == {
        "assessments.json": _sha256(out / "assessments.json")
    }


def test_cli_outputs_are_byte_deterministic(mixed_facts, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert _run(mixed_facts, first).returncode == 0
    assert _run(mixed_facts, second).returncode == 0

    for name in ("assessments.json", "metrics.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
        assert _sha256(first / name) == _sha256(second / name)


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        (
            {"UNKNOWN_RULE": {"facts": {}, "subject_ids": []}},
            "unknown rules in facts",
        ),
        (
            {
                "TRIP_HAZARD_IN_PATH": {
                    "facts": {
                        **_all_true("TRIP_HAZARD_IN_PATH"),
                        "hallucinated_safe": _fact(True),
                    },
                    "subject_ids": [],
                }
            },
            "unexpected facts",
        ),
    ],
)
def test_unknown_rule_or_fact_fails_closed_without_outputs(payload, error, tmp_path):
    facts = tmp_path / "invalid.json"
    facts.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "out"

    proc = _run(facts, out)

    assert proc.returncode != 0
    assert error in proc.stderr
    assert not (out / "assessments.json").exists()
    assert not (out / "metrics.json").exists()
