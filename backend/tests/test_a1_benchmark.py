import json
import math
import wave
from array import array

import pytest

from backend.tools.a1_benchmark import (
    aggregate_scores,
    build_plan,
    parse_prediction,
    required_wilson_trials,
    score_case,
    wilson_interval,
)
from scripts.a1_robustness import (
    add_noise_at_snr,
    backend_runtime_manifests,
    build_parser,
    canonical_wav,
    git_revision,
)


def test_plan_is_deterministic_balanced_and_capped():
    kwargs = dict(
        seed=20260716,
        base_cases=6,
        condition_ids=["clean", "noise20"],
        voices=["voice-a", "voice-b"],
        target_half_width=0.08,
        minimum_per_condition=2,
        maximum_observations=20,
    )
    first = build_plan(**kwargs)
    second = build_plan(**kwargs)

    assert first == second
    assert first["observation_count_per_backend"] == 12
    assert {case["style"] for case in first["cases"]} == {
        "direct",
        "reordered",
        "conversational",
    }
    assert {case["completeness"] for case in first["cases"]} == {"full", "partial"}
    assert {case["voice"] for case in first["cases"]} == {"voice-a", "voice-b"}
    assert all("家庭" not in case["narration"] for case in first["cases"])

    reordered = next(case for case in first["cases"] if case["style"] == "reordered")
    for expected, fragment in zip(
        reordered["expected"], reordered["narration"].split("接下来，"), strict=True
    ):
        label_position = fragment.index(expected["label_zh"])
        if expected["target_location"]:
            assert label_position < fragment.index(expected["target_location"])


def test_wilson_target_uses_worst_case_precision():
    required = required_wilson_trials(0.08)
    at_required = wilson_interval(required // 2, required)
    before = wilson_interval((required - 1) // 2, required - 1)

    assert at_required["half_width"] <= 0.08
    assert before["half_width"] > 0.08


def test_parser_accepts_json_fence_but_rejects_explanation():
    item = {
        "label_zh": "水杯",
        "label_en": "water bottle",
        "owner": "小明",
        "source_location": "旧卧室书桌上",
        "target_location": None,
        "pack_group": "学习用品",
        "attributes": {"color": "蓝色"},
    }
    fenced = "```json\n" + json.dumps([item], ensure_ascii=False) + "\n```"
    parsed, valid, error = parse_prediction(fenced)
    assert parsed == [item]
    assert valid is True
    assert error is None

    _, valid, error = parse_prediction("结果如下：" + json.dumps([item], ensure_ascii=False))
    assert valid is False
    assert error == "schema_mismatch"


def test_case_scoring_normalizes_aliases_and_counts_missed_slots():
    case = {
        "case_id": "case-0001",
        "style": "direct",
        "voice": "voice-a",
        "completeness": "full",
        "expected": [
            {
                "canonical_id": "water_bottle",
                "label_zh": "水杯",
                "label_en": "water bottle",
                "owner": "小明",
                "source_location": "旧卧室书桌上",
                "target_location": "新家卧室书桌上",
                "pack_group": "学习用品",
                "color": "蓝色",
            },
            {
                "canonical_id": "table_lamp",
                "label_zh": "台灯",
                "label_en": "table lamp",
                "owner": "妈妈",
                "source_location": "旧卧室床头柜上",
                "target_location": "新家卧室床头柜上",
                "pack_group": "睡前用品",
                "color": "白色",
            },
        ],
    }
    prediction = [{
        "label_zh": "蓝色水杯",
        "label_en": "blue water bottle",
        "owner": "小明的",
        "source_location": "旧卧室的书桌上",
        "target_location": "新家卧室的书桌上",
        "pack_group": "学习用品组",
        "attributes": {"color": "蓝"},
    }]
    scored = score_case(case, json.dumps(prediction, ensure_ascii=False))

    assert scored["item_true_positives"] == 1
    assert scored["expected_items"] == 2
    assert scored["slot_total"] == 10
    assert scored["slot_correct"] == 5
    assert scored["exact_case"] is False


def test_aggregate_requires_every_condition_to_reach_stop_gate():
    plan = build_plan(
        seed=1,
        base_cases=2,
        condition_ids=["clean", "noise20"],
        voices=["voice-a"],
        target_half_width=0.2,
        minimum_per_condition=1,
        maximum_observations=10,
    )
    plan["stopping_rule"]["worst_case_required_per_condition"] = 2
    plan["stopping_rule"]["target_half_width"] = 0.5
    records = []
    for condition in plan["condition_ids"]:
        for case in plan["cases"]:
            records.append({
                "backend": "cloud",
                "condition_id": condition,
                "style": case["style"],
                "voice": case["voice"],
                "completeness": case["completeness"],
                "schema_valid": True,
                "exact_case": True,
                "item_true_positives": len(case["expected"]),
                "expected_items": len(case["expected"]),
                "predicted_items": len(case["expected"]),
                "slot_correct": len(case["expected"]) * 5,
                "slot_total": len(case["expected"]) * 5,
                "parse_error": None,
            })

    report = aggregate_scores(records, plan)
    assert report["backends"]["cloud"]["stopping"]["reached"] is True
    assert report["backends"]["cloud"]["coverage"]["condition_id"] == {
        "clean": 2,
        "noise20": 2,
    }


def test_noise_injection_is_deterministic_and_hits_requested_snr(tmp_path):
    source = tmp_path / "source.wav"
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    samples = array("h", [round(10_000 * math.sin(index / 20)) for index in range(16_000)])
    with wave.open(str(source), "wb") as writer:
        writer.setparams((1, 2, 16_000, 0, "NONE", "not compressed"))
        writer.writeframes(samples.tobytes())

    add_noise_at_snr(source, first, snr_db=20.0, seed=42)
    add_noise_at_snr(source, second, snr_db=20.0, seed=42)
    assert first.read_bytes() == second.read_bytes()

    with wave.open(str(first), "rb") as reader:
        noisy = array("h")
        noisy.frombytes(reader.readframes(reader.getnframes()))
    signal_power = sum(value * value for value in samples) / len(samples)
    noise_power = sum((a - b) ** 2 for a, b in zip(noisy, samples, strict=True)) / len(samples)
    measured_snr = 10 * math.log10(signal_power / noise_power)
    assert measured_snr == pytest.approx(20.0, abs=0.25)


def test_pcm16_normalization_without_ffmpeg(monkeypatch, tmp_path):
    source = tmp_path / "source-24k.wav"
    clean = tmp_path / "clean-16k.wav"
    slow = tmp_path / "slow-16k.wav"
    samples = array("h", [round(10_000 * math.sin(index / 20)) for index in range(24_000)])
    with wave.open(str(source), "wb") as writer:
        writer.setparams((1, 2, 24_000, 0, "NONE", "not compressed"))
        writer.writeframes(samples.tobytes())

    monkeypatch.setattr("scripts.a1_robustness.shutil.which", lambda _: None)
    canonical_wav(source, clean)
    canonical_wav(source, slow, speed_ratio=0.9)

    with wave.open(str(clean), "rb") as reader:
        assert reader.getparams()[:3] == (1, 2, 16_000)
        assert reader.getnframes() == 16_000
    with wave.open(str(slow), "rb") as reader:
        assert reader.getparams()[:3] == (1, 2, 16_000)
        assert reader.getnframes() == pytest.approx(16_000 / 0.9, abs=1)


def test_formal_cloud_runner_defaults_to_greedy_decoding():
    args = build_parser().parse_args(["cloud", "--run-dir", "example"])
    assert args.temperature == 0.0


def test_git_revision_falls_back_to_deploy_stamp(tmp_path):
    (tmp_path / "COMMIT").write_text("abc123\n", encoding="utf-8")
    assert git_revision(tmp_path) == "abc123"


def test_backend_runtime_manifests_only_collect_existing_backends(tmp_path):
    path = tmp_path / "predictions" / "local" / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"model_load_seconds": 12.5}), encoding="utf-8")

    assert backend_runtime_manifests(tmp_path, ["cloud", "local"]) == {
        "local": {"model_load_seconds": 12.5}
    }
