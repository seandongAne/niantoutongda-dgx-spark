"""S2.5 hardval 固定口径与 prompt 排序测试（不含真实/人工验证数据）。"""

import copy
import json

import pytest

from backend.tools.detection_eval import (
    EvaluationInputError,
    evaluate_detection,
    rank_prompt_candidates,
    score_prompt,
)
from scripts.hardval_eval import main as hardval_main
from scripts.prompt_search import main as prompt_search_main


def _gt():
    return {
        "dataset_id": "dev_a_hardval_unit",
        "frames": [
            {
                "sequence_id": "v1",
                "frame_id": "f2",
                "instances": [
                    {
                        "instance_id": "anchor_lamp",
                        "canonical_id": "table_lamp",
                        "bbox": [0, 0, 10, 10],
                    },
                    {
                        "instance_id": "anchor_box",
                        "canonical_id": "storage_box",
                        "bbox": [20, 0, 30, 10],
                    },
                ],
            },
            {
                "sequence_id": "v1",
                "frame_id": "f1",
                "instances": [
                    {
                        "instance_id": "anchor_lamp",
                        "canonical_id": "table_lamp",
                        "bbox": [0, 0, 10, 10],
                    },
                    {
                        "instance_id": "anchor_box",
                        "canonical_id": "storage_box",
                        "bbox": [20, 0, 30, 10],
                    },
                    {
                        "instance_id": "anchor_hidden",
                        "canonical_id": "hidden_object",
                        "bbox": [40, 0, 50, 10],
                        "visible": False,
                    },
                ],
            },
        ],
    }


def _predictions():
    return {
        "dataset_id": "dev_a_hardval_unit",
        "frames": [
            {
                "sequence_id": "v1",
                "frame_id": "f1",
                "predictions": [
                    {
                        "track_id": "lamp_track_a",
                        "canonical_id": "table_lamp",
                        "bbox": [0, 0, 10, 10],
                    },
                    {
                        "track_id": "box_track",
                        "canonical_id": "storage_box",
                        "bbox": [20, 0, 30, 10],
                    },
                    {
                        "track_id": "wrong_canonical",
                        "canonical_id": "storage_box",
                        "bbox": [0, 0, 10, 10],
                    },
                ],
            },
            {
                "sequence_id": "v1",
                "frame_id": "f2",
                "predictions": [
                    {
                        "track_id": "lamp_track_b",
                        "canonical_id": "table_lamp",
                        "bbox": [0, 0, 10, 10],
                    },
                    {
                        "track_id": "far_box",
                        "canonical_id": "storage_box",
                        "bbox": [100, 0, 110, 10],
                    },
                ],
            },
        ],
    }


def test_frozen_metrics_and_fragmentation_denominator():
    result = evaluate_detection(_gt(), _predictions())

    assert result.matched_visible_gt_boxes == 3
    assert result.visible_gt_boxes == 4
    assert result.visible_instance_recall == 0.75
    # 两个 anchor 都至少命中；只有 anchor_lamp 被两个轨 ID 命中。未命中 GT 不进分母。
    assert result.fragmented_gt_instances == 1
    assert result.matched_gt_instances == 2
    assert result.fragmentation_rate == 0.5
    assert result.unmatched_predictions == 2
    assert result.false_positives_per_frame == 1.0
    assert [frame.frame_id for frame in result.frames] == ["f1", "f2"]


def test_one_to_one_matching_maximizes_match_count_not_greedy_first_edge():
    gt = {
        "dataset_id": "dev_a_cardinality",
        "frames": [
            {
                "frame_id": "f1",
                "instances": [
                    {"instance_id": "g1", "canonical_id": "c", "bbox": [0, 0, 10, 10]},
                    {"instance_id": "g2", "canonical_id": "c", "bbox": [3, 0, 13, 10]},
                ],
            }
        ],
    }
    predictions = {
        "dataset_id": "dev_a_cardinality",
        "frames": [
            {
                "frame_id": "f1",
                "predictions": [
                    {"track_id": "p1", "canonical_id": "c", "bbox": [0, 0, 10, 10]},
                    {"track_id": "p2", "canonical_id": "c", "bbox": [-2, 0, 8, 10]},
                ],
            }
        ],
    }

    result = evaluate_detection(gt, predictions)
    assert result.visible_instance_recall == 1.0
    assert result.false_positives_per_frame == 0.0
    assert {(match.instance_id, match.track_id) for match in result.frames[0].matches} == {
        ("g1", "p2"),
        ("g2", "p1"),
    }


def test_semantically_identical_array_order_produces_identical_output():
    gt_reordered = copy.deepcopy(_gt())
    predictions_reordered = copy.deepcopy(_predictions())
    gt_reordered["frames"].reverse()
    predictions_reordered["frames"].reverse()
    for frame in gt_reordered["frames"]:
        frame["instances"].reverse()
    for frame in predictions_reordered["frames"]:
        frame["predictions"].reverse()

    expected = evaluate_detection(_gt(), _predictions()).to_dict()
    actual = evaluate_detection(gt_reordered, predictions_reordered).to_dict()
    assert actual == expected


@pytest.mark.parametrize("dataset_id", ["task_b", "release-DEV_B-secret", "x/task_b/y"])
def test_task_b_dataset_ids_are_rejected(dataset_id):
    gt = _gt()
    predictions = _predictions()
    gt["dataset_id"] = dataset_id
    predictions["dataset_id"] = dataset_id
    with pytest.raises(EvaluationInputError, match="forbidden task B"):
        evaluate_detection(gt, predictions)


def test_extra_prediction_frame_is_rejected_but_missing_frame_is_empty():
    predictions = _predictions()
    predictions["frames"] = predictions["frames"][:1]
    result = evaluate_detection(_gt(), predictions)
    assert result.frame_count == 2
    assert result.visible_instance_recall == 0.5

    predictions["frames"].append(
        {"sequence_id": "v1", "frame_id": "not_annotated", "predictions": []}
    )
    with pytest.raises(EvaluationInputError, match="absent from ground truth"):
        evaluate_detection(_gt(), predictions)


def test_prompt_score_and_tie_break_are_deterministic():
    ground_truth = _gt()
    predictions = _predictions()
    ranked = rank_prompt_candidates(
        ground_truth,
        {"z_prompt": predictions, "a_prompt": predictions},
        lambda_fp=0.2,
        mu_fragmentation=0.4,
    )
    assert [item.candidate_id for item in ranked] == ["a_prompt", "z_prompt"]
    assert ranked[0].score == pytest.approx(0.75 - 0.2 * 1.0 - 0.4 * 0.5)
    assert score_prompt(ranked[0].evaluation, lambda_fp=0.2, mu_fragmentation=0.4) == pytest.approx(
        ranked[0].score
    )


def test_cli_json_outputs(tmp_path, capsys):
    gt_path = tmp_path / "gt.json"
    pred_path = tmp_path / "pred.json"
    gt_path.write_text(json.dumps(_gt()), encoding="utf-8")
    pred_path.write_text(json.dumps(_predictions()), encoding="utf-8")

    assert hardval_main([str(gt_path), str(pred_path)]) == 0
    hardval_payload = json.loads(capsys.readouterr().out)
    assert hardval_payload["iou_threshold"] == 0.5
    assert hardval_payload["metrics"]["visible_instance_recall"]["value"] == 0.75

    assert (
        prompt_search_main(
            [
                str(gt_path),
                "--candidate",
                f"z={pred_path}",
                "--candidate",
                f"a={pred_path}",
                "--lambda-fp",
                "0.2",
                "--mu-fragmentation",
                "0.4",
            ]
        )
        == 0
    )
    prompt_payload = json.loads(capsys.readouterr().out)
    assert [item["candidate_id"] for item in prompt_payload["ranking"]] == ["a", "z"]
