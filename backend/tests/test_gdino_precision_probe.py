import numpy as np

from scripts.gdino_torch_precision_bench import (
    _raw_diff,
    _raw_outputs_have_valid_nonfinite_patterns,
)


def test_identical_nonfinite_outputs_do_not_pass_finite_gate():
    outputs = {
        "logits": np.asarray([[0.0, np.nan, -np.inf]], dtype=np.float32),
        "pred_boxes": np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
    }

    report = _raw_diff(outputs, outputs)

    assert report["logits"]["nonfinite_pattern_equal"] is True
    assert report["logits"]["reference_all_finite"] is False
    assert _raw_outputs_have_valid_nonfinite_patterns(report) is False


def test_finite_equal_outputs_pass_finite_gate():
    outputs = {
        "logits": np.asarray([[0.0, 1.0]], dtype=np.float32),
        "pred_boxes": np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
    }

    assert _raw_outputs_have_valid_nonfinite_patterns(_raw_diff(outputs, outputs)) is True


def test_matching_negative_infinity_mask_is_allowed():
    outputs = {
        "logits": np.asarray([[0.0, -np.inf]], dtype=np.float32),
        "pred_boxes": np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
    }

    assert (
        _raw_outputs_have_valid_nonfinite_patterns(_raw_diff(outputs, outputs))
        is True
    )
