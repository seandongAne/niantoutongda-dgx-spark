import json
from pathlib import Path

import numpy as np
import pytest

from scripts.reid_multiview_proxy_sweep import (
    FORBIDDEN_GT,
    _all_set_similarities,
    load_views,
    reject_frozen_gt_paths,
    run_proxy_sweep,
)
from backend.tools.reid.multiview import set_similarity
from backend.tools.sf1.projection import NumpyProjectionHead, sha256_file


def _write_views(path: Path, *, plural_index: bool = False) -> None:
    rows = {
        "v1_ta": ([1, 0, 0, 0], [0, 1, 0, 0]),
        "v2_ta": ([0, 1, 0, 0], [1, 0, 0, 0]),
        "v1_tb": ([0, 0, 1, 0], [0, 0, 0, 1]),
        "v2_tb": ([0, 0, 0, 1], [0, 0, 1, 0]),
    }
    ids, indices, vectors = [], [], []
    for tracklet_id, view_set in rows.items():
        for index, vector in enumerate(view_set):
            ids.append(tracklet_id)
            indices.append(index)
            vectors.append(vector)
    payload = {
        "format_version": np.asarray("reid-multiview-embeddings-v1"),
        "tracklet_ids": np.asarray(ids),
        "vectors": np.asarray(vectors, dtype=np.float32),
        ("view_indices" if plural_index else "view_index"): np.asarray(
            indices, dtype=np.uint16
        ),
    }
    np.savez_compressed(path, **payload)


def _write_tutor(path: Path) -> None:
    rows = [
        ("v1_ta", "v2_ta", True),
        ("v1_tb", "v2_tb", True),
        ("v1_ta", "v2_tb", False),
        ("v1_tb", "v2_ta", False),
        # Same-video evidence is deliberately outside the cross-video proxy.
        ("v1_ta", "v1_tb", False),
    ]
    path.write_text(
        "".join(
            json.dumps(
                {
                    "tracklet_a": a,
                    "tracklet_b": b,
                    "tutor": {"same": same, "confidence": 0.99},
                }
            )
            + "\n"
            for a, b, same in rows
        ),
        encoding="utf-8",
    )


def _write_pseudo(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "_provenance": "synthetic proxy labels, not GT",
                "entities": [
                    {
                        "anchor_id": "pseudo_a",
                        "confirmed_tracklet_ids_by_video": {
                            "v1": ["v1_ta"],
                            "v2": ["v2_ta"],
                        },
                    },
                    {
                        "anchor_id": "pseudo_b",
                        "confirmed_tracklet_ids_by_video": {
                            "v1": ["v1_tb"],
                            "v2": ["v2_tb"],
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_load_views_accepts_both_index_spellings_and_normalizes(tmp_path):
    singular = tmp_path / "singular.npz"
    plural = tmp_path / "plural.npz"
    _write_views(singular)
    _write_views(plural, plural_index=True)

    first = load_views(singular)
    second = load_views(plural)

    assert first == second
    assert sorted(first) == ["v1_ta", "v1_tb", "v2_ta", "v2_tb"]
    assert all(
        abs(sum(value * value for value in vector) - 1.0) < 1e-6
        for view_set in first.values()
        for vector in view_set
    )


def test_vectorized_proxy_formulas_match_production_scorer():
    left = ((1.0, 0.0, 0.0), (0.0, 0.8, 0.6), (0.5, 0.5, 0.70710678))
    right = ((0.9, 0.4358899, 0.0), (0.0, 0.6, 0.8))

    values = _all_set_similarities(left, right, max_views=6)

    for method, value in values.items():
        assert value == pytest.approx(
            set_similarity(left, right, method=method, max_views=6), abs=1e-6
        )


def test_proxy_sweep_is_deterministic_and_auditable(tmp_path):
    views = tmp_path / "views.npz"
    tutor = tmp_path / "tutor.jsonl"
    pseudo = tmp_path / "pseudo.json"
    _write_views(views)
    _write_tutor(tutor)
    _write_pseudo(pseudo)

    kwargs = {
        "views_path": views,
        "tutor_pairs_path": tutor,
        "pseudo_labels_path": pseudo,
        "holdout_modulus": 2,
        "holdout_bucket": 0,
    }
    first = run_proxy_sweep(**kwargs)
    second = run_proxy_sweep(**kwargs)

    assert first == second
    assert first["scope"] == "AUTOTUNE_MULTIVIEW_PROXY_ONLY_NO_FROZEN_GT"
    assert first["frozen_gt_policy"]["read"] is False
    assert len(first["grid"]) == 9
    assert first["winner"]["method"] in {
        "max_pair",
        "mean_chamfer",
        "symmetric_top2",
    }
    assert first["winner"]["space"] == "raw"
    assert first["winner"]["blend"] in {0.25, 0.5, 0.75}
    assert first["winner"]["pseudo_retrieval"]["recall_at_1"] == 1.0
    assert first["counts"]["tutor"]["same_video"] == 1
    assert first["counts"]["pseudo_split"]["queries"] == 2
    assert first["inputs"]["views"]["sha256"]


def test_proxy_sweep_compares_raw_and_projected_spaces(tmp_path):
    views = tmp_path / "views.npz"
    tutor = tmp_path / "tutor.jsonl"
    pseudo = tmp_path / "pseudo.json"
    projection = tmp_path / "projection.npz"
    _write_views(views)
    _write_tutor(tutor)
    _write_pseudo(pseudo)
    NumpyProjectionHead(
        weight1=np.eye(4, dtype=np.float32),
        bias1=np.zeros(4, dtype=np.float32),
        weight2=np.eye(4, dtype=np.float32),
        bias2=np.zeros(4, dtype=np.float32),
        mode="plain",
    ).save(projection)

    report = run_proxy_sweep(
        views_path=views,
        tutor_pairs_path=tutor,
        pseudo_labels_path=pseudo,
        projection_path=projection,
        projection_sha256=sha256_file(projection),
        holdout_modulus=2,
        holdout_bucket=0,
    )

    assert report["parameters"]["spaces"] == ["raw", "projected"]
    assert set(report["baseline"]) == {"raw", "projected"}
    assert len(report["grid"]) == 18
    assert report["winner"]["space"] == "projected"
    assert report["projection"]["output_dim"] == 4
    assert report["projection"]["sha256"] == sha256_file(projection)


def test_frozen_gt_path_is_rejected_before_any_read(tmp_path, monkeypatch):
    touched = []
    original = Path.read_text

    def recording_read(path, *args, **kwargs):
        touched.append(path.resolve())
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", recording_read)

    with pytest.raises(ValueError, match="frozen hero GT is forbidden"):
        reject_frozen_gt_paths([FORBIDDEN_GT])

    assert touched == []


def test_run_rejects_gt_even_when_passed_as_a_proxy_argument(tmp_path):
    views = tmp_path / "views.npz"
    _write_views(views)

    with pytest.raises(ValueError, match="frozen hero GT is forbidden"):
        run_proxy_sweep(
            views_path=views,
            tutor_pairs_path=FORBIDDEN_GT,
            pseudo_labels_path=tmp_path / "unused.json",
        )
