import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.reid_multiview_embed import (
    FORMAT_VERSION,
    PrototypeRecord,
    _valid_existing_artifact,
    embed_records,
    input_sha256,
    load_prototype_records,
    write_artifact,
)


class FakeBatchEmbedder:
    model_version = "fake-batch@test"

    def __init__(self):
        self.calls = []

    def embed_many(self, image_paths, *, batch_size=32):
        self.calls.append((tuple(image_paths), batch_size))
        return [[float(index + 1), 1.0] for index, _ in enumerate(image_paths)]


def _image(path: Path, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color).save(path)


def test_load_records_sorts_and_reports_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "results" / "ingest"
    crop_a = root / "video_a" / "evidence" / "a.jpg"
    crop_b = root / "video_b" / "evidence" / "b.jpg"
    _image(crop_a, "red")
    _image(crop_b, "blue")
    for video, rows in (
        (
            "video_b",
            [
                {
                    "tracklet_id": "video_b_t002",
                    "prototype_refs": [
                        str(crop_b.relative_to(tmp_path)),
                        "missing.jpg",
                    ],
                }
            ],
        ),
        (
            "video_a",
            [
                {
                    "tracklet_id": "video_a_t001",
                    "prototype_refs": [str(crop_a.relative_to(tmp_path))],
                }
            ],
        ),
    ):
        path = root / video / "tracklets.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    records, missing, sources = load_prototype_records(root)

    assert [(row.tracklet_id, row.view_index) for row in records] == [
        ("video_a_t001", 0),
        ("video_b_t002", 0),
    ]
    assert [(row.tracklet_id, row.view_index) for row in missing] == [
        ("video_b_t002", 1)
    ]
    assert [source["path"] for source in sources] == [
        "video_a/tracklets.jsonl",
        "video_b/tracklets.jsonl",
    ]


def test_embed_records_batches_and_normalizes(tmp_path):
    records = [
        PrototypeRecord(f"t{index}", 0, f"{index}.jpg", tmp_path / f"{index}.jpg")
        for index in range(5)
    ]
    embedder = FakeBatchEmbedder()

    vectors = embed_records(records, embedder, batch_size=2)

    assert vectors.dtype == np.float32
    assert vectors.shape == (5, 2)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)
    assert [call[1] for call in embedder.calls] == [2, 2, 1]
    assert [len(call[0]) for call in embedder.calls] == [2, 2, 1]


def test_write_artifact_is_pickle_free_and_hash_validated(tmp_path):
    crop_a = tmp_path / "a.jpg"
    crop_b = tmp_path / "b.jpg"
    _image(crop_a, "red")
    _image(crop_b, "blue")
    records = [
        PrototypeRecord("same_track", 0, "a.jpg", crop_a),
        PrototypeRecord("same_track", 1, "b.jpg", crop_b),
    ]
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    inputs_hash = input_sha256(records, [])
    output = tmp_path / "views.npz"
    manifest = tmp_path / "views.manifest.json"

    payload = write_artifact(
        output,
        manifest,
        records,
        vectors,
        model_version="fake-batch@test",
        source_files=[],
        missing=[],
        inputs_sha256=inputs_hash,
    )

    with np.load(output, allow_pickle=False) as data:
        assert str(data["format_version"].item()) == FORMAT_VERSION
        assert data["tracklet_ids"].tolist() == ["same_track", "same_track"]
        assert data["view_index"].tolist() == [0, 1]
        assert data["vectors"].dtype == np.float32
    assert payload == json.loads(manifest.read_text())
    assert _valid_existing_artifact(
        output,
        manifest,
        model_version="fake-batch@test",
        inputs_sha256=inputs_hash,
    ) == payload

    output.write_bytes(output.read_bytes() + b"corrupt")
    assert (
        _valid_existing_artifact(
            output,
            manifest,
            model_version="fake-batch@test",
            inputs_sha256=inputs_hash,
        )
        is None
    )
