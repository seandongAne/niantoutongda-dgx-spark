"""Canonical detection vocabulary and prompt compiler tests."""

import json
from pathlib import Path

import pytest

from backend.pipeline.vocab import VocabMatch, load_vocabulary


VOCAB_PATH = Path(__file__).parents[2] / "fixtures" / "dev_a" / "vocab.json"


def test_dev_a_vocab_layers_and_aliases():
    vocab = load_vocabulary(VOCAB_PATH)

    assert len(vocab.entries) == 15
    assert sum(len(entry.detection_prompts) for entry in vocab.entries) == 17
    assert vocab.match("smart speaker") == VocabMatch("night_light", "lamp")
    assert vocab.match("cylinder lamp") == VocabMatch("night_light", "lamp")
    assert vocab.match("night_light") == VocabMatch("night_light", "lamp")
    assert vocab.match("toy storage organizer") == VocabMatch(
        "toy_storage_organizer", "cabinet"
    )
    assert vocab.match("wardrobe") == VocabMatch("wardrobe", "cabinet")
    assert vocab.match("smart speaker cylinder lamp") == VocabMatch("night_light", "lamp")
    assert vocab.match("smart speaker table lamp") == VocabMatch(None, None)
    # GDINO text_labels 偶尔只返回 prompt 的唯一 token 子串。
    assert vocab.match("camera") == VocabMatch("security_camera", "camera")
    assert vocab.match("animal") == VocabMatch("stuffed_animal", "stuffed_animal")
    assert vocab.match("lamp") == VocabMatch(None, None)  # night/table lamp 间仍有歧义


def test_transcriber_reconciles_detector_vlm_and_multilingual_labels():
    vocab = load_vocabulary(VOCAB_PATH)

    exact = vocab.transcribe("smart speaker", "夜灯")
    assert exact.match == VocabMatch("night_light", "lamp")
    assert exact.confidence == 1.0
    assert exact.status == "mapped"
    assert len(exact.evidence) == 2

    display_only = vocab.transcribe("夜灯")
    assert display_only.match == VocabMatch("night_light", "lamp")
    assert display_only.confidence == 0.95

    compound = vocab.transcribe("detected smart speaker cylinder lamp")
    assert compound.match == VocabMatch("night_light", "lamp")
    assert compound.confidence == 0.8


def test_transcriber_fails_closed_on_cross_source_conflict():
    vocab = load_vocabulary(VOCAB_PATH)

    result = vocab.transcribe("smart speaker", "table lamp")

    assert result.match == VocabMatch(None, None)
    assert result.confidence == 0.0
    assert result.status == "conflict"

    ambiguous_compound = vocab.transcribe(
        "smart speaker", "smart speaker table lamp"
    )
    assert ambiguous_compound.match == VocabMatch(None, None)
    assert ambiguous_compound.status == "conflict"
    assert any(
        item.startswith("ambiguous_compound_alias:")
        for item in ambiguous_compound.evidence
    )


def test_compiler_separates_aliases_and_confusable_concepts():
    compiled = load_vocabulary(VOCAB_PATH).compile()

    assert len(compiled) == 17
    assert all(1 <= len(batch) <= 4 for batch in compiled.prompt_batches)
    for batch in compiled.prompt_batches:
        canonical_ids = [prompt.canonical_id for prompt in batch]
        assert len(canonical_ids) == len(set(canonical_ids))
        groups = [prompt.confusable_group for prompt in batch if prompt.confusable_group]
        assert len(groups) == len(set(groups))

    prompt_batch = {
        prompt.text: batch_index
        for batch_index, batch in enumerate(compiled.prompt_batches)
        for prompt in batch
    }
    assert prompt_batch["smart speaker"] != prompt_batch["cylinder lamp"]
    assert prompt_batch["luggage"] != prompt_batch["laundry bag"]
    assert prompt_batch["water bottle"] != prompt_batch["tumbler"]


def test_loader_rejects_truth_anchor_id(tmp_path):
    raw = json.loads(VOCAB_PATH.read_text())
    raw["entries"][0]["anchor_id"] = "anchor_08"
    path = tmp_path / "leaky-vocab.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="ground-truth-only"):
        load_vocabulary(path)


def test_loader_rejects_alias_assigned_to_two_concepts(tmp_path):
    raw = json.loads(VOCAB_PATH.read_text())
    raw["entries"][1]["detection_prompts"] = ["desk"]
    path = tmp_path / "ambiguous-vocab.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="ambiguous detection prompt"):
        load_vocabulary(path)
