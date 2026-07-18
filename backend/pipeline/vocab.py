"""Detection vocabulary loading, validation, and prompt-batch compilation.

The vocabulary deliberately keeps three identities separate:

* ``canonical_id`` is a detector concept and merges prompt aliases.
* ``category_id`` is the broader semantic class consumed by S3.
* ``anchor_id`` belongs to ground truth only and is rejected here.

Compiled prompts retain explicit batch boundaries.  ``CompiledPrompts`` is a
``list[str]`` subclass so it can pass through the existing ingest interface
unchanged while :mod:`backend.pipeline.detect` can still recover the plan.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


def normalize_label(value: str) -> str:
    """Return the stable comparison form used for prompt/label matching."""

    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def normalize_display_label(value: str) -> str:
    """Normalize multilingual display text without deleting CJK characters."""

    return " ".join(re.sub(r"[\W_]+", " ", value.casefold(), flags=re.UNICODE).split())


@dataclass(frozen=True)
class VocabularyEntry:
    canonical_id: str
    category_id: str
    display_label_zh: str
    detection_prompts: tuple[str, ...]
    confusable_group: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class VocabMatch:
    canonical_id: str | None
    category_id: str | None


@dataclass(frozen=True)
class VocabTranscription:
    """Auditable reconciliation of detector and VLM label candidates."""

    canonical_id: str | None
    category_id: str | None
    confidence: float
    status: str
    evidence: tuple[str, ...] = ()

    @property
    def match(self) -> VocabMatch:
        return VocabMatch(self.canonical_id, self.category_id)


@dataclass(frozen=True)
class DetectionPrompt:
    text: str
    canonical_id: str
    category_id: str
    confusable_group: str | None


class CompiledPrompts(list[str]):
    """Flat prompt list carrying an immutable, explicit batching plan."""

    def __init__(self, prompt_batches: tuple[tuple[DetectionPrompt, ...], ...]):
        batches = tuple(tuple(prompt.text for prompt in batch) for batch in prompt_batches)
        super().__init__(prompt for batch in batches for prompt in batch)
        self.prompt_batches = prompt_batches
        self.batches = batches
        self.prompt_to_canonical: Mapping[str, str] = MappingProxyType(
            {
                normalize_label(prompt.text): prompt.canonical_id
                for batch in prompt_batches
                for prompt in batch
            }
        )
        self.prompt_to_category: Mapping[str, str] = MappingProxyType(
            {
                normalize_label(prompt.text): prompt.category_id
                for batch in prompt_batches
                for prompt in batch
            }
        )


class Vocabulary:
    """Validated inference vocabulary with no instance-level truth IDs."""

    def __init__(
        self,
        entries: tuple[VocabularyEntry, ...],
        *,
        version: str = "",
        schema_version: str = "1.0",
        batch_size: int = 4,
    ):
        if not entries:
            raise ValueError("vocab entries must not be empty")
        if batch_size <= 0:
            raise ValueError("vocab batch_size must be positive")

        self.entries = entries
        self.version = version
        self.schema_version = schema_version
        self.batch_size = batch_size

        canonical_ids: set[str] = set()
        aliases: dict[str, VocabMatch] = {}
        display_aliases: dict[str, set[VocabMatch]] = {}
        for entry in entries:
            if entry.canonical_id in canonical_ids:
                raise ValueError(f"duplicate canonical_id in vocab: {entry.canonical_id}")
            canonical_ids.add(entry.canonical_id)
            match = VocabMatch(entry.canonical_id, entry.category_id)
            display = normalize_display_label(entry.display_label_zh)
            if display:
                display_aliases.setdefault(display, set()).add(match)
            for prompt in entry.detection_prompts:
                alias = normalize_label(prompt)
                previous = aliases.get(alias)
                if previous is not None and previous != match:
                    raise ValueError(f"ambiguous detection prompt in vocab: {prompt}")
                aliases[alias] = match

        # Downstream artifacts use canonical labels, so canonical IDs resolve too.
        for entry in entries:
            key = normalize_label(entry.canonical_id)
            match = VocabMatch(entry.canonical_id, entry.category_id)
            previous = aliases.get(key)
            if previous is not None and previous != match:
                raise ValueError(f"canonical_id collides with a detection prompt: {entry.canonical_id}")
            aliases[key] = match
        self._aliases: Mapping[str, VocabMatch] = MappingProxyType(aliases)
        self._display_aliases: Mapping[str, frozenset[VocabMatch]] = MappingProxyType(
            {key: frozenset(value) for key, value in display_aliases.items()}
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "Vocabulary":
        return load_vocabulary(path)

    def match(self, raw_label: str) -> VocabMatch:
        """Resolve exact aliases and unambiguous GDINO compound labels."""

        label = normalize_label(raw_label)
        exact = self._aliases.get(label)
        if exact is not None:
            return exact

        hits = self._compound_matches(label)
        if len(hits) == 1:
            return next(iter(hits))
        return VocabMatch(canonical_id=None, category_id=None)

    def _compound_matches(self, normalized_label: str) -> frozenset[VocabMatch]:
        if not normalized_label:
            return frozenset()
        hits = {
            match
            for alias, match in self._aliases.items()
            if alias
            and (
                re.search(rf"(?:^| ){re.escape(alias)}(?: |$)", normalized_label)
                or re.search(rf"(?:^| ){re.escape(normalized_label)}(?: |$)", alias)
            )
        }
        return frozenset(hits)

    def transcribe(self, *raw_labels: str | None) -> VocabTranscription:
        """Reconcile detector/VLM names into one schema identity.

        Exact detector aliases and canonical IDs are strongest.  Exact
        multilingual display labels are accepted as schema evidence, while the
        historical unique compound-label matcher remains a lower-confidence
        fallback.  Conflicting candidates fail closed instead of silently
        choosing one side.
        """

        candidates: dict[VocabMatch, tuple[float, list[str]]] = {}
        ambiguous_evidence: list[str] = []
        for raw in raw_labels:
            if not isinstance(raw, str) or not raw.strip():
                continue
            value = raw.strip()
            english = normalize_label(value)
            display = normalize_display_label(value)
            exact = self._aliases.get(english) if english else None
            confidence = 1.0
            source = "exact_alias"
            if exact is None and display:
                display_hits = self._display_aliases.get(display, frozenset())
                if len(display_hits) == 1:
                    exact = next(iter(display_hits))
                    confidence = 0.95
                    source = "exact_display_label"
                elif len(display_hits) > 1:
                    ambiguous_evidence.append(f"ambiguous_display_label:{value}")
            if exact is None and english:
                compound_hits = self._compound_matches(english)
                if len(compound_hits) == 1:
                    exact = next(iter(compound_hits))
                    confidence = 0.80
                    source = "unique_compound_alias"
                elif len(compound_hits) > 1:
                    ambiguous_evidence.append(f"ambiguous_compound_alias:{value}")
            if exact is None:
                continue
            best, evidence = candidates.setdefault(exact, (confidence, []))
            evidence.append(f"{source}:{value}")
            candidates[exact] = (max(best, confidence), evidence)

        if ambiguous_evidence:
            evidence = tuple(
                ambiguous_evidence
                + [
                    item
                    for _, (_, items) in sorted(
                        candidates.items(),
                        key=lambda row: (row[0].category_id or "", row[0].canonical_id or ""),
                    )
                    for item in items
                ]
            )
            return VocabTranscription(None, None, 0.0, "conflict", evidence)
        if not candidates:
            return VocabTranscription(None, None, 0.0, "unmapped")
        if len(candidates) > 1:
            evidence = tuple(
                item
                for _, (_, items) in sorted(
                    candidates.items(), key=lambda row: (row[0].category_id or "", row[0].canonical_id or "")
                )
                for item in items
            )
            return VocabTranscription(None, None, 0.0, "conflict", evidence)
        match, (confidence, evidence) = next(iter(candidates.items()))
        return VocabTranscription(
            match.canonical_id,
            match.category_id,
            confidence,
            "mapped",
            tuple(evidence),
        )

    def compile(self, batch_size: int | None = None) -> CompiledPrompts:
        return compile_prompt_batches(self, batch_size=batch_size)


def _require_string(raw: Mapping[str, Any], field: str, *, allow_empty: bool = False) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"vocab field {field!r} must be a non-empty string")
    return value.strip()


def _contains_anchor_id(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key == "anchor_id" or _contains_anchor_id(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_anchor_id(item) for item in value)
    return False


def load_vocabulary(path: str | Path) -> Vocabulary:
    """Load and strictly validate a vocabulary JSON file."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("vocab root must be an object")
    if _contains_anchor_id(raw):
        raise ValueError("anchor_id is ground-truth-only and forbidden in inference vocab")

    raw_entries = raw.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("vocab must contain a non-empty entries list")

    entries: list[VocabularyEntry] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("each vocab entry must be an object")
        prompts = raw_entry.get("detection_prompts")
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("detection_prompts must be a non-empty list")
        cleaned_prompts: list[str] = []
        for prompt in prompts:
            if not isinstance(prompt, str) or not normalize_label(prompt):
                raise ValueError("each detection prompt must be a non-empty string")
            cleaned_prompts.append(prompt.strip())
        if len({normalize_label(prompt) for prompt in cleaned_prompts}) != len(cleaned_prompts):
            raise ValueError("detection_prompts must be unique within an entry")

        confusable_group = raw_entry.get("confusable_group")
        if confusable_group is not None:
            if not isinstance(confusable_group, str) or not confusable_group.strip():
                raise ValueError("confusable_group must be null or a non-empty string")
            confusable_group = confusable_group.strip()
        notes = raw_entry.get("notes", "")
        if not isinstance(notes, str):
            raise ValueError("notes must be a string")

        entries.append(
            VocabularyEntry(
                canonical_id=_require_string(raw_entry, "canonical_id"),
                category_id=_require_string(raw_entry, "category_id"),
                display_label_zh=_require_string(raw_entry, "display_label_zh"),
                detection_prompts=tuple(cleaned_prompts),
                confusable_group=confusable_group,
                notes=notes.strip(),
            )
        )

    batch_size = raw.get("batch_size", 4)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise ValueError("vocab batch_size must be an integer")
    schema_version = raw.get("schema_version", "1.0")
    version = raw.get("vocab_version", "")
    if not isinstance(schema_version, str) or not isinstance(version, str):
        raise ValueError("schema_version and vocab_version must be strings")
    return Vocabulary(
        tuple(entries),
        version=version,
        schema_version=schema_version,
        batch_size=batch_size,
    )


# Short spelling for callers that mirror the CLI flag.
load_vocab = load_vocabulary


def _compatible(prompt: DetectionPrompt, batch: list[DetectionPrompt]) -> bool:
    for existing in batch:
        if existing.canonical_id == prompt.canonical_id:
            return False
        if (
            prompt.confusable_group is not None
            and prompt.confusable_group == existing.confusable_group
        ):
            return False
    return True


def compile_prompt_batches(
    vocabulary: Vocabulary, batch_size: int | None = None
) -> CompiledPrompts:
    """Greedily pack prompts while separating aliases and confusable concepts.

    First-fit packing is deterministic in vocabulary declaration order.  A new
    batch is opened whenever every existing batch is full or conflicts with the
    prompt's canonical/confusable group.
    """

    size = vocabulary.batch_size if batch_size is None else batch_size
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError("prompt batch_size must be a positive integer")

    batches: list[list[DetectionPrompt]] = []
    for entry in vocabulary.entries:
        for text in entry.detection_prompts:
            prompt = DetectionPrompt(
                text=text,
                canonical_id=entry.canonical_id,
                category_id=entry.category_id,
                confusable_group=entry.confusable_group,
            )
            for batch in batches:
                if len(batch) < size and _compatible(prompt, batch):
                    batch.append(prompt)
                    break
            else:
                batches.append([prompt])

    return CompiledPrompts(tuple(tuple(batch) for batch in batches))
