"""S3 配置、词表与 v5 Tracklet 特征读取。

v5 只保存每条轨迹的 DINOv2 Top-3 均值向量和 raw detector label；属性、
上下文与逐视角向量尚未生成。因此本模块显式保留缺失值，不能把 baseline
包装成完整特征消融。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import yaml

from backend.pipeline.vocab import Vocabulary
from backend.schemas.core import Observation, Tracklet


@dataclass(frozen=True)
class WeightConfig:
    instance: float
    semantic: float
    attribute: float
    context: float
    geometry: float

    @property
    def total(self) -> float:
        return self.instance + self.semantic + self.attribute + self.context + self.geometry


@dataclass(frozen=True)
class ThresholdConfig:
    match: float
    new: float
    margin: float
    min_quality: float


@dataclass(frozen=True)
class StitchConfig:
    """同视频短轨 stitch。enabled=False 时 S3 行为与历史基线逐字节一致。"""

    enabled: bool = False
    min_cosine: float = 0.90
    max_gap_ms: int = 0  # 0 = 不限制片段间时间间隔


@dataclass(frozen=True)
class FilterConfig:
    """低证据轨标记:观测数不足的轨保留匹配与自动链接资格,但不进澄清队列。"""

    min_observations: int = 1  # 1 = 不标记


@dataclass(frozen=True)
class ClarifyConfig:
    """澄清请求封顶:每条轨在每个视频对里最多保留的歧义伙伴数。"""

    max_partners_per_tracklet: int = 0  # 0 = 不封顶


@dataclass(frozen=True)
class ReIDConfig:
    version: str
    embedding_dim: int
    top_k: int
    weights: WeightConfig
    thresholds: ThresholdConfig
    stitch: StitchConfig = StitchConfig()
    filter: FilterConfig = FilterConfig()
    clarify: ClarifyConfig = ClarifyConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ReIDConfig":
        raw = yaml.safe_load(Path(path).read_text())
        weights = WeightConfig(**raw["weights"])
        thresholds = ThresholdConfig(**raw["thresholds"])
        config = cls(
            version=str(raw["version"]),
            embedding_dim=int(raw["embedding_dim"]),
            top_k=int(raw["top_k"]),
            weights=weights,
            thresholds=thresholds,
            stitch=StitchConfig(**raw.get("stitch", {})),
            filter=FilterConfig(**raw.get("filter", {})),
            clarify=ClarifyConfig(**raw.get("clarify", {})),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.weights.total <= 0:
            raise ValueError("at least one ReID weight must be positive")
        if any(value < 0 for value in self.weights.__dict__.values()):
            raise ValueError("ReID weights cannot be negative")
        t = self.thresholds
        if not 0 <= t.new < t.match <= 1:
            raise ValueError("thresholds must satisfy 0 <= new < match <= 1")
        if not 0 <= t.margin <= 1 or not 0 <= t.min_quality <= 1:
            raise ValueError("margin/min_quality must be in [0, 1]")
        if not 0 <= self.stitch.min_cosine <= 1:
            raise ValueError("stitch.min_cosine must be in [0, 1]")
        if self.stitch.max_gap_ms < 0:
            raise ValueError("stitch.max_gap_ms cannot be negative")
        if self.filter.min_observations < 1:
            raise ValueError("filter.min_observations must be >= 1")
        if self.clarify.max_partners_per_tracklet < 0:
            raise ValueError("clarify.max_partners_per_tracklet cannot be negative")


@dataclass(frozen=True)
class TrackFeature:
    tracklet: Tracklet
    vector: tuple[float, ...]
    raw_label: str
    canonical_id: str | None
    category_id: str | None
    quality: float
    aspect_ratio: float | None
    area: float | None
    timestamps_ms: tuple[int, ...] = ()  # 已链接观测的时间戳,升序;stitch 共现否决用

    @property
    def tracklet_id(self) -> str:
        return self.tracklet.tracklet_id

    @property
    def video_id(self) -> str:
        return self.tracklet.video_id

    @property
    def observation_count(self) -> int:
        return len(self.timestamps_ms)


def _resolve_ref(ref: str, ingest_root: Path) -> Path:
    path = Path(ref)
    if path.is_absolute() and path.exists():
        return path
    for base in (Path.cwd(), ingest_root, *ingest_root.parents):
        candidate = base / path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"cannot resolve artifact reference: {ref}")


def _read_jsonl(path: Path, model_type):
    if not path.exists():
        return []
    return [model_type.model_validate_json(line) for line in path.read_text().splitlines() if line]


def _load_vector(ref: str, ingest_root: Path, expected_dim: int) -> tuple[float, ...]:
    path = _resolve_ref(ref, ingest_root)
    raw = json.loads(path.read_text())
    values = tuple(float(value) for value in raw["vector"])
    if len(values) != expected_dim:
        raise ValueError(f"{path}: expected {expected_dim} dims, got {len(values)}")
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"{path}: invalid embedding norm {norm}")
    return tuple(value / norm for value in values)


# S5 属性接线:只有这些键参与 _attribute_score 比较(白名单,杜绝流水线元数据
# 混入打分的 hero_scoring_version 一类事故);命名键(label_en/label_zh)只供展示。
COMPARABLE_ATTRIBUTE_KEYS = (
    "color_primary",
    "color_secondary",
    "material",
    "pattern",
    "shape",
    "text_marks",
)
# 归一化后视为"未知/缺失"的值 —— 不进分子也不进分母(missing/unknown 语义)。
UNKNOWN_ATTRIBUTE_VALUES = frozenset({"", "unknown", "none", "null"})


def load_attribute_enrichment(path: str | Path) -> dict[str, dict[str, str]]:
    """读 S5 属性抽取产物 JSONL({tracklet_id, attributes:{...}}),只保留可比键。"""

    enrichment: dict[str, dict[str, str]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        attrs = {
            key: str(value)
            for key, value in (row.get("attributes") or {}).items()
            if key in COMPARABLE_ATTRIBUTE_KEYS
        }
        if attrs:
            enrichment[str(row["tracklet_id"])] = attrs
    return enrichment


def load_features(
    ingest_root: str | Path,
    *,
    vocab: Vocabulary,
    embedding_dim: int,
    attributes: dict[str, dict[str, str]] | None = None,
) -> list[TrackFeature]:
    """读取每个视频目录的 Tracklet、Observation 与嵌入，顺序确定。

    ``attributes``(tracklet_id → 可比属性键值)在读入时合并进
    ``tracklet.attributes``;stitch 发生在其后,合并组自动继承 hero 成员属性。
    """

    root = Path(ingest_root)
    if not root.is_dir():
        raise FileNotFoundError(f"ingest root not found: {root}")

    features: list[TrackFeature] = []
    for tracklet_path in sorted(root.glob("*/tracklets.jsonl")):
        observations = {
            observation.observation_id: observation
            for observation in _read_jsonl(tracklet_path.parent / "observations.jsonl", Observation)
        }
        for tracklet in _read_jsonl(tracklet_path, Tracklet):
            if not tracklet.embedding_ref:
                continue
            if attributes and tracklet.tracklet_id in attributes:
                tracklet.attributes.update(attributes[tracklet.tracklet_id])
            vector = _load_vector(tracklet.embedding_ref, root, embedding_dim)
            raw_label = str(tracklet.attributes.get("label", ""))
            vocab_match = vocab.match(raw_label)
            linked = [observations[ref] for ref in tracklet.observation_ids if ref in observations]
            quality = max((observation.quality for observation in linked), default=0.0)
            timestamps = tuple(sorted(observation.timestamp_ms for observation in linked))
            ratios: list[float] = []
            areas: list[float] = []
            for observation in linked:
                x1, y1, x2, y2 = observation.bbox
                width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
                if width > 0 and height > 0:
                    ratios.append(width / height)
                    areas.append(width * height)
            features.append(
                TrackFeature(
                    tracklet=tracklet,
                    vector=vector,
                    raw_label=raw_label,
                    canonical_id=vocab_match.canonical_id,
                    category_id=vocab_match.category_id,
                    quality=quality,
                    aspect_ratio=median(ratios) if ratios else None,
                    area=median(areas) if areas else None,
                    timestamps_ms=timestamps,
                )
            )
    features.sort(key=lambda feature: feature.tracklet_id)
    return features
