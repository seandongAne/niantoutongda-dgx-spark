"""S3 配置、词表与 Tracklet 特征读取。

baseline 保存每条轨迹的 DINOv2 Top-3 均值向量。可选的 multiview
侧车产物另存逐 crop 向量，只在首轮均值向量 Top-K 召回后重排，避免
改变 baseline 的候选召回边界。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
import yaml

from backend.pipeline.vocab import Vocabulary
from backend.schemas.core import Observation, Tracklet
from backend.tools.reid.multiview import ARTIFACT_FORMAT_VERSION
from backend.tools.reid.neighborhood import (
    CONTEXT_FORMAT_VERSION,
    LEGACY_CONTEXT_FORMAT_VERSION,
    SUPPORTED_CONTEXT_FORMAT_VERSIONS,
)
from backend.tools.sf1.projection import NumpyProjectionHead


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
class ProjectionConfig:
    """SF1-L1 冻结投影头；启用时 artifact 与 sha256 缺一不可。"""

    enabled: bool = False
    artifact: str = ""
    sha256: str = ""


@dataclass(frozen=True)
class MultiViewConfig:
    """逐 crop 集合重排。

    artifact 是 ``reid-multiview-embeddings-v1`` NPZ；候选仍由原均值向量 Top-K
    决定，只对候选并集重算 instance 分，所以成本有界且可回放。
    """

    enabled: bool = False
    artifact: str = ""
    sha256: str = ""
    method: str = "symmetric_top2"
    blend: float = 1.0
    space: str = "raw"  # raw / projected
    calibration: str = "per_video_pair_quantile"
    max_views_per_rep: int = 6


@dataclass(frozen=True)
class NeighborhoodConfig:
    """冻结的静态邻域 sidecar，只在 baseline Top-K 候选内重排。"""

    enabled: bool = False
    artifact: str = ""
    sha256: str = ""
    blend: float = 0.10
    calibration: str = "per_video_pair_quantile"
    scope: str = "uncertain_only"
    confidence_weighting: bool = False


@dataclass(frozen=True)
class CycleConfig:
    """两条独立边支撑的三视频组件闭环。"""

    enabled: bool = False


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
    projection: ProjectionConfig = ProjectionConfig()
    multiview: MultiViewConfig = MultiViewConfig()
    neighborhood: NeighborhoodConfig = NeighborhoodConfig()
    cycle: CycleConfig = CycleConfig()

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
            projection=ProjectionConfig(**raw.get("projection", {})),
            multiview=MultiViewConfig(**raw.get("multiview", {})),
            neighborhood=NeighborhoodConfig(**raw.get("neighborhood", {})),
            cycle=CycleConfig(**raw.get("cycle", {})),
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
        if self.projection.enabled and not (
            self.projection.artifact and self.projection.sha256
        ):
            raise ValueError("enabled projection requires artifact and sha256")
        if self.multiview.enabled and not (
            self.multiview.artifact and self.multiview.sha256
        ):
            raise ValueError("enabled multiview requires artifact and sha256")
        if self.multiview.method not in {"symmetric_top2", "max_pair", "mean_chamfer"}:
            raise ValueError("unsupported multiview.method")
        if not 0 <= self.multiview.blend <= 1:
            raise ValueError("multiview.blend must be in [0, 1]")
        if self.multiview.space not in {"raw", "projected"}:
            raise ValueError("multiview.space must be raw or projected")
        if self.multiview.calibration not in {"none", "per_video_pair_quantile"}:
            raise ValueError("unsupported multiview.calibration")
        if self.multiview.max_views_per_rep < 2:
            raise ValueError("multiview.max_views_per_rep must be >= 2")
        if self.multiview.space == "projected" and not self.projection.enabled:
            raise ValueError("projected multiview space requires projection.enabled")
        if self.neighborhood.enabled and not (
            self.neighborhood.artifact and self.neighborhood.sha256
        ):
            raise ValueError("enabled neighborhood requires artifact and sha256")
        if not 0 <= self.neighborhood.blend <= 1:
            raise ValueError("neighborhood.blend must be in [0, 1]")
        if self.neighborhood.calibration != "per_video_pair_quantile":
            raise ValueError("unsupported neighborhood.calibration")
        if self.neighborhood.scope != "uncertain_only":
            raise ValueError("unsupported neighborhood.scope")
        if not isinstance(self.neighborhood.confidence_weighting, bool):
            raise ValueError("neighborhood.confidence_weighting must be boolean")


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
    view_vectors: tuple[tuple[float, ...], ...] = ()

    @property
    def tracklet_id(self) -> str:
        return self.tracklet.tracklet_id

    @property
    def video_id(self) -> str:
        return self.tracklet.video_id

    @property
    def observation_count(self) -> int:
        return len(self.timestamps_ms)


@dataclass(frozen=True)
class NeighborhoodPairEvidence:
    score: float | None
    shared_anchors: tuple[str, ...] = ()
    overlap: float | None = None
    relation_agreement: float | None = None
    confidence: float = 1.0
    support: float | None = None


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


def _load_multiview_vectors(
    ref: str,
    ingest_root: Path,
    expected_dim: int,
    *,
    expected_sha256: str,
) -> dict[str, tuple[tuple[float, ...], ...]]:
    """读取逐视角侧车产物，并对 schema/维度/有限性/范数 fail closed。"""

    artifact = _resolve_ref(ref, ingest_root)
    from backend.tools.sf1.projection import sha256_file

    actual_sha256 = sha256_file(artifact)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"multiview sha256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    grouped: dict[str, list[tuple[int, tuple[float, ...]]]] = {}
    with np.load(artifact, allow_pickle=False) as data:
        version = str(data["format_version"].item())
        if version != ARTIFACT_FORMAT_VERSION:
            raise ValueError(f"unsupported multiview format: {version}")
        tracklet_ids = np.asarray(data["tracklet_ids"])
        view_indices = np.asarray(data["view_index"])
        vectors = np.asarray(data["vectors"])
    if tracklet_ids.ndim != 1 or tracklet_ids.dtype.kind not in {"U", "S"}:
        raise ValueError("multiview tracklet_ids must be a 1-D string array")
    tracklet_ids = tracklet_ids.astype(str)
    if any(not tracklet_id for tracklet_id in tracklet_ids):
        raise ValueError("multiview tracklet_ids cannot contain empty values")
    if view_indices.ndim != 1 or view_indices.dtype.kind not in {"i", "u"}:
        raise ValueError("multiview view_index must be a 1-D integer array")
    view_indices = view_indices.astype(np.int64, copy=False)
    if vectors.dtype != np.float32:
        raise ValueError(f"multiview vectors must be float32, got {vectors.dtype}")
    if vectors.ndim != 2 or vectors.shape[1] != expected_dim:
        raise ValueError(
            f"multiview expects [N,{expected_dim}] vectors, got {tuple(vectors.shape)}"
        )
    if len(tracklet_ids) != len(view_indices) or len(tracklet_ids) != len(vectors):
        raise ValueError("multiview arrays have inconsistent row counts")
    if not np.isfinite(vectors).all():
        raise ValueError("multiview vectors contain non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if (norms < 1e-12).any():
        raise ValueError("multiview vectors contain zero-norm rows")
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
        raise ValueError("multiview vectors must be L2-normalized")
    vectors = vectors / norms[:, None]
    for tracklet_id, view_index, vector in zip(tracklet_ids, view_indices, vectors):
        grouped.setdefault(str(tracklet_id), []).append(
            (int(view_index), tuple(float(value) for value in vector))
        )
    result: dict[str, tuple[tuple[float, ...], ...]] = {}
    for tracklet_id, rows in grouped.items():
        indices = [index for index, _ in sorted(rows)]
        if indices != list(range(len(indices))):
            raise ValueError(f"multiview indices are not contiguous for {tracklet_id}")
        result[tracklet_id] = tuple(vector for _, vector in sorted(rows))
    return result


def load_neighborhood_evidence(
    ref: str,
    ingest_root: str | Path,
    *,
    expected_sha256: str,
) -> dict[tuple[str, str], NeighborhoodPairEvidence]:
    """读取冻结邻域 sidecar，并对 hash/schema/分数范围 fail closed。"""

    root = Path(ingest_root)
    artifact = _resolve_ref(ref, root)
    from backend.tools.sf1.projection import sha256_file

    actual_sha256 = sha256_file(artifact)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"neighborhood sha256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    evidence: dict[tuple[str, str], NeighborhoodPairEvidence] = {}
    for line_number, line in enumerate(
        artifact.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        format_version = row.get("schema_version")
        if format_version not in SUPPORTED_CONTEXT_FORMAT_VERSIONS:
            raise ValueError(
                f"unsupported neighborhood format at line {line_number}: "
                f"{format_version}"
            )
        a, b = str(row.get("tracklet_a", "")), str(row.get("tracklet_b", ""))
        if not a or not b or a == b:
            raise ValueError(f"invalid neighborhood pair at line {line_number}")
        key = tuple(sorted((a, b)))
        if key in evidence:
            raise ValueError(f"duplicate neighborhood pair: {key}")
        raw_score = row.get("score")
        raw_overlap = row.get("overlap")
        raw_agreement = row.get("relation_agreement")
        raw_confidence = row.get("confidence")
        raw_support = row.get("support")
        anchors = tuple(str(value) for value in row.get("shared_anchors", []))
        if anchors != tuple(sorted(set(anchors))):
            raise ValueError(f"neighborhood anchors must be sorted unique: {key}")
        if raw_score is None:
            if (
                anchors
                or raw_overlap is not None
                or raw_agreement is not None
                or raw_confidence is not None
                or raw_support is not None
            ):
                raise ValueError(f"uncovered neighborhood pair carries evidence: {key}")
            item = NeighborhoodPairEvidence(score=None, confidence=0.0)
        else:
            score = float(raw_score)
            overlap = float(raw_overlap)
            agreement = float(raw_agreement)
            if format_version == CONTEXT_FORMAT_VERSION:
                if raw_confidence is None or raw_support is None:
                    raise ValueError(
                        f"v2 neighborhood pair lacks confidence/support: {key}"
                    )
                confidence = float(raw_confidence)
                support = float(raw_support)
            else:
                # v1 已封存产物保持原行为；新 v2 才启用单锚点置信度。
                if format_version != LEGACY_CONTEXT_FORMAT_VERSION:
                    raise ValueError(f"unsupported neighborhood format: {format_version}")
                confidence = 1.0
                support = None
            if not anchors:
                raise ValueError(f"covered neighborhood pair lacks anchors: {key}")
            values = [score, overlap, agreement, confidence]
            if support is not None:
                values.append(support)
            if not all(
                math.isfinite(value) and 0 <= value <= 1 for value in values
            ):
                raise ValueError(f"invalid neighborhood score at line {line_number}")
            item = NeighborhoodPairEvidence(
                score=score,
                shared_anchors=anchors,
                overlap=overlap,
                relation_agreement=agreement,
                confidence=confidence,
                support=support,
            )
        evidence[key] = item
    if not evidence:
        raise ValueError("neighborhood sidecar contains no pairs")
    return evidence


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
    projection: ProjectionConfig | None = None,
    multiview: MultiViewConfig | None = None,
) -> list[TrackFeature]:
    """读取每个视频目录的 Tracklet、Observation 与嵌入，顺序确定。

    ``attributes``(tracklet_id → 可比属性键值)在读入时合并进
    ``tracklet.attributes``;stitch 发生在其后,合并组自动继承 hero 成员属性。
    """

    root = Path(ingest_root)
    if not root.is_dir():
        raise FileNotFoundError(f"ingest root not found: {root}")

    projection = projection or ProjectionConfig()
    multiview = multiview or MultiViewConfig()
    projection_head = None
    if projection.enabled:
        artifact = _resolve_ref(projection.artifact, root)
        projection_head = NumpyProjectionHead.load(
            artifact, expected_sha256=projection.sha256
        )
        if projection_head.input_dim != embedding_dim:
            raise ValueError(
                f"projection expects {projection_head.input_dim} dims, config has {embedding_dim}"
            )
    view_vectors_by_tracklet: dict[str, tuple[tuple[float, ...], ...]] = {}
    if multiview.enabled:
        view_vectors_by_tracklet = _load_multiview_vectors(
            multiview.artifact,
            root,
            embedding_dim,
            expected_sha256=multiview.sha256,
        )
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
            view_vectors = view_vectors_by_tracklet.get(tracklet.tracklet_id, ())
            if multiview.enabled and not view_vectors:
                raise ValueError(f"multiview artifact missing tracklet {tracklet.tracklet_id}")
            if multiview.enabled and len(view_vectors) != len(tracklet.prototype_refs):
                raise ValueError(
                    f"multiview view count mismatch for {tracklet.tracklet_id}: "
                    f"expected {len(tracklet.prototype_refs)}, got {len(view_vectors)}"
                )
            if projection_head is not None:
                vector = tuple(float(value) for value in projection_head.apply(vector))
                if multiview.enabled and multiview.space == "projected":
                    projected_views = projection_head.apply(np.asarray(view_vectors, dtype=np.float32))
                    view_vectors = tuple(
                        tuple(float(value) for value in projected) for projected in projected_views
                    )
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
                    view_vectors=view_vectors,
                )
            )
    features.sort(key=lambda feature: feature.tracklet_id)
    return features
