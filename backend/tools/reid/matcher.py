"""S3 确定性跨视频匹配 baseline。

流程严格保持：Top-K 召回 → 硬门控 → 视频对一对一分配 → 三段式决策
→ 每视频最多一个成员/cannot-link 约束聚类。没有机器真值时只输出诊断指标，
不会生成 Recall@1 或 13/15 等伪验收数字。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from backend.schemas.core import ClarificationRequest, IdentityState, ObjectEntity
from backend.tools.reid.assignment import maximise_assignment
from backend.tools.reid.model import (
    COMPARABLE_ATTRIBUTE_KEYS,
    UNKNOWN_ATTRIBUTE_VALUES,
    ReIDConfig,
    TrackFeature,
    Vocabulary,
    load_features,
)
from backend.tools.reid.stitch import stitch_features, tag_low_evidence


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


@dataclass(frozen=True)
class IdentityConstraints:
    same: frozenset[tuple[str, str]] = frozenset()
    different: frozenset[tuple[str, str]] = frozenset()

    @classmethod
    def from_json(cls, path: str | Path | None) -> "IdentityConstraints":
        if path is None:
            return cls()
        raw = json.loads(Path(path).read_text())
        same = frozenset(_pair_key(str(a), str(b)) for a, b in raw.get("same", []))
        different = frozenset(_pair_key(str(a), str(b)) for a, b in raw.get("different", []))
        overlap = same & different
        if overlap:
            raise ValueError(f"identity constraints conflict: {sorted(overlap)}")
        return cls(same=same, different=different)


@dataclass(frozen=True)
class PairScore:
    a: str
    b: str
    instance: float
    semantic: float
    attribute: float | None  # None = 该对没有可比属性键(权重已让渡)
    context: float
    geometry: float
    total: float
    gate_reasons: tuple[str, ...] = ()

    @property
    def viable(self) -> bool:
        return not self.gate_reasons

    def as_dict(self) -> dict[str, Any]:
        return {
            "tracklet_a": self.a,
            "tracklet_b": self.b,
            "score": round(self.total, 8),
            "components": {
                "instance": round(self.instance, 8),
                "semantic": round(self.semantic, 8),
                "attribute": round(self.attribute, 8) if self.attribute is not None else None,
                "context": round(self.context, 8),
                "geometry": round(self.geometry, 8),
            },
            "gate_reasons": list(self.gate_reasons),
        }


@dataclass
class ReIDRun:
    config_version: str
    entities: list[ObjectEntity]
    clarifications: list[ClarificationRequest]
    candidates: list[dict[str, Any]]
    accepted_links: list[dict[str, Any]]
    metrics: dict[str, Any]
    stitch_report: dict[str, Any] | None = None
    filtered_tracklets: list[dict[str, Any]] | None = None

    def write(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out / "entities.jsonl", (entity.model_dump(mode="json") for entity in self.entities))
        _write_jsonl(
            out / "clarifications.jsonl",
            (request.model_dump(mode="json") for request in self.clarifications),
        )
        _write_jsonl(out / "candidates.jsonl", self.candidates)
        _write_jsonl(out / "accepted-links.jsonl", self.accepted_links)
        (out / "metrics.json").write_text(
            json.dumps(self.metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        if self.stitch_report is not None:
            (out / "stitch-map.json").write_text(
                json.dumps(self.stitch_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
        if self.filtered_tracklets is not None:
            _write_jsonl(out / "filtered-tracklets.jsonl", self.filtered_tracklets)


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b))))


def _attribute_score(a: TrackFeature, b: TrackFeature) -> float | None:
    """S5 属性相似度,白名单键逐一比较。

    missing/unknown 语义:任一侧未知的键不进分子也不进分母;零可比键返回
    None,由 score_pair 把 attribute 权重让渡给其余分量(而不是给 0.5 偏置)。
    """
    hits = comparable = 0
    for key in COMPARABLE_ATTRIBUTE_KEYS:
        value_a = str(a.tracklet.attributes.get(key, "")).strip().lower()
        value_b = str(b.tracklet.attributes.get(key, "")).strip().lower()
        if value_a in UNKNOWN_ATTRIBUTE_VALUES or value_b in UNKNOWN_ATTRIBUTE_VALUES:
            continue
        comparable += 1
        hits += value_a == value_b
    if not comparable:
        return None
    return hits / comparable


def _geometry_score(a: TrackFeature, b: TrackFeature) -> float:
    if not a.aspect_ratio or not b.aspect_ratio:
        return 0.5
    return math.exp(-abs(math.log(a.aspect_ratio / b.aspect_ratio)))


def score_pair(
    a: TrackFeature,
    b: TrackFeature,
    config: ReIDConfig,
    constraints: IdentityConstraints,
) -> PairScore:
    reasons: list[str] = []
    if a.video_id == b.video_id:
        reasons.append("SAME_VIDEO_MUTEX")
    if _pair_key(a.tracklet_id, b.tracklet_id) in constraints.different:
        reasons.append("USER_CANNOT_LINK")
    if min(a.quality, b.quality) < config.thresholds.min_quality:
        reasons.append("LOW_QUALITY")
    if a.category_id and b.category_id and a.category_id != b.category_id:
        reasons.append("CATEGORY_CONFLICT")

    instance = _cosine(a.vector, b.vector)
    if a.category_id and a.category_id == b.category_id:
        semantic = 1.0
    elif a.canonical_id and a.canonical_id == b.canonical_id:
        semantic = 0.9
    elif a.raw_label.strip().lower() == b.raw_label.strip().lower():
        semantic = 0.75
    else:
        semantic = 0.25
    attribute = _attribute_score(a, b)
    context = 0.5  # v5 没有上下文字段，保持中性且默认权重为 0。
    geometry = _geometry_score(a, b)
    weights = config.weights
    # 零可比属性键 → attribute 权重让渡:分子不含该项,分母同步扣除,
    # 其余分量按原比例放大;等价于"该对不测属性",不引入 0.5 中性偏置。
    effective_attribute_weight = weights.attribute if attribute is not None else 0.0
    denominator = weights.total - weights.attribute + effective_attribute_weight
    total = (
        weights.instance * instance
        + weights.semantic * semantic
        + effective_attribute_weight * (attribute if attribute is not None else 0.0)
        + weights.context * context
        + weights.geometry * geometry
    ) / denominator if denominator > 0 else 0.0
    return PairScore(
        a=a.tracklet_id,
        b=b.tracklet_id,
        instance=instance,
        semantic=semantic,
        attribute=attribute,
        context=context,
        geometry=geometry,
        total=max(0.0, min(1.0, total)),
        gate_reasons=tuple(sorted(reasons)),
    )


def _second_best_margin(chosen: PairScore, alternatives: list[PairScore]) -> float:
    other_scores = [score.total for score in alternatives if score.viable and score != chosen]
    return chosen.total - max(other_scores, default=0.0)


def _pairwise_assignments(
    features: list[TrackFeature],
    config: ReIDConfig,
    constraints: IdentityConstraints,
) -> tuple[list[PairScore], dict[tuple[str, str], tuple[tuple[str, ...], float]], list[dict[str, Any]]]:
    by_video: dict[str, list[TrackFeature]] = defaultdict(list)
    for feature in features:
        by_video[feature.video_id].append(feature)
    accepted: list[PairScore] = []
    # pair → (reason codes, 供澄清封顶排序用的分数)
    ambiguous: dict[tuple[str, str], tuple[tuple[str, ...], float]] = {}
    candidates_out: list[dict[str, Any]] = []

    for video_a, video_b in combinations(sorted(by_video), 2):
        left = sorted(by_video[video_a], key=lambda feature: feature.tracklet_id)
        right = sorted(by_video[video_b], key=lambda feature: feature.tracklet_id)
        score_rows = [
            [score_pair(a, b, config, constraints) for b in right]
            for a in left
        ]
        left_ranked = {
            a.tracklet_id: sorted(
                (score for score in row if score.viable),
                key=lambda score: (-score.total, score.b),
            )[: config.top_k]
            for a, row in zip(left, score_rows)
        }
        right_ranked: dict[str, list[PairScore]] = {}
        for column, b in enumerate(right):
            right_ranked[b.tracklet_id] = sorted(
                (row[column] for row in score_rows if row[column].viable),
                key=lambda score: (-score.total, score.a),
            )[: config.top_k]

        recalled = {
            _pair_key(score.a, score.b)
            for rows in (left_ranked.values(), right_ranked.values())
            for row in rows
            for score in row
        }
        matrix = [
            [
                score.total if score.viable and _pair_key(score.a, score.b) in recalled else -math.inf
                for score in row
            ]
            for row in score_rows
        ]
        assignment = maximise_assignment(matrix, config.thresholds.new + 1e-10)
        assigned_pairs: set[tuple[str, str]] = set()
        for row_index, column in enumerate(assignment):
            if column is None or not math.isfinite(matrix[row_index][column]):
                continue
            chosen = score_rows[row_index][column]
            assigned_pairs.add(_pair_key(chosen.a, chosen.b))
            margin = min(
                _second_best_margin(chosen, left_ranked[chosen.a]),
                _second_best_margin(chosen, right_ranked[chosen.b]),
            )
            decision = "NEW_ENTITY"
            if chosen.total >= config.thresholds.match and margin >= config.thresholds.margin:
                decision = "MATCHED"
                accepted.append(chosen)
            elif chosen.total > config.thresholds.new:
                decision = "SUSPECTED_DUPLICATE"
                ambiguous[_pair_key(chosen.a, chosen.b)] = (("SCORE_OR_MARGIN_UNCERTAIN",), chosen.total)
            row = chosen.as_dict()
            row.update(
                {
                    "video_pair": [video_a, video_b],
                    "assigned": True,
                    "margin": round(margin, 8),
                    "decision": decision,
                }
            )
            candidates_out.append(row)

        # 被全局一对一挤掉但仍高于 T_new 的轨迹进入确认队列，不能静默丢弃。
        for ranked in list(left_ranked.values()) + list(right_ranked.values()):
            if not ranked:
                continue
            best = ranked[0]
            key = _pair_key(best.a, best.b)
            if best.total > config.thresholds.new and key not in assigned_pairs:
                ambiguous.setdefault(key, (("GLOBAL_ASSIGNMENT_CONTENTION",), best.total))

        for a, row in zip(left, score_rows):
            recalled_for_left = {_pair_key(score.a, score.b) for score in left_ranked[a.tracklet_id]}
            for score in row:
                key = _pair_key(score.a, score.b)
                if key not in recalled_for_left or key in assigned_pairs:
                    continue
                record = score.as_dict()
                record.update(
                    {
                        "video_pair": [video_a, video_b],
                        "assigned": False,
                        "decision": "CANDIDATE",
                    }
                )
                candidates_out.append(record)

    accepted.sort(key=lambda score: (-score.total, score.a, score.b))
    candidates_out.sort(
        key=lambda row: (
            row["video_pair"],
            not row["assigned"],
            -row["score"],
            row["tracklet_a"],
            row["tracklet_b"],
        )
    )
    return accepted, ambiguous, candidates_out


class _UnionFind:
    def __init__(self, ids: list[str]):
        self.parent = {item: item for item in ids}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            item, self.parent[item] = self.parent[item], root
        return root

    def members(self, item: str) -> set[str]:
        root = self.find(item)
        return {candidate for candidate in self.parent if self.find(candidate) == root}

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        low, high = sorted((root_a, root_b))
        self.parent[high] = low


def _can_union(
    union_find: _UnionFind,
    a: str,
    b: str,
    feature_by_id: dict[str, TrackFeature],
    constraints: IdentityConstraints,
) -> bool:
    if union_find.find(a) == union_find.find(b):
        return True
    members_a, members_b = union_find.members(a), union_find.members(b)
    videos_a = {feature_by_id[item].video_id for item in members_a}
    videos_b = {feature_by_id[item].video_id for item in members_b}
    if videos_a & videos_b:
        return False
    return not any(_pair_key(x, y) in constraints.different for x in members_a for y in members_b)


def _entity_id(members: list[str]) -> str:
    digest = hashlib.sha256("\n".join(members).encode()).hexdigest()[:12]
    return f"entity_{digest}"


def _cap_clarifications(
    ambiguous: dict[tuple[str, str], tuple[tuple[str, ...], float]],
    max_partners: int,
    video_of: dict[str, str],
) -> tuple[dict[tuple[str, str], tuple[tuple[str, ...], float]], int]:
    """互选封顶:pair 必须同时进入两端点各自(按对手视频)的 top-N 才保留。

    "任一端点保留即保留"挡不住星型扇出——对面碎轨只有这一条请求,
    自身 top-1 永远兜底。互选语义下,一条轨对某视频只发起 top-N 个提问,
    对面也必须认可。实体的 SUSPECTED 状态必须在封顶前的歧义集上计算,
    封顶只裁剪人工澄清队列,不粉饰不确定性。
    """

    if max_partners <= 0:
        return ambiguous, 0
    partners: dict[tuple[str, str], list[tuple[float, str, tuple[str, str]]]] = defaultdict(list)
    for pair, (_, score) in ambiguous.items():
        a, b = pair
        partners[(a, video_of[b])].append((score, b, pair))
        partners[(b, video_of[a])].append((score, a, pair))
    allowed: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for key, entries in partners.items():
        ranked = sorted(entries, key=lambda item: (-item[0], item[1]))
        allowed[key] = {pair for _, _, pair in ranked[:max_partners]}
    keep = {
        pair
        for pair in ambiguous
        if pair in allowed[(pair[0], video_of[pair[1]])]
        and pair in allowed[(pair[1], video_of[pair[0]])]
    }
    suppressed = len(ambiguous) - len(keep)
    return {pair: value for pair, value in ambiguous.items() if pair in keep}, suppressed


def _clarification(pair: tuple[str, str], reasons: tuple[str, ...]) -> ClarificationRequest:
    digest = hashlib.sha256("\n".join(pair).encode()).hexdigest()[:12]
    return ClarificationRequest(
        request_id=f"clarify_{digest}",
        candidate_a=pair[0],
        candidate_b=pair[1],
        reason_codes=list(reasons),
    )


def run_reid(
    *,
    ingest_root: str | Path,
    config: ReIDConfig,
    vocab: Vocabulary,
    constraints: IdentityConstraints | None = None,
    attributes: dict[str, dict[str, str]] | None = None,
) -> ReIDRun:
    constraints = constraints or IdentityConstraints()
    original_features = load_features(
        ingest_root, vocab=vocab, embedding_dim=config.embedding_dim, attributes=attributes
    )
    if not original_features:
        raise ValueError("no embedded tracklets found")
    original_ids = {feature.tracklet_id for feature in original_features}
    unknown_constraints = {
        item
        for pair in constraints.same | constraints.different
        for item in pair
        if item not in original_ids
    }
    if unknown_constraints:
        raise ValueError(f"constraints reference unknown tracklets: {sorted(unknown_constraints)}")

    # 同视频碎轨先收拢;约束在原始 id 空间校验后重映射到代表 id 空间。
    stitch_result = stitch_features(
        original_features,
        config,
        forced_same=constraints.same,
        forbidden=constraints.different,
    )
    member_to_rep = {
        member: rep for rep, members in stitch_result.members_by_rep.items() for member in members
    }

    def _rep(item: str) -> str:
        return member_to_rep.get(item, item)

    remapped_same = {
        _pair_key(_rep(a), _rep(b)) for a, b in constraints.same if _rep(a) != _rep(b)
    }
    for a, b in constraints.different:
        if _rep(a) == _rep(b):
            raise ValueError(f"cannot-link pair was stitched into one track: {(a, b)}")
    remapped_different = {_pair_key(_rep(a), _rep(b)) for a, b in constraints.different}
    constraints = IdentityConstraints(
        same=frozenset(remapped_same), different=frozenset(remapped_different)
    )

    low_evidence_ids, filtered_records = tag_low_evidence(
        stitch_result.features,
        config,
        stitch_result.members_by_rep,
        protected=frozenset(
            item for pair in constraints.same | constraints.different for item in pair
        ),
    )
    matching_features = stitch_result.features
    feature_by_id = {feature.tracklet_id: feature for feature in matching_features}
    stitched_by_id = feature_by_id

    accepted, ambiguous, candidates = _pairwise_assignments(matching_features, config, constraints)
    union_find = _UnionFind(sorted(feature_by_id))
    accepted_records: list[dict[str, Any]] = []

    for a, b in sorted(constraints.same):
        if not _can_union(union_find, a, b, feature_by_id, constraints):
            raise ValueError(f"positive constraint violates one-track-per-video/cannot-link: {(a, b)}")
        union_find.union(a, b)
        accepted_records.append({"tracklet_a": a, "tracklet_b": b, "mode": "user_same", "score": 1.0})

    accepted_score_by_pair: dict[tuple[str, str], float] = {}
    for score in accepted:
        key = _pair_key(score.a, score.b)
        if _can_union(union_find, score.a, score.b, feature_by_id, constraints):
            union_find.union(score.a, score.b)
            accepted_score_by_pair[key] = score.total
            accepted_records.append(
                {"tracklet_a": score.a, "tracklet_b": score.b, "mode": "automatic", "score": round(score.total, 8)}
            )
        else:
            ambiguous.setdefault(key, (("GLOBAL_CLUSTER_CONFLICT",), score.total))

    # 实体状态基于完整歧义集(诚实);澄清队列先摘低证据端点的对,再互选封顶。
    eligible = {
        pair: value
        for pair, value in ambiguous.items()
        if pair[0] not in low_evidence_ids and pair[1] not in low_evidence_ids
    }
    suppressed_low_evidence = len(ambiguous) - len(eligible)
    clarify_pairs, suppressed_count = _cap_clarifications(
        eligible,
        config.clarify.max_partners_per_tracklet,
        {feature.tracklet_id: feature.video_id for feature in matching_features},
    )

    clusters: dict[str, list[str]] = defaultdict(list)
    for tracklet_id in sorted(feature_by_id):
        clusters[union_find.find(tracklet_id)].append(tracklet_id)
    ambiguous_members = {item for pair in ambiguous for item in pair}
    entities: list[ObjectEntity] = []
    member_lists = [sorted(value) for value in clusters.values()]
    for members in sorted(member_lists, key=lambda value: value[0]):
        member_features = [stitched_by_id[item] for item in members]
        labels = [
            feature.category_id or feature.canonical_id or feature.raw_label or "unknown"
            for feature in member_features
        ]
        label_counts = Counter(labels)
        label = sorted(label_counts, key=lambda value: (-label_counts[value], value))[0]
        link_scores = [
            score
            for pair, score in accepted_score_by_pair.items()
            if pair[0] in members and pair[1] in members
        ]
        if len(members) > 1:
            state = IdentityState.MATCHED
            confidence = sum(link_scores) / len(link_scores) if link_scores else 1.0
        elif members[0] in ambiguous_members:
            state = IdentityState.SUSPECTED_DUPLICATE
            confidence = max(
                (row["score"] for row in candidates if members[0] in (row["tracklet_a"], row["tracklet_b"])),
                default=0.0,
            )
        else:
            state = IdentityState.NEW_ENTITY
            confidence = 1.0 - max(
                (row["score"] for row in candidates if members[0] in (row["tracklet_a"], row["tracklet_b"])),
                default=0.0,
            )
        evidence = sorted(
            {reference for feature in member_features for reference in feature.tracklet.prototype_refs}
        )
        # 对外一律展开回原始 tracklet id 空间;stitch 只是 S3 内部的视图。
        full_members = sorted(
            {original for item in members for original in stitch_result.expand(item)}
        )
        entities.append(
            ObjectEntity(
                entity_id=_entity_id(full_members),
                tracklet_ids=full_members,
                label=label,
                identity_state=state,
                confidence=max(0.0, min(1.0, confidence)),
                evidence_refs=evidence,
            )
        )

    clarifications = [_clarification(pair, clarify_pairs[pair][0]) for pair in sorted(clarify_pairs)]
    metrics = {
        "config_version": config.version,
        "baseline_only": True,
        "g2_evaluated": False,
        "g2_blocker": "machine-readable anchor-to-tracklet ground truth is absent",
        "tracklet_count": len(original_features),
        "known_category_tracklets": sum(
            feature.category_id is not None for feature in original_features
        ),
        "attribute_enriched_tracklet_count": sum(
            any(
                str(feature.tracklet.attributes.get(key, "")).strip().lower()
                not in UNKNOWN_ATTRIBUTE_VALUES
                for key in COMPARABLE_ATTRIBUTE_KEYS
            )
            for feature in original_features
        ),
        "stitch_enabled": config.stitch.enabled,
        "stitch_merge_count": stitch_result.report["merge_count"],
        "tracklet_count_after_stitch": len(stitch_result.features),
        "low_evidence_tracklet_count": len(filtered_records),
        "clarifications_suppressed_low_evidence": suppressed_low_evidence,
        "clarifications_suppressed_by_cap": suppressed_count,
        "entity_count": len(entities),
        "matched_entity_count": sum(entity.identity_state == IdentityState.MATCHED for entity in entities),
        "new_entity_count": sum(entity.identity_state == IdentityState.NEW_ENTITY for entity in entities),
        "suspected_entity_count": sum(
            entity.identity_state == IdentityState.SUSPECTED_DUPLICATE for entity in entities
        ),
        "clarification_count": len(clarifications),
        "automatic_link_count": sum(record["mode"] == "automatic" for record in accepted_records),
        "user_same_link_count": sum(record["mode"] == "user_same" for record in accepted_records),
    }
    return ReIDRun(
        config_version=config.version,
        entities=entities,
        clarifications=clarifications,
        candidates=candidates,
        accepted_links=accepted_records,
        metrics=metrics,
        stitch_report=stitch_result.report,
        filtered_tracklets=filtered_records,
    )
