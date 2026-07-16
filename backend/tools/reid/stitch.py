"""S3 前置:同视频短轨 stitch 与低证据过滤。

目标是把跨视频匹配前的碎轨先在视频内收拢,直接压低澄清请求量;
禁止通过放宽跨视频 match 阈值达成同样数字。

不变量:
* 合并轨的代表 id 取成员中字典序最小者,因此对外输出的一切 id 仍是
  合法的原始 tracklet id;完整成员清单见 stitch report(stitch-map.json)。
* 同帧共现 = 物理上是两个物体,任何相似度都不能推翻(用户 same 约束除外)。
* 全流程确定性:候选对按 (-cosine, id) 排序贪心合并,无随机性。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from backend.schemas.core import Tracklet
from backend.tools.reid.model import ReIDConfig, TrackFeature


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b))))


def _label_compatible(a: TrackFeature, b: TrackFeature) -> bool:
    if a.canonical_id and b.canonical_id:
        return a.canonical_id == b.canonical_id
    if a.canonical_id or b.canonical_id:
        return False
    return a.raw_label.strip().lower() == b.raw_label.strip().lower()


def _co_occurs(a: TrackFeature, b: TrackFeature) -> bool:
    return bool(set(a.timestamps_ms) & set(b.timestamps_ms))


def _gap_ms(a: TrackFeature, b: TrackFeature) -> int:
    if not a.timestamps_ms or not b.timestamps_ms:
        return 0
    first, second = sorted((a, b), key=lambda f: f.timestamps_ms[0])
    return max(0, second.timestamps_ms[0] - first.timestamps_ms[-1])


@dataclass
class StitchResult:
    features: list[TrackFeature]
    # 代表 id → 全部成员原始 id(含只有自己的未合并轨不入表)
    members_by_rep: dict[str, list[str]]
    report: dict[str, Any]

    def expand(self, rep_id: str) -> list[str]:
        return self.members_by_rep.get(rep_id, [rep_id])


def _merge_cluster(members: list[TrackFeature]) -> TrackFeature:
    members = sorted(members, key=lambda f: f.tracklet_id)
    rep_id = members[0].tracklet_id
    # hero 代表:成员里 hero_score 最高者提供 label/hero 元数据与几何代表值
    def _hero_score(feature: TrackFeature) -> float:
        try:
            return float(feature.tracklet.attributes.get("hero_score", "0"))
        except ValueError:
            return 0.0

    ordered_by_hero = sorted(members, key=lambda f: (-_hero_score(f), f.tracklet_id))
    hero = ordered_by_hero[0]

    total_obs = sum(max(1, f.observation_count) for f in members)
    dim = len(members[0].vector)
    mean = [0.0] * dim
    for feature in members:
        weight = max(1, feature.observation_count) / total_obs
        for i, value in enumerate(feature.vector):
            mean[i] += weight * value
    norm = max(sum(v * v for v in mean) ** 0.5, 1e-12)
    vector = tuple(v / norm for v in mean)

    observation_ids: list[str] = []
    prototype_refs: list[str] = []
    timestamps: list[int] = []
    for feature in ordered_by_hero:
        prototype_refs.extend(r for r in feature.tracklet.prototype_refs if r not in prototype_refs)
    for feature in members:
        observation_ids.extend(feature.tracklet.observation_ids)
        timestamps.extend(feature.timestamps_ms)

    attributes = dict(hero.tracklet.attributes)
    attributes["stitched_members"] = ",".join(f.tracklet_id for f in members)
    attributes["stitch_version"] = "same-video-stitch-v1"

    ratios = [f.aspect_ratio for f in members if f.aspect_ratio]
    weights = [max(1, f.observation_count) for f in members if f.aspect_ratio]
    aspect = sum(r * w for r, w in zip(ratios, weights)) / sum(weights) if ratios else None
    areas = [f.area for f in members if f.area]
    area_weights = [max(1, f.observation_count) for f in members if f.area]
    area = sum(a * w for a, w in zip(areas, area_weights)) / sum(area_weights) if areas else None

    merged = Tracklet(
        tracklet_id=rep_id,
        video_id=members[0].video_id,
        observation_ids=sorted(observation_ids),
        prototype_refs=prototype_refs,
        embedding_ref=hero.tracklet.embedding_ref,
        attributes=attributes,
    )
    return TrackFeature(
        tracklet=merged,
        vector=vector,
        raw_label=hero.raw_label,
        canonical_id=hero.canonical_id,
        category_id=hero.category_id,
        quality=max(f.quality for f in members),
        aspect_ratio=aspect,
        area=area,
        timestamps_ms=tuple(sorted(timestamps)),
    )


def stitch_features(
    features: list[TrackFeature],
    config: ReIDConfig,
    *,
    forced_same: frozenset[tuple[str, str]] = frozenset(),
    forbidden: frozenset[tuple[str, str]] = frozenset(),
) -> StitchResult:
    """按视频内规则合并碎轨。forced_same/forbidden 只处理同视频约束对。"""

    by_id = {f.tracklet_id: f for f in features}
    parent = {f.tracklet_id: f.tracklet_id for f in features}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def cluster(item: str) -> list[str]:
        root = find(item)
        return [i for i in parent if find(i) == root]

    def cluster_blocked(a: str, b: str, *, respect_cooccur: bool) -> bool:
        members_a, members_b = cluster(a), cluster(b)
        for x in members_a:
            for y in members_b:
                if tuple(sorted((x, y))) in forbidden:
                    return True
                if respect_cooccur and _co_occurs(by_id[x], by_id[y]):
                    return True
        return False

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            low, high = sorted((root_a, root_b))
            parent[high] = low

    merges: list[dict[str, Any]] = []
    vetoes = {"co_occurrence": 0, "cannot_link": 0}

    # 用户 same 约束是最高优先级证据,先合并(共现让位于用户;与 cannot-link 冲突必须报错)。
    for a, b in sorted(forced_same):
        if a in by_id and b in by_id and by_id[a].video_id == by_id[b].video_id:
            if cluster_blocked(a, b, respect_cooccur=False):
                raise ValueError(f"user same-constraint conflicts with cannot-link inside video: {(a, b)}")
            union(a, b)
            merges.append({"a": a, "b": b, "cosine": None, "mode": "user_same"})

    if config.stitch.enabled:
        candidates = []
        by_video: dict[str, list[TrackFeature]] = defaultdict(list)
        for feature in features:
            by_video[feature.video_id].append(feature)
        for video_features in by_video.values():
            ordered = sorted(video_features, key=lambda f: f.tracklet_id)
            for i, a in enumerate(ordered):
                for b in ordered[i + 1 :]:
                    if not _label_compatible(a, b):
                        continue
                    if config.stitch.max_gap_ms and _gap_ms(a, b) > config.stitch.max_gap_ms:
                        continue
                    cosine = _cosine(a.vector, b.vector)
                    if cosine < config.stitch.min_cosine:
                        continue
                    candidates.append((cosine, a.tracklet_id, b.tracklet_id))
        for cosine, a, b in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
            if find(a) == find(b):
                continue
            pair_forbidden = cluster_blocked(a, b, respect_cooccur=False)
            if pair_forbidden:
                vetoes["cannot_link"] += 1
                continue
            if cluster_blocked(a, b, respect_cooccur=True):
                vetoes["co_occurrence"] += 1
                continue
            union(a, b)
            merges.append({"a": a, "b": b, "cosine": round(cosine, 8), "mode": "stitch"})

    clusters: dict[str, list[str]] = defaultdict(list)
    for tracklet_id in sorted(by_id):
        clusters[find(tracklet_id)].append(tracklet_id)

    stitched: list[TrackFeature] = []
    members_by_rep: dict[str, list[str]] = {}
    for members in sorted(clusters.values(), key=lambda item: item[0]):
        if len(members) == 1:
            stitched.append(by_id[members[0]])
            continue
        merged = _merge_cluster([by_id[m] for m in members])
        members_by_rep[merged.tracklet_id] = sorted(members)
        stitched.append(merged)
    stitched.sort(key=lambda f: f.tracklet_id)

    per_video: dict[str, dict[str, int]] = {}
    for video_id in sorted({f.video_id for f in features}):
        per_video[video_id] = {
            "before": sum(f.video_id == video_id for f in features),
            "after": sum(f.video_id == video_id for f in stitched),
        }

    report = {
        "enabled": config.stitch.enabled,
        "min_cosine": config.stitch.min_cosine,
        "max_gap_ms": config.stitch.max_gap_ms,
        "tracklets_before": len(features),
        "tracklets_after": len(stitched),
        "merge_count": len(merges),
        "merges": merges,
        "vetoes": vetoes,
        "per_video": per_video,
        "groups": {rep: members for rep, members in sorted(members_by_rep.items())},
    }
    return StitchResult(features=stitched, members_by_rep=members_by_rep, report=report)


def split_low_evidence(
    features: list[TrackFeature],
    config: ReIDConfig,
    members_by_rep: dict[str, list[str]],
    protected: frozenset[str] = frozenset(),
) -> tuple[list[TrackFeature], list[dict[str, Any]]]:
    """把观测数不足的轨从跨视频配对里摘出来;它们保留为单例实体,不消失。

    protected 里的 id(用户约束点名的轨)不过滤——用户证据优先于启发式。
    """

    if config.filter.min_observations <= 1:
        return features, []
    kept: list[TrackFeature] = []
    excluded: list[dict[str, Any]] = []
    for feature in features:
        if (
            feature.observation_count >= config.filter.min_observations
            or feature.tracklet_id in protected
        ):
            kept.append(feature)
        else:
            excluded.append(
                {
                    "tracklet_id": feature.tracklet_id,
                    "members": members_by_rep.get(feature.tracklet_id, [feature.tracklet_id]),
                    "video_id": feature.video_id,
                    "observation_count": feature.observation_count,
                    "quality": round(feature.quality, 8),
                    "reason": "LOW_EVIDENCE_MIN_OBSERVATIONS",
                }
            )
    return kept, excluded
