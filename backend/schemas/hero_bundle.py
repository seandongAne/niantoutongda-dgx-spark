"""英雄场景统一数据合同 — S0 冻结契约之外的链路连接对象。

初赛范围冻结(docs/初赛范围冻结_2026-07-16.md)下的唯一演示链路:
旧房间 3 段视频 + 新家 1 段 + 逐件旁白
  → 实体(S3)→ 实体展示名(S5 VLM)→ 生活组合(GROUP)
  → 新家区域(manifest)→ CP-SAT 布局 → 结构化任务卡 → 验收 trace。

core.py 保持冻结:凡能落到 core 模型(LifeGroup / Region / PlacementPlan /
MoveTask / AuditEvent / 验收消息族)的一律转换后落 core;本模块只补
core 没有的连接对象(旁白、组证据、区域 manifest、任务卡、bundle 清单)。

GROUP 证据优先级(2026-07-16 用户裁决):
    旁白明确分组 > 用户轻确认 > 模板语义 > 画面共现(只佐证,不主导)。
移动镜头里的二维邻近不等于真实空间关系;共现证据永远不能单独把实体拉进组。
"""

from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.schemas.core import CapacityClass, LifeGroup, Region, SupportType, TaskStatus

HERO_BUNDLE_SCHEMA_VERSION = "1.0"


class _HeroContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = HERO_BUNDLE_SCHEMA_VERSION


class EvidenceSource(str, Enum):
    NARRATION = "narration"
    CONFIRMATION = "confirmation"
    TEMPLATE = "template"
    COOCCURRENCE = "cooccurrence"


# 数值越小优先级越高;共现只能佐证,不允许作为组员归属的唯一来源。
EVIDENCE_PRIORITY: dict[EvidenceSource, int] = {
    EvidenceSource.NARRATION: 0,
    EvidenceSource.CONFIRMATION: 1,
    EvidenceSource.TEMPLATE: 2,
    EvidenceSource.COOCCURRENCE: 3,
}
CORROBORATION_ONLY: frozenset[EvidenceSource] = frozenset({EvidenceSource.COOCCURRENCE})


class NarrationItem(_HeroContract):
    """逐件旁白的结构化条目 — 拍摄要求 §二第 5 行的五要素。"""

    item_id: str
    raw_text: str
    label_zh: str  # 是什么
    owner: str = ""  # 谁的
    source_location: str = ""  # 现在在哪
    target_location: str = ""  # 搬到新家想放哪
    group_partners: list[str] = []  # 和谁一组(旁白提到的同组物品名)
    color_words: list[str] = []  # 同款不同色消歧词(英文色值,与 S5 color_primary 同域)
    audio_ref: str = ""
    transcript_source: Literal["asr", "manual"] = "manual"


class NarrationResolution(_HeroContract):
    """旁白条目 → 实体的解析结果;解析不到或歧义一律进轻确认,不猜。"""

    item_id: str
    entity_id: Optional[str] = None
    method: Literal[
        "name_unique", "name_color", "unresolved_ambiguous", "unresolved_no_match"
    ]
    candidate_entity_ids: list[str] = []


class GroupEvidence(_HeroContract):
    entity_id: str
    source: EvidenceSource
    detail: str
    refs: list[str] = []


class GroupConfirmation(_HeroContract):
    """用户轻确认:只补旁白没覆盖的实体;与旁白冲突时旁白胜出并记录冲突。"""

    entity_id: str
    group_name_zh: str
    decision: Literal["assign", "remove"] = "assign"
    note: str = ""


class GroupClarification(_HeroContract):
    entity_id: str
    question_zh: str
    reason: Literal[
        "unresolved_narration", "unassigned_entity", "confirmation_conflict"
    ]
    candidate_group_ids: list[str] = []


class HeroGroup(_HeroContract):
    group_id: str
    name_zh: str
    entity_ids: list[str]
    dominant_source: EvidenceSource
    member_evidence: list[GroupEvidence]
    target_region_hint: str = ""

    @model_validator(mode="after")
    def _cooccurrence_never_dominant(self) -> "HeroGroup":
        if self.dominant_source in CORROBORATION_ONLY:
            raise ValueError("共现只能佐证,不能作为组的主导证据来源")
        return self

    def to_life_group(self) -> LifeGroup:
        source_map = {
            EvidenceSource.NARRATION: "user",
            EvidenceSource.CONFIRMATION: "user",
            EvidenceSource.TEMPLATE: "template",
        }
        return LifeGroup(
            group_id=self.group_id,
            entity_ids=sorted(self.entity_ids),
            source=source_map[self.dominant_source],
            evidence_refs=[
                f"{e.source.value}:{e.entity_id}:{e.detail}" for e in self.member_evidence
            ],
        )


class RegionEntry(_HeroContract):
    """新家候选区域 — 必须能回指 new_1.mp4 的画面证据(时间戳/帧路径)。"""

    region_id: str
    anchor: str  # bed / desk / closet / shelf / corner ...
    display_name_zh: str
    support_type: SupportType
    capacity_class: CapacityClass
    near_power: bool = False
    evidence_refs: list[str] = Field(min_length=1)

    def to_core_region(self) -> Region:
        return Region(
            region_id=self.region_id,
            anchor=self.anchor,
            support_type=self.support_type,
            capacity_class=self.capacity_class,
            attributes={"near_power": "true"} if self.near_power else {},
            evidence_refs=self.evidence_refs,
        )


class RegionManifest(_HeroContract):
    video_id: str
    entries: list[RegionEntry] = Field(min_length=1)
    notes: str = ""

    @model_validator(mode="after")
    def _unique_region_ids(self) -> "RegionManifest":
        ids = [e.region_id for e in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("region_id 重复")
        return self


class TaskCardItem(_HeroContract):
    entity_id: str
    display_name_zh: str
    hero_crop_ref: str = ""


class TaskCard(_HeroContract):
    card_id: str
    group_id: str
    box_label_zh: str
    items: list[TaskCardItem] = Field(min_length=1)
    target_region_id: str
    target_region_name_zh: str
    alternative_region_id: Optional[str] = None
    placement_notes: list[str] = []
    verification_checklist: list[str] = []
    priority: int = Field(default=3, ge=1, le=5)
    status: TaskStatus = TaskStatus.REGION_PLANNED


class AcceptanceMatch(_HeroContract):
    """验收照片里对单个实体的匹配结论 — 只答"出现与否",不答摆放。"""

    entity_id: str
    present: bool
    match_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    evidence_refs: list[str] = []


class AcceptancePhoto(_HeroContract):
    """一张验收照片:声明它拍的是哪个区域,以及照片内的实体匹配。

    matches 的来源要么是 spark 侧 ReID 匹配(reid),要么是对账 UI 人工
    勾选(manual)——两者都只是 presence 证据,verdict 一律走消息族裁决。
    """

    photo_ref: str
    region_id: str
    matches: list[AcceptanceMatch] = []
    match_source: Literal["reid", "manual"] = "manual"


class AcceptanceManifest(_HeroContract):
    photos: list[AcceptancePhoto] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_photo_refs(self) -> "AcceptanceManifest":
        refs = [p.photo_ref for p in self.photos]
        if len(refs) != len(set(refs)):
            raise ValueError("photo_ref 重复")
        return self


class StageArtifact(_HeroContract):
    stage: str
    path: str
    sha256: str


class HeroBundleManifest(_HeroContract):
    """一次英雄链路复跑的顶层清单 — 串起全部阶段产物与配置指纹。"""

    bundle_id: str
    created_at: str  # ISO-8601 UTC
    config_refs: dict[str, str] = {}  # 配置名 → sha256
    artifacts: list[StageArtifact] = []
    notes: str = ""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
