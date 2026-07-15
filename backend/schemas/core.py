"""S0 数据契约 — 手册 §6 冻结的 schema。所有对象带 schema_version。

设计文档 §7.1 是字段的权威来源;这里只做最小必要扩展(枚举、校验)。
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = SCHEMA_VERSION


class IdentityState(str, Enum):
    EXPECTED = "EXPECTED"
    OBSERVED = "OBSERVED"
    MATCHED = "MATCHED"
    NEW_ENTITY = "NEW_ENTITY"
    SUSPECTED_DUPLICATE = "SUSPECTED_DUPLICATE"
    NOT_SEEN = "NOT_SEEN"


class TaskStatus(str, Enum):
    GROUPED = "GROUPED"
    BOXED = "BOXED"
    REGION_PLANNED = "REGION_PLANNED"
    PLACED = "PLACED"
    VERIFIED = "VERIFIED"
    NOT_SEEN = "NOT_SEEN"
    USER_OVERRIDDEN = "USER_OVERRIDDEN"
    NEW_SPACE_INCOMPATIBLE = "NEW_SPACE_INCOMPATIBLE"


class SupportType(str, Enum):
    SURFACE = "surface"
    DRAWER = "drawer"
    SHELF = "shelf"
    FLOOR = "floor"
    WALL = "wall"


class CapacityClass(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class Observation(_Contract):
    observation_id: str
    video_id: str
    timestamp_ms: int = Field(ge=0)
    bbox: tuple[float, float, float, float]  # x1,y1,x2,y2 归一化或像素,由 capture 配置声明
    crop_ref: str
    quality: float = Field(ge=0.0, le=1.0)
    model_version: str


class Tracklet(_Contract):
    tracklet_id: str
    video_id: str
    observation_ids: list[str]
    prototype_refs: list[str] = []  # Top-K 高质量视角裁剪
    embedding_ref: Optional[str] = None
    attributes: dict[str, str] = {}


class ObjectEntity(_Contract):
    entity_id: str
    tracklet_ids: list[str]
    label: str
    identity_state: IdentityState
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_refs: list[str]


class ClarificationRequest(_Contract):
    request_id: str
    candidate_a: str  # tracklet/entity id
    candidate_b: str
    reason_codes: list[str]
    decision: Optional[Literal["same", "different"]] = None


class VerificationCheckRequest(_Contract):
    request_id: str
    task_id: str
    expected_entity_ids: list[str]
    photo_refs: list[str]
    result: Optional[Literal["VERIFIED", "NOT_SEEN"]] = None
    reason_codes: list[str] = []


class RelationEdge(_Contract):
    src: str
    dst: str
    relation: Literal["NEAR", "SAME_SURFACE", "CO_USED", "REQUIRES_POWER", "STORED_WITH", "ANCHORED_TO"]
    evidence_refs: list[str] = []


class LifeGroup(_Contract):
    group_id: str
    entity_ids: list[str]
    relation_edges: list[RelationEdge] = []
    source: Literal["auto", "template", "user"]
    evidence_refs: list[str]


class Region(_Contract):
    region_id: str
    anchor: str  # bed / desk / closet / door / window ...
    support_type: SupportType
    capacity_class: CapacityClass
    attributes: dict[str, str] = {}  # near_power="true" 必须有证据,由规则校验
    evidence_refs: list[str]


class Assignment(_Contract):
    group_id: str
    region_id: str
    score_breakdown: dict[str, int] = {}
    alternative_region_id: Optional[str] = None


class PlacementPlan(_Contract):
    plan_id: str
    assignments: list[Assignment]
    hard_constraints: list[str]
    soft_scores: dict[str, int] = {}
    solver_status: Literal["PLAN_READY", "NEW_SPACE_INCOMPATIBLE", "PLANNER_ERROR"]
    conflicts: list[str] = []  # NEW_SPACE_INCOMPATIBLE 时的冲突硬约束说明


class MoveTask(_Contract):
    task_id: str
    box_id: str
    group_id: str
    region_id: str
    priority: int = Field(ge=1, le=5)
    status: TaskStatus


class AuditEvent(_Contract):
    event_id: str
    event_type: str
    actor: str  # agent 名 / user / system
    input_refs: list[str] = []
    output_refs: list[str] = []
    config_version: str
    created_at: str  # ISO-8601 UTC
