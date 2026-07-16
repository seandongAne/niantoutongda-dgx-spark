"""S0 数据契约 — 手册 §6 冻结的 schema。所有对象带 schema_version。

设计文档 §7.1 是字段的权威来源;这里只做最小必要扩展(枚举、校验)。
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = SCHEMA_VERSION


class AgentRole(str, Enum):
    MEM = "MEM"  # 物品记忆
    GROUP = "GROUP"  # 生活组合
    SPACE = "SPACE"  # 空间规划
    EXEC = "EXEC"  # 搬家执行
    USER = "USER"
    SYSTEM = "SYSTEM"


class _Message(_Contract):
    """跨 Agent 消息基类 — 追加写入后不可变。

    correlation_id 串起一次完整往返(如一次验收复核);causation_id 指向
    触发本消息的上游 message_id;payload_hash 由 compute_payload_hash
    计算(排除自身),回放时校验消息未被篡改。
    """

    message_id: str
    correlation_id: str
    causation_id: Optional[str] = None
    producer: AgentRole
    payload_hash: str = ""


def compute_payload_hash(msg: _Message) -> str:
    data = msg.model_dump(mode="json")
    data.pop("payload_hash", None)
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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


class ClarificationRequest(_Message):
    """MEM → UI 的实例二选一请求。

    早期 S3 证据只含 request_id/candidates。``_legacy_message_fields`` 让这些
    已归档 JSONL 仍可读取；当前生产者必须显式写消息基字段并在落盘前计算
    payload_hash，严格 trace 回放会拒绝空 hash。
    """

    target: Literal[AgentRole.USER] = AgentRole.USER
    request_id: str
    candidate_a: str  # tracklet/entity id
    candidate_b: str
    reason_codes: list[str]
    decision: Optional[Literal["same", "different"]] = None

    @model_validator(mode="before")
    @classmethod
    def _legacy_message_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "message_id" in value:
            return value
        data = dict(value)
        request_id = str(data.get("request_id", ""))
        data.update(
            message_id=request_id,
            correlation_id=f"clarify-{request_id}",
            producer=AgentRole.MEM,
            payload_hash="",
        )
        return data


class ClarificationDecision(_Message):
    """UI → MEM 的二选一答复；与 request 分离，避免原消息被回写。"""

    target: Literal[AgentRole.MEM] = AgentRole.MEM
    request_id: str
    decision: Literal["same", "different"]
    note: str = ""


class AgentHandoff(_Message):
    """四 Agent 主链的不可变阶段交接消息。"""

    target: AgentRole
    action: Literal[
        "ENTITIES_READY",
        "GROUPS_READY",
        "PLACEMENT_READY",
        "TASKS_READY",
    ]
    item_ids: list[str] = []
    artifact_refs: list[str] = []
    summary: dict[str, str | int | float | bool] = {}

    @model_validator(mode="after")
    def _enforce_route(self) -> "AgentHandoff":
        routes = {
            "ENTITIES_READY": (AgentRole.MEM, AgentRole.GROUP),
            "GROUPS_READY": (AgentRole.GROUP, AgentRole.SPACE),
            "PLACEMENT_READY": (AgentRole.SPACE, AgentRole.EXEC),
            "TASKS_READY": (AgentRole.EXEC, AgentRole.USER),
        }
        expected = routes[self.action]
        if (self.producer, self.target) != expected:
            raise ValueError(
                f"{self.action} route must be {expected[0].value}->{expected[1].value}"
            )
        return self


class RelationEdge(_Contract):
    src: str
    dst: str
    relation: Literal["NEAR", "SAME_SURFACE", "CO_USED", "REQUIRES_POWER", "STORED_WITH", "ANCHORED_TO"]
    evidence_refs: list[str] = []


# ---- 验收复核消息族(EXEC 发起的反向协同) ----
# 职责拆分:MEM 只回答"物品是否出现"(presence),SPACE/确定性校验器只回答
# "目标区域与关系是否满足"(compliance);EXEC 汇总为 verdict,
# VERIFIED 必须 presence 与 compliance 同时通过。任何一方不得代写结论。


class VerificationCheckRequest(_Message):
    """EXEC → MEM 与 SPACE:请求对验收照片复核。producer=EXEC。"""

    task_id: str
    expected_entity_ids: list[str]
    target_region_id: str
    expected_relation_edges: list[RelationEdge] = []
    photo_refs: list[str]


class EntityPresence(_Contract):
    entity_id: str
    present: bool
    match_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    evidence_refs: list[str] = []


class ObjectPresenceCheckResult(_Message):
    """MEM → EXEC:只回答物品是否出现在照片中,不判断摆放。producer=MEM。"""

    request_id: str
    presences: list[EntityPresence]
    reason_codes: list[str] = []


class PlacementCompliance(_Contract):
    entity_id: str
    region_ok: bool
    relations_ok: bool
    violated_constraints: list[str] = []


class PlacementComplianceResult(_Message):
    """SPACE/确定性校验器 → EXEC:只回答区域与关系约束是否满足。producer=SPACE。"""

    request_id: str
    compliances: list[PlacementCompliance]
    reason_codes: list[str] = []


class VerificationVerdict(_Message):
    """EXEC 汇总:presence ∧ compliance 才 VERIFIED。producer=EXEC。"""

    request_id: str
    presence_result_id: str
    compliance_result_id: str
    verdict: Literal["VERIFIED", "FAILED", "NEEDS_USER"]
    reason_codes: list[str] = []


class UserAdjudication(_Message):
    """用户对 FAILED/NEEDS_USER 的裁决 — 独立不可变消息。producer=USER。"""

    verdict_id: str
    decision: Literal["accept_override", "reject_redo"]
    note: str = ""


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
