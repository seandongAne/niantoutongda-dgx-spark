#!/usr/bin/env python
"""英雄链路一键主链 — 可断点续跑,阶段 = 独立子进程。

不同模型/依赖绝不塞进同一个 Python 进程:每个阶段是独立命令(本地子进程
或 ssh spark 远程命令),产物落盘,state/<stage>.json 记录输入输出 sha256。

续跑规则:输入哈希没变且全部产物哈希在盘 → 跳过;--from-stage X 强制从
X 起重跑;--only X 只跑 X。没有本地产物声明的阶段(纯远程)永不判 fresh。

Spark 阶段遵守项目纪律:执行前自动插入 healthcheck(退出码 0 才继续,
不缓存);命令以 nohup + done 标记发射,本地只轮询,断连免疫;
ssh 返回 255 重试一次。
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.schemas.hero_bundle import (  # noqa: E402
    HeroBundleManifest,
    StageArtifact,
    sha256_file,
)

# attributes(S5 读 ingest hero 图)在 reid 之前:reid 的属性分量吃 S5 产物
STAGE_ORDER = [
    "healthcheck",
    "ingest",
    "attributes",
    "reid",
    "space_observe",
    "pull",
    "inventory",
    "naming",
    "narration",
    "space",
    "space_score",
    "space_review",
    "regions",
    "group",
    "layout",
    "taskcards",
    "risk",
    "verify",
    "trace",
    "report",
    "bundle",
]


@dataclass
class Stage:
    name: str
    kind: str  # local / spark / internal
    argv: list[str] = field(default_factory=list)
    spark_cmd: str = ""
    inputs: list[Path] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    code_dependencies: list[Path] = field(default_factory=list)
    remote_sync_inputs: list[Path] = field(default_factory=list)
    remote_pull_outputs: list[Path] = field(default_factory=list)
    always_run: bool = False

    @property
    def cmd_repr(self) -> str:
        return self.spark_cmd if self.kind == "spark" else " ".join(self.argv)

    @property
    def code_paths(self) -> list[Path]:
        """阶段脚本与显式 backend 依赖均进入内容寻址续跑判据。"""
        argv_paths = [
            Path(a) for a in self.argv if a.endswith(".py") and Path(a).exists()
        ]
        return sorted({*argv_paths, *self.code_dependencies}, key=str)


def _p(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJ / path


def _acceptance_photo_paths(manifest: Path, photo_root: Path) -> list[Path]:
    """把 manifest 中的照片纳入阶段输入 hash；缺文件由主循环 fail-closed。"""

    if not manifest.exists():
        return []
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = payload.get("photos", [])
    if not isinstance(rows, list):
        raise ValueError(f"{manifest}: photos must be a list")
    paths = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not str(row.get("photo_ref", "")).strip():
            raise ValueError(f"{manifest}: photos[{index}] missing photo_ref")
        ref = Path(str(row["photo_ref"]))
        paths.append(ref if ref.is_absolute() else photo_root / ref)
    return paths


def _project_relative(path: Path) -> Path:
    """把本地项目文件映射为 spark:~/proj 下的同路径，拒绝越界。"""

    try:
        return path.resolve().relative_to(PROJ.resolve())
    except ValueError as exc:
        raise ValueError(f"Spark stage path must stay under project root: {path}") from exc


def _remote_project_arg(path: Path) -> str:
    relative = _project_relative(path)
    return relative.as_posix() if relative.parts else "."


def build_stages(
    cfg: dict, py: str, config_path: Path | None = None
) -> dict[str, Stage]:
    run = _p(cfg["run_dir"])
    trace_id = str(cfg.get("trace_id") or run.name)
    stage_cfg: dict = cfg.get("stages", {}) or {}

    def sc(name: str) -> dict:
        return stage_cfg.get(name) or {}

    stages: dict[str, Stage] = {}
    inventory_enabled = bool(sc("inventory").get("enabled"))
    space_enabled = bool(sc("space").get("enabled"))
    space_shadow_only = bool(space_enabled and sc("space").get("shadow_only"))
    space_score_enabled = bool(sc("space_score").get("enabled"))
    space_review_enabled = bool(sc("space_review").get("enabled"))
    trusted_inventory_mode = bool(
        sc("group").get("enabled")
        and sc("group").get("trusted_inventory")
    )

    if sc("healthcheck").get("enabled"):
        stages["healthcheck"] = Stage(
            "healthcheck", "local",
            argv=[str(PROJ / "scripts/spark_healthcheck.sh")], always_run=True,
        )

    for name in ("ingest", "reid", "attributes", "space_observe"):
        c = sc(name)
        if c.get("enabled"):
            stages[name] = Stage(
                name, "spark",
                spark_cmd=c["spark_cmd"],
                outputs=[_p(o) for o in c.get("local_outputs", [])],
            )

    if sc("pull").get("enabled"):
        c = sc("pull")
        stages["pull"] = Stage(
            "pull", "local",
            argv=[str(_p(a)) if i == 0 else a for i, a in enumerate(c["cmd"])],
            outputs=[_p(o) for o in c.get("local_outputs", [])],
            always_run=bool(c.get("always_run")),
        )

    if inventory_enabled:
        c = sc("inventory")
        out_dir = run / "inventory"
        trace_out = out_dir / "trace.jsonl"
        entities = _p(c["entities"])
        items = _p(c["items"])
        anchor_review = _p(c["anchor_review"])
        stages["inventory"] = Stage(
            "inventory",
            "local",
            argv=[
                py,
                str(PROJ / "scripts/inventory_task.py"),
                "--entities",
                str(entities),
                "--items",
                str(items),
                "--anchor-review",
                str(anchor_review),
                "--out-dir",
                str(out_dir),
                "--max-clarifications",
                str(c.get("max_clarifications", 4)),
                "--trace-id",
                trace_id,
                "--trace-out",
                str(trace_out),
            ],
            inputs=[entities, items, anchor_review],
            outputs=[
                out_dir / "inventory.jsonl",
                out_dir / "trusted_entities.jsonl",
                out_dir / "display.jsonl",
                out_dir / "clarifications.jsonl",
                out_dir / "metrics.json",
                out_dir / "manifest.json",
                out_dir / "hashes.json",
                trace_out,
            ],
        )

    if sc("naming").get("enabled"):
        c = sc("naming")
        out = run / "naming/display.jsonl"
        trace_out = run / "naming/trace.jsonl"
        argv = [py, str(PROJ / "scripts/entity_naming.py"),
                "--entities", str(_p(c["entities"])),
                "--attributes", str(_p(c["attributes"])),
                "--out", str(out),
                "--trace-id", trace_id, "--trace-out", str(trace_out)]
        inputs = [_p(c["entities"]), _p(c["attributes"])]
        if c.get("clarifications"):
            clarifications = _p(c["clarifications"])
            argv += ["--clarifications", str(clarifications)]
            inputs.append(clarifications)
        if c.get("tracklets_dir"):
            argv += ["--tracklets-dir", str(_p(c["tracklets_dir"]))]
        stages["naming"] = Stage("naming", "local", argv=argv,
                                 inputs=inputs, outputs=[out, trace_out])

    if sc("narration").get("enabled"):
        c = sc("narration")
        out = run / "narration/narration.jsonl"
        if c.get("transcript"):
            src = _p(c["transcript"])
            argv = [py, str(PROJ / "scripts/narration_task.py"),
                    "--transcript", str(src), "--out", str(out)]
        else:
            src = _p(c["items"])
            argv = [py, str(PROJ / "scripts/narration_task.py"),
                    "--items", str(src), "--out", str(out)]
        if c.get("audio_ref"):
            argv += ["--audio-ref", c["audio_ref"]]
        stages["narration"] = Stage("narration", "local", argv=argv,
                                    inputs=[src], outputs=[out])

    if space_enabled:
        c = sc("space")
        if c.get("anchor_candidates") and not c.get("anchor_contract"):
            raise ValueError(
                "space.anchor_candidates requires production space.anchor_contract"
            )
        if c.get("anchor_contract") and not c.get("anchor_candidates"):
            raise ValueError(
                "space.anchor_contract requires automatic space.anchor_candidates"
            )
        observations = _p(c["observations"])
        out_dir = run / "spatial"
        argv = [
            py,
            str(PROJ / "scripts/space_task.py"),
            "--video-id",
            str(c["video_id"]),
            "--observations",
            str(observations),
            "--out-dir",
            str(out_dir),
        ]
        inputs = [observations]
        for key, flag in (
            ("observation_hashes", "--observation-hashes"),
            ("anchor_candidates", "--anchor-candidates"),
            ("anchor_hashes", "--anchor-hashes"),
            ("anchor_contract", "--anchor-contract"),
        ):
            if c.get(key):
                path = _p(c[key])
                argv += [flag, str(path)]
                inputs.append(path)
        for keys, flag in (
            (("min_regions", "min"), "--min-regions"),
            (("min_observations", "min_observations_per_region"), "--min-observations"),
            (("min_confidence", "min_model_confidence"), "--min-confidence"),
            (("min_hard_field_confidence",), "--min-hard-field-confidence"),
            (("min_power_confidence",), "--min-power-confidence"),
            (("min_field_consensus",), "--min-field-consensus"),
            (("dedupe_iou", "dedupe_iou_threshold"), "--dedupe-iou"),
            (("min_anchor_vote_share",), "--min-anchor-vote-share"),
            (("min_vlm_mean_confidence",), "--min-vlm-mean-confidence"),
            (("min_assignment_score",), "--min-assignment-score"),
            (("min_assignment_margin",), "--min-assignment-margin"),
            (
                ("support_saturation_observations",),
                "--support-saturation-observations",
            ),
        ):
            value = next((c[key] for key in keys if key in c), None)
            if value is not None:
                argv += [flag, str(value)]
        expected_anchors = c.get("expected_anchor", c.get("expected_anchors", []))
        if isinstance(expected_anchors, str):
            expected_anchors = [expected_anchors]
        for anchor in expected_anchors:
            argv += ["--expected-anchor", str(anchor)]
        if c.get("allow_partial_expected_coverage"):
            argv.append("--allow-partial-expected-coverage")
        if space_shadow_only:
            argv.append("--shadow-only")
        outputs = [out_dir / "candidate_manifest.json"]
        if c.get("anchor_candidates"):
            outputs.append(out_dir / "assignment.json")
        if not space_shadow_only:
            outputs.append(out_dir / "regions.json")
        outputs.extend(
            [out_dir / "metrics.json", out_dir / "normalized.sha256"]
        )
        stages["space"] = Stage(
            "space",
            "local",
            argv=argv,
            inputs=inputs,
            outputs=outputs,
            code_dependencies=[
                PROJ / "backend/tools/spatial/producer.py",
                PROJ / "backend/tools/spatial/assignment.py",
            ],
        )

    if space_score_enabled:
        if not space_enabled or space_shadow_only:
            raise ValueError("space_score requires non-shadow automatic space")
        c = sc("space_score")
        truth = _p(c["truth"])
        source_regions = run / "spatial/regions.json"
        out_dir = run / "spatial_score"
        stages["space_score"] = Stage(
            "space_score",
            "local",
            argv=[
                py,
                str(PROJ / "scripts/space_score_task.py"),
                "--regions",
                str(source_regions),
                "--truth",
                str(truth),
                "--out-dir",
                str(out_dir),
                "--expected-count",
                str(c.get("required_expected_anchor_count", 5)),
            ],
            inputs=[source_regions, truth],
            outputs=[
                out_dir / "score_manifest.json",
                out_dir / "metrics.json",
                out_dir / "normalized.sha256",
            ],
            code_dependencies=[PROJ / "backend/tools/spatial/scoring.py"],
        )

    if space_review_enabled:
        if not space_enabled:
            raise ValueError("space_review requires the automatic space stage")
        c = sc("space_review")
        review = _p(c["review"])
        evidence_frames = [_p(path) for path in c.get("evidence_frames", [])]
        out_dir = run / "spatial_review"
        source_candidates = run / "spatial/candidate_manifest.json"
        source_hash = run / "spatial/normalized.sha256"
        stages["space_review"] = Stage(
            "space_review",
            "local",
            argv=[
                py,
                str(PROJ / "scripts/space_adjudication_task.py"),
                "--candidates",
                str(source_candidates),
                "--source-hash",
                str(source_hash),
                "--review",
                str(review),
                "--out-dir",
                str(out_dir),
            ],
            inputs=[source_candidates, source_hash, review, *evidence_frames],
            outputs=[
                out_dir / "adjudication_manifest.json",
                out_dir / "regions.json",
                out_dir / "metrics.json",
                out_dir / "normalized.sha256",
            ],
            code_dependencies=[PROJ / "backend/tools/spatial/adjudication.py"],
        )

    if sc("regions").get("enabled"):
        c = sc("regions")
        source = c.get("source")
        if source is None:
            # Backward-compatible default for old development fixtures. Hero
            # production configs set this explicitly so an automatic failure
            # can never silently fall through to a fixture.
            source = "auto" if space_enabled and not space_shadow_only else "fixture"
        if source == "auto":
            if not space_enabled or space_shadow_only:
                raise ValueError(
                    "regions.source=auto requires non-shadow automatic space"
                )
            if c.get("manifest"):
                raise ValueError(
                    "regions.source=auto forbids manifest; automatic regions have one source"
                )
            src = run / "spatial/regions.json"
        elif source == "visual_adjudication":
            if not space_review_enabled:
                raise ValueError(
                    "regions.source=visual_adjudication requires space_review"
                )
            src = run / "spatial_review/regions.json"
        elif source == "fixture":
            if not c.get("manifest"):
                raise ValueError("regions.source=fixture requires manifest")
            src = _p(c["manifest"])
        else:
            raise ValueError(
                "regions.source must be auto, visual_adjudication, or fixture"
            )
        out_dir = run / "regions"
        stages["regions"] = Stage(
            "regions", "local",
            argv=[py, str(PROJ / "scripts/regions_task.py"),
                  "--manifest", str(src), "--out-dir", str(out_dir)],
            inputs=[src],
            outputs=[out_dir / "regions.json", out_dir / "regions_core.jsonl"],
        )

    if sc("group").get("enabled"):
        c = sc("group")
        out_dir = run / "group"
        trace_out = out_dir / "trace.jsonl"
        if trusted_inventory_mode:
            closure = _p(c["closure"])
            inventory = run / "inventory/inventory.jsonl"
            display = run / "inventory/display.jsonl"
            trace_parent = run / "inventory/trace.jsonl"
            stages["group"] = Stage(
                "group",
                "local",
                argv=[
                    py,
                    str(PROJ / "scripts/trusted_group_task.py"),
                    "--closure",
                    str(closure),
                    "--inventory",
                    str(inventory),
                    "--display",
                    str(display),
                    "--out-dir",
                    str(out_dir),
                    "--trace-id",
                    trace_id,
                    "--trace-parent",
                    str(trace_parent),
                    "--trace-out",
                    str(trace_out),
                ],
                inputs=[closure, inventory, display, trace_parent],
                outputs=[
                    out_dir / "groups.jsonl",
                    out_dir / "life_groups.jsonl",
                    out_dir / "placement_groups.jsonl",
                    out_dir / "independent_pack_items.jsonl",
                    out_dir / "boxlist.json",
                    out_dir / "metrics.json",
                    trace_out,
                ],
            )
        else:
            display = run / "naming/display.jsonl"
            narration = run / "narration/narration.jsonl"
            trace_parent = run / "naming/trace.jsonl"
            argv = [py, str(PROJ / "scripts/group_task.py"),
                    "--display", str(display), "--narration", str(narration),
                    "--config-version", c.get("config_version", "group-v1"),
                    "--out-dir", str(out_dir),
                    "--trace-id", trace_id,
                    "--trace-parent", str(trace_parent),
                    "--trace-out", str(trace_out)]
            inputs = [display, narration, trace_parent]
            for opt in ("confirmations", "template_rules", "cooccurrence"):
                if c.get(opt):
                    argv += [f"--{opt.replace('_', '-')}", str(_p(c[opt]))]
                    inputs.append(_p(c[opt]))
            stages["group"] = Stage(
                "group", "local", argv=argv, inputs=inputs,
                outputs=[out_dir / "groups.jsonl", out_dir / "life_groups.jsonl",
                         out_dir / "resolutions.jsonl",
                         out_dir / "clarifications.jsonl", out_dir / "conflicts.json",
                         trace_out],
            )

    if sc("layout").get("enabled"):
        c = sc("layout")
        groups = run / (
            "group/placement_groups.jsonl"
            if trusted_inventory_mode
            else "group/groups.jsonl"
        )
        regions = run / "regions/regions.json"
        out_dir = run / "layout"
        trace_parent = run / "group/trace.jsonl"
        trace_out = out_dir / "trace.jsonl"
        argv = [py, str(PROJ / "scripts/layout_task.py"),
                "--groups", str(groups), "--regions", str(regions),
                "--out-dir", str(out_dir),
                "--trace-id", trace_id,
                "--trace-parent", str(trace_parent),
                "--trace-out", str(trace_out)]
        if c.get("requires_power_groups"):
            argv += ["--requires-power-groups", c["requires_power_groups"]]
        stages["layout"] = Stage(
            "layout", "local", argv=argv, inputs=[groups, regions, trace_parent],
            outputs=[out_dir / "plan.json", out_dir / "layout.json", trace_out],
        )

    if sc("taskcards").get("enabled"):
        out_dir = run / "taskcards"
        group_input = run / (
            "group/placement_groups.jsonl"
            if trusted_inventory_mode
            else "group/groups.jsonl"
        )
        display_input = run / (
            "inventory/display.jsonl"
            if inventory_enabled
            else "naming/display.jsonl"
        )
        inputs = [group_input, run / "layout/layout.json",
                  run / "regions/regions.json", display_input,
                  run / "layout/trace.jsonl"]
        trace_out = out_dir / "trace.jsonl"
        stages["taskcards"] = Stage(
            "taskcards", "local",
            argv=[py, str(PROJ / "scripts/taskcards_task.py"),
                  "--groups", str(inputs[0]), "--layout", str(inputs[1]),
                  "--regions", str(inputs[2]), "--display", str(inputs[3]),
                  "--out-dir", str(out_dir),
                  "--trace-id", trace_id,
                  "--trace-parent", str(inputs[4]),
                  "--trace-out", str(trace_out)],
            inputs=inputs,
            outputs=[out_dir / "taskcards.jsonl", out_dir / "taskcards.md", trace_out],
        )

    if sc("risk").get("enabled"):
        c = sc("risk")
        closure = _p(c["closure"])
        facts = _p(c["facts"])
        out_dir = run / "risk"
        stages["risk"] = Stage(
            "risk",
            "local",
            argv=[
                py,
                str(PROJ / "scripts/risk_task.py"),
                "--closure",
                str(closure),
                "--facts",
                str(facts),
                "--out-dir",
                str(out_dir),
            ],
            inputs=[closure, facts],
            outputs=[out_dir / "assessments.json", out_dir / "metrics.json"],
        )

    if sc("verify").get("enabled"):
        c = sc("verify")
        execution = str(c.get("execution", "local"))
        if execution not in {"local", "spark"}:
            raise ValueError("verify.execution must be 'local' or 'spark'")
        photos = _p(c["photos"])
        photo_root = _p(c.get("photo_root", "."))
        photo_inputs = _acceptance_photo_paths(photos, photo_root)
        cards = run / "taskcards/taskcards.jsonl"
        out_dir = run / "verify"
        trace_parent = run / "taskcards/trace.jsonl"
        argv = [py, str(PROJ / "scripts/verify_task.py"),
                "--cards", str(cards), "--photos", str(photos),
                "--photo-root", str(photo_root),
                "--out-dir", str(out_dir),
                "--trace-parent", str(trace_parent),
                "--trace-out", str(out_dir / "messages.jsonl")]
        if c.get("worker_timeout_seconds") is not None:
            argv += ["--worker-timeout-seconds", str(c["worker_timeout_seconds"])]
        outputs = [
            out_dir / "requests.jsonl",
            out_dir / "mem-results.jsonl",
            out_dir / "space-results.jsonl",
            out_dir / "messages.jsonl",
            out_dir / "verdicts.json",
            out_dir / "taskcards_verified.jsonl",
            out_dir / "fanout-run.json",
        ]
        inputs = [cards, photos, trace_parent, *photo_inputs]
        spark_cmd = ""
        if execution == "spark":
            remote_argv = [
                "python",
                "scripts/verify_task.py",
                "--cards",
                _remote_project_arg(cards),
                "--photos",
                _remote_project_arg(photos),
                "--photo-root",
                _remote_project_arg(photo_root),
                "--out-dir",
                _remote_project_arg(out_dir),
                "--trace-parent",
                _remote_project_arg(trace_parent),
                "--trace-out",
                _remote_project_arg(out_dir / "messages.jsonl"),
            ]
            if c.get("worker_timeout_seconds") is not None:
                remote_argv += [
                    "--worker-timeout-seconds",
                    str(c["worker_timeout_seconds"]),
                ]
            spark_cmd = "source ~/venv/bin/activate && " + shlex.join(remote_argv)
        stages["verify"] = Stage(
            "verify", execution,
            argv=argv if execution == "local" else [],
            spark_cmd=spark_cmd,
            inputs=inputs,
            outputs=outputs,
            code_dependencies=[
                PROJ / "scripts/verification_worker.py",
                PROJ / "backend/schemas/core.py",
                PROJ / "backend/schemas/hero_bundle.py",
                PROJ / "backend/tools/trace/store.py",
                PROJ / "backend/tools/verification/acceptance.py",
                PROJ / "backend/tools/verification/verdict.py",
            ],
            remote_sync_inputs=inputs if execution == "spark" else [],
            remote_pull_outputs=outputs if execution == "spark" else [],
        )

    if sc("trace").get("enabled"):
        c = sc("trace")
        fragments = [
            run / (
                "inventory/trace.jsonl"
                if trusted_inventory_mode
                else "naming/trace.jsonl"
            ),
            run / "group/trace.jsonl",
            run / "layout/trace.jsonl",
            run / "taskcards/trace.jsonl",
        ]
        if sc("verify").get("enabled"):
            fragments.append(run / "verify/messages.jsonl")
        out = run / "audit/events.jsonl"
        report = run / "audit/replay-report.json"
        argv = [
            py,
            str(PROJ / "scripts/replay_trace.py"),
            "--fragments",
            *[str(path) for path in fragments],
            "--out",
            str(out),
            "--report",
            str(report),
            "--require-main-chain",
        ]
        if c.get("strict"):
            argv.append("--strict")
        else:
            for key, flag in (
                ("require_verification", "--require-verification"),
                ("require_closed_choices", "--require-closed-choices"),
                ("require_adjudication", "--require-adjudication"),
            ):
                if c.get(key):
                    argv.append(flag)
        stages["trace"] = Stage(
            "trace",
            "local",
            argv=argv,
            inputs=fragments,
            outputs=[out, report],
        )

    if sc("report").get("enabled"):
        display = run / (
            "inventory/display.jsonl"
            if inventory_enabled
            else "naming/display.jsonl"
        )
        groups = run / (
            "group/placement_groups.jsonl"
            if trusted_inventory_mode
            else "group/groups.jsonl"
        )
        inputs = [display, groups, run / "layout/layout.json",
                  run / "regions/regions.json", run / "taskcards/taskcards.jsonl"]
        if inventory_enabled:
            inputs.extend([
                run / "inventory/metrics.json",
                run / "inventory/clarifications.jsonl",
            ])
        if trusted_inventory_mode:
            inputs.extend([
                run / "group/boxlist.json",
                run / "group/metrics.json",
            ])
            closure_ref = sc("group").get("closure")
            if closure_ref:
                inputs.append(_p(closure_ref))
        else:
            inputs.extend([
                run / "group/clarifications.jsonl",
                run / "group/conflicts.json",
            ])
        if space_enabled:
            inputs.append(run / "spatial/metrics.json")
            if sc("space").get("anchor_candidates"):
                inputs.append(run / "spatial/assignment.json")
        if space_score_enabled:
            inputs.append(run / "spatial_score/metrics.json")
        if space_review_enabled:
            inputs.append(run / "spatial_review/metrics.json")
        if sc("verify").get("enabled"):
            inputs.append(run / "verify/verdicts.json")
        if sc("trace").get("enabled"):
            inputs.append(run / "audit/replay-report.json")
        report_argv = [
            py,
            str(PROJ / "scripts/results_page.py"),
            "--run-dir",
            str(run),
        ]
        if config_path is not None:
            report_argv += ["--config", str(config_path)]
            inputs.append(config_path)
        stages["report"] = Stage(
            "report", "local",
            argv=report_argv,
            inputs=inputs,
            outputs=[run / "index.html"],
        )

    if sc("bundle").get("enabled", True):
        stages["bundle"] = Stage("bundle", "internal", always_run=True,
                                 outputs=[run / "bundle.json"])

    return stages


def state_path(run: Path, name: str) -> Path:
    return run / "state" / f"{name}.json"


def is_fresh(run: Path, stage: Stage) -> bool:
    if stage.always_run or not stage.outputs:
        return False
    sp = state_path(run, stage.name)
    if not sp.exists():
        return False
    state = json.loads(sp.read_text(encoding="utf-8"))
    if state.get("status") != "done":
        return False
    if state.get("cmd") != stage.cmd_repr:
        return False
    recorded_code = state.get("code", {})
    if set(recorded_code) != {str(p) for p in stage.code_paths}:
        return False
    for path_s, sha in recorded_code.items():
        if sha256_file(Path(path_s)) != sha:
            return False
    for path_s, sha in state.get("inputs", {}).items():
        path = Path(path_s)
        if not path.exists() or sha256_file(path) != sha:
            return False
    recorded = state.get("outputs", {})
    if set(recorded) != {str(o) for o in stage.outputs}:
        return False
    for path_s, sha in recorded.items():
        path = Path(path_s)
        if not path.exists() or sha256_file(path) != sha:
            return False
    return True


def write_state(run: Path, stage: Stage, started: str) -> None:
    cmd_repr = stage.cmd_repr
    sp = state_path(run, stage.name)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(
        json.dumps(
            {
                "stage": stage.name,
                "status": "done",
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "cmd": cmd_repr,
                "code": {str(p): sha256_file(p) for p in stage.code_paths},
                "inputs": {str(p): sha256_file(p) for p in stage.inputs},
                "outputs": {str(p): sha256_file(p) for p in stage.outputs},
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


SSH_HUNG = 254  # 本地合成码:连接假死被超时斩断
REMOTE_STAGE_TRANSFER_LIMIT = 50 * 1024 * 1024


def ssh(
    remote_cmd: str, timeout: int = 120, retry_255: bool = True
) -> subprocess.CompletedProcess:
    """跨境网络纪律:255(连接层失败)重试一次;假死连接超时斩断,合成 254。

    254 不自动重试:假死时远端命令可能已经执行(回包丢失),盲目重发
    会造成双重发射;交调用方处置。非幂等命令(发射类)须 retry_255=False,
    由调用方核实远端状态。管线所有 ssh 都是控制面短命令,真正的长任务
    在远端 setsid 脱离,120s 上限只斩连接不斩任务。
    """
    argv = ["ssh", "spark", remote_cmd]
    for attempt in (1, 2):
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(
                argv, SSH_HUNG, "",
                f"ssh 假死,{timeout}s 超时斩断;远端可能已执行,不自动重试",
            )
        if proc.returncode != 255 or attempt == 2 or not retry_255:
            return proc
        print(f"  ssh 255,重试一次…", file=sys.stderr)
        time.sleep(5)
    return proc


def _run_transfer(argv: list[str], *, label: str) -> None:
    """跨境小文件传输失败重试一次；仍失败则拒绝继续。"""

    proc: subprocess.CompletedProcess[str] | None = None
    for attempt in (1, 2):
        proc = subprocess.run(
            argv,
            cwd=PROJ,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return
        if attempt == 1:
            print(f"  {label} 失败,重试一次…", file=sys.stderr)
            time.sleep(1)
    assert proc is not None
    detail = proc.stderr.strip() or proc.stdout.strip()
    raise RuntimeError(f"{label} 失败(已重试一次): {detail}")


def sync_spark_stage_inputs(stage: Stage) -> None:
    """把 Spark stage 的小体积、冻结输入同步到远端同路径。"""

    if not stage.remote_sync_inputs:
        return
    relative_paths: list[str] = []
    total_bytes = 0
    seen: set[Path] = set()
    for path in stage.remote_sync_inputs:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{stage.name}: remote input must be a regular file: {path}")
        total_bytes += path.stat().st_size
        relative_paths.append(_project_relative(path).as_posix())
    if total_bytes > REMOTE_STAGE_TRANSFER_LIMIT:
        raise ValueError(
            f"{stage.name}: remote inputs total {total_bytes} bytes exceed "
            f"{REMOTE_STAGE_TRANSFER_LIMIT} byte SSH limit"
        )
    _run_transfer(
        ["rsync", "-az", "--relative", *relative_paths, "spark:~/proj/"],
        label=f"{stage.name} 输入同步",
    )


def pull_spark_stage_outputs(stage: Stage) -> None:
    """逐文件拉回声明产物，避免把 results/ 中无关大目录带回。"""

    for path in stage.remote_pull_outputs:
        relative = _project_relative(path).as_posix()
        path.parent.mkdir(parents=True, exist_ok=True)
        _run_transfer(
            ["rsync", "-az", f"spark:~/proj/{relative}", str(path.parent) + "/"],
            label=f"{stage.name} 产物拉回:{relative}",
        )


def run_spark_stage(
    stage: Stage, poll_interval: int, timeout: int, adopt: bool = False
) -> None:
    if '"' in stage.spark_cmd:
        raise ValueError(f"{stage.name}: spark_cmd 不得包含双引号(setsid 包装限制)")
    done = f"~/proj/logs/hero_{stage.name}.done"
    log = f"~/proj/logs/hero_{stage.name}.log"
    if adopt:
        # 收养模式:远端任务已在跑(或已完成),不重新发射以免双重发射,
        # 直接轮询既有 done 标记。用于本地编排器重启后接上长任务。
        print(f"  收养模式:不重新发射,轮询 {done} …")
    else:
        sync_spark_stage_inputs(stage)
        for output in stage.remote_pull_outputs:
            if output.exists():
                if not output.is_file() and not output.is_symlink():
                    raise ValueError(
                        f"{stage.name}: refusing to replace non-file output: {output}"
                    )
                output.unlink()
        # setsid -f:长任务立即脱离 ssh 会话(过继 init),发射命令前台毫秒级返回。
        # 旧 nohup+& 形状下远端 shell 会陪跑整个长任务,发射连接被拖死(s1 首跑实锤)。
        launch = (
            f"cd ~/proj && rm -f {done} {log} && "
            f'setsid -f bash -c "{stage.spark_cmd}; rc=\\$?; '
            f"printf '%s\\n' \\$rc > {done}" + '" '
            f"> {log} 2>&1 < /dev/null && echo launched"
        )
        proc = ssh(launch, retry_255=False)
        if proc.returncode != 0:
            # 连接层失败时远端可能已经开跑:以新连接核实日志(本次发射前已 rm),
            # 绝不盲目重发造成双重发射。
            verify = ssh(f"test -f {log} && echo STARTED || echo ABSENT")
            if "STARTED" not in verify.stdout:
                raise RuntimeError(
                    f"{stage.name}: 远程发射失败且未见日志\n{proc.stderr}"
                )
            print(f"  发射回包丢失但远端日志已建,视为已发射")
        print(f"  已发射(setsid),轮询 {done} …")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(poll_interval)
        check = ssh(
            f"if test -f {done}; then printf 'EXIT:'; cat {done}; "
            f"else tail -c 300 {log}; fi"
        )
        exit_lines = [
            line for line in check.stdout.splitlines() if line.startswith("EXIT:")
        ]
        if exit_lines:
            raw_code = exit_lines[-1].partition(":")[2].strip()
            returncode = int(raw_code or "0")  # 兼容旧版空 done marker
            if returncode != 0:
                failure = ssh(f"tail -c 2000 {log}")
                detail = failure.stdout.strip() or failure.stderr.strip()
                raise RuntimeError(
                    f"{stage.name}: 远程任务退出码 {returncode}\n{detail}"
                )
            pull_spark_stage_outputs(stage)
            return
        note = check.stdout.strip() or check.stderr.strip()
        print(f"  [{stage.name}] 进行中: {note[-120:]}")
    raise TimeoutError(f"{stage.name}: 超时({timeout}s),日志见 spark:{log}")


def write_bundle(run: Path, config_path: Path, stages: dict[str, Stage]) -> None:
    artifacts: list[StageArtifact] = []
    for name in STAGE_ORDER:
        sp = state_path(run, name)
        if name == "bundle" or name not in stages or not sp.exists():
            continue
        state = json.loads(sp.read_text(encoding="utf-8"))
        for path_s, sha in sorted(state.get("outputs", {}).items()):
            artifacts.append(StageArtifact(stage=name, path=path_s, sha256=sha))
    manifest = HeroBundleManifest(
        bundle_id=run.name,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        config_refs={config_path.name: sha256_file(config_path)},
        artifacts=artifacts,
    )
    (run / "bundle.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--run-dir", type=Path, default=None, help="覆盖配置中的 run_dir")
    ap.add_argument("--from-stage", default=None)
    ap.add_argument("--until-stage", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument(
        "--adopt-stage",
        default=None,
        help="该 spark 阶段不重新发射,直接轮询既有 done 标记(收养已在跑/已完成的远端任务)",
    )
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--poll-interval", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.run_dir:
        cfg["run_dir"] = str(args.run_dir)
    run = _p(cfg["run_dir"])
    py = cfg.get("python", sys.executable)
    stages = build_stages(cfg, py, config_path=args.config.resolve())
    order = [n for n in STAGE_ORDER if n in stages]

    for flag in (args.from_stage, args.until_stage, args.only, args.adopt_stage):
        if flag and flag not in stages:
            ap.error(f"阶段 {flag} 未启用;已启用: {', '.join(order)}")
    if args.adopt_stage and stages[args.adopt_stage].kind != "spark":
        ap.error(f"--adopt-stage 只适用于 spark 阶段: {args.adopt_stage}")

    if args.only:
        selected, forced = [args.only], {args.only}
    else:
        selected = list(order)
        if args.until_stage:
            selected = selected[: selected.index(args.until_stage) + 1]
        forced = set()
        if args.from_stage:
            selected_from = selected[selected.index(args.from_stage):]
            selected = selected_from
            forced = set(selected_from)

    plan: list[tuple[Stage, str]] = []
    for name in selected:
        stage = stages[name]
        action = "run" if name in forced or not is_fresh(run, stage) else "skip"
        plan.append((stage, action))

    # 项目纪律:任何 spark 阶段实际执行前必须过 healthcheck
    if any(s.kind == "spark" and a == "run" for s, a in plan) and not any(
        s.name == "healthcheck" and a == "run" for s, a in plan
    ):
        hc = stages.get("healthcheck") or Stage(
            "healthcheck", "local",
            argv=[str(PROJ / "scripts/spark_healthcheck.sh")], always_run=True,
        )
        plan.insert(0, (hc, "run"))

    if args.list or args.dry_run:
        for stage, action in plan:
            cmd = stage.spark_cmd if stage.kind == "spark" else " ".join(stage.argv)
            print(f"[{action:4}] {stage.name:<12} ({stage.kind}) {cmd[:100]}")
        return 0

    for stage, planned_action in plan:
        # 上游在本次运行中重写了产物时，计划阶段的 skip 可能已经过期。
        # 执行到每一阶段前重新做一次内容寻址判断；若上游重跑但字节未变，
        # 该阶段仍会保持 skip，保留“产物不变则下游不陪跑”的语义。
        action = planned_action
        if action == "skip" and not is_fresh(run, stage):
            action = "run"
        if action == "skip":
            print(f"[skip] {stage.name}(输入未变,产物在盘)")
            continue
        print(f"[run ] {stage.name} …")
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        missing = [str(p) for p in stage.inputs if not p.exists()]
        if missing:
            print(f"  缺少输入: {missing}", file=sys.stderr)
            return 2
        if stage.kind == "internal":
            write_bundle(run, args.config, stages)
        elif stage.kind == "spark":
            run_spark_stage(
                stage,
                args.poll_interval,
                args.timeout,
                adopt=(stage.name == args.adopt_stage),
            )
        else:
            proc = subprocess.run(stage.argv, cwd=PROJ, capture_output=True, text=True)
            if proc.stdout.strip():
                print("  " + proc.stdout.strip().replace("\n", "\n  "))
            if proc.returncode != 0:
                print(proc.stderr, file=sys.stderr)
                print(f"[fail] {stage.name} 退出码 {proc.returncode}", file=sys.stderr)
                return proc.returncode
        lost = [str(p) for p in stage.outputs if not p.exists()]
        if lost:
            print(f"[fail] {stage.name} 声明产物缺失: {lost}", file=sys.stderr)
            return 2
        write_state(run, stage, started)
    print(f"✅ 主链完成: {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
