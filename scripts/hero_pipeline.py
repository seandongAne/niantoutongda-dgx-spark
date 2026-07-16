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

STAGE_ORDER = [
    "healthcheck",
    "ingest",
    "reid",
    "attributes",
    "pull",
    "naming",
    "narration",
    "regions",
    "group",
    "layout",
    "taskcards",
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
    always_run: bool = False

    @property
    def cmd_repr(self) -> str:
        return self.spark_cmd if self.kind == "spark" else " ".join(self.argv)

    @property
    def code_paths(self) -> list[Path]:
        """阶段脚本本身也是续跑判据的一部分;backend 模块改动请用 --from-stage。"""
        return [Path(a) for a in self.argv if a.endswith(".py") and Path(a).exists()]


def _p(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJ / path


def build_stages(cfg: dict, py: str) -> dict[str, Stage]:
    run = _p(cfg["run_dir"])
    stage_cfg: dict = cfg.get("stages", {}) or {}

    def sc(name: str) -> dict:
        return stage_cfg.get(name) or {}

    stages: dict[str, Stage] = {}

    if sc("healthcheck").get("enabled"):
        stages["healthcheck"] = Stage(
            "healthcheck", "local",
            argv=[str(PROJ / "scripts/spark_healthcheck.sh")], always_run=True,
        )

    for name in ("ingest", "reid", "attributes"):
        c = sc(name)
        if c.get("enabled"):
            stages[name] = Stage(
                name, "spark",
                spark_cmd=c["spark_cmd"],
                outputs=[_p(o) for o in c.get("local_outputs", [])],
            )

    if sc("pull").get("enabled"):
        stages["pull"] = Stage(
            "pull", "local",
            argv=[str(_p(a)) if i == 0 else a for i, a in enumerate(sc("pull")["cmd"])],
            outputs=[_p(o) for o in sc("pull").get("local_outputs", [])],
        )

    if sc("naming").get("enabled"):
        c = sc("naming")
        out = run / "naming/display.jsonl"
        argv = [py, str(PROJ / "scripts/entity_naming.py"),
                "--entities", str(_p(c["entities"])),
                "--attributes", str(_p(c["attributes"])),
                "--out", str(out)]
        inputs = [_p(c["entities"]), _p(c["attributes"])]
        if c.get("tracklets_dir"):
            argv += ["--tracklets-dir", str(_p(c["tracklets_dir"]))]
        stages["naming"] = Stage("naming", "local", argv=argv,
                                 inputs=inputs, outputs=[out])

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

    if sc("regions").get("enabled"):
        c = sc("regions")
        src = _p(c["manifest"])
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
        display = run / "naming/display.jsonl"
        narration = run / "narration/narration.jsonl"
        out_dir = run / "group"
        argv = [py, str(PROJ / "scripts/group_task.py"),
                "--display", str(display), "--narration", str(narration),
                "--config-version", c.get("config_version", "group-v1"),
                "--out-dir", str(out_dir)]
        inputs = [display, narration]
        for opt in ("confirmations", "template_rules", "cooccurrence"):
            if c.get(opt):
                argv += [f"--{opt.replace('_', '-')}", str(_p(c[opt]))]
                inputs.append(_p(c[opt]))
        stages["group"] = Stage(
            "group", "local", argv=argv, inputs=inputs,
            outputs=[out_dir / "groups.jsonl", out_dir / "life_groups.jsonl",
                     out_dir / "resolutions.jsonl",
                     out_dir / "clarifications.jsonl", out_dir / "conflicts.json"],
        )

    if sc("layout").get("enabled"):
        c = sc("layout")
        groups = run / "group/groups.jsonl"
        regions = run / "regions/regions.json"
        out_dir = run / "layout"
        argv = [py, str(PROJ / "scripts/layout_task.py"),
                "--groups", str(groups), "--regions", str(regions),
                "--out-dir", str(out_dir)]
        if c.get("requires_power_groups"):
            argv += ["--requires-power-groups", c["requires_power_groups"]]
        stages["layout"] = Stage(
            "layout", "local", argv=argv, inputs=[groups, regions],
            outputs=[out_dir / "plan.json", out_dir / "layout.json"],
        )

    if sc("taskcards").get("enabled"):
        out_dir = run / "taskcards"
        inputs = [run / "group/groups.jsonl", run / "layout/layout.json",
                  run / "regions/regions.json", run / "naming/display.jsonl"]
        stages["taskcards"] = Stage(
            "taskcards", "local",
            argv=[py, str(PROJ / "scripts/taskcards_task.py"),
                  "--groups", str(inputs[0]), "--layout", str(inputs[1]),
                  "--regions", str(inputs[2]), "--display", str(inputs[3]),
                  "--out-dir", str(out_dir)],
            inputs=inputs,
            outputs=[out_dir / "taskcards.jsonl", out_dir / "taskcards.md"],
        )

    if sc("report").get("enabled"):
        inputs = [run / "naming/display.jsonl", run / "group/groups.jsonl",
                  run / "layout/layout.json", run / "regions/regions.json",
                  run / "taskcards/taskcards.jsonl"]
        stages["report"] = Stage(
            "report", "local",
            argv=[py, str(PROJ / "scripts/results_page.py"),
                  "--run-dir", str(run)],
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


def ssh(remote_cmd: str) -> subprocess.CompletedProcess:
    """跨境网络纪律:255(连接层失败)重试一次。"""
    for attempt in (1, 2):
        proc = subprocess.run(
            ["ssh", "spark", remote_cmd], capture_output=True, text=True
        )
        if proc.returncode != 255 or attempt == 2:
            return proc
        print(f"  ssh 255,重试一次…", file=sys.stderr)
        time.sleep(5)
    return proc


def run_spark_stage(stage: Stage, poll_interval: int, timeout: int) -> None:
    if '"' in stage.spark_cmd:
        raise ValueError(f"{stage.name}: spark_cmd 不得包含双引号(nohup 包装限制)")
    done = f"~/proj/logs/hero_{stage.name}.done"
    log = f"~/proj/logs/hero_{stage.name}.log"
    launch = (
        f"cd ~/proj && rm -f {done} && "
        f'nohup bash -c "{stage.spark_cmd} && touch {done}" > {log} 2>&1 & echo launched'
    )
    proc = ssh(launch)
    if proc.returncode != 0:
        raise RuntimeError(f"{stage.name}: 远程发射失败\n{proc.stderr}")
    print(f"  已发射(nohup),轮询 {done} …")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(poll_interval)
        check = ssh(f"test -f {done} && echo DONE || tail -c 300 {log}")
        if "DONE" in check.stdout:
            return
        print(f"  [{stage.name}] 进行中: {check.stdout.strip()[-120:]}")
    raise TimeoutError(f"{stage.name}: 超时({timeout}s),日志见 spark:{log}")


def write_bundle(run: Path, config_path: Path, stages: dict[str, Stage]) -> None:
    artifacts: list[StageArtifact] = []
    for name in STAGE_ORDER:
        sp = state_path(run, name)
        if name == "bundle" or not sp.exists():
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
    stages = build_stages(cfg, py)
    order = [n for n in STAGE_ORDER if n in stages]

    for flag in (args.from_stage, args.until_stage, args.only):
        if flag and flag not in stages:
            ap.error(f"阶段 {flag} 未启用;已启用: {', '.join(order)}")

    if args.only:
        selected, forced = [args.only], {args.only}
    else:
        selected = list(order)
        if args.until_stage:
            selected = selected[: selected.index(args.until_stage) + 1]
        forced = set()
        if args.from_stage:
            selected_from = selected[selected.index(args.from_stage):]
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

    for stage, action in plan:
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
            run_spark_stage(stage, args.poll_interval, args.timeout)
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
