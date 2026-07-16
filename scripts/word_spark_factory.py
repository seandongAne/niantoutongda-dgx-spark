#!/usr/bin/env python
"""词表工厂原地版 — 云候选生成 + GDINO 扫描 + 判卷全部在 Spark 数据平面。

SSH 限流后的形态:本地只发射(launch)和收小报告;物品清单/GT/帧
全部走 deploy 或本就长在节点上;逐帧预测(大头)永不过境。

  launch(本地) --items fixtures/.../items.json --frames-dir ... [--gt ...]
    → SSH stdin 送一次性 key → worker(spark)脱离后依次:
      gen(云,一次 chat 调用,key 用完即弹) → scan(GDINO,先 free -h)
      → rank(有 GT)或 deadwords 摘要(无 GT) → 小报告落 report-dir。

凭据纪律见 spark_factory_common.py;云端输出只是候选,入词表前
必须过本判卷回路(有 GT)或后续人工裁决(无 GT)。
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Sequence

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "scripts"))

from spark_factory_common import (  # noqa: E402
    ACK_REFUSAL,
    launch_worker,
    read_stdin_key,
    redirect_and_detach,
    utc_now,
    write_json,
    write_status,
)
from stepfun_api import load_key  # noqa: E402

STATUS_SCHEMA = "word-spark-factory-v1"


def build_remote_command(args: argparse.Namespace) -> str:
    """固定代码的远端命令;key 有意不在参数里。"""
    worker_args = [
        "scripts/word_spark_factory.py", "worker",
        "--items", args.items,
        "--frames-dir", args.frames_dir,
        "--run-dir", args.run_dir,
        "--report-dir", args.report_dir,
        "--gen-model", args.gen_model,
        "--temperature", str(args.temperature),
        "--gdino-model", args.gdino_model,
        "--box-threshold", str(args.box_threshold),
        "--max-phrases", str(args.max_phrases),
        "--log", args.log,
    ]
    if args.gt:
        worker_args += ["--gt", args.gt]
    return "cd ~/proj && exec ~/venv/bin/python " + shlex.join(worker_args)


def cmd_launch(args: argparse.Namespace) -> int:
    if not args.acknowledge_key_exposure:
        raise SystemExit(ACK_REFUSAL)
    return launch_worker(args.host, build_remote_command(args), load_key())


def _run_gen(args: argparse.Namespace, key: str) -> Path:
    out = Path(args.run_dir) / "candidates.json"
    env = dict(os.environ)
    env["STEPFUN_API_KEY"] = key
    subprocess.run(
        [sys.executable, str(PROJ / "scripts/vocab_candidates_gen.py"),
         "--items", args.items, "--model", args.gen_model,
         "--temperature", str(args.temperature), "--out", str(out)],
        cwd=PROJ, check=True, env=env,
    )
    candidates = json.loads(out.read_text(encoding="utf-8"))
    n_phrases = sum(len(v) for v in candidates.values())
    if n_phrases > args.max_phrases:
        raise RuntimeError(
            f"候选短语 {n_phrases} 条超过 --max-phrases {args.max_phrases},"
            "拒绝扫描;提高上限或减物品重跑"
        )
    print(f"gen ok: {len(candidates)} categories, {n_phrases} phrases", flush=True)
    return out


def _run_scan(args: argparse.Namespace, candidates: Path) -> Path:
    subprocess.run(["free", "-h"], check=True)
    scan_dir = Path(args.run_dir) / "scan"
    subprocess.run(
        [sys.executable, str(PROJ / "scripts/word_candidate_scan.py"),
         "--candidates", str(candidates), "--frames-dir", args.frames_dir,
         "--model", args.gdino_model, "--box-threshold", str(args.box_threshold),
         "--out-dir", str(scan_dir)],
        cwd=PROJ, check=True,
    )
    return scan_dir


def _run_rank(args: argparse.Namespace, scan_dir: Path) -> None:
    subprocess.run(
        [sys.executable, str(PROJ / "scripts/word_candidate_rank.py"),
         "--gt", args.gt, "--scan-dir", str(scan_dir),
         "--out", str(Path(args.report_dir) / "ranking.json")],
        cwd=PROJ, check=True,
    )


def _write_deadwords(args: argparse.Namespace, scan_dir: Path) -> None:
    """无 GT 时的最小裁决材料:哪些短语一框都打不出来(死词)。"""
    manifest = json.loads((scan_dir / "scan-manifest.json").read_text(encoding="utf-8"))
    rows = {
        key: entry["detections"]
        for key, entry in manifest["candidates"].items()
    }
    write_json(Path(args.report_dir) / "deadwords.json", {
        "note": "无 GT 判卷,仅死词/触发计数;入词表仍需 GT 判卷或人工裁决",
        "dead": sorted(k for k, n in rows.items() if n == 0),
        "detections_per_phrase": rows,
    })


def run_worker(args: argparse.Namespace, key: str) -> int:
    status_path = Path(args.run_dir) / "factory-status.json"
    started_at = utc_now()

    def status(phase: str, state: str, **extra) -> None:
        write_status(
            status_path, schema=STATUS_SCHEMA, phase=phase, state=state,
            run_dir=args.run_dir, report_dir=args.report_dir,
            caps={"max_phrases": args.max_phrases},
            started_at=started_at, **extra,
        )

    try:
        status("gen", "running")
        candidates = _run_gen(args, key)
        key = ""
        status("scan", "running", credential_released_at=utc_now())

        scan_dir = _run_scan(args, candidates)

        status("report", "running")
        Path(args.report_dir).mkdir(parents=True, exist_ok=True)
        if args.gt:
            _run_rank(args, scan_dir)
        else:
            _write_deadwords(args, scan_dir)
        # 报告目录自带扫描指纹与 token 账
        for name in ("scan-manifest.json",):
            (Path(args.report_dir) / name).write_bytes((scan_dir / name).read_bytes())
        (Path(args.report_dir) / "candidates.manifest.json").write_bytes(
            candidates.with_suffix(".manifest.json").read_bytes()
        )

        status("complete", "complete", completed_at=utc_now())
        print("WORD_SPARK_FACTORY_DONE", flush=True)
        return 0
    except BaseException as exc:
        key = ""
        try:
            status(
                "failed", "failed", failed_at=utc_now(),
                error_type=type(exc).__name__, error=str(exc)[:1000],
            )
        except Exception:
            traceback.print_exc()
        traceback.print_exc()
        return 1


def cmd_worker(args: argparse.Namespace) -> int:
    key = read_stdin_key()
    child_pid = redirect_and_detach(Path(args.log))
    if child_pid is not None:
        key = ""
        print(f"WORD_SPARK_FACTORY_STARTED pid={child_pid} log={args.log}", flush=True)
        return 0
    exit_code = run_worker(args, key)
    os._exit(exit_code)


def add_job_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--items", required=True, help="物品清单 JSON(仓库内路径,经 deploy 上节点)")
    parser.add_argument("--frames-dir", required=True, help="扫描帧目录(节点上)")
    parser.add_argument("--gt", default=None, help="GT JSON;缺省只出死词摘要不判卷")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--report-dir", required=True, help="小报告目录(可拉回)")
    parser.add_argument("--gen-model", default="step-3.5-flash")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--gdino-model",
        default="/home/Developer/models/IDEA-Research__grounding-dino-base",
    )
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--max-phrases", type=int, default=90, help="扫描上限(GPU 时长护栏)")
    parser.add_argument("--log", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    launch = sub.add_parser("launch", help="send a disposable key on stdin and start Spark worker")
    launch.add_argument("--host", default="spark")
    launch.add_argument(
        "--acknowledge-key-exposure",
        action="store_true",
        help="confirm this one-run key is treated as compromised and will be revoked",
    )
    add_job_arguments(launch)
    launch.set_defaults(func=cmd_launch)
    worker = sub.add_parser("worker", help="Spark-side detached worker")
    add_job_arguments(worker)
    worker.set_defaults(func=cmd_worker)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
