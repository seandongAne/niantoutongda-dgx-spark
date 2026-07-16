#!/usr/bin/env python
"""Launch the complete synthetic A1 benchmark inside Spark without moving audio.

The local ``launch`` command sends one disposable StepFun key over SSH stdin.  The
remote ``worker`` detaches, generates the deterministic plan and synthetic audio,
runs cloud extraction, drops the key from its environment, runs local Step-Audio,
and exports a small acceptance report.  The credential is never written to disk,
put in an argv, or rendered in logs.  Because Spark was previously compromised,
the key must still be treated as exposed and revoked after the job.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "scripts"))

from backend.tools.a1_benchmark import build_plan  # noqa: E402
from scripts import a1_robustness  # noqa: E402
from stepfun_api import load_key  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _csv(value: str) -> list[str]:
    parsed = [part.strip() for part in value.split(",") if part.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("value must contain at least one item")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_remote_command(args: argparse.Namespace) -> str:
    """Build a fixed-code remote command; the API key deliberately is not an arg."""
    worker_args = [
        "scripts/a1_spark_factory.py",
        "worker",
        "--run-dir",
        args.run_dir,
        "--report-dir",
        args.report_dir,
        "--base-cases",
        str(args.base_cases),
        "--conditions",
        args.conditions,
        "--voices",
        args.voices,
        "--seed",
        str(args.seed),
        "--ci-half-width",
        str(args.ci_half_width),
        "--min-per-condition",
        str(args.min_per_condition),
        "--max-observations",
        str(args.max_observations),
        "--max-new-tts",
        str(args.max_new_tts),
        "--max-new-extractions",
        str(args.max_new_extractions),
        "--max-new-local",
        str(args.max_new_local),
        "--revocation-delay-days",
        str(args.revocation_delay_days),
        "--log",
        args.log,
    ]
    return "cd ~/proj && exec ~/envs/stepaudio/bin/python " + shlex.join(worker_args)


def cmd_launch(args: argparse.Namespace) -> int:
    if not args.acknowledge_key_exposure:
        raise SystemExit(
            "refusing to expose a credential to Spark; create a disposable key, "
            "accept that it is compromised on injection, and pass "
            "--acknowledge-key-exposure only for that one run"
        )
    key = load_key()
    command = build_remote_command(args)
    try:
        process = subprocess.run(
            ["ssh", args.host, command],
            input=key + "\n",
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        key = ""
    if process.stdout:
        print(process.stdout, end="")
    if process.stderr:
        print(process.stderr, end="", file=sys.stderr)
    if process.returncode != 0:
        raise SystemExit(f"Spark factory launch failed with exit code {process.returncode}")
    return 0


def _redirect_and_detach(log_path: Path) -> int | None:
    """Fork once, detach the child, and redirect all standard streams.

    The parent returns the child PID.  The detached child returns ``None``.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_pid = os.fork()
    if child_pid:
        return child_pid
    os.setsid()
    os.umask(0o077)
    devnull = os.open(os.devnull, os.O_RDONLY)
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.dup2(devnull, sys.stdin.fileno())
    os.dup2(log_fd, sys.stdout.fileno())
    os.dup2(log_fd, sys.stderr.fileno())
    if devnull > 2:
        os.close(devnull)
    if log_fd > 2:
        os.close(log_fd)
    return None


def _expected_plan(args: argparse.Namespace) -> dict[str, Any]:
    return build_plan(
        seed=args.seed,
        base_cases=args.base_cases,
        condition_ids=_csv(args.conditions),
        voices=_csv(args.voices),
        target_half_width=args.ci_half_width,
        minimum_per_condition=args.min_per_condition,
        maximum_observations=args.max_observations,
    )


def ensure_plan(args: argparse.Namespace) -> None:
    plan_path = Path(args.run_dir) / "plan.json"
    expected = _expected_plan(args)
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != expected:
            raise RuntimeError(f"existing plan differs from requested deterministic plan: {plan_path}")
        print(f"plan resume {plan_path}", flush=True)
        return
    write_json(plan_path, expected)
    print(
        f"plan created base_cases={expected['base_case_count']} "
        f"observations_per_backend={expected['observation_count_per_backend']}",
        flush=True,
    )


def _set_status(path: Path, *, phase: str, state: str, args: argparse.Namespace, **extra: Any) -> None:
    write_json(path, {
        "schema_version": "a1-spark-factory-v1",
        "updated_at": utc_now(),
        "state": state,
        "phase": phase,
        "factory_code_revision": a1_robustness.git_revision(PROJ),
        "run_dir": args.run_dir,
        "report_dir": args.report_dir,
        "credential_policy": "stdin_memory_only_treat_as_exposed",
        "revocation_delay_days_after_completion": args.revocation_delay_days,
        "transfer_policy": "code_and_small_reports_only_no_audio_over_ssh",
        "caps": {
            "tts": args.max_new_tts,
            "cloud_extractions": args.max_new_extractions,
            "local_extractions": args.max_new_local,
        },
        **extra,
    })


def _run_cloud(args: argparse.Namespace) -> None:
    result = a1_robustness.main([
        "cloud",
        "--run-dir",
        args.run_dir,
        "--temperature",
        "0",
        "--max-new-tts",
        str(args.max_new_tts),
        "--max-new-extractions",
        str(args.max_new_extractions),
    ])
    if result != 0:
        raise RuntimeError(f"cloud stage exited with {result}")


def _run_local(args: argparse.Namespace) -> None:
    subprocess.run(
        [
            sys.executable,
            str(PROJ / "scripts" / "a1_stepaudio_local.py"),
            "--run-dir",
            args.run_dir,
            "--max-new",
            str(args.max_new_local),
        ],
        cwd=PROJ,
        check=True,
    )


def _score(args: argparse.Namespace) -> None:
    result = a1_robustness.main([
        "score",
        "--run-dir",
        args.run_dir,
        "--report-dir",
        args.report_dir,
    ])
    if result != 0:
        raise RuntimeError(f"score stage exited with {result}")


def run_worker(args: argparse.Namespace, key: str) -> int:
    run_dir = Path(args.run_dir)
    status_path = run_dir / "factory-status.json"
    started_at = utc_now()
    try:
        _set_status(status_path, phase="plan", state="running", args=args, started_at=started_at)
        ensure_plan(args)

        _set_status(status_path, phase="cloud", state="running", args=args, started_at=started_at)
        os.environ["STEPFUN_API_KEY"] = key
        key = ""
        _run_cloud(args)
        os.environ.pop("STEPFUN_API_KEY", None)
        _set_status(
            status_path,
            phase="local",
            state="running",
            args=args,
            started_at=started_at,
            credential_released_at=utc_now(),
        )

        subprocess.run(["free", "-h"], check=True)
        _run_local(args)

        _set_status(status_path, phase="score", state="running", args=args, started_at=started_at)
        _score(args)
        _set_status(
            status_path,
            phase="complete",
            state="complete",
            args=args,
            started_at=started_at,
            completed_at=utc_now(),
        )
        report_status = Path(args.report_dir) / "factory-status.json"
        report_status.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(status_path, report_status)
        print("A1_SPARK_FACTORY_DONE", flush=True)
        return 0
    except BaseException as exc:
        os.environ.pop("STEPFUN_API_KEY", None)
        try:
            _set_status(
                status_path,
                phase="failed",
                state="failed",
                args=args,
                started_at=started_at,
                failed_at=utc_now(),
                error_type=type(exc).__name__,
                error=str(exc)[:1000],
            )
        except Exception:
            traceback.print_exc()
        traceback.print_exc()
        return 1


def cmd_worker(args: argparse.Namespace) -> int:
    key = sys.stdin.readline().strip()
    if not key:
        raise SystemExit("missing disposable StepFun key on stdin")
    child_pid = _redirect_and_detach(Path(args.log))
    if child_pid is not None:
        key = ""
        print(f"A1_SPARK_FACTORY_STARTED pid={child_pid} log={args.log}", flush=True)
        return 0
    exit_code = run_worker(args, key)
    os._exit(exit_code)


def add_job_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--base-cases", type=int, default=6)
    parser.add_argument("--conditions", default="clean,noise20,speed090")
    parser.add_argument("--voices", default="cixingnansheng,linjiajiejie")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--ci-half-width", type=float, default=0.08)
    parser.add_argument("--min-per-condition", type=int, default=100)
    parser.add_argument("--max-observations", type=int, default=2000)
    parser.add_argument("--max-new-tts", type=int, default=6)
    parser.add_argument("--max-new-extractions", type=int, default=18)
    parser.add_argument("--max-new-local", type=int, default=18)
    parser.add_argument(
        "--revocation-delay-days",
        type=_nonnegative_int,
        default=0,
        help="audited delay after completion before the disposable key is deleted",
    )
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
