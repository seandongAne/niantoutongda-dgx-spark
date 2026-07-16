"""Spark 原地 StepFun 工厂公共通道 — 一次性密钥纪律的唯一实现点。

通道纪律(docs/STEPFUN_API_PLAYBOOK.md,SSH 限流后的原地工厂通道):
- 只接受一次性、可撤销 key;launch 必须显式 --acknowledge-key-exposure,
  从注入时刻起按已泄露处理,批次结束即撤销。
- key 经 SSH stdin 单行进入远端 worker 进程内存;绝不进 argv、日志、
  状态文件、.env 或任何磁盘落点;云阶段短暂进入进程 env,用完立即弹出。
- worker fork+setsid 脱离 SSH 会话(断连免疫),日志 0600。
- 状态文件只记阶段/上限/错误/凭据策略,不含凭据。

a1_spark_factory.py 先于本模块存在且自带同一纪律的实现;其所在农场
空闲时迁移到这里,迁移前两处实现必须保持逐字段一致。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CREDENTIAL_POLICY = "stdin_memory_only_treat_as_exposed_revoke_after_job"
TRANSFER_POLICY = "code_and_small_reports_only_no_bulk_over_ssh"

ACK_REFUSAL = (
    "refusing to expose a credential to Spark; create a disposable key, "
    "accept that it is compromised on injection, and pass "
    "--acknowledge-key-exposure only for that one run"
)


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


def launch_worker(host: str, remote_command: str, key: str | None) -> int:
    """本地 → SSH 发射远端 worker;key 只走 stdin,绝不进命令行。

    key=None 表示本次任务不含云阶段(免凭据),stdin 不送任何内容。
    """
    if key == "":
        raise SystemExit("missing disposable StepFun key")
    try:
        process = subprocess.run(
            ["ssh", host, remote_command],
            input=(key + "\n") if key is not None else "",
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


def read_stdin_key() -> str:
    key = sys.stdin.readline().strip()
    if not key:
        raise SystemExit("missing disposable StepFun key on stdin")
    return key


def redirect_and_detach(log_path: Path) -> int | None:
    """fork 一次并 setsid 脱离;父进程返回子 PID,脱离后的子进程返回 None。"""
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


def write_status(
    path: Path, *, schema: str, phase: str, state: str, **extra: Any
) -> None:
    write_json(path, {
        "schema_version": schema,
        "updated_at": utc_now(),
        "state": state,
        "phase": phase,
        "credential_policy": CREDENTIAL_POLICY,
        "transfer_policy": TRANSFER_POLICY,
        **extra,
    })
