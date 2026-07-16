"""hero_pipeline 端到端:合成夹具全链路 + 断点续跑语义。

真子进程执行(阶段隔离是被测行为的一部分),使用 --run-dir 隔离产物。
"""

import json
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent.parent
CONFIG = PROJ / "configs/hero_pipeline_dev.yaml"


def run_pipeline(run_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PROJ / "scripts/hero_pipeline.py"),
         "--config", str(CONFIG), "--run-dir", str(run_dir), *extra],
        capture_output=True, text=True, cwd=PROJ,
    )


def test_full_chain_then_resume_then_from_stage(tmp_path):
    run_dir = tmp_path / "run"

    first = run_pipeline(run_dir)
    assert first.returncode == 0, first.stderr
    assert "[run ] naming" in first.stdout
    assert "✅ 主链完成" in first.stdout
    bundle = json.loads((run_dir / "bundle.json").read_text(encoding="utf-8"))
    stages_in_bundle = {a["stage"] for a in bundle["artifacts"]}
    assert {"naming", "narration", "regions", "group", "layout", "taskcards",
            "verify"} <= stages_in_bundle
    cards = (run_dir / "taskcards/taskcards.md").read_text(encoding="utf-8")
    assert "水壶(蓝色)" in cards and "水壶(粉色)" in cards

    verdicts = json.loads(
        (run_dir / "verify/verdicts.json").read_text(encoding="utf-8")
    )
    assert {v["verdict"] for v in verdicts.values()} == {
        "VERIFIED", "FAILED", "NEEDS_USER"
    }
    assert (run_dir / "index.html").exists()

    second = run_pipeline(run_dir)
    assert second.returncode == 0, second.stderr
    assert second.stdout.count("[skip]") == 8
    assert "[run ] naming" not in second.stdout

    third = run_pipeline(run_dir, "--from-stage", "layout")
    assert third.returncode == 0, third.stderr
    assert "[skip] group" in third.stdout
    assert "[run ] layout" in third.stdout
    assert "[run ] taskcards" in third.stdout


def test_dry_run_lists_plan_without_side_effects(tmp_path):
    run_dir = tmp_path / "dry"
    proc = run_pipeline(run_dir, "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert "[run ] naming" in proc.stdout
    assert not run_dir.exists()


def test_stale_stage_reruns_but_unchanged_output_spares_downstream(tmp_path):
    """内容寻址续跑:上游被判 stale 后重跑,但产物字节不变时下游不陪跑。"""
    run_dir = tmp_path / "run"
    assert run_pipeline(run_dir).returncode == 0
    state_path = run_dir / "state/narration.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    transcript = next(iter(state["inputs"]))
    state["inputs"][transcript] = "0" * 64  # 伪造输入哈希 → narration 判 stale
    state_path.write_text(json.dumps(state), encoding="utf-8")
    proc = run_pipeline(run_dir)
    assert proc.returncode == 0, proc.stderr
    assert "[skip] naming" in proc.stdout
    assert "[run ] narration" in proc.stdout
    # 重跑产物与上次字节一致 → group 及以下按内容寻址跳过
    assert "[skip] group" in proc.stdout
    assert "[skip] taskcards" in proc.stdout
