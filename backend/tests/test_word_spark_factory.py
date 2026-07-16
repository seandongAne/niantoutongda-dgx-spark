import json
from types import SimpleNamespace

import pytest

from scripts import word_spark_factory


def launch_args(tmp_path, **overrides):
    argv = [
        "launch",
        "--items", "fixtures/wordgen/smoke_items.json",
        "--frames-dir", "fixtures/dev_a/hardval/frames",
        "--gt", "fixtures/dev_a/hardval/gt.json",
        "--run-dir", str(tmp_path / "run"),
        "--report-dir", str(tmp_path / "report"),
        "--log", str(tmp_path / "factory.log"),
        "--acknowledge-key-exposure",
    ]
    args = word_spark_factory.build_parser().parse_args(argv)
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def test_remote_command_contains_job_config_but_never_a_credential(tmp_path):
    command = word_spark_factory.build_remote_command(launch_args(tmp_path))
    assert "word_spark_factory.py worker" in command
    assert "--items fixtures/wordgen/smoke_items.json" in command
    assert "--gt fixtures/dev_a/hardval/gt.json" in command
    assert "~/venv/bin/python" in command
    assert "STEPFUN_API_KEY" not in command


def test_launch_sends_disposable_key_only_on_stdin(monkeypatch, tmp_path):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="started\n", stderr="")

    monkeypatch.setattr(word_spark_factory, "load_key", lambda: "disposable-secret")
    import scripts.spark_factory_common as common
    monkeypatch.setattr(common.subprocess, "run", fake_run)

    assert word_spark_factory.cmd_launch(launch_args(tmp_path)) == 0
    assert captured["input"] == "disposable-secret\n"
    assert "disposable-secret" not in " ".join(captured["command"])


def test_launch_requires_explicit_key_exposure_acknowledgement(monkeypatch, tmp_path):
    import scripts.spark_factory_common as common
    monkeypatch.setattr(
        common.subprocess, "run",
        lambda *_a, **_k: pytest.fail("SSH must not start without acknowledgement"),
    )
    with pytest.raises(SystemExit, match="refusing to expose"):
        word_spark_factory.cmd_launch(
            launch_args(tmp_path, acknowledge_key_exposure=False)
        )


def test_gen_cap_refuses_oversized_candidate_set(monkeypatch, tmp_path):
    args = launch_args(tmp_path, max_phrases=2)
    out = tmp_path / "run" / "candidates.json"
    out.parent.mkdir(parents=True)

    def fake_run(cmd, **kwargs):
        out.write_text(json.dumps({"a": ["x", "y"], "b": ["z"]}), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(word_spark_factory.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="超过 --max-phrases"):
        word_spark_factory._run_gen(args, "key")


def test_worker_status_never_contains_credential(monkeypatch, tmp_path):
    args = launch_args(tmp_path)

    def fail_gen(_args, _key):
        raise RuntimeError("boom")

    monkeypatch.setattr(word_spark_factory, "_run_gen", fail_gen)
    assert word_spark_factory.run_worker(args, "disposable-secret") == 1
    status = (tmp_path / "run" / "factory-status.json").read_text(encoding="utf-8")
    assert "disposable-secret" not in status
    data = json.loads(status)
    assert data["state"] == "failed"
    assert data["credential_policy"].startswith("stdin_memory_only")


def test_deadwords_report_flags_zero_detection_phrases(tmp_path):
    args = launch_args(tmp_path, gt=None)
    scan_dir = tmp_path / "run" / "scan"
    scan_dir.mkdir(parents=True)
    (scan_dir / "scan-manifest.json").write_text(json.dumps({
        "candidates": {
            "a/c0": {"phrase": "live word", "detections": 7},
            "a/c1": {"phrase": "dead word", "detections": 0},
        }
    }), encoding="utf-8")
    (tmp_path / "report").mkdir()
    word_spark_factory._write_deadwords(args, scan_dir)
    report = json.loads(
        (tmp_path / "report" / "deadwords.json").read_text(encoding="utf-8")
    )
    assert report["dead"] == ["a/c1"]
    assert report["detections_per_phrase"]["a/c0"] == 7
