import json
from types import SimpleNamespace

import pytest

from scripts import a1_spark_factory


def launch_args(tmp_path):
    return a1_spark_factory.build_parser().parse_args([
        "launch",
        "--run-dir",
        str(tmp_path / "run"),
        "--report-dir",
        str(tmp_path / "report"),
        "--log",
        str(tmp_path / "factory.log"),
        "--acknowledge-key-exposure",
    ])


def test_remote_command_contains_job_config_but_never_a_credential(tmp_path):
    args = launch_args(tmp_path)
    command = a1_spark_factory.build_remote_command(args)

    assert "a1_spark_factory.py worker" in command
    assert "--conditions clean,noise20,speed090" in command
    assert "--ci-half-width 0.08" in command
    assert "--revocation-delay-days 0" in command
    assert "STEPFUN_API_KEY" not in command


def test_launch_sends_disposable_key_only_on_stdin(monkeypatch, tmp_path):
    args = launch_args(tmp_path)
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="started\n", stderr="")

    monkeypatch.setattr(a1_spark_factory, "load_key", lambda: "disposable-secret")
    monkeypatch.setattr(a1_spark_factory.subprocess, "run", fake_run)

    assert a1_spark_factory.cmd_launch(args) == 0
    assert captured["input"] == "disposable-secret\n"
    assert "disposable-secret" not in " ".join(captured["command"])
    assert "disposable-secret" not in captured["command"][-1]


def test_launch_requires_explicit_key_exposure_acknowledgement(monkeypatch, tmp_path):
    args = launch_args(tmp_path)
    args.acknowledge_key_exposure = False
    monkeypatch.setattr(
        a1_spark_factory.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("SSH must not start without acknowledgement"),
    )

    with pytest.raises(SystemExit, match="refusing to expose"):
        a1_spark_factory.cmd_launch(args)


def test_ensure_plan_is_resumable_but_rejects_a_different_plan(tmp_path):
    args = launch_args(tmp_path)
    a1_spark_factory.ensure_plan(args)
    plan_path = tmp_path / "run" / "plan.json"
    first = json.loads(plan_path.read_text(encoding="utf-8"))

    a1_spark_factory.ensure_plan(args)
    assert json.loads(plan_path.read_text(encoding="utf-8")) == first

    first["seed"] = 0
    plan_path.write_text(json.dumps(first), encoding="utf-8")
    with pytest.raises(RuntimeError, match="existing plan differs"):
        a1_spark_factory.ensure_plan(args)


def test_status_records_the_factory_code_revision(monkeypatch, tmp_path):
    args = launch_args(tmp_path)
    status_path = tmp_path / "factory-status.json"
    monkeypatch.setattr(
        a1_spark_factory.a1_robustness,
        "git_revision",
        lambda _repo: "deploy-revision",
    )

    a1_spark_factory._set_status(
        status_path,
        phase="plan",
        state="running",
        args=args,
    )

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["factory_code_revision"] == "deploy-revision"
