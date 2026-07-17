#!/usr/bin/env python
"""Stepfun Step Plan API 本地客户端(赛方 2000M token 配额)。

红线:
- STEPFUN_API_KEY 的持久副本只存本地 .env / 环境变量;永不进 git、永不写入
  Spark 磁盘(.gitignore 与 deploy.sh 均已排除 .env)。赛方限制 SSH 文件传输后，
  ``a1_spark_factory.py`` 可把一次性 key 仅经 stdin 注入 Spark 进程内存；该 key
  必须视为已暴露并在任务后撤销，绝不进入 argv、日志、状态文件或远端 .env。
- 演示主链不调用云 API。本工具只服务 dev-time 用途——A1 prompt 预热、
  G0 预标注/词表、LLM-judge、文案润色,见 docs/STEPFUN_API_PLAYBOOK.md。
- 云端输出永远是"候选/评审意见",不得自动写入真值或契约对象。

用法:
  python scripts/stepfun_api.py models
  python scripts/stepfun_api.py chat --model step-3.5-flash --prompt "..."
  echo "..." | python scripts/stepfun_api.py chat --model step-3.5-flash
  python scripts/stepfun_api.py chat --model step-audio-2.5 \
      --audio memo.wav --prompt "列出旁白里提到的每一件物品"
  (音频 content part 采用 OpenAI input_audio 约定;若阶跃实际格式不同,
   以第一次真实调用的报错为准修正。)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BASE_URL = os.environ.get("STEPFUN_BASE_URL", "https://api.stepfun.com/step_plan/v1")


class StepfunAPIError(RuntimeError):
    """A redacted API/network failure safe to persist in benchmark logs."""


def load_key() -> str:
    key = os.environ.get("STEPFUN_API_KEY", "")
    if not key:
        env = Path(__file__).resolve().parent.parent / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("STEPFUN_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        sys.exit("missing STEPFUN_API_KEY — 写入本地仓库根 .env(已 gitignore)或导出环境变量")
    return key


def _request(path: str, payload: dict | None = None) -> bytes:
    req = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={
            "Authorization": f"Bearer {load_key()}",
            "Content-Type": "application/json",
        },
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise StepfunAPIError(
            f"HTTP {e.code} {path}: {e.read().decode(errors='replace')[:2000]}"
        ) from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise StepfunAPIError(f"network error {path}: {e}") from e


def call(path: str, payload: dict | None = None) -> dict[str, Any]:
    try:
        return json.loads(_request(path, payload))
    except json.JSONDecodeError as e:
        raise StepfunAPIError(f"invalid JSON response {path}: {e}") from e


def chat_completion(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return call("/chat/completions", payload)


def synthesize_speech(
    *,
    text: str,
    model: str,
    voice: str,
    instruction: str | None = None,
    response_format: str = "wav",
) -> bytes:
    payload: dict[str, Any] = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": response_format,
    }
    if instruction:
        payload["instruction"] = instruction
    return _request("/audio/speech", payload)


def cmd_models(_: argparse.Namespace) -> None:
    data = call("/models")
    for m in data.get("data", []):
        print(m.get("id"))


def image_part(path: str | Path) -> dict:
    """图片 → OpenAI image_url data-URI content part(视觉调用统一走这里)。"""
    p = Path(path)
    suffix = p.suffix.lstrip(".").lower() or "jpeg"
    mime = {"jpg": "jpeg"}.get(suffix, suffix)
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/{mime};base64,"
            + base64.b64encode(p.read_bytes()).decode()
        },
    }


def cmd_chat(args: argparse.Namespace) -> None:
    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    content: list[dict] | str
    if args.audio:
        audio = Path(args.audio)
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "input_audio",
                "input_audio": {
                    "data": base64.b64encode(audio.read_bytes()).decode(),
                    "format": audio.suffix.lstrip(".").lower() or "wav",
                },
            },
        ]
    elif args.image:
        content = [{"type": "text", "text": prompt}] + [
            image_part(img) for img in args.image
        ]
    else:
        content = prompt
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": content})
    data = chat_completion(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
    )
    print(data["choices"][0]["message"]["content"])
    usage = data.get("usage", {})
    print(
        f"[usage] prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')}",
        file=sys.stderr,
    )


def cmd_tts(args: argparse.Namespace) -> None:
    """文本 → 语音(A1 预热用合成测试音频,不涉家庭素材)。"""
    audio = synthesize_speech(
        text=args.text,
        model=args.model,
        voice=args.voice,
        instruction=args.instruction,
    )
    Path(args.out).write_bytes(audio)
    print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models").set_defaults(func=cmd_models)
    tts = sub.add_parser("tts")
    tts.add_argument("--model", default="stepaudio-2.5-tts")
    tts.add_argument("--text", required=True)
    tts.add_argument("--voice", default="linjiajiejie")
    tts.add_argument("--instruction", default=None)
    tts.add_argument("--out", required=True)
    tts.set_defaults(func=cmd_tts)
    chat = sub.add_parser("chat")
    chat.add_argument("--model", required=True)
    chat.add_argument("--prompt", default=None, help="缺省从 stdin 读")
    chat.add_argument("--system", default=None)
    chat.add_argument("--audio", default=None, help="音频文件路径(wav/mp3)")
    chat.add_argument(
        "--image", action="append", default=None, help="图片路径,可重复(jpg/png)"
    )
    chat.add_argument("--temperature", type=float, default=0.2)
    chat.set_defaults(func=cmd_chat)
    args = ap.parse_args()
    try:
        args.func(args)
    except StepfunAPIError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
