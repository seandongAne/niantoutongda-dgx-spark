#!/usr/bin/env bash
# 幂等环境自举 — 在 spark 上执行。用法: spark_bootstrap.sh min|torch|deps
# min   : venv + modelscope(下载器),快,可同步等待
# torch : aarch64 CUDA torch(GB10/Blackwell,先 cu130 后 cu128),>1min,必须 nohup
# deps  : transformers 等推理依赖,>1min,必须 nohup(与 torch 串行,勿并发写 venv)
set -uo pipefail
MIRROR="https://mirrors.aliyun.com/pypi/simple/"
[ -d ~/venv ] || python3 -m venv ~/venv
# shellcheck disable=SC1090
source ~/venv/bin/activate

case "${1:-min}" in
  min)
    pip install -q -U pip -i "$MIRROR"
    pip install -q -U modelscope -i "$MIRROR"
    modelscope --help >/dev/null 2>&1 && echo "modelscope CLI OK"
    ;;
  torch)
    pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu130 \
      || pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu128
    python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.rand(2048, 2048, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("cuda_matmul_ok", tuple(y.shape), "device", torch.cuda.get_device_name(0))
PY
    ;;
  deps)
    pip install -U transformers accelerate pillow numpy opencv-python-headless \
      pyyaml librosa soundfile ortools fastapi uvicorn pydantic pytest -i "$MIRROR"
    ;;
  *)
    echo "usage: $0 min|torch|deps" >&2; exit 2 ;;
esac
echo "BOOTSTRAP_${1:-min}_DONE"
