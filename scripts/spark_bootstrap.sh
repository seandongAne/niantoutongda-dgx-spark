#!/usr/bin/env bash
# 幂等环境自举 — 在 spark 上执行。
# 用法: spark_bootstrap.sh min|torch|deps|env <name>
#   min        : ~/venv + modelscope(下载器),快,可同步等待
#   torch      : ~/venv 装 aarch64 CUDA torch(先 cu130 后 cu128),>1min,必须 nohup
#   deps       : ~/venv 装视觉主链推理依赖,>1min,必须 nohup(与 torch 串行,勿并发写 venv)
#   env <name> : 建独立 venv ~/envs/<name>,装 torch + configs/env_<name>.txt
#
# 环境拆分(评审 P0-3):Step-Audio 钉 transformers==4.49.0,Nemotron VL 要
# >4.53,<4.54,单一 venv 不可能同时满足。三套环境:
#   ~/venv               = 下载器 + 视觉主链(grounding-dino / dinov2 / CP-SAT)
#   ~/envs/stepaudio     = Step-Audio 2 mini / TTS-3B
#   ~/envs/nemotron_vl   = Nemotron VL-12B / 9B(mamba-ssm 需现场编译,见 env 文件注释)
# pip 缓存共享(~/.cache/pip),第二三次 torch 安装走缓存不重复跨境下载。
set -uo pipefail
MIRROR="https://mirrors.aliyun.com/pypi/simple/"
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"

install_torch() { # $1 = 额外包(如 torchaudio / torchvision)
  pip install -U torch "$1" --index-url https://download.pytorch.org/whl/cu130 \
    || pip install -U torch "$1" --index-url https://download.pytorch.org/whl/cu128
}

cuda_probe() {
  python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.rand(2048, 2048, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("cuda_matmul_ok", tuple(y.shape), "device", torch.cuda.get_device_name(0))
PY
}

PHASE="${1:-min}"
case "$PHASE" in
  min)
    [ -d ~/venv ] || python3 -m venv ~/venv
    source ~/venv/bin/activate
    pip install -q -U pip -i "$MIRROR"
    pip install -q -U modelscope -i "$MIRROR"
    modelscope --help >/dev/null 2>&1 && echo "modelscope CLI OK"
    ;;
  torch)
    source ~/venv/bin/activate
    install_torch torchvision
    cuda_probe
    ;;
  deps)
    source ~/venv/bin/activate
    pip install -U transformers accelerate pillow numpy opencv-python-headless \
      pyyaml ortools fastapi uvicorn pydantic pytest -i "$MIRROR"
    ;;
  env)
    NAME="${2:?usage: $0 env stepaudio|nemotron_vl}"
    REQ="$PROJ_DIR/configs/env_${NAME}.txt"
    [ -f "$REQ" ] || { echo "missing $REQ" >&2; exit 2; }
    mkdir -p ~/envs
    [ -d ~/envs/"$NAME" ] || python3 -m venv ~/envs/"$NAME"
    source ~/envs/"$NAME"/bin/activate
    pip install -q -U pip -i "$MIRROR"
    case "$NAME" in
      stepaudio)   install_torch torchaudio ;;
      nemotron_vl) install_torch torchvision
                   pip install -U ninja packaging -i "$MIRROR" ;;
      *)           install_torch torchvision ;;
    esac
    pip install -U -r "$REQ" -i "$MIRROR"
    if [ "$NAME" = "nemotron_vl" ]; then
      # CUDA 扩展现场编译,失败不静默 — 记录后走 NGC 容器兜底
      pip install "mamba-ssm==2.2.5" causal_conv1d --no-build-isolation -i "$MIRROR" \
        || { echo "MAMBA_BUILD_FAILED — fallback: NGC PyTorch container" >&2; exit 3; }
    fi
    cuda_probe
    PHASE="env_${NAME}"
    ;;
  *)
    echo "usage: $0 min|torch|deps|env <name>" >&2; exit 2 ;;
esac
echo "BOOTSTRAP_${PHASE}_DONE"
