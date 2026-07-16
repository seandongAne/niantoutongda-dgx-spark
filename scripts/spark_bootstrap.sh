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
      # CUDA 扩展现场编译。2026-07-15 三个根因逐一修复:
      # ① 节点缺 python3.12-dev 且无 sudo → apt-get download + dpkg -x 解包到
      #    ~/local,经 CPATH 注入。父目录与 python3.12 叶目录都要在:
      #    <Python.h> 走叶目录,pyconfig.h 桩按
      #    <aarch64-linux-gnu/python3.12/pyconfig.h> 相对引用走父目录。
      # ② aarch64 上 torch cpp_extension 默认 Jetson 架构表 → 钉真实算力。
      # ③ mamba-ssm 2.2.5 与 CUDA 13 有两处源码级不兼容(架构表 + CCCL 3
      #    删除的 CUB 符号) → 取 sdist 经 patch_mamba_gb10.py 补丁后本地安装。
      TORCH_CUDA_ARCH_LIST="$(python -c 'import torch; print(".".join(map(str, torch.cuda.get_device_capability(0))))')"
      export TORCH_CUDA_ARCH_LIST
      # 跳过 GitHub 预编译轮子探测(本平台必 404,纯噪音)
      export MAMBA_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE
      if [ ! -f /usr/include/python3.12/Python.h ]; then
        if [ ! -f "$HOME/local/usr/include/python3.12/Python.h" ]; then
          mkdir -p ~/tmp/pydev ~/local
          ( cd ~/tmp/pydev && apt-get download libpython3.12-dev python3.12-dev \
              && for d in *.deb; do dpkg -x "$d" ~/local; done ) \
            || { echo "PYDEV_HEADERS_FAILED — fallback: NGC PyTorch container" >&2; exit 4; }
        fi
        export CPATH="$HOME/local/usr/include/python3.12:$HOME/local/usr/include${CPATH:+:$CPATH}"
      fi
      # 运行时同样要 CPATH:NemotronH 推理时 triton JIT 现场编译 cuda_utils.c,
      # 也吃 Python.h — 用 .pth 的 import 行在解释器启动时自动注入,不依赖调用方
      # 记得传。(不能用 sitecustomize.py:Debian 在 /usr/lib/python3.12 有同名
      # 文件且路径序在前,venv 内的会被遮蔽 — 实测踩过。)
      SITE_DIR="$(python -c 'import site; print(site.getsitepackages()[0])')"
      cat > "$SITE_DIR/zz_gb10_cpath.pth" <<'PYEOF'
import os; _i = os.path.expanduser("~/local/usr/include"); (os.environ.__setitem__("CPATH", _i + "/python3.12:" + _i + ((":" + os.environ["CPATH"]) if os.environ.get("CPATH") else "")) if (os.path.isdir(_i) and _i not in os.environ.get("CPATH", "")) else None)
PYEOF
      echo "build env: TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST CPATH=${CPATH:-<system>}"
      pip install causal_conv1d --no-build-isolation -i "$MIRROR" \
        || { echo "CAUSAL_BUILD_FAILED — fallback: NGC PyTorch container" >&2; exit 3; }
      MAMBA_SRC="$HOME/tmp/mamba_src/mamba_ssm-2.2.5"
      if [ ! -f "$MAMBA_SRC/setup.py" ]; then
        mkdir -p "$HOME/tmp/mamba_src"
        curl -fsSLo "$HOME/tmp/mamba_src/mamba_ssm-2.2.5.tar.gz" \
          "https://mirrors.aliyun.com/pypi/packages/ba/2d/fbd909f6e6d48c491a9ed7ae68e8a890d8409aba4a6356741e2a9c6adad5/mamba_ssm-2.2.5.tar.gz" \
          || { echo "MAMBA_SDIST_FETCH_FAILED" >&2; exit 5; }
        tar xf "$HOME/tmp/mamba_src/mamba_ssm-2.2.5.tar.gz" -C "$HOME/tmp/mamba_src"
      fi
      python "$PROJ_DIR/scripts/patch_mamba_gb10.py" "$MAMBA_SRC" "$TORCH_CUDA_ARCH_LIST" \
        || { echo "MAMBA_PATCH_FAILED" >&2; exit 5; }
      pip install "$MAMBA_SRC" --no-build-isolation -i "$MIRROR" \
        || { echo "MAMBA_BUILD_FAILED — fallback: NGC PyTorch container" >&2; exit 3; }
    fi
    cuda_probe
    PHASE="env_${NAME}"
    ;;
  *)
    echo "usage: $0 min|torch|deps|env <name>" >&2; exit 2 ;;
esac
echo "BOOTSTRAP_${PHASE}_DONE"
