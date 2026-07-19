#!/usr/bin/env bash
# DAY-07 二分器 v3:穿透搬运算子上溯两层,自动定罪对比头路径"输入净/输出脏"的真实算子。
# vLLM 保持在线(与 chain3 B / chain4 B2 同型:trtexec 构建负载可与常驻服务共存)。
set -uo pipefail
cd "$HOME/proj"
mkdir -p logs

PY="$HOME/venv/bin/python"
TRT_ROOT="$HOME/local/tensorrt-10.14.1.48"
TRTEXEC="$(find "$TRT_ROOT" -type f -name trtexec -print -quit)"
TRT_PYTHON="$(ls -d "$TRT_ROOT"/usr/lib/python*/dist-packages 2>/dev/null | head -1)"
export LD_LIBRARY_PATH="$TRT_ROOT/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$TRT_PYTHON${PYTHONPATH:+:$PYTHONPATH}"
COMMIT="$(cat COMMIT 2>/dev/null || echo unknown)"
V10DIR="results/acceptance/SF1/trt-gdino-20260718-v10-dynamo-static"

echo "=== $(date -u +%FT%TZ) bisect v3 start commit=$COMMIT ==="
"$PY" scripts/gdino_slice_scatter_bisect.py \
  --onnx "$V10DIR/grounding_dino.onnx" \
  --inputs "$V10DIR/sample_inputs.npz" \
  --engine results/acceptance/SF1/trt-scatter-bisect-20260719-v3/scatter_bisect_fp32_notf32.engine \
  --output results/acceptance/SF1/trt-scatter-bisect-20260719-v3/result.json \
  --trtexec "$TRTEXEC" \
  --code-commit "$COMMIT"
echo "BISECT_V3_EXIT=$?"
echo "=== $(date -u +%FT%TZ) BISECT_V3_DONE ==="
