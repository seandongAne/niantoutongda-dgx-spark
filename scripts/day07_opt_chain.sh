#!/usr/bin/env bash
# DAY-07 优化链:①选择性 autocast bench → ②插桩 ONNX 的 TRT 哨兵定位。
# 在 Spark 上运行;两个阶段串行以避免 GPU 竞争污染基准。
# 阶段退出码只记录不中断——每个阶段的 JSON 里有完整门禁结论。
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
V10="results/acceptance/SF1/trt-gdino-20260718-v10-dynamo-static"
INTR="results/acceptance/SF1/onnx-intermediate-20260718-v1"
MODEL_DIR="$HOME/models/IDEA-Research__grounding-dino-base"

echo "=== $(date -u +%FT%TZ) day07 opt chain start commit=$COMMIT ==="
echo "trtexec=$TRTEXEC trt_python=$TRT_PYTHON"
for required in "$V10/sample_inputs.npz" "$V10/torch_outputs.npz" \
    "$V10/baseline_manifest.json" "$INTR/instrumented.onnx" \
    "$INTR/result.json" "$MODEL_DIR" "$TRTEXEC"; do
  [ -e "$required" ] || { echo "MISSING_PREREQ: $required"; exit 2; }
done
free -h

echo "=== $(date -u +%FT%TZ) stage 1: selective autocast bench ==="
"$PY" scripts/gdino_selective_autocast_bench.py \
  --model-dir "$MODEL_DIR" \
  --inputs "$V10/sample_inputs.npz" \
  --baseline-manifest "$V10/baseline_manifest.json" \
  --baseline-outputs "$V10/torch_outputs.npz" \
  --output results/acceptance/SF1/selective-autocast-20260719-v1/bench.json \
  --code-commit "$COMMIT"
echo "STAGE1_EXIT=$?"
free -h

echo "=== $(date -u +%FT%TZ) stage 2: instrumented TRT sentinel dump ==="
"$PY" scripts/gdino_trt_instrumented_sentinel_dump.py \
  --onnx "$INTR/instrumented.onnx" \
  --sentinel-manifest "$INTR/result.json" \
  --inputs "$V10/sample_inputs.npz" \
  --engine results/acceptance/SF1/trt-sentinel-20260719-v1/instrumented_fp32_notf32.engine \
  --output results/acceptance/SF1/trt-sentinel-20260719-v1/result.json \
  --baseline-manifest "$V10/baseline_manifest.json" \
  --torch-outputs "$V10/torch_outputs.npz" \
  --model-dir "$MODEL_DIR" \
  --trtexec "$TRTEXEC" \
  --code-commit "$COMMIT"
echo "STAGE2_EXIT=$?"
free -h
echo "=== $(date -u +%FT%TZ) CHAIN_DONE ==="
