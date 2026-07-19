#!/usr/bin/env bash
# DAY-07 融合拆分修复实验 v6:FP16 + FP32 标记精度岛。
# 复用 v5 的 marked.onnx;bench 的严格位置门预期 FAIL(FP16 分数漂移>1e-3),
# 终审=集合门三档(strict/diagnostic/revised-0.98)。
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
V5DIR="results/acceptance/SF1/trt-fusion-split-20260719-v5"
OUTDIR="results/acceptance/SF1/trt-fusion-split-20260719-v6-fp16"
mkdir -p "$OUTDIR"

echo "=== $(date -u +%FT%TZ) fusion-split v6 fp16 start commit=$COMMIT ==="
free -h | head -2

echo "=== stage 1: trtexec FP16 (+FP32 marker islands) build ==="
"$TRTEXEC" \
  --onnx="$V5DIR/marked.onnx" \
  --saveEngine="$OUTDIR/repaired_fp16.engine" \
  --fp16 --noTF32 --skipInference --memPoolSize=workspace:16384M \
  > "$OUTDIR/trtexec_build.log" 2>&1
echo "BUILD_EXIT=$?"
tail -3 "$OUTDIR/trtexec_build.log"

echo "=== stage 2: runtime bench (positional strict gate informational) ==="
"$PY" scripts/gdino_trt_runtime_bench.py \
  --engine "$OUTDIR/repaired_fp16.engine" \
  --inputs "$V10DIR/sample_inputs.npz" \
  --torch-outputs "$V10DIR/torch_outputs.npz" \
  --baseline-manifest "$V10DIR/baseline_manifest.json" \
  --model-dir "$HOME/models/IDEA-Research__grounding-dino-base" \
  --output "$OUTDIR/bench.json" \
  --runs 100 --warmup 10 \
  --code-commit "$COMMIT" > "$OUTDIR/bench_stdout.log" 2>&1
echo "BENCH_EXIT=$?"

echo "=== stage 3: set-based three-tier gate (final adjudication) ==="
"$PY" scripts/gdino_npz_decision_set_gate.py \
  --model-dir "$HOME/models/IDEA-Research__grounding-dino-base" \
  --inputs "$V10DIR/sample_inputs.npz" \
  --baseline-manifest "$V10DIR/baseline_manifest.json" \
  --reference-npz "$V10DIR/torch_outputs.npz" \
  --candidate-npz "$OUTDIR/bench.outputs.npz" \
  --output "$OUTDIR/set_gate.json" \
  --code-commit "$COMMIT"
echo "SETGATE_EXIT=$?"
free -h | head -2
echo "=== $(date -u +%FT%TZ) FUSION_SPLIT_V6_DONE ==="
