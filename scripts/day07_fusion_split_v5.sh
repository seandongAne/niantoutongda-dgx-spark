#!/usr/bin/env bash
# DAY-07 融合拆分修复实验 v5(用户批准时间盒):
#   prep(v3∪v4 标记集,带类型)→ trtexec FP32 --noTF32 → runtime bench(严格决策门+计时)
# vLLM 保持在线(与二分 v2-v4 同型负载)。
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
OUTDIR="results/acceptance/SF1/trt-fusion-split-20260719-v5"

echo "=== $(date -u +%FT%TZ) fusion-split v5 start commit=$COMMIT ==="
free -h | head -2

echo "=== stage 1: prep marked ONNX ==="
"$PY" scripts/gdino_fusion_split_repair_prep.py \
  --onnx "$V10DIR/grounding_dino.onnx" \
  --from-results \
    results/acceptance/SF1/trt-scatter-bisect-20260719-v3/result.json \
    results/acceptance/SF1/trt-scatter-bisect-20260719-v4/result.json \
  --output-onnx "$OUTDIR/marked.onnx" \
  --output-manifest "$OUTDIR/prep_manifest.json" \
  --code-commit "$COMMIT"
echo "PREP_EXIT=$?"

echo "=== stage 2: trtexec FP32 --noTF32 build ==="
"$TRTEXEC" \
  --onnx="$OUTDIR/marked.onnx" \
  --saveEngine="$OUTDIR/repaired_fp32_notf32.engine" \
  --noTF32 --skipInference --memPoolSize=workspace:16384M \
  > "$OUTDIR/trtexec_build.log" 2>&1
echo "BUILD_EXIT=$?"
tail -3 "$OUTDIR/trtexec_build.log"

echo "=== stage 3: runtime bench + strict decision gate ==="
"$PY" scripts/gdino_trt_runtime_bench.py \
  --engine "$OUTDIR/repaired_fp32_notf32.engine" \
  --inputs "$V10DIR/sample_inputs.npz" \
  --torch-outputs "$V10DIR/torch_outputs.npz" \
  --baseline-manifest "$V10DIR/baseline_manifest.json" \
  --model-dir "$HOME/models/IDEA-Research__grounding-dino-base" \
  --output "$OUTDIR/bench.json" \
  --runs 100 --warmup 10 \
  --code-commit "$COMMIT" > "$OUTDIR/bench_stdout.log" 2>&1
echo "BENCH_EXIT=$?"
free -h | head -2
echo "=== $(date -u +%FT%TZ) FUSION_SPLIT_V5_DONE ==="
