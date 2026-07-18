#!/usr/bin/env bash
# DAY-07 优化链第二段:③v2 选择性 autocast(含 encoder-only)→ ④v7 Torch-TensorRT
# 文本外置探针(容器内,FP32-only——按纪律 FP32 集合门 PASS 后才谈 FP16)。
# 前提:day07_opt_chain.sh 已 CHAIN_DONE,GPU 空闲。
set -uo pipefail
cd "$HOME/proj"
mkdir -p logs

PY="$HOME/venv/bin/python"
COMMIT="$(cat COMMIT 2>/dev/null || echo unknown)"
V10="results/acceptance/SF1/trt-gdino-20260718-v10-dynamo-static"
IMAGE="dgx-spark/torch-tensorrt-gdino:26.06"

echo "=== $(date -u +%FT%TZ) day07 opt chain2 start commit=$COMMIT ==="
free -h

echo "=== $(date -u +%FT%TZ) stage 3: selective autocast bench v2 (encoder-only) ==="
"$PY" scripts/gdino_selective_autocast_bench.py \
  --model-dir "$HOME/models/IDEA-Research__grounding-dino-base" \
  --inputs "$V10/sample_inputs.npz" \
  --baseline-manifest "$V10/baseline_manifest.json" \
  --baseline-outputs "$V10/torch_outputs.npz" \
  --output results/acceptance/SF1/selective-autocast-20260719-v2/bench.json \
  --code-commit "$COMMIT"
echo "STAGE3_EXIT=$?"
free -h

echo "=== $(date -u +%FT%TZ) stage 4: torch-tensorrt v7 text-outside-export (container, fp32-only) ==="
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "MISSING_IMAGE: $IMAGE"
else
  docker run --rm --gpus all \
    -v "$HOME/proj:/workspace/proj" \
    -v "$HOME/models:/models:ro" \
    -w /workspace/proj \
    "$IMAGE" \
    python scripts/gdino_torch_tensorrt_hybrid_probe.py \
      --model-dir /models/IDEA-Research__grounding-dino-base \
      --inputs "$V10/sample_inputs.npz" \
      --baseline-manifest "$V10/manifest.json" \
      --baseline-outputs results/acceptance/SF1/topk-stage-host-20260718-v2/stage-boundaries.npz \
      --output results/acceptance/SF1/torch-tensorrt-hybrid-20260719-v7-text-outside-export/probe.json \
      --container-eager-gate decision-set \
      --text-outside-export \
      --skip-fp16 \
      --container-image "$IMAGE" \
      --code-commit "$COMMIT"
  echo "STAGE4_EXIT=$?"
fi
free -h
echo "=== $(date -u +%FT%TZ) CHAIN2_DONE ==="
