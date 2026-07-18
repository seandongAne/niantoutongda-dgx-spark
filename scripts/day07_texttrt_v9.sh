#!/usr/bin/env bash
# DAY-07 v9:Torch-TensorRT 文本外置探针重跑(v7 编译成功但在 profile/benchmark
# v9:去掉 offload(与 cuda 示例输入不兼容),保留 workspace 上限;探针内已加 empty_cache。
# workspace 上限,并缩短 benchmark;Nemotron vLLM 保持驻留不动。
set -uo pipefail
cd "$HOME/proj"
mkdir -p logs

COMMIT="$(cat COMMIT 2>/dev/null || echo unknown)"
V10="results/acceptance/SF1/trt-gdino-20260718-v10-dynamo-static"
IMAGE="dgx-spark/torch-tensorrt-gdino:26.06"

echo "=== $(date -u +%FT%TZ) v9 text-outside-export start commit=$COMMIT ==="
free -h
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "MISSING_IMAGE: $IMAGE"
  exit 2
fi
docker run --rm --gpus all --ipc=host \
  -v "$HOME/proj:/workspace/proj" \
  -v "$HOME/models:/models:ro" \
  -w /workspace/proj \
  "$IMAGE" \
  python scripts/gdino_torch_tensorrt_hybrid_probe.py \
    --model-dir /models/IDEA-Research__grounding-dino-base \
    --inputs "$V10/sample_inputs.npz" \
    --baseline-manifest "$V10/manifest.json" \
    --baseline-outputs results/acceptance/SF1/topk-stage-host-20260718-v2/stage-boundaries.npz \
    --output results/acceptance/SF1/torch-tensorrt-hybrid-20260719-v9-text-outside-export/probe.json \
    --container-eager-gate decision-set \
    --text-outside-export \
    --workspace-size-gib 8 \
    --runs 20 --warmup 5 \
    --skip-fp16 \
    --container-image "$IMAGE" \
    --code-commit "$COMMIT"
echo "V9_EXIT=$?"
free -h
echo "=== $(date -u +%FT%TZ) V9_DONE ==="
