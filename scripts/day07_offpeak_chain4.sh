#!/usr/bin/env bash
# DAY-07 错峰窗口链3(用户已批准临时停 Nemotron vLLM):
#   A) v10 文本外置 FP32 engine 门禁+计时(容器,runs 50)


# trap EXIT 无条件恢复 vLLM;各阶段退出码只记录不中断。
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
IMAGE="dgx-spark/torch-tensorrt-gdino:26.06"
MODEL_DIR="$HOME/models/IDEA-Research__grounding-dino-base"

VLLM_ID="$(docker ps -q --filter ancestor=nvcr.io/nvidia/vllm:26.06-py3 | head -1)"
restore_vllm() {
  if [ -n "$VLLM_ID" ]; then
    echo "=== restoring vllm $VLLM_ID ==="
    if docker start "$VLLM_ID" >/dev/null 2>&1; then
      sleep 5
      docker ps --format "{{.ID}} {{.Status}}" | grep "^${VLLM_ID}" \
        && echo "VLLM_RESTARTED" || echo "VLLM_RESTART_STATUS_UNKNOWN"
    else
      echo "VLLM_RESTART_FAILED"
    fi
  fi
}
trap restore_vllm EXIT

echo "=== $(date -u +%FT%TZ) offpeak chain4 start commit=$COMMIT vllm_id=$VLLM_ID ==="

echo "=== $(date -u +%FT%TZ) stage B2: slice_scatter bisect v2 (typed outputs, vllm stays up) ==="
"$PY" scripts/gdino_slice_scatter_bisect.py \
  --onnx "$V10DIR/grounding_dino.onnx" \
  --inputs "$V10DIR/sample_inputs.npz" \
  --engine results/acceptance/SF1/trt-scatter-bisect-20260719-v2/scatter_bisect_fp32_notf32.engine \
  --output results/acceptance/SF1/trt-scatter-bisect-20260719-v2/result.json \
  --trtexec "$TRTEXEC" \
  --code-commit "$COMMIT"
echo "STAGEB2_EXIT=$?"
free -h
if [ -n "$VLLM_ID" ]; then
  docker stop "$VLLM_ID" >/dev/null && echo "VLLM_STOPPED"
else
  echo "VLLM_NOT_RUNNING"
fi
free -h

echo "=== $(date -u +%FT%TZ) stage A: v11 text-outside-export (container, fp32, runs 50) ==="
docker run --rm --gpus all --ipc=host \
  -v "$HOME/proj:/workspace/proj" \
  -v "$HOME/models:/models:ro" \
  -w /workspace/proj \
  "$IMAGE" \
  python scripts/gdino_torch_tensorrt_hybrid_probe.py \
    --model-dir /models/IDEA-Research__grounding-dino-base \
    --inputs "$V10DIR/sample_inputs.npz" \
    --baseline-manifest "$V10DIR/manifest.json" \
    --baseline-outputs results/acceptance/SF1/topk-stage-host-20260718-v2/stage-boundaries.npz \
    --output results/acceptance/SF1/torch-tensorrt-hybrid-20260719-v11-text-outside-export/probe.json \
    --container-eager-gate decision-set \
    --text-outside-export \
    --workspace-size-gib 8 \
    --runs 50 --warmup 10 \
    --skip-fp16 \
    --skip-runtime-profile \
    --container-image "$IMAGE" \
    --code-commit "$COMMIT"
echo "STAGEA_EXIT=$?"
free -h

free -h

free -h

restore_vllm
trap - EXIT
echo "=== $(date -u +%FT%TZ) CHAIN4_DONE ==="
