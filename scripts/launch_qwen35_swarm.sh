#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-0.8B}"
HOST="${HOST:-0.0.0.0}"
BASE_PORT="${BASE_PORT:-8100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
LOG_DIR="${LOG_DIR:-logs/qwen35_swarm}"

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
mkdir -p "$LOG_DIR"

if [[ "${#GPU_ARRAY[@]}" -eq 0 ]]; then
  echo "No GPUs specified in GPUS" >&2
  exit 1
fi

for idx in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$idx]}"
  port=$((BASE_PORT + idx))
  log_path="${LOG_DIR}/qwen35_0p8b_gpu${gpu}_port${port}.log"

  echo "Launching replica ${idx} on GPU ${gpu} at port ${port}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup vllm serve "${MODEL_NAME}" \
    --host "${HOST}" \
    --port "${port}" \
    --tensor-parallel-size 1 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    >"${log_path}" 2>&1 &
done

echo
echo "Replica endpoints:"
for idx in "${!GPU_ARRAY[@]}"; do
  port=$((BASE_PORT + idx))
  echo "http://127.0.0.1:${port}/v1"
done
