#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-/data/AnyaCoder/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17}"
HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-8100}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
GPU_GROUPS="${GPU_GROUPS:-0,1;2,3;4,5;6,7}"
LOG_DIR="${LOG_DIR:-logs/qwen35_sglang_swarm}"
PYTHON_BIN="${PYTHON_BIN:-/data/AnyaCoder/miniconda3/envs/qwen35sg510/bin/python}"
HF_ENDPOINT="${HF_ENDPOINT:-}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-0.8B}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-0}"
SLEEP_ON_IDLE="${SLEEP_ON_IDLE:-1}"

mkdir -p "$LOG_DIR"

IFS=';' read -r -a GROUP_ARRAY <<< "$GPU_GROUPS"
if [[ "${#GROUP_ARRAY[@]}" -eq 0 ]]; then
  echo "No GPU groups specified in GPU_GROUPS" >&2
  exit 1
fi

for idx in "${!GROUP_ARRAY[@]}"; do
  group="${GROUP_ARRAY[$idx]}"
  gpu_csv="$(echo "$group" | tr -d ' ')"
  if [[ -z "$gpu_csv" ]]; then
    echo "Encountered empty GPU group at index ${idx}" >&2
    exit 1
  fi
  IFS=',' read -r -a GROUP_GPUS <<< "$gpu_csv"
  if [[ "${#GROUP_GPUS[@]}" -eq 0 ]]; then
    echo "Invalid GPU group: ${group}" >&2
    exit 1
  fi
  tp_size="$TENSOR_PARALLEL_SIZE"
  if [[ "$tp_size" == "0" ]]; then
    tp_size="${#GROUP_GPUS[@]}"
  fi
  if [[ "$tp_size" -ne "${#GROUP_GPUS[@]}" ]]; then
    echo "TENSOR_PARALLEL_SIZE=${tp_size} does not match GPU group '${gpu_csv}'" >&2
    exit 1
  fi
  port=$((BASE_PORT + idx))
  group_tag="${gpu_csv//,/g}"
  log_path="${LOG_DIR}/qwen35_gpus${group_tag}_port${port}.log"
  launcher_path="${LOG_DIR}/launch_gpus${group_tag}_port${port}.sh"

  echo "Launching replica ${idx} on GPUs ${gpu_csv} at port ${port}"
  cat >"${launcher_path}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PATH="$(dirname "${PYTHON_BIN}"):\$PATH"
if [[ -n "${HF_ENDPOINT}" ]]; then
  export HF_ENDPOINT="${HF_ENDPOINT}"
fi
export CUDA_VISIBLE_DEVICES="${gpu_csv}"
exec "${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${MODEL_NAME}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --enable-multimodal \
  --host "${HOST}" \
  --port "${port}" \
  --tensor-parallel-size "${tp_size}" \
  --context-length "${CONTEXT_LENGTH}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  --disable-cuda-graph \
EOF
  if [[ "${SLEEP_ON_IDLE}" == "1" ]]; then
    cat >>"${launcher_path}" <<EOF
  --sleep-on-idle \
EOF
  fi
  cat >>"${launcher_path}" <<EOF
  ${EXTRA_ARGS}
EOF
  chmod +x "${launcher_path}"
  nohup bash "${launcher_path}" >"${log_path}" 2>&1 &
done

echo
echo "Replica endpoints:"
for idx in "${!GROUP_ARRAY[@]}"; do
  port=$((BASE_PORT + idx))
  echo "http://127.0.0.1:${port}/v1"
done
