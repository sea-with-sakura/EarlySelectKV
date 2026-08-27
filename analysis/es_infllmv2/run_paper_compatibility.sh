#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${MODEL_DIR:-modelzoo/InfLLM-V2-Long-Sparse-Base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-result/infllmv2_compatibility}"
LOWRANK_CACHE="${LOWRANK_CACHE:-${OUTPUT_ROOT}/cache/infllmv2_long_sparse_rank256.pt}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
SAMPLE_COUNT="${SAMPLE_COUNT:-16}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-12288}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DATASETS="${DATASETS:-gov_report qmsum musique hotpotqa}"

run_cmd() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_dataset() {
  local dataset="$1"
  local eval_config="config/eval_config/longbench/${dataset}.json"
  local rank_dir="${OUTPUT_ROOT}/rank256/${dataset}"
  local exact_dir="${OUTPUT_ROOT}/exact_wq/${dataset}"

  run_cmd "${PYTHON_BIN}" analysis/es_infllmv2/run_infllmv2_longbench_accuracy.py \
    --model-dir "${MODEL_DIR}" \
    --eval-config "${eval_config}" \
    --output-dir "${rank_dir}" \
    --modes oracle_sparse,es_sparse \
    --sample-count "${SAMPLE_COUNT}" \
    --max-input-length "${MAX_INPUT_LENGTH}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --rank 256 \
    --route-query lowrank_next_wq \
    --lookahead-source mid \
    --lowrank-cache "${LOWRANK_CACHE}"

  run_cmd "${PYTHON_BIN}" analysis/es_infllmv2/run_infllmv2_longbench_accuracy.py \
    --model-dir "${MODEL_DIR}" \
    --eval-config "${eval_config}" \
    --output-dir "${exact_dir}" \
    --modes es_sparse \
    --sample-count "${SAMPLE_COUNT}" \
    --max-input-length "${MAX_INPUT_LENGTH}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --rank 256 \
    --route-query exact_next_wq \
    --lookahead-source mid
}

for dataset in ${DATASETS}; do
  run_dataset "${dataset}"
done

run_cmd "${PYTHON_BIN}" analysis/es_infllmv2/summarize_compatibility.py \
  --root "${OUTPUT_ROOT}" \
  --datasets "${DATASETS}"
