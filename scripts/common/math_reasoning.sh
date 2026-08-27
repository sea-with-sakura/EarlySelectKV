#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

method="${1:-}"
output_dir_root="${2:-}"
token_budget="${3:-}"
datasets_override="${4:-}"

task="${TASK:-math_reasoning}"
output_task="${MATH_REASONING_OUTPUT_TASK:-${TASK_OUTPUT_NAME:-${task}}}"
model="${MODEL:?MODEL must be set by the model wrapper}"
pipeline_config="${PIPELINE_CONFIG:-config/pipeline_config/${task}/${model}.json}"
default_datasets="${DATASETS:-math500}"
datasets="${datasets_override:-$default_datasets}"
seed="${SEED:-42}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-${seed}}"
planned_runs=0

if [ -z "${method}" ] || [ -z "${output_dir_root}" ]; then
  echo "Usage:"
  echo "bash scripts/${task}/${model}.sh <method> <output_dir_root> [token_budget] [datasets]"
  echo "token_budget is required for non-baseline methods."
  echo "datasets example: \"math500\""
  print_method_help
  exit 1
fi

if ! runner="$(resolve_pipeline_runner "${method}")"; then
  echo "Invalid method: ${method}"
  print_method_help
  exit 1
fi

if [ "${method}" = "baseline" ]; then
  for dataset in ${datasets}; do
    output_dir="${output_dir_root}/${output_task}/${method}/${model}/${dataset}/"
    if ! should_run_eval "${output_dir}" "${pipeline_config}" "config/eval_config/${task}/${dataset}.json" "${method}" "" "${dataset}" "" "${SCDQ_MODE:-0}"; then
      continue
    fi
    planned_runs=$((planned_runs + 1))
    if [ "${QUEUE_CHECK_ONLY:-0}" = "1" ]; then
      continue
    fi
    run_cmd python -m "${runner}" \
      --method "${method}" \
      --exp_desc "${task}_${dataset}_${model}_${method}" \
      --pipeline_config_dir "${pipeline_config}" \
      --eval_config_dir "config/eval_config/${task}/${dataset}.json" \
      --seed "${seed}" \
      --output_folder_dir "${output_dir}"
  done
  queue_check_finalize "${planned_runs}" "${output_dir_root}/${output_task}/${method}/${model}/" || exit $?
  if [ "${QUEUE_CHECK_ONLY:-0}" = "1" ] || [ "${planned_runs}" -eq 0 ]; then
    exit 0
  fi
  run_cmd python -m visualization.math_reasoning_summary \
    --output_dir "${output_dir_root}/${output_task}/${method}/${model}/" \
    --datasets "${datasets}"
  exit 0
fi

if [ -z "${token_budget}" ]; then
  echo "${method} mode requires <token_budget> as the third argument."
  exit 1
fi

for dataset in ${datasets}; do
  output_dir="${output_dir_root}/${output_task}/${method}/${token_budget}/${model}/${dataset}/"
  if ! should_run_eval "${output_dir}" "${pipeline_config}" "config/eval_config/${task}/${dataset}.json" "${method}" "${token_budget}" "${dataset}" "" "${SCDQ_MODE:-0}"; then
    continue
  fi
  planned_runs=$((planned_runs + 1))
  if [ "${QUEUE_CHECK_ONLY:-0}" = "1" ]; then
    continue
  fi
  run_cmd python -m "${runner}" \
    --method "${method}" \
    --token_budget "${token_budget}" \
    --exp_desc "${task}_${dataset}_${model}_${method}_${token_budget}" \
    --pipeline_config_dir "${pipeline_config}" \
    --eval_config_dir "config/eval_config/${task}/${dataset}.json" \
    --seed "${seed}" \
    --output_folder_dir "${output_dir}"
done

queue_check_finalize "${planned_runs}" "${output_dir_root}/${output_task}/${method}/${token_budget}/${model}/" || exit $?
if [ "${QUEUE_CHECK_ONLY:-0}" = "1" ] || [ "${planned_runs}" -eq 0 ]; then
  exit 0
fi

run_cmd python -m visualization.math_reasoning_summary \
  --output_dir "${output_dir_root}/${output_task}/${method}/${token_budget}/${model}/" \
  --datasets "${datasets}"
