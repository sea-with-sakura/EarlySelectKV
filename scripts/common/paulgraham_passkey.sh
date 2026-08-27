#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

method="${1:-}"
output_dir_root="${2:-}"
token_budget="${3:-}"
dataset_override="${4:-}"

task="${TASK:-paulgraham_passkey}"
model="${MODEL:?MODEL must be set by the model wrapper}"
default_dataset="${DATASET:?DATASET must be set by the model wrapper}"
datasets="${dataset_override:-${default_dataset}}"
pipeline_config="${PIPELINE_CONFIG:-config/pipeline_config/${task}/${model}.json}"
seed="${SEED:-42}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-${seed}}"
planned_runs=0

run_paulgraham_visualization() {
  local output_dir="$1"
  run_cmd python -m visualization.paulgraham_heatmap.paulgraham_visualization \
    --output_dir "${output_dir}"
}

needs_paulgraham_summary() {
  local output_dir="${1%/}"
  [ ! -s "${output_dir}/paulgraham_result_summary.json" ] || [ ! -s "${output_dir}/paulgraham_result_summary.csv" ]
}

finalize_paulgraham_runs() {
  local planned_runs="$1"
  local output_dir="${2%/}"

  if [ "${QUEUE_CHECK_ONLY:-0}" = "1" ]; then
    if [ "${planned_runs}" -gt 0 ] || needs_paulgraham_summary "${output_dir}"; then
      return 0
    fi
    return "${QUEUE_CHECK_SKIP_EXIT_CODE}"
  fi

  if [ "${planned_runs}" -eq 0 ] && ! needs_paulgraham_summary "${output_dir}"; then
    echo "Skip all completed: ${output_dir}"
    return 1
  fi

  run_paulgraham_visualization "${output_dir}"
}

if [ -z "${method}" ] || [ -z "${output_dir_root}" ]; then
  echo "Usage:"
  echo "bash scripts/${task}/${model}.sh <method> <output_dir_root> [token_budget] [datasets]"
  echo "token_budget is required for non-baseline methods."
  echo "datasets is optional; the model wrapper default is used when omitted."
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
    eval_config="${EVAL_CONFIG:-config/eval_config/${task}/${dataset}.json}"
    output_dir="${output_dir_root}/${task}/${method}/${model}/${dataset}/"
    if ! should_run_eval "${output_dir}" "${pipeline_config}" "${eval_config}" "${method}" "" "${dataset}" "" "${SCDQ_MODE:-0}"; then
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
      --eval_config_dir "${eval_config}" \
      --seed "${seed}" \
      --output_folder_dir "${output_dir}"
  done
  if finalize_paulgraham_runs "${planned_runs}" "${output_dir_root}/${task}/${method}/${model}/"; then
    status=0
  else
    status=$?
  fi
  if [ "${status}" -eq 1 ]; then
    exit 0
  fi
  if [ "${status}" -ne 0 ]; then
    exit "${status}"
  fi
  exit 0
fi

if [ -z "${token_budget}" ]; then
  echo "${method} mode requires <token_budget> as the third argument."
  exit 1
fi

for dataset in ${datasets}; do
  eval_config="${EVAL_CONFIG:-config/eval_config/${task}/${dataset}.json}"
  output_dir="${output_dir_root}/${task}/${method}/${token_budget}/${model}/${dataset}/"
  if ! should_run_eval "${output_dir}" "${pipeline_config}" "${eval_config}" "${method}" "${token_budget}" "${dataset}" "" "${SCDQ_MODE:-0}"; then
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
    --eval_config_dir "${eval_config}" \
    --seed "${seed}" \
    --output_folder_dir "${output_dir}"
done
if finalize_paulgraham_runs "${planned_runs}" "${output_dir_root}/${task}/${method}/${token_budget}/${model}/"; then
  status=0
else
  status=$?
fi
if [ "${status}" -eq 1 ]; then
  exit 0
fi
if [ "${status}" -ne 0 ]; then
  exit "${status}"
fi
