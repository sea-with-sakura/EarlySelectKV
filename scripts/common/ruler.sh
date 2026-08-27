#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

method="${1:-}"
output_dir_root="${2:-}"
token_budget="${3:-}"
datasets_override="${4:-}"
seq_lengths_override="${5:-}"

task="${TASK:-ruler}"
model="${MODEL:?MODEL must be set by the model wrapper}"
pipeline_config="${PIPELINE_CONFIG:-config/pipeline_config/${task}/${model}.json}"
eval_config="${EVAL_CONFIG:-config/eval_config/${task}.json}"
default_datasets="${DATASETS:-niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2}"
default_seq_lengths="${MAX_SEQ_LENGTHS:?MAX_SEQ_LENGTHS must be set by the model wrapper}"
datasets="${datasets_override:-$default_datasets}"
summary_datasets="${RULER_SUMMARY_DATASETS:-$datasets}"
seq_lengths="${seq_lengths_override:-$default_seq_lengths}"
seed="${SEED:-42}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-${seed}}"
planned_runs=0

if [ -z "${method}" ] || [ -z "${output_dir_root}" ]; then
  echo "Usage:"
  echo "bash scripts/${task}/${model}.sh <method> <output_dir_root> [token_budget] [datasets] [max_seq_lengths]"
  echo "token_budget is required for non-baseline methods."
  print_method_help
  exit 1
fi

if ! runner="$(resolve_pipeline_runner "${method}")"; then
  echo "Invalid method: ${method}"
  print_method_help
  exit 1
fi

if [ "${method}" != "baseline" ] && [ -z "${token_budget}" ]; then
  echo "${method} mode requires <token_budget> as the third argument."
  exit 1
fi

for max_seq_length in ${seq_lengths}; do
  if [ "${method}" = "baseline" ]; then
    group_dir="${output_dir_root}/${task}/${max_seq_length}/${method}/${model}"
  else
    group_dir="${output_dir_root}/${task}/${max_seq_length}/${method}/${token_budget}/${model}"
  fi

  if [ "${method}" = "baseline" ]; then
    for dataset in ${datasets}; do
      output_dir="${group_dir}/${dataset}/"
      if ! should_run_eval "${output_dir}" "${pipeline_config}" "${eval_config}" "${method}" "" "${dataset}" "${max_seq_length}" "${SCDQ_MODE:-0}"; then
        continue
      fi
      planned_runs=$((planned_runs + 1))
      if [ "${QUEUE_CHECK_ONLY:-0}" = "1" ]; then
        continue
      fi
      run_cmd python -m "${runner}" \
        --method "${method}" \
        --dataset "${dataset}" \
        --max_seq_length "${max_seq_length}" \
        --exp_desc "${task}_${dataset}_${model}_${method}" \
        --pipeline_config_dir "${pipeline_config}" \
        --eval_config_dir "${eval_config}" \
        --seed "${seed}" \
        --output_folder_dir "${output_dir}"
    done
  else
    for dataset in ${datasets}; do
      output_dir="${group_dir}/${dataset}/"
      if ! should_run_eval "${output_dir}" "${pipeline_config}" "${eval_config}" "${method}" "${token_budget}" "${dataset}" "${max_seq_length}" "${SCDQ_MODE:-0}"; then
        continue
      fi
      planned_runs=$((planned_runs + 1))
      if [ "${QUEUE_CHECK_ONLY:-0}" = "1" ]; then
        continue
      fi
      run_cmd python -m "${runner}" \
        --method "${method}" \
        --dataset "${dataset}" \
        --max_seq_length "${max_seq_length}" \
        --token_budget "${token_budget}" \
        --exp_desc "${task}_${dataset}_${model}_${method}_${token_budget}" \
        --pipeline_config_dir "${pipeline_config}" \
        --eval_config_dir "${eval_config}" \
        --seed "${seed}" \
        --output_folder_dir "${output_dir}"
    done
  fi

  if [ "${QUEUE_CHECK_ONLY:-0}" != "1" ]; then
    run_cmd python "${SCRIPT_DIR}/ruler_collect_summary.py" \
      --group-dir "${group_dir}" \
      --datasets "${summary_datasets}"
  fi
done

queue_check_finalize "${planned_runs}" "${output_dir_root}/${task}/${method}/${model}/" || exit $?
if [ "${QUEUE_CHECK_ONLY:-0}" = "1" ]; then
  exit 0
fi
