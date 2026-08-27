#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV:-env_torch}"

models="${MODELS:-mistral-7b-instruct-v0.2}"
budgets="${BUDGETS:-1024}"
methods="${METHODS:-anchor_quality_stage2}"
datasets="${DATASETS:-qasper}"
cuda_devices="${CUDA_DEVICES:-1}"
output_dir_root="${OUTPUT_DIR_ROOT:-analysis/figure_stage2_anchor_quality/out}"
sample_limit="${LONGBENCH_SAMPLE_LIMIT:-50}"
max_records="${STAGE2_PROBE_MAX_RECORDS:-0}"
session_prefix="${SESSION_PREFIX:-anchor-quality-stage2}"

export LONGBENCH_SAMPLE_LIMIT="${sample_limit}"
export STAGE2_PROBE_MAX_RECORDS="${max_records}"

mapfile -t devices < <(printf '%s\n' ${cuda_devices})
num_devices="${#devices[@]}"
if [ "${num_devices}" -eq 0 ]; then
  echo "No CUDA devices provided."
  exit 1
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
  idx=0
  for model in ${models}; do
    for budget in ${budgets}; do
      for method in ${methods}; do
        for dataset in ${datasets}; do
          device_idx=$((idx % num_devices))
          printf 'CUDA_VISIBLE_DEVICES=%q MODEL=%q LONGBENCH_SAMPLE_LIMIT=%q bash scripts/longbench/%q.sh %q %q %q %q\n' \
            "${devices[$device_idx]}" "${model}" "${sample_limit}" "${model}" "${method}" "${output_dir_root}" "${budget}" "${dataset}"
          idx=$((idx + 1))
        done
      done
    done
  done
  echo "Planned ${idx} anchor_quality_stage2 probe commands."
  exit 0
fi

for i in "${!devices[@]}"; do
  session="${session_prefix}-${i}"
  tmux kill-session -t "${session}" 2>/dev/null || true
  tmux new-session -d -s "${session}" -c "${REPO_ROOT}"
  tmux send-keys -t "${session}" "source ${HOME}/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ${HOME}/anaconda3/etc/profile.d/conda.sh" C-m
  tmux send-keys -t "${session}" "conda activate ${CONDA_ENV:-env_torch}" C-m
  tmux send-keys -t "${session}" "export LONGBENCH_SAMPLE_LIMIT=${sample_limit}" C-m
  tmux send-keys -t "${session}" "export STAGE2_PROBE_MAX_RECORDS=${max_records}" C-m
  tmux send-keys -t "${session}" "export CUDA_VISIBLE_DEVICES=${devices[$i]}" C-m
done

idx=0
for model in ${models}; do
  for budget in ${budgets}; do
    for method in ${methods}; do
      for dataset in ${datasets}; do
        device_idx=$((idx % num_devices))
        session="${session_prefix}-${device_idx}"
        cmd="MODEL=${model} bash scripts/longbench/${model}.sh ${method} ${output_dir_root} ${budget} ${dataset}"
        tmux send-keys -t "${session}" "${cmd}" C-m
        idx=$((idx + 1))
      done
    done
  done
done

echo "Queued ${idx} anchor_quality_stage2 probe commands into ${num_devices} tmux sessions (${session_prefix}-0 ...)."
