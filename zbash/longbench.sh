#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SESSION_NAME="${SESSION_NAME:-longbench-earlyselect-rocket}"
CONDA_ENV="${CONDA_ENV:-env_torch}"
TASK="${TASK:-longbench}"

MODELS="${MODELS:-mistral-7b-instruct-v0.2}"
METHODS="${METHODS:-rocketkv earlyselectkv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-result_new}"
BUDGETS="${BUDGETS:-512}"
CUDA_DEVICES="${CUDA_DEVICES:-0 1 2 3}"

DATASETS="${DATASETS:-narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_retrieval_en passage_count lcc repobench-p}"

read -r -a CUDA_ARRAY <<<"${CUDA_DEVICES}"
if [ "${#CUDA_ARRAY[@]}" -eq 0 ]; then
  echo "CUDA_DEVICES must contain at least one GPU id." >&2
  exit 1
fi

setup_window() {
  local window="$1"
  local cuda="$2"
  local target="${SESSION_NAME}:${window}"

  tmux send-keys -t "${target}" "source ~/.bashrc; if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then source ~/miniconda3/etc/profile.d/conda.sh; elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then source ~/anaconda3/etc/profile.d/conda.sh; fi; source ${REPO_ROOT@Q}/scripts/common/hf_env.sh; hf_export_env; conda activate ${CONDA_ENV}" C-m
  tmux send-keys -t "${target}" "cd ${REPO_ROOT@Q}" C-m
  tmux send-keys -t "${target}" "export CUDA_VISIBLE_DEVICES=${cuda}" C-m
}

ensure_window() {
  local cuda="$1"
  local window="gpu${cuda}"

  if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    tmux new-session -d -s "${SESSION_NAME}" -n "${window}" -c "${REPO_ROOT}"
    setup_window "${window}" "${cuda}"
    return 0
  fi

  if tmux list-windows -t "${SESSION_NAME}" -F "#{window_name}" | grep -Fxq "${window}"; then
    return 0
  fi

  tmux new-window -d -t "${SESSION_NAME}" -n "${window}" -c "${REPO_ROOT}"
  setup_window "${window}" "${cuda}"
}

dataset_group_for_slot() {
  local slot="$1"
  local group=""
  local i=0
  local dataset

  for dataset in ${DATASETS}; do
    if [ $((i % ${#CUDA_ARRAY[@]})) -eq "${slot}" ]; then
      if [ -n "${group}" ]; then
        group+=" "
      fi
      group+="${dataset}"
    fi
    i=$((i + 1))
  done

  printf '%s' "${group}"
}

queue_gpu() {
  local slot="$1"
  local cuda="${CUDA_ARRAY[$slot]}"
  local window="gpu${cuda}"
  local target="${SESSION_NAME}:${window}"
  local datasets
  local model
  local method
  local budget
  local cmd

  datasets="$(dataset_group_for_slot "${slot}")"
  if [ -z "${datasets}" ]; then
    return 0
  fi

  ensure_window "${cuda}"
  tmux send-keys -t "${target}" "export CUDA_VISIBLE_DEVICES=${cuda}" C-m

  for model in ${MODELS}; do
    for budget in ${BUDGETS}; do
      for method in ${METHODS}; do
        cmd=$(printf 'unset LOOKAHEAD_Q_PROBE LOOKAHEAD_Q_RES_SCALES LONGBENCH_SAMPLE_LIMIT PASSKEY_SAMPLE_LIMIT; MODEL=%q TASK=%q DATASETS=%q bash scripts/common/longbench.sh %q %q %q %q' \
          "${model}" "${TASK}" "${datasets}" "${method}" "${OUTPUT_ROOT}" "${budget}" "${datasets}")

        tmux send-keys -t "${target}" "echo '===== ${TASK} model=${model} method=${method} budget=${budget} cuda=${cuda} datasets=${datasets} ====='" C-m
        tmux send-keys -t "${target}" "${cmd}" C-m
      done
    done
  done
}

for slot in "${!CUDA_ARRAY[@]}"; do
  queue_gpu "${slot}"
done

echo "Queued LongBench into tmux session: ${SESSION_NAME}"
echo "Models: ${MODELS}"
echo "Methods: ${METHODS}"
echo "Budgets: ${BUDGETS}"
echo "CUDA devices: ${CUDA_DEVICES}"
echo "Output root: ${OUTPUT_ROOT}/longbench/<method>/<budget>/<model>/"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
