#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
source "${SCRIPT_DIR}/hf_env.sh"

task="${1:-}"
models="${2:-}"
methods="${3:-}"
output_dir_root="${4:-}"
cuda="${5:-0}"
token_budgets="${6:-}"
datasets="${7:-}"
session_name="${8:-}"

conda_env="${CONDA_ENV:-env_torch}"
reuse_existing_session="${TMUX_REUSE_EXISTING_SESSION:-0}"

print_usage() {
  echo "Usage:"
  echo "bash scripts/common/tmux_budget_sweep.sh <task> \"<models>\" \"<methods>\" <output_dir_root> <cuda> \"<token_budgets>\" [datasets] [session_name]"
  echo "Examples:"
  echo "bash scripts/common/tmux_budget_sweep.sh longbench \"mistral-7b-instruct-v0.2 llama3-8b-instruct\" \"earlyselectkv_in_mid_local rocketkv\" result_new 0 \"1024 2048 4096\""
  echo "bash scripts/common/tmux_budget_sweep.sh longbench \"llama3-8b-instruct\" \"baseline topk\" result_new 0 \"1024 2048\""
  echo "bash scripts/common/tmux_budget_sweep.sh math_reasoning \"llama3.1-8b-instruct qwen2.5-7b-instruct\" \"baseline rocketkv earlyselectkv\" result_math 0 \"1024 4096\" \"math500\""
}

validate_runner_and_method() {
  local model="$1"
  local method="$2"
  local runner="scripts/${task}/${model}.sh"

  if [ ! -f "${runner}" ]; then
    echo "Runner not found: ${runner}" >&2
    return 1
  fi
  if ! resolve_pipeline_runner "${method}" >/dev/null; then
    echo "Invalid method: ${method}" >&2
    print_method_help
    return 1
  fi
}

build_runner_cmd() {
  local runner="$1"
  local method="$2"
  local budget="$3"
  local datasets_arg="$4"
  printf 'bash %q %q %q %q %q' "${runner}" "${method}" "${output_dir_root}" "${budget}" "${datasets_arg}"
}

queue_check_combo() {
  local runner="$1"
  local model="$2"
  local method="$3"
  local budget="$4"
  local output=""
  local status=0

  if output="$(QUEUE_CHECK_ONLY=1 DRY_RUN=0 bash "${runner}" "${method}" "${output_dir_root}" "${budget}" "${datasets}" 2>&1)"; then
    status=0
  else
    status=$?
  fi

  if [ "${status}" -eq "${QUEUE_CHECK_SKIP_EXIT_CODE}" ]; then
    if [ -n "${budget}" ]; then
      echo "Skip queued combo (already complete): model=${model}; method=${method}; budget=${budget}"
    else
      echo "Skip queued combo (already complete): model=${model}; method=${method}"
    fi
    return 1
  fi

  if [ "${status}" -eq 0 ]; then
    return 0
  fi

  echo "Queue pre-check failed for model=${model}, method=${method}, budget=${budget:-baseline}" >&2
  if [ -n "${output}" ]; then
    echo "${output}" >&2
  fi
  return "${status}"
}

ensure_session() {
  if tmux has-session -t "${session_name}" 2>/dev/null; then
    return 0
  fi
  tmux new-session -d -s "${session_name}"
  tmux send-keys -t "${session_name}" "source ~/.bashrc; if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then source ~/miniconda3/etc/profile.d/conda.sh; elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then source ~/anaconda3/etc/profile.d/conda.sh; fi; source ${SCRIPT_DIR@Q}/hf_env.sh; hf_export_env; conda activate ${conda_env}" C-m
  tmux send-keys -t "${session_name}" "export CUDA_VISIBLE_DEVICES=${cuda}" C-m
  tmux send-keys -t "${session_name}" "export DRY_RUN=${DRY_RUN:-0}" C-m
}

queue_combo() {
  local runner="$1"
  local model="$2"
  local method="$3"
  local budget="$4"
  local label="$5"
  local cmd

  ensure_session
  tmux send-keys -t "${session_name}" "echo ${label@Q}" C-m
  cmd="$(build_runner_cmd "${runner}" "${method}" "${budget}" "${datasets}")"
  tmux send-keys -t "${session_name}" "${cmd}" C-m
}

if [ -z "${task}" ] || [ -z "${models}" ] || [ -z "${methods}" ] || [ -z "${output_dir_root}" ]; then
  print_usage
  exit 1
fi

if [ -z "${session_name}" ]; then
  session_name="${task}-model-method-budget-${cuda}"
fi

if tmux has-session -t "${session_name}" 2>/dev/null; then
  if [ "${reuse_existing_session}" != "1" ]; then
    echo "tmux session already exists: ${session_name}"
    echo "Attach directly instead of queueing more commands. Set TMUX_REUSE_EXISTING_SESSION=1 to keep the old queue-into-existing behavior."
    if [ "${TMUX_ATTACH:-1}" = "1" ]; then
      if [ -n "${TMUX:-}" ]; then
        tmux switch-client -t "${session_name}"
      else
        tmux attach -t "${session_name}"
      fi
    fi
    exit 0
  fi
fi

needs_budget=0
for method in ${methods}; do
  if [ "${method}" != "baseline" ]; then
    needs_budget=1
  fi
done

if [ "${needs_budget}" -eq 1 ] && [ -z "${token_budgets}" ]; then
  echo "Non-baseline methods require a non-empty token budget list." >&2
  print_usage
  exit 1
fi

for model in ${models}; do
  for method in ${methods}; do
    validate_runner_and_method "${model}" "${method}"
  done
done

session_preexisting=0
if tmux has-session -t "${session_name}" 2>/dev/null; then
  session_preexisting=1
fi

queued_commands=0
skipped_commands=0
queue_status=0

for model in ${models}; do
  runner="scripts/${task}/${model}.sh"
  for method in ${methods}; do
    if [ "${method}" = "baseline" ]; then
      if queue_check_combo "${runner}" "${model}" "${method}" ""; then
        queue_status=0
      else
        queue_status=$?
      fi
      if [ "${queue_status}" -eq 0 ]; then
        queue_combo \
          "${runner}" \
          "${model}" \
          "${method}" \
          "" \
          "===== Start model: ${model}; method: ${method} ====="
        queued_commands=$((queued_commands + 1))
      elif [ "${queue_status}" -eq 1 ]; then
        skipped_commands=$((skipped_commands + 1))
      else
        exit "${queue_status}"
      fi
      continue
    fi

    for token_budget in ${token_budgets}; do
      if queue_check_combo "${runner}" "${model}" "${method}" "${token_budget}"; then
        queue_status=0
      else
        queue_status=$?
      fi
      if [ "${queue_status}" -eq 0 ]; then
        queue_combo \
          "${runner}" \
          "${model}" \
          "${method}" \
          "${token_budget}" \
          "===== Start model: ${model}; method: ${method}; budget: ${token_budget} ====="
        queued_commands=$((queued_commands + 1))
      elif [ "${queue_status}" -eq 1 ]; then
        skipped_commands=$((skipped_commands + 1))
      else
        exit "${queue_status}"
      fi
    done
  done
done

if [ "${queued_commands}" -eq 0 ]; then
  echo "No pending combinations to queue. skipped=${skipped_commands}"
else
  if [ "${session_preexisting}" -eq 1 ]; then
    echo "Queued ${queued_commands} command(s) into existing tmux session: ${session_name}"
  else
    echo "Queued ${queued_commands} command(s) into new tmux session: ${session_name}"
  fi
  echo "Pre-check skipped ${skipped_commands} already-completed combination(s)."
fi

if [ "${TMUX_ATTACH:-1}" = "1" ] && tmux has-session -t "${session_name}" 2>/dev/null; then
  tmux attach -t "${session_name}"
fi
