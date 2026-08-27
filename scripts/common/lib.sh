#!/usr/bin/env bash

QUEUE_CHECK_SKIP_EXIT_CODE=10

resolve_pipeline_runner() {
  local method="${1,,}"

  case "${method}" in
    baseline|rocketkv|topk|quest|loki)
      echo "pipeline.${method}.main"
      ;;
    rocketkv_topk)
      echo "pipeline.rocketkv.main"
      ;;
    anchor_quality_stage2|dense_stage2|topk_stage2|qout_stage2|qmid_stage2|qin_stage2|qinwithmid_stage2)
      echo "pipeline.stage2_anchor_probe.main"
      ;;
    earlyselectkv|earlyselectkv_in|earlyselectkv_in_mid|earlyselectkv_in_mid_local|earlyselectkv_topk|earlyselectkv_local|earlyselectkv_hsa_local|earlyselectkv_mid_svd_r64|earlyselectkv_mid_svd_r128|earlyselectkv_mid_svd_r256|earlyselectkv_mid_svd_r512)
      echo "pipeline.lookaheadkv.main"
      ;;
    lookaheadkv|lookaheadkv_in|lookahead_in_mid|lookahead_in_mid_local|lookaheadkv_topk|lookaheadkv_local|lookaheadkv_hsa_local|lookaheadkv_mid_svd_r64|lookaheadkv_mid_svd_r128|lookaheadkv_mid_svd_r256|lookaheadkv_mid_svd_r512)
      echo "pipeline.lookaheadkv.main"
      ;;
    earlyselect_quest|lookahead_quest)
      echo "pipeline.lookahead_quest.main"
      ;;
    earlyselect_loki|lookahead_loki)
      echo "pipeline.lookahead_loki.main"
      ;;
    *)
      return 1
      ;;
  esac
}

print_method_help() {
  echo "method can be: baseline, topk, rocketkv, rocketkv_topk, quest, earlyselect_quest, loki, earlyselect_loki, earlyselectkv, earlyselectkv_in, earlyselectkv_in_mid, earlyselectkv_in_mid_local, earlyselectkv_topk, earlyselectkv_local, earlyselectkv_hsa_local"
}

run_cmd() {
  if [ "${DRY_RUN:-0}" = "1" ]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

should_run_eval() {
  local output_dir="$1"
  local pipeline_config="$2"
  local eval_config="$3"
  local method="$4"
  local token_budget="$5"
  local dataset="$6"
  local max_seq_length="$7"
  local scdq_mode="$8"
  local model_name_or_path="${9:-}"
  local seed="${SEED:-42}"

  local output_dir_clean="${output_dir%/}"

  if [ ! -d "${output_dir_clean}" ]; then
    return 0
  fi

  local status=0
  python scripts/common/should_run_eval.py \
    --output-dir "${output_dir_clean}" \
    --pipeline-config "${pipeline_config}" \
    --eval-config "${eval_config}" \
    --method "${method}" \
    --token-budget "${token_budget}" \
    --dataset "${dataset}" \
    --max-seq-length "${max_seq_length}" \
    --scdq-mode "${scdq_mode}" \
    --seed "${seed}" \
    --model-name-or-path "${model_name_or_path}" || status=$?

  if [ ${status} -eq 1 ]; then
    echo "Skip completed (config match): ${output_dir_clean}"
    return 1
  fi
  if [ ${status} -eq 2 ]; then
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    mv "${output_dir_clean}" "${output_dir_clean}__mismatch_${ts}"
    return 0
  fi
  if [ ${status} -ne 0 ]; then
    echo "Resume check failed (status=${status}); rerunning ${output_dir_clean}"
  fi
  return 0
}

queue_check_finalize() {
  local planned_runs="$1"
  local label="${2:-task}"

  if [ "${QUEUE_CHECK_ONLY:-0}" = "1" ]; then
    if [ "${planned_runs}" -gt 0 ]; then
      return 0
    fi
    return "${QUEUE_CHECK_SKIP_EXIT_CODE}"
  fi

  if [ "${planned_runs}" -eq 0 ]; then
    echo "Skip all completed: ${label}"
    return 0
  fi

  return 0
}

queue_check_runner() {
  local label="$1"
  shift

  local output=""
  local status=0

  if output="$(QUEUE_CHECK_ONLY=1 DRY_RUN=0 "$@" 2>&1)"; then
    status=0
  else
    status=$?
  fi

  if [ "${status}" -eq "${QUEUE_CHECK_SKIP_EXIT_CODE}" ]; then
    echo "Skip queued combo (already complete): ${label}"
    return "${QUEUE_CHECK_SKIP_EXIT_CODE}"
  fi

  if [ "${status}" -eq 0 ]; then
    return 0
  fi

  echo "Queue pre-check failed: ${label}" >&2
  if [ -n "${output}" ]; then
    echo "${output}" >&2
  fi
  return "${status}"
}
