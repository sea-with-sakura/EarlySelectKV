#!/usr/bin/env bash
set -euo pipefail

sample_count="${SAMPLE_COUNT:-20}"
token_budget="${TOKEN_BUDGET:-1024}"
output_root="${OUTPUT_ROOT:-analysis/figure_w6_multistep_fidelity/out}"
devices_csv="${DEVICES:-0,1}"
samples_per_shard="${SAMPLES_PER_SHARD:-10}"
python_bin="${PYTHON_BIN:-python}"

pipeline_config="${PIPELINE_CONFIG:-config/pipeline_config/longbench/mistral-7b-instruct-v0.2.json}"
eval_config="${EVAL_CONFIG:-config/eval_config/longbench/multi_news.json}"
method="${METHOD:-anchor_quality_stage2}"

IFS=',' read -r -a devices <<< "${devices_csv}"
if [[ "${#devices[@]}" -eq 0 ]]; then
  echo "No CUDA devices specified via DEVICES." >&2
  exit 1
fi

mkdir -p "${output_root}"

if ! command -v "${python_bin}" >/dev/null 2>&1 && [[ ! -x "${python_bin}" ]]; then
  echo "Python interpreter not found: ${python_bin}" >&2
  echo "Set PYTHON_BIN to an environment with torch/litgpt installed." >&2
  exit 1
fi

offset=0
shard=0
pids=()
while [[ "${offset}" -lt "${sample_count}" ]]; do
  limit="${samples_per_shard}"
  remaining=$((sample_count - offset))
  if [[ "${remaining}" -lt "${limit}" ]]; then
    limit="${remaining}"
  fi

  device="${devices[$((shard % ${#devices[@]}))]}"
  shard_dir="${output_root}/shard_${shard}_offset_${offset}_limit_${limit}"
  mkdir -p "${shard_dir}"

  echo "Launching shard ${shard}: CUDA_VISIBLE_DEVICES=${device}, offset=${offset}, limit=${limit}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q LONGBENCH_SAMPLE_OFFSET=%q LONGBENCH_SAMPLE_LIMIT=%q STAGE2_PROBE_HEAD_ROWS=0 STAGE2_PROBE_CANDIDATES=rocket_qout,lookahead_qmid %q -m pipeline.stage2_anchor_probe.main --pipeline_config_dir %q --eval_config_dir %q --output_folder_dir %q --job_post_via terminal --method %q --token_budget %q\n' \
      "${device}" "${offset}" "${limit}" "${python_bin}" "${pipeline_config}" "${eval_config}" "${shard_dir}" "${method}" "${token_budget}"
    offset=$((offset + limit))
    shard=$((shard + 1))
    continue
  fi
  (
    export CUDA_VISIBLE_DEVICES="${device}"
    export LONGBENCH_SAMPLE_OFFSET="${offset}"
    export LONGBENCH_SAMPLE_LIMIT="${limit}"
    export STAGE2_PROBE_HEAD_ROWS=0
    export STAGE2_PROBE_CANDIDATES=rocket_qout,lookahead_qmid
    "${python_bin}" -m pipeline.stage2_anchor_probe.main \
      --pipeline_config_dir "${pipeline_config}" \
      --eval_config_dir "${eval_config}" \
      --output_folder_dir "${shard_dir}" \
      --job_post_via terminal \
      --method "${method}" \
      --token_budget "${token_budget}"
  ) > "${shard_dir}/run.log" 2>&1 &
  pids+=("$!")

  offset=$((offset + limit))
  shard=$((shard + 1))
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Planned ${shard} fixed-trajectory shards."
  exit 0
fi

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "At least one W6 shard failed. Inspect ${output_root}/shard_*/run.log" >&2
  exit "${status}"
fi

"${python_bin}" analysis/figure_w6_multistep_fidelity/plot_multistep_fidelity.py \
  --root "${output_root}" \
  --out-dir "${output_root}/plots"

echo "W6 outputs written to ${output_root}/plots"
