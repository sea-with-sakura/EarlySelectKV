#!/usr/bin/env bash
set -euo pipefail

models="${MODELS:-mistral-7b-instruct-v0.2 llama3-8b-instruct llama3.1-8b-instruct qwen2.5-7b-instruct}"
domains="${DOMAINS:-alpaca12k_decode_calib qasper qmsum}"
budget="${TOKEN_BUDGET:-1024}"
output_root="${OUTPUT_ROOT:-result/esmix_domain_stability}"
sample_limit="${SAMPLE_COUNT:-50}"
seed="${SEED:-42}"

export LONGBENCH_SAMPLE_LIMIT="${sample_limit}"
export LONGBENCH_SAMPLE_SEED="${seed}"

for model in ${models}; do
  for domain in ${domains}; do
    MODEL="${model}" bash scripts/common/longbench.sh \
      anchor_quality_stage2 "${output_root}" "${budget}" "${domain}"
  done
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

python analysis/esmix_domain_stability.py \
  --root "${output_root}" \
  --budget "${budget}" \
  --models ${models} \
  --domains ${domains}
