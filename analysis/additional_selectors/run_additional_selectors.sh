#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-python}"
output_root="${OUTPUT_ROOT:-result/additional_selectors}"
pca_root="${PCA_ROOT:-dataset/loki_pca}"
pca_max_tokens="${PCA_MAX_TOKENS:-4096}"

run_eval() {
  local model="$1" method="$2" budget="$3"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    DRY_RUN=1 bash "scripts/longbench/${model}.sh" "${method}" "${output_root}" "${budget}"
  else
    bash "scripts/longbench/${model}.sh" "${method}" "${output_root}" "${budget}"
  fi
}

ensure_pca() {
  local model="$1"
  local pca_dir="${pca_root}/${model}/wikitext-valid_t${pca_max_tokens}"
  if [[ "${DRY_RUN:-0}" != "1" ]] && [[ ! -f "${pca_dir}/key/pca_components/pca_components_layer_1.pt" ]]; then
    "${python_bin}" -m pipeline.loki.pca_analysis \
      --pipeline-config "config/pipeline_config/longbench/${model}.json" \
      --output-dir "${pca_dir}" \
      --calibration-dataset wikitext-valid \
      --max-tokens "${pca_max_tokens}" \
      --rotary-type postrotary
  fi
  export LOKI_PCA_DIR="${pca_dir}"
}

for method in quest earlyselect_quest; do
  run_eval qwen2.5-32b-instruct "${method}" 256
  run_eval qwen2.5-7b-instruct "${method}" 128
  for budget in 128 1024 2048 4096; do
    run_eval mistral-7b-instruct-v0.2 "${method}" "${budget}"
  done
done

for model in qwen2.5-7b-instruct mistral-7b-instruct-v0.2; do
  ensure_pca "${model}"
  budgets="128"
  if [[ "${model}" == "mistral-7b-instruct-v0.2" ]]; then
    budgets="128 1024 2048 4096"
  fi
  for method in loki earlyselect_loki; do
    for budget in ${budgets}; do
      run_eval "${model}" "${method}" "${budget}"
    done
  done
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  "${python_bin}" analysis/additional_selectors/plot_additional_selectors.py \
    --root "${output_root}" --model mistral-7b-instruct-v0.2 --budget 1024
fi
