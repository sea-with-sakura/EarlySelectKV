#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-python}"
model="${MODEL:-mistral-7b-instruct-v0.2}"
pipeline_config="${PIPELINE_CONFIG:-config/pipeline_config/longbench/${model}.json}"
model_path="${MODEL_PATH:-modelzoo/Mistral-7B-Instruct-v0.2}"
artifact_dir="${ARTIFACT_DIR:-dataset/wq_svd/Mistral-7B-Instruct-v0.2}"
output_root="${OUTPUT_ROOT:-result/svd_rank_longbench}"
budget="${TOKEN_BUDGET:-512}"
ranks="${RANKS:-64 128 256 512}"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  "${python_bin}" scripts/probe/export_wq_svd_artifacts.py \
    --model-dir "${model_path}" \
    --output-dir "${artifact_dir}" \
    --ranks "${ranks}"
fi

methods="rocketkv"
for rank in ${ranks}; do
  methods+=" earlyselectkv_mid_svd_r${rank}"
done

for method in ${methods}; do
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    DRY_RUN=1 MODEL="${model}" PIPELINE_CONFIG="${pipeline_config}" \
      bash scripts/common/longbench.sh "${method}" "${output_root}" "${budget}"
  else
    MODEL="${model}" PIPELINE_CONFIG="${pipeline_config}" \
      bash scripts/common/longbench.sh "${method}" "${output_root}" "${budget}"
  fi
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  "${python_bin}" analysis/figure_svd_rank/plot_svd_delta.py \
    --root "${output_root}" --model "${model}" --budget "${budget}"
fi
