#!/usr/bin/env bash
set -euo pipefail

output_root="${OUTPUT_ROOT:-result/paper_quality}"
budgets_longbench="${LONGBENCH_BUDGETS:-128 256 512 1024 2048 4096}"
budgets_variants="${VARIANT_BUDGETS:-64 128 256 512 1024 2048 4096}"
budgets_ruler="${RULER_BUDGETS:-128 256 512 1024 2048}"
budgets_niah="${NIAH_BUDGETS:-64 128 256 512 1024 2048 4096}"

run() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    DRY_RUN=1 "$@"
  else
    "$@"
  fi
}

# Main LongBench curves and appendix averages.
for model in llama3-8b-instruct llama3.1-8b-instruct mistral-7b-instruct-v0.2 qwen2.5-7b-instruct; do
  run bash "scripts/longbench/${model}.sh" baseline "${output_root}"
  for method in topk rocketkv earlyselectkv; do
    for budget in ${budgets_longbench}; do
      run bash "scripts/longbench/${model}.sh" "${method}" "${output_root}" "${budget}"
    done
  done
  for method in rocketkv_topk earlyselectkv_topk; do
    for budget in ${budgets_longbench}; do
      run bash "scripts/longbench/${model}.sh" "${method}" "${output_root}" "${budget}"
    done
  done
done

# ES-In / ES-Mid / ES-Mix appendix sweep. Qwen uses ES-Mid as the documented
# conservative fallback because no stable mixed profile was found.
for model in llama3-8b-instruct llama3.1-8b-instruct mistral-7b-instruct-v0.2; do
  for method in rocketkv earlyselectkv_in earlyselectkv earlyselectkv_in_mid; do
    for budget in ${budgets_variants}; do
      run bash "scripts/longbench/${model}.sh" "${method}" "${output_root}" "${budget}"
    done
  done
done

# RULER rows in the paper use exactly 8K, 16K, and 32K contexts.
for model in mistral-7b-instruct-v0.2 llama3.1-8b-instruct qwen2.5-7b-instruct; do
  run bash "scripts/ruler/${model}.sh" baseline "${output_root}" "" "" "8000 16000 32000"
  for method in topk rocketkv earlyselectkv; do
    for budget in ${budgets_ruler}; do
      run bash "scripts/ruler/${model}.sh" "${method}" "${output_root}" "${budget}" "" "8000 16000 32000"
    done
  done
done

for model in mistral-7b-instruct-v0.2 llama3.1-8b-instruct; do
  for method in earlyselectkv_in earlyselectkv earlyselectkv_in_mid; do
    for budget in ${budgets_ruler}; do
      run bash "scripts/ruler/${model}.sh" "${method}" "${output_root}" "${budget}" "" "8000 16000 32000"
    done
  done
done

# NIAH/Passkey 10x10x3_7digits sweeps at the model-specific paper lengths.
for model in mistral-7b-instruct-v0.2 llama3-8b-instruct llama3.1-8b-instruct qwen2.5-7b-instruct; do
  methods="topk rocketkv earlyselectkv_in earlyselectkv"
  if [[ "${model}" != "qwen2.5-7b-instruct" ]]; then
    methods+=" earlyselectkv_in_mid"
  fi
  for method in ${methods}; do
    for budget in ${budgets_niah}; do
      run bash "scripts/paulgraham_passkey/${model}.sh" "${method}" "${output_root}" "${budget}"
    done
  done
done

# Independent long-generation evaluation on the full MATH-500 test split.
for model in llama3.1-8b-instruct qwen2.5-7b-instruct; do
  run bash "scripts/math_reasoning/${model}.sh" baseline "${output_root}" "" "math500"
  for method in rocketkv earlyselectkv; do
    run bash "scripts/math_reasoning/${model}.sh" "${method}" "${output_root}" 1024 "math500"
  done
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  python analysis/benchmark_figures/plot_longbench_ruler.py --root "${output_root}"
fi
