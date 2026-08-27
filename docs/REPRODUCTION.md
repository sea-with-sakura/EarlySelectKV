# Reproduction Guide

This guide covers setup and evaluation for the non-TensorRT experiments in
the EarlySelectKV paper. For an experiment-by-experiment coverage matrix and
known paper/code consistency decisions, see
[`OPEN_SOURCE_AUDIT.md`](../OPEN_SOURCE_AUDIT.md). The paper's TensorRT system
implementation and performance-reproduction workflow are available in the
[`EarlySelectKV` branch of our TensorRT-LLM fork](https://github.com/sea-with-sakura/TensorRT-LLM/tree/EarlySelectKV).

## Included Methods

- `baseline`: dense LitGPT decoding.
- `rocketkv`: RocketKV-compatible two-stage KV selection.
- `rocketkv_topk`: exact top-k Stage-2 routing variant.
- `topk`: standalone Exact-TopK over the full cache.
- `quest`, `earlyselect_quest`: Quest and ES-Quest page routing.
- `loki`, `earlyselect_loki`: Loki and ES-Loki low-rank routing; offline PCA is required.
- `earlyselectkv`: computes the next-layer routing query ahead of use.
- `earlyselectkv_in`: uses the current-layer input as the lookahead source.
- `earlyselectkv_in_mid`: uses the model-configured input/midpoint source.
- `earlyselectkv_topk`: exact top-k routing variant.
- `earlyselectkv_in_mid_local`, `earlyselectkv_local`, and
  `earlyselectkv_hsa_local`: local-token routing variants.

Paper diagnostics for proxy fidelity, fixed-trajectory generation, ES-Mix
calibration, and low-rank proxy projection are under `analysis/`.

## Setup

Install a CUDA-compatible PyTorch build and FlashAttention first, then install
the remaining dependencies. The requirements pin the public LitGPT fork and
revision used by the paper experiments:

```bash
pip install -r requirements.txt
```

For private Hugging Face models, provide credentials through the standard
Hugging Face mechanism:

```bash
export HF_TOKEN=<your-token>
```

No access tokens or model weights are stored in this repository.

The scripts expect LitGPT-format checkpoints under `modelzoo/` by default.
Each checkpoint directory must contain `lit_model.pth`, `model_config.yaml`,
and tokenizer files. The pinned LitGPT CLI can download and convert supported
Hugging Face models:

```bash
litgpt download <organization>/<model> --checkpoint_dir modelzoo
```

Set `model_name` in the relevant `config/pipeline_config/**/*.json` if the
checkpoint is stored elsewhere.

## Data Preparation

Prepare LongBench and MATH-500 with:

```bash
python scripts/dataset_prep/download_longbench.py
python scripts/dataset_prep/download_math_reasoning.py
```

Prepare RULER assets with:

```bash
bash scripts/setup_external_assets.sh ruler
```

For the ES-Mix cross-domain analysis, place the LongAlpaca-12k source JSON at
`dataset/alpaca12k/LongAlpaca-12k.json`, then convert it to the repository's
LongBench-compatible calibration format:

```bash
bash scripts/setup_external_assets.sh longalpaca
```

## Reproduce the Main Quality Matrix

The main launcher covers LongBench, RULER, Needle-in-a-Haystack/Passkey, and
MATH-500:

```bash
bash scripts/reproduce_paper_quality.sh
```

Inspect the complete command matrix without loading models:

```bash
DRY_RUN=1 bash scripts/reproduce_paper_quality.sh
```

Run the repository's model-independent regression tests with:

```bash
pytest -q
```

Use a clean environment so that an unrelated editable LitGPT checkout or an
incompatible preinstalled PyTorch/torchvision pair does not shadow the pinned
dependencies in `requirements.txt`.

## Run Individual Benchmarks

Run a LongBench task group directly:

```bash
MODEL=mistral-7b-instruct-v0.2 \
bash scripts/common/longbench.sh rocketkv result_public 512 "qasper"

MODEL=mistral-7b-instruct-v0.2 \
bash scripts/common/longbench.sh earlyselectkv result_public 512 "qasper"
```

Or use a model-specific wrapper:

```bash
bash scripts/longbench/mistral-7b-instruct-v0.2.sh rocketkv result_public 512 "qasper"
bash scripts/longbench/mistral-7b-instruct-v0.2.sh earlyselectkv result_public 512 "qasper"
```

For a tmux sweep over LongBench datasets:

```bash
METHODS="rocketkv earlyselectkv" BUDGETS="512 1024" \
CUDA_DEVICES="0 1 2 3" bash zbash/longbench.sh
```

Run MATH-500 with:

```bash
bash scripts/math_reasoning/llama3.1-8b-instruct.sh baseline result_math
bash scripts/math_reasoning/llama3.1-8b-instruct.sh rocketkv result_math 1024
bash scripts/math_reasoning/llama3.1-8b-instruct.sh earlyselectkv result_math 1024 math500
bash scripts/math_reasoning/qwen2.5-7b-instruct.sh earlyselectkv result_math 1024 math500
```

Run NIAH/Passkey and RULER with:

```bash
bash scripts/paulgraham_passkey/llama3.1-8b-instruct.sh earlyselectkv result_public 512
bash scripts/ruler/llama3.1-8b-instruct.sh earlyselectkv result_public 512
```

Qwen2.5-32B LongBench and RULER configurations are also included:

```bash
bash scripts/longbench/qwen2.5-32b-instruct.sh earlyselectkv result_public 512 "qasper"
bash scripts/ruler/qwen2.5-32b-instruct.sh earlyselectkv result_public 512
```

## Paper Analyses

The theoretical routing-overhead figure requires neither model weights nor
evaluation data:

```bash
python analysis/figure2_router_overhead/plot_router_overhead.py
```

The remaining paper analyses have focused launchers:

```bash
bash analysis/additional_selectors/run_additional_selectors.sh
bash analysis/figure_stage2_anchor_quality/run_anchor_quality_stage2.sh
bash analysis/run_esmix_domain_stability.sh
bash analysis/figure_w6_multistep_fidelity/run_w6_multistep_fidelity.sh
bash analysis/figure_svd_rank/run_svd_rank_longbench.sh
bash analysis/es_infllmv2/run_paper_compatibility.sh
```

The InfLLM-V2 launcher reproduces only the four-dataset task-quality study. It
uses a precision-oriented PyTorch sparse-mask emulation and does not execute or
benchmark the OpenBMB CUDA kernel. See
[`analysis/es_infllmv2/README.md`](../analysis/es_infllmv2/README.md) for its
checkpoint requirements and exact interpretation.

## Configuration Layout

Shared sparse-method defaults are in:

```text
config/pipeline_config/_common/sparse_methods.json
```

Model and benchmark settings are in:

```text
config/pipeline_config/longbench/
config/pipeline_config/math_reasoning/
config/pipeline_config/paulgraham_passkey/
config/pipeline_config/ruler/
config/eval_config/
```

## Reproduction Boundary

This repository reproduces the non-TensorRT quality experiments and PyTorch
analyses. TensorRT-only throughput, batching, GPU-memory, and decode-breakdown
measurements are maintained in the separate
[`EarlySelectKV` TensorRT-LLM branch](https://github.com/sea-with-sakura/TensorRT-LLM/tree/EarlySelectKV).
