# InfLLM-V2 compatibility experiment

This directory reproduces the paper's post-hoc EarlySelectKV integration with
`openbmb/InfLLM-V2-Long-Sparse-Base`. It is an accuracy experiment, not a
runtime benchmark: both Real-Q and EarlySelectKV construct InfLLM-V2-style
block masks in PyTorch and consume them with dense attention logits plus the
same sparse token mask. The released sparse checkpoint weights are used, while
the checkpoint's native CUDA sparse-attention class is disabled during model
loading.

In the paper, "native routing" distinguishes the normal target-layer real
query from EarlySelectKV's proxy query. It does not mean that this artifact
executes or benchmarks the upstream CUDA routing kernel. Only task quality is
reported for this compatibility study.

The three paper columns differ only in the Stage-1 routing query:

- `Real-Q`: the current layer's true query (`oracle_sparse` in the code);
- `Exact-Wq`: the previous layer's mid hidden state projected with the next
  layer's exact final-checkpoint `Wq`;
- `Rank-256`: the same mid hidden state projected with a rank-256 SVD of the
  final-checkpoint `Wq`.

The final attention computation always uses the target layer's real query.

## Prerequisites

Prepare the four LongBench datasets with the repository data script, and place
the complete checkpoint under the default path:

```text
modelzoo/InfLLM-V2-Long-Sparse-Base/
```

The checkpoint must include its `config.json`, tokenizer files, remote-code
Python sources, weight index, and all seven safetensors shards. The runner uses
`trust_remote_code=True`; review and trust the checkpoint's Python files before
running them.

The experiment does not import or require the upstream InfLLM-V2 CUDA
repository. The CUDA kernels are outside this precision-only experiment and
are intentionally not vendored here.

## Reproduce the paper table

```bash
bash analysis/es_infllmv2/run_paper_compatibility.sh
```

Useful overrides include:

```bash
MODEL_DIR=/path/to/InfLLM-V2-Long-Sparse-Base \
DEVICE=cuda:0 DTYPE=bfloat16 \
bash analysis/es_infllmv2/run_paper_compatibility.sh
```

For a one-task smoke run or command inspection:

```bash
DATASETS=gov_report SAMPLE_COUNT=1 MAX_NEW_TOKENS=4 \
bash analysis/es_infllmv2/run_paper_compatibility.sh

DRY_RUN=1 bash analysis/es_infllmv2/run_paper_compatibility.sh
```

The rank-256 factors are generated from the final sparse checkpoint on first
use and cached below the output root. Model weights, generated predictions,
and the SVD cache are not source artifacts and are ignored by Git.

## Reference values

The recovered run used the first 16 examples per task, middle truncation to
12,288 input tokens, greedy decoding, and the checkpoint sparse configuration
(`dense_len=8192`, `block_size=64`, `window_size=2048`, `topk=64`). It produced:

| Task | Sparse | Real-Q | Exact-Wq | Rank-256 |
|---|---:|---:|---:|---:|
| GovReport | 13/16 | 25.21 | 25.91 | 24.68 |
| QMSum | 15/16 | 24.05 | 24.47 | 23.80 |
| MuSiQue | 16/16 | 6.96 | 6.89 | 6.78 |
| HotpotQA | 15/16 | 15.11 | 15.11 | 14.96 |

Across the four task scores, Exact-Wq minus Real-Q is `+0.26` points and
Rank-256 minus Real-Q is `-0.28` points, matching the paper table. The
generated summary also records route preparation statistics; the recovered
run reported zero missing EarlySelectKV routes.
