# Low-Rank Proxy Projection

This directory reproduces the non-TensorRT rank ablation in the paper on
Mistral-7B-Instruct-v0.2, LongBench, and KV-access budget 512.

The launcher first exports offline SVD factors for ranks 64, 128, 256, and
512, then runs RocketKV and the four ES-RocketKV low-rank variants:

```bash
bash analysis/figure_svd_rank/run_svd_rank_longbench.sh
```

Use `DRY_RUN=1` to print the evaluation commands without running models. The
plotter reads the standard `longbench_result_summary.json` files and writes
`svd_delta.{png,pdf}`.
