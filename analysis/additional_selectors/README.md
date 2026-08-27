# Quest and Loki Generalization

`run_additional_selectors.sh` reproduces the non-TensorRT LongBench rows for
Quest/Loki and their EarlySelect variants. Loki requires an offline PCA fit;
the launcher generates it from WikiText validation tokens before evaluation
and disables the non-reproducible identity fallback.

```bash
bash analysis/additional_selectors/run_additional_selectors.sh
```

The default matrix matches the paper: Qwen2.5-32B Quest at budget 256;
Qwen2.5-7B Quest/Loki at 128; and Mistral Quest/Loki at 128, 1024,
2048, and 4096. The 1024 runs generate the main task-level figure.
