# Figure Assets

This directory stores the figures retained from the final paper source so that
the public repository does not depend on the removable `paper/` directory.
The PDF files are the publication-quality originals.

The repository landing page embeds two PNG derivatives for reliable rendering
on GitHub:

- `earlyselectkv-overview.png`, rendered from `earlyselectkv-overview.pdf`;
- `longbench-ruler-results.png`, rendered from `longbench_ruler.pdf`.

## Contents

- Method: `earlyselectkv-overview.pdf`,
  `earlyselectkv-method-details.pdf`, and `routing_overhead.pdf`.
- Quality: `longbench_ruler.pdf`, `additional_selector_results.pdf`, and the
  three Qwen NIAH heatmaps (`rocket-256-qwen.pdf`, `ES-in-256-qwen.pdf`, and
  `ES-mid-256-qwen.pdf`).
- Diagnostics: `proxy_fidelity.pdf`,
  `appendix_anchor_cosine_page_recall.pdf`,
  `long_generation_trajectory.pdf`, and `svd_delta.pdf`.
- TensorRT system results: `throughput.pdf`, `multi_batch.pdf`,
  `decode_breakdown.pdf`, and `runtime_scope.pdf`.

TensorRT figures are retained here as paper assets. Their runtime implementation
and reproduction workflow are released in the
[`EarlySelectKV` branch of our TensorRT-LLM fork](https://github.com/sea-with-sakura/TensorRT-LLM/tree/EarlySelectKV).
