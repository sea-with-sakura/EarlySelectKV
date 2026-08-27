# Fixed-Trajectory Long-Generation Fidelity

This experiment reproduces the fixed-trajectory long-generation diagnostic in
the paper.

Default setting:

- model: `Mistral-7B-Instruct-v0.2`
- task: LongBench `multi_news`
- KV budget: `1024`
- samples: `20`
- common trajectory: the shared greedy trajectory produced by the
  `anchor_quality_stage2` diagnostic path after the same Stage2 prompt-cache
  compression. RocketKV and ES-RocketKV routing candidates are evaluated
  losslessly on the same generated token, hidden state, real query, and KV
  cache. This trajectory is used only for mechanism diagnosis.
- methods compared in the same lossless probe:
  - `exact_topk` -> Exact-TopK oracle from the real-query attention distribution
  - `rocket_qout` -> RocketKV
  - `lookahead_qmid` -> ES-RocketKV
- decode buckets: `0-64`, `64-128`, `128-256`, `256-384`, `384-512`
- metrics:
  - retained attention mass under the real-query attention distribution
  - signed Top-R channel recall for the proxy query versus the real query
  - page overlap, reported as oracle-page recall after mapping token indices to
    HSA page ids
- uncertainty:
  - per-example paired ES-RocketKV-minus-RocketKV gaps with bootstrap 95% CIs
  - linear trend slopes versus decode position, bootstrapped over examples

Run on GPUs 0 and 1:

```bash
bash analysis/figure_w6_multistep_fidelity/run_w6_multistep_fidelity.sh
```

Useful overrides:

```bash
SAMPLE_COUNT=50 SAMPLES_PER_SHARD=10 DEVICES=0,1 \
  bash analysis/figure_w6_multistep_fidelity/run_w6_multistep_fidelity.sh
```

Outputs:

```text
analysis/figure_w6_multistep_fidelity/out/plots/
  w6_multistep_fidelity_summary.csv
  w6_multistep_fidelity_sample_bucket_summary.csv
  w6_multistep_fidelity_paired_gaps.csv
  w6_multistep_fidelity_gap_summary.csv
  w6_multistep_fidelity_trends.csv
  w6_multistep_fidelity.png
  w6_multistep_fidelity.pdf
  long_generation_trajectory.pdf
  w6_rebuttal_draft.md
```
