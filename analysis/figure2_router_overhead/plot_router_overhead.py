#!/usr/bin/env python3
"""Plot theoretical routing compute overhead for Figure 2.

The default setting is a single Llama-3.1-8B attention layer with an 8K KV
cache. The accounting is in MAC-equivalent units and only models routing
compute relative to the selected sparse attention compute.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

@dataclass(frozen=True)
class ModelConfig:
    seq_len: int = 8192
    attention_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    quest_page_size: int = 16
    sparq_rank: int = 16
    loki_rank: int = 16


@dataclass(frozen=True)
class MethodPoint:
    method: str
    budget: int
    compression_ratio: int
    route_compute: float
    sparse_compute: float
    compute_share: float
    details: str


def sparse_attention_compute(cfg: ModelConfig, budget: int) -> float:
    return 2.0 * cfg.attention_heads * float(budget) * cfg.head_dim


def compute_share(route_compute: float, sparse_compute: float) -> float:
    return route_compute / (route_compute + sparse_compute)


def quest_point(cfg: ModelConfig, budget: int) -> MethodPoint:
    route_compute = cfg.attention_heads * (cfg.seq_len / cfg.quest_page_size) * cfg.head_dim
    sparse_compute = sparse_attention_compute(cfg, budget)
    return MethodPoint(
        method="Quest",
        budget=budget,
        compression_ratio=int(round(cfg.seq_len / budget)),
        route_compute=route_compute,
        sparse_compute=sparse_compute,
        compute_share=compute_share(route_compute, sparse_compute),
        details=f"P={cfg.quest_page_size}",
    )


def sparq_point(cfg: ModelConfig, budget: int) -> MethodPoint:
    route_compute = cfg.attention_heads * cfg.seq_len * cfg.sparq_rank
    sparse_compute = sparse_attention_compute(cfg, budget)
    return MethodPoint(
        method="SparQ",
        budget=budget,
        compression_ratio=int(round(cfg.seq_len / budget)),
        route_compute=route_compute,
        sparse_compute=sparse_compute,
        compute_share=compute_share(route_compute, sparse_compute),
        details=f"r={cfg.sparq_rank}",
    )


def loki_point(cfg: ModelConfig, budget: int, *, projection: str) -> MethodPoint:
    if projection == "paper":
        projection_compute_per_head = 2.0 * cfg.head_dim * cfg.head_dim
    elif projection == "rank":
        projection_compute_per_head = 2.0 * cfg.head_dim * cfg.loki_rank
    elif projection == "none":
        projection_compute_per_head = 0.0
    else:
        raise ValueError(f"unsupported Loki projection mode: {projection}")

    route_compute = cfg.attention_heads * (
        cfg.seq_len * cfg.loki_rank + projection_compute_per_head
    )
    sparse_compute = sparse_attention_compute(cfg, budget)
    return MethodPoint(
        method="Loki",
        budget=budget,
        compression_ratio=int(round(cfg.seq_len / budget)),
        route_compute=route_compute,
        sparse_compute=sparse_compute,
        compute_share=compute_share(route_compute, sparse_compute),
        details=f"r={cfg.loki_rank}, projection={projection}",
    )


def loki_cached_point(cfg: ModelConfig, budget: int) -> MethodPoint:
    """Loki route compute with cached projected keys.

    This is the comparable theoretical model for Loki: the rank-r projected K
    cache is maintained alongside the KV cache, so decode routing only projects
    the current query/current key and scores cached rank-r keys.
    """

    query_projection = cfg.attention_heads * cfg.head_dim * cfg.loki_rank
    current_key_projection = cfg.kv_heads * cfg.head_dim * cfg.loki_rank
    approximate_scores = cfg.attention_heads * cfg.seq_len * cfg.loki_rank
    route_compute = query_projection + current_key_projection + approximate_scores
    sparse_compute = sparse_attention_compute(cfg, budget)
    return MethodPoint(
        method="Loki",
        budget=budget,
        compression_ratio=int(round(cfg.seq_len / budget)),
        route_compute=route_compute,
        sparse_compute=sparse_compute,
        compute_share=compute_share(route_compute, sparse_compute),
        details=(
            f"r={cfg.loki_rank}, projection=cached-rank, "
            f"q_proj={query_projection:.0f}, k_new_proj={current_key_projection:.0f}"
        ),
    )


def loki_naive_pipeline_point(cfg: ModelConfig, budget: int) -> MethodPoint:
    """Loki route compute for the current naive PyTorch implementation.

    pipeline/loki/runtime.py currently projects the full K cache at every decode
    step before scoring in rank-r space. This is useful as an implementation
    diagnostic, but it is not the fair paper-level Loki cost model.
    """

    query_projection = cfg.attention_heads * cfg.head_dim * cfg.loki_rank
    key_projection = cfg.kv_heads * cfg.seq_len * cfg.head_dim * cfg.loki_rank
    approximate_scores = cfg.attention_heads * cfg.seq_len * cfg.loki_rank
    route_compute = query_projection + key_projection + approximate_scores
    sparse_compute = sparse_attention_compute(cfg, budget)
    return MethodPoint(
        method="Loki",
        budget=budget,
        compression_ratio=int(round(cfg.seq_len / budget)),
        route_compute=route_compute,
        sparse_compute=sparse_compute,
        compute_share=compute_share(route_compute, sparse_compute),
        details=(
            f"r={cfg.loki_rank}, projection=naive-pipeline-full-k, "
            f"q_proj={query_projection:.0f}, k_proj={key_projection:.0f}"
        ),
    )


def hsa_params(cfg: ModelConfig, budget: int) -> tuple[float, float, float, int, int]:
    compression_ratio = max(1.0, float(cfg.seq_len) / float(budget))
    rho = min(0.2 + 0.06 * math.log2(compression_ratio), 0.8)
    token_capacity_budget = float(cfg.seq_len) / (compression_ratio**rho)
    stage2_ratio = max(1.0, token_capacity_budget / float(budget))
    chunk_size = min(math.floor(stage2_ratio), math.ceil(math.sqrt(stage2_ratio)))
    chunk_size = max(1, int(chunk_size))
    channel_budget = min(
        cfg.head_dim,
        max(1, round(cfg.head_dim * chunk_size / stage2_ratio)),
    )
    return rho, token_capacity_budget, stage2_ratio, chunk_size, int(channel_budget)


def hsa_point(cfg: ModelConfig, budget: int) -> MethodPoint:
    rho, token_capacity_budget, stage2_ratio, chunk_size, channel_budget = hsa_params(cfg, budget)
    route_compute = cfg.attention_heads * (cfg.seq_len / chunk_size) * channel_budget
    sparse_compute = sparse_attention_compute(cfg, budget)
    details = (
        f"rho={rho:.4f}, capacity={token_capacity_budget:.2f}, "
        f"c2={stage2_ratio:.4f}, p={chunk_size}, r_hsa={channel_budget}"
    )
    return MethodPoint(
        method="HSA",
        budget=budget,
        compression_ratio=int(round(cfg.seq_len / budget)),
        route_compute=route_compute,
        sparse_compute=sparse_compute,
        compute_share=compute_share(route_compute, sparse_compute),
        details=details,
    )


def build_rows(
    cfg: ModelConfig,
    budgets: list[int],
    *,
    cost_model: str,
    loki_projection: str,
) -> list[MethodPoint]:
    rows: list[MethodPoint] = []
    if cost_model == "paper":
        loki_builder = lambda budget: loki_point(cfg, budget, projection=loki_projection)
    elif cost_model == "cached":
        loki_builder = lambda budget: loki_cached_point(cfg, budget)
    elif cost_model == "naive-pipeline":
        loki_builder = lambda budget: loki_naive_pipeline_point(cfg, budget)
    else:
        raise ValueError(f"unsupported cost model: {cost_model}")
    for budget in budgets:
        rows.extend(
            [
                quest_point(cfg, budget),
                sparq_point(cfg, budget),
                loki_builder(budget),
                hsa_point(cfg, budget),
            ]
        )
    return rows


def write_csv(rows: list[MethodPoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "compression_ratio",
                "budget",
                "route_compute",
                "sparse_compute",
                "route_over_sparse_compute",
                "compute_share",
                "compute_share_percent",
                "details",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row.method,
                    "compression_ratio": row.compression_ratio,
                    "budget": row.budget,
                    "route_compute": row.route_compute,
                    "sparse_compute": row.sparse_compute,
                    "route_over_sparse_compute": row.route_compute / row.sparse_compute,
                    "compute_share": row.compute_share,
                    "compute_share_percent": 100.0 * row.compute_share,
                    "details": row.details,
                }
            )


def compression_tick_labels(cfg: ModelConfig, compression_ratios: list[int]) -> list[str]:
    return [f"{ratio}x\n(B={cfg.seq_len // ratio})" for ratio in compression_ratios]


def plot_combined(rows: list[MethodPoint], compression_ratios: list[int], out_base: Path, cfg: ModelConfig) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 17,
            "xtick.labelsize": 12,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
        }
    )

    methods = ["Quest", "SparQ", "Loki", "HSA"]
    colors = {
        "Quest": "#2A6FBB",
        "SparQ": "#D37A22",
        "Loki": "#6B4FB3",
        "HSA": "#238B45",
    }
    markers = {
        "Quest": "o",
        "SparQ": "s",
        "Loki": "^",
        "HSA": "D",
    }

    by_method: dict[str, list[MethodPoint]] = {method: [] for method in methods}
    for row in rows:
        by_method[row.method].append(row)
    for method in methods:
        by_method[method].sort(key=lambda item: item.compression_ratio)

    fig, ax = plt.subplots(figsize=(7.4, 5.0))

    for method in methods:
        xs = [row.compression_ratio for row in by_method[method]]
        ys_compute = [100.0 * row.compute_share for row in by_method[method]]
        ax.plot(
            xs,
            ys_compute,
            label=method,
            color=colors[method],
            marker=markers[method],
            linewidth=2.2,
            markersize=5.5,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(compression_ratios)
    ax.set_xticklabels(compression_tick_labels(cfg, compression_ratios))
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_xlabel("")
    ax.set_ylabel("Routing compute share (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Llama-3.1-8B single-layer decode, S={cfg.seq_len}")
    ax.legend(loc="best", frameon=False, ncol=2)
    fig.tight_layout()

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--attention-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--quest-page-size", type=int, default=16)
    parser.add_argument("--sparq-rank", type=int, default=16)
    parser.add_argument("--loki-rank", type=int, default=16)
    parser.add_argument("--compression-ratios", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument(
        "--loki-projection",
        choices=["paper", "rank", "none"],
        default="paper",
        help="Projection-cost model for Loki routing compute when --cost-model=paper.",
    )
    parser.add_argument(
        "--cost-model",
        choices=["paper", "cached", "naive-pipeline"],
        default="paper",
        help=(
            "paper keeps the original idealized paper formulas; cached uses a fair "
            "projected-K-cache model for Loki; naive-pipeline includes the current "
            "repo's per-step full-K projection for Loki."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "out",
    )
    return parser.parse_args()


def budgets_from_compression_ratios(seq_len: int, compression_ratios: list[int]) -> list[int]:
    budgets: list[int] = []
    for ratio in compression_ratios:
        if ratio <= 0:
            raise ValueError(f"compression ratio must be positive, got {ratio}")
        if seq_len % ratio != 0:
            raise ValueError(f"seq_len={seq_len} must be divisible by compression ratio {ratio}")
        budgets.append(seq_len // ratio)
    return budgets


def main() -> None:
    args = parse_args()
    cfg = ModelConfig(
        seq_len=args.seq_len,
        attention_heads=args.attention_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        quest_page_size=args.quest_page_size,
        sparq_rank=args.sparq_rank,
        loki_rank=args.loki_rank,
    )
    compression_ratios = sorted(set(int(ratio) for ratio in args.compression_ratios))
    budgets = budgets_from_compression_ratios(cfg.seq_len, compression_ratios)
    rows = build_rows(
        cfg,
        budgets,
        cost_model=args.cost_model,
        loki_projection=args.loki_projection,
    )

    out_name = "routing_compute_vs_compression_llama31_8b_8k"
    if args.cost_model == "cached":
        out_name += "_corrected"
    elif args.cost_model == "naive-pipeline":
        out_name += "_naive_pipeline"
    out_base = args.out_dir / out_name
    write_csv(rows, out_base.with_suffix(".csv"))
    plot_combined(rows, compression_ratios, out_base, cfg)

    print(f"Wrote {out_base.with_suffix('.png')}")
    print(f"Wrote {out_base.with_suffix('.pdf')}")
    print(f"Wrote {out_base.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
