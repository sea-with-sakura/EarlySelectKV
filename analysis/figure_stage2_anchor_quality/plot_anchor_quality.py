from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns


METHOD_ORDER = ["exact_topk", "rocket_qout", "lookahead_qmid", "lookahead_qin"]
ANCHOR_METHODS = ["rocket_qout", "lookahead_qmid", "lookahead_qin"]
PAPER_QUERY_METHODS = ["lookahead_qmid", "lookahead_qin"]
PAPER_ROUTING_METHODS = ["rocket_qout", "lookahead_qmid", "lookahead_qin"]
METHOD_LABELS = {
    "exact_topk": "Exact-TopK",
    "rocket_qout": "Qout",
    "lookahead_qmid": "Qmid",
    "lookahead_qin": "Qin",
}
METHOD_COLORS = {
    "rocket_qout": "#2A6FBB",
    "lookahead_qmid": "#238B45",
    "lookahead_qin": "#D37A22",
}
METHOD_STYLES = {
    "rocket_qout": "-",
    "lookahead_qmid": "-",
    "lookahead_qin": "-",
}
METHOD_WIDTHS = {
    "rocket_qout": 1.8,
    "lookahead_qmid": 1.5,
    "lookahead_qin": 1.5,
}


def _register_times_new_roman() -> None:
    for font_path in (
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf",
    ):
        path = Path(font_path)
        if path.exists():
            font_manager.fontManager.addfont(str(path))


def _save(fig: plt.Figure, plots_dir: Path, name: str) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(plots_dir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _style_axis(ax: plt.Axes, *, title: str, ylabel: str, xlabel: str = "") -> None:
    ax.set_title(title, fontweight="normal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.7, alpha=0.35)


def _plot_metric_lines(
    ax: plt.Axes,
    layer_df: pd.DataFrame,
    *,
    methods: list[str],
    metric: str,
    title: str,
    ylabel: str,
    ada_split_layer: int | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    for method in methods:
        sub = layer_df[layer_df["method"] == method].sort_values("layer_idx")
        if sub.empty or metric not in sub.columns:
            continue
        ax.plot(
            sub["layer_idx"],
            sub[metric],
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method),
            linestyle=METHOD_STYLES.get(method, "-"),
            linewidth=METHOD_WIDTHS.get(method, 2.2),
        )
    if ada_split_layer is not None:
        ax.axvline(ada_split_layer, color="#9ca3af", linewidth=1.0, linestyle=":")
    _style_axis(ax, title=title, ylabel=ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _plot_single_metric_panel(
    ax: plt.Axes,
    layer_df: pd.DataFrame,
    *,
    methods: list[str],
    metric: str,
    title: str,
    ylabel: str,
    ada_split_layer: int,
) -> None:
    for method in methods:
        sub = layer_df[layer_df["method"] == method].sort_values("layer_idx")
        if sub.empty or metric not in sub.columns:
            continue
        ax.plot(
            sub["layer_idx"],
            sub[metric],
            color=METHOD_COLORS.get(method),
            linewidth=METHOD_WIDTHS.get(method, 2.0),
            label=METHOD_LABELS.get(method, method),
        )
    ax.axvline(ada_split_layer, color="#9ca3af", linewidth=1.0, linestyle=":")
    ax.set_title(title, fontfamily="Times New Roman", fontweight="normal", color="#111111")
    ax.set_xlabel("Layer index")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.margins(x=0.02)


def _paper_legend(axes: list[plt.Axes], *, include_qout: bool = True) -> None:
    methods = ["lookahead_qmid", "lookahead_qin"]
    if include_qout:
        methods.append("rocket_qout")
    handles = [
        Line2D([0], [0], color=METHOD_COLORS[method], linewidth=2.2, label=METHOD_LABELS[method])
        for method in methods
    ]
    for ax in axes:
        ax.legend(
            handles=handles,
            loc="lower right",
            ncol=1,
            frameon=False,
            bbox_to_anchor=(0.98, 0.04),
            handlelength=1.6,
            labelspacing=0.25,
            prop={"weight": "normal", "size": 9},
        )


def plot_paper_main(layer_df: pd.DataFrame, plots_dir: Path, *, ada_split_layer: int = 20) -> None:
    _register_times_new_roman()
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.35))
    ax_query, ax_route = axes

    _plot_single_metric_panel(
        ax_query,
        layer_df,
        methods=PAPER_QUERY_METHODS,
        metric="signed_channel_recall",
        title="Query fidelity by layer",
        ylabel="Signed top-r recall",
        ada_split_layer=ada_split_layer,
    )
    ax_query.axhline(
        1.0,
        color=METHOD_COLORS["rocket_qout"],
        linewidth=METHOD_WIDTHS["rocket_qout"],
        label=METHOD_LABELS["rocket_qout"],
    )

    _plot_single_metric_panel(
        ax_route,
        layer_df,
        methods=PAPER_ROUTING_METHODS,
        metric="mass",
        title="Routing fidelity by layer",
        ylabel="Attention mass",
        ada_split_layer=ada_split_layer,
    )

    _paper_legend(list(axes), include_qout=True)
    fig.tight_layout(rect=(0, 0, 1, 1), w_pad=1.6)
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "proxy_fidelity.pdf", bbox_inches="tight")
    _save(fig, plots_dir, "paper_anchor_quality_main")


def plot_appendix_metrics(layer_df: pd.DataFrame, plots_dir: Path, *, ada_split_layer: int = 20) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.35))
    ax_cos, ax_page = axes
    _plot_single_metric_panel(
        ax_cos,
        layer_df,
        methods=PAPER_QUERY_METHODS,
        metric="cosine_similarity",
        title="Query cosine similarity",
        ylabel="Cosine similarity",
        ada_split_layer=ada_split_layer,
    )
    ax_cos.axhline(
        1.0,
        color=METHOD_COLORS["rocket_qout"],
        linewidth=METHOD_WIDTHS["rocket_qout"],
        label=METHOD_LABELS["rocket_qout"],
    )
    _plot_single_metric_panel(
        ax_page,
        layer_df,
        methods=PAPER_ROUTING_METHODS,
        metric="page_recall",
        title="Routing page recall",
        ylabel="Page recall vs Exact-TopK",
        ada_split_layer=ada_split_layer,
    )
    _paper_legend(list(axes), include_qout=True)
    fig.tight_layout(rect=(0, 0, 1, 1), w_pad=1.6)
    _save(fig, plots_dir, "appendix_anchor_cosine_page_recall")


def plot_query_quality(layer_df: pd.DataFrame, plots_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    metrics = [
        ("cosine_similarity", "Cosine similarity"),
        ("l2_relative_error", "L2 relative error"),
        ("topr_channel_recall", "Top-r channel recall"),
        ("topr_sign_agreement", "Top-r sign agreement"),
        ("signed_channel_recall", "Signed channel recall"),
    ]
    sub = layer_df[layer_df["method"].isin(ANCHOR_METHODS)].copy()
    sub["method"] = pd.Categorical(sub["method"], categories=ANCHOR_METHODS, ordered=True)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 16), sharex=True)
    for ax, (metric, title) in zip(axes, metrics, strict=False):
        sns.lineplot(data=sub, x="layer_idx", y=metric, hue="method", linewidth=2.0, ax=ax)
        ax.set_title(title, fontweight="normal")
        ax.set_xlabel("")
        ax.set_ylabel(metric)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    axes[-1].set_xlabel("Target layer")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, plots_dir, "anchor_query_quality_by_layer")


def plot_selection_fidelity(layer_df: pd.DataFrame, plots_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    metrics = [
        ("topk_recall", "Recall@k vs Exact-TopK"),
        ("topk_jaccard", "Jaccard vs Exact-TopK"),
        ("page_recall", "Page recall vs Exact-TopK"),
        ("mass", "Attention mass retained"),
        ("mass_gap_to_topk", "Mass gap to Exact-TopK"),
    ]
    sub = layer_df[layer_df["method"].isin(METHOD_ORDER)].copy()
    sub["method"] = pd.Categorical(sub["method"], categories=METHOD_ORDER, ordered=True)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 16), sharex=True)
    for ax, (metric, title) in zip(axes, metrics, strict=False):
        sns.lineplot(data=sub, x="layer_idx", y=metric, hue="method", linewidth=2.0, ax=ax)
        if metric == "mass_gap_to_topk":
            ax.axhline(0.0, color="#9ca3af", linestyle="--", linewidth=1.2)
        ax.set_title(title, fontweight="normal")
        ax.set_xlabel("")
        ax.set_ylabel(metric)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    axes[-1].set_xlabel("Target layer")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, plots_dir, "anchor_selection_fidelity_by_layer")


def plot_selection_fidelity_compact(layer_df: pd.DataFrame, plots_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    metrics = [
        ("page_recall", "Page recall vs Exact-TopK"),
        ("mass", "Attention mass retained"),
    ]
    sub = layer_df[layer_df["method"].isin(METHOD_ORDER)].copy()
    sub["method"] = pd.Categorical(sub["method"], categories=METHOD_ORDER, ordered=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharex=True)
    for ax, (metric, title) in zip(axes, metrics, strict=False):
        sns.lineplot(data=sub, x="layer_idx", y=metric, hue="method", linewidth=2.2, ax=ax)
        ax.set_title(title, fontweight="normal")
        ax.set_xlabel("Target layer")
        ax.set_ylabel(metric)
        ax.set_ylim(0.0, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    _save(fig, plots_dir, "anchor_selection_page_recall_mass_by_layer")


def plot_head_delta(head_df: pd.DataFrame, plots_dir: Path) -> None:
    sns.set_theme(style="white", context="talk")
    for metric in ("cosine_similarity", "signed_channel_recall"):
        pivot = head_df.pivot_table(index=["layer_idx", "head_idx"], columns="method", values=metric)
        if "lookahead_qin" not in pivot or "lookahead_qmid" not in pivot:
            continue
        delta = (pivot["lookahead_qin"] - pivot["lookahead_qmid"]).reset_index(name="qin_minus_qmid")
        heat = delta.pivot(index="head_idx", columns="layer_idx", values="qin_minus_qmid")
        vmax = float(heat.abs().max().max())
        if not vmax or pd.isna(vmax):
            vmax = 1.0
        fig, ax = plt.subplots(figsize=(13, 8))
        sns.heatmap(
            heat,
            cmap="RdBu_r",
            center=0.0,
            vmin=-vmax,
            vmax=vmax,
            linewidths=0.1,
            linecolor="#f3f4f6",
            cbar_kws={"label": "positive = qin better"},
            ax=ax,
        )
        ax.set_title(f"Qin - Qmid {metric}", fontweight="normal")
        ax.set_xlabel("Target layer")
        ax.set_ylabel("Head")
        _save(fig, plots_dir, f"anchor_head_qin_minus_qmid_{metric}")


def plot_head_anchor_to_qout(head_df: pd.DataFrame, plots_dir: Path) -> None:
    sns.set_theme(style="white", context="talk")
    specs = [
        ("lookahead_qmid", "cosine_similarity", "Qmid vs Qout cosine", "anchor_head_qmid_to_qout_cosine_similarity"),
        ("lookahead_qin", "cosine_similarity", "Qin vs Qout cosine", "anchor_head_qin_to_qout_cosine_similarity"),
        ("lookahead_qmid", "signed_channel_recall", "Qmid vs Qout signed-channel recall", "anchor_head_qmid_to_qout_signed_channel_recall"),
        ("lookahead_qin", "signed_channel_recall", "Qin vs Qout signed-channel recall", "anchor_head_qin_to_qout_signed_channel_recall"),
    ]
    for method, metric, title, filename in specs:
        sub = head_df[head_df["method"] == method].copy()
        if sub.empty or metric not in sub.columns:
            continue
        heat = sub.pivot_table(index="head_idx", columns="layer_idx", values=metric, aggfunc="mean")
        fig, ax = plt.subplots(figsize=(13, 8))
        sns.heatmap(
            heat,
            cmap="Reds_r",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.0,
            cbar_kws={"label": metric, "ticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]},
            ax=ax,
        )
        ax.set_title(title, fontweight="normal")
        ax.set_xlabel("Target layer")
        ax.set_ylabel("Head")
        _save(fig, plots_dir, filename)


def write_compact_tables(summary_df: pd.DataFrame, layer_df: pd.DataFrame, out_dir: Path) -> None:
    keep = [
        "method",
        "count",
        "cosine_similarity",
        "l2_relative_error",
        "topr_channel_recall",
        "topr_sign_agreement",
        "signed_channel_recall",
        "topk_recall",
        "topk_jaccard",
        "page_recall",
        "mass",
        "topk_mass",
        "mass_gap_to_topk",
        "sink_page_ratio",
        "local_page_ratio",
        "local_window_ratio",
        "generated_token_ratio",
    ]
    summary_df[[c for c in keep if c in summary_df.columns]].to_csv(out_dir / "anchor_quality_compact_summary.csv", index=False)

    rows = []
    for method, part in layer_df[layer_df["method"].isin(["lookahead_qmid", "lookahead_qin"])].groupby("method"):
        for bucket, group in part.groupby(pd.cut(part["layer_idx"], bins=[0, 10, 20, 32], labels=["early", "middle", "late"]), observed=True):
            rows.append(
                {
                    "method": method,
                    "layer_bucket": bucket,
                    "cosine_similarity": group["cosine_similarity"].mean(),
                    "signed_channel_recall": group["signed_channel_recall"].mean(),
                    "topk_recall": group["topk_recall"].mean(),
                    "mass": group["mass"].mean(),
                    "mass_gap_to_topk": group["mass_gap_to_topk"].mean(),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "anchor_quality_layer_bucket_compact.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot anchor_quality_stage2 layer/head summaries.")
    parser.add_argument(
        "--probe-dir",
        default="analysis/figure_stage2_anchor_quality/out/longbench/anchor_quality_stage2/1024/mistral-7b-instruct-v0.2/qasper",
        help="Directory containing stage2_probe_*.csv files from anchor_quality_stage2.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Output directory for figures. Defaults to <probe-dir>/plots.",
    )
    parser.add_argument(
        "--ada-split-layer",
        type=int,
        default=20,
        help="Layer index shown as the Ada Qmid/Qin switch boundary in the paper figure.",
    )
    parser.add_argument(
        "--include-diagnostics",
        action="store_true",
        help="Also write the exploratory multi-panel and heatmap diagnostics.",
    )
    args = parser.parse_args()
    probe_dir = Path(args.probe_dir)
    plots_dir = args.plots_dir or (probe_dir / "plots")
    summary_df = pd.read_csv(probe_dir / "stage2_probe_summary.csv")
    layer_df = pd.read_csv(probe_dir / "stage2_probe_layer_detail_summary.csv")
    plot_paper_main(layer_df, plots_dir, ada_split_layer=args.ada_split_layer)
    plot_appendix_metrics(layer_df, plots_dir, ada_split_layer=args.ada_split_layer)
    if args.include_diagnostics:
        head_df = pd.read_csv(probe_dir / "stage2_probe_head_summary.csv")
        plot_query_quality(layer_df, plots_dir)
        plot_selection_fidelity(layer_df, plots_dir)
        plot_selection_fidelity_compact(layer_df, plots_dir)
        plot_head_delta(head_df, plots_dir)
        plot_head_anchor_to_qout(head_df, plots_dir)
    write_compact_tables(summary_df, layer_df, probe_dir)


if __name__ == "__main__":
    main()
