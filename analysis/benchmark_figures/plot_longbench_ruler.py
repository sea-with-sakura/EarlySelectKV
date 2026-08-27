from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


LB_MODELS = [
    ("mistral-7b-instruct-v0.2", "Mistral-7B-Instruct-v0.2"),
    ("llama3-8b-instruct", "Llama-3-8B-Instruct"),
    ("llama3.1-8b-instruct", "Llama-3.1-8B-Instruct"),
    ("qwen2.5-7b-instruct", "Qwen2.5-7B-Instruct"),
]
LB_BUDGETS = [128, 256, 512, 1024, 2048, 4096]
RULER_MODELS = [
    ("mistral-7b-instruct-v0.2", "Mistral-7B-Instruct-v0.2"),
    ("llama3.1-8b-instruct", "Llama-3.1-8B-Instruct"),
]
RULER_LENGTHS = [8000, 16000, 32000]
RULER_BUDGETS = [128, 256, 512, 1024, 2048]
METHODS = {
    "baseline": ("FullCache", "black", "-", "o"),
    "earlyselectkv": ("ES-RocketKV", "#1f77b4", "-", "o"),
    "earlyselectkv_topk": ("ES-RocketKV_topk", "#1f77b4", "--", "s"),
    "rocketkv": ("RocketKV", "#ff7f0e", "-", "o"),
    "rocketkv_topk": ("RocketKV_topk", "#ff7f0e", "--", "s"),
}


def longbench_score(root: Path, method: str, model: str, budget: int | None) -> float:
    if method == "baseline":
        path = root / "longbench" / method / model / "longbench_result_summary.json"
    else:
        path = root / "longbench" / method / str(budget) / model / "longbench_result_summary.json"
    return float(json.loads(path.read_text(encoding="utf-8"))["LB_average_result"])


def ruler_score(root: Path, method: str, model: str, length: int, budget: int | None) -> float:
    if method == "baseline":
        path = root / "ruler" / str(length) / method / model / "results.json"
    else:
        path = root / "ruler" / str(length) / method / str(budget) / model / "results.json"
    return float(json.loads(path.read_text(encoding="utf-8"))["avg"])


def draw(ax: plt.Axes, budgets: list[int], values: dict[str, list[float]], caption: str) -> None:
    for method, scores in values.items():
        label, color, style, marker = METHODS[method]
        ax.plot(budgets, scores, label=label, color=color, linestyle=style, marker=marker, markersize=3, linewidth=1.4)
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(x) for x in budgets], rotation=30)
    ax.set_xlabel("Token Budget")
    ax.set_ylabel("Average Score")
    ax.grid(alpha=0.25)
    ax.text(0.5, -0.42, caption, transform=ax.transAxes, ha="center", va="top", fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the paper LongBench/RULER quality grid.")
    parser.add_argument("--root", default="result/paper_quality")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.output_dir) if args.output_dir else root / "plots"
    out.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.0, 7.0))
    grid = fig.add_gridspec(3, 12, hspace=0.95, wspace=0.85)
    axes = []
    for idx, (model, label) in enumerate(LB_MODELS):
        ax = fig.add_subplot(grid[0, idx * 3:(idx + 1) * 3])
        values = {}
        for method in METHODS:
            values[method] = [longbench_score(root, method, model, None if method == "baseline" else b) for b in LB_BUDGETS]
        draw(ax, LB_BUDGETS, values, f"LongBench, {label}")
        axes.append(ax)
    row = 1
    for model, label in RULER_MODELS:
        for col, length in enumerate(RULER_LENGTHS):
            ax = fig.add_subplot(grid[row, col * 4:(col + 1) * 4])
            values = {
                method: [ruler_score(root, method, model, length, None if method == "baseline" else b) for b in RULER_BUDGETS]
                for method in ("baseline", "earlyselectkv", "rocketkv")
            }
            draw(ax, RULER_BUDGETS, values, f"RULER-{length // 1000}K, {label}")
            axes.append(ax)
        row += 1
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "longbench_ruler.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "longbench_ruler.pdf", bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
