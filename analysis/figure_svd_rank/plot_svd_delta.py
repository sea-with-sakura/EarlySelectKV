from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASETS = [
    ("narrativeqa", "NarrativeQA"),
    ("qasper", "QASPER"),
    ("multifieldqa_en", "MultiFieldQA"),
    ("hotpotqa", "HotpotQA"),
    ("2wikimqa", "2WikiMQA"),
    ("musique", "MuSiQue"),
    ("gov_report", "GovReport"),
    ("qmsum", "QMSum"),
    ("multi_news", "MultiNews"),
    ("trec", "TREC"),
    ("triviaqa", "TriviaQA"),
    ("samsum", "Samsum"),
    ("passage_retrieval_en", "PassageRetrieval"),
    ("passage_count", "PassageCount"),
    ("lcc", "LCC"),
    ("repobench-p", "RepoBench"),
]
RANKS = [64, 128, 256, 512]


def load_summary(root: Path, method: str, budget: int, model: str) -> dict:
    path = root / "longbench" / method / str(budget) / model / "longbench_result_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing LongBench summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LongBench score deltas for low-rank proxy Wq.")
    parser.add_argument("--root", default="result/svd_rank_longbench")
    parser.add_argument("--model", default="mistral-7b-instruct-v0.2")
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.output_dir) if args.output_dir else root / "plots"
    out.mkdir(parents=True, exist_ok=True)
    baseline = load_summary(root, "rocketkv", args.budget, args.model)
    variants = {
        rank: load_summary(root, f"earlyselectkv_mid_svd_r{rank}", args.budget, args.model)
        for rank in RANKS
    }
    base_tasks = baseline["individual_dataset_result"]
    task_deltas = {
        rank: [float(variants[rank]["individual_dataset_result"][key]) - float(base_tasks[key]) for key, _ in DATASETS]
        for rank in RANKS
    }
    avg_deltas = {
        rank: float(variants[rank]["LB_average_result"]) - float(baseline["LB_average_result"])
        for rank in RANKS
    }

    x = np.arange(len(DATASETS))
    width = 0.19
    colors = ["#dcefed", "#b6d9d5", "#79aaa5", "#3c817c"]
    hatches = ["///", "\\\\", "xx", "++"]
    fig, ax = plt.subplots(figsize=(14.2, 4.4))
    for idx, rank in enumerate(RANKS):
        ax.bar(
            x + (idx - 1.5) * width,
            task_deltas[rank],
            width,
            color=colors[idx],
            edgecolor="#315b59",
            linewidth=0.7,
            hatch=hatches[idx],
            label=f"r{rank}",
        )
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_ylabel("Delta vs. RocketKV")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in DATASETS], rotation=38, ha="right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(ncol=4, frameon=False, loc="upper center")
    inset = ax.inset_axes([0.87, 0.70, 0.12, 0.28])
    inset.bar(range(4), [avg_deltas[r] for r in RANKS], color=colors, edgecolor="#315b59", linewidth=0.7)
    inset.axhline(0, color="black", linewidth=0.8)
    inset.set_xticks([])
    inset.set_title("Avg.", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "svd_delta.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "svd_delta.pdf", bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
