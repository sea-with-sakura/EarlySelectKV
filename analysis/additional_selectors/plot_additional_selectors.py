from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASETS = [
    ("narrativeqa", "NarrativeQA"), ("qasper", "QASPER"),
    ("multifieldqa_en", "MultiFieldQA"), ("hotpotqa", "HotpotQA"),
    ("2wikimqa", "2WikiMQA"), ("musique", "MuSiQue"),
    ("gov_report", "GovReport"), ("qmsum", "QMSum"),
    ("multi_news", "MultiNews"), ("trec", "TREC"),
    ("triviaqa", "TriviaQA"), ("samsum", "Samsum"),
    ("passage_retrieval_en", "PassageRetrieval"), ("passage_count", "PassageCount"),
    ("lcc", "LCC"), ("repobench-p", "RepoBench"),
]
METHODS = ["loki", "earlyselect_loki", "quest", "earlyselect_quest"]
LABELS = ["Loki", "ES-Loki", "Quest", "ES-Quest"]
COLORS = ["#4d9994", "#a6d7d3", "#879bd7", "#e8f2f0"]


def load(root: Path, method: str, budget: int, model: str) -> dict[str, float]:
    path = root / "longbench" / method / str(budget) / model / "longbench_result_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    return {key: float(value) for key, value in summary["individual_dataset_result"].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Quest/Loki LongBench task scores and ES deltas.")
    parser.add_argument("--root", default="result/additional_selectors")
    parser.add_argument("--model", default="mistral-7b-instruct-v0.2")
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.output_dir) if args.output_dir else root / "plots"
    out.mkdir(parents=True, exist_ok=True)
    values = {method: load(root, method, args.budget, args.model) for method in METHODS}

    x = np.arange(len(DATASETS), dtype=float)
    width = 0.18
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(14.5, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [4.0, 1.15], "hspace": 0.04},
    )
    for idx, (method, label, color) in enumerate(zip(METHODS, LABELS, COLORS, strict=True)):
        scores = [values[method][key] for key, _ in DATASETS]
        top.bar(x + (idx - 1.5) * width, scores, width, label=label, color=color, edgecolor="#557b8a", linewidth=0.65)
    deltas = {
        "ES-Loki minus Loki": [values["earlyselect_loki"][key] - values["loki"][key] for key, _ in DATASETS],
        "ES-Quest minus Quest": [values["earlyselect_quest"][key] - values["quest"][key] for key, _ in DATASETS],
    }
    for idx, (label, delta) in enumerate(deltas.items()):
        bottom.bar(x + (idx - 0.5) * width, delta, width, label=label, color=COLORS[idx * 2 + 1], edgecolor="#557b8a", linewidth=0.65)
    bottom.axhline(0, color="black", linewidth=1.0)
    top.set_ylabel("Task Score")
    bottom.set_ylabel("Delta")
    top.set_ylim(bottom=0)
    top.grid(axis="y", alpha=0.25, linestyle="--")
    bottom.grid(axis="y", alpha=0.25, linestyle="--")
    bottom.set_xticks(x)
    bottom.set_xticklabels([label for _, label in DATASETS], rotation=32, ha="right")
    handles1, labels1 = top.get_legend_handles_labels()
    handles2, labels2 = bottom.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc="upper center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "additional_selector_results.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "additional_selector_results.pdf", bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
