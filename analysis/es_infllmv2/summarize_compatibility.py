from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASK_LABELS = {
    "gov_report": "GovReport",
    "qmsum": "QMSum",
    "musique": "MuSiQue",
    "hotpotqa": "HotpotQA",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing InfLLM-V2 result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _count_sparse_examples(summary: dict[str, Any], prediction_path: Path) -> int:
    dense_len = int(summary["sparse_config"]["dense_len"])
    count = 0
    with prediction_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and int(json.loads(line)["input_tokens"]) >= dense_len:
                count += 1
    return count


def summarize(root: Path, datasets: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        rank_dir = root / "rank256" / dataset
        exact_dir = root / "exact_wq" / dataset
        rank_summary = _load_json(rank_dir / "summary.json")
        exact_summary = _load_json(exact_dir / "summary.json")
        modes = rank_summary["modes"]
        prediction_path = rank_dir / f"{dataset}_oracle_sparse.jsonl"
        row = {
            "dataset": dataset,
            "label": TASK_LABELS.get(dataset, dataset),
            "metric": modes["oracle_sparse"]["metric"],
            "sample_count": int(rank_summary["sample_count"]),
            "sparse_count": _count_sparse_examples(rank_summary, prediction_path),
            "real_q": float(modes["oracle_sparse"]["score"]),
            "exact_wq": float(exact_summary["modes"]["es_sparse"]["score"]),
            "rank256": float(modes["es_sparse"]["score"]),
        }
        rows.append(row)

    exact_delta = sum(row["exact_wq"] - row["real_q"] for row in rows) / len(rows)
    rank_delta = sum(row["rank256"] - row["real_q"] for row in rows) / len(rows)
    return {
        "rows": rows,
        "mean_exact_wq_minus_real_q": round(exact_delta, 4),
        "mean_rank256_minus_real_q": round(rank_delta, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the paper InfLLM-V2 compatibility table.")
    parser.add_argument("--root", default="result/infllmv2_compatibility")
    parser.add_argument("--datasets", default="gov_report qmsum musique hotpotqa")
    args = parser.parse_args()

    root = Path(args.root)
    result = summarize(root, args.datasets.split())
    output = root / "infllmv2_compatibility_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("| Task | Sparse | Real-Q | Exact-Wq | Rank-256 |")
    print("|---|---:|---:|---:|---:|")
    for row in result["rows"]:
        print(
            f"| {row['label']} | {row['sparse_count']}/{row['sample_count']} "
            f"| {row['real_q']:.2f} | {row['exact_wq']:.2f} | {row['rank256']:.2f} |"
        )
    print(f"Mean Exact-Wq minus Real-Q: {result['mean_exact_wq_minus_real_q']:+.4f}")
    print(f"Mean Rank-256 minus Real-Q: {result['mean_rank256_minus_real_q']:+.4f}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
