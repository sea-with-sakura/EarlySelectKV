import argparse
import csv
import json
from pathlib import Path


def read_summary(summary_path: Path, dataset: str) -> dict | None:
    if not summary_path.is_file():
        return None

    task_row = None
    score_row = None
    nulls_row = None
    with summary_path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if row[0] == "Tasks":
                task_row = row
            elif row[0] == "Score":
                score_row = row
            elif row[0] == "Nulls":
                nulls_row = row

    if task_row is None or score_row is None:
        return None

    try:
        idx = task_row.index(dataset)
    except ValueError:
        try:
            idx = task_row.index("avg")
        except ValueError:
            return None

    return {
        "score": float(score_row[idx]),
        "nulls": nulls_row[idx] if nulls_row is not None and idx < len(nulls_row) else None,
        "summary_path": str(summary_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-dir", required=True)
    parser.add_argument("--datasets", required=True, help="Space-separated RULER dataset names.")
    args = parser.parse_args()

    group_dir = Path(args.group_dir)
    datasets = args.datasets.split()
    results = {}
    for dataset in datasets:
        summary = read_summary(group_dir / dataset / "summary.csv", dataset)
        if summary is not None:
            results[dataset] = summary

    if not results:
        print(f"No completed RULER dataset summaries found under {group_dir}")
        return 0

    scores = [results[dataset]["score"] for dataset in datasets if dataset in results]
    avg = sum(scores) / len(scores)

    group_dir.mkdir(parents=True, exist_ok=True)
    with (group_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Tasks", *results.keys(), "avg"])
        writer.writerow(["Score", *[results[dataset]["score"] for dataset in results], avg])
        writer.writerow(["Nulls", *[results[dataset]["nulls"] for dataset in results], "N/A"])

    with (group_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump({"results": results, "avg": avg}, f, indent=4)

    print(f"Collected {len(results)} RULER summaries to {group_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
