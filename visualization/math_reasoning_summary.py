import argparse
import json
from pathlib import Path

import numpy as np


def load_score(dataset_dir: Path):
    config_path = dataset_dir / "output_config.json"
    if not config_path.is_file():
        return "N/A"
    data = json.loads(config_path.read_text(encoding="utf-8"))
    result = data.get("eval_results", {}).get("processed_results", {})
    if not result:
        return "N/A"
    value = next(iter(result.values()))
    return float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--datasets", default="math500", help="Space-separated math_reasoning datasets.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    datasets = args.datasets.split()

    individual = {}
    scores = []
    for dataset in datasets:
        score = load_score(output_dir / dataset)
        individual[dataset] = score
        if score != "N/A":
            scores.append(float(score))

    row = " ".join("N/A" if individual[d] == "N/A" else str(np.round(float(individual[d]), 2)) for d in datasets)
    average = "N/A" if len(scores) != len(datasets) else float(np.round(sum(scores) / len(scores), 2))
    summary = {
        "individual_dataset_result": individual,
        "individual_result": row,
        "average_result": average,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "math_reasoning_result_summary.json"
    out_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")
    print(f"Complete writing math_reasoning summary to {out_path}")


if __name__ == "__main__":
    main()
