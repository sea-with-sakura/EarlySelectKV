import json
from pathlib import Path

from datasets import load_dataset


DATASETS = {
    "math500": ("HuggingFaceH4/MATH-500", "test"),
}


def main():
    save_dir = Path("dataset/math_reasoning")
    save_dir.mkdir(parents=True, exist_ok=True)

    for name, (repo_id, split) in DATASETS.items():
        data = load_dataset(repo_id, split=split)
        out_path = save_dir / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in data:
                json.dump(dict(row), f, ensure_ascii=False)
                f.write("\n")
        print(f"Saved {len(data)} rows to {out_path}")


if __name__ == "__main__":
    main()
