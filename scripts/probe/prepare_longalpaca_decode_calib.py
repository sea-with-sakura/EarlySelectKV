from __future__ import annotations

import argparse
import json
from pathlib import Path


def estimate_length(text: str) -> int:
    # LongBench uses a length field for reporting only; tokenization happens later.
    return max(1, len(text.split()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LongAlpaca-12k to the local LongBench jsonl format.")
    parser.add_argument("--input", default="dataset/alpaca12k/LongAlpaca-12k.json")
    parser.add_argument("--output", default="dataset/longbench/alpaca12k_decode_calib.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of examples to export; 0 exports all.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if args.limit > 0:
        data = data[: args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(data):
            instruction = str(item.get("instruction", "")).strip()
            output = str(item.get("output", "")).strip()
            row = {
                "input": "",
                "context": instruction,
                "answers": [output],
                "all_classes": None,
                "length": estimate_length(instruction),
                "dataset": "alpaca12k_decode_calib",
                "_id": str(idx),
            }
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")

    print(f"Wrote {len(data)} examples to {output_path}")


if __name__ == "__main__":
    main()
