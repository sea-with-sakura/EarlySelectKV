from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split() if item]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export low-rank SVD factors for LitGPT attention Wq.")
    parser.add_argument("--model-dir", default="modelzoo/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--output-dir", default="dataset/wq_svd/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--ranks", default="64,128,256,512")
    parser.add_argument("--layers", default="1-31", help="Target layers to export; use A-B or comma list.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    hidden_size = int(config["hidden_size"])
    n_head = int(config["num_attention_heads"])
    head_dim = hidden_size // n_head
    q_out = n_head * head_dim

    if "-" in args.layers:
        start, end = [int(x) for x in args.layers.split("-", 1)]
        layers = list(range(start, end + 1))
    else:
        layers = parse_int_list(args.layers)
    ranks = parse_int_list(args.ranks)
    max_rank = max(ranks)

    save_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    state = torch.load(model_dir / "lit_model.pth", map_location="cpu", mmap=True, weights_only=True)
    by_rank: dict[int, dict[str, object]] = {
        rank: {
            "metadata": {
                "model_dir": str(model_dir),
                "rank": rank,
                "layers": layers,
                "hidden_size": hidden_size,
                "q_out": q_out,
                "n_head": n_head,
                "head_dim": head_dim,
                "factorization": "Wq ~= up @ down, q = (x @ down.T) @ up.T",
                "dtype": args.dtype,
            },
            "layers": {},
        }
        for rank in ranks
    }

    for layer in layers:
        key = f"transformer.h.{layer}.attn.qkv.weight"
        if key not in state:
            raise KeyError(f"Missing {key} in {model_dir / 'lit_model.pth'}")
        wq = state[key][:q_out].to(device=device, dtype=torch.float32)
        print(f"SVD layer {layer}: {tuple(wq.shape)}", flush=True)
        u, s, vh = torch.linalg.svd(wq, full_matrices=False)
        for rank in ranks:
            r = min(int(rank), int(s.numel()))
            down = vh[:r, :].to(device="cpu", dtype=save_dtype).contiguous()
            up = (u[:, :r] * s[:r].unsqueeze(0)).to(device="cpu", dtype=save_dtype).contiguous()
            by_rank[rank]["layers"][int(layer)] = {
                "down": down,
                "up": up,
            }
        del u, s, vh, wq
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for rank, artifact in by_rank.items():
        path = output_dir / f"wq_svd_rank{rank}.pth"
        torch.save(artifact, path)
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
