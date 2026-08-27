from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.es_infllmv2.es_infllmv2 import (  # noqa: E402
    config_from_sparse_config,
    install_esinfllmv2,
    uninstall_esinfllmv2,
)
from utils.longbench_utils.eval_long_bench import dataset2metric  # noqa: E402


def parse_modes(value: str) -> list[str]:
    return [item.strip() for item in value.replace(",", " ").split() if item.strip()]


def build_prompt(row: dict[str, Any], instruction: str) -> str:
    context = str(row.get("context", "")).replace("NEWLINE_CHAR", "\n")
    question = str(row.get("input", "")).replace("NEWLINE_CHAR", "\n")
    return instruction.format(context=context, input=question)


def iter_rows(dataset_path: Path, sample_count: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if sample_count is not None and len(rows) >= sample_count:
                break
    return rows


def middle_truncate(input_ids: torch.Tensor, max_length: int) -> torch.Tensor:
    if input_ids.size(1) <= max_length:
        return input_ids
    first = max_length // 2
    second = max_length - first
    return torch.cat([input_ids[:, :first], input_ids[:, -second:]], dim=1)


def encode_prompt(tokenizer: Any, prompt: str, max_input_length: int, device: torch.device) -> torch.Tensor:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = middle_truncate(encoded["input_ids"], max_input_length)
    return input_ids.to(device)


def get_eos_ids(tokenizer: Any, model: torch.nn.Module) -> set[int]:
    eos = getattr(model.config, "eos_token_id", None)
    ids: set[int] = set()
    if isinstance(eos, int):
        ids.add(eos)
    elif isinstance(eos, (list, tuple)):
        ids.update(int(item) for item in eos)
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    return ids


def make_cache(model: torch.nn.Module) -> Any:
    cache_cls = model.model.forward.__globals__.get("InfLLMv2Cache")
    if cache_cls is not None:
        return cache_cls(config=model.config, num_hidden_layers=model.config.num_hidden_layers)
    from transformers.cache_utils import DynamicCache

    try:
        return DynamicCache(config=model.config)
    except TypeError:
        return DynamicCache()


def model_forward(model: torch.nn.Module, **kwargs: Any) -> Any:
    runtime = getattr(model, "_esinfllmv2_runtime", None)
    if runtime is not None:
        runtime.reset_step()
    try:
        return model(**kwargs, logits_to_keep=1)
    except TypeError:
        return model(**kwargs)


@torch.inference_mode()
def greedy_generate(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> str:
    device = input_ids.device
    cache = make_cache(model)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
    position_ids = torch.arange(input_ids.size(1), device=device, dtype=torch.long).unsqueeze(0)
    outputs = model_forward(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    cache = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1)
    generated: list[int] = []
    eos_ids = get_eos_ids(tokenizer, model)

    for _ in range(max_new_tokens):
        token_id = int(next_token.item())
        generated.append(token_id)
        if token_id in eos_ids:
            break
        current = next_token.view(1, 1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)],
            dim=1,
        )
        position_ids = torch.tensor([[attention_mask.size(1) - 1]], dtype=torch.long, device=device)
        outputs = model_forward(
            model,
            input_ids=current,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)

    return tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def score_predictions(dataset: str, predictions: list[str], rows: list[dict[str, Any]]) -> tuple[float, list[float]]:
    metric = dataset2metric[dataset]
    scores: list[float] = []
    for pred, row in zip(predictions, rows, strict=True):
        answers = row.get("answers", [])
        if not isinstance(answers, list):
            answers = [answers]
        all_classes = row.get("all_classes", [])
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            pred = pred.lstrip("\n").split("\n")[0]
        best = 0.0
        for answer in answers:
            best = max(best, float(metric(pred, answer, all_classes=all_classes)))
        scores.append(best)
    return round(100.0 * sum(scores) / max(1, len(scores)), 2), scores


def load_sparse_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return dict(raw.get("sparse_config") or {})


def missing_shards(model_dir: Path) -> list[str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return ["model.safetensors.index.json"]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    files = sorted(set(index["weight_map"].values()))
    return [name for name in files if not (model_dir / name).exists()]


def load_model(model_dir: Path, device: torch.device, dtype: torch.dtype) -> tuple[Any, torch.nn.Module, dict[str, Any]]:
    missing = missing_shards(model_dir)
    if missing:
        raise SystemExit(
            "Model checkpoint is incomplete. Missing files:\n"
            + "\n".join(f"  - {name}" for name in missing)
            + "\nRun the download command first, then rerun this script."
        )

    sparse_config = load_sparse_config(model_dir)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    config.sparse_config = None
    config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        dtype=dtype,
        device_map={"": str(device)},
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    return tokenizer, model, sparse_config


def main() -> None:
    parser = argparse.ArgumentParser(description="InfLLM-V2 Long-Sparse precision-only ES sparse evaluation.")
    parser.add_argument("--model-dir", default="modelzoo/InfLLM-V2-Long-Sparse-Base")
    parser.add_argument("--eval-config", default="config/eval_config/longbench/qasper.json")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--output-dir", default="analysis/es_infllmv2/out/qasper")
    parser.add_argument("--modes", default="oracle_sparse,es_sparse")
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--max-input-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--route-query", choices=["lowrank_next_wq", "exact_next_wq"], default="lowrank_next_wq")
    parser.add_argument("--remote-topk", type=int, default=None)
    parser.add_argument("--route-q-scale", type=float, default=1.0)
    parser.add_argument("--lookahead-source", choices=["in", "mid", "out"], default="mid")
    parser.add_argument("--lowrank-cache", default="analysis/es_infllmv2/cache/infllmv2_long_sparse_rank256.pt")
    parser.add_argument("--use-k2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-prefill-sparse", action="store_true")
    args = parser.parse_args()

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_config = json.loads(Path(args.eval_config).read_text(encoding="utf-8"))
    eval_params = eval_config["eval_params"]
    dataset = str(eval_params["dataset"])
    instruction = str(eval_params["instruction"])
    dataset_root = Path(eval_params["dataset_path"])
    dataset_path = Path(args.dataset_path) if args.dataset_path else dataset_root / f"{dataset}.jsonl"
    rows = iter_rows(dataset_path, args.sample_count)
    max_new_tokens = min(int(args.max_new_tokens), int(eval_params.get("max_new_tokens", args.max_new_tokens)))

    tokenizer, model, sparse_config = load_model(model_dir, device=device, dtype=dtype)
    modes = parse_modes(args.modes)
    summary: dict[str, Any] = {
        "dataset": dataset,
        "sample_count": len(rows),
        "max_input_length": args.max_input_length,
        "max_new_tokens": max_new_tokens,
        "sparse_config": sparse_config,
        "modes": {},
    }

    for mode in modes:
        if mode not in {"dense", "oracle_sparse", "es_sparse"}:
            raise ValueError(f"Unsupported mode={mode!r}")
        print(f"running mode={mode}", flush=True)
        runtime = None
        if mode != "dense":
            cfg = config_from_sparse_config(
                sparse_config,
                mode=mode,
                rank=args.rank,
                route_query=args.route_query,
                remote_topk=args.remote_topk,
                route_q_scale=args.route_q_scale,
                lookahead_source=args.lookahead_source,
                apply_decode_only=not args.include_prefill_sparse,
                use_k2=args.use_k2,
                lowrank_cache_path=args.lowrank_cache if mode == "es_sparse" else None,
            )
            runtime = install_esinfllmv2(model, cfg)

        predictions: list[str] = []
        pred_path = out_dir / f"{dataset}_{mode}.jsonl"
        with pred_path.open("w", encoding="utf-8") as f:
            for sample_id, row in enumerate(rows):
                prompt = build_prompt(row, instruction)
                input_ids = encode_prompt(tokenizer, prompt, args.max_input_length, device)
                pred = greedy_generate(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                )
                predictions.append(pred)
                out_row = {
                    "sample_id": sample_id,
                    "mode": mode,
                    "pred": pred,
                    "answers": row.get("answers", []),
                    "all_classes": row.get("all_classes", []),
                    "length": row.get("length", int(input_ids.size(1))),
                    "input_tokens": int(input_ids.size(1)),
                }
                f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                print(f"  sample={sample_id} input_tokens={input_ids.size(1)} pred={pred[:80]!r}", flush=True)

        score, per_sample_scores = score_predictions(dataset, predictions, rows)
        mode_summary = {
            "score": score,
            "metric": dataset2metric[dataset].__name__,
            "prediction_path": str(pred_path),
            "per_sample_scores": per_sample_scores,
            "runtime_stats": asdict(runtime.stats) if runtime is not None else {},
        }
        summary["modes"][mode] = mode_summary
        (out_dir / f"{dataset}_{mode}_summary.json").write_text(
            json.dumps(mode_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if mode != "dense":
            uninstall_esinfllmv2(model)
        torch.cuda.empty_cache()

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
