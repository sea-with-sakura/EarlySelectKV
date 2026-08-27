from __future__ import annotations

import argparse
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import torch

from litgpt import LLM
from litgpt.model import CausalSelfAttention, apply_rope, apply_rope_interleave

from pipeline.config_utils import load_json_with_extends
from pipeline.model_utils import build_chat
from utils.longbench_utils.eval_long_bench import load_data

logger = logging.getLogger("loki-pca")


class OnlineKeyPCA:
    def __init__(self) -> None:
        self.counts: dict[int, int] = {}
        self.sums: dict[int, torch.Tensor] = {}
        self.xtx: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def observe(self, layer_idx: int, key_states: torch.Tensor) -> None:
        key_states = key_states.detach().to(dtype=torch.float32)
        if key_states.ndim != 4:
            raise ValueError(f"Expected key states [B, H, T, D], got {tuple(key_states.shape)}")
        _, _, time_steps, _ = key_states.shape
        if time_steps <= 0:
            return

        layer_idx = int(layer_idx)
        layer_sum = key_states.sum(dim=(0, 2)).cpu()
        layer_xtx = torch.einsum("bhtd,bhte->hde", key_states, key_states).cpu()

        if layer_idx not in self.sums:
            self.counts[layer_idx] = int(key_states.size(0) * time_steps)
            self.sums[layer_idx] = layer_sum
            self.xtx[layer_idx] = layer_xtx
            return

        self.counts[layer_idx] += int(key_states.size(0) * time_steps)
        self.sums[layer_idx] += layer_sum
        self.xtx[layer_idx] += layer_xtx

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        component_dir = output_dir / "key" / "pca_components"
        mean_dir = output_dir / "key" / "pca_means"
        variance_dir = output_dir / "key" / "pca_explained_variance"
        component_dir.mkdir(parents=True, exist_ok=True)
        mean_dir.mkdir(parents=True, exist_ok=True)
        variance_dir.mkdir(parents=True, exist_ok=True)

        for layer_idx in sorted(self.sums):
            count = int(self.counts[layer_idx])
            means = self.sums[layer_idx] / max(count, 1)
            cov = self.xtx[layer_idx] / max(count, 1) - torch.einsum("hd,he->hde", means, means)
            cov = 0.5 * (cov + cov.transpose(-1, -2))

            components_by_head = []
            variances_by_head = []
            for head_cov in cov:
                eigvals, eigvecs = torch.linalg.eigh(head_cov)
                eigvals = eigvals.clamp_min(0)
                order = torch.argsort(eigvals, descending=True)
                eigvals = eigvals[order]
                eigvecs = eigvecs[:, order]
                denom = eigvals.sum().clamp_min(torch.finfo(torch.float32).eps)
                variances_by_head.append(eigvals / denom)
                components_by_head.append(eigvecs.transpose(0, 1).contiguous())

            components = torch.stack(components_by_head, dim=0).to(dtype=torch.float32)
            explained = torch.stack(variances_by_head, dim=0).to(dtype=torch.float32)
            torch.save(components, component_dir / f"pca_components_layer_{layer_idx}.pt")
            torch.save(means.to(dtype=torch.float32), mean_dir / f"pca_means_layer_{layer_idx}.pt")
            torch.save(explained, variance_dir / f"pca_explained_variance_layer_{layer_idx}.pt")
            logger.info(
                "saved layer %s PCA: heads=%s head_dim=%s samples_per_head=%s",
                layer_idx,
                components.size(0),
                components.size(-1),
                count,
            )


@contextmanager
def collect_loki_keys(collector: OnlineKeyPCA, *, rotary_type: str):
    if rotary_type not in {"postrotary", "prerotary"}:
        raise ValueError(f"Unsupported rotary_type={rotary_type}")
    original_forward = CausalSelfAttention.forward

    def patched_forward(
        attn_self: CausalSelfAttention,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        input_pos_maxp1: Optional[int] = None,
    ) -> torch.Tensor:
        del input_pos, input_pos_maxp1
        head_size = attn_self.config.head_size
        n_head = attn_self.config.n_head
        n_query_groups = attn_self.config.n_query_groups
        rope_n_elem = attn_self.config.rope_n_elem
        batch, time_steps, _ = x.size()

        qkv = attn_self.qkv(x)
        query_size = n_head * head_size
        key_size = value_size = n_query_groups * head_size
        q, k, _ = qkv.split((query_size, key_size, value_size), dim=-1)

        if attn_self.config.norm_qk and attn_self.config.norm_qk_type == "olmo2":
            q = attn_self.norm_q(q)
            k = attn_self.norm_k(k)

        q = q.view(batch, time_steps, n_head, head_size).transpose(1, 2)
        k = k.view(batch, time_steps, n_query_groups, head_size).transpose(1, 2)

        if attn_self.config.norm_qk and attn_self.config.norm_qk_type == "default":
            q = attn_self.norm_q(q)
            k = attn_self.norm_k(k)

        if rotary_type == "prerotary":
            collector.observe(attn_self.block_idx, k)
            return original_forward(attn_self, x, cos, sin, mask, None, None)

        if attn_self.config.rope_interleave:
            k_roped = apply_rope_interleave(k[..., :rope_n_elem], cos, sin)
        else:
            k_roped = apply_rope(k[..., :rope_n_elem], cos, sin)
        k = torch.cat((k_roped, k[..., rope_n_elem:]), dim=-1)
        collector.observe(attn_self.block_idx, k)
        return original_forward(attn_self, x, cos, sin, mask, None, None)

    CausalSelfAttention.forward = patched_forward
    try:
        yield
    finally:
        CausalSelfAttention.forward = original_forward


def _load_eval_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_calibration_prompts(model: LLM, tokenizer, pipeline_params: dict, eval_params: dict, limit: int) -> list[str]:
    data = load_data(eval_params)
    prompts = []
    for idx, item in enumerate(data):
        if limit > 0 and idx >= limit:
            break
        prompt = eval_params["instruction"].format(**item)
        prompt = build_chat(model, tokenizer, prompt, pipeline_params=pipeline_params)
        prompts.append(prompt)
    return prompts


def _build_longbench_token_batches(
    model: LLM,
    tokenizer,
    pipeline_params: dict,
    eval_params: dict,
    limit: int,
    max_tokens: int,
    device: torch.device,
) -> list[torch.Tensor]:
    prompts = _build_calibration_prompts(
        model,
        tokenizer,
        pipeline_params=pipeline_params,
        eval_params=eval_params,
        limit=limit,
    )
    batches = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, bos=False, device=device)
        tokens = _truncate_tokens(tokens, max_tokens).view(1, -1)
        if tokens.size(1) > 0:
            batches.append(tokens)
    return batches


def _build_wikitext_token_batches(
    tokenizer,
    split: str,
    limit: int,
    max_tokens: int,
    device: torch.device,
    keep_last_partial: bool,
) -> list[torch.Tensor]:
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(row["text"].strip() for row in dataset if row["text"].strip())
    tokens = tokenizer.encode(text, bos=False, device=device).view(-1)
    block_size = int(max_tokens) if max_tokens > 0 else int(tokens.numel())
    if block_size <= 0:
        return []

    sample_limit = int(limit) if limit > 0 else 2**31 - 1
    batches = []
    for start in range(0, int(tokens.numel()), block_size):
        if len(batches) >= sample_limit:
            break
        chunk = tokens[start : start + block_size]
        if chunk.numel() < block_size and not keep_last_partial:
            break
        if chunk.numel() > 0:
            batches.append(chunk.view(1, -1))
    return batches


def _build_text_file_token_batches(
    tokenizer,
    text_file: str | Path,
    limit: int,
    max_tokens: int,
    device: torch.device,
    keep_last_partial: bool,
) -> list[torch.Tensor]:
    text = Path(text_file).read_text(encoding="utf-8")
    tokens = tokenizer.encode(text, bos=False, device=device).view(-1)
    block_size = int(max_tokens) if max_tokens > 0 else int(tokens.numel())
    if block_size <= 0:
        return []

    sample_limit = int(limit) if limit > 0 else 2**31 - 1
    batches = []
    for start in range(0, int(tokens.numel()), block_size):
        if len(batches) >= sample_limit:
            break
        chunk = tokens[start : start + block_size]
        if chunk.numel() < block_size and not keep_last_partial:
            break
        if chunk.numel() > 0:
            batches.append(chunk.view(1, -1))
    return batches


def _build_calibration_token_batches(
    model: LLM,
    tokenizer,
    pipeline_params: dict,
    eval_params: dict | None,
    args: argparse.Namespace,
    device: torch.device,
) -> list[torch.Tensor]:
    calibration_dataset = str(args.calibration_dataset)
    if calibration_dataset.startswith("wikitext-"):
        split_name = calibration_dataset.removeprefix("wikitext-")
        split = "validation" if split_name == "valid" else split_name
        logger.info("using WikiText-2 raw %s split for Loki PCA calibration", split)
        return _build_wikitext_token_batches(
            tokenizer,
            split=split,
            limit=int(args.max_samples),
            max_tokens=int(args.max_tokens),
            device=device,
            keep_last_partial=bool(args.keep_last_partial),
        )

    if calibration_dataset == "text-file":
        if args.calibration_text_file is None:
            raise ValueError("--calibration-text-file is required when --calibration-dataset text-file")
        logger.info("using text file %s for Loki PCA calibration", args.calibration_text_file)
        return _build_text_file_token_batches(
            tokenizer,
            text_file=args.calibration_text_file,
            limit=int(args.max_samples),
            max_tokens=int(args.max_tokens),
            device=device,
            keep_last_partial=bool(args.keep_last_partial),
        )

    if calibration_dataset == "longbench":
        if eval_params is None:
            raise ValueError("--eval-config is required when --calibration-dataset longbench")
        logger.warning("using LongBench task data for PCA calibration; this is only for debugging, not fair evaluation")
        return _build_longbench_token_batches(
            model,
            tokenizer,
            pipeline_params=pipeline_params,
            eval_params=eval_params,
            limit=int(args.max_samples),
            max_tokens=int(args.max_tokens),
            device=device,
        )

    raise ValueError(f"Unsupported calibration dataset: {calibration_dataset}")


def _truncate_tokens(tokens: torch.Tensor, max_tokens: int) -> torch.Tensor:
    if max_tokens <= 0 or tokens.numel() <= max_tokens:
        return tokens
    half = max_tokens // 2
    right = max_tokens - half
    return torch.cat((tokens[:half], tokens[-right:]), dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Loki PCA transforms from LitGPT keys.")
    parser.add_argument("--pipeline-config", required=True)
    parser.add_argument("--eval-config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--calibration-dataset",
        default="wikitext-valid",
        choices=["wikitext-valid", "wikitext-test", "wikitext-train", "text-file", "longbench"],
        help="Dataset used to fit PCA transforms. Use longbench only for debugging because it leaks eval data.",
    )
    parser.add_argument("--calibration-text-file")
    parser.add_argument("--max-samples", type=int, default=0, help="Maximum calibration chunks; <=0 uses the full split.")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--rotary-type",
        choices=["postrotary", "prerotary"],
        default="postrotary",
        help="Whether to fit PCA on keys before or after applying RoPE.",
    )
    parser.add_argument(
        "--keep-last-partial",
        action="store_true",
        help="Keep a final calibration chunk shorter than --max-tokens. Official Loki PCA drops it.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s : %(message)s")
    pipeline_config = load_json_with_extends(args.pipeline_config)
    eval_config = _load_eval_config(args.eval_config) if args.eval_config else None
    pipeline_params = pipeline_config["pipeline_params"]
    eval_params = eval_config["eval_params"] if eval_config is not None else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("loading LitGPT model from %s", pipeline_params["model_name"])
    model = LLM.load(model=pipeline_params["model_name"])
    model.model.eval()
    tokenizer = model.tokenizer
    collector = OnlineKeyPCA()

    token_batches = _build_calibration_token_batches(
        model,
        tokenizer,
        pipeline_params=pipeline_params,
        eval_params=eval_params,
        args=args,
        device=device,
    )
    logger.info(
        "calibration dataset=%s samples=%s max_tokens=%s output=%s",
        args.calibration_dataset,
        len(token_batches),
        args.max_tokens,
        args.output_dir,
    )

    logger.info("collecting %s Loki keys", args.rotary_type)
    with collect_loki_keys(collector, rotary_type=str(args.rotary_type)), torch.inference_mode():
        for idx, tokens in enumerate(token_batches):
            if tokens.size(1) > model.model.max_seq_length:
                model.model.max_seq_length = int(tokens.size(1))
                model.model.cos, model.model.sin = model.model.rope_cache(device=device)
            logger.info("PCA sample %s/%s tokens=%s", idx + 1, len(token_batches), tokens.size(1))
            model.model(tokens)
            torch.cuda.empty_cache()

    collector.save(args.output_dir)


if __name__ == "__main__":
    main()
