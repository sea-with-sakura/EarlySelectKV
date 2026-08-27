from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from litgpt import LLM
from litgpt.model import apply_rope

from pipeline.auxiliary_kv_cache import build_chunk_summary_values, select_hsa_indices_from_summary


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split() if item]


def compute_rocketkv_params(prompt_len: int, max_new_tokens: int, token_budget: int) -> dict[str, int | float]:
    sequence_length = int(prompt_len) + int(max_new_tokens)
    token_budget = min(max(1, int(token_budget)), max(1, sequence_length))
    compression_ratio = max(1.0, float(sequence_length) / float(token_budget))
    rho = min(0.2 + 0.06 * math.log2(compression_ratio), 0.8)
    token_capacity_budget = int(float(sequence_length) / (compression_ratio**rho))
    topk_budget = max(1, min(round(token_budget / 2), int(token_capacity_budget)))
    stage2_ratio = max(1.0, float(token_capacity_budget) / float(token_budget))
    chunk_size = math.ceil(math.sqrt(stage2_ratio))
    if chunk_size > stage2_ratio:
        chunk_size = 1
    return {
        "rho": rho,
        "token_capacity_budget": token_capacity_budget,
        "topk_budget": topk_budget,
        "compression_ratio": stage2_ratio,
        "chunk_size": max(1, int(chunk_size)),
    }


def load_longbench_prompts(
    *,
    dataset_path: Path,
    prompt_path: Path,
    dataset: str,
    limit: int,
) -> list[str]:
    prompt_map = json.loads(prompt_path.read_text(encoding="utf-8"))
    template = prompt_map[dataset]
    prompts: list[str] = []
    with (dataset_path / f"{dataset}.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= limit:
                break
            row = json.loads(line)
            prompts.append(template.format(**row))
    return prompts


def middle_truncate(ids: torch.Tensor, max_tokens: int) -> torch.Tensor:
    if ids.numel() <= max_tokens:
        return ids
    left_len = max_tokens // 2
    right_len = max_tokens - left_len
    return torch.cat((ids[:left_len], ids[-right_len:]), dim=0)


def reshape_q(q_flat: torch.Tensor, *, n_head: int, head_dim: int) -> torch.Tensor:
    return q_flat.view(q_flat.size(0), q_flat.size(1), n_head, head_dim).transpose(1, 2).contiguous()


def reshape_k(k_flat: torch.Tensor, *, kv_heads: int, head_dim: int) -> torch.Tensor:
    return k_flat.view(k_flat.size(0), k_flat.size(1), kv_heads, head_dim).transpose(1, 2).contiguous()


def rope_tensor(t: torch.Tensor, *, cos: torch.Tensor, sin: torch.Tensor, rope_n_elem: int) -> torch.Tensor:
    roped = apply_rope(t[..., :rope_n_elem], cos, sin)
    return torch.cat((roped, t[..., rope_n_elem:]), dim=-1).contiguous()


def top_r_metrics(
    *,
    q_true: torch.Tensor,
    q_approx: torch.Tensor,
    kv_heads: int,
    r: int,
) -> dict[str, float]:
    batch, query_heads, q_len, head_dim = q_true.shape
    group = query_heads // kv_heads
    true_grouped = q_true.view(batch, kv_heads, group, q_len, head_dim)
    approx_grouped = q_approx.view(batch, kv_heads, group, q_len, head_dim)

    true_sum = true_grouped.sum(dim=2, keepdim=True)
    approx_sum = approx_grouped.sum(dim=2, keepdim=True)
    true_abs = true_grouped.abs().sum(dim=2, keepdim=True)
    approx_abs = approx_grouped.abs().sum(dim=2, keepdim=True)

    true_idx = torch.topk(true_abs, k=r, dim=-1, sorted=False).indices
    approx_idx = torch.topk(approx_abs, k=r, dim=-1, sorted=False).indices
    overlap = (true_idx.unsqueeze(-1) == approx_idx.unsqueeze(-2)).any(dim=-1).float().mean()

    true_sign_at_true = torch.gather(true_sum, -1, true_idx) > 0
    approx_sign_at_true = torch.gather(approx_sum, -1, true_idx) > 0
    sign_on_true = (true_sign_at_true == approx_sign_at_true).float().mean()

    true_signed = torch.where(true_sign_at_true, true_idx + head_dim, true_idx)
    approx_sign_at_approx = torch.gather(approx_sum, -1, approx_idx) > 0
    approx_signed = torch.where(approx_sign_at_approx, approx_idx + head_dim, approx_idx)
    signed_overlap = (true_signed.unsqueeze(-1) == approx_signed.unsqueeze(-2)).any(dim=-1).float().mean()

    return {
        "topr_recall": float(overlap.item()),
        "sign_match_on_true_topr": float(sign_on_true.item()),
        "signed_topr_recall": float(signed_overlap.item()),
    }


def selected_attention_mass(
    *,
    q_true: torch.Tensor,
    k: torch.Tensor,
    indices: torch.Tensor,
    kv_heads: int,
) -> torch.Tensor:
    q_grouped = q_true.view(q_true.size(0), kv_heads, -1, q_true.size(2), q_true.size(-1))
    scores = torch.matmul(q_grouped, k.unsqueeze(2).transpose(-1, -2)) / math.sqrt(q_true.size(-1))
    probs = torch.softmax(scores, dim=-1).mean(dim=2)
    gathered = torch.gather(probs, dim=-1, index=indices)
    return gathered.sum(dim=-1)


def hsa_metrics(
    *,
    q_true: torch.Tensor,
    q_approx: torch.Tensor,
    k: torch.Tensor,
    summary: torch.Tensor,
    key_len: int,
    chunk_size: int,
    compression_ratio: float,
    topk_budget: int,
    kv_heads: int,
) -> dict[str, float]:
    true_idx = select_hsa_indices_from_summary(
        q_true,
        summary,
        key_len=key_len,
        chunk_size=chunk_size,
        compression_ratio=compression_ratio,
        topk_budget=topk_budget,
        kv_heads=kv_heads,
        sorted_topk=False,
    )
    approx_idx = select_hsa_indices_from_summary(
        q_approx,
        summary,
        key_len=key_len,
        chunk_size=chunk_size,
        compression_ratio=compression_ratio,
        topk_budget=topk_budget,
        kv_heads=kv_heads,
        sorted_topk=False,
    )

    token_recall = (true_idx.unsqueeze(-1) == approx_idx.unsqueeze(-2)).any(dim=-1).float().mean()
    true_pages = true_idx // chunk_size
    approx_pages = approx_idx // chunk_size
    page_recall = (true_pages.unsqueeze(-1) == approx_pages.unsqueeze(-2)).any(dim=-1).float().mean()

    true_mass = selected_attention_mass(q_true=q_true, k=k, indices=true_idx, kv_heads=kv_heads)
    approx_mass = selected_attention_mass(q_true=q_true, k=k, indices=approx_idx, kv_heads=kv_heads)
    mass_ratio = (approx_mass / true_mass.clamp_min(1e-8)).mean()
    mass_gap = (approx_mass - true_mass).mean()

    return {
        "hsa_token_recall": float(token_recall.item()),
        "hsa_page_recall": float(page_recall.item()),
        "mass_ratio_to_true_hsa": float(mass_ratio.item()),
        "mass_gap_to_true_hsa": float(mass_gap.item()),
    }


def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    merged: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            merged[key].append(float(value))
    return {key: sum(values) / len(values) for key, values in merged.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether previous-layer hidden_in/hidden_mid plus low-rank next-layer Wq "
            "preserves RocketKV routing signals for the next layer."
        )
    )
    parser.add_argument("--model-dir", default="modelzoo/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--dataset-path", default="dataset/longbench")
    parser.add_argument("--prompt-path", default="utils/longbench_utils/config/dataset2prompt.json")
    parser.add_argument("--dataset", default="qasper")
    parser.add_argument("--output-dir", default="result/wq_svd_rank_probe/mistral_qasper")
    parser.add_argument("--layers", default="1,5,9,13,17,21,25,29,31", help="Target layers; source is layer-1.")
    parser.add_argument("--ranks", default="64,128,256,384,512,768,1024,1536,2048")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--window-tokens", type=int, default=24)
    parser.add_argument("--route-queries", type=int, default=8)
    parser.add_argument("--token-budget", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    layers = parse_int_list(args.layers)
    if any(layer <= 0 for layer in layers):
        raise ValueError("lookahead Wq-SVD probe expects target layers > 0 because source is target_layer - 1.")
    ranks = parse_int_list(args.ranks)

    model = LLM.load(model=args.model_dir)
    model.eval()
    base_model = model.model
    gpt = getattr(base_model, "module", base_model)
    tokenizer = model.tokenizer
    cfg = gpt.config
    n_head = int(cfg.n_head)
    kv_heads = int(cfg.n_query_groups)
    head_dim = int(cfg.head_size)
    rope_n_elem = int(cfg.rope_n_elem)

    source_layers = sorted({layer - 1 for layer in layers})
    layer_data: dict[int, list[dict[str, torch.Tensor | int]]] = {layer: [] for layer in layers}
    current_capture: dict[int, dict[str, torch.Tensor]] = {}
    hooks = []

    def make_source_prehook(layer_idx: int):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            hidden = inputs[0].detach()
            tail = min(int(args.window_tokens), int(hidden.size(1)))
            current_capture.setdefault(layer_idx, {})["hidden_in_full"] = hidden
            current_capture[layer_idx]["hidden_in_tail"] = hidden[:, -tail:, :].detach().cpu().contiguous()

        return hook

    def make_source_attn_hook(layer_idx: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            hidden_in = current_capture.setdefault(layer_idx, {}).get("hidden_in_full")
            if hidden_in is None:
                return
            block = gpt.transformer.h[layer_idx]
            attn_out = block.post_attention_norm(output.detach())
            hidden_mid = attn_out + hidden_in
            tail = min(int(args.window_tokens), int(hidden_mid.size(1)))
            current_capture[layer_idx]["hidden_mid_tail"] = hidden_mid[:, -tail:, :].detach().cpu().contiguous()

        return hook

    def make_target_qkv_hook(layer_idx: int):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            del inputs
            qkv = output.detach()
            query_size = n_head * head_dim
            key_size = kv_heads * head_dim
            q_flat = qkv[..., :query_size]
            k_flat = qkv[..., query_size : query_size + key_size]
            tail = min(int(args.window_tokens), int(q_flat.size(1)))
            current_capture.setdefault(layer_idx, {})["q_tail"] = q_flat[:, -tail:, :].detach().cpu().contiguous()
            current_capture[layer_idx]["k_all"] = k_flat.detach().cpu().contiguous()

        return hook

    for layer in source_layers:
        hooks.append(gpt.transformer.h[layer].register_forward_pre_hook(make_source_prehook(layer)))
        hooks.append(gpt.transformer.h[layer].attn.register_forward_hook(make_source_attn_hook(layer)))
    for layer in layers:
        hooks.append(gpt.transformer.h[layer].attn.qkv.register_forward_hook(make_target_qkv_hook(layer)))

    prompts = load_longbench_prompts(
        dataset_path=Path(args.dataset_path),
        prompt_path=Path(args.prompt_path),
        dataset=args.dataset,
        limit=int(args.samples),
    )

    try:
        with torch.no_grad():
            for sample_idx, prompt in enumerate(prompts):
                ids = tokenizer.encode(prompt, bos=False, device=device)
                ids = middle_truncate(ids, int(args.max_tokens)).to(device=device, dtype=torch.long)
                idx = ids.unsqueeze(0)
                input_pos = torch.arange(idx.size(1), device=device, dtype=torch.long)
                current_capture.clear()
                _ = model.model(idx)
                for layer in layers:
                    target = current_capture.get(layer)
                    source = current_capture.get(layer - 1)
                    if target is None:
                        raise RuntimeError(f"Missing target capture for layer {layer}.")
                    if source is None:
                        raise RuntimeError(f"Missing source capture for layer {layer - 1}.")
                    if "hidden_in_tail" not in source or "hidden_mid_tail" not in source:
                        raise RuntimeError(f"Missing hidden_in/hidden_mid capture for source layer {layer - 1}.")
                    layer_data[layer].append(
                        {
                            "sample_idx": sample_idx,
                            "seq_len": int(idx.size(1)),
                            "source_layer": layer - 1,
                            "hidden_in_tail": source["hidden_in_tail"],
                            "hidden_mid_tail": source["hidden_mid_tail"],
                            "q_tail": target["q_tail"],
                            "k_all": target["k_all"],
                        }
                    )
                print(f"captured sample {sample_idx}: seq_len={idx.size(1)}", flush=True)
    finally:
        for hook in hooks:
            hook.remove()

    rows: list[dict[str, Any]] = []
    layer_summaries: list[dict[str, Any]] = []
    state = torch.load(Path(args.model_dir) / "lit_model.pth", map_location="cpu", mmap=True, weights_only=True)

    for layer in layers:
        print(f"SVD target layer {layer} (source layer {layer - 1})", flush=True)
        wq = state[f"transformer.h.{layer}.attn.qkv.weight"][: n_head * head_dim].to(device=device, dtype=torch.float32)
        u, s, vh = torch.linalg.svd(wq, full_matrices=False)
        energy = torch.cumsum(s.square(), dim=0) / s.square().sum().clamp_min(1e-8)
        rank_energy = {f"energy_rank_{rank}": float(energy[min(rank, energy.numel()) - 1].item()) for rank in ranks}
        target_block = gpt.transformer.h[layer]

        for rank in ranks:
            rank_eff = min(int(rank), int(u.size(1)))
            u_r = u[:, :rank_eff]
            s_r = s[:rank_eff]
            vh_r = vh[:rank_eff, :]

            for anchor_mode in ("hidden_in", "hidden_mid"):
                rank_rows: list[dict[str, float]] = []

                for sample in layer_data[layer]:
                    seq_len = int(sample["seq_len"])
                    budgets = compute_rocketkv_params(seq_len, args.max_new_tokens, args.token_budget)
                    chunk_size = int(budgets["chunk_size"])
                    compression_ratio = float(budgets["compression_ratio"])
                    topk_budget = int(budgets["topk_budget"])
                    r = round(head_dim * chunk_size / max(compression_ratio, 1.0))
                    r = max(1, min(head_dim, int(r)))

                    hidden_tail = sample[f"{anchor_mode}_tail"].to(
                        device=device,
                        dtype=target_block.norm_1.weight.dtype,
                    )
                    hidden_tail = target_block.norm_1(hidden_tail).to(dtype=torch.float32)
                    q_true_raw = reshape_q(
                        sample["q_tail"].to(device=device, dtype=torch.float32),
                        n_head=n_head,
                        head_dim=head_dim,
                    )
                    k_raw = reshape_k(
                        sample["k_all"].to(device=device, dtype=torch.float32),
                        kv_heads=kv_heads,
                        head_dim=head_dim,
                    )
                    q_flat_approx = ((hidden_tail @ vh_r.t()) * s_r) @ u_r.t()
                    q_approx_raw = reshape_q(q_flat_approx, n_head=n_head, head_dim=head_dim)

                    tail_len = int(q_true_raw.size(2))
                    pos_all = torch.arange(seq_len, device=device, dtype=torch.long)
                    pos_tail = pos_all[-tail_len:]
                    cos_all = gpt.cos.index_select(0, pos_all).unsqueeze(0).to(device=device, dtype=torch.float32)
                    sin_all = gpt.sin.index_select(0, pos_all).unsqueeze(0).to(device=device, dtype=torch.float32)
                    cos_tail = gpt.cos.index_select(0, pos_tail).unsqueeze(0).to(device=device, dtype=torch.float32)
                    sin_tail = gpt.sin.index_select(0, pos_tail).unsqueeze(0).to(device=device, dtype=torch.float32)
                    q_true = rope_tensor(q_true_raw, cos=cos_tail, sin=sin_tail, rope_n_elem=rope_n_elem)
                    q_approx = rope_tensor(q_approx_raw, cos=cos_tail, sin=sin_tail, rope_n_elem=rope_n_elem)
                    k = rope_tensor(k_raw, cos=cos_all, sin=sin_all, rope_n_elem=rope_n_elem)
                    summary = build_chunk_summary_values(k, chunk_size=chunk_size, capacity_len=seq_len)

                    q_cos = F.cosine_similarity(
                        q_true.flatten(0, 2),
                        q_approx.flatten(0, 2),
                        dim=-1,
                    ).mean()
                    sign_match = ((q_true > 0) == (q_approx > 0)).float().mean()
                    pos_frac_gap = (q_approx > 0).float().mean() - (q_true > 0).float().mean()
                    metrics = {
                        "q_cosine": float(q_cos.item()),
                        "sign_match_all": float(sign_match.item()),
                        "positive_fraction_gap": float(pos_frac_gap.item()),
                    }
                    metrics.update(top_r_metrics(q_true=q_true, q_approx=q_approx, kv_heads=kv_heads, r=r))

                    route_queries = min(int(args.route_queries), tail_len)
                    hsa_rows = []
                    for q_offset in range(tail_len - route_queries, tail_len):
                        hsa_rows.append(
                            hsa_metrics(
                                q_true=q_true[:, :, q_offset : q_offset + 1, :],
                                q_approx=q_approx[:, :, q_offset : q_offset + 1, :],
                                k=k,
                                summary=summary,
                                key_len=seq_len,
                                chunk_size=chunk_size,
                                compression_ratio=compression_ratio,
                                topk_budget=topk_budget,
                                kv_heads=kv_heads,
                            )
                        )
                    metrics.update(mean_rows(hsa_rows))
                    metrics.update(
                        {
                            "hsa_channel_budget_r": float(r),
                            "chunk_size": float(chunk_size),
                            "compression_ratio": float(compression_ratio),
                            "topk_budget": float(topk_budget),
                        }
                    )
                    rank_rows.append(metrics)

                row: dict[str, Any] = {
                    "target_layer": layer,
                    "source_layer": layer - 1,
                    "anchor_mode": anchor_mode,
                    "rank": rank,
                    "wq_flop_ratio": (2.0 * rank_eff) / float(n_head * head_dim),
                    "wq_flop_saving": 1.0 - (2.0 * rank_eff) / float(n_head * head_dim),
                    "weight_energy": rank_energy[f"energy_rank_{rank}"],
                }
                row.update(mean_rows(rank_rows))
                rows.append(row)

        layer_summary = {"layer": layer}
        layer_summary.update(rank_energy)
        layer_summaries.append(layer_summary)
        del u, s, vh, wq
        torch.cuda.empty_cache()

    summary_by_rank = []
    for anchor_mode in ("hidden_in", "hidden_mid"):
        for rank in ranks:
            rank_rows = [
                {k: v for k, v in row.items() if isinstance(v, (int, float))}
                for row in rows
                if row["rank"] == rank and row["anchor_mode"] == anchor_mode
            ]
            merged = mean_rows(rank_rows)
            merged["anchor_mode"] = anchor_mode
            merged["rank"] = rank
            summary_by_rank.append(merged)

    with (output_dir / "wq_svd_rank_rows.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "wq_svd_rank_summary.json").write_text(
        json.dumps(
            {
                "model_dir": args.model_dir,
                "dataset": args.dataset,
                "samples": args.samples,
                "max_tokens": args.max_tokens,
                "window_tokens": args.window_tokens,
                "route_queries": args.route_queries,
                "layers": layers,
                "ranks": ranks,
                "layer_energy": layer_summaries,
                "summary_by_rank": summary_by_rank,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary_by_rank, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
