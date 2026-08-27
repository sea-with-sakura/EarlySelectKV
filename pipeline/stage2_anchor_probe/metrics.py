from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from pipeline.runtime_decode_utils import sorted_topk_indices


METRIC_KEYS = [
    "cosine_similarity",
    "l2_relative_error",
    "sign_agreement",
    "topr_channel_recall",
    "topr_channel_jaccard",
    "topr_sign_agreement",
    "signed_channel_recall",
    "signed_channel_jaccard",
    "topk_recall",
    "topk_jaccard",
    "page_recall",
    "page_jaccard",
    "mass",
    "topk_mass",
    "mass_gap_to_topk",
    "sink_page_ratio",
    "local_page_ratio",
    "local_window_ratio",
    "generated_token_ratio",
    "selected_len",
    "snap_len",
    "channel_budget_r",
]


def _row_sets(x: torch.Tensor) -> list[set[int]]:
    if x.dim() == 4 and x.size(2) == 1:
        x = x.squeeze(2)
    flat = x.detach().to(device="cpu", dtype=torch.long).reshape(-1, x.size(-1))
    return [set(row.tolist()) for row in flat]


def _overlap(candidate: torch.Tensor, truth: torch.Tensor) -> tuple[float, float]:
    recalls = []
    jaccards = []
    for cand, true in zip(_row_sets(candidate), _row_sets(truth), strict=False):
        if not true:
            recalls.append(1.0)
            jaccards.append(1.0 if not cand else 0.0)
            continue
        inter = len(cand & true)
        union = len(cand | true)
        recalls.append(inter / len(true))
        jaccards.append(inter / union if union else 1.0)
    return float(sum(recalls) / len(recalls)), float(sum(jaccards) / len(jaccards))


def _page_indices(indices: torch.Tensor, page_size: int) -> torch.Tensor:
    return torch.div(indices, max(1, int(page_size)), rounding_mode="floor")


def grouped_true_probs(
    *,
    q_true: torch.Tensor,
    k: torch.Tensor,
    attention_heads: int,
    kv_heads: int,
) -> torch.Tensor:
    q_grouped = q_true.reshape(q_true.size(0), kv_heads, attention_heads // kv_heads, q_true.size(2), q_true.size(-1))
    scores = torch.matmul(q_grouped, k.unsqueeze(2).transpose(-1, -2)) / math.sqrt(q_true.size(-1))
    probs = torch.softmax(scores, dim=-1)
    return probs.mean(dim=2).squeeze(-2)


def exact_topk_from_probs(probs: torch.Tensor, k: int) -> torch.Tensor:
    return sorted_topk_indices(probs, max(0, min(int(k), int(probs.size(-1)))))


def route_mass(probs: torch.Tensor, indices: torch.Tensor) -> float:
    if indices.numel() == 0:
        return 0.0
    if indices.dim() == 4 and indices.size(2) == 1:
        indices = indices.squeeze(2)
    gathered = torch.gather(probs, dim=-1, index=indices)
    return float(gathered.sum(dim=-1).mean().item())


def route_token_ratios(
    *,
    indices: torch.Tensor,
    snap_len: int,
    active_prompt_len: int,
    page_size: int,
    window_size: int,
) -> dict[str, float]:
    if indices.numel() == 0:
        return {
            "sink_page_ratio": 0.0,
            "local_page_ratio": 0.0,
            "local_window_ratio": 0.0,
            "generated_token_ratio": 0.0,
        }
    flat = indices.detach().to(dtype=torch.long)
    sink_len = max(1, int(page_size))
    local_page_start = max(0, int(snap_len) - max(1, int(page_size)))
    local_window_start = max(0, int(snap_len) - max(1, int(window_size)))
    return {
        "sink_page_ratio": float((flat < sink_len).to(torch.float32).mean().item()),
        "local_page_ratio": float((flat >= local_page_start).to(torch.float32).mean().item()),
        "local_window_ratio": float((flat >= local_window_start).to(torch.float32).mean().item()),
        "generated_token_ratio": float((flat >= int(active_prompt_len)).to(torch.float32).mean().item()),
    }


def layer_bucket(layer_idx: int, num_layers: int) -> str:
    if num_layers <= 0:
        return "unknown"
    frac = (int(layer_idx) + 1) / float(num_layers)
    if frac <= 1.0 / 3.0:
        return "early"
    if frac <= 2.0 / 3.0:
        return "middle"
    return "late"


def decode_bucket(decode_step: int) -> str:
    step = int(decode_step)
    if step <= 0:
        return "d000"
    if step <= 4:
        return "d001_004"
    if step <= 16:
        return "d005_016"
    return "d017_plus"


def build_stage2_record(
    *,
    method: str,
    task: str,
    model_name: str,
    token_budget: int,
    sample_id: str,
    layer_idx: int,
    num_layers: int,
    logical_pos: int,
    decode_step: int,
    q_true: torch.Tensor,
    k: torch.Tensor,
    route_indices: torch.Tensor | None,
    topk_budget: int,
    attention_heads: int,
    kv_heads: int,
    page_size: int,
    active_prompt_len: int = 0,
    window_size: int = 32,
    probs: torch.Tensor | None = None,
    oracle: torch.Tensor | None = None,
    topk_mass: float | None = None,
    candidate_method: str | None = None,
) -> dict[str, object]:
    if probs is None:
        probs = grouped_true_probs(q_true=q_true, k=k, attention_heads=attention_heads, kv_heads=kv_heads)
    if oracle is None:
        oracle = exact_topk_from_probs(probs, topk_budget)
    if topk_mass is None:
        topk_mass = route_mass(probs, oracle)
    snap_len = int(k.size(2))

    if route_indices is None:
        recall = 1.0
        jaccard = float(oracle.size(-1)) / float(snap_len) if snap_len > 0 else 1.0
        page_recall = 1.0
        page_jaccard = 1.0
        mass = 1.0
        selected_len = snap_len
        ratios = route_token_ratios(
            indices=torch.arange(snap_len).view(1, 1, snap_len),
            snap_len=snap_len,
            active_prompt_len=active_prompt_len,
            page_size=page_size,
            window_size=window_size,
        )
    else:
        recall, jaccard = _overlap(route_indices, oracle)
        page_recall, page_jaccard = _overlap(_page_indices(route_indices, page_size), _page_indices(oracle, page_size))
        mass = route_mass(probs, route_indices)
        selected_len = int(route_indices.size(-1))
        ratios = route_token_ratios(
            indices=route_indices,
            snap_len=snap_len,
            active_prompt_len=active_prompt_len,
            page_size=page_size,
            window_size=window_size,
        )

    return {
        "method": candidate_method or method,
        "probe_run_method": method,
        "task": task,
        "model_name": model_name,
        "token_budget": int(token_budget),
        "sample_id": sample_id,
        "layer_idx": int(layer_idx),
        "layer_bucket": layer_bucket(layer_idx, num_layers),
        "logical_pos": int(logical_pos),
        "decode_step": int(decode_step),
        "decode_bucket": decode_bucket(decode_step),
        "topk_recall": float(recall),
        "topk_jaccard": float(jaccard),
        "page_recall": float(page_recall),
        "page_jaccard": float(page_jaccard),
        "mass": float(mass),
        "topk_mass": float(topk_mass),
        "mass_gap_to_topk": float(mass - topk_mass),
        **ratios,
        "selected_len": int(selected_len),
        "snap_len": int(snap_len),
    }


def _mean_overlap_from_sets(candidates: list[set[tuple[int, int]]], truths: list[set[tuple[int, int]]]) -> tuple[float, float]:
    recalls = []
    jaccards = []
    for cand, true in zip(candidates, truths, strict=False):
        if not true:
            recalls.append(1.0)
            jaccards.append(1.0 if not cand else 0.0)
            continue
        inter = len(cand & true)
        union = len(cand | true)
        recalls.append(inter / len(true))
        jaccards.append(inter / union if union else 1.0)
    return float(sum(recalls) / len(recalls)), float(sum(jaccards) / len(jaccards))


def query_quality_records(
    *,
    method: str,
    task: str,
    model_name: str,
    token_budget: int,
    sample_id: str,
    layer_idx: int,
    num_layers: int,
    logical_pos: int,
    decode_step: int,
    q_candidate: torch.Tensor,
    q_true: torch.Tensor,
    channel_budget_r: int,
    head_rows: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    q_candidate = q_candidate.detach().to(dtype=torch.float32)
    q_true = q_true.detach().to(device=q_candidate.device, dtype=torch.float32)
    if q_candidate.dim() == 4 and q_candidate.size(2) == 1:
        q_candidate_flat = q_candidate.squeeze(2)
    else:
        q_candidate_flat = q_candidate.reshape(q_candidate.size(0), q_candidate.size(1), -1)
    if q_true.dim() == 4 and q_true.size(2) == 1:
        q_true_flat = q_true.squeeze(2)
    else:
        q_true_flat = q_true.reshape(q_true.size(0), q_true.size(1), -1)

    eps = 1e-12
    cos = F.cosine_similarity(q_candidate_flat, q_true_flat, dim=-1)
    l2_rel = torch.linalg.vector_norm(q_candidate_flat - q_true_flat, dim=-1) / (
        torch.linalg.vector_norm(q_true_flat, dim=-1) + eps
    )
    sign_agree = (torch.sign(q_candidate_flat) == torch.sign(q_true_flat)).to(torch.float32).mean(dim=-1)

    r = max(1, min(int(channel_budget_r), int(q_true_flat.size(-1))))
    true_top = torch.topk(q_true_flat.abs(), k=r, dim=-1).indices
    cand_top = torch.topk(q_candidate_flat.abs(), k=r, dim=-1).indices

    flat_true_top = true_top.reshape(-1, r)
    flat_cand_top = cand_top.reshape(-1, r)
    channel_recalls = []
    channel_jaccards = []
    signed_cand_sets = []
    signed_true_sets = []
    top_sign_agree_values = []
    flat_cand = q_candidate_flat.reshape(-1, q_candidate_flat.size(-1))
    flat_true = q_true_flat.reshape(-1, q_true_flat.size(-1))
    for row_idx, (cand_idx, true_idx) in enumerate(zip(flat_cand_top, flat_true_top, strict=False)):
        cand_set = set(int(x) for x in cand_idx.detach().cpu().tolist())
        true_set = set(int(x) for x in true_idx.detach().cpu().tolist())
        inter = len(cand_set & true_set)
        union = len(cand_set | true_set)
        channel_recalls.append(inter / max(1, len(true_set)))
        channel_jaccards.append(inter / union if union else 1.0)

        cand_sign_at_true = torch.sign(flat_cand[row_idx, true_idx])
        true_sign_at_true = torch.sign(flat_true[row_idx, true_idx])
        top_sign_agree_values.append(float((cand_sign_at_true == true_sign_at_true).to(torch.float32).mean().item()))

        cand_signs = torch.sign(flat_cand[row_idx, cand_idx]).detach().cpu().to(torch.int8).tolist()
        true_signs = torch.sign(flat_true[row_idx, true_idx]).detach().cpu().to(torch.int8).tolist()
        signed_cand_sets.append(set(zip((int(x) for x in cand_idx.detach().cpu().tolist()), (int(s) for s in cand_signs))))
        signed_true_sets.append(set(zip((int(x) for x in true_idx.detach().cpu().tolist()), (int(s) for s in true_signs))))

    signed_recall, signed_jaccard = _mean_overlap_from_sets(signed_cand_sets, signed_true_sets)
    base = {
        "method": method,
        "task": task,
        "model_name": model_name,
        "token_budget": int(token_budget),
        "sample_id": sample_id,
        "layer_idx": int(layer_idx),
        "layer_bucket": layer_bucket(layer_idx, num_layers),
        "logical_pos": int(logical_pos),
        "decode_step": int(decode_step),
        "decode_bucket": decode_bucket(decode_step),
        "cosine_similarity": float(cos.mean().item()),
        "l2_relative_error": float(l2_rel.mean().item()),
        "sign_agreement": float(sign_agree.mean().item()),
        "topr_channel_recall": float(sum(channel_recalls) / len(channel_recalls)),
        "topr_channel_jaccard": float(sum(channel_jaccards) / len(channel_jaccards)),
        "topr_sign_agreement": float(sum(top_sign_agree_values) / len(top_sign_agree_values)),
        "signed_channel_recall": float(signed_recall),
        "signed_channel_jaccard": float(signed_jaccard),
        "channel_budget_r": int(r),
    }

    per_head = []
    if head_rows:
        cos_h = cos.mean(dim=0).detach().cpu().tolist()
        l2_h = l2_rel.mean(dim=0).detach().cpu().tolist()
        sign_h = sign_agree.mean(dim=0).detach().cpu().tolist()
        heads = int(q_true_flat.size(1))
        batch = int(q_true_flat.size(0))
        for head_idx in range(heads):
            start = head_idx
            head_channel_recall = sum(channel_recalls[start::heads]) / max(1, batch)
            head_channel_jaccard = sum(channel_jaccards[start::heads]) / max(1, batch)
            head_top_sign = sum(top_sign_agree_values[start::heads]) / max(1, batch)
            head_signed_recall, head_signed_jaccard = _mean_overlap_from_sets(
                signed_cand_sets[start::heads],
                signed_true_sets[start::heads],
            )
            item = dict(base)
            item.update(
                {
                    "head_idx": int(head_idx),
                    "cosine_similarity": float(cos_h[head_idx]),
                    "l2_relative_error": float(l2_h[head_idx]),
                    "sign_agreement": float(sign_h[head_idx]),
                    "topr_channel_recall": float(head_channel_recall),
                    "topr_channel_jaccard": float(head_channel_jaccard),
                    "topr_sign_agreement": float(head_top_sign),
                    "signed_channel_recall": float(head_signed_recall),
                    "signed_channel_jaccard": float(head_signed_jaccard),
                }
            )
            per_head.append(item)
    return base, per_head


def append_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def _read_rows(metrics_path: Path) -> list[dict[str, object]]:
    if not metrics_path.exists():
        return []
    rows = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _summarize(rows: list[dict[str, object]], keys: list[str]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], dict[str, object]] = defaultdict(
        lambda: {"count": 0, "sums": defaultdict(float), "metric_counts": defaultdict(int)}
    )
    for row in rows:
        group_key = tuple(row.get(k) for k in keys)
        group = groups[group_key]
        group["count"] += 1
        for metric in METRIC_KEYS:
            value = row.get(metric)
            if isinstance(value, (int, float)):
                group["sums"][metric] += float(value)
                group["metric_counts"][metric] += 1

    out = []
    for group_key, group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        count = int(group["count"])
        item = {key: value for key, value in zip(keys, group_key, strict=False)}
        item["count"] = count
        for metric in METRIC_KEYS:
            metric_count = int(group["metric_counts"][metric])
            item[metric] = group["sums"][metric] / metric_count if metric_count else None
        out.append(item)
    return out


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_probe_dir(output_dir: str | Path) -> None:
    output = Path(output_dir)
    rows = _read_rows(output / "stage2_probe_metrics.jsonl")
    head_rows_raw = _read_rows(output / "stage2_probe_head_metrics.jsonl")
    layer_rows = _summarize(rows, ["method", "task", "model_name", "token_budget", "layer_bucket"])
    layer_detail_rows = _summarize(rows, ["method", "task", "model_name", "token_budget", "layer_idx"])
    decode_rows = _summarize(rows, ["method", "task", "model_name", "token_budget", "decode_bucket"])
    overall_rows = _summarize(rows, ["method", "task", "model_name", "token_budget"])
    head_rows = _summarize(head_rows_raw, ["method", "task", "model_name", "token_budget", "layer_idx", "head_idx"])

    _write_csv(output / "stage2_probe_layer_summary.csv", layer_rows)
    _write_csv(output / "stage2_probe_layer_detail_summary.csv", layer_detail_rows)
    _write_csv(output / "stage2_probe_decode_summary.csv", decode_rows)
    _write_csv(output / "stage2_probe_summary.csv", overall_rows)
    _write_csv(output / "stage2_probe_head_summary.csv", head_rows)
    with (output / "stage2_probe_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "overall": overall_rows,
                "by_layer_bucket": layer_rows,
                "by_layer": layer_detail_rows,
                "by_decode_bucket": decode_rows,
                "by_layer_head": head_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
