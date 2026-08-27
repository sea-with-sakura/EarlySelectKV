from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from litgpt import LLM
from litgpt.generate.base import sample

from pipeline.auxiliary_kv_cache import ChunkSummaryCache, build_chunk_summary_values
from pipeline.litgpt_prefill import decode_next_token_logits, prefill_last_token_logits, setup_standard_kv_cache
from pipeline.runtime_decode_utils import group_queries_by_kv
from .quest_monkey_patch import install_quest_attention, uninstall_quest_attention


@dataclass
class QuestConfig:
    prompt_len: int
    token_budget: int
    page_size: int
    skip_layers: int
    attention_heads: int
    kv_heads: int
    min_select_pages: int = 1
    include_last_page: bool = True
    group_shared_within_kv_group: bool = True


@dataclass
class QuestSelection:
    indices: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None


class QuestRuntime:
    """Quest query-aware page selection over LitGPT's standard KV cache."""

    def __init__(self, cfg: QuestConfig) -> None:
        self.cfg = cfg
        if self.cfg.skip_layers < 0:
            raise ValueError("skip_layers must be non-negative.")
        if self.cfg.page_size <= 0:
            raise ValueError("page_size must be positive.")
        if self.cfg.token_budget <= 0:
            raise ValueError("token_budget must be positive.")
        if self.cfg.attention_heads % self.cfg.kv_heads != 0:
            raise ValueError(
                f"attention_heads={self.cfg.attention_heads} must be divisible by kv_heads={self.cfg.kv_heads}"
            )
        self.summary_cache = ChunkSummaryCache(chunk_size=int(self.cfg.page_size))

    def observe_layer_hidden(self, **_: object) -> None:
        return

    def cache_position_for_input(self, input_pos: torch.Tensor) -> torch.Tensor:
        return input_pos

    def prepare_summary_cache(
        self,
        *,
        layer_idx: int,
        k_cache: torch.Tensor,
        key_len: int,
        cache_input_pos: torch.Tensor,
        k_update: torch.Tensor,
    ) -> None:
        if int(layer_idx) < int(self.cfg.skip_layers):
            return
        state = self.summary_cache.by_layer.get(int(layer_idx))
        capacity_len = int(k_cache.size(2))
        key_len = min(max(int(key_len), 0), capacity_len)
        if state is None or state.capacity_len < capacity_len:
            self.summary_cache.build_layer(
                int(layer_idx),
                k_cache[:, :, :key_len, :].contiguous(),
                capacity_len=capacity_len,
            )
            return
        self.summary_cache.update_layer(
            int(layer_idx),
            cache_input_pos=cache_input_pos,
            k_update=k_update,
        )

    def select_attention_indices(
        self,
        *,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Optional[QuestSelection]:
        if layer_idx < int(self.cfg.skip_layers):
            return None
        if q.size(2) != 1:
            return None

        key_len = int(k.size(2))
        if key_len <= 0 or int(self.cfg.token_budget) >= key_len:
            return None

        num_pages = math.ceil(key_len / int(self.cfg.page_size))
        page_budget = self._resolve_page_budget(num_pages)
        if page_budget >= num_pages:
            return None

        if bool(self.cfg.include_last_page):
            indices, attention_mask = self._select_with_last_page(
                layer_idx=layer_idx,
                q=q,
                k=k,
                key_len=key_len,
                page_budget=page_budget,
            )
        else:
            indices, attention_mask = self._select_pages(
                layer_idx=layer_idx,
                q=q,
                k=k,
                key_len=key_len,
                page_budget=page_budget,
            )

        if indices.size(-1) <= 0 or indices.size(-1) >= key_len:
            return None
        return QuestSelection(indices=indices, attention_mask=attention_mask)

    def _resolve_page_budget(self, num_pages: int) -> int:
        page_budget = int(self.cfg.token_budget) // int(self.cfg.page_size)
        page_budget = max(page_budget, int(self.cfg.min_select_pages), 1)
        return min(page_budget, int(num_pages))

    def _select_with_last_page(
        self,
        *,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        key_len: int,
        page_budget: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        page_size = int(self.cfg.page_size)
        last_page_start = ((int(key_len) - 1) // page_size) * page_size
        route_len = int(last_page_start)
        route_pages = route_len // page_size
        route_budget = min(max(int(page_budget) - 1, 0), route_pages)

        if route_budget > 0:
            route_idx, _ = self._select_pages(
                layer_idx=layer_idx,
                q=q,
                k=k[..., :route_len, :],
                key_len=route_len,
                page_budget=route_budget,
            )
        else:
            route_heads = int(self.cfg.kv_heads) if bool(self.cfg.group_shared_within_kv_group) else int(self.cfg.attention_heads)
            route_idx = torch.empty(
                q.size(0),
                route_heads,
                q.size(2),
                0,
                dtype=torch.long,
                device=q.device,
            )

        last_idx = torch.arange(last_page_start, key_len, device=q.device, dtype=torch.long)
        last_idx = last_idx.view(1, 1, 1, -1).expand(q.size(0), route_idx.size(1), q.size(2), -1)
        return torch.cat((route_idx, last_idx), dim=-1), None

    def _select_pages(
        self,
        *,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        key_len: int,
        page_budget: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        page_size = int(self.cfg.page_size)
        key_len = int(key_len)
        if key_len <= 0 or page_budget <= 0:
            route_heads = int(self.cfg.kv_heads) if bool(self.cfg.group_shared_within_kv_group) else int(self.cfg.attention_heads)
            empty = torch.empty(
                q.size(0),
                route_heads,
                q.size(2),
                0,
                dtype=torch.long,
                device=q.device,
            )
            return empty, None

        scores = self._quest_page_scores(layer_idx=layer_idx, q=q, k=k, key_len=key_len)
        page_count = int(scores.size(-1))
        page_budget = min(int(page_budget), page_count)
        page_idx = torch.topk(scores, k=page_budget, dim=-1, sorted=False).indices.sort(dim=-1).values
        offsets = torch.arange(page_size, device=q.device, dtype=torch.long)
        token_idx = page_idx.unsqueeze(-1) * page_size + offsets.view(*((1,) * page_idx.ndim), page_size)
        token_idx = token_idx.flatten(start_dim=-2)
        token_idx = token_idx[..., : page_budget * page_size]
        valid_mask = token_idx < int(key_len)
        if not bool(valid_mask.all()):
            token_idx = token_idx.clamp_max(max(int(key_len) - 1, 0))
            return token_idx, valid_mask
        return token_idx, None

    def _quest_page_scores(
        self,
        *,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        key_len: int,
    ) -> torch.Tensor:
        summary = self.summary_cache.get(int(layer_idx), key_len=int(key_len))
        if summary is None:
            summary = build_chunk_summary_values(
                k,
                chunk_size=int(self.cfg.page_size),
                capacity_len=int(key_len),
            )

        head_dim = int(q.size(-1))
        if summary.size(-1) == head_dim * 2:
            page_min, page_max = summary.split(head_dim, dim=-1)
        else:
            page_min = page_max = summary

        q_grouped = group_queries_by_kv(
            q,
            attention_heads=int(self.cfg.attention_heads),
            kv_heads=int(self.cfg.kv_heads),
        )
        page_bound = torch.where(
            q_grouped.unsqueeze(4) >= 0,
            page_max.unsqueeze(2).unsqueeze(3),
            page_min.unsqueeze(2).unsqueeze(3),
        )
        head_scores = (
            q_grouped.unsqueeze(4).to(dtype=torch.float32) * page_bound.to(dtype=torch.float32)
        ).sum(dim=-1)
        if bool(self.cfg.group_shared_within_kv_group):
            return torch.softmax(head_scores, dim=-1, dtype=torch.float32).mean(dim=2)
        return head_scores.reshape(q.size(0), int(self.cfg.attention_heads), q.size(2), -1)


def decode_quest(
    *,
    prompt_ids: torch.Tensor,
    model: LLM,
    runtime: QuestRuntime,
    max_new_tokens: int,
) -> str:
    gpt = model.model
    prompt_ids = prompt_ids.to(torch.long)
    prompt_len = int(prompt_ids.numel())
    max_new_tokens = int(max_new_tokens)
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")

    max_returned_tokens = prompt_len + max_new_tokens
    if max_returned_tokens > model.config.block_size:
        raise ValueError(
            f"Requested total length {max_returned_tokens} exceeds model context {model.config.block_size}"
        )
    if max_new_tokens == 0:
        return ""

    device = prompt_ids.device
    gpt.clear_kv_cache()
    setup_standard_kv_cache(
        gpt,
        logical_max_seq_length=max_returned_tokens,
        cache_max_seq_length=max_returned_tokens,
        device=device,
        build_mask=False,
    )
    gpt.eval()

    generated = torch.empty(max_new_tokens, dtype=torch.long, device=device)
    generated_count = 0
    eos_id = getattr(model.tokenizer, "eos_id", None)

    install_quest_attention(gpt, runtime)
    try:
        with torch.inference_mode():
            prefill_tokens = prompt_ids.view(1, -1)
            prefill_pos = torch.arange(0, prompt_len, device=device, dtype=torch.int64)
            prefill_logits = prefill_last_token_logits(
                gpt,
                prefill_tokens,
                prefill_pos,
                input_pos_maxp1=prompt_len,
            )

            token = sample(prefill_logits, temperature=0.0, top_p=0.0, top_k=None).to(dtype=torch.int64).view(1, 1)
            generated[generated_count] = token.view(-1)[0]
            generated_count += 1

            input_pos = torch.tensor([prompt_len], device=device, dtype=torch.int64)
            input_pos_maxp1 = prompt_len + 1
            for _ in range(max_new_tokens - 1):
                if eos_id is not None and int(token.item()) == int(eos_id):
                    break
                logits = decode_next_token_logits(gpt, token, input_pos, input_pos_maxp1=input_pos_maxp1)
                token = sample(logits, temperature=0.0, top_p=0.0, top_k=None).to(dtype=torch.int64).view(1, 1)
                generated[generated_count] = token.view(-1)[0]
                generated_count += 1
                input_pos.add_(1)
                input_pos_maxp1 += 1
    finally:
        uninstall_quest_attention(gpt)

    generated_ids = generated[:generated_count]
    return model.preprocessor.decode(generated_ids) if generated_ids.numel() > 0 else ""
