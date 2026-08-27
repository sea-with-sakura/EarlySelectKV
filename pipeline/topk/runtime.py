import math
from dataclasses import dataclass
from typing import Optional

import torch

from litgpt import LLM

from pipeline.runtime_decode_utils import run_full_cache_decode, sorted_topk_indices
from .topk_monkey_patch import install_topk_attention, uninstall_topk_attention


@dataclass
class TopKConfig:
    prompt_len: int
    topk_budget: int
    group_shared_within_kv_group: bool
    skip_layers: int
    attention_heads: int
    kv_heads: int
    attention_scores_scalar: Optional[int]
    head_size: int


@dataclass
class TopKSelection:
    indices: torch.Tensor


class TopKRuntime:
    """Decode-time exact top-k attention on the full KV cache."""

    _sorted_topk = staticmethod(sorted_topk_indices)

    def __init__(self, cfg: TopKConfig) -> None:
        self.cfg = cfg
        if self.cfg.skip_layers < 0:
            raise ValueError("skip_layers must be non-negative.")
        if self.cfg.attention_heads % self.cfg.kv_heads != 0:
            raise ValueError(
                f"attention_heads={self.cfg.attention_heads} must be divisible by kv_heads={self.cfg.kv_heads}"
            )

    def observe_layer_hidden(self, **_: object) -> None:
        return

    def cache_position_for_input(self, input_pos: torch.Tensor) -> torch.Tensor:
        return input_pos

    def select_attention_indices(
        self,
        *,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Optional[TopKSelection]:
        if layer_idx < self.cfg.skip_layers:
            return None
        if q.size(2) != 1:
            return None
        key_len = int(k.size(2))
        if key_len <= 0 or self.cfg.topk_budget >= key_len:
            return None

        scale = 1.0 / math.sqrt(self.cfg.attention_scores_scalar or self.cfg.head_size)
        if q.size(1) == k.size(1):
            scores = q @ k.mT
            scores = scores * scale
            topk_idx = self._sorted_topk(scores, min(int(self.cfg.topk_budget), key_len))
            return TopKSelection(indices=topk_idx)

        q_per_kv = self.cfg.attention_heads // self.cfg.kv_heads
        q_grouped = q.reshape(q.size(0), self.cfg.kv_heads, q_per_kv, q.size(2), q.size(3))
        scores = torch.einsum("bgqtd,bgkd->bgqtk", q_grouped, k) * scale

        if self.cfg.group_shared_within_kv_group:
            scores = torch.softmax(scores, dim=-1).mean(dim=2)
        else:
            scores = scores.reshape(q.size(0), q.size(1), q.size(2), key_len)

        topk_idx = self._sorted_topk(scores, min(int(self.cfg.topk_budget), key_len))
        return TopKSelection(indices=topk_idx)


def decode_topk(
    *,
    prompt_ids: torch.Tensor,
    model: LLM,
    runtime: TopKRuntime,
    max_new_tokens: int,
) -> str:
    return run_full_cache_decode(
        prompt_ids=prompt_ids,
        model=model,
        max_new_tokens=max_new_tokens,
        attention_state=runtime,
        install_attention=install_topk_attention,
        uninstall_attention=uninstall_topk_attention,
    )
