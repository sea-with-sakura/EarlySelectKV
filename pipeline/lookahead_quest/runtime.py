from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from litgpt import LLM
from litgpt.model import apply_rope, apply_rope_interleave

from pipeline.quest.runtime import QuestConfig, QuestRuntime
from pipeline.runtime_decode_utils import run_full_cache_decode

from .lookahead_quest_monkey_patch import install_lookahead_quest_attention, uninstall_lookahead_quest_attention


@dataclass
class LookaheadQuestConfig(QuestConfig):
    lookahead_source: str = "mid"
    lookahead_mid_until_layer: int = 20
    use_real_q_fallback: bool = False


class LookaheadQuestRuntime(QuestRuntime):
    """Quest page routing with a one-layer-ahead route query."""

    cfg: LookaheadQuestConfig

    def __init__(self, cfg: LookaheadQuestConfig) -> None:
        super().__init__(cfg)
        if self.cfg.lookahead_source not in {"mid", "in", "in_mid"}:
            raise ValueError(f"Unsupported lookahead_source={self.cfg.lookahead_source}")
        if self.cfg.lookahead_mid_until_layer < 0:
            raise ValueError("lookahead_mid_until_layer must be non-negative.")
        self._blocks = None
        self.lookahead_q_by_layer: dict[int, torch.Tensor] = {}

    def attach_model(self, model: torch.nn.Module) -> None:
        self._blocks = list(model.transformer.h)

    def reset(self) -> None:
        self.summary_cache.clear()
        self.lookahead_q_by_layer.clear()

    def observe_layer_input(
        self,
        *,
        layer_idx: int,
        hidden_in: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> None:
        if hidden_in.size(1) != 1 or self._blocks is None:
            return
        if int(layer_idx) == 0:
            self.lookahead_q_by_layer.clear()

        next_layer_idx = int(layer_idx) + 1
        source = str(self.cfg.lookahead_source)
        if source == "in_mid":
            if next_layer_idx <= int(self.cfg.lookahead_mid_until_layer):
                return
        elif source != "in":
            return

        q = self._compute_next_layer_route_query(layer_idx=layer_idx, hidden=hidden_in, cos=cos, sin=sin)
        if q is not None:
            self.lookahead_q_by_layer[next_layer_idx] = q

    def observe_layer_mid(
        self,
        *,
        layer_idx: int,
        hidden_mid: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> None:
        if hidden_mid.size(1) != 1 or self._blocks is None:
            return
        if int(layer_idx) == 0 and self.cfg.lookahead_source == "mid":
            self.lookahead_q_by_layer.clear()

        next_layer_idx = int(layer_idx) + 1
        source = str(self.cfg.lookahead_source)
        if source == "in_mid":
            if next_layer_idx > int(self.cfg.lookahead_mid_until_layer):
                return
        elif source != "mid":
            return

        q = self._compute_next_layer_route_query(layer_idx=layer_idx, hidden=hidden_mid, cos=cos, sin=sin)
        if q is not None:
            self.lookahead_q_by_layer[next_layer_idx] = q

    def _compute_next_layer_route_query(
        self,
        *,
        layer_idx: int,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        next_idx = int(layer_idx) + 1
        if hidden.size(1) != 1 or self._blocks is None or next_idx >= len(self._blocks):
            return None

        next_block = self._blocks[next_idx]
        next_attn = next_block.attn
        hidden_normed = next_block.norm_1(hidden)
        qkv = next_attn.qkv(hidden_normed)
        query_size = next_attn.config.n_head * next_attn.config.head_size
        q = qkv[..., :query_size]

        if next_attn.config.norm_qk and next_attn.config.norm_qk_type == "olmo2":
            q = next_attn.norm_q(q)

        q = q.view(hidden.size(0), hidden.size(1), next_attn.config.n_head, next_attn.config.head_size)
        q = q.transpose(1, 2)

        if next_attn.config.norm_qk and next_attn.config.norm_qk_type == "default":
            q = next_attn.norm_q(q)

        rope_n_elem = next_attn.config.rope_n_elem
        if next_attn.config.rope_interleave:
            q_roped = apply_rope_interleave(q[..., :rope_n_elem], cos, sin)
        else:
            q_roped = apply_rope(q[..., :rope_n_elem], cos, sin)
        return torch.cat((q_roped, q[..., rope_n_elem:]), dim=-1).detach()

    def select_attention_indices(self, *, layer_idx: int, q: torch.Tensor, k: torch.Tensor):
        routing_q = self.lookahead_q_by_layer.get(int(layer_idx))
        if routing_q is None:
            if bool(self.cfg.use_real_q_fallback):
                routing_q = q
            else:
                return None
        routing_q = routing_q.to(device=q.device, dtype=q.dtype)
        return super().select_attention_indices(layer_idx=layer_idx, q=routing_q, k=k)


def decode_lookahead_quest(
    *,
    prompt_ids: torch.Tensor,
    model: LLM,
    runtime: LookaheadQuestRuntime,
    max_new_tokens: int,
) -> str:
    runtime.attach_model(model.model)
    runtime.reset()
    return run_full_cache_decode(
        prompt_ids=prompt_ids,
        model=model,
        max_new_tokens=max_new_tokens,
        attention_state=runtime,
        install_attention=install_lookahead_quest_attention,
        uninstall_attention=uninstall_lookahead_quest_attention,
    )
