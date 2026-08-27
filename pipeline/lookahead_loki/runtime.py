from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from litgpt import LLM
from litgpt.model import apply_rope, apply_rope_interleave

from pipeline.loki.runtime import LokiConfig, LokiRuntime
from pipeline.runtime_decode_utils import run_full_cache_decode

from .lookahead_loki_monkey_patch import install_lookahead_loki_attention, uninstall_lookahead_loki_attention


@dataclass
class LookaheadLokiConfig(LokiConfig):
    lookahead_source: str = "mid"
    lookahead_mid_until_layer: int = 20
    use_real_q_fallback: bool = False


class LookaheadLokiRuntime(LokiRuntime):
    """Loki route with a query predicted one layer ahead.

    The route query is computed from the previous block's hidden state using the
    next layer's norm/Wq, then RoPE is applied when that next layer runs. The
    attention output still uses the current layer's real q and selected K/V.
    """

    cfg: LookaheadLokiConfig

    def __init__(self, cfg: LookaheadLokiConfig) -> None:
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
        super().reset()
        self.lookahead_q_by_layer.clear()

    def observe_layer_input(self, *, layer_idx: int, hidden_in: torch.Tensor) -> None:
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

        q = self._compute_next_layer_query(layer_idx=int(layer_idx), hidden=hidden_in)
        if q is not None:
            self.lookahead_q_by_layer[next_layer_idx] = q

    def observe_layer_mid(self, *, layer_idx: int, hidden_mid: torch.Tensor) -> None:
        if hidden_mid.size(1) != 1 or self._blocks is None:
            return

        next_layer_idx = int(layer_idx) + 1
        source = str(self.cfg.lookahead_source)
        if source == "in_mid":
            if next_layer_idx > int(self.cfg.lookahead_mid_until_layer):
                return
        elif source != "mid":
            return

        q = self._compute_next_layer_query(layer_idx=int(layer_idx), hidden=hidden_mid)
        if q is not None:
            self.lookahead_q_by_layer[next_layer_idx] = q

    def _compute_next_layer_query(self, *, layer_idx: int, hidden: torch.Tensor) -> Optional[torch.Tensor]:
        if hidden.size(1) != 1 or self._blocks is None:
            return None
        next_idx = int(layer_idx) + 1
        if next_idx >= len(self._blocks):
            return None

        next_block = self._blocks[next_idx]
        hidden_normed = next_block.norm_1(hidden)
        next_attn = next_block.attn
        qkv = next_attn.qkv(hidden_normed)
        query_size = next_attn.config.n_head * next_attn.config.head_size
        q = qkv[..., :query_size]

        if next_attn.config.norm_qk and next_attn.config.norm_qk_type == "olmo2":
            q = next_attn.norm_q(q)

        q = q.view(hidden.size(0), hidden.size(1), next_attn.config.n_head, next_attn.config.head_size)
        q = q.transpose(1, 2)

        if next_attn.config.norm_qk and next_attn.config.norm_qk_type == "default":
            q = next_attn.norm_q(q)

        return q.detach()

    def _get_routing_query(
        self,
        *,
        layer_idx: int,
        real_q: torch.Tensor,
        real_q_pre: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        rope_n_elem: int,
        rope_interleave: bool,
    ) -> Optional[torch.Tensor]:
        lookahead_q = self.lookahead_q_by_layer.get(int(layer_idx))
        if lookahead_q is None:
            if not bool(self.cfg.use_real_q_fallback):
                return None
            if self.uses_prerotary_route() and real_q_pre is not None:
                return real_q_pre
            return real_q

        lookahead_q = lookahead_q.to(device=real_q.device, dtype=real_q.dtype)
        if self.uses_prerotary_route():
            return lookahead_q
        if rope_interleave:
            q_roped = apply_rope_interleave(lookahead_q[..., :rope_n_elem], cos, sin)
        else:
            q_roped = apply_rope(lookahead_q[..., :rope_n_elem], cos, sin)
        return torch.cat((q_roped, lookahead_q[..., rope_n_elem:]), dim=-1)

    def compute_attention_output(
        self,
        *,
        attn_self: object,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rope_n_elem: int,
        rope_interleave: bool,
        q_pre: Optional[torch.Tensor] = None,
        route_k: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if int(layer_idx) < int(self.cfg.skip_layers):
            return None
        if q.size(2) != 1:
            return None

        key_len = int(k.size(2))
        top_k = min(int(self.cfg.token_budget), key_len)
        if key_len <= 0 or top_k >= key_len:
            return None

        routing_q = self._get_routing_query(
            layer_idx=int(layer_idx),
            real_q=q,
            real_q_pre=q_pre,
            cos=cos,
            sin=sin,
            rope_n_elem=int(rope_n_elem),
            rope_interleave=bool(rope_interleave),
        )
        if routing_q is None:
            return None

        route_k = k if route_k is None else route_k
        if int(route_k.size(2)) < key_len:
            return None

        route_scores = self._approx_route_scores(
            attn_self=attn_self,
            layer_idx=int(layer_idx),
            q=routing_q,
            k=route_k[:, :, :key_len, :],
        )
        indices = self._sorted_topk(route_scores, top_k)

        if indices.size(1) == int(self.cfg.kv_heads):
            return self._grouped_exact_output(
                attn_self=attn_self,
                q=q,
                k=k,
                v=v,
                indices=indices,
            )
        return self._per_head_exact_output(
            attn_self=attn_self,
            q=q,
            k=k,
            v=v,
            indices=indices,
        )


def decode_lookahead_loki(
    *,
    prompt_ids: torch.Tensor,
    model: LLM,
    runtime: LookaheadLokiRuntime,
    max_new_tokens: int,
) -> str:
    runtime.reset()
    return run_full_cache_decode(
        prompt_ids=prompt_ids,
        model=model,
        max_new_tokens=max_new_tokens,
        attention_state=runtime,
        install_attention=install_lookahead_loki_attention,
        uninstall_attention=uninstall_lookahead_loki_attention,
    )
