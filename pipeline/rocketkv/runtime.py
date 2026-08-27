import math
from dataclasses import dataclass
from typing import Optional

import torch

from litgpt import LLM

from pipeline.auxiliary_kv_cache import (
    ChunkSummaryCache,
    build_chunk_summary_values,
    select_hsa_indices_from_summary,
)
from pipeline.runtime_decode_utils import (
    compress_standard_prompt_kv,
    compute_chunk_size,
    run_sparse_decode,
    setup_prompt_kv_cache,
    sorted_topk_indices,
)
from .rocket_monkey_patch import install_rocketkv_attention, uninstall_rocketkv_attention


@dataclass
class RocketKVConfig:
    prompt_len: int
    prompt_budget: int
    topk_budget: int
    compression_ratio: float
    window_size: int
    kernel_size: int
    skip_layers: int
    attention_heads: int
    kv_heads: int
    routing_mode: str = "hsa"


class RocketKVRuntime:
    """Runtime state for LitGPT RocketKV-family decoding."""

    _sorted_topk = staticmethod(sorted_topk_indices)

    def __init__(self, cfg: RocketKVConfig) -> None:
        self.cfg = cfg
        if self.cfg.skip_layers < 0:
            raise ValueError("skip_layers must be non-negative.")
        if self.cfg.attention_heads % self.cfg.kv_heads != 0:
            raise ValueError(
                f"attention_heads={self.cfg.attention_heads} must be divisible by kv_heads={self.cfg.kv_heads}"
            )
        if self.cfg.routing_mode not in {"hsa", "exact_topk"}:
            raise ValueError(f"Unsupported routing_mode={self.cfg.routing_mode}")
        self.active_prompt_len = min(int(self.cfg.prompt_len), int(self.cfg.prompt_budget))
        self._compressed_layers: set[int] = set()
        self.prompt_k_by_layer: dict[int, torch.Tensor] = {}
        self.prompt_v_by_layer: dict[int, torch.Tensor] = {}
        self.chunk_size = compute_chunk_size(self.cfg.compression_ratio)
        self.aux_cache = ChunkSummaryCache(chunk_size=self.chunk_size)

    @property
    def uses_aux_cache(self) -> bool:
        return self.cfg.routing_mode == "hsa" and self.chunk_size >= 2

    def reset(self) -> None:
        self._compressed_layers.clear()
        self.prompt_k_by_layer.clear()
        self.prompt_v_by_layer.clear()
        self.aux_cache.clear()

    def observe_layer_hidden(self, **_: object) -> None:
        return

    def is_decode_call(self, *, input_pos: Optional[torch.Tensor], time_steps: int) -> bool:
        if time_steps != 1 or input_pos is None:
            return False
        pos = input_pos if input_pos.dim() == 1 else input_pos[0]
        if pos.numel() != 1:
            return False
        logical_pos = int(pos.item())
        return logical_pos >= int(self.cfg.prompt_len)

    def maybe_record_decode_step(self, *, layer_idx: int, input_pos: Optional[torch.Tensor], time_steps: int) -> bool:
        is_decode = self.is_decode_call(input_pos=input_pos, time_steps=time_steps)
        return is_decode

    def maybe_compress_prefill(
        self,
        *,
        layer_idx: int,
        attn_module: torch.nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        input_pos_maxp1: Optional[int],
    ) -> None:
        if layer_idx in self._compressed_layers:
            return

        key_len = int(input_pos_maxp1) if input_pos_maxp1 is not None else int(k.size(2))
        if key_len != int(self.cfg.prompt_len):
            return

        del attn_module
        self._compressed_layers.add(layer_idx)
        compressed = compress_standard_prompt_kv(
            q=q,
            k=k,
            v=v,
            key_len=key_len,
            active_prompt_len=self.active_prompt_len,
            window_size=self.cfg.window_size,
            kernel_size=self.cfg.kernel_size,
            attention_heads=self.cfg.attention_heads,
            kv_heads=self.cfg.kv_heads,
        )
        self.prompt_k_by_layer[layer_idx] = compressed.prompt_k
        self.prompt_v_by_layer[layer_idx] = compressed.prompt_v

    def setup_decode_kv_cache(
        self,
        *,
        gpt: torch.nn.Module,
        logical_max_seq_length: int,
        max_new_tokens: int,
        device: torch.device,
    ) -> None:
        setup_prompt_kv_cache(
            gpt=gpt,
            prompt_k_by_layer=self.prompt_k_by_layer,
            prompt_v_by_layer=self.prompt_v_by_layer,
            active_prompt_len=self.active_prompt_len,
            logical_max_seq_length=logical_max_seq_length,
            max_new_tokens=max_new_tokens,
            device=device,
            runtime_name="RocketKV",
        )
        if self.uses_aux_cache:
            self.aux_cache.clear()
            capacity_len = self.active_prompt_len + int(max_new_tokens)
            for layer_idx, block in enumerate(gpt.transformer.h):
                cache = block.attn.kv_cache
                if cache is None:
                    raise RuntimeError("KV cache is not initialized.")
                prompt_k = cache.k[:1, :, : self.active_prompt_len, :]
                self.aux_cache.build_layer(layer_idx, prompt_k, capacity_len=capacity_len)
        else:
            self.aux_cache.clear()
        self.prompt_k_by_layer.clear()
        self.prompt_v_by_layer.clear()

    def update_aux_cache(
        self,
        *,
        layer_idx: int,
        cache_input_pos: torch.Tensor,
        k_update: torch.Tensor,
    ) -> None:
        if not self.uses_aux_cache:
            return
        self.aux_cache.update_layer(
            layer_idx,
            cache_input_pos=cache_input_pos,
            k_update=k_update,
        )

    def cache_position_for_input(self, input_pos: torch.Tensor) -> torch.Tensor:
        if input_pos.dim() == 1:
            pos = input_pos.to(dtype=torch.long)
        else:
            pos = input_pos[0].to(dtype=torch.long)
        if pos.numel() == 0:
            return pos
        if pos.numel() > 1:
            return pos
        logical_pos = int(pos.item())
        if logical_pos < self.cfg.prompt_len:
            return pos
        compact_pos = self.active_prompt_len + (logical_pos - self.cfg.prompt_len)
        return torch.tensor([compact_pos], device=pos.device, dtype=torch.long)

    def select_attention_indices(
        self,
        *,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if layer_idx < self.cfg.skip_layers:
            return None
        len_q = int(q.size(2))
        key_len = int(k.size(2))
        if len_q > 1:
            return None
        if len_q <= 0 or key_len <= 0 or self.cfg.topk_budget >= key_len:
            return None

        if self.cfg.routing_mode == "exact_topk":
            return self._select_exact_topk_indices(q=q, k=k)
        return self._select_hsa_indices(layer_idx=layer_idx, q=q, k=k)

    def _select_exact_topk_indices(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        key_len = int(k.size(2))
        q_grouped = q.view(q.size(0), self.cfg.kv_heads, -1, q.size(2), q.size(-1)).contiguous()
        k_grouped = k.unsqueeze(2)

        scores = torch.matmul(q_grouped, k_grouped.transpose(-1, -2)) / math.sqrt(q.size(-1))
        probs = torch.softmax(scores, dim=-1)
        mean_scores = probs.mean(dim=2)

        k_eff = min(int(self.cfg.topk_budget), key_len)
        topk_idx = self._sorted_topk(mean_scores, k_eff)
        return topk_idx

    def _select_hsa_indices(
        self,
        *,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        key_len = int(k.size(2))
        summary = self.aux_cache.get(layer_idx, key_len=key_len) if self.uses_aux_cache else None
        if summary is None:
            summary = build_chunk_summary_values(
                k,
                chunk_size=self.chunk_size,
                capacity_len=key_len,
            )
        return select_hsa_indices_from_summary(
            q,
            summary,
            key_len=key_len,
            chunk_size=self.chunk_size,
            compression_ratio=self.cfg.compression_ratio,
            topk_budget=self.cfg.topk_budget,
            kv_heads=self.cfg.kv_heads,
            sorted_topk=False,
        )


def decode_rocketkv(
    *,
    prompt_ids: torch.Tensor,
    model: LLM,
    runtime: RocketKVRuntime,
    max_new_tokens: int,
) -> str:
    return run_sparse_decode(
        prompt_ids=prompt_ids,
        model=model,
        runtime=runtime,
        max_new_tokens=max_new_tokens,
        install_attention=install_rocketkv_attention,
        uninstall_attention=uninstall_rocketkv_attention,
    )
