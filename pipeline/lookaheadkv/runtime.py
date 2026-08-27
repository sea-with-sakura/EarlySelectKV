import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from litgpt import LLM
from litgpt.model import apply_rope

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
from .lookahead_monkey_patch import install_lookaheadkv_attention, uninstall_lookaheadkv_attention


DEFAULT_DECODE_LOCAL_WINDOW_SIZE = 32
_SVD_WQ_ARTIFACT_CACHE: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
_SVD_WQ_DEVICE_CACHE: dict[tuple[str, str, torch.dtype], dict[int, dict[str, torch.Tensor]]] = {}


@dataclass
class LookaheadKVConfig:
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
    decode_local_budget: Optional[int] = None
    lookahead_source: str = "mid"
    lookahead_mid_until_layer: int = 20
    lookahead_query_mode: str = "next_wq"
    lookahead_svd_path: Optional[str] = None


class LookaheadKVRuntime:
    """Runtime state for LitGPT EarlySelectKV decoding."""

    _sorted_topk = staticmethod(sorted_topk_indices)

    def __init__(self, cfg: LookaheadKVConfig) -> None:
        self.cfg = cfg
        if self.cfg.skip_layers < 0:
            raise ValueError("skip_layers must be non-negative.")
        if self.cfg.attention_heads % self.cfg.kv_heads != 0:
            raise ValueError(
                f"attention_heads={self.cfg.attention_heads} must be divisible by kv_heads={self.cfg.kv_heads}"
            )
        if self.cfg.routing_mode not in {"hsa", "chunk1_local", "hsa_local", "exact_topk"}:
            raise ValueError(f"Unsupported routing_mode={self.cfg.routing_mode}")
        if self.cfg.lookahead_source not in {"mid", "in", "in_mid"}:
            raise ValueError(f"Unsupported lookahead_source={self.cfg.lookahead_source}")
        if self.cfg.lookahead_query_mode not in {"next_wq", "svd_wq"}:
            raise ValueError(f"Unsupported lookahead_query_mode={self.cfg.lookahead_query_mode}")
        if self.cfg.lookahead_query_mode == "svd_wq" and not self.cfg.lookahead_svd_path:
            raise ValueError("lookahead_svd_path is required when lookahead_query_mode='svd_wq'.")
        if self.cfg.lookahead_mid_until_layer < 0:
            raise ValueError("lookahead_mid_until_layer must be non-negative.")
        self.active_prompt_len = min(int(self.cfg.prompt_len), int(self.cfg.prompt_budget))
        self._compressed_layers: set[int] = set()
        self._blocks = None
        self.lookahead_q_by_layer: dict[int, torch.Tensor] = {}
        self.prompt_k_by_layer: dict[int, torch.Tensor] = {}
        self.prompt_v_by_layer: dict[int, torch.Tensor] = {}
        self.chunk_size = compute_chunk_size(self.cfg.compression_ratio)
        self.aux_cache = ChunkSummaryCache(chunk_size=self.chunk_size)

    def _get_svd_wq_factors(
        self,
        *,
        layer_idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[dict[str, torch.Tensor]]:
        path = self.cfg.lookahead_svd_path
        if not path:
            return None
        resolved = str(Path(path).resolve())
        if resolved not in _SVD_WQ_ARTIFACT_CACHE:
            artifact = torch.load(resolved, map_location="cpu", weights_only=True)
            layers = artifact.get("layers", artifact)
            _SVD_WQ_ARTIFACT_CACHE[resolved] = {int(k): v for k, v in layers.items()}
        cache_key = (resolved, str(device), dtype)
        if cache_key not in _SVD_WQ_DEVICE_CACHE:
            _SVD_WQ_DEVICE_CACHE[cache_key] = {
                int(layer): {
                    "down": factors["down"].to(device=device, dtype=dtype, non_blocking=True),
                    "up": factors["up"].to(device=device, dtype=dtype, non_blocking=True),
                }
                for layer, factors in _SVD_WQ_ARTIFACT_CACHE[resolved].items()
            }
        return _SVD_WQ_DEVICE_CACHE[cache_key].get(int(layer_idx))

    @property
    def uses_aux_cache(self) -> bool:
        return self.cfg.routing_mode in {"hsa", "hsa_local"} and self.chunk_size >= 2

    @property
    def uses_local_route(self) -> bool:
        return self.cfg.routing_mode in {"chunk1_local", "hsa_local"}

    def attach_model(self, model: torch.nn.Module) -> None:
        self._blocks = list(model.transformer.h)

    def reset(self) -> None:
        self._compressed_layers.clear()
        self.lookahead_q_by_layer.clear()
        self.prompt_k_by_layer.clear()
        self.prompt_v_by_layer.clear()
        self.aux_cache.clear()

    def observe_layer_hidden(self, **_: object) -> None:
        return

    def observe_layer_input(self, *, layer_idx: int, hidden_in: torch.Tensor) -> None:
        if hidden_in.size(1) != 1 or self._blocks is None:
            return
        if layer_idx == 0:
            self.lookahead_q_by_layer.clear()

        source = str(self.cfg.lookahead_source)
        next_layer_idx = int(layer_idx) + 1
        if source == "in_mid":
            if next_layer_idx <= int(self.cfg.lookahead_mid_until_layer):
                return
        elif source != "in":
            return

        q = self._compute_next_layer_query(layer_idx=layer_idx, hidden=hidden_in)
        if q is not None:
            self.lookahead_q_by_layer[next_layer_idx] = q

    def _compute_next_layer_query(self, *, layer_idx: int, hidden: torch.Tensor) -> Optional[torch.Tensor]:
        if hidden.size(1) != 1 or self._blocks is None:
            return
        next_idx = layer_idx + 1
        if next_idx >= len(self._blocks):
            return

        next_block = self._blocks[next_idx]
        next_attn = next_block.attn
        hidden_normed = next_block.norm_1(hidden)
        if self.cfg.lookahead_query_mode == "svd_wq":
            factors = self._get_svd_wq_factors(
                layer_idx=next_idx,
                device=hidden_normed.device,
                dtype=hidden_normed.dtype,
            )
            if factors is None:
                return None
            q = F.linear(F.linear(hidden_normed, factors["down"]), factors["up"])
        else:
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

    def observe_layer_mid(self, *, layer_idx: int, hidden_mid: torch.Tensor) -> None:
        if hidden_mid.size(1) != 1 or self._blocks is None:
            return
        source = str(self.cfg.lookahead_source)
        if layer_idx == 0 and source == "mid":
            self.lookahead_q_by_layer.clear()

        next_layer_idx = int(layer_idx) + 1
        if source == "in_mid":
            if next_layer_idx > int(self.cfg.lookahead_mid_until_layer):
                return
        elif source != "mid":
            return

        q = self._compute_next_layer_query(layer_idx=layer_idx, hidden=hidden_mid)
        if q is not None:
            self.lookahead_q_by_layer[next_layer_idx] = q

    def observe_layer_output(
        self,
        *,
        layer_idx: int,
        hidden_in: torch.Tensor,
        hidden_mid: torch.Tensor,
        hidden_out: torch.Tensor,
    ) -> None:
        del layer_idx, hidden_in, hidden_mid, hidden_out

    def _resolve_decode_local_budget(self, key_len: int) -> int:
        if not self.uses_local_route:
            return 0
        topk_budget = min(int(self.cfg.topk_budget), int(key_len))
        if topk_budget <= 0:
            return 0
        if self.cfg.decode_local_budget is None or int(self.cfg.decode_local_budget) < 0:
            local_budget = min(DEFAULT_DECODE_LOCAL_WINDOW_SIZE, max(16, topk_budget // 8))
        else:
            local_budget = int(self.cfg.decode_local_budget)
        return max(0, min(local_budget, topk_budget, int(key_len)))

    def is_decode_call(self, *, input_pos: Optional[torch.Tensor], time_steps: int) -> bool:
        if time_steps != 1 or input_pos is None:
            return False
        pos = input_pos if input_pos.dim() == 1 else input_pos[0]
        if pos.numel() != 1:
            return False
        logical_pos = int(pos.item())
        return logical_pos >= int(self.cfg.prompt_len)

    def maybe_record_decode_step(self, *, layer_idx: int, input_pos: Optional[torch.Tensor], time_steps: int) -> bool:
        del layer_idx
        return self.is_decode_call(input_pos=input_pos, time_steps=time_steps)

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
            runtime_name="EarlySelectKV",
        )
        if self.uses_aux_cache:
            self.aux_cache.clear()
            capacity_len = self.active_prompt_len + int(max_new_tokens)
            for layer_idx, block in enumerate(gpt.transformer.h):
                cache = block.attn.kv_cache
                if cache is None:
                    raise RuntimeError("KV cache is not initialized.")
                prompt_k = cache.k[:1, :, : self.active_prompt_len, :].contiguous()
                self.aux_cache.build_layer(layer_idx, prompt_k, capacity_len=capacity_len)
        else:
            self.aux_cache.clear()

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

    def _get_routing_query(
        self,
        *,
        layer_idx: int,
        q: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rope_n_elem: int,
    ) -> Optional[torch.Tensor]:
        if layer_idx == 0:
            return None
        lookahead_q = self.lookahead_q_by_layer.get(layer_idx)
        if lookahead_q is None:
            return None
        lookahead_q = lookahead_q.to(device=q.device, dtype=q.dtype)
        q_roped = apply_rope(lookahead_q[..., :rope_n_elem], cos, sin)
        return torch.cat((q_roped, lookahead_q[..., rope_n_elem:]), dim=-1)

    def select_attention_indices(
        self,
        *,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rope_n_elem: int,
        v: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor | dict[str, torch.Tensor]]:
        del v
        if layer_idx < self.cfg.skip_layers:
            return None
        len_q = int(q.size(2))
        key_len = int(k.size(2))
        if len_q > 1:
            return None
        if len_q <= 0 or key_len <= 0 or self.cfg.topk_budget >= key_len:
            return None

        routing_q = self._get_routing_query(
            layer_idx=layer_idx,
            q=q,
            cos=cos,
            sin=sin,
            rope_n_elem=rope_n_elem,
        )
        if routing_q is None:
            return None

        if self.cfg.routing_mode == "exact_topk":
            return self._compute_exact_route(q=routing_q, k=k)["indices"]
        if self.cfg.routing_mode == "chunk1_local":
            return self._compute_local_route(q=routing_q, k=k, route_chunk_size=1)["indices"]
        if self.cfg.routing_mode == "hsa_local":
            return self._compute_local_route(
                layer_idx=layer_idx,
                q=routing_q,
                k=k,
                route_chunk_size=self.chunk_size,
            )["indices"]
        return self._compute_hsa_route(layer_idx=layer_idx, q=routing_q, k=k)["indices"]

    def _compute_exact_route(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        key_len = int(k.size(2))
        q_grouped = q.view(q.size(0), self.cfg.kv_heads, -1, q.size(2), q.size(-1)).contiguous()
        k_grouped = k.unsqueeze(2)

        scores = torch.matmul(q_grouped, k_grouped.transpose(-1, -2)) / math.sqrt(q.size(-1))
        probs = torch.softmax(scores, dim=-1)
        mean_scores = probs.mean(dim=2)

        k_eff = min(int(self.cfg.topk_budget), key_len)
        topk_idx = self._sorted_topk(mean_scores, k_eff)
        return {"indices": topk_idx, "scores": mean_scores}

    def _compute_hsa_route(
        self,
        *,
        layer_idx: Optional[int] = None,
        q: torch.Tensor,
        k: torch.Tensor,
        topk_budget: Optional[int] = None,
        chunk_size: Optional[int] = None,
        trim_cached_summary: bool = False,
    ) -> dict[str, torch.Tensor | int | float]:
        key_len = int(k.size(2))
        if chunk_size is None:
            chunk_size = self.chunk_size
        chunk_size = max(int(chunk_size), 1)
        requested_topk = int(self.cfg.topk_budget) if topk_budget is None else int(topk_budget)
        summary = self._get_hsa_summary(
            layer_idx=layer_idx,
            k=k,
            key_len=key_len,
            chunk_size=chunk_size,
            trim_cached_summary=trim_cached_summary,
        )
        topk_idx = select_hsa_indices_from_summary(
            q,
            summary,
            key_len=key_len,
            chunk_size=chunk_size,
            compression_ratio=self.cfg.compression_ratio,
            topk_budget=requested_topk,
            kv_heads=self.cfg.kv_heads,
            sorted_topk=False,
        )
        base_r = round(q.size(-1) * chunk_size / max(float(self.cfg.compression_ratio), 1.0))
        r = max(1, min(q.size(-1), int(base_r)))
        return {
            "indices": topk_idx,
            "channel_budget_r": int(r),
            "base_channel_budget_r": int(base_r),
        }

    def _get_hsa_summary(
        self,
        *,
        layer_idx: Optional[int],
        k: torch.Tensor,
        key_len: int,
        chunk_size: int,
        trim_cached_summary: bool,
    ) -> torch.Tensor:
        summary = None
        use_layer_summary = layer_idx is not None and chunk_size == self.chunk_size and self.uses_aux_cache
        if use_layer_summary:
            summary = self.aux_cache.get(layer_idx, key_len=key_len)
        if summary is None:
            return build_chunk_summary_values(
                k,
                chunk_size=chunk_size,
                capacity_len=key_len,
            )

        if not trim_cached_summary or chunk_size < 2 or key_len % chunk_size == 0:
            return summary

        full_chunks = key_len // chunk_size
        boundary_start = full_chunks * chunk_size
        boundary_k = k[..., boundary_start:key_len, :]
        boundary_summary = build_chunk_summary_values(
            boundary_k,
            chunk_size=chunk_size,
            capacity_len=boundary_k.size(2),
        )
        if full_chunks <= 0:
            return boundary_summary
        return torch.cat((summary[:, :, :full_chunks, :], boundary_summary), dim=2).contiguous()

    def _compute_local_route(
        self,
        *,
        layer_idx: Optional[int] = None,
        q: torch.Tensor,
        k: torch.Tensor,
        route_chunk_size: int,
    ) -> dict[str, torch.Tensor]:
        len_q = int(q.size(2))
        key_len = int(k.size(2))
        k_eff = min(int(self.cfg.topk_budget), key_len)
        local_budget = self._resolve_decode_local_budget(key_len)
        route_budget = max(k_eff - local_budget, 0)
        local_start = key_len - local_budget

        if route_budget > 0 and local_start > 0:
            route = self._compute_hsa_route(
                layer_idx=layer_idx,
                q=q,
                k=k[..., :local_start, :],
                topk_budget=min(route_budget, local_start),
                chunk_size=route_chunk_size,
                trim_cached_summary=True,
            )
            route_idx = route["indices"]
        else:
            route_idx = torch.empty(
                q.size(0),
                int(self.cfg.kv_heads),
                len_q,
                0,
                device=q.device,
                dtype=torch.long,
            )

        if local_budget > 0:
            local_idx = torch.arange(local_start, key_len, device=q.device, dtype=torch.long)
            local_idx = local_idx.view(1, 1, 1, local_budget).expand(q.size(0), int(self.cfg.kv_heads), len_q, -1)
            topk_idx = torch.cat((route_idx, local_idx), dim=-1)
        else:
            topk_idx = route_idx

        return {"indices": topk_idx}


def decode_lookaheadkv(
    *,
    prompt_ids: torch.Tensor,
    model: LLM,
    runtime: LookaheadKVRuntime,
    max_new_tokens: int,
) -> str:
    return run_sparse_decode(
        prompt_ids=prompt_ids,
        model=model,
        runtime=runtime,
        max_new_tokens=max_new_tokens,
        install_attention=install_lookaheadkv_attention,
        uninstall_attention=uninstall_lookaheadkv_attention,
    )
