from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from litgpt import LLM
from litgpt.model import do_softcapping

from pipeline.runtime_decode_utils import (
    group_queries_by_kv,
    repeat_kv_for_queries,
    run_full_cache_decode,
    sorted_topk_indices,
)
from .loki_monkey_patch import install_loki_attention, uninstall_loki_attention

logger = logging.getLogger("main")


@dataclass
class LokiConfig:
    prompt_len: int
    token_budget: int
    rank: float
    skip_layers: int
    attention_heads: int
    kv_heads: int
    attention_scores_scalar: Optional[int]
    head_size: int
    model_name: str = ""
    pca_dir: Optional[str] = None
    pca_model_name: Optional[str] = None
    transform_dataset: str = "wikitext"
    rotary_type: str = "postrotary"
    group_shared_within_kv_group: bool = True
    allow_identity_fallback: bool = True


class LokiRuntime:
    """Naive PyTorch Loki decode attention over LitGPT's standard KV cache."""

    _sorted_topk = staticmethod(sorted_topk_indices)

    def __init__(self, cfg: LokiConfig) -> None:
        self.cfg = cfg
        if self.cfg.skip_layers < 0:
            raise ValueError("skip_layers must be non-negative.")
        if self.cfg.token_budget <= 0:
            raise ValueError("token_budget must be positive.")
        if self.cfg.rank == 0:
            raise ValueError("rank must be non-zero. Use -1 to keep all dimensions.")
        if self.cfg.attention_heads % self.cfg.kv_heads != 0:
            raise ValueError(
                f"attention_heads={self.cfg.attention_heads} must be divisible by kv_heads={self.cfg.kv_heads}"
            )
        self._pca_base_dir = self._resolve_pca_base_dir()
        self._component_cache: dict[tuple[int, str, Optional[int], torch.dtype], torch.Tensor] = {}
        self._prerotary_k_cache: dict[int, torch.Tensor] = {}
        self._warned_identity_fallback = False
        if self.cfg.rotary_type not in {"postrotary", "prerotary"}:
            raise ValueError(f"Unsupported Loki rotary_type={self.cfg.rotary_type}")

    def reset(self) -> None:
        self._prerotary_k_cache.clear()

    def uses_prerotary_route(self) -> bool:
        return str(self.cfg.rotary_type) == "prerotary"

    def observe_layer_hidden(self, **_: object) -> None:
        return

    def cache_position_for_input(self, input_pos: torch.Tensor) -> torch.Tensor:
        return input_pos

    def update_prerotary_key_cache(
        self,
        *,
        layer_idx: int,
        cache_input_pos: torch.Tensor,
        k_update: torch.Tensor,
        capacity_len: int,
    ) -> None:
        if not self.uses_prerotary_route():
            return
        if k_update.ndim != 4:
            raise ValueError(f"expected k_update with shape [B, H, T, D], got {tuple(k_update.shape)}")

        positions = cache_input_pos.reshape(-1).to(device=k_update.device, dtype=torch.long)
        if positions.numel() != k_update.size(2):
            raise ValueError(
                f"cache positions ({positions.numel()}) must match k_update tokens ({k_update.size(2)})"
            )

        layer_idx = int(layer_idx)
        capacity_len = max(int(capacity_len), int(positions.max().item()) + 1 if positions.numel() else 0)
        cached = self._prerotary_k_cache.get(layer_idx)
        if (
            cached is None
            or cached.device != k_update.device
            or cached.dtype != k_update.dtype
            or cached.size(0) != k_update.size(0)
            or cached.size(1) != k_update.size(1)
            or cached.size(3) != k_update.size(3)
            or cached.size(2) < capacity_len
        ):
            new_cache = torch.zeros(
                k_update.size(0),
                k_update.size(1),
                capacity_len,
                k_update.size(3),
                dtype=k_update.dtype,
                device=k_update.device,
            )
            if cached is not None:
                keep_len = min(cached.size(2), new_cache.size(2))
                new_cache[:, :, :keep_len, :] = cached[:, :, :keep_len, :].to(
                    device=k_update.device, dtype=k_update.dtype
                )
            cached = new_cache
            self._prerotary_k_cache[layer_idx] = cached

        cached[:, :, positions, :] = k_update

    def get_prerotary_key_cache(self, *, layer_idx: int, key_len: int) -> Optional[torch.Tensor]:
        cached = self._prerotary_k_cache.get(int(layer_idx))
        if cached is None:
            return None
        key_len = min(max(0, int(key_len)), int(cached.size(2)))
        return cached[:, :, :key_len, :]

    def compute_attention_output(
        self,
        *,
        attn_self: object,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        route_q: Optional[torch.Tensor] = None,
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

        route_q = q if route_q is None else route_q
        route_k = k if route_k is None else route_k
        if int(route_k.size(2)) < key_len:
            return None

        route_scores = self._approx_route_scores(
            attn_self=attn_self,
            layer_idx=int(layer_idx),
            q=route_q,
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

    def _approx_route_scores(self, *, attn_self: object, layer_idx: int, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        components = self._get_layer_components(layer_idx, device=q.device, dtype=torch.float32)
        q_float = q.to(dtype=torch.float32)
        k_float = k.to(dtype=torch.float32)

        if components.size(0) in {1, int(self.cfg.kv_heads)}:
            if components.size(0) == 1:
                components = components.expand(int(self.cfg.kv_heads), -1, -1)
            q_grouped = group_queries_by_kv(
                q_float,
                attention_heads=int(self.cfg.attention_heads),
                kv_heads=int(self.cfg.kv_heads),
            )
            q_proj = torch.einsum("bgqtd,gdr->bgqtr", q_grouped, components)
            k_proj = torch.einsum("bgsd,gdr->bgsr", k_float, components)
            scores = torch.matmul(q_proj, k_proj.unsqueeze(2).transpose(-1, -2))
            scores = self._scale_scores(attn_self, scores)
            if bool(self.cfg.group_shared_within_kv_group):
                return F.softmax(scores, dim=-1, dtype=torch.float32).mean(dim=2)
            return scores.reshape(q.size(0), int(self.cfg.attention_heads), q.size(2), k.size(2))

        if components.size(0) != int(self.cfg.attention_heads):
            raise ValueError(
                "PCA component head count must be 1, kv_heads, or attention_heads; "
                f"got {components.size(0)} for kv_heads={self.cfg.kv_heads}, "
                f"attention_heads={self.cfg.attention_heads}."
            )

        k_full = repeat_kv_for_queries(
            k_float,
            attention_heads=int(self.cfg.attention_heads),
            kv_heads=int(self.cfg.kv_heads),
        )
        q_proj = torch.einsum("bhtd,hdr->bhtr", q_float, components)
        k_proj = torch.einsum("bhsd,hdr->bhsr", k_full, components)
        scores = torch.matmul(q_proj, k_proj.transpose(-1, -2))
        scores = self._scale_scores(attn_self, scores)
        if bool(self.cfg.group_shared_within_kv_group) and int(self.cfg.kv_heads) != int(self.cfg.attention_heads):
            q_per_kv = int(self.cfg.attention_heads) // int(self.cfg.kv_heads)
            scores = scores.reshape(q.size(0), int(self.cfg.kv_heads), q_per_kv, q.size(2), k.size(2))
            return F.softmax(scores, dim=-1, dtype=torch.float32).mean(dim=2)
        return scores

    def _grouped_exact_output(
        self,
        *,
        attn_self: object,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        head_dim = int(q.size(-1))
        q_grouped = group_queries_by_kv(
            q,
            attention_heads=int(self.cfg.attention_heads),
            kv_heads=int(self.cfg.kv_heads),
        )
        gather_idx = indices.squeeze(2).unsqueeze(-1).expand(-1, -1, -1, head_dim)
        selected_k = torch.gather(k, dim=2, index=gather_idx)
        selected_v = torch.gather(v, dim=2, index=gather_idx)

        scores = torch.matmul(
            q_grouped.to(dtype=torch.float32),
            selected_k.unsqueeze(2).transpose(-1, -2).to(dtype=torch.float32),
        )
        scores = self._scale_scores(attn_self, scores)
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=v.dtype)
        output = torch.matmul(weights, selected_v.unsqueeze(2))
        return output.permute(0, 3, 1, 2, 4).reshape(q.size(0), q.size(2), int(self.cfg.attention_heads), head_dim)

    def _per_head_exact_output(
        self,
        *,
        attn_self: object,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        head_dim = int(q.size(-1))
        if int(self.cfg.kv_heads) == int(self.cfg.attention_heads):
            gather_idx = indices.squeeze(2).unsqueeze(-1).expand(-1, -1, -1, head_dim)
            selected_k = torch.gather(k, dim=2, index=gather_idx)
            selected_v = torch.gather(v, dim=2, index=gather_idx)
        else:
            q_per_kv = int(self.cfg.attention_heads) // int(self.cfg.kv_heads)
            grouped_indices = indices.squeeze(2).reshape(
                indices.size(0),
                int(self.cfg.kv_heads),
                q_per_kv,
                indices.size(-1),
            )
            k_grouped = k.unsqueeze(2).expand(-1, -1, q_per_kv, -1, -1)
            v_grouped = v.unsqueeze(2).expand(-1, -1, q_per_kv, -1, -1)
            gather_idx = grouped_indices.unsqueeze(-1).expand(-1, -1, -1, -1, head_dim)
            selected_k = torch.gather(k_grouped, dim=3, index=gather_idx).reshape(
                k.size(0),
                int(self.cfg.attention_heads),
                indices.size(-1),
                head_dim,
            )
            selected_v = torch.gather(v_grouped, dim=3, index=gather_idx).reshape(
                v.size(0),
                int(self.cfg.attention_heads),
                indices.size(-1),
                head_dim,
            )

        scores = torch.matmul(q.to(dtype=torch.float32), selected_k.transpose(-1, -2).to(dtype=torch.float32))
        scores = self._scale_scores(attn_self, scores)
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=v.dtype)
        return torch.matmul(weights, selected_v).transpose(1, 2).contiguous()

    def _scale_scores(self, attn_self: object, scores: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.cfg.attention_scores_scalar or self.cfg.head_size)
        scale = scale * float(getattr(attn_self, "mscale", 1.0)) * float(getattr(attn_self, "mscale", 1.0))
        scores = scores * scale
        softcap = getattr(getattr(attn_self, "config", None), "attention_logit_softcapping", None)
        if softcap is not None:
            scores = do_softcapping(scores, softcap)
        return scores

    def _get_layer_components(self, layer_idx: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cache_key = (int(layer_idx), device.type, device.index, dtype)
        cached = self._component_cache.get(cache_key)
        if cached is not None:
            return cached

        components = self._load_layer_components(int(layer_idx)).to(device=device, dtype=dtype)
        self._component_cache[cache_key] = components
        return components

    def _load_layer_components(self, layer_idx: int) -> torch.Tensor:
        if self._pca_base_dir is None:
            return self._identity_components(layer_idx)

        component_file = self._pca_base_dir / "pca_components" / f"pca_components_layer_{layer_idx}.pt"
        if not component_file.is_file():
            if bool(self.cfg.allow_identity_fallback):
                return self._identity_components(layer_idx, missing_file=component_file)
            raise FileNotFoundError(f"Missing Loki PCA component file: {component_file}")

        components = torch.load(component_file, map_location="cpu")
        if components.ndim != 3:
            raise ValueError(f"Expected PCA components with shape [heads, D, D], got {tuple(components.shape)}")
        if int(components.size(-1)) != int(self.cfg.head_size) or int(components.size(-2)) != int(self.cfg.head_size):
            raise ValueError(
                f"PCA component head_dim mismatch: got {tuple(components.shape)}, expected D={self.cfg.head_size}"
            )

        rank = self._resolve_rank(layer_idx=layer_idx, head_dim=int(components.size(-1)))
        return components.to(dtype=torch.float32).transpose(-1, -2).contiguous()[..., :rank]

    def _resolve_rank(self, *, layer_idx: int, head_dim: int) -> int:
        rank = float(self.cfg.rank)
        if rank < 0:
            return int(head_dim)
        if 0 < rank < 1:
            variance_file = None
            if self._pca_base_dir is not None:
                variance_file = (
                    self._pca_base_dir
                    / "pca_explained_variance"
                    / f"pca_explained_variance_layer_{layer_idx}.pt"
                )
            if variance_file is not None and variance_file.is_file():
                variances = torch.load(variance_file, map_location="cpu").to(dtype=torch.float32)
                rank_count = int((variances.cumsum(-1) < rank).sum(-1).max().item())
                return max(1, min(rank_count, int(head_dim)))
            return max(1, min(int(math.ceil(rank * head_dim)), int(head_dim)))
        return max(1, min(int(rank), int(head_dim)))

    def _identity_components(self, layer_idx: int, missing_file: Optional[Path] = None) -> torch.Tensor:
        if not self._warned_identity_fallback:
            detail = f" because {missing_file} is missing" if missing_file is not None else ""
            logger.warning(
                "Loki PCA components are unavailable%s; using the first rank dimensions as an identity fallback.",
                detail,
            )
            self._warned_identity_fallback = True
        rank = self._resolve_rank(layer_idx=layer_idx, head_dim=int(self.cfg.head_size))
        eye = torch.eye(int(self.cfg.head_size), dtype=torch.float32)
        return eye[:, :rank].unsqueeze(0).contiguous()

    def _resolve_pca_base_dir(self) -> Optional[Path]:
        if self.cfg.pca_dir is None:
            return None

        root = Path(str(self.cfg.pca_dir)).expanduser()
        model_key = str(self.cfg.pca_model_name or Path(str(self.cfg.model_name)).name)
        candidates = [
            root,
            root / "key",
            root / f"{model_key}-PCA" / self.cfg.transform_dataset / self.cfg.rotary_type / "key",
            root / model_key / self.cfg.transform_dataset / self.cfg.rotary_type / "key",
        ]
        for candidate in candidates:
            if (candidate / "pca_components").is_dir():
                return candidate

        if bool(self.cfg.allow_identity_fallback):
            logger.warning("Could not resolve Loki PCA directory under %s; identity fallback is enabled.", root)
            return None
        raise FileNotFoundError(f"Could not resolve Loki PCA directory from pca_dir={root}")


def decode_loki(
    *,
    prompt_ids: torch.Tensor,
    model: LLM,
    runtime: LokiRuntime,
    max_new_tokens: int,
) -> str:
    runtime.reset()
    return run_full_cache_decode(
        prompt_ids=prompt_ids,
        model=model,
        max_new_tokens=max_new_tokens,
        attention_state=runtime,
        install_attention=install_loki_attention,
        uninstall_attention=uninstall_loki_attention,
    )
