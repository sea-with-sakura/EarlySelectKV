import math
from dataclasses import dataclass

import torch

from pipeline.runtime_decode_utils import gather_along_dim


@dataclass
class ChunkSummaryState:
    values: torch.Tensor
    chunk_size: int
    head_dim: int
    capacity_len: int
    key_len: int


def build_chunk_summary_values(
    k: torch.Tensor,
    *,
    chunk_size: int,
    capacity_len: int | None = None,
) -> torch.Tensor:
    """Build RocketKV-style K_min/K_max page summaries.

    For chunk_size >= 2 the last dim is [K_min, K_max]. For chunk_size == 1
    the summary is just a token-aligned K buffer, matching the gpt-fast path.
    """
    if k.ndim != 4:
        raise ValueError(f"expected k with shape [B, H, T, D], got {tuple(k.shape)}")
    chunk_size = max(1, int(chunk_size))
    batch, heads, key_len, head_dim = k.shape
    capacity_len = key_len if capacity_len is None else max(int(capacity_len), key_len)

    if chunk_size < 2:
        values = torch.zeros(
            batch,
            heads,
            capacity_len,
            head_dim,
            dtype=k.dtype,
            device=k.device,
        )
        values[:, :, :key_len, :] = k
        return values.contiguous()

    num_chunks = max(1, math.ceil(capacity_len / chunk_size))
    padded_len = num_chunks * chunk_size
    pad_len = padded_len - key_len

    min_pad = torch.full(
        (batch, heads, pad_len, head_dim),
        float("inf"),
        dtype=k.dtype,
        device=k.device,
    )
    max_pad = torch.full(
        (batch, heads, pad_len, head_dim),
        -float("inf"),
        dtype=k.dtype,
        device=k.device,
    )
    k_min = torch.cat((k, min_pad), dim=2)
    k_max = torch.cat((k, max_pad), dim=2)
    k_min = k_min.view(batch, heads, num_chunks, chunk_size, head_dim).amin(dim=3)
    k_max = k_max.view(batch, heads, num_chunks, chunk_size, head_dim).amax(dim=3)
    return torch.cat((k_min, k_max), dim=-1).contiguous()


class ChunkSummaryCache:
    """Per-layer auxiliary storage for RocketKV-style page routing."""

    def __init__(self, *, chunk_size: int) -> None:
        self.chunk_size = max(1, int(chunk_size))
        self.by_layer: dict[int, ChunkSummaryState] = {}

    def clear(self) -> None:
        self.by_layer.clear()

    def build_layer(
        self,
        layer_idx: int,
        k: torch.Tensor,
        *,
        capacity_len: int | None = None,
    ) -> None:
        if k.ndim != 4:
            raise ValueError(f"expected k with shape [B, H, T, D], got {tuple(k.shape)}")
        capacity_len = k.size(2) if capacity_len is None else max(int(capacity_len), k.size(2))
        values = build_chunk_summary_values(
            k,
            chunk_size=self.chunk_size,
            capacity_len=capacity_len,
        )
        self.by_layer[int(layer_idx)] = ChunkSummaryState(
            values=values,
            chunk_size=self.chunk_size,
            head_dim=int(k.size(-1)),
            capacity_len=capacity_len,
            key_len=int(k.size(2)),
        )

    def build_all(
        self,
        prompt_k_by_layer: dict[int, torch.Tensor],
        *,
        capacity_len: int,
    ) -> None:
        self.clear()
        for layer_idx, k in prompt_k_by_layer.items():
            self.build_layer(layer_idx, k, capacity_len=capacity_len)

    def get(self, layer_idx: int, *, key_len: int) -> torch.Tensor | None:
        state = self.by_layer.get(int(layer_idx))
        if state is None:
            return None
        key_len = min(max(0, int(key_len)), state.capacity_len)
        if state.chunk_size < 2:
            return state.values[:, :, :key_len, :]
        needed_chunks = max(1, math.ceil(key_len / state.chunk_size))
        return state.values[:, :, :needed_chunks, :]

    def update_layer(
        self,
        layer_idx: int,
        *,
        cache_input_pos: torch.Tensor,
        k_update: torch.Tensor,
    ) -> None:
        state = self.by_layer.get(int(layer_idx))
        if state is None:
            return
        if k_update.ndim != 4:
            raise ValueError(f"expected k_update with shape [B, H, T, D], got {tuple(k_update.shape)}")

        positions = cache_input_pos.reshape(-1).to(device=k_update.device, dtype=torch.long)
        if positions.numel() != k_update.size(2):
            raise ValueError(
                f"cache positions ({positions.numel()}) must match k_update tokens ({k_update.size(2)})"
            )

        values = state.values
        for token_offset, pos_tensor in enumerate(positions):
            pos = int(pos_tensor.item())
            if pos < 0 or pos >= state.capacity_len:
                continue
            token = k_update[:, :, token_offset, :].to(dtype=values.dtype)
            if state.chunk_size < 2:
                values[:, :, pos, :] = token
            else:
                chunk_idx = pos // state.chunk_size
                min_slice = values[:, :, chunk_idx, : state.head_dim]
                max_slice = values[:, :, chunk_idx, state.head_dim :]
                values[:, :, chunk_idx, : state.head_dim] = torch.minimum(min_slice, token)
                values[:, :, chunk_idx, state.head_dim :] = torch.maximum(max_slice, token)
            state.key_len = max(state.key_len, pos + 1)


def select_hsa_indices_from_summary(
    q: torch.Tensor,
    summary: torch.Tensor,
    *,
    key_len: int,
    chunk_size: int,
    compression_ratio: float,
    topk_budget: int,
    kv_heads: int,
    sorted_topk: bool = False,
) -> torch.Tensor:
    """Select token indices with RocketKV gpt-fast two-stage HSA routing."""
    if q.ndim != 4:
        raise ValueError(f"expected q with shape [B, H, T, D], got {tuple(q.shape)}")
    if summary.ndim != 4:
        raise ValueError(f"expected summary with shape [B, H, C, D], got {tuple(summary.shape)}")
    key_len = int(key_len)
    if key_len <= 0:
        return torch.empty(q.size(0), int(kv_heads), q.size(2), 0, dtype=torch.long, device=q.device)

    batch, query_heads, q_len, head_dim = q.shape
    if q_len != 1:
        raise ValueError("RocketKV HSA routing is only implemented for single-token decode.")
    if query_heads % kv_heads != 0:
        raise ValueError(f"query heads ({query_heads}) must be divisible by kv heads ({kv_heads})")

    chunk_size = max(1, int(chunk_size))
    topk_budget = min(max(1, int(topk_budget)), key_len)
    r = round(head_dim * chunk_size / max(float(compression_ratio), 1.0))
    r = min(head_dim, max(1, int(r)))

    q_grouped = q.view(batch, kv_heads, query_heads // kv_heads, q_len, head_dim).contiguous()
    abs_q = q_grouped.abs()
    channel_idx = torch.topk(
        abs_q.sum(dim=2, keepdim=True),
        k=r,
        dim=-1,
        sorted=sorted_topk,
    ).indices
    q_hat = gather_along_dim(q_grouped, -1, channel_idx)

    if chunk_size >= 2:
        signed_channel_idx = torch.where(
            q_hat.sum(dim=2, keepdim=True) > 0,
            channel_idx + head_dim,
            channel_idx,
        )
    else:
        signed_channel_idx = channel_idx

    k_hat = gather_along_dim(summary.unsqueeze(2), -1, signed_channel_idx)
    qk_hat = torch.matmul(q_hat, k_hat.transpose(-1, -2))
    qk_hat = qk_hat.repeat_interleave(chunk_size, dim=-1)[..., :key_len]

    denom = abs_q.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(abs_q.dtype).eps)
    scale = torch.sqrt(head_dim * q_hat.abs().sum(dim=-1, keepdim=True) / denom)
    scale = scale.clamp_min(torch.finfo(scale.dtype).eps)
    stage2_scores = torch.softmax(qk_hat / scale, dim=-1).sum(dim=2)
    return torch.topk(stage2_scores, k=topk_budget, dim=-1, sorted=sorted_topk).indices
