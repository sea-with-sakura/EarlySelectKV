from __future__ import annotations

import torch


def gather_selected_kv(
    *,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    attention_heads: int,
    expand_to_attention_heads: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = k.size(-1)
    if indices.size(1) != k.size(1):
        if indices.size(1) == attention_heads and attention_heads % k.size(1) == 0:
            q_per_kv = attention_heads // k.size(1)
            grouped_indices = indices.squeeze(2).reshape(
                indices.size(0),
                k.size(1),
                q_per_kv,
                indices.size(-1),
            )
            k_grouped = k.unsqueeze(2).expand(-1, -1, q_per_kv, -1, -1)
            v_grouped = v.unsqueeze(2).expand(-1, -1, q_per_kv, -1, -1)
            gather_idx = grouped_indices.unsqueeze(-1).expand(-1, -1, -1, -1, head_dim)
            selected_k = torch.gather(k_grouped, dim=3, index=gather_idx).reshape(
                k.size(0),
                attention_heads,
                indices.size(-1),
                head_dim,
            )
            selected_v = torch.gather(v_grouped, dim=3, index=gather_idx).reshape(
                v.size(0),
                attention_heads,
                indices.size(-1),
                head_dim,
            )
            return selected_k.contiguous(), selected_v.contiguous()
        repeats = indices.size(1) // k.size(1)
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    gather_idx = indices.squeeze(2).unsqueeze(-1).expand(-1, -1, -1, head_dim)
    selected_k = torch.gather(k, dim=2, index=gather_idx)
    selected_v = torch.gather(v, dim=2, index=gather_idx)
    if expand_to_attention_heads and selected_k.size(1) != attention_heads:
        repeats = attention_heads // selected_k.size(1)
        selected_k = selected_k.repeat_interleave(repeats, dim=1)
        selected_v = selected_v.repeat_interleave(repeats, dim=1)
    return selected_k.contiguous(), selected_v.contiguous()
