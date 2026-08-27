from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from litgpt.model import do_softcapping


def grouped_decode_attention(
    attn_self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    attention_heads = int(q.size(1))
    kv_heads = int(k.size(1))
    if attention_heads == kv_heads:
        return attn_self.scaled_dot_product_attention(q, k, v, None)
    if attention_heads % kv_heads != 0:
        raise ValueError(f"attention_heads={attention_heads} must be divisible by kv_heads={kv_heads}")

    q_per_kv = attention_heads // kv_heads
    q_grouped = q.view(q.size(0), kv_heads, q_per_kv, q.size(2), q.size(3)).contiguous()
    k_grouped = k.unsqueeze(2)
    v_grouped = v.unsqueeze(2)

    scale = 1.0 / math.sqrt(attn_self.config.attention_scores_scalar or attn_self.config.head_size)
    scale = scale * attn_self.mscale * attn_self.mscale

    scores = torch.matmul(q_grouped, k_grouped.transpose(-1, -2)) * scale
    if attn_self.config.attention_logit_softcapping is not None:
        scores = do_softcapping(scores, attn_self.config.attention_logit_softcapping)
    if mask is not None:
        if mask.ndim == 4:
            mask = mask.unsqueeze(2)
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + mask
    probs = F.softmax(scores, dim=-1, dtype=torch.float).to(dtype=q.dtype)
    y = torch.matmul(probs, v_grouped)
    return y.permute(0, 3, 1, 2, 4).reshape(q.size(0), q.size(2), attention_heads, q.size(3))
