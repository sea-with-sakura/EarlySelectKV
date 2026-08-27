from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from litgpt.model import do_softcapping


def _materialize_attention_bias(mask, *, device: torch.device) -> torch.Tensor:
    materialize = getattr(mask, "_materialize", None)
    if callable(materialize):
        mask = materialize()
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask, device=device)
    return mask.to(device=device)


def _normalize_grouped_mask(mask, *, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    mask = _materialize_attention_bias(mask, device=q.device)
    q_len = int(q.size(2))
    key_len = int(k.size(2))
    if mask.size(-1) != key_len:
        mask = mask[..., :key_len]
    if mask.dim() >= 2 and mask.size(-2) != q_len:
        mask = mask[..., -q_len:, :]

    if mask.dim() == 2:
        mask = mask.view(1, 1, 1, mask.size(-2), mask.size(-1))
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1).unsqueeze(2)
    elif mask.dim() == 4:
        mask = mask.unsqueeze(2)
    elif mask.dim() != 5:
        raise ValueError(f"Unsupported grouped attention mask shape: {tuple(mask.shape)}")
    return mask


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
        mask = _normalize_grouped_mask(mask, q=q, k=k)
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + mask
    probs = F.softmax(scores, dim=-1, dtype=torch.float).to(dtype=q.dtype)
    y = torch.matmul(probs, v_grouped)
    return y.permute(0, 3, 1, 2, 4).reshape(q.size(0), q.size(2), attention_heads, q.size(3))
