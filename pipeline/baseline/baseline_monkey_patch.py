from __future__ import annotations

from typing import Optional

import torch

from litgpt.model import CausalSelfAttention, KVCache, apply_rope, apply_rope_interleave

_ORIG_ATTN_FORWARD = None


def _is_full_prefill(
    *,
    x: torch.Tensor,
    input_pos: Optional[torch.Tensor],
    input_pos_maxp1: Optional[int],
) -> bool:
    if input_pos is None or input_pos_maxp1 is None:
        return False
    time_steps = int(x.size(1))
    if time_steps <= 1 or int(input_pos_maxp1) != time_steps:
        return False
    pos = input_pos if input_pos.dim() == 1 else input_pos[0]
    if pos.numel() != time_steps:
        return False
    expected = torch.arange(0, time_steps, device=pos.device, dtype=pos.dtype)
    return bool(torch.equal(pos, expected))


def install_baseline_monkey_patch() -> None:
    global _ORIG_ATTN_FORWARD
    if _ORIG_ATTN_FORWARD is not None:
        return

    _ORIG_ATTN_FORWARD = CausalSelfAttention.forward

    def patched_forward(
        attn_self: CausalSelfAttention,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        input_pos_maxp1: Optional[int] = None,
    ) -> torch.Tensor:
        runtime = getattr(attn_self, "baseline_runtime", None)
        if runtime is None or input_pos is None or attn_self.apply_sliding_window_attention:
            return _ORIG_ATTN_FORWARD(attn_self, x, cos, sin, mask, input_pos, input_pos_maxp1)

        if not _is_full_prefill(x=x, input_pos=input_pos, input_pos_maxp1=input_pos_maxp1):
            return _ORIG_ATTN_FORWARD(attn_self, x, cos, sin, mask, input_pos, input_pos_maxp1)

        head_size = attn_self.config.head_size
        n_head = attn_self.config.n_head
        n_query_groups = attn_self.config.n_query_groups
        rope_n_elem = attn_self.config.rope_n_elem
        batches, time_steps, _ = x.size()

        qkv = attn_self.qkv(x)
        query_size = n_head * head_size
        key_size = value_size = n_query_groups * head_size
        q, k, v = qkv.split((query_size, key_size, value_size), dim=-1)

        if attn_self.config.norm_qk and attn_self.config.norm_qk_type == "olmo2":
            q = attn_self.norm_q(q)
            k = attn_self.norm_k(k)

        q = q.view(batches, time_steps, n_head, head_size).transpose(1, 2)
        k = k.view(batches, time_steps, n_query_groups, head_size).transpose(1, 2)
        v = v.view(batches, time_steps, n_query_groups, head_size).transpose(1, 2)

        if attn_self.config.norm_qk and attn_self.config.norm_qk_type == "default":
            q = attn_self.norm_q(q)
            k = attn_self.norm_k(k)

        if attn_self.config.rope_interleave:
            q_roped = apply_rope_interleave(q[..., :rope_n_elem], cos, sin)
            k_roped = apply_rope_interleave(k[..., :rope_n_elem], cos, sin)
        else:
            q_roped = apply_rope(q[..., :rope_n_elem], cos, sin)
            k_roped = apply_rope(k[..., :rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., rope_n_elem:]), dim=-1)
        k = torch.cat((k_roped, k[..., rope_n_elem:]), dim=-1)

        if not isinstance(attn_self.kv_cache, KVCache):
            raise TypeError("You need to call `gpt.set_kv_cache()`")
        k, v = attn_self.kv_cache(input_pos, k, v)

        if input_pos_maxp1 is not None:
            k = k[..., :input_pos_maxp1, :]
            v = v[..., :input_pos_maxp1, :]

        if n_query_groups != n_head and (input_pos is None or n_query_groups != 1):
            q_per_kv = n_head // n_query_groups
            k = k.repeat_interleave(q_per_kv, dim=1)
            v = v.repeat_interleave(q_per_kv, dim=1)

        y = attn_self.scaled_dot_product_attention(q, k, v, None)
        y = y.reshape(batches, time_steps, head_size * n_head)
        return attn_self.proj(y)

    CausalSelfAttention.forward = patched_forward


def uninstall_baseline_monkey_patch() -> None:
    global _ORIG_ATTN_FORWARD
    if _ORIG_ATTN_FORWARD is None:
        return
    CausalSelfAttention.forward = _ORIG_ATTN_FORWARD
    _ORIG_ATTN_FORWARD = None


def attach_baseline_runtime(model: torch.nn.Module, runtime: object) -> None:
    for block in model.transformer.h:
        block.attn.baseline_runtime = runtime


def detach_baseline_runtime(model: torch.nn.Module) -> None:
    for block in model.transformer.h:
        block.attn.baseline_runtime = None


def install_baseline_attention(model: torch.nn.Module, runtime: object) -> None:
    install_baseline_monkey_patch()
    attach_baseline_runtime(model, runtime)


def uninstall_baseline_attention(model: torch.nn.Module) -> None:
    detach_baseline_runtime(model)
    uninstall_baseline_monkey_patch()
