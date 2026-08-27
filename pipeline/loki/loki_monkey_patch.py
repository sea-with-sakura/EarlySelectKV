from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from litgpt.model import CausalSelfAttention, KVCache, apply_rope, apply_rope_interleave, do_softcapping
from pipeline.decode_attention import grouped_decode_attention

_ORIG_ATTN_FORWARD = None
_ORIG_ATTN_SDPA = None


def install_loki_monkey_patch() -> None:
    global _ORIG_ATTN_FORWARD, _ORIG_ATTN_SDPA
    if _ORIG_ATTN_FORWARD is not None:
        return

    _ORIG_ATTN_FORWARD = CausalSelfAttention.forward
    _ORIG_ATTN_SDPA = CausalSelfAttention.scaled_dot_product_attention

    def project_qkv_pre_rope(attn_self: CausalSelfAttention, x: torch.Tensor):
        head_size = attn_self.config.head_size
        n_head = attn_self.config.n_head
        n_query_groups = attn_self.config.n_query_groups
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

        return q, k, v

    def patched_forward(
        attn_self: CausalSelfAttention,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        input_pos_maxp1: Optional[int] = None,
    ) -> torch.Tensor:
        runtime = getattr(attn_self, "loki_runtime", None)
        if runtime is None or input_pos is None:
            return _ORIG_ATTN_FORWARD(attn_self, x, cos, sin, mask, input_pos, input_pos_maxp1)
        if x.size(1) > 1:
            if bool(runtime.uses_prerotary_route()):
                if not isinstance(attn_self.kv_cache, KVCache):
                    raise TypeError("You need to call `gpt.set_kv_cache()`")
                _, k_pre, _ = project_qkv_pre_rope(attn_self, x)
                cache_input_pos = runtime.cache_position_for_input(input_pos)
                runtime.update_prerotary_key_cache(
                    layer_idx=attn_self.block_idx,
                    cache_input_pos=cache_input_pos,
                    k_update=k_pre,
                    capacity_len=int(attn_self.kv_cache.k.size(2)),
                )
            return _ORIG_ATTN_FORWARD(attn_self, x, cos, sin, mask, input_pos, input_pos_maxp1)

        head_size = attn_self.config.head_size
        n_head = attn_self.config.n_head
        n_query_groups = attn_self.config.n_query_groups
        rope_n_elem = attn_self.config.rope_n_elem
        batches, time_steps, _ = x.size()

        q, k, v = project_qkv_pre_rope(attn_self, x)
        q_pre = q
        k_pre = k

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
        cache_input_pos = runtime.cache_position_for_input(input_pos)
        k, v = attn_self.kv_cache(cache_input_pos, k, v)
        route_q = None
        route_k = None
        if bool(runtime.uses_prerotary_route()):
            runtime.update_prerotary_key_cache(
                layer_idx=attn_self.block_idx,
                cache_input_pos=cache_input_pos,
                k_update=k_pre,
                capacity_len=int(attn_self.kv_cache.k.size(2)),
            )
            route_q = q_pre

        if attn_self.apply_sliding_window_attention:
            actual_kv_len = k.size(2)
            if mask is not None and mask.size(-1) != actual_kv_len:
                mask = mask[..., :actual_kv_len]

        if input_pos_maxp1 is not None:
            k = k[..., :input_pos_maxp1, :]
            v = v[..., :input_pos_maxp1, :]
            if mask is not None and mask.size(-1) != input_pos_maxp1:
                mask = mask[..., :input_pos_maxp1]

        if bool(runtime.uses_prerotary_route()):
            route_k = runtime.get_prerotary_key_cache(layer_idx=attn_self.block_idx, key_len=int(k.size(2)))

        runtime.observe_layer_hidden(layer_idx=attn_self.block_idx, hidden=x, input_pos=input_pos)
        y = runtime.compute_attention_output(
            attn_self=attn_self,
            layer_idx=attn_self.block_idx,
            q=q,
            k=k,
            v=v,
            route_q=route_q,
            route_k=route_k,
        )

        if y is not None:
            y = y.reshape(batches, time_steps, head_size * n_head)
            return attn_self.proj(y)

        if time_steps == 1 and n_query_groups != n_head:
            y = grouped_decode_attention(attn_self, q, k, v, None)
            y = y.reshape(batches, time_steps, head_size * n_head)
            return attn_self.proj(y)

        if mask is None:
            mask = torch.ones(
                k.size(0),
                1,
                q.size(2),
                k.size(2),
                dtype=torch.bool,
                device=q.device,
            )
        y = attn_self.scaled_dot_product_attention(q, k, v, mask)
        y = y.reshape(batches, time_steps, head_size * n_head)
        return attn_self.proj(y)

    def patched_scaled_dot_product_attention(
        attn_self: CausalSelfAttention,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(attn_self.config.attention_scores_scalar or attn_self.config.head_size)
        scale = scale * attn_self.mscale * attn_self.mscale

        if attn_self.config.attention_logit_softcapping is not None:
            scores = q @ k.mT * scale
            scores = do_softcapping(scores, attn_self.config.attention_logit_softcapping)
            if mask is None:
                mask = torch.ones(q.size(2), k.size(2), dtype=torch.bool, device=q.device).tril()
                mask = mask.view(1, 1, *mask.shape)
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, torch.finfo(q.dtype).min)
            else:
                scores = scores + mask
            scores = F.softmax(scores, dim=-1, dtype=torch.float).to(dtype=q.dtype)
            y = scores @ v
        else:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=0.0,
                scale=scale,
                is_causal=mask is None,
            )
        return y.transpose(1, 2)

    CausalSelfAttention.forward = patched_forward
    CausalSelfAttention.scaled_dot_product_attention = patched_scaled_dot_product_attention


def uninstall_loki_monkey_patch() -> None:
    global _ORIG_ATTN_FORWARD, _ORIG_ATTN_SDPA
    if _ORIG_ATTN_FORWARD is None:
        return
    CausalSelfAttention.forward = _ORIG_ATTN_FORWARD
    CausalSelfAttention.scaled_dot_product_attention = _ORIG_ATTN_SDPA
    _ORIG_ATTN_FORWARD = None
    _ORIG_ATTN_SDPA = None


def attach_loki_runtime(model: torch.nn.Module, runtime: object) -> None:
    for block in model.transformer.h:
        block.attn.loki_runtime = runtime


def detach_loki_runtime(model: torch.nn.Module) -> None:
    for block in model.transformer.h:
        block.attn.loki_runtime = None


def install_loki_attention(model: torch.nn.Module, runtime: object) -> None:
    install_loki_monkey_patch()
    attach_loki_runtime(model, runtime)


def uninstall_loki_attention(model: torch.nn.Module) -> None:
    detach_loki_runtime(model)
    uninstall_loki_monkey_patch()
