from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from litgpt.model import Block, CausalSelfAttention, KVCache, apply_rope, do_softcapping
from pipeline.decode_attention import grouped_decode_attention
from pipeline.monkey_patch_utils import gather_selected_kv

_ORIG_BLOCK_FORWARD = None
_ORIG_ATTN_FORWARD = None
_ORIG_ATTN_SDPA = None

def install_rocketkv_monkey_patch() -> None:
    global _ORIG_BLOCK_FORWARD, _ORIG_ATTN_FORWARD, _ORIG_ATTN_SDPA
    if _ORIG_ATTN_FORWARD is not None:
        return

    _ORIG_BLOCK_FORWARD = Block.forward
    _ORIG_ATTN_FORWARD = CausalSelfAttention.forward
    _ORIG_ATTN_SDPA = CausalSelfAttention.scaled_dot_product_attention

    def patched_block_forward(
        block_self: Block,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
        input_pos_maxp1: int | None = None,
    ) -> torch.Tensor:
        runtime = getattr(block_self.attn, "rocketkv_runtime", None)
        if runtime is None:
            return _ORIG_BLOCK_FORWARD(block_self, x, cos, sin, mask, input_pos, input_pos_maxp1)

        x_normed = block_self.norm_1(x)
        attention_output = block_self.attn(x_normed, cos, sin, mask, input_pos, input_pos_maxp1)
        attention_output = block_self.post_attention_norm(attention_output)

        if block_self.config.parallel_residual:
            if not block_self.config.shared_attention_norm:
                x_normed = block_self.norm_2(x)
            x = attention_output + x
        else:
            x = attention_output + x
            x_normed = block_self.norm_2(x)

        runtime.maybe_record_decode_step(
            layer_idx=block_self.attn.block_idx,
            input_pos=input_pos,
            time_steps=int(x.size(1)),
        )
        mlp_out = block_self.post_mlp_norm(block_self.mlp(x_normed))
        return mlp_out + x

    def patched_forward(
        attn_self: CausalSelfAttention,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        input_pos_maxp1: Optional[int] = None,
    ) -> torch.Tensor:
        runtime = getattr(attn_self, "rocketkv_runtime", None)
        if runtime is None or input_pos is None:
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

        q_roped = apply_rope(q[..., :rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., :rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., rope_n_elem:]), dim=-1)
        k = torch.cat((k_roped, k[..., rope_n_elem:]), dim=-1)

        if not isinstance(attn_self.kv_cache, KVCache):
            raise TypeError("You need to call `gpt.set_kv_cache()`")
        cache_input_pos = runtime.cache_position_for_input(input_pos)
        k_update = k
        k, v = attn_self.kv_cache(cache_input_pos, k_update, v)
        if runtime.uses_aux_cache:
            runtime.update_aux_cache(
                layer_idx=attn_self.block_idx,
                cache_input_pos=cache_input_pos,
                k_update=k_update,
            )

        if attn_self.apply_sliding_window_attention:
            actual_kv_len = k.size(2)
            if mask is not None and mask.size(-1) != actual_kv_len:
                mask = mask[..., :actual_kv_len]

        if input_pos_maxp1 is not None:
            k = k[..., :input_pos_maxp1, :]
            v = v[..., :input_pos_maxp1, :]
        runtime.observe_layer_hidden(layer_idx=attn_self.block_idx, hidden=x, input_pos=input_pos)
        selection = None
        if runtime.is_decode_call(input_pos=input_pos, time_steps=time_steps):
            selection = runtime.select_attention_indices(
                layer_idx=attn_self.block_idx,
                q=q,
                k=k,
            )
        k_for_cache = k
        v_for_cache = v

        if selection is not None:
            selected_k, selected_v = gather_selected_kv(
                k=k,
                v=v,
                indices=selection,
                attention_heads=n_head,
                expand_to_attention_heads=False,
            )
            if selected_k.size(1) != n_head:
                y = grouped_decode_attention(attn_self, q, selected_k, selected_v)
            else:
                y = attn_self.scaled_dot_product_attention(q, selected_k, selected_v, None)
            y = y.reshape(batches, time_steps, head_size * n_head)
            y = attn_self.proj(y)
            runtime.maybe_compress_prefill(
                layer_idx=attn_self.block_idx,
                attn_module=attn_self,
                q=q,
                k=k_for_cache,
                v=v_for_cache,
                input_pos_maxp1=input_pos_maxp1,
            )
            return y

        if n_query_groups != n_head and (input_pos is None or n_query_groups != 1):
            q_per_kv = n_head // n_query_groups
            k = k.repeat_interleave(q_per_kv, dim=1)
            v = v.repeat_interleave(q_per_kv, dim=1)

        if attn_self.apply_sliding_window_attention and input_pos is None:
            if mask is None:
                mask = torch.ones(time_steps, time_steps, dtype=q.dtype, device=q.device).triu(diagonal=1)
                mask.masked_fill_(mask.bool(), float("-inf"))
                mask = mask.view(1, 1, *mask.shape)

            sliding_window_mask = torch.full((time_steps, time_steps), float("-inf"), dtype=q.dtype, device=q.device)
            for i in range(time_steps):
                window_start = max(0, i - attn_self.config.sliding_window_size + 1)
                sliding_window_mask[i, window_start : i + 1] = 0.0
            mask = sliding_window_mask.view(1, 1, time_steps, time_steps)

        if (
            selection is None
            and time_steps > 1
            and not attn_self.apply_sliding_window_attention
            and input_pos is not None
            and (input_pos_maxp1 is None or int(input_pos_maxp1) == time_steps)
        ):
            mask = None

        y = attn_self.scaled_dot_product_attention(q, k, v, mask if time_steps > 1 else None)
        y = y.reshape(batches, time_steps, head_size * n_head)
        y = attn_self.proj(y)
        runtime.maybe_compress_prefill(
            layer_idx=attn_self.block_idx,
            attn_module=attn_self,
            q=q,
            k=k_for_cache,
            v=v_for_cache,
            input_pos_maxp1=input_pos_maxp1,
        )
        return y

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
                mask = torch.ones(q.size(2), q.size(2), dtype=q.dtype, device=q.device).triu(diagonal=1)
                mask.masked_fill_(mask.bool(), torch.finfo(q.dtype).min)
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
                is_causal=mask is None and q.size(2) > 1,
            )
        return y.transpose(1, 2)

    Block.forward = patched_block_forward
    CausalSelfAttention.forward = patched_forward
    CausalSelfAttention.scaled_dot_product_attention = patched_scaled_dot_product_attention


def uninstall_rocketkv_monkey_patch() -> None:
    global _ORIG_BLOCK_FORWARD, _ORIG_ATTN_FORWARD, _ORIG_ATTN_SDPA
    if _ORIG_ATTN_FORWARD is None:
        return
    Block.forward = _ORIG_BLOCK_FORWARD
    CausalSelfAttention.forward = _ORIG_ATTN_FORWARD
    CausalSelfAttention.scaled_dot_product_attention = _ORIG_ATTN_SDPA
    _ORIG_BLOCK_FORWARD = None
    _ORIG_ATTN_FORWARD = None
    _ORIG_ATTN_SDPA = None


def attach_rocketkv_runtime(model: torch.nn.Module, runtime: object) -> None:
    for block in model.transformer.h:
        block.attn.rocketkv_runtime = runtime


def detach_rocketkv_runtime(model: torch.nn.Module) -> None:
    for block in model.transformer.h:
        block.attn.rocketkv_runtime = None


def install_rocketkv_attention(model: torch.nn.Module, runtime: object) -> None:
    install_rocketkv_monkey_patch()
    attach_rocketkv_runtime(model, runtime)


def uninstall_rocketkv_attention(model: torch.nn.Module) -> None:
    detach_rocketkv_runtime(model)
    uninstall_rocketkv_monkey_patch()
