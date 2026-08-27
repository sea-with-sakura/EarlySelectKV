from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from typing import Any

import torch
import torch.nn as nn

from litgpt.model import batched_index_select, build_mask_cache, do_softcapping

DEFAULT_PREFILL_CHUNK_SIZE = 4096

try:
    from torch.nn.attention.bias import causal_lower_right
except (ImportError, AttributeError):
    causal_lower_right = None


def _get_base_model_and_precision_ctx(model: torch.nn.Module):
    base_model = getattr(model, "module", model)
    precision_ctx = nullcontext()
    strategy = getattr(model, "_strategy", None)
    if strategy is not None:
        precision_ctx = strategy.precision.forward_context()
    return base_model, precision_ctx


def _slice_input_pos(input_pos: torch.Tensor, *, start: int, end: int) -> torch.Tensor:
    if input_pos.dim() == 1:
        return input_pos[start:end]
    return input_pos[..., start:end]


def _resolve_chunk_input_pos_maxp1(input_pos_chunk: torch.Tensor, *, limit: int) -> int:
    if input_pos_chunk.numel() == 0:
        return 0
    return min(int(input_pos_chunk.max().item()) + 1, int(limit))


def _build_causal_mask_for_input_pos(
    input_pos: torch.Tensor,
    *,
    input_pos_maxp1: int,
    use_causal_bias: bool = True,
) -> Any:
    input_pos_maxp1 = int(input_pos_maxp1)
    pos = None
    if input_pos.dim() == 1:
        pos = input_pos
    elif input_pos.dim() == 2 and input_pos.size(0) == 1:
        pos = input_pos[0]
    if (
        use_causal_bias
        and causal_lower_right is not None
        and pos is not None
        and pos.numel() > 0
        and input_pos_maxp1 >= pos.numel()
    ):
        start = input_pos_maxp1 - int(pos.numel())
        expected = torch.arange(start, input_pos_maxp1, device=pos.device, dtype=pos.dtype)
        if torch.equal(pos.reshape(-1), expected):
            return causal_lower_right(int(pos.numel()), input_pos_maxp1)

    key_positions = torch.arange(int(input_pos_maxp1), device=input_pos.device, dtype=input_pos.dtype)
    if input_pos.dim() == 1:
        return key_positions.view(1, 1, 1, -1) <= input_pos.view(1, 1, -1, 1)
    if input_pos.dim() == 2:
        return key_positions.view(1, 1, 1, -1) <= input_pos.unsqueeze(1).unsqueeze(-1)
    raise ValueError(f"input_pos must have 1 or 2 dimensions, input_pos.shape = {input_pos.shape}")


def _resolve_prefill_chunk_size(
    model: torch.nn.Module,
    *,
    requested_chunk_size: int | None,
) -> int | None:
    if requested_chunk_size is not None:
        resolved = int(requested_chunk_size)
    else:
        resolved = getattr(model, "_prefill_chunk_size", None)
        if resolved is None:
            base_model = getattr(model, "module", model)
            resolved = getattr(base_model, "_prefill_chunk_size", None)
        if resolved is None:
            resolved = DEFAULT_PREFILL_CHUNK_SIZE
    if int(resolved) <= 0:
        return None
    return int(resolved)


def _iter_chunk_ranges(seq_len: int, chunk_size: int):
    start = 0
    while start < seq_len:
        remaining = seq_len - start
        step = min(chunk_size, remaining)
        # Keep the final sparse-prefill chunk from degenerating into a single token
        # when we can borrow one token from the preceding chunk.
        if step == chunk_size and step > 1 and remaining - step == 1:
            step -= 1
        end = start + step
        yield start, end
        start = end


def _forward_prefill_chunk(
    base_model: torch.nn.Module,
    *,
    idx: torch.Tensor,
    input_pos: torch.Tensor,
    input_pos_maxp1: int,
) -> torch.Tensor:
    seq_len = idx.size(1)
    if input_pos.dim() > 2:
        raise ValueError(f"input_pos must have 1 or 2 dimensions, input_pos.shape = {input_pos.shape}")
    if input_pos.shape[-1] != seq_len:
        raise ValueError(f"input_pos.shape[-1] = {input_pos.shape[-1]} != {seq_len} = idx.shape[1], must be the same")

    cos = batched_index_select(base_model.cos, 0, input_pos)
    sin = batched_index_select(base_model.sin, 0, input_pos)
    if input_pos.dim() == 1:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    use_causal_bias = base_model.config.attention_logit_softcapping is None
    if base_model.mask_cache is None or use_causal_bias:
        mask = _build_causal_mask_for_input_pos(
            input_pos,
            input_pos_maxp1=input_pos_maxp1,
            use_causal_bias=use_causal_bias,
        )
    else:
        mask = batched_index_select(base_model.mask_cache, 2, input_pos)
        if mask.dim() > 4:
            mask = mask.view(*(mask.shape[0:1] + mask.shape[2:]))
        mask = mask[..., :input_pos_maxp1]

    x = base_model.transformer.wte(idx)
    if base_model.config.scale_embeddings:
        x = x * torch.tensor(base_model.config.n_embd**0.5, dtype=x.dtype, device=x.device)

    for block_idx, block in enumerate(base_model.transformer.h):
        if base_model.config.rope_indices is not None:
            x = block(
                x,
                cos[..., base_model.config.rope_indices[block_idx]],
                sin[..., base_model.config.rope_indices[block_idx]],
                mask,
                input_pos,
                input_pos_maxp1,
            )
        else:
            x = block(x, cos, sin, mask, input_pos, input_pos_maxp1)
    return x


def setup_standard_kv_cache(
    gpt: torch.nn.Module,
    *,
    logical_max_seq_length: int,
    cache_max_seq_length: int,
    device: torch.device,
    build_mask: bool = True,
) -> None:
    gpt.mask_cache = None
    gpt.max_seq_length = int(logical_max_seq_length)
    if build_mask:
        gpt.mask_cache = build_mask_cache(int(logical_max_seq_length), device)
    rope_cache_length = gpt.rope_cache_length() if hasattr(gpt, "rope_cache_length") else None
    for block in gpt.transformer.h:
        block.attn.kv_cache = block.attn.build_kv_cache(
            batch_size=1,
            max_seq_length=int(cache_max_seq_length),
            rope_cache_length=rope_cache_length,
            device=device,
            dtype=block.attn.qkv.weight.dtype,
        )


def snapshot_standard_kv_caches(
    gpt: torch.nn.Module,
    *,
    seq_len: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    cache_entries: list[tuple[torch.Tensor, torch.Tensor]] = []
    for block in gpt.transformer.h:
        cache = block.attn.kv_cache
        if cache is None:
            raise RuntimeError("KV cache is not initialized.")
        cache_entries.append(
            (
                cache.k[:1, :, :seq_len, :].detach().clone().contiguous(),
                cache.v[:1, :, :seq_len, :].detach().clone().contiguous(),
            )
        )
    return cache_entries


def load_standard_kv_caches(
    gpt: torch.nn.Module,
    cache_entries: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    logical_max_seq_length: int,
    cache_max_seq_length: int,
    device: torch.device,
) -> None:
    setup_standard_kv_cache(
        gpt,
        logical_max_seq_length=logical_max_seq_length,
        cache_max_seq_length=cache_max_seq_length,
        device=device,
    )
    for block, (k_states, v_states) in zip(gpt.transformer.h, cache_entries):
        cache = block.attn.kv_cache
        if cache is None:
            raise RuntimeError("KV cache is not initialized.")
        seq_len = k_states.size(2)
        cache.k[:1, :, :seq_len, :] = k_states.to(device=device, dtype=cache.k.dtype)
        cache.v[:1, :, :seq_len, :] = v_states.to(device=device, dtype=cache.v.dtype)


def compact_standard_kv_cache(
    gpt: torch.nn.Module,
    *,
    seq_len: int,
    logical_max_seq_length: int,
    cache_max_seq_length: int,
    device: torch.device,
) -> None:
    gpt.mask_cache = None
    gpt.max_seq_length = int(logical_max_seq_length)
    gpt.mask_cache = build_mask_cache(int(logical_max_seq_length), device)
    rope_cache_length = gpt.rope_cache_length() if hasattr(gpt, "rope_cache_length") else None
    seq_len = int(seq_len)

    for block in gpt.transformer.h:
        old_cache = block.attn.kv_cache
        if old_cache is None:
            raise RuntimeError("KV cache is not initialized.")

        new_cache = block.attn.build_kv_cache(
            batch_size=1,
            max_seq_length=int(cache_max_seq_length),
            rope_cache_length=rope_cache_length,
            device=device,
            dtype=block.attn.qkv.weight.dtype,
        )
        new_cache.k[:1, :, :seq_len, :] = old_cache.k[:1, :, :seq_len, :].to(device=device, dtype=new_cache.k.dtype)
        new_cache.v[:1, :, :seq_len, :] = old_cache.v[:1, :, :seq_len, :].to(device=device, dtype=new_cache.v.dtype)
        block.attn.kv_cache = new_cache


def prefill_last_token_logits_layerwise(
    model: torch.nn.Module,
    idx: torch.Tensor,
    input_pos: torch.Tensor,
    *,
    input_pos_maxp1: int,
    chunk_size: int | None = None,
) -> torch.Tensor:
    base_model, precision_ctx = _get_base_model_and_precision_ctx(model)
    resolved_chunk_size = _resolve_prefill_chunk_size(model, requested_chunk_size=chunk_size)

    if idx.dim() != 2:
        raise ValueError(f"idx must have shape [batch, seq], got {idx.shape}")
    if idx.size(0) != 1:
        raise ValueError(f"Layer-wise sparse prefill currently supports batch size 1, got {idx.size(0)}")
    seq_len = int(idx.size(1))
    if input_pos.shape[-1] != seq_len:
        raise ValueError(f"input_pos.shape[-1] = {input_pos.shape[-1]} != {seq_len} = idx.shape[1], must be the same")
    if resolved_chunk_size is None:
        resolved_chunk_size = seq_len
    chunk_ranges = list(_iter_chunk_ranges(seq_len, resolved_chunk_size))

    base_model.mask_cache = None
    rope_cache_length = base_model.rope_cache_length() if hasattr(base_model, "rope_cache_length") else None

    with precision_ctx:
        hidden = torch.empty(
            (idx.size(0), seq_len, base_model.config.n_embd),
            device=idx.device,
            dtype=base_model.transformer.wte.weight.dtype,
        )
        for chunk_start, chunk_end in chunk_ranges:
            x = base_model.transformer.wte(idx[:, chunk_start:chunk_end])
            if base_model.config.scale_embeddings:
                x = x * torch.tensor(base_model.config.n_embd**0.5, dtype=x.dtype, device=x.device)
            hidden[:, chunk_start:chunk_end, :] = x

        for block_idx, block in enumerate(base_model.transformer.h):
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size=idx.size(0),
                max_seq_length=int(input_pos_maxp1),
                rope_cache_length=rope_cache_length,
                device=idx.device,
                dtype=block.attn.qkv.weight.dtype,
            )
            try:
                for chunk_start, chunk_end in chunk_ranges:
                    chunk_input_pos = _slice_input_pos(input_pos, start=chunk_start, end=chunk_end)
                    chunk_input_pos_maxp1 = _resolve_chunk_input_pos_maxp1(chunk_input_pos, limit=input_pos_maxp1)
                    cos = batched_index_select(base_model.cos, 0, chunk_input_pos)
                    sin = batched_index_select(base_model.sin, 0, chunk_input_pos)
                    if chunk_input_pos.dim() == 1:
                        cos = cos.unsqueeze(0)
                        sin = sin.unsqueeze(0)
                    mask = _build_causal_mask_for_input_pos(
                        chunk_input_pos,
                        input_pos_maxp1=chunk_input_pos_maxp1,
                        use_causal_bias=base_model.config.attention_logit_softcapping is None,
                    )

                    x = hidden[:, chunk_start:chunk_end, :]
                    if base_model.config.rope_indices is not None:
                        x = block(
                            x,
                            cos[..., base_model.config.rope_indices[block_idx]],
                            sin[..., base_model.config.rope_indices[block_idx]],
                            mask,
                            chunk_input_pos,
                            chunk_input_pos_maxp1,
                        )
                    else:
                        x = block(x, cos, sin, mask, chunk_input_pos, chunk_input_pos_maxp1)
                    hidden[:, chunk_start:chunk_end, :] = x
            finally:
                block.attn.kv_cache = None

        x = base_model.transformer.ln_f(hidden[:, -1:, :])
        clamp_head = (
            partial(do_softcapping, thresh=base_model.config.final_logit_softcapping)
            if base_model.config.final_logit_softcapping is not None
            else nn.Identity()
        )
        return clamp_head(base_model.lm_head(x))


def decode_next_token_logits(
    model: torch.nn.Module,
    idx: torch.Tensor,
    input_pos: torch.Tensor,
    *,
    input_pos_maxp1: int,
) -> torch.Tensor:
    base_model, precision_ctx = _get_base_model_and_precision_ctx(model)
    if idx.dim() != 2 or idx.size(1) != 1:
        raise ValueError(f"Single-token decode expects idx with shape [batch, 1], got {idx.shape}")
    if input_pos.shape[-1] != 1:
        raise ValueError(f"Single-token decode expects input_pos.shape[-1] == 1, got {input_pos.shape}")

    with precision_ctx:
        cos = batched_index_select(base_model.cos, 0, input_pos)
        sin = batched_index_select(base_model.sin, 0, input_pos)
        if input_pos.dim() == 1:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        x = base_model.transformer.wte(idx)
        if base_model.config.scale_embeddings:
            x = x * torch.tensor(base_model.config.n_embd**0.5, dtype=x.dtype, device=x.device)

        for block_idx, block in enumerate(base_model.transformer.h):
            if base_model.config.rope_indices is not None:
                x = block(
                    x,
                    cos[..., base_model.config.rope_indices[block_idx]],
                    sin[..., base_model.config.rope_indices[block_idx]],
                    None,
                    input_pos,
                    input_pos_maxp1,
                )
            else:
                x = block(x, cos, sin, None, input_pos, input_pos_maxp1)

        x = base_model.transformer.ln_f(x)
        clamp_head = (
            partial(do_softcapping, thresh=base_model.config.final_logit_softcapping)
            if base_model.config.final_logit_softcapping is not None
            else nn.Identity()
        )
        return clamp_head(base_model.lm_head(x))


def prefill_last_token_logits(
    model: torch.nn.Module,
    idx: torch.Tensor,
    input_pos: torch.Tensor,
    *,
    input_pos_maxp1: int,
    chunk_size: int | None = None,
) -> torch.Tensor:
    base_model, precision_ctx = _get_base_model_and_precision_ctx(model)
    resolved_chunk_size = _resolve_prefill_chunk_size(model, requested_chunk_size=chunk_size)

    with precision_ctx:
        if resolved_chunk_size is None or idx.size(1) <= resolved_chunk_size:
            x = _forward_prefill_chunk(base_model, idx=idx, input_pos=input_pos, input_pos_maxp1=input_pos_maxp1)
        else:
            x = None
            seq_len = int(idx.size(1))
            for chunk_start in range(0, seq_len, resolved_chunk_size):
                chunk_end = min(chunk_start + resolved_chunk_size, seq_len)
                chunk_input_pos = _slice_input_pos(input_pos, start=chunk_start, end=chunk_end)
                x = _forward_prefill_chunk(
                    base_model,
                    idx=idx[:, chunk_start:chunk_end],
                    input_pos=chunk_input_pos,
                    input_pos_maxp1=_resolve_chunk_input_pos_maxp1(chunk_input_pos, limit=input_pos_maxp1),
                )
            if x is None:
                raise RuntimeError("Chunked prefill produced no chunks.")
        x = base_model.transformer.ln_f(x[:, -1:, :])
        clamp_head = (
            partial(do_softcapping, thresh=base_model.config.final_logit_softcapping)
            if base_model.config.final_logit_softcapping is not None
            else nn.Identity()
        )
        return clamp_head(base_model.lm_head(x))
