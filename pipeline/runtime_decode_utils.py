from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import torch
import torch.nn.functional as F

from litgpt import LLM
from litgpt.generate.base import sample

from pipeline.litgpt_prefill import (
    decode_next_token_logits,
    prefill_last_token_logits,
    prefill_last_token_logits_layerwise,
    setup_standard_kv_cache,
)


def compute_chunk_size(compression_ratio: float) -> int:
    chunk_size = math.ceil(math.sqrt(float(compression_ratio)))
    if chunk_size > float(compression_ratio):
        chunk_size = 1
    return max(int(chunk_size), 1)


def repeat_kv_for_queries(
    k: torch.Tensor,
    *,
    attention_heads: int,
    kv_heads: int,
) -> torch.Tensor:
    if kv_heads == attention_heads:
        return k
    q_per_kv = attention_heads // kv_heads
    return k.repeat_interleave(q_per_kv, dim=1)


def group_queries_by_kv(
    q: torch.Tensor,
    *,
    attention_heads: int,
    kv_heads: int,
) -> torch.Tensor:
    if q.size(1) == kv_heads:
        return q.unsqueeze(2)
    q_per_kv = attention_heads // kv_heads
    return q.reshape(q.size(0), kv_heads, q_per_kv, q.size(2), q.size(3))


def gather_along_dim(t: torch.Tensor, dim: int, idx: torch.Tensor) -> torch.Tensor:
    dim += (dim < 0) * t.ndim
    return t.gather(dim, idx.expand(*t.shape[:dim], idx.shape[dim], *t.shape[dim + 1 :]))


def sorted_topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        empty_shape = (*scores.shape[:-1], 0)
        return scores.new_empty(empty_shape, dtype=torch.long)
    return scores.topk(k=k, dim=-1).indices.sort(dim=-1).values


def _build_observation_mask(
    *,
    obs_window: int,
    key_len: int,
    device: torch.device,
) -> torch.Tensor:
    dist = (
        torch.arange(0, obs_window, device=device)[:, None]
        - torch.arange(0, key_len, device=device)[None, :]
        + key_len
        - obs_window
    )
    return dist >= 0


@dataclass
class StandardPromptCompressionResult:
    prompt_k: torch.Tensor
    prompt_v: torch.Tensor
    keep_idx: Optional[torch.Tensor]
    scores_full: Optional[torch.Tensor]
    obs_window: int
    key_len: int


def compress_standard_prompt_kv(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_len: int,
    active_prompt_len: int,
    window_size: int,
    kernel_size: int,
    attention_heads: int,
    kv_heads: int,
    score_chunk_size: int = 16,
) -> StandardPromptCompressionResult:
    key_len = int(key_len)
    active_prompt_len = int(active_prompt_len)
    if active_prompt_len <= 0:
        return StandardPromptCompressionResult(
            prompt_k=k[:, :, :0, :].contiguous(),
            prompt_v=v[:, :, :0, :].contiguous(),
            keep_idx=None,
            scores_full=None,
            obs_window=0,
            key_len=key_len,
        )
    if key_len <= active_prompt_len:
        return StandardPromptCompressionResult(
            prompt_k=k[:, :, :key_len, :].contiguous(),
            prompt_v=v[:, :, :key_len, :].contiguous(),
            keep_idx=None,
            scores_full=None,
            obs_window=key_len,
            key_len=key_len,
        )

    obs_window = min(int(window_size), int(q.size(2)), active_prompt_len, key_len)
    if obs_window <= 0:
        return StandardPromptCompressionResult(
            prompt_k=k[:, :, :0, :].contiguous(),
            prompt_v=v[:, :, :0, :].contiguous(),
            keep_idx=None,
            scores_full=None,
            obs_window=0,
            key_len=key_len,
        )

    q_observe = group_queries_by_kv(
        q[:, :, -obs_window:, :],
        attention_heads=attention_heads,
        kv_heads=kv_heads,
    )
    attention_mask = _build_observation_mask(obs_window=obs_window, key_len=key_len, device=q.device)

    scores_full = torch.zeros(k.size(0), kv_heads, key_len, dtype=q.dtype, device=q.device)
    score_chunk_size = max(1, int(score_chunk_size))
    for obs_start in range(0, obs_window, score_chunk_size):
        obs_end = min(obs_window, obs_start + score_chunk_size)
        q_chunk = q_observe[..., obs_start:obs_end, :]
        chunk_mask = attention_mask[obs_start:obs_end, :]
        scores = torch.einsum("bgqtd,bgkd->bgqtk", q_chunk, k) / math.sqrt(q.size(-1))
        scores = torch.masked_fill(
            scores,
            ~chunk_mask.view(1, 1, 1, obs_end - obs_start, key_len),
            torch.scalar_tensor(float("-inf"), device=scores.device, dtype=scores.dtype),
        )
        probs = F.softmax(scores, dim=-1)
        probs = torch.masked_fill(
            probs,
            ~chunk_mask.view(1, 1, 1, obs_end - obs_start, key_len),
            torch.scalar_tensor(0.0, device=probs.device, dtype=probs.dtype),
        )
        scores_full.add_(probs.sum(dim=-2).sum(dim=2))
        del scores, probs

    scores_prefix = scores_full[..., :-obs_window]
    pooled_scores = F.max_pool1d(scores_prefix, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)

    keep_count = active_prompt_len - obs_window
    if keep_count > 0:
        keep_idx = sorted_topk_indices(pooled_scores, keep_count)
        k_idx = keep_idx.unsqueeze(-1).expand(-1, -1, -1, k.size(-1))
        v_idx = keep_idx.unsqueeze(-1).expand(-1, -1, -1, v.size(-1))
        k_keep = k[:, :, :-obs_window, :].gather(dim=2, index=k_idx)
        v_keep = v[:, :, :-obs_window, :].gather(dim=2, index=v_idx)
    else:
        keep_idx = None
        k_keep = k[:, :, :0, :]
        v_keep = v[:, :, :0, :]

    return StandardPromptCompressionResult(
        prompt_k=torch.cat((k_keep, k[:, :, -obs_window:, :]), dim=2).contiguous(),
        prompt_v=torch.cat((v_keep, v[:, :, -obs_window:, :]), dim=2).contiguous(),
        keep_idx=keep_idx,
        scores_full=scores_full,
        obs_window=obs_window,
        key_len=key_len,
    )


def setup_prompt_kv_cache(
    *,
    gpt: torch.nn.Module,
    prompt_k_by_layer: dict[int, torch.Tensor],
    prompt_v_by_layer: dict[int, torch.Tensor],
    active_prompt_len: int,
    logical_max_seq_length: int,
    max_new_tokens: int,
    device: torch.device,
    runtime_name: str,
    allow_head_mismatch: bool = False,
) -> bool:
    setup_standard_kv_cache(
        gpt,
        logical_max_seq_length=logical_max_seq_length,
        cache_max_seq_length=int(active_prompt_len) + int(max_new_tokens),
        device=device,
        build_mask=False,
    )

    loaded_any = False
    for layer_idx, block in enumerate(gpt.transformer.h):
        prompt_k = prompt_k_by_layer.get(layer_idx)
        prompt_v = prompt_v_by_layer.get(layer_idx)
        if prompt_k is None or prompt_v is None:
            raise RuntimeError(f"Missing compressed prompt KV for {runtime_name} layer {layer_idx}.")

        cache = block.attn.kv_cache
        if cache is None:
            raise RuntimeError("KV cache is not initialized.")

        if prompt_k.size(1) != cache.k.size(1):
            if allow_head_mismatch:
                continue
            raise RuntimeError(
                f"{runtime_name} layer {layer_idx} prompt KV heads {prompt_k.size(1)} != cache heads {cache.k.size(1)}."
            )

        seq_len = int(prompt_k.size(2))
        cache.k[:1, :, :seq_len, :] = prompt_k.to(device=device, dtype=cache.k.dtype)
        cache.v[:1, :, :seq_len, :] = prompt_v.to(device=device, dtype=cache.v.dtype)
        loaded_any = True

    return loaded_any


class SparseDecodeRuntime(Protocol):
    active_prompt_len: int

    def reset(self) -> None: ...

    def setup_decode_kv_cache(
        self,
        *,
        gpt: torch.nn.Module,
        logical_max_seq_length: int,
        max_new_tokens: int,
        device: torch.device,
    ) -> None: ...


def run_full_cache_decode(
    *,
    prompt_ids: torch.Tensor,
    model: LLM,
    max_new_tokens: int,
    attention_state: object,
    install_attention: Callable[[torch.nn.Module, object], None],
    uninstall_attention: Callable[[torch.nn.Module], None],
) -> str:
    gpt = model.model
    prompt_ids = prompt_ids.to(torch.long)
    prompt_len = int(prompt_ids.numel())
    max_new_tokens = int(max_new_tokens)
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")

    max_returned_tokens = prompt_len + max_new_tokens
    if max_returned_tokens > model.config.block_size:
        raise ValueError(
            f"Requested total length {max_returned_tokens} exceeds model context {model.config.block_size}"
        )
    if max_new_tokens == 0:
        return ""

    device = prompt_ids.device
    gpt.clear_kv_cache()
    gpt.max_seq_length = max_returned_tokens
    gpt.set_kv_cache(batch_size=1, max_seq_length=max_returned_tokens, device=device)
    gpt.eval()

    generated = torch.empty(max_new_tokens, dtype=torch.long, device=device)
    generated_count = 0
    eos_id = getattr(model.tokenizer, "eos_id", None)

    install_attention(gpt, attention_state)
    try:
        with torch.inference_mode():
            prefill_tokens = prompt_ids.view(1, -1)
            prefill_pos = torch.arange(0, prompt_len, device=device, dtype=torch.int64)
            prefill_logits = prefill_last_token_logits(
                gpt,
                prefill_tokens,
                prefill_pos,
                input_pos_maxp1=prompt_len,
            )

            token = sample(prefill_logits, temperature=0.0, top_p=0.0, top_k=None).to(dtype=torch.int64).view(1, 1)
            generated[generated_count] = token.view(-1)[0]
            generated_count += 1

            input_pos = torch.tensor([prompt_len], device=device, dtype=torch.int64)
            input_pos_maxp1 = prompt_len + 1
            for _ in range(max_new_tokens - 1):
                if eos_id is not None and int(token.item()) == int(eos_id):
                    break
                logits = gpt(token, input_pos, input_pos_maxp1=input_pos_maxp1)
                token = sample(logits, temperature=0.0, top_p=0.0, top_k=None).to(dtype=torch.int64).view(1, 1)
                generated[generated_count] = token.view(-1)[0]
                generated_count += 1
                input_pos.add_(1)
                input_pos_maxp1 += 1
    finally:
        uninstall_attention(gpt)
        gpt.clear_kv_cache()

    generated_ids = generated[:generated_count]
    return model.preprocessor.decode(generated_ids) if generated_ids.numel() > 0 else ""


def run_sparse_decode(
    *,
    prompt_ids: torch.Tensor,
    model: LLM,
    runtime: SparseDecodeRuntime,
    max_new_tokens: int,
    install_attention: Callable[[torch.nn.Module, SparseDecodeRuntime], None],
    uninstall_attention: Callable[[torch.nn.Module], None],
) -> str:
    gpt = model.model
    prompt_ids = prompt_ids.to(torch.long)
    prompt_len = int(prompt_ids.numel())
    max_new_tokens = int(max_new_tokens)
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")

    max_returned_tokens = prompt_len + max_new_tokens
    if max_returned_tokens > model.config.block_size:
        raise ValueError(
            f"Requested total length {max_returned_tokens} exceeds model context {model.config.block_size}"
        )
    if max_new_tokens == 0:
        return ""

    device = prompt_ids.device
    gpt.clear_kv_cache()
    gpt.max_seq_length = max_returned_tokens
    gpt.mask_cache = None
    gpt.eval()
    runtime.reset()

    generated = torch.empty(max_new_tokens, dtype=torch.long, device=device)
    generated_count = 0
    eos_id = getattr(model.tokenizer, "eos_id", None)

    install_attention(gpt, runtime)
    try:
        with torch.inference_mode():
            prefill_tokens = prompt_ids.view(1, -1)
            prefill_pos = torch.arange(0, prompt_len, device=device, dtype=torch.int64)
            prefill_logits = prefill_last_token_logits_layerwise(
                gpt,
                prefill_tokens,
                prefill_pos,
                input_pos_maxp1=prompt_len,
            )
            runtime.setup_decode_kv_cache(
                gpt=gpt,
                logical_max_seq_length=max_returned_tokens,
                max_new_tokens=max_new_tokens,
                device=device,
            )

            token = sample(prefill_logits, temperature=0.0, top_p=0.0, top_k=None).to(dtype=torch.int64).view(1, 1)
            generated[generated_count] = token.view(-1)[0]
            generated_count += 1

            input_pos = torch.tensor([prompt_len], device=device, dtype=torch.int64)
            input_pos_maxp1 = int(runtime.active_prompt_len) + 1
            for _ in range(max_new_tokens - 1):
                if eos_id is not None and int(token.item()) == int(eos_id):
                    break
                logits = decode_next_token_logits(gpt, token, input_pos, input_pos_maxp1=input_pos_maxp1)
                token = sample(logits, temperature=0.0, top_p=0.0, top_k=None).to(dtype=torch.int64).view(1, 1)
                generated[generated_count] = token.view(-1)[0]
                generated_count += 1
                input_pos.add_(1)
                input_pos_maxp1 += 1
    finally:
        uninstall_attention(gpt)
        flush_probe = getattr(runtime, "flush_q_probe", None)
        if callable(flush_probe):
            flush_probe()
        gpt.clear_kv_cache()

    generated_ids = generated[:generated_count]
    return model.preprocessor.decode(generated_ids) if generated_ids.numel() > 0 else ""
