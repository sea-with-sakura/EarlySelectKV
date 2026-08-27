from __future__ import annotations

import math
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class ESInfLLMV2Config:
    mode: str = "es_sparse"
    rank: int = 256
    route_query: str = "lowrank_next_wq"
    lookahead_source: str = "mid"
    apply_decode_only: bool = True
    skip_layers: int = 1
    dense_len: int = 8192
    kernel_size: int = 32
    kernel_stride: int = 16
    block_size: int = 64
    init_blocks: int = 1
    window_size: int = 2048
    remote_topk: int = 64
    route_q_scale: float = 1.0
    use_k2: bool = True
    svd_niter: int = 2
    lowrank_cache_path: str | None = None
    enabled: bool = True


@dataclass
class ESInfLLMV2Stats:
    routed_attn_calls: int = 0
    dense_attn_calls: int = 0
    oracle_route_calls: int = 0
    prepared_routes: int = 0
    prepared_from_input: int = 0
    prepared_from_mid: int = 0
    prepared_from_output: int = 0
    missing_route_calls: int = 0


@dataclass
class ESInfLLMV2Runtime:
    config: ESInfLLMV2Config
    lowrank: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    pending_block_masks: dict[int, torch.Tensor] = field(default_factory=dict)
    stats: ESInfLLMV2Stats = field(default_factory=ESInfLLMV2Stats)

    def reset_step(self) -> None:
        self.pending_block_masks.clear()


def config_from_sparse_config(
    sparse_config: dict[str, Any] | None,
    **overrides: Any,
) -> ESInfLLMV2Config:
    sparse_config = sparse_config or {}
    cfg = ESInfLLMV2Config(
        kernel_size=int(sparse_config.get("kernel_size", 32)),
        kernel_stride=int(sparse_config.get("kernel_stride", 16)),
        init_blocks=int(sparse_config.get("init_blocks", 1)),
        block_size=int(sparse_config.get("block_size", 64)),
        window_size=int(sparse_config.get("window_size", 2048)),
        remote_topk=int(sparse_config.get("topk", 64)),
        dense_len=int(sparse_config.get("dense_len", 8192)),
    )
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    return cfg


def _bound_method(obj: Any, name: str):
    method = getattr(obj, name)
    return method.__func__ if hasattr(method, "__func__") else method


def _get_forward_globals(module: torch.nn.Module) -> dict[str, Any]:
    saved = getattr(module, "_es_forward_globals", None)
    if saved is not None:
        return saved
    return _bound_method(module, "forward").__globals__


def _lowrank_factors(weight: torch.Tensor, rank: int, niter: int) -> tuple[torch.Tensor, torch.Tensor]:
    q = min(min(weight.shape), max(rank + 32, rank))
    u, s, v = torch.svd_lowrank(weight.float(), q=q, niter=niter)
    u = u[:, :rank].contiguous()
    s = s[:rank].contiguous()
    v = v[:, :rank].contiguous()
    down = v.T.contiguous()
    up = (u * s.unsqueeze(0)).contiguous()
    return down.cpu(), up.cpu()


def _project_lowrank(hidden: torch.Tensor, down: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    down = down.to(device=hidden.device, dtype=hidden.dtype)
    up = up.to(device=hidden.device, dtype=hidden.dtype)
    return F.linear(F.linear(hidden, down), up)


def _shape_q(q_flat: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    batch, q_len, _ = q_flat.shape
    return q_flat.view(batch, q_len, num_heads, head_dim).transpose(1, 2).contiguous()


def _get_cached_key(cache: Any, layer_idx: int) -> torch.Tensor | None:
    if cache is None:
        return None
    key_cache = getattr(cache, "key_cache", None)
    if key_cache is not None and layer_idx < len(key_cache):
        key = key_cache[layer_idx]
        if key is not None and key.numel() > 0:
            return key
    layers = getattr(cache, "layers", None)
    if layers is not None and layer_idx < len(layers):
        key = getattr(layers[layer_idx], "keys", None)
        if key is not None and key.numel() > 0:
            return key
    return None


def _repeat_kv_to_query_heads(kv: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    if num_key_value_groups == 1:
        return kv
    batch, kv_heads, seq_len, head_dim = kv.shape
    kv = kv[:, :, None, :, :].expand(batch, kv_heads, num_key_value_groups, seq_len, head_dim)
    return kv.reshape(batch, kv_heads * num_key_value_groups, seq_len, head_dim)


def _block_mask_to_token_mask(
    block_mask: torch.Tensor,
    *,
    kv_seq_len: int,
    block_size: int,
    num_key_value_groups: int,
) -> torch.Tensor:
    batch, kv_heads, q_len, block_count = block_mask.shape
    token_mask = torch.zeros(
        (batch, kv_heads, q_len, kv_seq_len),
        dtype=torch.bool,
        device=block_mask.device,
    )
    for block_idx in range(block_count):
        start = block_idx * block_size
        end = min(start + block_size, kv_seq_len)
        if start < end:
            token_mask[..., start:end] |= block_mask[..., block_idx : block_idx + 1]
    return _repeat_kv_to_query_heads(token_mask, num_key_value_groups)


def _pad_block_mask(block_mask: torch.Tensor, block_count: int) -> torch.Tensor:
    if block_mask.size(-1) == block_count:
        return block_mask
    padded = torch.zeros(
        (*block_mask.shape[:-1], block_count),
        dtype=block_mask.dtype,
        device=block_mask.device,
    )
    copy = min(block_mask.size(-1), block_count)
    if copy > 0:
        padded[..., :copy] = block_mask[..., :copy]
    return padded


def _forced_block_mask(
    *,
    batch_size: int,
    kv_heads: int,
    q_len: int,
    kv_seq_len: int,
    query_positions: torch.Tensor,
    cfg: ESInfLLMV2Config,
    device: torch.device,
) -> torch.Tensor:
    block_count = math.ceil(kv_seq_len / cfg.block_size)
    block_ids = torch.arange(block_count, device=device)
    if query_positions.dim() == 1:
        query_positions = query_positions.unsqueeze(0)
    q_blocks = query_positions.to(device) // cfg.block_size
    forced = torch.zeros((batch_size, kv_heads, q_len, block_count), dtype=torch.bool, device=device)
    if cfg.init_blocks > 0:
        forced[..., : min(cfg.init_blocks, block_count)] = True
    local_blocks = max(0, cfg.window_size // cfg.block_size)
    if local_blocks > 0:
        local = (block_ids.view(1, 1, 1, -1) <= q_blocks.view(batch_size, 1, q_len, 1)) & (
            q_blocks.view(batch_size, 1, q_len, 1) <= block_ids.view(1, 1, 1, -1) + local_blocks
        )
        forced |= local
    causal_valid = block_ids.view(1, 1, 1, -1) <= q_blocks.view(batch_size, 1, q_len, 1)
    return forced & causal_valid


def _compress_keys_mean(k: torch.Tensor, kernel_size: int, kernel_stride: int) -> torch.Tensor:
    batch, heads, seq_len, head_dim = k.shape
    if seq_len < kernel_size:
        return k.new_empty(batch, heads, 0, head_dim)
    starts = torch.arange(0, seq_len - kernel_size + 1, kernel_stride, device=k.device)
    offsets = torch.arange(kernel_size, device=k.device)
    indices = starts[:, None] + offsets[None, :]
    chunks = k[:, :, indices, :]
    return chunks.mean(dim=3).contiguous()


def _pooling_offsets(kernel_size: int, kernel_stride: int, block_size: int, device: torch.device) -> torch.Tensor:
    left = max(1, kernel_size // kernel_stride)
    right = max(1, block_size // kernel_stride)
    raw = (torch.arange(left, device=device)[:, None] + torch.arange(right, device=device)[None, :]).reshape(-1)
    return torch.bincount(raw, minlength=int(raw.max().item()) + 1).float()


def _grouped_compressed_probs(q: torch.Tensor, compressed_k: torch.Tensor) -> torch.Tensor:
    batch, q_heads, q_len, head_dim = q.shape
    _, kv_heads, compressed_len, _ = compressed_k.shape
    if compressed_len == 0:
        return q.new_empty(batch, kv_heads, q_len, 0)
    group = q_heads // kv_heads
    q_grouped = q.view(batch, kv_heads, group, q_len, head_dim)
    scores = torch.matmul(q_grouped, compressed_k.unsqueeze(2).transpose(-1, -2)) / math.sqrt(head_dim)
    return torch.softmax(scores.float(), dim=-1).mean(dim=2).to(q.dtype)


def _compressed_probs_to_block_scores(
    compressed_probs: torch.Tensor,
    *,
    query_positions: torch.Tensor,
    seq_len: int,
    kernel_size: int,
    kernel_stride: int,
    cfg: ESInfLLMV2Config,
) -> torch.Tensor:
    batch, kv_heads, q_len, compressed_len = compressed_probs.shape
    block_count = math.ceil(seq_len / cfg.block_size)
    block_scores = compressed_probs.new_full((batch, kv_heads, q_len, block_count), float("-inf"))
    if compressed_len == 0:
        return block_scores

    weights = _pooling_offsets(kernel_size, kernel_stride, cfg.block_size, compressed_probs.device).to(
        compressed_probs.dtype
    )
    pad_len = max(0, kernel_size // kernel_stride - 1)
    block_stride = max(1, cfg.block_size // kernel_stride)
    for block_idx in range(block_count):
        start = block_idx * block_stride - pad_len
        vals: list[torch.Tensor] = []
        for off, weight in enumerate(weights):
            src = start + off
            if 0 <= src < compressed_len:
                vals.append(compressed_probs[..., src] * weight)
        if vals:
            block_scores[..., block_idx] = torch.stack(vals, dim=-1).amax(dim=-1)

    if query_positions.dim() == 1:
        query_positions = query_positions.unsqueeze(0)
    key_blocks = torch.arange(block_count, device=compressed_probs.device)
    q_blocks = query_positions.to(compressed_probs.device) // cfg.block_size
    causal_valid = key_blocks.view(1, 1, 1, -1) <= q_blocks.view(batch, 1, q_len, 1)
    return block_scores.masked_fill(~causal_valid, float("-inf"))


def _stage1_route_mask(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    cfg: ESInfLLMV2Config,
) -> torch.Tensor:
    batch, _, q_len, _ = q.shape
    _, kv_heads, kv_seq_len, _ = k.shape
    query_positions = position_ids[:, -q_len:]
    forced = _forced_block_mask(
        batch_size=batch,
        kv_heads=kv_heads,
        q_len=q_len,
        kv_seq_len=kv_seq_len,
        query_positions=query_positions,
        cfg=cfg,
        device=q.device,
    )

    compressed_k = _compress_keys_mean(k, cfg.kernel_size, cfg.kernel_stride)
    probs = _grouped_compressed_probs(q, compressed_k)
    block_scores = _compressed_probs_to_block_scores(
        probs,
        query_positions=query_positions,
        seq_len=kv_seq_len,
        kernel_size=cfg.kernel_size,
        kernel_stride=cfg.kernel_stride,
        cfg=cfg,
    )
    if cfg.use_k2 and kv_seq_len >= cfg.kernel_size * 4:
        compressed_k2 = _compress_keys_mean(k, cfg.kernel_size * 4, cfg.kernel_stride * 4)
        probs2 = _grouped_compressed_probs(q, compressed_k2)
        block_scores2 = _compressed_probs_to_block_scores(
            probs2,
            query_positions=query_positions,
            seq_len=kv_seq_len,
            kernel_size=cfg.kernel_size * 4,
            kernel_stride=cfg.kernel_stride * 4,
            cfg=cfg,
        )
        block_scores = torch.maximum(block_scores, block_scores2)

    remote_scores = block_scores.masked_fill(forced, float("-inf"))
    finite_count = int(torch.isfinite(remote_scores).sum(dim=-1).min().item())
    if cfg.remote_topk <= 0 or finite_count <= 0:
        return forced
    remote_budget = min(int(cfg.remote_topk), finite_count)
    selected = torch.topk(remote_scores, k=remote_budget, dim=-1, largest=True, sorted=False).indices
    selected_mask = torch.zeros_like(forced)
    selected_mask.scatter_(-1, selected, True)
    return forced | selected_mask


def _should_apply_sparse(hidden_states: torch.Tensor, kv_seq_len: int, layer_idx: int, cfg: ESInfLLMV2Config) -> bool:
    if not cfg.enabled or layer_idx < cfg.skip_layers:
        return False
    if cfg.apply_decode_only and hidden_states.size(1) != 1:
        return False
    return kv_seq_len >= cfg.dense_len


def _prepare_next_layer_route(
    layer: torch.nn.Module,
    hidden: torch.Tensor,
    position_ids: torch.Tensor | None,
    past_key_value: Any,
    *,
    source: str,
) -> None:
    runtime: ESInfLLMV2Runtime = layer._esinfllmv2_runtime
    cfg = runtime.config
    if cfg.mode != "es_sparse" or not cfg.enabled or position_ids is None:
        return
    if source != cfg.lookahead_source:
        return
    if cfg.apply_decode_only and hidden.size(1) != 1:
        return

    next_idx = int(layer._esinfllmv2_layer_idx) + 1
    next_layer = getattr(layer, "_esinfllmv2_next_layer", None)
    if next_layer is None or next_idx < cfg.skip_layers:
        return

    cached_k = _get_cached_key(past_key_value, next_idx)
    if cached_k is None or cached_k.size(-2) < cfg.dense_len:
        return

    next_attn = next_layer.self_attn
    route_hidden = next_layer.input_layernorm(hidden)
    if cfg.route_query == "exact_next_wq":
        q_flat = next_attn.q_proj(route_hidden)
    elif cfg.route_query == "lowrank_next_wq":
        down, up = runtime.lowrank[next_idx]
        q_flat = _project_lowrank(route_hidden, down, up)
    else:
        raise ValueError(f"Unsupported route_query={cfg.route_query!r}")

    if cfg.route_q_scale != 1.0:
        q_flat = q_flat * float(cfg.route_q_scale)
    q = _shape_q(q_flat, next_attn.num_heads, next_attn.head_dim)
    seq_len = int(position_ids.max().item()) + 1
    cos, sin = next_attn.rotary_emb(q.to(torch.float32), seq_len=seq_len)
    apply_rotary_pos_emb = _get_forward_globals(next_attn)["apply_rotary_pos_emb"]
    q, _ = apply_rotary_pos_emb(q, q, cos, sin, position_ids)
    block_mask = _stage1_route_mask(q=q, k=cached_k, position_ids=position_ids, cfg=cfg)

    runtime.pending_block_masks[next_idx] = block_mask.detach()
    runtime.stats.prepared_routes += 1
    if source == "in":
        runtime.stats.prepared_from_input += 1
    elif source == "mid":
        runtime.stats.prepared_from_mid += 1
    elif source == "out":
        runtime.stats.prepared_from_output += 1


def _es_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_value: Any = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    if "padding_mask" in kwargs:
        warnings = _get_forward_globals(self.self_attn)["warnings"]
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. "
            "Please make sure use `attention_mask` instead.`"
        )

    _prepare_next_layer_route(self, hidden_states, position_ids, past_key_value, source="in")

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        **kwargs,
    )
    hidden_mid = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))
    _prepare_next_layer_route(self, hidden_mid, position_ids, past_key_value, source="mid")

    residual = hidden_mid
    hidden_states = self.post_attention_layernorm(hidden_mid)
    hidden_states = self.mlp(hidden_states)
    hidden_out = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))
    _prepare_next_layer_route(self, hidden_out, position_ids, past_key_value, source="out")

    outputs = (hidden_out,)
    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)
    return outputs


def _es_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_value: Any = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    runtime: ESInfLLMV2Runtime = self._esinfllmv2_runtime
    cfg = runtime.config
    layer_idx = int(self.layer_idx)
    block_mask = runtime.pending_block_masks.pop(layer_idx, None)

    if cfg.mode not in {"oracle_sparse", "es_sparse"}:
        runtime.stats.dense_attn_calls += 1
        return self._esinfllmv2_original_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

    if "padding_mask" in kwargs:
        warnings = _get_forward_globals(self)["warnings"]
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. "
            "Please make sure use `attention_mask` instead.`"
        )

    bsz, q_len, _ = hidden_states.size()
    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)
        query_states = torch.cat([F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
        key_states = torch.cat([F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
        value_states = torch.cat([F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = int(position_ids.max().item()) + 1 if position_ids is not None else q_len
    cos, sin = self.rotary_emb(value_states.to(torch.float32), seq_len=kv_seq_len)
    apply_rotary_pos_emb = _get_forward_globals(self)["apply_rotary_pos_emb"]
    repeat_kv = _get_forward_globals(self)["repeat_kv"]
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    actual_kv_seq_len = key_states.size(-2)
    should_sparse = _should_apply_sparse(hidden_states, actual_kv_seq_len, layer_idx, cfg)
    if cfg.mode == "es_sparse" and block_mask is None:
        should_sparse = False
        if cfg.enabled and hidden_states.size(1) == 1 and layer_idx >= cfg.skip_layers and actual_kv_seq_len >= cfg.dense_len:
            runtime.stats.missing_route_calls += 1

    if should_sparse and cfg.mode == "oracle_sparse":
        block_mask = _stage1_route_mask(q=query_states, k=key_states, position_ids=position_ids, cfg=cfg)
        runtime.stats.oracle_route_calls += 1

    key_states_repeated = repeat_kv(key_states, self.num_key_value_groups)
    value_states_repeated = repeat_kv(value_states, self.num_key_value_groups)
    attn_weights = torch.matmul(query_states, key_states_repeated.transpose(2, 3)) / math.sqrt(self.head_dim)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    if should_sparse and block_mask is not None:
        block_count = math.ceil(actual_kv_seq_len / cfg.block_size)
        block_mask = _pad_block_mask(block_mask.to(device=attn_weights.device), block_count)
        forced = _forced_block_mask(
            batch_size=bsz,
            kv_heads=self.num_key_value_heads,
            q_len=q_len,
            kv_seq_len=actual_kv_seq_len,
            query_positions=position_ids[:, -q_len:],
            cfg=cfg,
            device=attn_weights.device,
        )
        block_mask = block_mask | forced
        token_mask = _block_mask_to_token_mask(
            block_mask,
            kv_seq_len=actual_kv_seq_len,
            block_size=cfg.block_size,
            num_key_value_groups=self.num_key_value_groups,
        )
        attn_weights = attn_weights.masked_fill(~token_mask, torch.finfo(attn_weights.dtype).min)
        runtime.stats.routed_attn_calls += 1
    else:
        runtime.stats.dense_attn_calls += 1

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states_repeated)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum(F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp))
    else:
        attn_output = self.o_proj(attn_output)
    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value


def _load_lowrank_cache(path: str | None) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    if not path:
        return {}
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    raw = torch.load(cache_path, map_location="cpu")
    return {int(k): (v[0].cpu(), v[1].cpu()) for k, v in raw.items()}


def _save_lowrank_cache(path: str | None, lowrank: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> None:
    if not path:
        return
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({int(k): (v[0].cpu(), v[1].cpu()) for k, v in lowrank.items()}, cache_path)


def install_esinfllmv2(model: torch.nn.Module, config: ESInfLLMV2Config) -> ESInfLLMV2Runtime:
    runtime = ESInfLLMV2Runtime(config)
    layers = list(model.model.layers)
    if config.mode == "es_sparse" and config.route_query == "lowrank_next_wq":
        runtime.lowrank = _load_lowrank_cache(config.lowrank_cache_path)
        device = next(model.parameters()).device
        changed = False
        for layer_idx, layer in enumerate(layers):
            if layer_idx == 0:
                continue
            if layer_idx in runtime.lowrank:
                continue
            weight = layer.self_attn.q_proj.weight.detach().to(device=device, dtype=torch.float32)
            runtime.lowrank[layer_idx] = _lowrank_factors(weight, rank=config.rank, niter=config.svd_niter)
            changed = True
            torch.cuda.empty_cache()
        if changed:
            _save_lowrank_cache(config.lowrank_cache_path, runtime.lowrank)

    for layer_idx, layer in enumerate(layers):
        layer._esinfllmv2_original_forward = layer.forward
        layer._esinfllmv2_runtime = runtime
        layer._esinfllmv2_layer_idx = layer_idx
        layer._esinfllmv2_next_layer = layers[layer_idx + 1] if layer_idx + 1 < len(layers) else None
        layer.forward = types.MethodType(_es_layer_forward, layer)

        attn = layer.self_attn
        attn._es_forward_globals = _get_forward_globals(attn)
        attn._esinfllmv2_original_forward = attn.forward
        attn._esinfllmv2_runtime = runtime
        attn.forward = types.MethodType(_es_attention_forward, attn)

    model._esinfllmv2_runtime = runtime
    return runtime


def uninstall_esinfllmv2(model: torch.nn.Module) -> None:
    for layer in model.model.layers:
        if hasattr(layer, "_esinfllmv2_original_forward"):
            layer.forward = layer._esinfllmv2_original_forward
            delattr(layer, "_esinfllmv2_original_forward")
        for name in ("_esinfllmv2_runtime", "_esinfllmv2_layer_idx", "_esinfllmv2_next_layer"):
            if hasattr(layer, name):
                delattr(layer, name)

        attn = layer.self_attn
        if hasattr(attn, "_esinfllmv2_original_forward"):
            attn.forward = attn._esinfllmv2_original_forward
            delattr(attn, "_esinfllmv2_original_forward")
        for name in ("_esinfllmv2_runtime", "_es_forward_globals"):
            if hasattr(attn, name):
                delattr(attn, name)

    if hasattr(model, "_esinfllmv2_runtime"):
        delattr(model, "_esinfllmv2_runtime")
