from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from litgpt import LLM
from litgpt.model import apply_rope

from pipeline.auxiliary_kv_cache import build_chunk_summary_values, select_hsa_indices_from_summary
from pipeline.runtime_decode_utils import (
    compress_standard_prompt_kv,
    compute_chunk_size,
    run_sparse_decode,
    setup_prompt_kv_cache,
    sorted_topk_indices,
)
from .metrics import append_jsonl, build_stage2_record, summarize_probe_dir
from .metrics import exact_topk_from_probs, grouped_true_probs, query_quality_records, route_mass
from .monkey_patch import install_stage2_anchor_attention, uninstall_stage2_anchor_attention


STAGE2_METHODS = {
    "anchor_quality_stage2",
    "dense_stage2",
    "topk_stage2",
    "qout_stage2",
    "qmid_stage2",
    "qin_stage2",
    "qinwithmid_stage2",
}


@dataclass
class Stage2AnchorConfig:
    method: str
    prompt_len: int
    prompt_budget: int
    topk_budget: int
    compression_ratio: float
    window_size: int
    kernel_size: int
    skip_layers: int
    attention_heads: int
    kv_heads: int
    output_dir: str
    task: str = ""
    model_name: str = ""
    token_budget: int = 0
    sample_id: str = ""
    max_records: int = 0
    head_rows: bool = True
    candidates: tuple[str, ...] = ("rocket_qout", "lookahead_qmid", "lookahead_qin")


class Stage2AnchorRuntime:
    """RocketKV-style stage-2 probe over a SnapKV-evicted prompt cache."""

    _sorted_topk = staticmethod(sorted_topk_indices)

    def __init__(self, cfg: Stage2AnchorConfig) -> None:
        self.cfg = cfg
        if cfg.method not in STAGE2_METHODS:
            raise ValueError(f"Unsupported stage2 anchor method: {cfg.method}")
        if cfg.attention_heads % cfg.kv_heads != 0:
            raise ValueError(f"attention_heads={cfg.attention_heads} must be divisible by kv_heads={cfg.kv_heads}")
        self.active_prompt_len = min(int(cfg.prompt_len), int(cfg.prompt_budget))
        self.chunk_size = compute_chunk_size(float(cfg.compression_ratio))
        self._compressed_layers: set[int] = set()
        self._blocks = None
        self.prompt_k_by_layer: dict[int, torch.Tensor] = {}
        self.prompt_v_by_layer: dict[int, torch.Tensor] = {}
        self.input_q_by_layer: dict[int, torch.Tensor] = {}
        self.mid_q_by_layer: dict[int, torch.Tensor] = {}
        self._probe_rows: list[dict[str, object]] = []
        self._head_rows: list[dict[str, object]] = []
        self._records_seen = 0

    def attach_model(self, model: torch.nn.Module) -> None:
        self._blocks = list(model.transformer.h)

    def reset(self) -> None:
        self._compressed_layers.clear()
        self.prompt_k_by_layer.clear()
        self.prompt_v_by_layer.clear()
        self.input_q_by_layer.clear()
        self.mid_q_by_layer.clear()
        self._probe_rows.clear()
        self._head_rows.clear()
        self._records_seen = 0

    def _is_decode(self, *, input_pos: Optional[torch.Tensor], time_steps: int) -> bool:
        if input_pos is None or int(time_steps) != 1:
            return False
        pos = input_pos if input_pos.dim() == 1 else input_pos[0]
        if pos.numel() != 1:
            return False
        return int(pos.item()) >= int(self.cfg.prompt_len)

    def _logical_pos(self, input_pos: Optional[torch.Tensor]) -> int:
        if input_pos is None:
            return -1
        pos = input_pos if input_pos.dim() == 1 else input_pos[0]
        return int(pos.reshape(-1)[0].item()) if pos.numel() else -1

    def _decode_step(self, input_pos: Optional[torch.Tensor]) -> int:
        pos = self._logical_pos(input_pos)
        return max(0, pos - int(self.cfg.prompt_len))

    def _compute_next_layer_query(self, *, layer_idx: int, hidden: torch.Tensor) -> Optional[torch.Tensor]:
        if hidden.size(1) != 1 or self._blocks is None:
            return None
        next_idx = int(layer_idx) + 1
        if next_idx >= len(self._blocks):
            return None
        next_block = self._blocks[next_idx]
        next_attn = next_block.attn
        hidden_normed = next_block.norm_1(hidden)
        qkv = next_attn.qkv(hidden_normed)
        query_size = next_attn.config.n_head * next_attn.config.head_size
        q = qkv[..., :query_size]
        if next_attn.config.norm_qk and next_attn.config.norm_qk_type == "olmo2":
            q = next_attn.norm_q(q)
        q = q.view(hidden.size(0), hidden.size(1), next_attn.config.n_head, next_attn.config.head_size)
        q = q.transpose(1, 2)
        if next_attn.config.norm_qk and next_attn.config.norm_qk_type == "default":
            q = next_attn.norm_q(q)
        return q.detach()

    def observe_layer_input(self, *, layer_idx: int, hidden: torch.Tensor) -> None:
        if layer_idx == 0:
            self.input_q_by_layer.clear()
        q = self._compute_next_layer_query(layer_idx=layer_idx, hidden=hidden)
        if q is not None:
            self.input_q_by_layer[layer_idx + 1] = q

    def observe_layer_mid(self, *, layer_idx: int, hidden: torch.Tensor) -> None:
        if layer_idx == 0:
            self.mid_q_by_layer.clear()
        q = self._compute_next_layer_query(layer_idx=layer_idx, hidden=hidden)
        if q is not None:
            self.mid_q_by_layer[layer_idx + 1] = q

    def maybe_compress_prefill(
        self,
        *,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        input_pos_maxp1: Optional[int],
    ) -> None:
        if layer_idx in self._compressed_layers:
            return
        key_len = int(input_pos_maxp1) if input_pos_maxp1 is not None else int(k.size(2))
        if key_len != int(self.cfg.prompt_len):
            return
        self._compressed_layers.add(layer_idx)
        compressed = compress_standard_prompt_kv(
            q=q,
            k=k,
            v=v,
            key_len=key_len,
            active_prompt_len=self.active_prompt_len,
            window_size=self.cfg.window_size,
            kernel_size=self.cfg.kernel_size,
            attention_heads=self.cfg.attention_heads,
            kv_heads=self.cfg.kv_heads,
        )
        self.prompt_k_by_layer[layer_idx] = compressed.prompt_k
        self.prompt_v_by_layer[layer_idx] = compressed.prompt_v

    def setup_decode_kv_cache(
        self,
        *,
        gpt: torch.nn.Module,
        logical_max_seq_length: int,
        max_new_tokens: int,
        device: torch.device,
    ) -> None:
        setup_prompt_kv_cache(
            gpt=gpt,
            prompt_k_by_layer=self.prompt_k_by_layer,
            prompt_v_by_layer=self.prompt_v_by_layer,
            active_prompt_len=self.active_prompt_len,
            logical_max_seq_length=logical_max_seq_length,
            max_new_tokens=max_new_tokens,
            device=device,
            runtime_name="Stage2AnchorProbe",
        )

    def cache_position_for_input(self, input_pos: torch.Tensor) -> torch.Tensor:
        pos = input_pos.to(dtype=torch.long) if input_pos.dim() == 1 else input_pos[0].to(dtype=torch.long)
        if pos.numel() == 0 or pos.numel() > 1:
            return pos
        logical_pos = int(pos.item())
        if logical_pos < int(self.cfg.prompt_len):
            return pos
        compact_pos = self.active_prompt_len + (logical_pos - int(self.cfg.prompt_len))
        return torch.tensor([compact_pos], device=pos.device, dtype=torch.long)

    def _apply_rope(self, *, q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_n_elem: int) -> torch.Tensor:
        q_roped = apply_rope(q[..., :rope_n_elem], cos, sin)
        return torch.cat((q_roped, q[..., rope_n_elem:]), dim=-1)

    def _routing_query(
        self,
        *,
        layer_idx: int,
        q_true: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rope_n_elem: int,
    ) -> torch.Tensor:
        if self.cfg.method in {"anchor_quality_stage2", "dense_stage2", "topk_stage2", "qout_stage2"}:
            return q_true
        if self.cfg.method == "qmid_stage2":
            q = self.mid_q_by_layer.get(layer_idx)
        elif self.cfg.method == "qin_stage2":
            q = self.input_q_by_layer.get(layer_idx)
        elif self.cfg.method == "qinwithmid_stage2":
            q = self.mid_q_by_layer.get(layer_idx) if int(layer_idx) <= 20 else self.input_q_by_layer.get(layer_idx)
        else:
            q = None
        if q is None or q.shape[:3] != q_true.shape[:3] or q.size(-1) != q_true.size(-1):
            return q_true
        return self._apply_rope(q=q.to(device=q_true.device, dtype=q_true.dtype), cos=cos, sin=sin, rope_n_elem=rope_n_elem)

    def select_attention_indices(
        self,
        *,
        layer_idx: int,
        q_true: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rope_n_elem: int,
        input_pos: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self._is_decode(input_pos=input_pos, time_steps=int(q_true.size(2))):
            return None
        if layer_idx < int(self.cfg.skip_layers):
            return None
        key_len = int(k.size(2))
        if key_len <= 0 or int(self.cfg.topk_budget) >= key_len:
            return None
        if self.cfg.method in {"anchor_quality_stage2", "dense_stage2"}:
            return None
        if self.cfg.method == "topk_stage2":
            return self._select_exact_topk(q=q_true, k=k)
        routing_q = self._routing_query(layer_idx=layer_idx, q_true=q_true, cos=cos, sin=sin, rope_n_elem=rope_n_elem)
        return self._select_hsa(q=routing_q, k=k)

    def _select_exact_topk(self, *, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        q_grouped = q.reshape(q.size(0), self.cfg.kv_heads, -1, q.size(2), q.size(-1))
        scores = torch.matmul(q_grouped, k.unsqueeze(2).transpose(-1, -2)) / math.sqrt(q.size(-1))
        probs = torch.softmax(scores, dim=-1).mean(dim=2).squeeze(-2)
        return self._sorted_topk(probs, min(int(self.cfg.topk_budget), int(k.size(2))))

    def _select_hsa(self, *, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        key_len = int(k.size(2))
        summary = build_chunk_summary_values(k, chunk_size=self.chunk_size, capacity_len=key_len)
        return select_hsa_indices_from_summary(
            q,
            summary,
            key_len=key_len,
            chunk_size=self.chunk_size,
            compression_ratio=self.cfg.compression_ratio,
            topk_budget=self.cfg.topk_budget,
            kv_heads=self.cfg.kv_heads,
            sorted_topk=False,
        )

    def record_stage2_probe(
        self,
        *,
        layer_idx: int,
        q_true: torch.Tensor,
        k: torch.Tensor,
        route_indices: Optional[torch.Tensor],
        input_pos: Optional[torch.Tensor],
    ) -> None:
        if not self._is_decode(input_pos=input_pos, time_steps=int(q_true.size(2))):
            return
        if layer_idx < int(self.cfg.skip_layers):
            return
        if int(self.cfg.topk_budget) >= int(k.size(2)):
            return
        if self.cfg.max_records > 0 and self._records_seen >= int(self.cfg.max_records):
            return
        num_layers = len(self._blocks) if self._blocks is not None else 0
        if self.cfg.method == "anchor_quality_stage2":
            return
        row = build_stage2_record(
            method=self.cfg.method,
            task=self.cfg.task,
            model_name=self.cfg.model_name,
            token_budget=self.cfg.token_budget,
            sample_id=self.cfg.sample_id,
            layer_idx=layer_idx,
            num_layers=num_layers,
            logical_pos=self._logical_pos(input_pos),
            decode_step=self._decode_step(input_pos),
            q_true=q_true,
            k=k,
            route_indices=route_indices,
            topk_budget=self.cfg.topk_budget,
            attention_heads=self.cfg.attention_heads,
            kv_heads=self.cfg.kv_heads,
            page_size=self.chunk_size,
            active_prompt_len=self.active_prompt_len,
            window_size=self.cfg.window_size,
        )
        self._probe_rows.append(row)
        self._records_seen += 1
        if len(self._probe_rows) >= 256:
            self.flush_q_probe()

    def _candidate_query(
        self,
        *,
        candidate_method: str,
        layer_idx: int,
        q_true: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        rope_n_elem: int | None = None,
    ) -> torch.Tensor | None:
        if candidate_method in {"exact_topk", "rocket_qout"}:
            return q_true
        if candidate_method == "lookahead_qmid":
            q = self.mid_q_by_layer.get(layer_idx)
        elif candidate_method == "lookahead_qin":
            q = self.input_q_by_layer.get(layer_idx)
        else:
            q = None
        if q is None or q.shape[:3] != q_true.shape[:3] or q.size(-1) != q_true.size(-1):
            return None
        if cos is None or sin is None or rope_n_elem is None:
            return q.to(device=q_true.device, dtype=q_true.dtype)
        return self._apply_rope(q=q.to(device=q_true.device, dtype=q_true.dtype), cos=cos, sin=sin, rope_n_elem=rope_n_elem)

    def _candidate_query_raw(
        self,
        *,
        candidate_method: str,
        layer_idx: int,
        q_true_raw: torch.Tensor,
    ) -> torch.Tensor | None:
        if candidate_method in {"exact_topk", "rocket_qout"}:
            return q_true_raw
        if candidate_method == "lookahead_qmid":
            q = self.mid_q_by_layer.get(layer_idx)
        elif candidate_method == "lookahead_qin":
            q = self.input_q_by_layer.get(layer_idx)
        else:
            q = None
        if q is None or q.shape[:3] != q_true_raw.shape[:3] or q.size(-1) != q_true_raw.size(-1):
            return None
        return q.to(device=q_true_raw.device, dtype=q_true_raw.dtype)

    def record_anchor_quality_probe(
        self,
        *,
        layer_idx: int,
        q_true: torch.Tensor,
        q_true_raw: Optional[torch.Tensor] = None,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rope_n_elem: int,
        input_pos: Optional[torch.Tensor],
    ) -> None:
        if self.cfg.method != "anchor_quality_stage2":
            self.record_stage2_probe(layer_idx=layer_idx, q_true=q_true, k=k, route_indices=None, input_pos=input_pos)
            return
        if not self._is_decode(input_pos=input_pos, time_steps=int(q_true.size(2))):
            return
        if layer_idx < int(self.cfg.skip_layers):
            return
        key_len = int(k.size(2))
        if int(self.cfg.topk_budget) >= key_len:
            return
        if self.cfg.max_records > 0 and self._records_seen >= int(self.cfg.max_records):
            return

        num_layers = len(self._blocks) if self._blocks is not None else 0
        logical_pos = self._logical_pos(input_pos)
        decode_step = self._decode_step(input_pos)
        probs = grouped_true_probs(
            q_true=q_true,
            k=k,
            attention_heads=self.cfg.attention_heads,
            kv_heads=self.cfg.kv_heads,
        )
        oracle = exact_topk_from_probs(probs, self.cfg.topk_budget)
        topk_mass = route_mass(probs, oracle)
        base_kwargs = {
            "task": self.cfg.task,
            "model_name": self.cfg.model_name,
            "token_budget": self.cfg.token_budget,
            "sample_id": self.cfg.sample_id,
            "layer_idx": layer_idx,
            "num_layers": num_layers,
            "logical_pos": logical_pos,
            "decode_step": decode_step,
            "q_true": q_true,
            "k": k,
            "topk_budget": self.cfg.topk_budget,
            "attention_heads": self.cfg.attention_heads,
            "kv_heads": self.cfg.kv_heads,
            "page_size": self.chunk_size,
            "active_prompt_len": self.active_prompt_len,
            "window_size": self.cfg.window_size,
            "probs": probs,
            "oracle": oracle,
            "topk_mass": topk_mass,
        }

        rows: list[dict[str, object]] = [
            build_stage2_record(
                method=self.cfg.method,
                route_indices=oracle,
                candidate_method="exact_topk",
                **base_kwargs,
            )
        ]
        head_rows: list[dict[str, object]] = []
        base_r = round(q_true.size(-1) * self.chunk_size / max(float(self.cfg.compression_ratio), 1.0))
        channel_r = max(1, min(q_true.size(-1), int(base_r)))

        for candidate_method in self.cfg.candidates:
            q_candidate_raw = self._candidate_query_raw(
                candidate_method=candidate_method,
                layer_idx=layer_idx,
                q_true_raw=q_true if q_true_raw is None else q_true_raw,
            )
            q_candidate = self._candidate_query(
                candidate_method=candidate_method,
                layer_idx=layer_idx,
                q_true=q_true,
                cos=cos,
                sin=sin,
                rope_n_elem=rope_n_elem,
            )
            if q_candidate is None or q_candidate_raw is None:
                continue
            route = self._select_hsa(q=q_candidate, k=k)
            query_row, per_head = query_quality_records(
                method=candidate_method,
                task=self.cfg.task,
                model_name=self.cfg.model_name,
                token_budget=self.cfg.token_budget,
                sample_id=self.cfg.sample_id,
                layer_idx=layer_idx,
                num_layers=num_layers,
                logical_pos=logical_pos,
                decode_step=decode_step,
                q_candidate=q_candidate_raw,
                q_true=q_true if q_true_raw is None else q_true_raw,
                channel_budget_r=channel_r,
                head_rows=bool(self.cfg.head_rows),
            )
            selection_row = build_stage2_record(
                method=self.cfg.method,
                route_indices=route,
                candidate_method=candidate_method,
                **base_kwargs,
            )
            selection_row.update(query_row)
            rows.append(selection_row)
            head_rows.extend(per_head)

        self._probe_rows.extend(rows)
        self._head_rows.extend(head_rows)
        self._records_seen += 1
        if len(self._probe_rows) >= 256 or len(self._head_rows) >= 4096:
            self.flush_q_probe()

    def flush_q_probe(self) -> None:
        if not self.cfg.output_dir:
            self._probe_rows.clear()
            self._head_rows.clear()
            return
        output = Path(self.cfg.output_dir)
        append_jsonl(output / "stage2_probe_metrics.jsonl", self._probe_rows)
        append_jsonl(output / "stage2_probe_head_metrics.jsonl", self._head_rows)
        self._probe_rows.clear()
        self._head_rows.clear()


def decode_stage2_anchor(
    *,
    prompt_ids: torch.Tensor,
    model: LLM,
    runtime: Stage2AnchorRuntime,
    max_new_tokens: int,
) -> str:
    return run_sparse_decode(
        prompt_ids=prompt_ids,
        model=model,
        runtime=runtime,
        max_new_tokens=max_new_tokens,
        install_attention=install_stage2_anchor_attention,
        uninstall_attention=uninstall_stage2_anchor_attention,
    )
