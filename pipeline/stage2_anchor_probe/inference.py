import logging
import math
import os
from typing import Any, Dict, Optional

import torch

from litgpt import LLM
from litgpt.tokenizer import Tokenizer

from pipeline.model_utils import empty_cuda_cache, get_prompt_ids, initialize_litgpt_model, normalize_batch_input
from .runtime import STAGE2_METHODS, Stage2AnchorConfig, Stage2AnchorRuntime, decode_stage2_anchor

logger = logging.getLogger("main")


def _get_cfg(pipeline_params: Dict[str, Any]) -> Dict[str, Any]:
    method = str(pipeline_params.get("method", "qmid_stage2"))
    if method not in STAGE2_METHODS:
        raise ValueError(f"Unsupported stage2 anchor method: {method}")
    cfg = dict(pipeline_params.get(method, {}))
    cfg["token_budget"] = int(pipeline_params.get("token_budget", cfg.get("token_budget", 1024)))
    cfg.setdefault("window_size", 32)
    cfg.setdefault("kernel_size", 63)
    cfg.setdefault("skip_layers", 1)
    cfg.setdefault("max_records", int(os.environ.get("STAGE2_PROBE_MAX_RECORDS", "0")))
    cfg.setdefault("head_rows", os.environ.get("STAGE2_PROBE_HEAD_ROWS", "1").lower() not in {"0", "false", "no"})
    if "candidates" not in cfg:
        raw_candidates = os.environ.get("STAGE2_PROBE_CANDIDATES", "")
        if raw_candidates:
            cfg["candidates"] = tuple(x.strip() for x in raw_candidates.split(",") if x.strip())
        else:
            cfg["candidates"] = ("rocket_qout", "lookahead_qmid", "lookahead_qin")
    return cfg


def initialize_model_tokenizer(pipeline_params: Dict[str, Any]):
    return initialize_litgpt_model(pipeline_params, path_label="Stage2AnchorProbe")


def _compute_params(prompt_len: int, max_new_tokens: int, token_budget: int) -> Dict[str, int | float]:
    sequence_length = int(prompt_len) + int(max_new_tokens)
    token_budget = min(max(1, int(token_budget)), max(1, sequence_length))
    compression_ratio = max(1.0, float(sequence_length) / float(token_budget))
    rho = min(0.2 + 0.06 * math.log2(compression_ratio), 0.8)
    token_capacity_budget = int(float(sequence_length) / (compression_ratio**rho))
    prompt_budget = max(min(int(prompt_len), int(token_capacity_budget) - int(max_new_tokens)), 0)
    topk_budget = max(1, min(round(token_budget / 2), int(token_capacity_budget)))
    stage2_ratio = max(1.0, float(token_capacity_budget) / float(token_budget))
    return {
        "rho": rho,
        "token_capacity_budget": token_capacity_budget,
        "prompt_budget": prompt_budget,
        "topk_budget": topk_budget,
        "compression_ratio": stage2_ratio,
    }


def _build_runtime(
    model: LLM,
    pipeline_params: Dict[str, Any],
    *,
    prompt_len: int,
    max_new_tokens: int,
    sample_id: str,
) -> Stage2AnchorRuntime:
    method = str(pipeline_params.get("method", "qmid_stage2"))
    method_cfg = _get_cfg(pipeline_params)
    budgets = _compute_params(prompt_len, max_new_tokens, int(method_cfg["token_budget"]))
    cfg = Stage2AnchorConfig(
        method=method,
        prompt_len=prompt_len,
        prompt_budget=int(budgets["prompt_budget"]),
        topk_budget=int(budgets["topk_budget"]),
        compression_ratio=float(budgets["compression_ratio"]),
        window_size=int(method_cfg["window_size"]),
        kernel_size=int(method_cfg["kernel_size"]),
        skip_layers=int(method_cfg["skip_layers"]),
        attention_heads=model.config.n_head,
        kv_heads=int(getattr(model.config, "n_query_groups", model.config.n_head)),
        output_dir=str(pipeline_params.get("stage2_probe_output_dir", "")),
        task=str(pipeline_params.get("stage2_probe_task", "")),
        model_name=str(pipeline_params.get("model_name", "")),
        token_budget=int(method_cfg["token_budget"]),
        sample_id=sample_id,
        max_records=int(method_cfg.get("max_records", 0)),
        head_rows=bool(method_cfg.get("head_rows", True)),
        candidates=tuple(str(x) for x in method_cfg.get("candidates", ())),
    )
    unsupported_candidates = set(cfg.candidates) - {"rocket_qout", "lookahead_qmid", "lookahead_qin"}
    if unsupported_candidates:
        raise ValueError(f"Unsupported stage2 probe candidates: {sorted(unsupported_candidates)}")
    runtime = Stage2AnchorRuntime(cfg)
    runtime.attach_model(model.model)
    logger.info(
        "%s budgets: token_budget=%s rho=%.4f prompt_budget=%s topk_budget=%s c2=%.4f skip_layers=%s output=%s sample=%s",
        method,
        method_cfg["token_budget"],
        budgets["rho"],
        budgets["prompt_budget"],
        budgets["topk_budget"],
        budgets["compression_ratio"],
        method_cfg["skip_layers"],
        cfg.output_dir,
        sample_id,
    )
    return runtime


def batch_generate(
    batched_input: torch.Tensor,
    model: LLM,
    tokenizer: Tokenizer,
    max_new_tokens: int,
    pipeline_params: Optional[Dict[str, Any]] = None,
):
    model.eval()
    params = pipeline_params or {}
    prompts, prompt_ids_by_idx = normalize_batch_input(batched_input, model, tokenizer)

    responses = []
    for idx, prompt in enumerate(prompts):
        prompt_ids = get_prompt_ids(
            prompt=prompt,
            prompt_ids_by_idx=prompt_ids_by_idx,
            idx=idx,
            tokenizer=tokenizer,
            model=model,
            bos=False,
        )
        sample_counter = int(params.get("_stage2_probe_sample_counter", 0))
        params["_stage2_probe_sample_counter"] = sample_counter + 1
        sample_offset = int(os.environ.get("LONGBENCH_SAMPLE_OFFSET", "0"))
        runtime = _build_runtime(
            model,
            params,
            prompt_len=int(prompt_ids.numel()),
            max_new_tokens=max_new_tokens,
            sample_id=str(sample_offset + sample_counter),
        )
        responses.append(
            decode_stage2_anchor(
                prompt_ids=prompt_ids,
                model=model,
                runtime=runtime,
                max_new_tokens=max_new_tokens,
            )
        )

    empty_cuda_cache()
    return responses
