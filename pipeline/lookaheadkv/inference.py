import logging
import math
from typing import Any, Dict, Optional

import torch

from litgpt import LLM
from litgpt.tokenizer import Tokenizer

from pipeline.config_utils import canonical_method_key, method_config_keys
from pipeline.model_utils import empty_cuda_cache, get_prompt_ids, initialize_litgpt_model, normalize_batch_input
from .runtime import (
    DEFAULT_DECODE_LOCAL_WINDOW_SIZE,
    LookaheadKVConfig,
    LookaheadKVRuntime,
    decode_lookaheadkv,
)

logger = logging.getLogger("main")

EARLYSELECTKV_METHODS = {
    "earlyselectkv",
    "earlyselectkv_in",
    "earlyselectkv_in_mid",
    "earlyselectkv_in_mid_local",
    "earlyselectkv_topk",
    "earlyselectkv_local",
    "earlyselectkv_hsa_local",
    "earlyselectkv_mid_svd_r64",
    "earlyselectkv_mid_svd_r128",
    "earlyselectkv_mid_svd_r256",
    "earlyselectkv_mid_svd_r512",
}


def _has_method_cfg(pipeline_params: Dict[str, Any], method: str) -> bool:
    return any(key in pipeline_params for key in method_config_keys(method))


def _get_method_cfg(pipeline_params: Dict[str, Any], method: str) -> Dict[str, Any]:
    for key in method_config_keys(method):
        value = pipeline_params.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _get_lookaheadkv_cfg(pipeline_params: Dict[str, Any]) -> Dict[str, Any]:
    method = str(pipeline_params.get("method", "earlyselectkv"))
    canonical_method = canonical_method_key(method)
    if canonical_method not in EARLYSELECTKV_METHODS:
        raise ValueError(f"Unsupported EarlySelectKV method: {method}")
    if canonical_method == "earlyselectkv_in_mid" and not _has_method_cfg(pipeline_params, canonical_method):
        raise KeyError(
            "Missing pipeline_params['earlyselectkv_in_mid']; this method is only enabled "
            "for model configs that explicitly opt in."
        )

    if canonical_method == "earlyselectkv_in_mid_local":
        lookahead_cfg = _get_method_cfg(pipeline_params, "earlyselectkv_in_mid")
        lookahead_cfg.update(_get_method_cfg(pipeline_params, canonical_method))
    else:
        lookahead_cfg = _get_method_cfg(pipeline_params, canonical_method)
    lookahead_cfg["token_budget"] = int(pipeline_params.get("token_budget", lookahead_cfg.get("token_budget", 1024)))
    lookahead_cfg.setdefault("window_size", 32)
    lookahead_cfg.setdefault("kernel_size", 63)
    lookahead_cfg.setdefault("skip_layers", 0)
    lookahead_cfg.setdefault("divide_budget_by_group_size", False)
    lookahead_cfg.setdefault("group_shared_within_kv_group", True)
    lookahead_cfg.setdefault("decode_local_budget", None)
    default_source = "in_mid" if canonical_method == "earlyselectkv_in_mid_local" else "mid"
    lookahead_cfg.setdefault("lookahead_source", default_source)
    lookahead_cfg.setdefault("lookahead_mid_until_layer", 20)
    lookahead_cfg.setdefault(
        "lookahead_query_mode",
        "svd_wq" if canonical_method.startswith("earlyselectkv_mid_svd") else "next_wq",
    )
    lookahead_cfg.setdefault("lookahead_svd_path", None)
    return lookahead_cfg


def initialize_model_tokenizer(pipeline_params: Dict[str, Any]):
    return initialize_litgpt_model(pipeline_params, path_label="EarlySelectKV")


def _compute_lookaheadkv_params(prompt_len: int, max_new_tokens: int, token_budget: int) -> Dict[str, int | float]:
    total_capacity = int(prompt_len) + int(max_new_tokens)
    token_budget = max(1, int(token_budget))
    compression_ratio = max(1.0, float(total_capacity) / float(token_budget))
    rho = min(0.2 + 0.06 * math.log2(compression_ratio), 0.8)
    token_capacity_budget = max(
        int(float(total_capacity) / (compression_ratio**rho)),
        min(2 * int(max_new_tokens), int(total_capacity)),
    )
    prompt_budget = max(int(token_capacity_budget) - int(max_new_tokens), 0)
    topk_budget = max(int(token_budget) // 2, 1)
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
) -> LookaheadKVRuntime:
    method = str(pipeline_params.get("method", "earlyselectkv"))
    canonical_method = canonical_method_key(method)
    lookahead_cfg = _get_lookaheadkv_cfg(pipeline_params)
    routing_mode = "hsa"
    if canonical_method == "earlyselectkv_topk":
        routing_mode = "exact_topk"
    elif canonical_method == "earlyselectkv_local":
        routing_mode = "chunk1_local"
    elif canonical_method in {"earlyselectkv_hsa_local", "earlyselectkv_in_mid_local"}:
        routing_mode = "hsa_local"

    decode_local_budget = lookahead_cfg.get("decode_local_budget")

    budgets = _compute_lookaheadkv_params(prompt_len, max_new_tokens, lookahead_cfg["token_budget"])
    cfg = LookaheadKVConfig(
        prompt_len=prompt_len,
        prompt_budget=int(budgets["prompt_budget"]),
        topk_budget=int(budgets["topk_budget"]),
        compression_ratio=float(budgets["compression_ratio"]),
        window_size=int(lookahead_cfg["window_size"]),
        kernel_size=int(lookahead_cfg["kernel_size"]),
        skip_layers=int(lookahead_cfg["skip_layers"]),
        attention_heads=model.config.n_head,
        kv_heads=int(getattr(model.config, "n_query_groups", model.config.n_head)),
        routing_mode=routing_mode,
        decode_local_budget=(
            None if decode_local_budget is None else int(decode_local_budget)
        ),
        lookahead_source=str(lookahead_cfg.get("lookahead_source", "mid")),
        lookahead_mid_until_layer=int(lookahead_cfg.get("lookahead_mid_until_layer", 20)),
        lookahead_query_mode=str(lookahead_cfg.get("lookahead_query_mode", "next_wq")),
        lookahead_svd_path=(
            None
            if lookahead_cfg.get("lookahead_svd_path") is None
            else str(lookahead_cfg["lookahead_svd_path"])
        ),
    )
    log_fields = [
        f"token_budget={lookahead_cfg['token_budget']}",
        f"rho={float(budgets['rho']):.4f}",
        f"prompt_budget={int(budgets['prompt_budget'])}",
        f"topk_budget={int(budgets['topk_budget'])}",
        f"c2={float(budgets['compression_ratio']):.4f}",
        f"divide_by_group={bool(lookahead_cfg['divide_budget_by_group_size'])}",
        f"group_shared={bool(lookahead_cfg['group_shared_within_kv_group'])}",
        f"routing_mode={routing_mode}",
        f"lookahead_source={str(lookahead_cfg.get('lookahead_source', 'mid'))}",
        f"lookahead_mid_until_layer={int(lookahead_cfg.get('lookahead_mid_until_layer', 20))}",
        f"lookahead_query_mode={cfg.lookahead_query_mode}",
    ]
    if cfg.lookahead_query_mode == "svd_wq":
        log_fields.append(f"lookahead_svd_path={cfg.lookahead_svd_path}")
    if routing_mode in {"chunk1_local", "hsa_local"}:
        local_budget = cfg.decode_local_budget
        if local_budget is None or int(local_budget) < 0:
            local_budget = min(DEFAULT_DECODE_LOCAL_WINDOW_SIZE, max(16, int(cfg.topk_budget) // 8))
        local_budget = max(0, min(int(local_budget), int(cfg.topk_budget)))
        log_fields.extend(
            [
                f"decode_local_budget={local_budget}",
                f"decode_local_cap={DEFAULT_DECODE_LOCAL_WINDOW_SIZE}",
                f"decode_route_budget={max(int(cfg.topk_budget) - local_budget, 0)}",
            ]
        )
    logger.info("%s budgets: %s", canonical_method, " ".join(log_fields))
    return LookaheadKVRuntime(cfg)


def batch_generate(
    batched_input: torch.Tensor,
    model: LLM,
    tokenizer: Tokenizer,
    max_new_tokens: int,
    pipeline_params: Optional[Dict[str, Any]] = None,
):
    model.eval()

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
        runtime = _build_runtime(
            model, pipeline_params or {}, prompt_len=int(prompt_ids.numel()), max_new_tokens=max_new_tokens
        )
        responses.append(
            decode_lookaheadkv(
                prompt_ids=prompt_ids,
                model=model,
                runtime=runtime,
                max_new_tokens=max_new_tokens,
            )
        )

    empty_cuda_cache()
    return responses
