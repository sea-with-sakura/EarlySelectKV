import logging
import math
from typing import Any, Dict, Optional

import torch

from litgpt import LLM
from litgpt.tokenizer import Tokenizer

from pipeline.model_utils import empty_cuda_cache, get_prompt_ids, initialize_litgpt_model, normalize_batch_input
from .runtime import RocketKVConfig, RocketKVRuntime, decode_rocketkv

logger = logging.getLogger("main")


def _get_rocketkv_cfg(pipeline_params: Dict[str, Any]) -> Dict[str, Any]:
    method = str(pipeline_params.get("method", "rocketkv"))
    if method not in {"rocketkv", "rocketkv_topk"}:
        raise ValueError(f"Unsupported RocketKV method: {method}")

    cfg_key = method
    rocket_cfg = dict(pipeline_params.get(cfg_key, {}))
    rocket_cfg["token_budget"] = int(pipeline_params.get("token_budget", rocket_cfg.get("token_budget", 1024)))
    rocket_cfg.setdefault("window_size", 32)
    rocket_cfg.setdefault("kernel_size", 63)
    rocket_cfg.setdefault("skip_layers", 0)
    return rocket_cfg


def initialize_model_tokenizer(pipeline_params: Dict[str, Any]):
    return initialize_litgpt_model(pipeline_params, path_label="RocketKV-family")


def _compute_rocketkv_params(prompt_len: int, max_new_tokens: int, token_budget: int) -> Dict[str, int | float]:
    sequence_length = int(prompt_len) + int(max_new_tokens)
    token_budget = min(max(1, int(token_budget)), max(1, sequence_length))
    compression_ratio = max(1.0, float(sequence_length) / float(token_budget))
    rho = min(0.2 + 0.06 * math.log2(compression_ratio), 0.8)
    token_capacity_budget = max(
        int(float(sequence_length) / (compression_ratio**rho)),
        min(2 * int(max_new_tokens), int(sequence_length)),
    )
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
) -> RocketKVRuntime:
    method = str(pipeline_params.get("method", "rocketkv"))
    rocket_cfg = _get_rocketkv_cfg(pipeline_params)
    budgets = _compute_rocketkv_params(prompt_len, max_new_tokens, rocket_cfg["token_budget"])
    cfg = RocketKVConfig(
        prompt_len=prompt_len,
        prompt_budget=int(budgets["prompt_budget"]),
        topk_budget=int(budgets["topk_budget"]),
        compression_ratio=float(budgets["compression_ratio"]),
        window_size=int(rocket_cfg["window_size"]),
        kernel_size=int(rocket_cfg["kernel_size"]),
        skip_layers=int(rocket_cfg["skip_layers"]),
        attention_heads=model.config.n_head,
        kv_heads=int(getattr(model.config, "n_query_groups", model.config.n_head)),
        routing_mode="exact_topk" if method == "rocketkv_topk" else "hsa",
    )
    logger.info(
        "%s budgets: token_budget=%s rho=%.4f prompt_budget=%s topk_budget=%s c2=%.4f",
        method,
        rocket_cfg["token_budget"],
        budgets["rho"],
        budgets["prompt_budget"],
        budgets["topk_budget"],
        budgets["compression_ratio"],
    )
    return RocketKVRuntime(cfg)


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
            decode_rocketkv(
                prompt_ids=prompt_ids,
                model=model,
                runtime=runtime,
                max_new_tokens=max_new_tokens,
            )
        )

    empty_cuda_cache()
    return responses
