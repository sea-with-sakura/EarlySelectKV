import logging
from typing import Any, Dict

import torch

from litgpt import LLM
from litgpt.tokenizer import Tokenizer

from pipeline.model_utils import empty_cuda_cache, get_prompt_ids, initialize_litgpt_model, normalize_batch_input
from .runtime import TopKConfig, TopKRuntime, decode_topk

logger = logging.getLogger("main")


def _get_topk_cfg(pipeline_params: Dict[str, Any]) -> Dict[str, Any]:
    topk_cfg = dict(pipeline_params.get("topk", {}))
    topk_cfg["token_budget"] = int(pipeline_params.get("token_budget", topk_cfg.get("token_budget", 1024)))
    topk_cfg.setdefault("group_shared_within_kv_group", True)
    topk_cfg.setdefault("skip_layers", 0)
    return topk_cfg


def initialize_model_tokenizer(pipeline_params):
    model, tokenizer = initialize_litgpt_model(pipeline_params, path_label="TopK")
    topk_cfg = _get_topk_cfg(pipeline_params)
    model._topk_bundle = {
        "token_budget": int(topk_cfg["token_budget"]),
        "group_shared_within_kv_group": bool(topk_cfg["group_shared_within_kv_group"]),
        "skip_layers": int(topk_cfg["skip_layers"]),
    }
    return model, tokenizer


def _build_runtime(model: LLM, prompt_len: int) -> TopKRuntime:
    bundle = getattr(model, "_topk_bundle", None)
    if bundle is None:
        raise RuntimeError("topk runtime is not initialized. Call initialize_model_tokenizer() first.")
    cfg = TopKConfig(
        prompt_len=int(prompt_len),
        topk_budget=int(bundle["token_budget"]),
        group_shared_within_kv_group=bool(bundle["group_shared_within_kv_group"]),
        skip_layers=int(bundle["skip_layers"]),
        attention_heads=model.config.n_head,
        kv_heads=int(getattr(model.config, "n_query_groups", model.config.n_head)),
        attention_scores_scalar=model.config.attention_scores_scalar,
        head_size=model.config.head_size,
    )
    return TopKRuntime(cfg)


def batch_generate(
    batched_input: torch.Tensor,  # [batch_size, seq_len]
    model: LLM,
    tokenizer: Tokenizer,
    max_new_tokens: int,
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
        runtime = _build_runtime(model, prompt_len=int(prompt_ids.numel()))
        responses.append(
            decode_topk(
                prompt_ids=prompt_ids,
                model=model,
                runtime=runtime,
                max_new_tokens=max_new_tokens,
            )
        )

    empty_cuda_cache()
    return responses
