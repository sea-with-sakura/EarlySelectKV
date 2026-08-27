import logging
from typing import Any, Dict, Optional

import torch

from litgpt import LLM
from litgpt.tokenizer import Tokenizer

from pipeline.model_utils import empty_cuda_cache, get_prompt_ids, initialize_litgpt_model, normalize_batch_input
from pipeline.config_utils import canonical_method_key, method_config_keys
from .runtime import LookaheadQuestConfig, LookaheadQuestRuntime, decode_lookahead_quest

logger = logging.getLogger("main")


def _get_lookahead_quest_cfg(pipeline_params: Dict[str, Any]) -> Dict[str, Any]:
    method = canonical_method_key(str(pipeline_params.get("method", "earlyselect_quest")))
    if method != "earlyselect_quest":
        raise ValueError(f"Unsupported EarlySelect-Quest method: {method}")

    quest_cfg = dict(pipeline_params.get("quest", {}))
    for key in reversed(method_config_keys(method)):
        quest_cfg.update(dict(pipeline_params.get(key, {})))
    quest_cfg["token_budget"] = int(pipeline_params.get("token_budget", quest_cfg.get("token_budget", 1024)))
    if "page_size" not in quest_cfg and "chunk_size" in quest_cfg:
        quest_cfg["page_size"] = quest_cfg["chunk_size"]
    quest_cfg.setdefault("page_size", 16)
    quest_cfg.setdefault("skip_layers", 1)
    quest_cfg.setdefault("min_select_pages", 1)
    quest_cfg.setdefault("include_last_page", True)
    quest_cfg.setdefault("group_shared_within_kv_group", True)
    quest_cfg.setdefault("lookahead_source", "mid")
    quest_cfg.setdefault("lookahead_mid_until_layer", 20)
    quest_cfg.setdefault("use_real_q_fallback", False)
    return quest_cfg


def initialize_model_tokenizer(pipeline_params: Dict[str, Any]):
    return initialize_litgpt_model(pipeline_params, path_label="EarlySelect-Quest")


def _build_runtime(
    model: LLM,
    pipeline_params: Dict[str, Any],
    *,
    prompt_len: int,
) -> LookaheadQuestRuntime:
    quest_cfg = _get_lookahead_quest_cfg(pipeline_params)
    cfg = LookaheadQuestConfig(
        prompt_len=int(prompt_len),
        token_budget=int(quest_cfg["token_budget"]),
        page_size=int(quest_cfg["page_size"]),
        skip_layers=int(quest_cfg["skip_layers"]),
        attention_heads=model.config.n_head,
        kv_heads=int(getattr(model.config, "n_query_groups", model.config.n_head)),
        min_select_pages=int(quest_cfg["min_select_pages"]),
        include_last_page=bool(quest_cfg["include_last_page"]),
        group_shared_within_kv_group=bool(quest_cfg["group_shared_within_kv_group"]),
        lookahead_source=str(quest_cfg["lookahead_source"]),
        lookahead_mid_until_layer=int(quest_cfg["lookahead_mid_until_layer"]),
        use_real_q_fallback=bool(quest_cfg["use_real_q_fallback"]),
    )
    logger.info(
        "earlyselect_quest budget: token_budget=%s page_size=%s page_budget=%s skip_layers=%s include_last_page=%s group_shared=%s lookahead_source=%s lookahead_mid_until_layer=%s real_q_fallback=%s kv_heads=%s attention_heads=%s",
        cfg.token_budget,
        cfg.page_size,
        max(1, cfg.token_budget // cfg.page_size),
        cfg.skip_layers,
        cfg.include_last_page,
        cfg.group_shared_within_kv_group,
        cfg.lookahead_source,
        cfg.lookahead_mid_until_layer,
        cfg.use_real_q_fallback,
        cfg.kv_heads,
        cfg.attention_heads,
    )
    return LookaheadQuestRuntime(cfg)


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
        runtime = _build_runtime(model, pipeline_params or {}, prompt_len=int(prompt_ids.numel()))
        responses.append(
            decode_lookahead_quest(
                prompt_ids=prompt_ids,
                model=model,
                runtime=runtime,
                max_new_tokens=max_new_tokens,
            )
        )

    empty_cuda_cache()
    return responses
