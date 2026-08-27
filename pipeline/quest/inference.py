import logging
from typing import Any, Dict, Optional

import torch

from litgpt import LLM
from litgpt.tokenizer import Tokenizer

from pipeline.model_utils import empty_cuda_cache, get_prompt_ids, initialize_litgpt_model, normalize_batch_input
from .runtime import QuestConfig, QuestRuntime, decode_quest

logger = logging.getLogger("main")


def _get_quest_cfg(pipeline_params: Dict[str, Any]) -> Dict[str, Any]:
    method = str(pipeline_params.get("method", "quest"))
    if method != "quest":
        raise ValueError(f"Unsupported Quest method: {method}")

    quest_cfg = dict(pipeline_params.get("quest", {}))
    quest_cfg["token_budget"] = int(pipeline_params.get("token_budget", quest_cfg.get("token_budget", 1024)))
    if "page_size" not in quest_cfg and "chunk_size" in quest_cfg:
        quest_cfg["page_size"] = quest_cfg["chunk_size"]
    quest_cfg.setdefault("page_size", 16)
    quest_cfg.setdefault("skip_layers", 1)
    quest_cfg.setdefault("min_select_pages", 1)
    quest_cfg.setdefault("include_last_page", True)
    quest_cfg.setdefault("group_shared_within_kv_group", True)
    return quest_cfg


def initialize_model_tokenizer(pipeline_params: Dict[str, Any]):
    return initialize_litgpt_model(pipeline_params, path_label="Quest")


def _build_runtime(
    model: LLM,
    pipeline_params: Dict[str, Any],
    *,
    prompt_len: int,
) -> QuestRuntime:
    quest_cfg = _get_quest_cfg(pipeline_params)
    cfg = QuestConfig(
        prompt_len=int(prompt_len),
        token_budget=int(quest_cfg["token_budget"]),
        page_size=int(quest_cfg["page_size"]),
        skip_layers=int(quest_cfg["skip_layers"]),
        attention_heads=model.config.n_head,
        kv_heads=int(getattr(model.config, "n_query_groups", model.config.n_head)),
        min_select_pages=int(quest_cfg["min_select_pages"]),
        include_last_page=bool(quest_cfg["include_last_page"]),
        group_shared_within_kv_group=bool(quest_cfg["group_shared_within_kv_group"]),
    )
    logger.info(
        "quest budget: token_budget=%s page_size=%s page_budget=%s skip_layers=%s include_last_page=%s group_shared=%s",
        cfg.token_budget,
        cfg.page_size,
        max(1, cfg.token_budget // cfg.page_size),
        cfg.skip_layers,
        cfg.include_last_page,
        cfg.group_shared_within_kv_group,
    )
    return QuestRuntime(cfg)


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
            decode_quest(
                prompt_ids=prompt_ids,
                model=model,
                runtime=runtime,
                max_new_tokens=max_new_tokens,
            )
        )

    empty_cuda_cache()
    return responses
