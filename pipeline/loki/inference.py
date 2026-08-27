import logging
import os
from typing import Any, Dict, Optional

import torch

from litgpt import LLM
from litgpt.tokenizer import Tokenizer

from pipeline.model_utils import empty_cuda_cache, get_prompt_ids, initialize_litgpt_model, normalize_batch_input
from .runtime import LokiConfig, LokiRuntime, decode_loki

logger = logging.getLogger("main")


def _get_loki_cfg(pipeline_params: Dict[str, Any]) -> Dict[str, Any]:
    method = str(pipeline_params.get("method", "loki"))
    if method not in {"loki", "loki_headwise", "loki_pre"}:
        raise ValueError(f"Unsupported Loki method: {method}")

    loki_cfg = dict(pipeline_params.get("loki", {}))
    if method != "loki":
        loki_cfg.update(dict(pipeline_params.get(method, {})))
    loki_cfg["token_budget"] = int(pipeline_params.get("token_budget", loki_cfg.get("token_budget", 1024)))
    if "rank" not in loki_cfg and "top_r" in loki_cfg:
        loki_cfg["rank"] = loki_cfg["top_r"]
    loki_cfg.setdefault("rank", 16)
    loki_cfg.setdefault("skip_layers", 1)
    loki_cfg.setdefault("transform_dataset", "wikitext")
    loki_cfg.setdefault("rotary_type", "postrotary")
    loki_cfg.setdefault("group_shared_within_kv_group", True)
    loki_cfg.setdefault("allow_identity_fallback", False)
    loki_cfg.setdefault("pca_dir", None)
    loki_cfg.setdefault("pca_model_name", None)
    if os.environ.get("LOKI_PCA_DIR"):
        loki_cfg["pca_dir"] = os.environ["LOKI_PCA_DIR"]
    return loki_cfg


def initialize_model_tokenizer(pipeline_params: Dict[str, Any]):
    return initialize_litgpt_model(pipeline_params, path_label="Loki")


def _build_runtime(
    model: LLM,
    pipeline_params: Dict[str, Any],
    *,
    prompt_len: int,
) -> LokiRuntime:
    loki_cfg = _get_loki_cfg(pipeline_params)
    cfg = LokiConfig(
        prompt_len=int(prompt_len),
        token_budget=int(loki_cfg["token_budget"]),
        rank=float(loki_cfg["rank"]),
        skip_layers=int(loki_cfg["skip_layers"]),
        attention_heads=model.config.n_head,
        kv_heads=int(getattr(model.config, "n_query_groups", model.config.n_head)),
        attention_scores_scalar=model.config.attention_scores_scalar,
        head_size=model.config.head_size,
        model_name=str(pipeline_params.get("model_name", "")),
        pca_dir=loki_cfg.get("pca_dir"),
        pca_model_name=loki_cfg.get("pca_model_name"),
        transform_dataset=str(loki_cfg["transform_dataset"]),
        rotary_type=str(loki_cfg["rotary_type"]),
        group_shared_within_kv_group=bool(loki_cfg["group_shared_within_kv_group"]),
        allow_identity_fallback=bool(loki_cfg["allow_identity_fallback"]),
    )
    logger.info(
        "loki budget: token_budget=%s rank=%s skip_layers=%s pca_dir=%s dataset=%s rotary=%s routing_group_shared=%s kv_heads=%s attention_heads=%s",
        cfg.token_budget,
        cfg.rank,
        cfg.skip_layers,
        cfg.pca_dir,
        cfg.transform_dataset,
        cfg.rotary_type,
        cfg.group_shared_within_kv_group,
        cfg.kv_heads,
        cfg.attention_heads,
    )
    return LokiRuntime(cfg)


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
            decode_loki(
                prompt_ids=prompt_ids,
                model=model,
                runtime=runtime,
                max_new_tokens=max_new_tokens,
            )
        )

    empty_cuda_cache()
    return responses
