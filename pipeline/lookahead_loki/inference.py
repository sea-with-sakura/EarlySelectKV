import logging
import os
from typing import Any, Dict, Optional

import torch

from litgpt import LLM
from litgpt.tokenizer import Tokenizer

from pipeline.model_utils import empty_cuda_cache, get_prompt_ids, initialize_litgpt_model, normalize_batch_input
from pipeline.config_utils import canonical_method_key, method_config_keys
from .runtime import LookaheadLokiConfig, LookaheadLokiRuntime, decode_lookahead_loki

logger = logging.getLogger("main")


def _get_lookahead_loki_cfg(pipeline_params: Dict[str, Any]) -> Dict[str, Any]:
    method = canonical_method_key(str(pipeline_params.get("method", "earlyselect_loki")))
    if method != "earlyselect_loki":
        raise ValueError(f"Unsupported EarlySelect-Loki method: {method}")

    loki_cfg = dict(pipeline_params.get("loki", {}))
    for key in reversed(method_config_keys(method)):
        loki_cfg.update(dict(pipeline_params.get(key, {})))
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
    loki_cfg.setdefault("lookahead_source", "mid")
    loki_cfg.setdefault("lookahead_mid_until_layer", 20)
    loki_cfg.setdefault("use_real_q_fallback", False)
    if os.environ.get("LOKI_PCA_DIR"):
        loki_cfg["pca_dir"] = os.environ["LOKI_PCA_DIR"]
    return loki_cfg


def initialize_model_tokenizer(pipeline_params: Dict[str, Any]):
    return initialize_litgpt_model(pipeline_params, path_label="EarlySelect-Loki")


def _build_runtime(
    model: LLM,
    pipeline_params: Dict[str, Any],
    *,
    prompt_len: int,
) -> LookaheadLokiRuntime:
    cfg_dict = _get_lookahead_loki_cfg(pipeline_params)
    cfg = LookaheadLokiConfig(
        prompt_len=int(prompt_len),
        token_budget=int(cfg_dict["token_budget"]),
        rank=float(cfg_dict["rank"]),
        skip_layers=int(cfg_dict["skip_layers"]),
        attention_heads=model.config.n_head,
        kv_heads=int(getattr(model.config, "n_query_groups", model.config.n_head)),
        attention_scores_scalar=model.config.attention_scores_scalar,
        head_size=model.config.head_size,
        model_name=str(pipeline_params.get("model_name", "")),
        pca_dir=cfg_dict.get("pca_dir"),
        pca_model_name=cfg_dict.get("pca_model_name"),
        transform_dataset=str(cfg_dict["transform_dataset"]),
        rotary_type=str(cfg_dict["rotary_type"]),
        group_shared_within_kv_group=bool(cfg_dict["group_shared_within_kv_group"]),
        allow_identity_fallback=bool(cfg_dict["allow_identity_fallback"]),
        lookahead_source=str(cfg_dict["lookahead_source"]),
        lookahead_mid_until_layer=int(cfg_dict["lookahead_mid_until_layer"]),
        use_real_q_fallback=bool(cfg_dict["use_real_q_fallback"]),
    )
    logger.info(
        "earlyselect_loki budget: token_budget=%s rank=%s skip_layers=%s pca_dir=%s dataset=%s rotary=%s routing_group_shared=%s lookahead_source=%s lookahead_mid_until_layer=%s real_q_fallback=%s kv_heads=%s attention_heads=%s",
        cfg.token_budget,
        cfg.rank,
        cfg.skip_layers,
        cfg.pca_dir,
        cfg.transform_dataset,
        cfg.rotary_type,
        cfg.group_shared_within_kv_group,
        cfg.lookahead_source,
        cfg.lookahead_mid_until_layer,
        cfg.use_real_q_fallback,
        cfg.kv_heads,
        cfg.attention_heads,
    )
    return LookaheadLokiRuntime(cfg)


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
            decode_lookahead_loki(
                prompt_ids=prompt_ids,
                model=model,
                runtime=runtime,
                max_new_tokens=max_new_tokens,
            )
        )

    empty_cuda_cache()
    return responses
