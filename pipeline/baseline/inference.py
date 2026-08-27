import logging
import torch

from litgpt import LLM
from litgpt.tokenizer import Tokenizer

from pipeline.model_utils import empty_cuda_cache, get_prompt_ids, initialize_litgpt_model, normalize_batch_input
from pipeline.runtime_decode_utils import run_full_cache_decode
from .baseline_monkey_patch import install_baseline_attention, uninstall_baseline_attention

logger = logging.getLogger("main")


def initialize_model_tokenizer(pipeline_params):
    return initialize_litgpt_model(pipeline_params, path_label="baseline", distribute=True)


def _decode_baseline(
    *,
    prompt_ids: torch.Tensor,
    model: LLM,
    max_new_tokens: int,
) -> str:
    return run_full_cache_decode(
        prompt_ids=prompt_ids,
        model=model,
        max_new_tokens=max_new_tokens,
        attention_state=True,
        install_attention=install_baseline_attention,
        uninstall_attention=uninstall_baseline_attention,
    )


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
        response = _decode_baseline(
            prompt_ids=prompt_ids,
            model=model,
            max_new_tokens=max_new_tokens,
        )
        responses.append(response)

    empty_cuda_cache()
    return responses
