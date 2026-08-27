import logging
import os
from pathlib import Path
from typing import Any

import torch

from litgpt import LLM
from litgpt.tokenizer import Tokenizer
from litgpt.prompts import Default, PromptStyle
from pipeline.litgpt_prefill import DEFAULT_PREFILL_CHUNK_SIZE

logger = logging.getLogger("main")
EMPTY_CUDA_CACHE_ENABLED = os.environ.get("SAKURA_EMPTY_CUDA_CACHE", "0").lower() in {"1", "true", "yes"}


def _get_pad_token_id(tokenizer: Tokenizer) -> int | None:
    pad_id = getattr(tokenizer, "pad_id", None)
    if pad_id is not None:
        return int(pad_id)

    processor = getattr(tokenizer, "processor", None)
    get_padding = getattr(processor, "get_padding", None)
    if get_padding is not None:
        padding = get_padding()
        if padding is not None and padding.get("pad_id") is not None:
            return int(padding["pad_id"])

    if getattr(tokenizer, "backend", None) == "huggingface":
        for token in ("<|endoftext|>", "<pad>", "[PAD]"):
            try:
                return int(tokenizer.token_to_id(token))
            except (ValueError, TypeError):
                continue
    return None


def trim_right_padding(row: torch.Tensor, tokenizer: Tokenizer) -> torch.Tensor:
    """Trim only trailing real padding tokens, preserving in-prompt EOS markers."""
    pad_id = _get_pad_token_id(tokenizer)
    eos_id = getattr(tokenizer, "eos_id", None)
    if pad_id is None or (eos_id is not None and int(pad_id) == int(eos_id)):
        return row

    keep_end = int(row.numel())
    while keep_end > 0 and int(row[keep_end - 1].item()) == int(pad_id):
        keep_end -= 1
    return row[:keep_end]


def initialize_litgpt_model(
    pipeline_params: dict[str, Any],
    *,
    path_label: str,
    distribute: bool = False,
) -> tuple[LLM, Tokenizer]:
    model_name = pipeline_params["model_name"]
    checkpoint_dir = Path(model_name)
    if not checkpoint_dir.is_dir():
        raise ValueError(f"pipeline_params['model_name'] must be a local directory, got: {model_name}")

    lit_ckpt = checkpoint_dir / "lit_model.pth"
    lit_cfg = checkpoint_dir / "model_config.yaml"
    if not lit_ckpt.is_file() or not lit_cfg.is_file():
        raise FileNotFoundError(
            f"Missing converted LitGPT checkpoint files. Expected {lit_ckpt} and {lit_cfg} to exist."
        )

    rope_theta_factor = pipeline_params.get("rope_theta_factor", 1.0)
    if rope_theta_factor != 1.0:
        logger.warning("rope_theta_factor=%s is currently ignored in LitGPT %s path.", rope_theta_factor, path_label)

    checkpoint_dir = checkpoint_dir.resolve()
    logger.info("Initializing LitGPT model from %s", checkpoint_dir)
    model = LLM.load(model=str(checkpoint_dir))

    prefill_chunk_size = int(pipeline_params.get("prefill_chunk_size", DEFAULT_PREFILL_CHUNK_SIZE))

    cuda_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if distribute and cuda_device_count > 1:
        model.distribute(
            generate_strategy="sequential",
            devices=cuda_device_count,
            fixed_kv_cache_size=pipeline_params.get("model_max_len", None),
        )
        logger.info("Distributed configuration enabled, using %s CUDA devices", cuda_device_count)
    elif cuda_device_count > 1:
        logger.warning("%s LitGPT path currently runs in single-process mode; skipping model.distribute().", path_label)
    elif distribute:
        device_info = f"{cuda_device_count} CUDA devices" if torch.cuda.is_available() else "CPU"
        logger.info("Running on %s, distributed configuration not enabled", device_info)

    if hasattr(model, "model"):
        setattr(model.model, "_prefill_chunk_size", prefill_chunk_size)
    setattr(model, "_prefill_chunk_size", prefill_chunk_size)
    if prefill_chunk_size > 0:
        logger.info("Chunked prefill enabled with chunk size %s tokens", prefill_chunk_size)
    else:
        logger.info("Chunked prefill disabled")

    return model, model.tokenizer


def normalize_batch_input(
    batched_input,
    model: LLM,
    tokenizer: Tokenizer,
) -> tuple[list[str | None], list[torch.Tensor] | None]:
    if isinstance(batched_input, torch.Tensor):
        prompt_ids_by_idx = []
        for row in batched_input:
            row = trim_right_padding(row, tokenizer)
            row = row.to(device=model.preprocessor.device, dtype=torch.long)
            prompt_ids_by_idx.append(row)
        return [None] * len(prompt_ids_by_idx), prompt_ids_by_idx

    if batched_input and isinstance(batched_input[0], str):
        return list(batched_input), None

    logger.error("Unknown batched_input:%s", batched_input)
    raise ValueError(f"Unknown batched_input type: {type(batched_input)!r}")


def get_prompt_ids(
    *,
    prompt: str | None,
    prompt_ids_by_idx: list[torch.Tensor] | None,
    idx: int,
    tokenizer: Tokenizer,
    model: LLM,
    bos: bool | None = False,
) -> torch.Tensor:
    if prompt_ids_by_idx is not None:
        return prompt_ids_by_idx[idx]
    if prompt is None:
        raise ValueError("prompt must be provided when prompt_ids_by_idx is None.")
    return tokenizer.encode(prompt, bos=bos, device=model.preprocessor.device)


def empty_cuda_cache(*, force: bool = False) -> None:
    if torch.cuda.is_available() and (force or EMPTY_CUDA_CACHE_ENABLED):
        torch.cuda.empty_cache()


def resolve_chat_template_name(
    *,
    pipeline_params: dict[str, Any],
    model: LLM | None = None,
) -> str | None:
    explicit = pipeline_params.get("chat_template")
    if explicit is not None:
        return str(explicit)

    model_name = str(pipeline_params.get("model_name", "")).lower()
    if "qwen2.5" in model_name or "qwq" in model_name or "qwen3" in model_name:
        return "qwen"
    if "llama-3" in model_name and "instruct" in model_name:
        return "llama3"
    if "mistral" in model_name and "instruct" in model_name:
        return "mistral_instruct"

    prompt_style = getattr(model, "prompt_style", None)
    if isinstance(prompt_style, Default):
        return None
    if prompt_style is not None:
        style_name = type(prompt_style).__name__.lower()
        if "llama3" in style_name:
            return "llama3"
        if "qwen" in style_name or "chatml" in style_name:
            return "qwen"
    return None


def _get_prompt_style_for_template(chat_template: str) -> PromptStyle:
    template_name = chat_template.lower()
    if "llama3" in template_name:
        return PromptStyle.from_name("llama3")
    if "qwen" in template_name:
        return PromptStyle.from_name("qwen2.5")
    raise NotImplementedError(f"No LitGPT PromptStyle is registered for chat template {chat_template!r}.")


def _get_active_prompt_style(model: LLM | None, chat_template: str) -> PromptStyle:
    prompt_style = getattr(model, "prompt_style", None)
    if prompt_style is None or isinstance(prompt_style, Default):
        return _get_prompt_style_for_template(chat_template)
    return prompt_style


def post_process(response, *, pipeline_params: dict[str, Any], model: LLM | None = None):
    chat_template = resolve_chat_template_name(pipeline_params=pipeline_params, model=model)
    if chat_template is None:
        return response

    template_name = chat_template.lower()
    if "qwen" in template_name:
        response = response.strip()
        if response.startswith("<|im_start|>assistant"):
            response = response[len("<|im_start|>assistant") :].lstrip()
        elif response.startswith("<|im_start|>"):
            response = response.split("\n", 1)[1] if "\n" in response else ""
        response = response.split("<|im_end|>")[0].split("<|im_start|>")[0].strip()
    return response


def get_chat_template_label(*, pipeline_params: dict[str, Any], model: LLM | None = None) -> str:
    return resolve_chat_template_name(pipeline_params=pipeline_params, model=model) or "plain"


def build_chat(
    model: LLM | None,
    tokenizer: Tokenizer,
    prompt,
    *,
    pipeline_params: dict[str, Any],
):
    chat_template = resolve_chat_template_name(pipeline_params=pipeline_params, model=model)
    if chat_template is None:
        return prompt

    template_name = chat_template.lower()

    if "llama3" in template_name:
        prompt_style = _get_active_prompt_style(model, chat_template)
        prompt = prompt_style.apply(prompt)
    elif "qwen" in template_name:
        prompt_style = _get_active_prompt_style(model, chat_template)
        prompt = prompt_style.apply(prompt)
    elif "mistral" in template_name:
        return prompt
    else:
        logger.error(f"{chat_template} is unsupported.")
        raise NotImplementedError

    return prompt
