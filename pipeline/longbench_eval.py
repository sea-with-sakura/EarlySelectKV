import json
import logging
import os
import random
from pathlib import Path
from typing import Callable

import torch
from litgpt import LLM
from litgpt.tokenizer import Tokenizer
from tqdm import tqdm

import pipeline.main_utils as main_utils
import utils.longbench_utils.eval_long_bench as longbench_eval
from pipeline.model_utils import build_chat, get_chat_template_label, post_process
from utils.longbench_utils.eval_long_bench import load_data

logger = logging.getLogger("main")

NO_CHAT_DATASETS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}


def _encode_longbench_input(
    *,
    prompt: str,
    tokenizer: Tokenizer,
    device: torch.device,
    pipeline_params: dict,
    eval_params: dict,
    bos: bool | None,
) -> tuple[torch.Tensor, str]:
    tokenized_prompt = tokenizer.encode(prompt, bos=bos)
    trunc_ids = None

    if pipeline_params.get("truncation_mode") == "middle":
        model_max = int(pipeline_params["model_max_len"])
        max_new = int(eval_params.get("max_new_tokens", 0))
        safety = 1
        bos_pad = 1 if getattr(tokenizer, "use_bos", False) else 0
        allowed = model_max - max_new - bos_pad - safety
        if allowed < 1:
            raise ValueError(f"model_max_len ({model_max}) too small for max_new_tokens ({max_new})")
        if len(tokenized_prompt) > allowed:
            half = allowed // 2
            left = tokenized_prompt[:half]
            right_len = allowed - half
            right = tokenized_prompt[-right_len:] if right_len > 0 else tokenized_prompt.new_empty(0)
            trunc_ids = torch.cat([left, right])
            prompt = tokenizer.decode(trunc_ids)

    if trunc_ids is not None:
        return trunc_ids.to(device).unsqueeze(0), prompt
    return tokenized_prompt.to(device).unsqueeze(0), prompt


def get_pred(
    *,
    model: LLM,
    tokenizer: Tokenizer,
    data,
    device: torch.device,
    pipeline_params: dict,
    eval_params: dict,
    batch_generate: Callable,
    pass_pipeline_params: bool,
    bos: bool | None = False,
) -> list[dict]:
    preds = []
    for json_obj in tqdm(data):
        prompt = eval_params["instruction"].format(**json_obj)

        if eval_params["dataset"] not in NO_CHAT_DATASETS:
            prompt = build_chat(model, tokenizer, prompt, pipeline_params=pipeline_params)

        input_ids, _ = _encode_longbench_input(
            prompt=prompt,
            tokenizer=tokenizer,
            device=device,
            pipeline_params=pipeline_params,
            eval_params=eval_params,
            bos=bos,
        )
        generate_kwargs = {"pipeline_params": pipeline_params} if pass_pipeline_params else {}
        pred = batch_generate(
            input_ids,
            model,
            tokenizer,
            eval_params["max_new_tokens"],
            **generate_kwargs,
        )[0]
        pred = post_process(pred, pipeline_params=pipeline_params, model=model)
        preds.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
        )

    return preds


def eval_longbench(
    *,
    config: dict,
    initialize_model_tokenizer: Callable,
    batch_generate: Callable,
    pass_pipeline_params: bool,
    bos: bool | None = False,
):
    eval_params = config["eval_params"]
    pipeline_params = config["pipeline_params"]
    data = load_data(eval_params)
    sample_limit = int(eval_params.get("sample_limit", os.environ.get("LONGBENCH_SAMPLE_LIMIT", "0")))
    if sample_limit > 0:
        limit = min(sample_limit, len(data))
        sample_seed_raw = eval_params.get("sample_seed", os.environ.get("LONGBENCH_SAMPLE_SEED"))
        if sample_seed_raw is None or str(sample_seed_raw) == "":
            indices = list(range(limit))
            logger.info("Limiting LongBench data to first %s samples.", limit)
        else:
            sample_seed = int(sample_seed_raw)
            indices = sorted(random.Random(sample_seed).sample(range(len(data)), limit))
            logger.info("Selecting %s LongBench samples with seed %s.", limit, sample_seed)
        data = data.select(indices) if hasattr(data, "select") else [data[i] for i in indices]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = initialize_model_tokenizer(pipeline_params=pipeline_params)
    model.eval()
    preds = get_pred(
        model=model,
        tokenizer=tokenizer,
        data=data,
        device=device,
        pipeline_params=pipeline_params,
        eval_params=eval_params,
        batch_generate=batch_generate,
        pass_pipeline_params=pass_pipeline_params,
        bos=bos,
    )

    out_dir = Path(config["management"]["output_folder_dir"]) / "pred" / pipeline_params["method"]
    out_dir.mkdir(parents=True, exist_ok=True)
    template_label = get_chat_template_label(pipeline_params=pipeline_params, model=model)
    out_path = out_dir / f"{eval_params['dataset']}_{template_label}.jsonl"
    for stale_path in sorted(out_dir.glob(f"{eval_params['dataset']}_*.jsonl")):
        if stale_path != out_path:
            stale_path.unlink()

    with open(out_path, "w", encoding="utf-8") as f:
        for pred in preds:
            json.dump(pred, f, ensure_ascii=False)
            f.write("\n")
    main_utils.log_prediction_file(str(out_path))

    return longbench_eval.eval(
        pred_dir=config["management"]["output_folder_dir"],
        model=pipeline_params["method"],
        eval_params=eval_params,
    )
