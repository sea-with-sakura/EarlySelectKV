import json
import logging
import math
import os
import re
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable

import torch
from litgpt import LLM
from litgpt.tokenizer import Tokenizer
from tqdm import tqdm

import pipeline.main_utils as main_utils
from pipeline.model_utils import build_chat, get_chat_template_label, post_process

logger = logging.getLogger("main")


DEFAULT_PROBLEM_FIELDS = ("problem", "question", "prompt", "input")
DEFAULT_ANSWER_FIELDS = ("answer", "final_answer", "ground_truth", "target", "label")


def _read_local_json(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "examples", "test", "rows"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported local math dataset format: {path}")


def _load_hf_dataset(eval_params: dict) -> list[dict]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "datasets is required to load math_reasoning data from Hugging Face. "
            "Install a compatible datasets/huggingface_hub pair or provide a local dataset_path."
        ) from exc

    hf_dataset = eval_params.get("hf_dataset")
    if not hf_dataset:
        raise ValueError("math_reasoning eval config must set hf_dataset or an existing local dataset_path.")
    hf_config = eval_params.get("hf_config")
    hf_split = eval_params.get("hf_split", "test")
    if hf_config:
        data = load_dataset(hf_dataset, hf_config, split=hf_split)
    else:
        data = load_dataset(hf_dataset, split=hf_split)
    return [dict(row) for row in data]


def load_data(eval_params: dict) -> list[dict]:
    dataset_path = eval_params.get("dataset_path")
    if dataset_path:
        path = Path(dataset_path)
        if path.is_file():
            logger.info("Loading math_reasoning data from %s.", path)
            return _read_local_json(path)
        if path.is_dir():
            candidate = path / f"{eval_params['dataset']}.jsonl"
            if candidate.is_file():
                logger.info("Loading math_reasoning data from %s.", candidate)
                return _read_local_json(candidate)

    logger.info("Loading math_reasoning data from Hugging Face dataset %s.", eval_params.get("hf_dataset"))
    return _load_hf_dataset(eval_params)


def _first_field(example: dict, fields: Iterable[str]):
    for field in fields:
        if field in example and example[field] not in (None, ""):
            return example[field]
    return None


def _message_content(value) -> str | None:
    if not isinstance(value, list):
        return None
    parts = []
    for item in value:
        if isinstance(item, dict) and item.get("content"):
            parts.append(str(item["content"]))
    return "\n".join(parts).strip() or None


def _coerce_problem(example: dict, eval_params: dict) -> str:
    fields = eval_params.get("problem_fields", DEFAULT_PROBLEM_FIELDS)
    value = _first_field(example, fields)
    if value is None:
        value = _message_content(example.get("messages")) or _message_content(example.get("prompt"))
    if value is None:
        raise KeyError(f"Could not find problem text in fields {fields}: {sorted(example.keys())}")
    return str(value)


def _coerce_answer(example: dict, eval_params: dict) -> str:
    fields = eval_params.get("answer_fields", DEFAULT_ANSWER_FIELDS)
    value = _first_field(example, fields)
    if isinstance(value, dict):
        value = _first_field(value, ("answer", "ground_truth", "value"))
    if value is None and example.get("solution"):
        value = extract_answer(str(example["solution"]))
    if value is None:
        raise KeyError(f"Could not find answer in fields {fields}: {sorted(example.keys())}")
    return str(value)


def _find_last_braced(text: str, command: str) -> str | None:
    marker = "\\" + command + "{"
    start = text.rfind(marker)
    if start < 0:
        return None
    i = start + len(marker)
    depth = 1
    out = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
        out.append(ch)
        i += 1
    return None


def extract_answer(text: str) -> str:
    for command in ("boxed", "fbox"):
        boxed = _find_last_braced(text, command)
        if boxed:
            return boxed

    patterns = [
        r"(?:final answer|answer)\s*(?:is|:)\s*\$?([^\n$]+)",
        r"答案\s*(?:是|:|：)\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    number_matches = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", text.replace(",", ""))
    if number_matches:
        return number_matches[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def extract_reference_answer(text: str) -> str:
    for command in ("boxed", "fbox"):
        boxed = _find_last_braced(text, command)
        if boxed:
            return boxed

    patterns = [
        r"(?:final answer|answer)\s*(?:is|:)\s*\$?([^\n$]+)",
        r"答案\s*(?:是|:|：)\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    return text.strip()


def _strip_latex_wrappers(text: str) -> str:
    text = text.strip()
    text = text.replace("\\\\", "\\")
    text = re.sub(r"^\\(?:boxed|fbox)\{(.+)\}$", r"\1", text)
    text = re.sub(r"^\$+|\$+$", "", text)
    text = re.sub(r"\\(?:left|right)", "", text)
    text = re.sub(r"\\!", "", text)
    text = re.sub(r"\\,", "", text)
    text = re.sub(r"\\;", "", text)
    text = re.sub(r"\\ ", "", text)
    text = text.replace("\\text{", "").replace("\\mathrm{", "")
    text = text.replace("{,}", ",")
    return text.strip()


def normalize_answer(text: str) -> str:
    text = _strip_latex_wrappers(str(text))
    text = text.strip().rstrip(".。")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1/\2", text)
    text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)
    text = text.replace("\\pi", "pi")
    text = text.replace("^\\circ", "")
    text = text.replace("\\%", "%")
    text = text.replace(" ", "")
    text = text.replace(",", "")
    text = text.replace("{", "").replace("}", "")
    return text.lower()


def _to_number(text: str) -> float | None:
    text = normalize_answer(text)
    if not text:
        return None
    if "%" in text:
        try:
            return float(text.replace("%", "")) / 100.0
        except ValueError:
            return None
    if re.fullmatch(r"[-+]?\d+/\d+", text):
        try:
            return float(Fraction(text))
        except ZeroDivisionError:
            return None
    try:
        return float(Fraction(text))
    except Exception:
        try:
            return float(text)
        except ValueError:
            return None


def answers_match(prediction: str, reference: str) -> bool:
    pred = extract_answer(prediction)
    ref = extract_reference_answer(reference)
    if normalize_answer(pred) == normalize_answer(ref):
        return True

    pred_num = _to_number(pred)
    ref_num = _to_number(ref)
    if pred_num is None or ref_num is None:
        return False
    return math.isclose(pred_num, ref_num, rel_tol=1e-4, abs_tol=1e-4)


def _encode_input(
    *,
    prompt: str,
    tokenizer: Tokenizer,
    model: LLM,
    pipeline_params: dict,
    eval_params: dict,
) -> torch.Tensor:
    tokenized_prompt = tokenizer.encode(prompt, bos=False)
    if pipeline_params.get("truncation_mode") != "middle":
        return tokenized_prompt.to(model.preprocessor.device).unsqueeze(0)

    model_max = int(pipeline_params["model_max_len"])
    max_new = int(eval_params.get("max_new_tokens", 0))
    allowed = model_max - max_new - 2
    if allowed < 1:
        raise ValueError(f"model_max_len ({model_max}) too small for max_new_tokens ({max_new})")
    if len(tokenized_prompt) > allowed:
        half = allowed // 2
        right_len = allowed - half
        tokenized_prompt = torch.cat([tokenized_prompt[:half], tokenized_prompt[-right_len:]])
    return tokenized_prompt.to(model.preprocessor.device).unsqueeze(0)


def _format_instruction(instruction: str, *, problem: str) -> str:
    instruction = instruction.replace("\\\\boxed", "\\boxed").replace("\\\\fbox", "\\fbox")
    safe_instruction = instruction.replace("\\boxed{}", "\\boxed{{}}").replace("\\fbox{}", "\\fbox{{}}")
    return safe_instruction.format(problem=problem)


def eval_math_reasoning(
    *,
    config: dict,
    initialize_model_tokenizer: Callable,
    batch_generate: Callable,
    pass_pipeline_params: bool,
):
    eval_params = config["eval_params"]
    pipeline_params = config["pipeline_params"]
    data = load_data(eval_params)
    sample_offset = int(eval_params.get("sample_offset", os.environ.get("MATH_REASONING_SAMPLE_OFFSET", "0")))
    sample_limit = int(eval_params.get("sample_limit", os.environ.get("MATH_REASONING_SAMPLE_LIMIT", "0")))
    if sample_offset > 0:
        logger.info("Skipping first %s math_reasoning samples.", sample_offset)
        data = data[sample_offset:]
    if sample_limit > 0:
        logger.info("Limiting math_reasoning data to first %s samples.", sample_limit)
        data = data[:sample_limit]

    model, tokenizer = initialize_model_tokenizer(pipeline_params=pipeline_params)
    model.eval()

    predictions = []
    correct = 0
    instruction = eval_params["instruction"]

    for idx, example in enumerate(tqdm(data), start=sample_offset):
        problem = _coerce_problem(example, eval_params)
        reference = _coerce_answer(example, eval_params)
        prompt = _format_instruction(instruction, problem=problem)
        if eval_params.get("use_chat_template", True):
            prompt = build_chat(model, tokenizer, prompt, pipeline_params=pipeline_params)

        input_ids = _encode_input(
            prompt=prompt,
            tokenizer=tokenizer,
            model=model,
            pipeline_params=pipeline_params,
            eval_params=eval_params,
        )
        generate_kwargs = {"pipeline_params": pipeline_params} if pass_pipeline_params else {}
        pred = batch_generate(
            input_ids,
            model,
            tokenizer,
            int(eval_params["max_new_tokens"]),
            **generate_kwargs,
        )[0]
        pred = post_process(pred, pipeline_params=pipeline_params, model=model)
        is_correct = answers_match(pred, reference)
        correct += int(is_correct)
        predictions.append(
            {
                "idx": idx,
                "pred": pred,
                "extracted_pred": extract_answer(pred),
                "answer": reference,
                "extracted_answer": extract_reference_answer(reference),
                "correct": is_correct,
                "problem": problem,
            }
        )

    total = len(predictions)
    accuracy = 100.0 * correct / total if total else 0.0

    out_dir = Path(config["management"]["output_folder_dir"]) / "pred" / pipeline_params["method"]
    out_dir.mkdir(parents=True, exist_ok=True)
    template_label = get_chat_template_label(pipeline_params=pipeline_params, model=model)
    out_path = out_dir / f"{eval_params['dataset']}_{template_label}.jsonl"
    for stale_path in sorted(out_dir.glob(f"{eval_params['dataset']}_*.jsonl")):
        if stale_path != out_path:
            stale_path.unlink()
    with open(out_path, "w", encoding="utf-8") as f:
        for item in predictions:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    main_utils.log_prediction_file(str(out_path))

    processed_results = {eval_params["dataset"]: accuracy}
    raw_results = {
        "dataset": eval_params["dataset"],
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "prediction_path": str(out_path),
        "predictions": predictions,
    }
    return processed_results, raw_results
