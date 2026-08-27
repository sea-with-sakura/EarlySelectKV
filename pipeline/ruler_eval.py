import csv
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import torch
from tqdm import tqdm

from pipeline.model_utils import get_chat_template_label, post_process

logger = logging.getLogger("main")

REPO_ROOT = Path(__file__).resolve().parents[1]
RULER_UTILS_DIR = REPO_ROOT / "utils" / "ruler_utils"


def _run_command(command: list[str], desc: str) -> None:
    logger.info("Starting %s: %s", desc, " ".join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        logger.info("%s: %s", desc, line.rstrip())

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def prepare_data(config: dict) -> None:
    eval_params = config["eval_params"]
    pipeline_params = config["pipeline_params"]
    management = config["management"]
    command = [
        sys.executable,
        str(RULER_UTILS_DIR / "data" / "prepare.py"),
        "--save_dir",
        management["output_folder_dir"],
        "--benchmark",
        eval_params["benchmark"],
        "--task",
        eval_params["dataset"],
        "--tokenizer_path",
        pipeline_params["tokenizer_name"],
        "--tokenizer_type",
        "hf",
        "--max_seq_length",
        str(eval_params["max_seq_length"]),
        "--model_template_type",
        get_chat_template_label(pipeline_params=pipeline_params),
        "--num_samples",
        str(eval_params["num_samples"]),
    ]
    _run_command(command, "RULER data preparation")


def get_eval(config: dict) -> None:
    eval_params = config["eval_params"]
    management = config["management"]
    command = [
        sys.executable,
        str(RULER_UTILS_DIR / "eval" / "evaluate.py"),
        "--data_dir",
        management["output_folder_dir"],
        "--benchmark",
        eval_params["benchmark"],
    ]
    _run_command(command, "RULER evaluation")


def _read_compact_raw_results(path: Path) -> list[dict]:
    keep_fields = ("index", "pred", "outputs", "others", "truncation", "length")
    results = []
    with path.open(encoding="utf-8") as json_file:
        for line in json_file:
            if not line.strip():
                continue
            row = json.loads(line)
            results.append({field: row[field] for field in keep_fields if field in row})
    return results


def _read_summary(summary_path: Path, dataset: str) -> dict:
    if not summary_path.is_file():
        raise FileNotFoundError(f"RULER summary file is missing: {summary_path}")

    task_row = None
    score_row = None
    nulls_row = None
    with summary_path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if row[0] == "Tasks":
                task_row = row
            elif row[0] == "Score":
                score_row = row
            elif row[0] == "Nulls":
                nulls_row = row

    if task_row is None or score_row is None:
        raise ValueError(f"RULER summary file has no Tasks/Score rows: {summary_path}")

    try:
        dataset_idx = task_row.index(dataset)
    except ValueError:
        dataset_idx = task_row.index("avg")

    score = float(score_row[dataset_idx])
    nulls = nulls_row[dataset_idx] if nulls_row is not None and dataset_idx < len(nulls_row) else None
    return {"score": score, "nulls": nulls}


def get_pred(
    *,
    model,
    tokenizer,
    data,
    pred_file: str,
    device: torch.device,
    pipeline_params: dict,
    eval_params: dict,
    batch_generate: Callable,
    pass_pipeline_params: bool,
) -> None:
    if os.path.exists(pred_file):
        os.remove(pred_file)
        logger.info("clear old pred file in %s", pred_file)

    with open(pred_file, "at", encoding="utf-8", buffering=1) as fout:
        for json_obj in tqdm(data):
            prompt = json_obj["input"]
            input_ids = tokenizer.encode(prompt, device=device, bos=False).unsqueeze(0)
            generate_kwargs = {"pipeline_params": pipeline_params} if pass_pipeline_params else {}
            pred = batch_generate(
                input_ids,
                model,
                tokenizer,
                eval_params["max_new_tokens"],
                **generate_kwargs,
            )[0]
            pred = post_process(pred, pipeline_params=pipeline_params, model=model)
            pred = {
                "index": json_obj["index"],
                "pred": pred,
                "input": prompt,
                "outputs": json_obj["outputs"],
                "others": json_obj.get("others", {}),
                "truncation": json_obj.get("truncation", -1),
                "length": json_obj.get("length", -1),
            }
            fout.write(json.dumps(pred, ensure_ascii=False) + "\n")
    logger.info("generate pred file to %s", pred_file)


def eval_ruler(
    *,
    config: dict,
    initialize_model_tokenizer: Callable,
    batch_generate: Callable,
    pass_pipeline_params: bool,
) -> tuple[dict, list[dict]]:
    prepare_data(config)
    eval_params = config["eval_params"]
    pipeline_params = config["pipeline_params"]
    management = config["management"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting RULER evaluation via %s.", pipeline_params["method"])
    model, tokenizer = initialize_model_tokenizer(pipeline_params=pipeline_params)
    model.eval()

    data_file = Path(management["output_folder_dir"]) / eval_params["dataset"] / "validation.jsonl"
    pred_file = os.path.join(management["output_folder_dir"], eval_params["dataset"] + ".jsonl")
    pred_path = Path(pred_file)
    with data_file.open(encoding="utf-8") as json_file:
        data = (json.loads(line) for line in json_file if line.strip())
        get_pred(
            model=model,
            tokenizer=tokenizer,
            data=data,
            pred_file=pred_file,
            device=device,
            pipeline_params=pipeline_params,
            eval_params=eval_params,
            batch_generate=batch_generate,
            pass_pipeline_params=pass_pipeline_params,
        )
    get_eval(config)
    summary = _read_summary(Path(management["output_folder_dir"]) / "summary.csv", eval_params["dataset"])
    return {eval_params["dataset"]: summary}, _read_compact_raw_results(pred_path)
