import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.config_utils import load_json_with_extends, method_config_keys

TOKEN_BUDGET_FREE_METHODS = {"baseline"}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_raw_results_path(input_eval_config: Path) -> Path:
    try:
        data = load_json(input_eval_config)
        raw_name = data.get("management", {}).get("sub_dir", {}).get("raw_results", "raw_results.json")
    except Exception:
        raw_name = "raw_results.json"
    return (input_eval_config.parent.parent / raw_name).resolve()


def resolve_output_config_path(input_eval_config: Path) -> Path:
    try:
        data = load_json(input_eval_config)
        config_name = data.get("management", {}).get("sub_dir", {}).get("output_config", "output_config.json")
    except Exception:
        config_name = "output_config.json"
    return (input_eval_config.parent.parent / config_name).resolve()


def is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def is_longbench_eval_config(path: Path) -> bool:
    return "longbench" in path.parts


def longbench_prediction_artifacts_complete(output_dir: Path, method: str, dataset: str) -> bool:
    pred_dir = output_dir / "pred" / method
    result_path = pred_dir / "result.json"
    if not is_nonempty_file(result_path):
        return False
    if not dataset:
        return True
    prediction_files = sorted(path for path in pred_dir.glob(f"{dataset}_*.jsonl") if is_nonempty_file(path))
    return len(prediction_files) == 1


def normalize_expected_eval(*, eval_config_path: Path, dataset: str, max_seq_length: str) -> dict:
    expected_eval = load_json(eval_config_path)
    eval_params = expected_eval.get("eval_params", {})
    if eval_params.get("benchmark") == "synthetic":
        eval_params["dataset"] = dataset
        if "niah" in dataset:
            eval_params["max_new_tokens"] = 128
        elif "vt" in dataset:
            eval_params["max_new_tokens"] = 30
        elif "cwe" in dataset:
            eval_params["max_new_tokens"] = 120
        elif "fwe" in dataset:
            eval_params["max_new_tokens"] = 50
        elif "qa" in dataset:
            eval_params["max_new_tokens"] = 32
        if max_seq_length:
            eval_params["max_seq_length"] = int(max_seq_length)
        expected_eval["eval_params"] = eval_params
    return expected_eval


def normalize_expected_pipeline(
    *,
    pipeline_config_path: Path,
    method: str,
    token_budget: str,
    scdq_mode: str,
    seed: str,
    model_name_or_path: str,
) -> dict:
    expected_pipeline = load_json_with_extends(pipeline_config_path)
    pipeline_params = expected_pipeline.get("pipeline_params", {})
    if model_name_or_path:
        pipeline_params["model_name"] = model_name_or_path
    pipeline_params["method"] = method
    pipeline_params["seed"] = int(seed)
    if method not in TOKEN_BUDGET_FREE_METHODS:
        pipeline_params["token_budget"] = int(token_budget) if token_budget else 1024
    else:
        pipeline_params.pop("token_budget", None)
    pipeline_params["scdq_mode"] = scdq_mode in {"1", "true", "True", "yes", "YES"}

    active_method_keys = set(method_config_keys(method))
    for key, value in list(pipeline_params.items()):
        if isinstance(value, dict) and key not in active_method_keys:
            pipeline_params.pop(key, None)
    expected_pipeline["pipeline_params"] = pipeline_params
    return expected_pipeline


def stable(obj: dict) -> dict:
    return json.loads(json.dumps(obj, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pipeline-config", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--token-budget", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--max-seq-length", default="")
    parser.add_argument("--scdq-mode", default="0")
    parser.add_argument("--seed", default="42")
    parser.add_argument("--model-name-or-path", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        return 0

    input_config_dir = output_dir / "input_config"
    input_eval_config = input_config_dir / "input_eval_config.json"
    input_pipeline_config = input_config_dir / "input_pipeline_config.json"

    if not input_eval_config.is_file() or not input_pipeline_config.is_file():
        return 2

    raw_results_path = resolve_raw_results_path(input_eval_config)
    output_config_path = resolve_output_config_path(input_eval_config)
    if not is_nonempty_file(raw_results_path) or not is_nonempty_file(output_config_path):
        return 0

    eval_config_path = Path(args.eval_config)
    if is_longbench_eval_config(eval_config_path) and not longbench_prediction_artifacts_complete(
        output_dir=output_dir,
        method=args.method,
        dataset=args.dataset,
    ):
        return 0

    try:
        existing_eval = load_json(input_eval_config)
        existing_pipeline = load_json(input_pipeline_config)
        expected_eval = normalize_expected_eval(
            eval_config_path=eval_config_path,
            dataset=args.dataset,
            max_seq_length=args.max_seq_length,
        )
        expected_pipeline = normalize_expected_pipeline(
            pipeline_config_path=Path(args.pipeline_config),
            method=args.method,
            token_budget=args.token_budget,
            scdq_mode=args.scdq_mode,
            seed=args.seed,
            model_name_or_path=args.model_name_or_path,
        )
    except Exception:
        return 3

    match = stable(existing_eval) == stable(expected_eval) and stable(existing_pipeline) == stable(expected_pipeline)
    return 1 if match else 2


if __name__ == "__main__":
    raise SystemExit(main())
