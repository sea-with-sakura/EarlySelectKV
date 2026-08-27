import logging
import argparse
import sys
import os
import json
import datetime
import copy
from zoneinfo import ZoneInfo
import random
import torch
import numpy as np

from pipeline.config_utils import load_json_with_extends, method_config_keys
from pipeline.hf_utils import setup_hf_auth
from utils.longbench_utils.constants import LONGBENCH_DATASET, LONGBENCH_E_DATASET

logger = logging.getLogger("main")
TOKEN_BUDGET_FREE_METHODS = {"baseline"}

RESET = "\033[0m"
LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",  # cyan
    logging.INFO: "\033[32m",  # green
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[1;31m",  # bold red
}


class ColoredFormatter(logging.Formatter):
    """Colorize console log lines by log level."""

    def __init__(self, fmt, use_color=True):
        super().__init__(fmt)
        self.use_color = use_color

    def format(self, record):
        message = super().format(record)
        if not self.use_color:
            return message
        color = LEVEL_COLORS.get(record.levelno)
        if not color:
            return message
        return f"{color}{message}{RESET}"


def configure_fast_reproducibility(seed):
    """Set reproducibility knobs without disabling fast CUDA kernels."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(False)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_desc", type=str, help="experiment description.", default="exp_test")
    parser.add_argument(
        "--pipeline_config_dir",
        type=str,
        help="file path of pipeline config.",
        default="config/pipeline_config/longbench/mistral-7b-instruct-v0.2.json",
    )
    parser.add_argument(
        "--eval_config_dir",
        type=str,
        help="file path of eval config.",
        default="config/eval_config/longbench/narrativeqa.json",
    )
    parser.add_argument("--output_folder_dir", type=str, help="path of output model", default="result/exp_test")
    parser.add_argument("--job_post_via", default="slurm_sbatch", type=str, help="slurm_sbatch or terminal")
    parser.add_argument("--method", type=str, help="evaluation method", default="baseline")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Optional local LitGPT checkpoint override for pipeline_params['model_name'].",
    )
    parser.add_argument("--token_budget", type=int, default=1024, help="token_budget")
    parser.add_argument(
        "--prefill_chunk_size",
        type=int,
        default=None,
        help="maximum prompt chunk size for incremental prefill; <=0 disables chunking",
    )
    parser.add_argument("--dataset", type=str, help="task for ruler benchmark", default="narrativeqa")
    parser.add_argument("--max_seq_length", type=int, default=4000, help="max seq length for ruler benchmark")
    parser.add_argument("--scdq_mode", action="store_true")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 42)), help="random seed")

    args = parser.parse_args()

    if args.output_folder_dir != "":
        if args.output_folder_dir[-1] != "/":
            args.output_folder_dir += "/"
    else:
        logger.error(f"Valid {args.output_folder_dir} is required.")

    return args


# Output in terminal and exp.log file under output_folder_dir.
def set_logger(output_folder_dir):
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    ct_timezone = ZoneInfo("Asia/Shanghai")
    log_formatter = logging.Formatter("%(asctime)s | %(levelname)s : %(message)s")
    log_formatter.converter = lambda *args: datetime.datetime.now(ct_timezone).timetuple()
    file_handler = logging.FileHandler(output_folder_dir + "exp.log", mode="w")
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    use_color = hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and "NO_COLOR" not in os.environ
    colored_formatter = ColoredFormatter("%(asctime)s | %(levelname)s : %(message)s", use_color=use_color)
    colored_formatter.converter = lambda *args: datetime.datetime.now(ct_timezone).timetuple()
    console_handler.setFormatter(colored_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    return logger


def register_args_and_configs(args):
    # Make outer output dir.
    os.makedirs(args.output_folder_dir, exist_ok=True)
    logger.info("Output folder dir %s ready.", args.output_folder_dir)

    # Copy input eval config to output dir.
    with open(args.eval_config_dir) as eval_config_f:
        eval_config = json.load(eval_config_f)
        logger.info(f"Input eval config file {args.eval_config_dir} loaded.")
        if eval_config["eval_params"].get("benchmark") == "synthetic":
            eval_config["eval_params"]["dataset"] = args.dataset
            if "niah" in args.dataset:
                eval_config["eval_params"]["max_new_tokens"] = 128
            elif "vt" in args.dataset:
                eval_config["eval_params"]["max_new_tokens"] = 30
            elif "cwe" in args.dataset:
                eval_config["eval_params"]["max_new_tokens"] = 120
            elif "fwe" in args.dataset:
                eval_config["eval_params"]["max_new_tokens"] = 50
            elif "qa" in args.dataset:
                eval_config["eval_params"]["max_new_tokens"] = 32
            eval_config["eval_params"]["max_seq_length"] = args.max_seq_length

    # Make subdir under output dir to store input configs.
    input_config_subdir = eval_config["management"]["sub_dir"]["input_config"]
    os.makedirs(args.output_folder_dir + input_config_subdir, exist_ok=True)
    logger.info("Input config subdir %s ready.", args.output_folder_dir + input_config_subdir)

    input_eval_config_path = args.output_folder_dir + input_config_subdir + "input_eval_config.json"
    with open(input_eval_config_path, "w+") as input_eval_config_f:
        json.dump(eval_config, input_eval_config_f, indent=4)
        logger.info(f"Input eval config file {args.eval_config_dir} saved to {input_eval_config_path}.")

    # Copy input pipeline config to output dir.
    pipeline_config = load_json_with_extends(args.pipeline_config_dir)
    logger.info(f"Input pipeline config file {args.pipeline_config_dir} loaded.")
    if args.model_name_or_path is not None:
        pipeline_config["pipeline_params"]["model_name"] = args.model_name_or_path
    pipeline_config["pipeline_params"]["method"] = args.method
    pipeline_config["pipeline_params"]["seed"] = args.seed
    if args.prefill_chunk_size is not None:
        pipeline_config["pipeline_params"]["prefill_chunk_size"] = args.prefill_chunk_size
    if args.method not in TOKEN_BUDGET_FREE_METHODS:
        pipeline_config["pipeline_params"]["token_budget"] = args.token_budget
    else:
        pipeline_config["pipeline_params"].pop("token_budget", None)
    pipeline_config["pipeline_params"]["scdq_mode"] = args.scdq_mode
    input_pipeline_config_path = args.output_folder_dir + input_config_subdir + "input_pipeline_config.json"
    pipeline_config_for_dump = prune_inactive_method_configs(pipeline_config)
    with open(input_pipeline_config_path, "w+") as input_pipeline_config_f:
        json.dump(pipeline_config_for_dump, input_pipeline_config_f, indent=4)
        logger.info(f"Input pipeline config file {args.pipeline_config_dir} saved to {input_pipeline_config_path}.")

    # Fuse and complete pipeline config, eval config, and args from argparser into a general config.
    config = dict()
    config["pipeline_params"] = pipeline_config["pipeline_params"]
    config["eval_params"] = eval_config["eval_params"]
    config["eval_results"] = dict()  # processed result

    config["management"] = dict()
    config["management"]["exp_desc"] = args.exp_desc
    config["management"]["pipeline_config_dir"] = args.pipeline_config_dir
    config["management"]["eval_config_dir"] = args.eval_config_dir
    config["management"]["output_folder_dir"] = args.output_folder_dir
    config["management"]["job_post_via"] = args.job_post_via
    if (
        config["management"]["job_post_via"] == "slurm_sbatch"
    ):  # Add slurm info to config['management'] if the job is triggered via slurm sbatch.
        try:
            config["management"]["slurm_info"] = register_slurm_sbatch_info()
        except Exception:
            config["management"]["job_post_via"] = "terminal"  # Likely not a slurm job, rollback to terminal post.
    config["management"]["sub_dir"] = eval_config["management"]["sub_dir"]

    return config


def register_slurm_sbatch_info():
    slurm_job_id = os.environ["SLURM_JOB_ID"]
    slurm_job_name = os.getenv("SLURM_JOB_NAME")
    slurm_out_file_dir = os.getenv("SLURM_SUBMIT_DIR") + "/slurm-" + os.getenv("SLURM_JOB_ID") + ".out"

    logger.info(f"Slurm job #{slurm_job_id} ({slurm_job_name}) running with slurm.out file at {slurm_out_file_dir}.")

    return {
        "slurm_job_id": slurm_job_id,
        "slurm_job_name": slurm_job_name,
        "slurm_out_file_dir": slurm_out_file_dir,
    }


def prune_inactive_method_configs(config):
    config_for_dump = copy.deepcopy(config)
    pipeline_params = config_for_dump.get("pipeline_params", {})
    method_key = str(pipeline_params.get("method", ""))
    active_method_keys = set(method_config_keys(method_key))
    for key, value in list(pipeline_params.items()):
        if isinstance(value, dict) and key not in active_method_keys:
            pipeline_params.pop(key, None)
    return config_for_dump


def register_result(processed_results, raw_results, config):
    raw_results_path = config["management"]["output_folder_dir"] + config["management"]["sub_dir"]["raw_results"]
    with open(raw_results_path, "w+") as raw_results_f:
        json.dump(raw_results, raw_results_f, indent=4)
        logger.info(f"raw_results file saved to {raw_results_path}.")

    config["eval_results"]["processed_results"] = processed_results
    logger.info("Processed results:")
    logger.info(json.dumps(config["eval_results"]["processed_results"], indent=4))


def register_exp_time(start_time, end_time, config):
    config["management"]["start_time"] = str(start_time)
    config["management"]["end_time"] = str(end_time)
    config["management"]["exp_duration"] = str(end_time - start_time)


def register_output_config(config):
    output_config_path = config["management"]["output_folder_dir"] + config["management"]["sub_dir"]["output_config"]
    output_config = prune_inactive_method_configs(config)
    with open(output_config_path, "w+") as output_config_f:
        json.dump(output_config, output_config_f, indent=4)
        logger.info(f"output_config file saved to {output_config_path}.")


def log_experiment_start(config, start_time, seed, include_config=False):
    logger.info(
        "Experiment %s (SEED=%s) started at %s.",
        config["management"]["exp_desc"],
        seed,
        start_time,
    )
    if include_config:
        config_for_log = prune_inactive_method_configs(config)
        logger.info("Input config:\n%s", json.dumps(config_for_log, indent=4))


def log_experiment_end(config, end_time):
    logger.info(
        "Experiment %s ended at %s. Duration: %s",
        config["management"]["exp_desc"],
        end_time,
        config["management"]["exp_duration"],
    )


def log_prediction_file(prediction_path):
    logger.info("Prediction file saved to %s.", prediction_path)


def run_pipeline_main(
    *,
    eval_longbench=None,
    eval_passkey_retrieval=None,
    eval_ruler=None,
    eval_math_reasoning=None,
    include_longbench_e=True,
    include_config=False,
    unsupported_message=None,
):
    args = parse_args()
    seed = int(args.seed)
    configure_fast_reproducibility(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    ct_timezone = ZoneInfo("Asia/Shanghai")
    start_time = datetime.datetime.now(ct_timezone)
    os.makedirs(args.output_folder_dir, exist_ok=True)
    active_logger = set_logger(args.output_folder_dir)
    setup_hf_auth(active_logger)
    config = register_args_and_configs(args)
    log_experiment_start(config, start_time, seed, include_config=include_config)

    try:
        dataset = config["eval_params"]["dataset"]
        is_passkey = dataset in {"passkey_retrieval", "magic_city_number_retrieval"}
        is_longbench = dataset in LONGBENCH_DATASET or (include_longbench_e and dataset in LONGBENCH_E_DATASET)

        if is_passkey and eval_passkey_retrieval is not None:
            processed_results, raw_results = eval_passkey_retrieval(config)
            register_result(processed_results, raw_results, config)
        elif is_longbench and eval_longbench is not None:
            processed_results, raw_results = eval_longbench(config)
            register_result(processed_results, raw_results, config)
        elif config["eval_params"].get("benchmark") == "synthetic" and eval_ruler is not None:
            result = eval_ruler(config)
            if result is not None:
                processed_results, raw_results = result
                register_result(processed_results, raw_results, config)
        elif config["eval_params"].get("benchmark") == "math_reasoning" and eval_math_reasoning is not None:
            processed_results, raw_results = eval_math_reasoning(config)
            register_result(processed_results, raw_results, config)
        else:
            message = unsupported_message or "Invalid config['eval_params']['dataset'] input: %s."
            error_text = message % dataset if "%s" in message else message
            active_logger.error(error_text)
            raise ValueError(error_text)

        end_time = datetime.datetime.now(ct_timezone)
        register_exp_time(start_time, end_time, config)
        register_output_config(config)
        log_experiment_end(config, end_time)
    except Exception:
        active_logger.exception("Experiment failed with an exception.")
        raise
