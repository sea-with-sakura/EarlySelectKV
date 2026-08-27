import logging
from typing import Callable

import utils.passkey_utils.passkey_main as passkey_main
import utils.passkey_utils.passkey_utils as passkey_utils

logger = logging.getLogger("main")


def eval_passkey_retrieval(
    *,
    config: dict,
    initialize_model_tokenizer: Callable,
    batch_generate: Callable,
    pass_pipeline_params: bool,
):
    raw_exp_results = passkey_main.prepare_passkey_retrieval_input(config)
    eval_params = config["eval_params"]
    pipeline_params = config["pipeline_params"]

    logger.info("Starting evaluation via %s", pipeline_params["method"])
    model, tokenizer = initialize_model_tokenizer(pipeline_params=pipeline_params)

    passkey_utils.check_if_out_of_context_window(
        longest_input=raw_exp_results[-1]["full_input"],
        model_max_len=pipeline_params["model_max_len"],
        tokenizer=tokenizer,
        out_of_max_len_allowed=pipeline_params["out_of_max_len_allowed"],
    )

    batch_size = int(pipeline_params["batch_size"])
    batched_raw_exp_results = [
        raw_exp_results[i : i + batch_size] for i in range(0, len(raw_exp_results), batch_size)
    ]

    for i, one_batch in enumerate(batched_raw_exp_results):
        batched_input = [item["full_input"] for item in one_batch]
        generate_kwargs = {"pipeline_params": pipeline_params} if pass_pipeline_params else {}
        batched_responses = batch_generate(
            batched_input=batched_input,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=eval_params["max_new_tokens"],
            **generate_kwargs,
        )

        for one_exp_results, one_response in zip(one_batch, batched_responses):
            one_exp_results["response"] = one_response

        logger.info(
            "Finished evaluating batch %s/%s (batch_size = %s).",
            i + 1,
            len(batched_raw_exp_results),
            batch_size,
        )
    logger.info("Finished evaluating all %s batches (batch_size = %s).", len(batched_raw_exp_results), batch_size)

    processed_results, raw_results = passkey_utils.process_raw_exp_results(
        raw_exp_results=raw_exp_results,
        metrics=eval_params["eval_metrics"],
    )
    logger.info("raw_exp_results processed.")
    return processed_results, raw_results
