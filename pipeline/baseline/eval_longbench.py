from pipeline.longbench_eval import eval_longbench as _eval_longbench

from .inference import batch_generate, initialize_model_tokenizer


def eval_longbench(config):
    return _eval_longbench(
        config=config,
        initialize_model_tokenizer=initialize_model_tokenizer,
        batch_generate=batch_generate,
        pass_pipeline_params=False,
        bos=False,
    )
