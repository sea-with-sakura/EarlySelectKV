from pipeline.math_reasoning_eval import eval_math_reasoning as _eval_math_reasoning

from .inference import batch_generate, initialize_model_tokenizer


def eval_math_reasoning(config):
    return _eval_math_reasoning(
        config=config,
        initialize_model_tokenizer=initialize_model_tokenizer,
        batch_generate=batch_generate,
        pass_pipeline_params=True,
    )
