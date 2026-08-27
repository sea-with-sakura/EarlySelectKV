from pipeline.ruler_eval import eval_ruler as _eval_ruler

from .inference import batch_generate, initialize_model_tokenizer


def eval_ruler(config):
    return _eval_ruler(
        config=config,
        initialize_model_tokenizer=initialize_model_tokenizer,
        batch_generate=batch_generate,
        pass_pipeline_params=False,
    )
