from pipeline.passkey_eval import eval_passkey_retrieval as _eval_passkey_retrieval

from .inference import batch_generate, initialize_model_tokenizer


def eval_passkey_retrieval(config):
    return _eval_passkey_retrieval(
        config=config,
        initialize_model_tokenizer=initialize_model_tokenizer,
        batch_generate=batch_generate,
        pass_pipeline_params=False,
    )
