from pipeline.lookaheadkv.inference import _compute_lookaheadkv_params
from pipeline.model_utils import post_process
from pipeline.rocketkv.inference import _compute_rocketkv_params


def test_qwen_post_process_keeps_assistant_response():
    response = "<|im_start|>assistant\nThe answer is 42.<|im_end|>"

    assert post_process(response, pipeline_params={"chat_template": "qwen"}) == "The answer is 42."


def test_qwen_post_process_handles_generic_chatml_prefix():
    response = "<|im_start|>\nA second answer.<|im_end|>"

    assert post_process(response, pipeline_params={"chat_template": "qwen2.5"}) == "A second answer."


def test_rocketkv_capacity_reserves_long_generation_tail():
    params = _compute_rocketkv_params(prompt_len=128, max_new_tokens=512, token_budget=64)

    assert params["token_capacity_budget"] == 640
    assert params["prompt_budget"] == 128


def test_rocketkv_and_earlyselectkv_use_same_capacity_guard():
    rocket = _compute_rocketkv_params(prompt_len=4096, max_new_tokens=2048, token_budget=512)
    earlyselect = _compute_lookaheadkv_params(prompt_len=4096, max_new_tokens=2048, token_budget=512)

    assert rocket["token_capacity_budget"] == earlyselect["token_capacity_budget"]
