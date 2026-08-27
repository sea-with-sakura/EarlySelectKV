from pipeline.math_reasoning_eval import answers_match
from pipeline.model_utils import post_process


def test_qwen_post_process_keeps_generated_answer_after_assistant_marker():
    response = "<|im_start|>assistant\nWe solve it. Therefore \\boxed{204}.<|im_end|>"
    processed = post_process(
        response,
        pipeline_params={"model_name": "modelzoo/Qwen2.5-7B-Instruct"},
    )

    assert processed == "We solve it. Therefore \\boxed{204}."
    assert answers_match(processed, "204")
