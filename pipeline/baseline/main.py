import pipeline.main_utils as main_utils

from .eval_longbench import eval_longbench
from .eval_math_reasoning import eval_math_reasoning
from .eval_passkey_retrieval import eval_passkey_retrieval
from .eval_ruler import eval_ruler


def main():
    main_utils.run_pipeline_main(
        eval_longbench=eval_longbench,
        eval_passkey_retrieval=eval_passkey_retrieval,
        eval_ruler=eval_ruler,
        eval_math_reasoning=eval_math_reasoning,
        include_config=True,
    )


if __name__ == "__main__":
    main()
