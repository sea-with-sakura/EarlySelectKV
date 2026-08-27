import pipeline.main_utils as main_utils

from .eval_longbench import eval_longbench


def main():
    main_utils.run_pipeline_main(
        eval_longbench=eval_longbench,
        include_longbench_e=False,
        unsupported_message="Stage2 anchor probe currently supports LongBench datasets, got %s.",
    )


if __name__ == "__main__":
    main()
