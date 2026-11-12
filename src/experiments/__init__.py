from .top_1_acc import (
    build_tuning_runs,
    build_eval_runs,
    aggregate_eval_results,
    prune_finished_runs,
)

experiments = {
    "build_tuning_runs": build_tuning_runs,
    "build_eval_runs": build_eval_runs,
    "aggregate_eval_results": aggregate_eval_results,
    "prune_finished_runs": prune_finished_runs,
}
