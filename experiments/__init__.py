from .top_1_acc import build_tuning_runs, build_eval_runs, aggregate_eval_results
from .functional_signatures import build_functional_signature_runs

experiments = {
    "build_tuning_runs": build_tuning_runs,
    "build_eval_runs": build_eval_runs,
    "aggregate_eval_results": aggregate_eval_results,
    "build_functional_signature_runs": build_functional_signature_runs,
}
