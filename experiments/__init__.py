from .vpt.tuning import (
    build_args as vpt_tuning_ba,
    build_results as vpt_tuning_br,
)

from .vpt.eval import (
    build_args as vpt_eval_ba,
    build_results as vpt_eval_br,
    aggregate_results as vpt_eval_ar,
)

build_args = {
    "vpt_tuning": vpt_tuning_ba,
    "vpt_eval": vpt_eval_ba,
}

build_results = {
    "vpt_tuning": vpt_tuning_br,
    "vpt_eval": vpt_eval_br,
}

aggregate_results = {
    "vpt_eval": vpt_eval_ar,
}
