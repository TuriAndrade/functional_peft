from .vpt.tuning import (
    build_args as vpt_tuning_ba,
    build_results as vpt_tuning_br,
)

from .vpt.eval import (
    build_args as vpt_eval_ba,
    build_results as vpt_eval_br,
    aggregate_results as vpt_eval_ar,
)

from .linear_probe.tuning import (
    build_args as linear_probe_tuning_ba,
    build_results as linear_probe_tuning_br,
)

from .linear_probe.eval import (
    build_args as linear_probe_eval_ba,
    build_results as linear_probe_eval_br,
    aggregate_results as linear_probe_eval_ar,
)

from .full_ft.tuning import (
    build_args as full_ft_tuning_ba,
    build_results as full_ft_tuning_br,
)

from .full_ft.eval import (
    build_args as full_ft_eval_ba,
    build_results as full_ft_eval_br,
    aggregate_results as full_ft_eval_ar,
)

from .task_functional_signatures.estimate_functional_signatures import (
    build_args as estimate_functional_signatures_ba,
)

from .task_functional_signatures.analyze_vtab_functional_semantics import (
    build_args as analyze_vtab_functional_semantics_ba,
)

build_args = {
    "vpt_tuning": vpt_tuning_ba,
    "vpt_eval": vpt_eval_ba,
    "linear_probe_tuning": linear_probe_tuning_ba,
    "linear_probe_eval": linear_probe_eval_ba,
    "full_ft_tuning": full_ft_tuning_ba,
    "full_ft_eval": full_ft_eval_ba,
    "estimate_functional_signatures": estimate_functional_signatures_ba,
    "analyze_vtab_functional_semantics": analyze_vtab_functional_semantics_ba,
}

build_results = {
    "vpt_tuning": vpt_tuning_br,
    "vpt_eval": vpt_eval_br,
    "linear_probe_tuning": linear_probe_tuning_br,
    "linear_probe_eval": linear_probe_eval_br,
    "full_ft_tuning": full_ft_tuning_br,
    "full_ft_eval": full_ft_eval_br,
}

aggregate_results = {
    "vpt_eval": vpt_eval_ar,
    "linear_probe_eval": linear_probe_eval_ar,
    "full_ft_eval": full_ft_eval_ar,
}
