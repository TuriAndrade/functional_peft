import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SUFFIX = "estimate_mc_fim"
_EVAL_SUFFIX = "eval"
_TOP1_ACC_DIR = "top1_acc_ft"
_RUN_SEED = 0

_LINEAR_PROBE_EXP = "linear_probe"
_FULL_FT_EXP = "full_ft"
_VPT_EXP = "vpt"
_EXPERIMENT_NAMES = (_LINEAR_PROBE_EXP, _FULL_FT_EXP, _VPT_EXP)

_VPT_DEEP_PROMPT_MODE = "deep"
_VPT_SHALLOW_PROMPT_MODE = "shallow"

_COMMON_ARGS = {
    "mc_samples": 20,
    "device": "cuda",
    "seed": _RUN_SEED,
    "batch_size": 1,
    "max_batches": None,
    "plot_norm": "percentile",
    "plot_percentile": 99.0,
}


def _experiments_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _backbone_results_dir(exp_name: str, backbone_name: str) -> Path:
    return _experiments_dir() / exp_name / _TOP1_ACC_DIR / backbone_name


def _load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _list_dataset_names(backbone_name: str) -> List[str]:
    dataset_names = None

    for exp_name in _EXPERIMENT_NAMES:
        results_dir = _backbone_results_dir(
            exp_name=exp_name,
            backbone_name=backbone_name,
        )

        if not results_dir.exists():
            raise FileNotFoundError(f"Missing results directory: {results_dir}")

        current_dataset_names = {
            path.name for path in results_dir.iterdir() if path.is_dir()
        }

        if dataset_names is None:
            dataset_names = current_dataset_names
            continue

        if current_dataset_names != dataset_names:
            missing = sorted(dataset_names - current_dataset_names)
            extra = sorted(current_dataset_names - dataset_names)
            raise ValueError(
                f"Dataset mismatch for {exp_name} at {results_dir}. "
                f"Missing: {missing}. Extra: {extra}."
            )

    if dataset_names is None or len(dataset_names) == 0:
        raise ValueError(f"No datasets found for backbone: {backbone_name}")

    return sorted(dataset_names)


def _validate_model_dir(
    model_dir: Path,
    dataset_name: str,
    backbone_name: str,
) -> Dict:
    config = _load_json(model_dir / "config.json")
    model_path = model_dir / "model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

    if config.get("dataset_name") != dataset_name:
        raise ValueError(
            f"Expected dataset '{dataset_name}' in {model_dir}, "
            f"found '{config.get('dataset_name')}'."
        )

    if config.get("backbone_name") != backbone_name:
        raise ValueError(
            f"Expected backbone '{backbone_name}' in {model_dir}, "
            f"found '{config.get('backbone_name')}'."
        )

    return config


def _find_eval_run_dir(
    exp_name: str,
    dataset_name: str,
    backbone_name: str,
    prompt_mode: Optional[str] = None,
) -> Path:
    eval_dir = (
        _backbone_results_dir(exp_name=exp_name, backbone_name=backbone_name)
        / dataset_name
        / _EVAL_SUFFIX
    )

    if not eval_dir.exists():
        raise FileNotFoundError(f"Missing eval directory: {eval_dir}")

    candidates = []

    for run_dir in sorted(eval_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.endswith(f"_{_RUN_SEED}"):
            continue

        config = _validate_model_dir(
            model_dir=run_dir,
            dataset_name=dataset_name,
            backbone_name=backbone_name,
        )

        if prompt_mode is not None and config.get("prompt_mode") != prompt_mode:
            continue

        candidates.append(run_dir)

    if len(candidates) == 0:
        suffix = f" with prompt_mode={prompt_mode}" if prompt_mode else ""
        raise FileNotFoundError(
            f"No seed {_RUN_SEED} eval run found in {eval_dir}{suffix}"
        )

    if len(candidates) > 1:
        raise ValueError(
            f"Expected one seed {_RUN_SEED} eval run in {eval_dir}, "
            f"found {len(candidates)}: {candidates}"
        )

    return candidates[0]


def _build_load_model_paths(dataset_name: str, backbone_name: str) -> List[str]:
    return [
        str(
            _find_eval_run_dir(
                exp_name=_LINEAR_PROBE_EXP,
                dataset_name=dataset_name,
                backbone_name=backbone_name,
            )
        ),
        str(
            _find_eval_run_dir(
                exp_name=_FULL_FT_EXP,
                dataset_name=dataset_name,
                backbone_name=backbone_name,
            )
        ),
        str(
            _find_eval_run_dir(
                exp_name=_VPT_EXP,
                dataset_name=dataset_name,
                backbone_name=backbone_name,
                prompt_mode=_VPT_DEEP_PROMPT_MODE,
            )
        ),
        str(
            _find_eval_run_dir(
                exp_name=_VPT_EXP,
                dataset_name=dataset_name,
                backbone_name=backbone_name,
                prompt_mode=_VPT_SHALLOW_PROMPT_MODE,
            )
        ),
    ]


def build_args(backbone_name: str) -> Tuple[List[Dict], str, str]:
    save_dir = Path(__file__).resolve().parent / "mc_fim_estimates" / backbone_name
    dataset_names = _list_dataset_names(backbone_name=backbone_name)

    args = [
        {
            **_COMMON_ARGS,
            "load_model_paths": _build_load_model_paths(
                dataset_name=dataset_name,
                backbone_name=backbone_name,
            ),
            "save_dir": str(save_dir),
            "save_name": dataset_name,
        }
        for dataset_name in dataset_names
    ]

    output_path = str(save_dir / f"{_SUFFIX}.txt")
    exe_prefix = "python estimate_mc_fim_exe.py"

    return args, output_path, exe_prefix
