import json
from pathlib import Path
from typing import Dict, List, Tuple

_SUFFIX = "estimate_mc_fim"
_EVAL_SUFFIX = "eval"
_TOP1_ACC_DIR = "top1_acc_ft"
_RUN_SEED = 0

_EXPERIMENT_NAMES = ("linear_probe", "full_ft", "vpt")

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
    return json.loads(path.read_text(encoding="utf-8"))


def _list_dataset_names(backbone_name: str) -> List[str]:
    datasets = None

    for exp_name in _EXPERIMENT_NAMES:
        results_dir = _backbone_results_dir(exp_name, backbone_name)

        if not results_dir.exists():
            raise FileNotFoundError(f"Missing results directory: {results_dir}")

        current = {p.name for p in results_dir.iterdir() if p.is_dir()}

        if datasets is None:
            datasets = current
        elif current != datasets:
            raise ValueError(
                f"Dataset mismatch for {exp_name} at {results_dir}. "
                f"Missing: {sorted(datasets - current)}. "
                f"Extra: {sorted(current - datasets)}."
            )

    if not datasets:
        raise ValueError(f"No datasets found for backbone: {backbone_name}")

    return sorted(datasets)


def _validate_run_dir(run_dir: Path, dataset_name: str, backbone_name: str) -> Dict:
    config = _load_json(run_dir / "config.json")

    if not (run_dir / "model.pt").exists():
        raise FileNotFoundError(f"Missing model checkpoint: {run_dir / 'model.pt'}")

    if config.get("dataset_name") != dataset_name:
        raise ValueError(
            f"Expected dataset '{dataset_name}' in {run_dir}, "
            f"found '{config.get('dataset_name')}'."
        )

    if config.get("backbone_name") != backbone_name:
        raise ValueError(
            f"Expected backbone '{backbone_name}' in {run_dir}, "
            f"found '{config.get('backbone_name')}'."
        )

    return config


def _model_name(exp_name: str, backbone_name: str, config: Dict) -> str:
    if exp_name == "full_ft":
        return f"{backbone_name}_full_ft"

    if exp_name == "linear_probe":
        return f"{backbone_name}_linear_probe"

    return f"{backbone_name}_vpt_{config['prompt_mode']}_{config['prompt_length']}"


def _find_eval_runs(
    exp_name: str,
    dataset_name: str,
    backbone_name: str,
) -> List[Tuple[Path, str]]:
    eval_dir = (
        _backbone_results_dir(exp_name, backbone_name) / dataset_name / _EVAL_SUFFIX
    )

    if not eval_dir.exists():
        raise FileNotFoundError(f"Missing eval directory: {eval_dir}")

    runs = []

    for run_dir in sorted(eval_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.endswith(f"_{_RUN_SEED}"):
            continue

        config = _validate_run_dir(run_dir, dataset_name, backbone_name)
        runs.append((run_dir, _model_name(exp_name, backbone_name, config)))

    if not runs:
        raise FileNotFoundError(f"No seed {_RUN_SEED} eval runs found in {eval_dir}")

    return runs


def _build_load_models(
    dataset_name: str, backbone_name: str
) -> Tuple[List[str], List[str]]:
    runs = [
        run
        for exp_name in _EXPERIMENT_NAMES
        for run in _find_eval_runs(exp_name, dataset_name, backbone_name)
    ]

    return [str(path) for path, _ in runs], [name for _, name in runs]


def build_args(backbone_name: str) -> Tuple[List[Dict], str, str]:
    save_dir = Path(__file__).resolve().parent / "mc_fim_estimates" / backbone_name
    args = []

    for dataset_name in _list_dataset_names(backbone_name):
        load_model_paths, load_model_names = _build_load_models(
            dataset_name=dataset_name,
            backbone_name=backbone_name,
        )

        args.append(
            {
                **_COMMON_ARGS,
                "load_model_paths": load_model_paths,
                "load_model_names": load_model_names,
                "save_dir": str(save_dir),
                "save_name": dataset_name,
            }
        )

    return args, str(save_dir / f"{_SUFFIX}.txt"), "python estimate_mc_fim_exe.py"
