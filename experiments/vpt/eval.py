import json
from pathlib import Path
from typing import List, Dict, Tuple
import math

from scipy.stats import t as student_t
import pandas as pd

from .tuning import _TUNING_SEED

_EVAL_SEEDS = [0, 1, 2]

_COMMON_ARGS = {
    "freeze_backbone": True,
    "save_ckpt": True,
    "save_ckpt_all_seeds": False,
    "early_stopping": False,
    "eval_on_test": True,
    "bootstrap": False,
}


def _load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_required(config: Dict, key: str):
    if key not in config:
        raise KeyError(f"Missing key '{key}' in config.json")
    return config[key]


def _make_eval_save_name(config: Dict, prefix: str = "") -> str:
    return (
        f"{prefix}"
        f"pm-{config['prompt_mode']}"
        f"_pl-{config['prompt_length']}"
        f"_lr-{config['ref_lr']}"
        f"_wd-{config['ref_wd']}"
    )


def _load_best_tuning_config(base_dir: Path) -> List[Dict]:
    tuning_suffix = "tuning"

    tuning_json_path = base_dir / f"{tuning_suffix}.json"
    tuning_dir = base_dir / tuning_suffix

    tuning_results = _load_json(tuning_json_path)

    best_configs = []

    for _, prompt_mode_results in tuning_results.items():
        for _, prompt_length_results in prompt_mode_results.items():
            best_entry = None

            for entry in prompt_length_results:
                if best_entry is None or entry["val_acc"] > best_entry["val_acc"]:
                    best_entry = entry

            if best_entry is None:
                continue

            tuning_save_name = best_entry["save_name"]
            run_dir = tuning_dir / f"{tuning_save_name}_{_TUNING_SEED}"

            config = _load_json(run_dir / "config.json")
            train_summary = _load_json(run_dir / "train_summary.json")

            best_configs.append(
                {
                    "best_val_acc": best_entry["val_acc"],
                    "best_epoch": _get_required(train_summary, "best_epoch"),
                    "batch_size": _get_required(config, "batch_size"),
                    "gpu_batch_size": _get_required(config, "gpu_batch_size"),
                    "start_lr": _get_required(config, "start_lr"),
                    "final_lr": _get_required(config, "final_lr"),
                    "warmup_epochs": _get_required(config, "warmup_epochs"),
                    "prompt_mode": _get_required(config, "prompt_mode"),
                    "prompt_length": _get_required(config, "prompt_length"),
                    "ref_lr": _get_required(config, "ref_lr"),
                    "ref_wd": _get_required(config, "ref_wd"),
                    "final_wd": _get_required(config, "final_wd"),
                }
            )

    if len(best_configs) == 0:
        raise ValueError(f"No valid tuning entries found in: {tuning_json_path}")

    best_by_prompt_mode = {}

    for config in best_configs:
        prompt_mode = config["prompt_mode"]

        if (
            prompt_mode not in best_by_prompt_mode
            or config["best_val_acc"] > best_by_prompt_mode[prompt_mode]["best_val_acc"]
        ):
            best_by_prompt_mode[prompt_mode] = config

    for config in best_configs:
        is_best_prompt_mode_config = (
            config is best_by_prompt_mode[config["prompt_mode"]]
        )

        config["save_name"] = _make_eval_save_name(
            config,
            prefix="best_" if is_best_prompt_mode_config else "",
        )

        config["run_seeds"] = (
            list(_EVAL_SEEDS) if is_best_prompt_mode_config else [_EVAL_SEEDS[0]]
        )

    return best_configs


def build_args(
    dataset_name: str,
    backbone_name: str,
) -> Tuple[List[Dict], str, str]:
    base_dir = (
        Path(__file__).resolve().parent / "top1_acc_ft" / backbone_name / dataset_name
    )
    suffix = "eval"

    best_configs = _load_best_tuning_config(base_dir=base_dir)

    args = []

    for config in best_configs:
        args.append(
            {
                **_COMMON_ARGS,
                "dataset_name": dataset_name,
                "backbone_name": backbone_name,
                "save_dir": str(base_dir / suffix),
                "run_seeds": config["run_seeds"],
                "batch_size": config["batch_size"],
                "gpu_batch_size": config["gpu_batch_size"],
                "start_lr": config["start_lr"],
                "final_lr": config["final_lr"],
                "warmup_epochs": config["warmup_epochs"],
                "epochs": config["best_epoch"],
                "save_name": config["save_name"],
                "prompt_mode": config["prompt_mode"],
                "prompt_length": config["prompt_length"],
                "ref_lr": config["ref_lr"],
                "ref_wd": config["ref_wd"],
                "final_wd": config["final_wd"],
            }
        )

    output_path = str(base_dir / f"{suffix}.txt")
    exe_prefix = "python vpt_train_exe.py"

    return args, output_path, exe_prefix


def _compute_mean_std_ci(values: List[float], confidence: float = 0.95) -> Dict:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    if len(values) == 0:
        raise ValueError("Cannot compute statistics with zero values")

    n = len(values)
    mean = sum(values) / n

    if n == 1:
        return {
            "mean": mean,
            "std": None,
            "ci_low": None,
            "ci_high": None,
            "ci_half_width": None,
            "confidence": confidence,
            "n": n,
        }

    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    std = math.sqrt(variance)
    standard_error = std / math.sqrt(n)

    alpha = 1.0 - confidence
    degrees_of_freedom = n - 1
    critical_value = student_t.ppf(1.0 - alpha / 2.0, degrees_of_freedom)

    ci_half_width = critical_value * standard_error

    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - ci_half_width,
        "ci_high": mean + ci_half_width,
        "ci_half_width": ci_half_width,
        "confidence": confidence,
        "n": n,
    }


def build_results(
    dataset_name: str,
    backbone_name: str,
    confidence: float = 0.95,
) -> None:
    base_dir = (
        Path(__file__).resolve().parent / "top1_acc_ft" / backbone_name / dataset_name
    )
    suffix = "eval"
    save_dir = base_dir / suffix

    configs = _load_best_tuning_config(base_dir=base_dir)

    results = []

    for config in configs:
        test_acc_values = []
        seed_results = []

        for seed in config["run_seeds"]:
            run_dir = save_dir / f"{config['save_name']}_{seed}"
            test_path = run_dir / "test_acc_eval.json"

            if not test_path.exists():
                print(f"Skipping missing result: {run_dir}")
                continue

            test_data = _load_json(test_path)
            test_acc = test_data["accuracy"]["top1"]

            test_acc_values.append(test_acc)
            seed_results.append(
                {
                    "seed": seed,
                    "test_acc": test_acc,
                    "run_dir": str(run_dir),
                }
            )

        if len(test_acc_values) == 0:
            print(f"Skipping config with no test results: {config['save_name']}")
            continue

        stats = _compute_mean_std_ci(
            values=test_acc_values,
            confidence=confidence,
        )

        results.append(
            {
                "test_acc": stats["mean"],
                "test_acc_std": stats["std"],
                "test_acc_ci_low": stats["ci_low"],
                "test_acc_ci_high": stats["ci_high"],
                "test_acc_ci_half_width": stats["ci_half_width"],
                "confidence": stats["confidence"],
                "n_seeds": stats["n"],
                "seed_results": seed_results,
                "save_name": config["save_name"],
                "best_val_acc": config["best_val_acc"],
                "lr": config["ref_lr"],
                "wd": config["ref_wd"],
                "prompt_mode": config["prompt_mode"],
                "prompt_length": config["prompt_length"],
            }
        )

    json_path = base_dir / f"{suffix}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved JSON results to: {json_path}")


def _format_result(entry: Dict) -> str:
    if entry["test_acc_ci_low"] is None or entry["test_acc_ci_high"] is None:
        return f"{entry['test_acc']:.4f}"

    return (
        f"{entry['test_acc']:.4f} "
        f"[{entry['test_acc_ci_low']:.4f}, {entry['test_acc_ci_high']:.4f}]"
    )


def aggregate_results(backbone_name: str) -> None:
    results_dir = Path(__file__).resolve().parent / "top1_acc_ft" / backbone_name
    suffix = "eval"

    rows = {}

    for dataset_dir in sorted(results_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        json_path = dataset_dir / f"{suffix}.json"

        if not json_path.exists():
            print(f"Skipping missing aggregate result: {json_path}")
            continue

        dataset_results = _load_json(json_path)

        for entry in dataset_results:
            if not entry["save_name"].startswith("best_"):
                continue

            row_name = entry["prompt_mode"]

            if row_name not in rows:
                rows[row_name] = {}

            rows[row_name][dataset_name] = _format_result(entry)

    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df.reindex(sorted(df.columns), axis=1)

    csv_path = results_dir / f"{suffix}.csv"
    df.to_csv(csv_path)

    print(f"Saved CSV results to: {csv_path}")
