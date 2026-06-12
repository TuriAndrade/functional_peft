import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from scipy.stats import t as student_t

from .tuning import _TUNING_SEED

_EVAL_SEEDS = [0, 1, 2]

_COMMON_ARGS = {
    "freeze_backbone": True,
    "save_ckpt": True,
    "save_ckpt_all_seeds": False,
    "early_stopping": False,
    "run_seeds": _EVAL_SEEDS,
    "eval_on_test": True,
    "bootstrap": False,
}


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_required(config: Dict, key: str):
    if key not in config:
        raise KeyError(f"Missing key '{key}' in config.json")
    return config[key]


def _load_best_tuning_config(base_dir: Path) -> Dict:
    tuning_suffix = "tuning"
    tuning_json_path = base_dir / f"{tuning_suffix}.json"
    tuning_dir = base_dir / tuning_suffix
    tuning_results = _load_json(tuning_json_path)

    if len(tuning_results) == 0:
        raise ValueError(f"No valid tuning entries found in: {tuning_json_path}")

    best_entry = max(tuning_results, key=lambda entry: entry["val_acc"])
    save_name = best_entry["save_name"]
    run_dir = tuning_dir / f"{save_name}_{_TUNING_SEED}"
    config = _load_json(run_dir / "config.json")
    train_summary = _load_json(run_dir / "train_summary.json")

    return {
        "save_name": save_name,
        "best_val_acc": best_entry["val_acc"],
        "best_epoch": _get_required(train_summary, "best_epoch"),
        "batch_size": _get_required(config, "batch_size"),
        "gpu_batch_size": _get_required(config, "gpu_batch_size"),
        "start_lr": _get_required(config, "start_lr"),
        "final_lr": _get_required(config, "final_lr"),
        "warmup_epochs": _get_required(config, "warmup_epochs"),
        "prompt_length": _get_required(config, "prompt_length"),
        "ref_lr": _get_required(config, "ref_lr"),
        "ref_wd": _get_required(config, "ref_wd"),
        "final_wd": _get_required(config, "final_wd"),
    }


def build_args(
    dataset_name: str,
    backbone_name: str,
) -> Tuple[List[Dict], str, str]:
    base_dir = (
        Path(__file__).resolve().parent / "top1_acc_ft" / backbone_name / dataset_name
    )
    suffix = "eval"
    best_config = _load_best_tuning_config(base_dir=base_dir)

    args = [
        {
            **_COMMON_ARGS,
            "dataset_name": dataset_name,
            "backbone_name": backbone_name,
            "save_dir": str(base_dir / suffix),
            "batch_size": best_config["batch_size"],
            "gpu_batch_size": best_config["gpu_batch_size"],
            "start_lr": best_config["start_lr"],
            "final_lr": best_config["final_lr"],
            "warmup_epochs": best_config["warmup_epochs"],
            "epochs": best_config["best_epoch"],
            "save_name": (f"lr-{best_config['ref_lr']}_wd-{best_config['ref_wd']}"),
            "prompt_length": 0,
            "ref_lr": best_config["ref_lr"],
            "ref_wd": best_config["ref_wd"],
            "final_wd": best_config["final_wd"],
        }
    ]

    output_path = str(base_dir / f"{suffix}.txt")
    exe_prefix = "python vpt_train_exe.py"

    return args, output_path, exe_prefix


def _compute_mean_std_ci(
    values: List[float],
    confidence: float = 0.95,
) -> Dict:
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
    critical_value = student_t.ppf(1.0 - alpha / 2.0, n - 1)
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
    args, output_path, _ = build_args(
        dataset_name=dataset_name,
        backbone_name=backbone_name,
    )

    output_path = Path(output_path)
    parent_dir = output_path.parent
    suffix = output_path.stem
    results = []

    for item in args:
        test_acc_values = []
        seed_results = []

        for seed in item["run_seeds"]:
            run_dir = Path(item["save_dir"]) / f"{item['save_name']}_{seed}"
            test_path = run_dir / "test_acc_eval.json"

            if not test_path.exists():
                print(f"Skipping missing result: {run_dir}")
                continue

            test_acc = _load_json(test_path)["accuracy"]["top1"]
            test_acc_values.append(test_acc)
            seed_results.append(
                {
                    "seed": seed,
                    "test_acc": test_acc,
                    "run_dir": str(run_dir),
                }
            )

        if len(test_acc_values) == 0:
            print(f"Skipping config with no test results: {item['save_name']}")
            continue

        stats = _compute_mean_std_ci(test_acc_values, confidence)
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
                "save_name": item["save_name"],
                "lr": item["ref_lr"],
                "wd": item["ref_wd"],
            }
        )

    json_path = parent_dir / f"{suffix}.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved JSON results to: {json_path}")


def aggregate_results(backbone_name: str) -> None:
    results_dir = Path(__file__).resolve().parent / "top1_acc_ft" / backbone_name
    suffix = "eval"
    rows = {"linear_probe": {}}

    for dataset_dir in sorted(results_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        json_path = dataset_dir / f"{suffix}.json"

        if not json_path.exists():
            print(f"Skipping missing aggregate result: {json_path}")
            continue

        dataset_results = _load_json(json_path)

        if len(dataset_results) == 0:
            print(f"Skipping empty aggregate result: {json_path}")
            continue

        entry = dataset_results[0]
        rows["linear_probe"][dataset_name] = (
            f"{entry['test_acc']:.4f} "
            f"[{entry['test_acc_ci_low']:.4f}, {entry['test_acc_ci_high']:.4f}]"
        )

    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df.reindex(sorted(df.columns), axis=1)
    csv_path = results_dir / f"{suffix}.csv"
    df.to_csv(csv_path)

    print(f"Saved CSV results to: {csv_path}")
