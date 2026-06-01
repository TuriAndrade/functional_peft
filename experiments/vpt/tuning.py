import os
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple

_SUFFIX = "tuning"
_TUNING_SEED = 0

_COMMON_ARGS = {
    "batch_size": 64,
    "gpu_batch_size": 64,
    "epochs": 100,
    "early_stopping": False,
    "start_lr": 1e-6,
    "final_lr": 1e-5,
    "warmup_epochs": 10,
    "save_ckpt": False,
    "run_seeds": [_TUNING_SEED],
    "eval_on_test": False,
    "bootstrap": False,
}

_LR_SEARCH = [1e-3, 1e-2]
_WD_SEARCH = [1e-4, 1e-3]

_PL_SEARCH = {
    "shallow": [5, 10, 50, 100, 200],
    "deep": [5, 10, 50, 100],
}

_PM = list(_PL_SEARCH.keys())


def build_args(
    dataset_name: str,
    backbone_name: str,
) -> Tuple[List[Dict], str, str]:
    _BASE_DIR = os.path.join(
        Path(__file__).resolve().parent,
        "top1_acc_ft",
        backbone_name,
        dataset_name,
    )

    ARGS = [
        {
            **_COMMON_ARGS,
            "dataset_name": dataset_name,
            "backbone_name": backbone_name,
            "save_dir": os.path.join(_BASE_DIR, _SUFFIX),
            "save_name": f"pm-{pm}_pl-{pl}_lr-{lr}_wd-{wd}",
            "prompt_mode": pm,
            "prompt_length": pl,
            "ref_lr": lr,
            "ref_wd": wd,
            "final_wd": wd,
        }
        for wd in _WD_SEARCH
        for lr in _LR_SEARCH
        for pm in _PM
        for pl in _PL_SEARCH[pm]
    ]

    OUTPUT_PATH = os.path.join(_BASE_DIR, f"{_SUFFIX}.txt")
    EXE_PREFIX = "python vpt_train_exe.py"

    return ARGS, OUTPUT_PATH, EXE_PREFIX


def build_results(
    dataset_name: str,
    backbone_name: str,
    topk: int = 3,
) -> None:
    args, output_path, _ = build_args(
        dataset_name=dataset_name,
        backbone_name=backbone_name,
    )

    output_path = Path(output_path)
    parent_dir = output_path.parent
    suffix = output_path.stem

    results = {
        prompt_mode: {
            str(prompt_length): [] for prompt_length in _PL_SEARCH[prompt_mode]
        }
        for prompt_mode in _PM
    }

    for item in args:
        run_dir = Path(item["save_dir"]) / f"{item['save_name']}_{_TUNING_SEED}"
        val_path = run_dir / "val_acc_eval.json"

        if not val_path.exists():
            print(f"Skipping missing result: {run_dir}")
            continue

        with open(val_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)

        val_acc = val_data["accuracy"]["top1"]

        prompt_mode = item["prompt_mode"]
        prompt_length = str(item["prompt_length"])

        results[prompt_mode][prompt_length].append(
            {
                "val_acc": val_acc,
                "save_name": item["save_name"],
                "lr": item["ref_lr"],
                "wd": item["ref_wd"],
            }
        )

    for prompt_mode in _PM:
        for prompt_length in _PL_SEARCH[prompt_mode]:
            prompt_length_key = str(prompt_length)

            results[prompt_mode][prompt_length_key] = sorted(
                results[prompt_mode][prompt_length_key],
                key=lambda x: x["val_acc"],
                reverse=True,
            )[:topk]

    json_path = parent_dir / f"{suffix}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    def get_top1_metric(prompt_mode: str, prompt_length: int, metric: str):
        entries = results[prompt_mode][str(prompt_length)]
        if len(entries) == 0:
            return None
        return entries[0][metric]

    fig, ax = plt.subplots(figsize=(8, 5))

    for prompt_mode in _PM:
        x_values = _PL_SEARCH[prompt_mode]
        val_y_values = [
            get_top1_metric(prompt_mode, prompt_length, "val_acc")
            for prompt_length in x_values
        ]

        ax.plot(x_values, val_y_values, marker="o", label=prompt_mode)

    ax.set_xlabel("Prompt length")
    ax.set_ylabel("Accuracy")
    ax.set_title("Top-1 validation accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = parent_dir / f"{suffix}.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"Saved JSON results to: {json_path}")
    print(f"Saved plots to: {plot_path}")
