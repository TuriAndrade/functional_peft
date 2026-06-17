import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

_SUFFIX = "tuning"
_TUNING_SEED = 0

_COMMON_ARGS = {
    "freeze_backbone": False,
    "batch_size": 64,
    "gpu_batch_size": 64,
    "epochs": 100,
    "early_stopping": False,
    "start_lr": 1e-6,
    "final_lr": 1e-5,
    "warmup_epochs": 5,
    "save_ckpt": False,
    "run_seeds": [_TUNING_SEED],
    "eval_on_test": False,
    "bootstrap": False,
    "prompt_length": 0,
}

_LR_SEARCH = [1e-5, 3e-5, 1e-4]
_WD_SEARCH = [0.0, 1e-2]


def build_args(
    dataset_name: str,
    backbone_name: str,
) -> Tuple[List[Dict], str, str]:
    base_dir = os.path.join(
        Path(__file__).resolve().parent,
        "top1_acc_ft",
        backbone_name,
        dataset_name,
    )

    args = [
        {
            **_COMMON_ARGS,
            "dataset_name": dataset_name,
            "backbone_name": backbone_name,
            "save_dir": os.path.join(base_dir, _SUFFIX),
            "save_name": f"lr-{lr}_wd-{wd}",
            "ref_lr": lr,
            "ref_wd": wd,
            "final_wd": wd,
        }
        for wd in _WD_SEARCH
        for lr in _LR_SEARCH
    ]

    output_path = os.path.join(base_dir, f"{_SUFFIX}.txt")
    exe_prefix = "python vpt_train_exe.py"

    return args, output_path, exe_prefix


def build_results(
    dataset_name: str,
    backbone_name: str,
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
        run_dir = Path(item["save_dir"]) / f"{item['save_name']}_{_TUNING_SEED}"
        val_path = run_dir / "val_acc_eval.json"

        if not val_path.exists():
            print(f"Skipping missing result: {run_dir}")
            continue

        with val_path.open("r", encoding="utf-8") as f:
            val_data = json.load(f)

        results.append(
            {
                "val_acc": val_data["accuracy"]["top1"],
                "save_name": item["save_name"],
                "lr": item["ref_lr"],
                "wd": item["ref_wd"],
            }
        )

    results.sort(key=lambda entry: entry["val_acc"], reverse=True)
    json_path = parent_dir / f"{suffix}.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved JSON results to: {json_path}")
