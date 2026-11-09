import csv
import json
import math
import os
import shlex
import statistics


# ======================================================================
# DATASETS
# ======================================================================

DATASETS = [
    "cifar100",
    "dtd",
    "eurosat",
    "food101",
    "gtsrb",
    "stanford_cars",
    "svhn",
]

DATASET_GROUPS = {
    "all": DATASETS,
}


# ======================================================================
# BACKBONES
# ======================================================================

BACKBONES = ["vit_b16"]

BACKBONE_GROUPS = {
    "all": BACKBONES,
}


# ======================================================================
# TUNING ARGS
# ======================================================================

TUNING_SEED = 0

TUNING_ARGS = {
    "batch_size": 64,
    "gpu_batch_size": 16,
    "epochs": 100,
    "early_stopping": True,
    "patience": 1,
    "start_lr": 1e-6,
    "final_lr": 1e-5,
    "warmup_epochs": 10,
    "save_ckpt": False,
    "run_seeds": [TUNING_SEED],
    "eval_on_test": False,
    "bootstrap": False,
}


# ======================================================================
# TUNING SEARCH GRIDS
# ======================================================================

PEFT_LR_SEARCH = [1e-3, 1e-2]

PEFT_WD_SEARCH = [1e-4]

VPT_SHALLOW_PROMPT_SEARCH = [10, 50, 100]

VPT_DEEP_PROMPT_SEARCH = [5, 10, 50]

LORA_RANK_SEARCH = [8, 16, 32]

ADAPTFORMER_BOTTLENECK_DIM_SEARCH = [4, 16, 32]

ADAPTFORMER_SCALE_SEARCH = [0.1, 0.2]


def linear_probe_args(lr, wd):
    return {
        **TUNING_ARGS,
        "peft_name": "vpt",
        "peft_config": {
            "prompt_length": 0,
        },
        "ref_lr": lr,
        "ref_wd": wd,
        "final_wd": wd,
        "save_name": f"lr-{lr}_wd-{wd}",
    }


def linear_probe_search():
    return [linear_probe_args(lr, wd) for lr in PEFT_LR_SEARCH for wd in PEFT_WD_SEARCH]


def vpt_args(lr, wd, prompt_length, prompt_mode):
    return {
        **TUNING_ARGS,
        "peft_name": "vpt",
        "peft_config": {
            "prompt_length": prompt_length,
            "prompt_mode": prompt_mode,
            "prompt_dropout": 0.0 if prompt_mode == "shallow" else 0.1,
        },
        "ref_lr": lr,
        "ref_wd": wd,
        "final_wd": wd,
        "save_name": f"lr-{lr}_wd-{wd}_pl-{prompt_length}",
    }


def vpt_shallow_search():
    return [
        vpt_args(lr, wd, pl, "shallow")
        for lr in PEFT_LR_SEARCH
        for wd in PEFT_WD_SEARCH
        for pl in VPT_SHALLOW_PROMPT_SEARCH
    ]


def vpt_deep_search():
    return [
        vpt_args(lr, wd, pl, "deep")
        for lr in PEFT_LR_SEARCH
        for wd in PEFT_WD_SEARCH
        for pl in VPT_DEEP_PROMPT_SEARCH
    ]


def lora_args(lr, wd, rank):
    return {
        **TUNING_ARGS,
        "peft_name": "lora",
        "peft_config": {
            "rank": rank,
            "lora_alpha": None,
            "lora_targets": ("query", "value"),
        },
        "ref_lr": lr,
        "ref_wd": wd,
        "final_wd": wd,
        "save_name": f"lr-{lr}_wd-{wd}_rank-{rank}",
    }


def lora_search():
    return [
        lora_args(lr, wd, rank)
        for lr in PEFT_LR_SEARCH
        for wd in PEFT_WD_SEARCH
        for rank in LORA_RANK_SEARCH
    ]


def adaptformer_args(lr, wd, bottleneck_dim, scale):
    return {
        **TUNING_ARGS,
        "peft_name": "adaptformer",
        "peft_config": {
            "bottleneck_dim": bottleneck_dim,
            "adapter_scale": scale,
            "adapter_dropout": 0.0,
        },
        "ref_lr": lr,
        "ref_wd": wd,
        "final_wd": wd,
        "save_name": f"lr-{lr}_wd-{wd}_bndim-{bottleneck_dim}_scale-{scale}",
    }


def adaptformer_search():
    return [
        adaptformer_args(lr, wd, dim, scale)
        for lr in PEFT_LR_SEARCH
        for wd in PEFT_WD_SEARCH
        for dim in ADAPTFORMER_BOTTLENECK_DIM_SEARCH
        for scale in ADAPTFORMER_SCALE_SEARCH
    ]


def ssf_args(lr, wd):
    return {
        **TUNING_ARGS,
        "peft_name": "ssf",
        "peft_config": {
            "ssf_targets": (
                "patch",
                "attention_norm",
                "query",
                "key",
                "value",
                "attention_out",
                "ffn_norm",
                "fc1",
                "fc2",
                "final_norm",
            ),
        },
        "ref_lr": lr,
        "ref_wd": wd,
        "final_wd": wd,
        "save_name": f"lr-{lr}_wd-{wd}",
    }


def ssf_search():
    return [ssf_args(lr, wd) for lr in PEFT_LR_SEARCH for wd in PEFT_WD_SEARCH]


PEFT_METHODS = {
    "vpt_shallow": vpt_shallow_search,
    "vpt_deep": vpt_deep_search,
    "lora": lora_search,
    "ssf": ssf_search,
    "adaptformer": adaptformer_search,
}

FINETUNING_METHODS = {
    "linear_probe": linear_probe_search,
    **PEFT_METHODS,
}

FINETUNING_METHOD_GROUPS = {
    "peft": PEFT_METHODS,
    "all": FINETUNING_METHODS,
}


# ======================================================================
# HELPERS
# ======================================================================


def _resolve_names(name_or_group, groups, names, label):
    if name_or_group in groups:
        return list(groups[name_or_group])
    if name_or_group in names:
        return [name_or_group]
    raise ValueError(f"Unknown {label} or group: {name_or_group}")


def _resolve_methods(name_or_group):
    if name_or_group in FINETUNING_METHOD_GROUPS:
        return FINETUNING_METHOD_GROUPS[name_or_group]
    if name_or_group in FINETUNING_METHODS:
        return {name_or_group: FINETUNING_METHODS[name_or_group]}
    raise ValueError(f"Unknown fine-tuning method or group: {name_or_group}")


def _write_commands(commands, output_name, max_runs_per_file):
    if max_runs_per_file < 0:
        raise ValueError("max_runs_per_file must be non-negative")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    if max_runs_per_file == 0:
        chunks = [commands]
    else:
        chunks = [
            commands[i : i + max_runs_per_file]
            for i in range(0, len(commands), max_runs_per_file)
        ]
    for i, chunk in enumerate(chunks):
        filename = (
            f"{output_name}.txt" if max_runs_per_file == 0 else f"{output_name}_{i}.txt"
        )
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(chunk) + "\n")


# ======================================================================
# BUILD TUNING RUNS
# ======================================================================


def build_tuning_runs(
    dataset_name_or_group: str = "all",
    backbone_name_or_group: str = "all",
    ft_name_or_group: str = "all",
    output_name: str = "top_1_acc_tuning",
    max_runs_per_file: int = 0,
):
    dataset_names = _resolve_names(
        dataset_name_or_group, DATASET_GROUPS, DATASETS, "dataset"
    )
    backbone_names = _resolve_names(
        backbone_name_or_group, BACKBONE_GROUPS, BACKBONES, "backbone"
    )
    methods = _resolve_methods(ft_name_or_group)
    experiments_dir = os.environ.get("EXPERIMENTS_DIR", "./experiments")
    commands = []
    for backbone_name in backbone_names:
        for dataset_name in dataset_names:
            for ft_name, search_fn in methods.items():
                save_dir = os.path.join(
                    experiments_dir,
                    "top_1_acc_tuning",
                    backbone_name,
                    dataset_name,
                    ft_name,
                )
                for tuning_args in search_fn():
                    args = {
                        "dataset_name": dataset_name,
                        "backbone_name": backbone_name,
                        **tuning_args,
                        "save_dir": save_dir,
                    }
                    args_json = json.dumps(args)
                    commands.append(
                        f"python -m scripts.train_peft_model --args-json {shlex.quote(args_json)}"
                    )
    _write_commands(commands, output_name, max_runs_per_file)


# ======================================================================
# EVAL ARGS
# ======================================================================

EVAL_SEEDS = [1, 2]

EVAL_ARGS = {
    "batch_size": 64,
    "gpu_batch_size": 16,
    "epochs": 100,
    "early_stopping": True,
    "patience": 3,
    "start_lr": 1e-6,
    "final_lr": 1e-5,
    "warmup_epochs": 10,
    "save_ckpt": True,
    "run_seeds": EVAL_SEEDS,
    "eval_on_test": True,
    "bootstrap": False,
}

EVAL_CONFIG_KEYS = (
    "peft_name",
    "peft_config",
    "ref_lr",
    "ref_wd",
    "final_wd",
    "save_name",
)


# ======================================================================
# BUILD EVAL RUNS
# ======================================================================


def _best_tuning_config(tuning_dir: str):
    best_acc = None
    best_config = None
    for run_name in sorted(os.listdir(tuning_dir)):
        run_dir = os.path.join(tuning_dir, run_name)
        if not os.path.isdir(run_dir):
            continue
        eval_path = os.path.join(run_dir, "val_acc_eval.json")
        config_path = os.path.join(run_dir, "config.json")
        if not all(os.path.isfile(path) for path in (eval_path, config_path)):
            continue
        with open(eval_path, "r", encoding="utf-8") as f:
            acc = json.load(f)["accuracy"]["top1"]
        if best_acc is None or acc > best_acc:
            with open(config_path, "r", encoding="utf-8") as f:
                best_config = json.load(f)
            best_acc = acc
    if best_config is None:
        raise FileNotFoundError(f"No completed tuning runs found in: {tuning_dir}")
    return best_config


def build_eval_runs(
    dataset_name_or_group: str = "all",
    backbone_name_or_group: str = "all",
    ft_name_or_group: str = "all",
    output_name: str = "top_1_acc_eval",
    max_runs_per_file: int = 0,
):
    dataset_names = _resolve_names(
        dataset_name_or_group, DATASET_GROUPS, DATASETS, "dataset"
    )
    backbone_names = _resolve_names(
        backbone_name_or_group, BACKBONE_GROUPS, BACKBONES, "backbone"
    )
    methods = _resolve_methods(ft_name_or_group)
    experiments_dir = os.environ.get("EXPERIMENTS_DIR", "./experiments")
    commands = []
    for backbone_name in backbone_names:
        for dataset_name in dataset_names:
            for ft_name in methods:
                tuning_dir = os.path.join(
                    experiments_dir,
                    "top_1_acc_tuning",
                    backbone_name,
                    dataset_name,
                    ft_name,
                )
                best = _best_tuning_config(tuning_dir)
                save_dir = os.path.join(
                    experiments_dir,
                    "top_1_acc_eval",
                    backbone_name,
                    dataset_name,
                    ft_name,
                )
                args = {
                    "dataset_name": dataset_name,
                    "backbone_name": backbone_name,
                    **{key: best[key] for key in EVAL_CONFIG_KEYS if key in best},
                    **EVAL_ARGS,
                    "save_dir": save_dir,
                }
                args_json = json.dumps(args)
                commands.append(
                    f"python -m scripts.train_peft_model "
                    f"--args-json {shlex.quote(args_json)}"
                )
    _write_commands(commands, output_name, max_runs_per_file)


# ======================================================================
# AGGREGATE EVAL RESULTS
# ======================================================================


def aggregate_eval_results(
    dataset_name_or_group: str = "all",
    backbone_name_or_group: str = "all",
    ft_name_or_group: str = "all",
    output_name: str = "top_1_acc_eval_results",
    split: str = "test",
):
    if split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test'")
    dataset_names = _resolve_names(
        dataset_name_or_group, DATASET_GROUPS, DATASETS, "dataset"
    )
    backbone_names = _resolve_names(
        backbone_name_or_group, BACKBONE_GROUPS, BACKBONES, "backbone"
    )
    method_names = list(_resolve_methods(ft_name_or_group))
    experiments_dir = os.environ.get("EXPERIMENTS_DIR", "./experiments")
    output_dir = os.path.join(experiments_dir, "top_1_acc_eval")

    # Two-sided 95% Student-t critical value for n=3 (df=2).
    t_crit = 4.302652729911275

    results = {
        f"{backbone}_{method}": {}
        for backbone in backbone_names
        for method in method_names
    }
    for backbone_name in backbone_names:
        for dataset_name in dataset_names:
            for method_name in method_names:
                eval_dir = os.path.join(
                    output_dir,
                    backbone_name,
                    dataset_name,
                    method_name,
                )
                accuracies = {}
                if os.path.isdir(eval_dir):
                    for run_name in sorted(os.listdir(eval_dir)):
                        run_dir = os.path.join(eval_dir, run_name)
                        if not os.path.isdir(run_dir):
                            continue
                        seed = next(
                            (
                                seed
                                for seed in EVAL_SEEDS
                                if run_name.endswith(f"_seed-{seed}")
                            ),
                            None,
                        )
                        if seed is None:
                            continue
                        eval_path = os.path.join(run_dir, f"{split}_acc_eval.json")
                        if not os.path.isfile(eval_path):
                            continue
                        with open(eval_path, "r", encoding="utf-8") as f:
                            accuracies[seed] = json.load(f)["accuracy"]["top1"]
                if set(accuracies) != set(EVAL_SEEDS):
                    missing = sorted(set(EVAL_SEEDS) - set(accuracies))
                    raise FileNotFoundError(
                        f"Missing eval seeds {missing} for "
                        f"{backbone_name}/{dataset_name}/{method_name}"
                    )
                values = [accuracies[seed] for seed in EVAL_SEEDS]
                mean = statistics.mean(values)
                ci95 = t_crit * statistics.stdev(values) / math.sqrt(len(values))
                results[f"{backbone_name}_{method_name}"][dataset_name] = (
                    f"{mean:.2f} ± {ci95:.2f}"
                )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{output_name}.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", *dataset_names])
        for backbone_name in backbone_names:
            for method_name in method_names:
                key = f"{backbone_name}_{method_name}"
                writer.writerow(
                    [
                        key,
                        *[results[key][dataset] for dataset in dataset_names],
                    ]
                )
