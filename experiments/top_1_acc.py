import json
import os
import shlex

# ======================================================================
# DATASETS
# ======================================================================
VTAB_NATURAL = [
    "vtab_caltech101",
    "vtab_cifar100",
    "vtab_dtd",
    "vtab_flowers102",
    "vtab_pets",
    "vtab_sun397",
    "vtab_svhn",
]
VTAB_SPECIALIZED = [
    "vtab_eurosat",
    "vtab_resisc45",
    "vtab_patch_camelyon",
    "vtab_retinopathy",
]
VTAB_STRUCTURED = [
    "vtab_clevr_count",
    "vtab_clevr_distance",
    "vtab_dmlab",
    "vtab_kitti_distance",
    "vtab_dsprites_location",
    "vtab_dsprites_orientation",
    "vtab_smallnorb_azimuth",
    "vtab_smallnorb_elevation",
]
VTAB = VTAB_NATURAL + VTAB_SPECIALIZED + VTAB_STRUCTURED

FGVC = [
    "fgvc_cub200",
    "fgvc_nabirds",
    "fgvc_oxford_flowers",
    "fgvc_stanford_dogs",
    "fgvc_stanford_cars",
]


DATASETS = VTAB + FGVC + ["imagenet_val"]
DATASET_GROUPS = {
    "vtab_natural": VTAB_NATURAL,
    "vtab_specialized": VTAB_SPECIALIZED,
    "vtab_structured": VTAB_STRUCTURED,
    "vtab": VTAB,
    "vtab+imagenet_val": VTAB + ["imagenet_val"],
    "fgvc": FGVC,
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
    "gpu_batch_size": 8,
    "epochs": 100,
    "early_stopping": True,
    "patience": 10,
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
LR_SEARCH = [1e-3, 1e-2]
WD_SEARCH = [1e-4, 1e-3]
VPT_SHALLOW_PROMPT_SEARCH = [5, 10, 50, 100, 200]
VPT_DEEP_PROMPT_SEARCH = [5, 10, 50, 100]
LORA_RANK_SEARCH = [1, 8, 16, 32]
ADAPTFORMER_BOTTLENECK_DIM_SEARCH = [4, 16, 32]
ADAPTFORMER_SCALE_SEARCH = [0.05, 0.1, 0.2]


def linear_probe_args(lr, wd):
    return {
        **TUNING_ARGS,
        "peft_name": "vpt",  # Any method works with peft deactivated
        "freeze_backbone": True,
        "peft_config": {
            "prompt_length": 0,  # Deactivates VPT
        },
        "ref_lr": lr,
        "ref_wd": wd,
        "final_wd": wd,
        "save_name": f"lr-{lr}_wd-{wd}",
    }


def linear_probe_search():
    return [linear_probe_args(lr, wd) for lr in LR_SEARCH for wd in WD_SEARCH]


def full_ft_args(lr, wd):
    return {
        **TUNING_ARGS,
        "peft_name": "vpt",  # Any method works with peft deactivated
        "freeze_backbone": False,
        "peft_config": {
            "prompt_length": 0,  # Deactivates VPT
        },
        "ref_lr": lr,
        "ref_wd": wd,
        "final_wd": wd,
        "save_name": f"lr-{lr}_wd-{wd}",
    }


def full_ft_search():
    return [full_ft_args(lr, wd) for lr in LR_SEARCH for wd in WD_SEARCH]


def vpt_args(lr, wd, prompt_length, prompt_mode):
    return {
        **TUNING_ARGS,
        "peft_name": "vpt",
        "freeze_backbone": True,
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
        for lr in LR_SEARCH
        for wd in WD_SEARCH
        for pl in VPT_SHALLOW_PROMPT_SEARCH
    ]


def vpt_deep_search():
    return [
        vpt_args(lr, wd, pl, "deep")
        for lr in LR_SEARCH
        for wd in WD_SEARCH
        for pl in VPT_DEEP_PROMPT_SEARCH
    ]


def lora_args(lr, wd, rank):
    return {
        **TUNING_ARGS,
        "peft_name": "lora",
        "freeze_backbone": True,
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
        for lr in LR_SEARCH
        for wd in WD_SEARCH
        for rank in LORA_RANK_SEARCH
    ]


def adaptformer_args(lr, wd, bottleneck_dim, scale):
    return {
        **TUNING_ARGS,
        "peft_name": "adaptformer",
        "freeze_backbone": True,
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
        adaptformer_args(lr, wd, dim, s)
        for lr in LR_SEARCH
        for wd in WD_SEARCH
        for dim in ADAPTFORMER_BOTTLENECK_DIM_SEARCH
        for s in ADAPTFORMER_SCALE_SEARCH
    ]


def ssf_args(lr, wd):
    return {
        **TUNING_ARGS,
        "peft_name": "ssf",
        "freeze_backbone": True,
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
    return [ssf_args(lr, wd) for lr in LR_SEARCH for wd in WD_SEARCH]


PEFT_METHODS = {
    "vpt_shallow": vpt_shallow_search,
    "vpt_deep": vpt_deep_search,
    "lora": lora_search,
    "ssf": ssf_search,
    "adaptformer": adaptformer_search,
}
FINETUNING_METHODS = {
    "linear_probe": linear_probe_search,
    "full_ft": full_ft_search,
    **PEFT_METHODS,
}
FINETUNING_METHOD_GROUPS = {
    "peft": PEFT_METHODS,
    "all": FINETUNING_METHODS,
}


# ======================================================================
# BUILD RUNS
# ======================================================================
def build_tuning_runs(
    dataset_name_or_group: str,
    backbone_name_or_group: str,
    ft_name_or_group: str,
    output_name: str = "top_1_acc_tuning",
    max_runs_per_file: int = 0,
):
    if max_runs_per_file < 0:
        raise ValueError("max_runs_per_file must be non-negative")

    if dataset_name_or_group in DATASET_GROUPS:
        dataset_names = DATASET_GROUPS[dataset_name_or_group]
    elif dataset_name_or_group in DATASETS:
        dataset_names = [dataset_name_or_group]
    else:
        raise ValueError(f"Unknown dataset or group: {dataset_name_or_group}")

    if backbone_name_or_group in BACKBONE_GROUPS:
        backbone_names = BACKBONE_GROUPS[backbone_name_or_group]
    elif backbone_name_or_group in BACKBONES:
        backbone_names = [backbone_name_or_group]
    else:
        raise ValueError(f"Unknown backbone or group: {backbone_name_or_group}")

    if ft_name_or_group in FINETUNING_METHOD_GROUPS:
        methods = FINETUNING_METHOD_GROUPS[ft_name_or_group]
    elif ft_name_or_group in FINETUNING_METHODS:
        methods = {ft_name_or_group: FINETUNING_METHODS[ft_name_or_group]}
    else:
        raise ValueError(f"Unknown fine-tuning method or group: {ft_name_or_group}")

    experiments_dir = os.environ.get("EXPERIMENTS_DIR", "./experiments")
    output_dir = os.path.dirname(os.path.abspath(__file__))

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
                        f"python peft_train_exe.py --args-json {shlex.quote(args_json)}"
                    )

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
