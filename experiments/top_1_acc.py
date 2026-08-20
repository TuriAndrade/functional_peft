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
FGVC = [
    "fgvc_cub200",
    "fgvc_nabirds",
    "fgvc_oxford_flowers",
    "fgvc_stanford_dogs",
    "fgvc_stanford_cars",
]
DATASET_GROUPS = {
    "vtab_natural": VTAB_NATURAL,
    "vtab_specialized": VTAB_SPECIALIZED,
    "vtab_structured": VTAB_STRUCTURED,
    "vtab": VTAB_NATURAL + VTAB_SPECIALIZED + VTAB_STRUCTURED,
    "fgvc": FGVC,
    "all": VTAB_NATURAL + VTAB_SPECIALIZED + VTAB_STRUCTURED + FGVC + ["imagenet_val"],
}

TUNING_SEED = 0
COMMON_ARGS = {
    "batch_size": 64,
    "gpu_batch_size": 64,
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

LR_SEARCH = [1e-3, 1e-2]
WD_SEARCH = [1e-4, 1e-3]
VPT_PROMPT_SEARCH = {
    "shallow": [5, 10, 50, 100, 200],
    "deep": [5, 10, 50, 100],
}
LORA_RANK_SEARCH = [1, 8, 16, 32]
ADAPTFORMER_BOTTLENECK_DIM = [4, 16, 32]
ADAPTFORMER_SCALE = [0.05, 0.1, 0.2]


def linear_probe_args(lr, wd, **backbone_args):
    return {
        **COMMON_ARGS,
        "peft_name": "vpt",  # Any method works with peft deactivated
        "freeze_backbone": True,
        "peft_config": {
            **backbone_args,
            "prompt_length": 0,  # Deactivates VPT
        },
        "ref_lr": lr,
        "ref_wd": wd,
        "final_wd": wd,
    }


def full_ft_args(lr, wd, **backbone_args):
    return {
        **COMMON_ARGS,
        "peft_name": "vpt",  # Any method works with peft deactivated
        "freeze_backbone": False,
        "peft_config": {
            **backbone_args,
            "prompt_length": 0,  # Deactivates VPT
        },
        "ref_lr": lr,
        "ref_wd": wd,
        "final_wd": wd,
    }


def vpt_args(lr, wd, prompt_length, prompt_mode, **backbone_args):
    return {
        **COMMON_ARGS,
        "peft_name": "vpt",
        "freeze_backbone": True,
        "peft_config": {
            **backbone_args,
            "prompt_length": prompt_length,
            "prompt_mode": prompt_mode,
            "prompt_dropout": 0.0 if prompt_mode == "shallow" else 0.1,
        },
        "ref_lr": lr,
        "ref_wd": wd,
        "final_wd": wd,
    }


def lora_args(lr, wd, rank, **backbone_args):
    return {
        **COMMON_ARGS,
        "peft_name": "lora",
        "freeze_backbone": True,
        "peft_config": {
            **backbone_args,
            "rank": rank,
            "lora_alpha": None,
            "lora_targets": ("query", "value"),
        },
        "ref_lr": lr,
        "ref_wd": wd,
        "final_wd": wd,
    }


def adaptformer_args(lr, wd, bottleneck_dim, scale, **backbone_args):
    return {
        **COMMON_ARGS,
        "peft_name": "adaptformer",
        "freeze_backbone": True,
        "peft_config": {
            **backbone_args,
            "bottleneck_dim": bottleneck_dim,
            "adapter_scale": scale,
            "adapter_dropout": 0.0,
        },
        "ref_lr": lr,
        "ref_wd": wd,
        "final_wd": wd,
    }


def ssf_args(lr, wd, **backbone_args):
    return {
        **COMMON_ARGS,
        "peft_name": "ssf",
        "freeze_backbone": True,
        "peft_config": {
            **backbone_args,
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
    }
