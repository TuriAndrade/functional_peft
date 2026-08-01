import json
from pathlib import Path
from typing import Dict, List, Tuple

_RUN_SEED = 0

_DATASET_NAMES = [
    "imagenet_val",
    "vtab_caltech101",
    "vtab_cifar100",
    "vtab_dtd",
    "vtab_flowers102",
    "vtab_pets",
    "vtab_sun397",
    "vtab_svhn",
    "vtab_eurosat",
    "vtab_resisc45",
    "vtab_patch_camelyon",
    "vtab_retinopathy",
    "vtab_clevr_count",
    "vtab_clevr_distance",
    "vtab_dmlab",
    "vtab_dsprites_location",
    "vtab_dsprites_orientation",
    "vtab_kitti_distance",
    "vtab_smallnorb_azimuth",
    "vtab_smallnorb_elevation",
]


_NATURAL = [
    "vtab_caltech101",
    "vtab_cifar100",
    "vtab_dtd",
    "vtab_flowers102",
    "vtab_pets",
    "vtab_sun397",
    "vtab_svhn",
]

_SPECIALIZED = [
    "vtab_eurosat",
    "vtab_resisc45",
    "vtab_patch_camelyon",
    "vtab_retinopathy",
]

_STRUCTURED = [
    "vtab_clevr_count",
    "vtab_clevr_distance",
    "vtab_dmlab",
    "vtab_dsprites_location",
    "vtab_dsprites_orientation",
    "vtab_kitti_distance",
    "vtab_smallnorb_azimuth",
    "vtab_smallnorb_elevation",
]


_DATASET_GROUPS = {
    # =========================================================
    # 1. Official VTAB taxonomy
    # =========================================================
    "vtab_natural": (
        _NATURAL,
        _SPECIALIZED + _STRUCTURED,
    ),
    "vtab_specialized": (
        _SPECIALIZED,
        _NATURAL + _STRUCTURED,
    ),
    "vtab_structured": (
        _STRUCTURED,
        _NATURAL + _SPECIALIZED,
    ),
    # =========================================================
    # 2. Official Specialized subdomains
    # =========================================================
    "domain_remote_sensing": (
        [
            "vtab_eurosat",
            "vtab_resisc45",
        ],
        [
            "vtab_patch_camelyon",
            "vtab_retinopathy",
        ],
    ),
    "domain_medical": (
        [
            "vtab_patch_camelyon",
            "vtab_retinopathy",
        ],
        [
            "vtab_eurosat",
            "vtab_resisc45",
        ],
    ),
    # =========================================================
    # 3. Shared input provenance
    # =========================================================
    "source_clevr": (
        [
            "vtab_clevr_count",
            "vtab_clevr_distance",
        ],
        [
            "vtab_dmlab",
            "vtab_dsprites_location",
            "vtab_dsprites_orientation",
            "vtab_kitti_distance",
            "vtab_smallnorb_azimuth",
            "vtab_smallnorb_elevation",
        ],
    ),
    "source_dsprites": (
        [
            "vtab_dsprites_location",
            "vtab_dsprites_orientation",
        ],
        [
            "vtab_clevr_count",
            "vtab_clevr_distance",
            "vtab_dmlab",
            "vtab_kitti_distance",
            "vtab_smallnorb_azimuth",
            "vtab_smallnorb_elevation",
        ],
    ),
    "source_smallnorb": (
        [
            "vtab_smallnorb_azimuth",
            "vtab_smallnorb_elevation",
        ],
        [
            "vtab_clevr_count",
            "vtab_clevr_distance",
            "vtab_dmlab",
            "vtab_dsprites_location",
            "vtab_dsprites_orientation",
            "vtab_kitti_distance",
        ],
    ),
    # =========================================================
    # 4. Prediction-task relations
    # =========================================================
    "task_entity_category_recognition": (
        [
            "vtab_caltech101",
            "vtab_cifar100",
            "vtab_flowers102",
            "vtab_pets",
        ],
        [
            "vtab_dtd",
            "vtab_sun397",
            "vtab_svhn",
        ],
    ),
    "task_fine_grained_recognition": (
        [
            "vtab_flowers102",
            "vtab_pets",
        ],
        [
            "vtab_caltech101",
            "vtab_cifar100",
            "vtab_dtd",
            "vtab_sun397",
            "vtab_svhn",
        ],
    ),
    "task_distance_related": (
        [
            "vtab_clevr_distance",
            "vtab_dmlab",
            "vtab_kitti_distance",
        ],
        [
            "vtab_clevr_count",
            "vtab_dsprites_location",
            "vtab_dsprites_orientation",
            "vtab_smallnorb_azimuth",
            "vtab_smallnorb_elevation",
        ],
    ),
    "task_angular_prediction": (
        [
            "vtab_dsprites_orientation",
            "vtab_smallnorb_azimuth",
            "vtab_smallnorb_elevation",
        ],
        [
            "vtab_clevr_count",
            "vtab_clevr_distance",
            "vtab_dmlab",
            "vtab_dsprites_location",
            "vtab_kitti_distance",
        ],
    ),
    # =========================================================
    # 5. External reference
    # =========================================================
    "pretraining_domain_alignment": (
        ["imagenet_val"],
        _NATURAL,
        _SPECIALIZED + _STRUCTURED,
    ),
}


_VPT_CONFIGS = [
    ("vpt_deep_5", "deep", 5),
    ("vpt_deep_10", "deep", 10),
    ("vpt_deep_50", "deep", 50),
    ("vpt_deep_100", "deep", 100),
    ("vpt_shallow_5", "shallow", 5),
    ("vpt_shallow_10", "shallow", 10),
    ("vpt_shallow_50", "shallow", 50),
    ("vpt_shallow_100", "shallow", 100),
    ("vpt_shallow_200", "shallow", 200),
]


_DEEP_MODELS = [name for name, prompt_mode, _ in _VPT_CONFIGS if prompt_mode == "deep"]

_SHALLOW_MODELS = [
    name for name, prompt_mode, _ in _VPT_CONFIGS if prompt_mode == "shallow"
]

_MODEL_GROUPS = {
    "vpt_deep": (_DEEP_MODELS,),
    "vpt_shallow": (_SHALLOW_MODELS,),
}


_F_SIG_KEYS = [
    "residual_branch",
    "attention_head",
    "backbone_attention_transport",
    "patch_spatial_transport",
    "mlp_neuron",
    "mlp_token_output",
]


def _find_signature(
    eval_dir: Path,
    prompt_mode: str = None,
    prompt_length: int = None,
) -> str:
    matches = []

    for run_dir in sorted(eval_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.endswith(f"_{_RUN_SEED}"):
            continue

        signature_path = run_dir / "functional_signatures.npz"
        if not signature_path.exists():
            continue

        if prompt_mode is not None:
            with open(run_dir / "config.json") as f:
                config = json.load(f)

            if (
                config["prompt_mode"] != prompt_mode
                or config["prompt_length"] != prompt_length
            ):
                continue

        matches.append(signature_path)

    if len(matches) != 1:
        raise ValueError(
            f"Expected one signature in {eval_dir}, found {len(matches)} "
            f"for prompt_mode={prompt_mode}, prompt_length={prompt_length}."
        )

    return str(matches[0])


def _find_best_signature(
    eval_dir: Path,
    prompt_mode: str,
) -> str:
    """
    Find the seed-specific Fisher signature for the VPT configuration
    with the highest validation accuracy for one prompt mode.
    """
    eval_path = eval_dir.parent / "eval.json"

    with open(eval_path) as f:
        results = json.load(f)

    candidates = [result for result in results if result["prompt_mode"] == prompt_mode]

    if not candidates:
        raise ValueError(f"No results for prompt_mode={prompt_mode} in {eval_path}.")

    best = max(
        candidates,
        key=lambda result: result["best_val_acc"],
    )

    seed_results = [
        result for result in best["seed_results"] if result["seed"] == _RUN_SEED
    ]

    if len(seed_results) != 1:
        raise ValueError(
            f"Expected one seed {_RUN_SEED} run for best "
            f"prompt_mode={prompt_mode} in {eval_path}, "
            f"found {len(seed_results)}."
        )

    signature_path = Path(seed_results[0]["run_dir"]) / "functional_signatures.npz"

    if not signature_path.exists():
        raise FileNotFoundError(signature_path)

    return str(signature_path)


def build_args(
    backbone_name: str,
) -> Tuple[List[Dict], str, str]:
    experiments_dir = Path(__file__).resolve().parent.parent

    model_names = (
        ["linear_probe"]
        + [name for name, _, _ in _VPT_CONFIGS]
        + ["vpt_shallow_best", "vpt_deep_best"]
    )

    f_sig_paths = []

    # Linear probe
    f_sig_paths.append(
        [
            _find_signature(
                experiments_dir
                / "linear_probe"
                / "top1_acc_ft"
                / backbone_name
                / dataset_name
                / "eval"
            )
            for dataset_name in _DATASET_NAMES
        ]
    )

    # Fixed-length VPT variants
    for _, prompt_mode, prompt_length in _VPT_CONFIGS:
        f_sig_paths.append(
            [
                _find_signature(
                    experiments_dir
                    / "vpt"
                    / "top1_acc_ft"
                    / backbone_name
                    / dataset_name
                    / "eval",
                    prompt_mode=prompt_mode,
                    prompt_length=prompt_length,
                )
                for dataset_name in _DATASET_NAMES
            ]
        )

    # Validation-selected VPT variants
    for prompt_mode in ("shallow", "deep"):
        f_sig_paths.append(
            [
                _find_best_signature(
                    experiments_dir
                    / "vpt"
                    / "top1_acc_ft"
                    / backbone_name
                    / dataset_name
                    / "eval",
                    prompt_mode=prompt_mode,
                )
                for dataset_name in _DATASET_NAMES
            ]
        )

    save_dir = Path(__file__).resolve().parent / backbone_name

    args = [
        {
            "model_names": model_names,
            "model_groups": _MODEL_GROUPS,
            "dataset_names": _DATASET_NAMES,
            "dataset_groups": _DATASET_GROUPS,
            "f_sig_paths": f_sig_paths,
            "f_sig_keys": _F_SIG_KEYS,
            "save_dir": str(save_dir),
            "save_name": None,
        }
    ]

    output_path = str(
        Path(__file__).resolve().parent
        / backbone_name
        / "analyze_fisher_signatures.txt"
    )

    exe_prefix = "python analyze_fisher_signatures_exe.py"

    return args, output_path, exe_prefix
