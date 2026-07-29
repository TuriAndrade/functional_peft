import json
from pathlib import Path
from typing import List, Dict, Tuple

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
    # ---------------------------------------------------------
    # Coarse VTAB taxonomy
    # ---------------------------------------------------------
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
    # ---------------------------------------------------------
    # Relation to ImageNet pretraining domain
    #
    # ImageNet is the singleton anchor:
    # Natural < non-Natural VTAB.
    # ---------------------------------------------------------
    "imagenet_reference": (
        ["imagenet_val"],
        _NATURAL,
        _SPECIALIZED + _STRUCTURED,
    ),
    # ---------------------------------------------------------
    # Specialized-domain subgroups defined by VTAB
    # ---------------------------------------------------------
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
    # ---------------------------------------------------------
    # Same-source / provenance
    # ---------------------------------------------------------
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
    # ---------------------------------------------------------
    # Cross-dataset functional relations
    # ---------------------------------------------------------
    "functional_fine_grained": (
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
    "functional_distance": (
        [
            "vtab_clevr_distance",
            "vtab_kitti_distance",
        ],
        [
            "vtab_clevr_count",
            "vtab_dmlab",
            "vtab_dsprites_location",
            "vtab_dsprites_orientation",
            "vtab_smallnorb_azimuth",
            "vtab_smallnorb_elevation",
        ],
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


def build_args(
    backbone_name: str,
) -> Tuple[List[Dict], str, str]:
    experiments_dir = Path(__file__).resolve().parent.parent

    model_names = ["linear_probe"] + [name for name, _, _ in _VPT_CONFIGS]
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

    # VPT variants
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

    save_dir = Path(__file__).resolve().parent / backbone_name

    args = [
        {
            "model_names": model_names,
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
        / "analyze_functional_signatures.txt"
    )
    exe_prefix = "python analyze_functional_signatures_exe.py"

    return args, output_path, exe_prefix
