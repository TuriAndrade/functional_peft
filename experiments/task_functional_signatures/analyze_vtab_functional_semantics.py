import json
from pathlib import Path
from typing import Any

_RUN_SEED = 0

_NATURAL = (
    "vtab_caltech101",
    "vtab_cifar100",
    "vtab_dtd",
    "vtab_flowers102",
    "vtab_pets",
    "vtab_sun397",
    "vtab_svhn",
)
_SPECIALIZED = (
    "vtab_eurosat",
    "vtab_resisc45",
    "vtab_patch_camelyon",
    "vtab_retinopathy",
)
_STRUCTURED = (
    "vtab_clevr_count",
    "vtab_clevr_distance",
    "vtab_dmlab",
    "vtab_dsprites_location",
    "vtab_dsprites_orientation",
    "vtab_kitti_distance",
    "vtab_smallnorb_azimuth",
    "vtab_smallnorb_elevation",
)
_DATASET_NAMES = _NATURAL + _SPECIALIZED + _STRUCTURED + ("imagenet_val",)

_VPT_CONFIGS = (
    ("vpt_deep_5", "deep", 5),
    ("vpt_deep_10", "deep", 10),
    ("vpt_deep_50", "deep", 50),
    ("vpt_deep_100", "deep", 100),
    ("vpt_shallow_5", "shallow", 5),
    ("vpt_shallow_10", "shallow", 10),
    ("vpt_shallow_50", "shallow", 50),
    ("vpt_shallow_100", "shallow", 100),
    ("vpt_shallow_200", "shallow", 200),
)

_MODEL_GROUPS = {
    "vpt_deep": tuple(
        name for name, prompt_mode, _ in _VPT_CONFIGS if prompt_mode == "deep"
    ),
    "vpt_shallow": tuple(
        name for name, prompt_mode, _ in _VPT_CONFIGS if prompt_mode == "shallow"
    ),
}


def _find_signature(
    eval_dir: Path,
    prompt_mode: str | None = None,
    prompt_length: int | None = None,
) -> str:
    matches = []

    for run_dir in sorted(eval_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.endswith(f"_{_RUN_SEED}"):
            continue

        signature_path = run_dir / "functional_signatures.npz"
        if not signature_path.exists():
            continue

        if prompt_mode is not None:
            with open(run_dir / "config.json") as file:
                config = json.load(file)

            if (
                config["prompt_mode"] != prompt_mode
                or config["prompt_length"] != prompt_length
            ):
                continue

        matches.append(signature_path)

    if len(matches) != 1:
        raise ValueError(
            f"Expected one signature in {eval_dir}, found {len(matches)} "
            f"for prompt_mode={prompt_mode}, "
            f"prompt_length={prompt_length}."
        )

    return str(matches[0])


def _find_best_signature(
    eval_dir: Path,
    prompt_mode: str,
) -> str:
    eval_path = eval_dir.parent / "eval.json"

    with open(eval_path) as file:
        results = json.load(file)

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
) -> tuple[list[dict[str, Any]], str, str]:
    experiments_dir = Path(__file__).resolve().parent.parent
    datasets = _DATASET_NAMES

    def eval_dir(
        method: str,
        dataset: str,
    ) -> Path:
        return (
            experiments_dir / method / "top1_acc_ft" / backbone_name / dataset / "eval"
        )

    model_names = [
        "linear_probe",
        *(name for name, _, _ in _VPT_CONFIGS),
        "vpt_shallow_best",
        "vpt_deep_best",
    ]

    signature_paths = [
        [_find_signature(eval_dir("linear_probe", dataset)) for dataset in datasets]
    ]

    signature_paths.extend(
        [
            _find_signature(
                eval_dir("vpt", dataset),
                prompt_mode=prompt_mode,
                prompt_length=prompt_length,
            )
            for dataset in datasets
        ]
        for _, prompt_mode, prompt_length in _VPT_CONFIGS
    )

    signature_paths.extend(
        [
            _find_best_signature(
                eval_dir("vpt", dataset),
                prompt_mode=prompt_mode,
            )
            for dataset in datasets
        ]
        for prompt_mode in ("shallow", "deep")
    )

    save_dir = Path(__file__).resolve().parent / backbone_name

    args = [
        {
            "model_names": model_names,
            "model_groups": _MODEL_GROUPS,
            "signature_paths": signature_paths,
            "save_dir": str(save_dir),
            "verbose": True,
            "create_plots": True,
        }
    ]

    output_path = str(save_dir / "analyze_vtab_functional_semantics.txt")
    exe_prefix = "python analyze_vtab_functional_semantics_exe.py"

    return args, output_path, exe_prefix
