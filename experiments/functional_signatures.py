import json
import os
import shlex
from pathlib import Path

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
DATASETS = VTAB + ["imagenet_val"]

DATASET_GROUPS = {
    "vtab_natural": VTAB_NATURAL,
    "vtab_specialized": VTAB_SPECIALIZED,
    "vtab_structured": VTAB_STRUCTURED,
    "vtab": VTAB,
    "vtab+imagenet_val": VTAB + ["imagenet_val"],
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
# FINE-TUNING METHODS
# ======================================================================
PEFT_METHODS = (
    "vpt_shallow",
    "vpt_deep",
    "lora",
    "ssf",
    "adaptformer",
)

FINETUNING_METHODS = (
    "linear_probe",
    "full_ft",
    *PEFT_METHODS,
)

FINETUNING_METHOD_GROUPS = {
    "peft": PEFT_METHODS,
    "all": FINETUNING_METHODS,
}


# ======================================================================
# FUNCTIONAL-SIGNATURE ESTIMATION ARGS
# ======================================================================
EVAL_SEEDS = (0, 1, 2)

MC_SAMPLES = 20
BATCH_SIZE = 1
EPS = 1e-12
DEVICE = "cuda"
VERBOSE = True

EVAL_EXPERIMENT_NAME = "top_1_acc_eval"
ESTIMATE_EXE = "estimate_functional_signatures_exe.py"
CHECKPOINT_NAME = "model.pt"
CONFIG_NAME = "config.json"


# ======================================================================
# HELPERS
# ======================================================================
def _resolve_names(name_or_group, names, groups, kind):
    if name_or_group in groups:
        return list(groups[name_or_group])

    if name_or_group in names:
        return [name_or_group]

    raise ValueError(f"Unknown {kind} or group: {name_or_group}")


def _find_eval_runs(eval_dir: Path):
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Eval directory not found: {eval_dir}")

    runs_by_seed = {}

    for run_dir in sorted(path for path in eval_dir.iterdir() if path.is_dir()):
        config_path = run_dir / CONFIG_NAME
        checkpoint_path = run_dir / CHECKPOINT_NAME

        if not config_path.is_file() or not checkpoint_path.is_file():
            continue

        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        seed = config.get("run_seed")
        if seed not in EVAL_SEEDS:
            continue

        if seed in runs_by_seed:
            raise RuntimeError(
                f"Found multiple completed eval runs for seed {seed} in {eval_dir}: "
                f"{runs_by_seed[seed].name}, {run_dir.name}"
            )

        runs_by_seed[seed] = run_dir

    missing = [seed for seed in EVAL_SEEDS if seed not in runs_by_seed]
    if missing:
        raise FileNotFoundError(
            f"Missing eval checkpoints for seeds {missing} in: {eval_dir}"
        )

    return [runs_by_seed[seed] for seed in EVAL_SEEDS]


# ======================================================================
# BUILD FUNCTIONAL-SIGNATURE RUNS
# ======================================================================
def build_functional_signature_runs(
    dataset_name_or_group: str = "all",
    backbone_name_or_group: str = "all",
    ft_name_or_group: str = "all",
    output_name: str = "functional_signatures",
    max_runs_per_file: int = 0,
):
    if max_runs_per_file < 0:
        raise ValueError("max_runs_per_file must be non-negative")

    dataset_names = _resolve_names(
        dataset_name_or_group,
        DATASETS,
        DATASET_GROUPS,
        "dataset",
    )
    backbone_names = _resolve_names(
        backbone_name_or_group,
        BACKBONES,
        BACKBONE_GROUPS,
        "backbone",
    )
    method_names = _resolve_names(
        ft_name_or_group,
        FINETUNING_METHODS,
        FINETUNING_METHOD_GROUPS,
        "fine-tuning method",
    )

    experiments_dir = Path(os.environ.get("EXPERIMENTS_DIR", "./experiments"))
    output_dir = Path(__file__).resolve().parent

    commands = []

    for backbone_name in backbone_names:
        for dataset_name in dataset_names:
            for ft_name in method_names:
                eval_dir = (
                    experiments_dir
                    / EVAL_EXPERIMENT_NAME
                    / backbone_name
                    / dataset_name
                    / ft_name
                )

                for model_path in _find_eval_runs(eval_dir):
                    with (model_path / CONFIG_NAME).open("r", encoding="utf-8") as f:
                        config = json.load(f)

                    args = {
                        "dataset_name": dataset_name,
                        "backbone_name": backbone_name,
                        "peft_name": config["peft_name"],
                        "model_path": str(model_path),
                        "batch_size": BATCH_SIZE,
                        "mc_samples": MC_SAMPLES,
                        "eps": EPS,
                        "device": DEVICE,
                        # Store the signatures next to the checkpoint/config for
                        # the corresponding training seed.
                        "save_dir": str(model_path),
                        "verbose": VERBOSE,
                    }

                    args_json = json.dumps(args)
                    commands.append(
                        f"python {ESTIMATE_EXE} --args-json {shlex.quote(args_json)}"
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

        path = output_dir / filename

        with path.open("w", encoding="utf-8") as f:
            f.write("\n".join(chunk) + "\n")
