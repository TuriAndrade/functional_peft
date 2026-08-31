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

# Independently trained checkpoints.
EVAL_SEEDS = (0, 1, 2)

# Independent MC estimation runs within each checkpoint.
MC_SEEDS = (0, 1, 2)

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


def _resolve_names(
    name_or_group,
    names,
    groups,
    kind,
):
    if name_or_group in groups:
        return list(groups[name_or_group])

    if name_or_group in names:
        return [name_or_group]

    raise ValueError(f"Unknown {kind} or group: " f"{name_or_group}")


def _find_eval_runs(
    eval_dir: Path,
):
    """
    Validate that eval_dir contains exactly one completed checkpoint
    for each required training seed.

    The returned directories are only used for validation and for
    reading a representative config. The estimation script receives
    eval_dir itself as model_path.
    """
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Eval directory not found: " f"{eval_dir}")

    run_dirs = sorted(path for path in eval_dir.iterdir() if path.is_dir())

    if not run_dirs:
        raise FileNotFoundError(
            f"No eval checkpoint directories " f"found in: {eval_dir}"
        )

    runs_by_seed = {}

    for run_dir in run_dirs:
        config_path = run_dir / CONFIG_NAME

        checkpoint_path = run_dir / CHECKPOINT_NAME

        # The estimation script treats every subdirectory as a
        # checkpoint, so every subdirectory must be complete.
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Missing {CONFIG_NAME} in " f"checkpoint directory: " f"{run_dir}"
            )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing {CHECKPOINT_NAME} in " f"checkpoint directory: " f"{run_dir}"
            )

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            config = json.load(f)

        seed = config.get("run_seed")

        if seed not in EVAL_SEEDS:
            raise RuntimeError(
                f"Found unexpected eval seed "
                f"{seed} in {run_dir}. "
                f"Expected only {EVAL_SEEDS}."
            )

        if seed in runs_by_seed:
            raise RuntimeError(
                "Found multiple completed "
                f"eval runs for seed {seed} "
                f"in {eval_dir}: "
                f"{runs_by_seed[seed].name}, "
                f"{run_dir.name}"
            )

        runs_by_seed[seed] = run_dir

    missing = [seed for seed in EVAL_SEEDS if seed not in runs_by_seed]

    if missing:
        raise FileNotFoundError(
            "Missing eval checkpoints " f"for seeds {missing} in: " f"{eval_dir}"
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
        raise ValueError("max_runs_per_file must " "be non-negative")

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

    experiments_dir = Path(
        os.environ.get(
            "EXPERIMENTS_DIR",
            "./experiments",
        )
    )

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

                # Validate the three independently trained checkpoints.
                eval_runs = _find_eval_runs(eval_dir)

                # All checkpoint configs should describe the same
                # adaptation method/configuration. The estimation
                # script validates this again across checkpoints.
                with (eval_runs[0] / CONFIG_NAME).open(
                    "r",
                    encoding="utf-8",
                ) as f:
                    config = json.load(f)

                args = {
                    "dataset_name": (dataset_name),
                    "backbone_name": (backbone_name),
                    "peft_name": (config["peft_name"]),
                    # Parent directory containing all independently
                    # trained checkpoint subdirectories.
                    "model_path": str(eval_dir),
                    "ckpt_name": (CHECKPOINT_NAME),
                    "batch_size": (BATCH_SIZE),
                    "mc_samples": (MC_SAMPLES),
                    "eps": (EPS),
                    "device": (DEVICE),
                    # MC estimation seeds within each checkpoint.
                    "run_seeds": list(MC_SEEDS),
                    "verbose": (VERBOSE),
                }

                args_json = json.dumps(args)

                commands.append(
                    f"python {ESTIMATE_EXE} "
                    f"--args-json "
                    f"{shlex.quote(args_json)}"
                )

    if max_runs_per_file == 0:
        chunks = [commands]

    else:
        chunks = [
            commands[i : i + max_runs_per_file]
            for i in range(
                0,
                len(commands),
                max_runs_per_file,
            )
        ]

    for i, chunk in enumerate(chunks):
        filename = (
            f"{output_name}.txt"
            if max_runs_per_file == 0
            else (f"{output_name}_" f"{i}.txt")
        )

        path = output_dir / filename

        with path.open(
            "w",
            encoding="utf-8",
        ) as f:
            f.write("\n".join(chunk) + "\n")
