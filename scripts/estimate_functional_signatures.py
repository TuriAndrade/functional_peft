import argparse
import json
from pathlib import Path

import numpy as np

from datasets import datasets
from models import models
from utils import parse_args_json, seed_worker, set_seed

PEFT_METHODS = ("vpt", "lora", "adaptformer", "ssf")

STATISTIC_COUNT_KEYS = (
    "input_count",
    "sample_count_m",
    "sample_count_e",
)

STATISTIC_TENSOR_KEYS = (
    "participation_sum",
    "fisher_sum_m",
    "score_sum_e",
    "fisher_sum_e",
    "cross_sum_m",
    "coupling_square_sum_m",
    "cross_sum_e",
    "coupling_score_sum_e",
    "coupling_square_sum_e",
)


def _build_parser():
    parser = argparse.ArgumentParser(description="Estimate PEFT functional signatures.")

    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--backbone-name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--peft-name",
        type=str,
        choices=PEFT_METHODS,
        default=None,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Directory containing one subdirectory per "
            "independently trained checkpoint."
        ),
    )

    parser.add_argument(
        "--ckpt-name",
        type=str,
        default="model.pt",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--mc-samples",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--run-seeds",
        type=json.loads,
        default=[0],
        help=("List of MC estimation seeds. " "Example: --run-seeds '[0, 1, 2]'"),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    return parser


def _get_checkpoint_dirs(
    model_path: Path,
    ckpt_name: str,
):
    checkpoint_dirs = sorted(path for path in model_path.iterdir() if path.is_dir())

    if not checkpoint_dirs:
        raise ValueError(f"No checkpoint directories found in {model_path}.")

    for checkpoint_dir in checkpoint_dirs:
        config_path = checkpoint_dir / "config.json"

        checkpoint_path = checkpoint_dir / ckpt_name

        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json in " f"{checkpoint_dir}.")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing {ckpt_name} in " f"{checkpoint_dir}.")

    return checkpoint_dirs


def _load_config(
    checkpoint_dir: Path,
):
    with open(
        checkpoint_dir / "config.json",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def _build_model(
    args,
    checkpoint_dir: Path,
    config,
):
    config_peft_name = config["peft_name"]

    if args.peft_name is not None and args.peft_name != config_peft_name:
        raise ValueError(
            "PEFT method mismatch: "
            f"requested '{args.peft_name}', "
            "but config.json contains "
            f"'{config_peft_name}'."
        )

    peft_name = args.peft_name or config_peft_name

    peft_config = config["peft_config"]

    model = models[args.backbone_name][peft_name](
        img_size=datasets[args.dataset_name].IMG_SIZE,
        n_classes=datasets[args.dataset_name].N_CLASSES,
        freeze_backbone=config["freeze_backbone"],
        **peft_config,
    )

    model.load_from_path(
        checkpoint_dir / args.ckpt_name,
        map_location="cpu",
    )

    model.to(args.device)

    return (
        model,
        peft_name,
        peft_config,
    )


def _save_statistics(
    path: Path,
    statistics,
    run_seed: int,
    mc_samples: int,
):
    arrays = {key: np.array(statistics[key]) for key in STATISTIC_COUNT_KEYS}

    arrays.update(
        {key: (statistics[key].detach().cpu().numpy()) for key in STATISTIC_TENSOR_KEYS}
    )

    np.savez_compressed(
        path,
        run_seed=np.array(run_seed),
        mc_samples=np.array(mc_samples),
        **arrays,
    )


def _load_statistics(
    path: Path,
):
    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        statistics = {key: int(data[key]) for key in STATISTIC_COUNT_KEYS}

        statistics.update({key: data[key].copy() for key in STATISTIC_TENSOR_KEYS})

    return statistics


def _save_signatures(
    path: Path,
    signatures,
    dataset_name: str,
    backbone_name: str,
    peft_name: str,
    peft_config,
    **metadata,
):
    arrays = {
        (f"{signature_name}" f"__{component_name}"): (values.detach().cpu().numpy())
        for (
            signature_name,
            signature,
        ) in signatures.items()
        for (
            component_name,
            values,
        ) in signature.items()
    }

    metadata_arrays = {name: np.array(value) for name, value in metadata.items()}

    np.savez_compressed(
        path,
        dataset_name=np.array(dataset_name),
        backbone_name=np.array(backbone_name),
        peft_name=np.array(peft_name),
        peft_config=np.array(json.dumps(peft_config)),
        **metadata_arrays,
        **arrays,
    )


def _save_json(
    path: Path,
    value,
):
    def clean(obj):
        if isinstance(
            obj,
            dict,
        ):
            return {key: clean(item) for key, item in obj.items()}

        if isinstance(
            obj,
            (list, tuple),
        ):
            return [clean(item) for item in obj]

        if isinstance(
            obj,
            float,
        ):
            return obj if np.isfinite(obj) else None

        return obj

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            clean(value),
            f,
            indent=2,
        )


def _estimate_checkpoint(
    args,
    checkpoint_dir: Path,
    reference_config=None,
):
    config = _load_config(checkpoint_dir)

    model, peft_name, peft_config = _build_model(
        args,
        checkpoint_dir,
        config,
    )

    if reference_config is not None:
        if peft_name != reference_config["peft_name"]:
            raise ValueError("Checkpoint PEFT methods " "do not match.")

        if peft_config != reference_config["peft_config"]:
            raise ValueError("Checkpoint PEFT " "configurations do not match.")

        if config["freeze_backbone"] != reference_config["freeze_backbone"]:
            raise ValueError("Checkpoint freeze_backbone " "settings do not match.")

    statistics = []

    for run_seed in args.run_seeds:
        statistics_path = checkpoint_dir / (
            "functional_statistics_" f"seed-{run_seed}.npz"
        )

        if statistics_path.exists():
            print("Statistics already " f"exist: {statistics_path}. " "Loading.")

        else:
            print("\n" f"===== {checkpoint_dir.name}: " f"MC seed {run_seed} =====")

            set_seed(run_seed)

            ds = datasets[args.dataset_name](split="train+val")

            loader = ds.get_dataloader(
                batch_size=(args.batch_size),
                worker_init_fn=(seed_worker),
            )

            stats = model.estimate_functional_statistics(
                loader,
                mc_samples=(args.mc_samples),
                verbose=(args.verbose),
            )

            _save_statistics(
                statistics_path,
                stats,
                run_seed,
                args.mc_samples,
            )

        statistics.append(_load_statistics(statistics_path))

    # --------------------------------------------------------------
    # Reliability across MC estimates of this checkpoint.
    # --------------------------------------------------------------

    mc_reliability = model.functional_statistics_reliability(
        statistics,
        eps=args.eps,
    )

    _save_json(
        checkpoint_dir / ("functional_statistics_" "reliability.json"),
        {
            "run_seeds": (args.run_seeds),
            "icc_a1": (mc_reliability),
        },
    )

    # --------------------------------------------------------------
    # Merge the MC statistics into one checkpoint signature.
    # --------------------------------------------------------------

    checkpoint_signatures = model.functional_signatures_from_statistics(
        statistics,
        eps=args.eps,
    )

    _save_signatures(
        checkpoint_dir / "functional_signatures.npz",
        checkpoint_signatures,
        dataset_name=(args.dataset_name),
        backbone_name=(args.backbone_name),
        peft_name=(peft_name),
        peft_config=(peft_config),
        run_seeds=np.array(args.run_seeds),
        eps=args.eps,
    )

    return (
        model,
        checkpoint_signatures,
        {
            "peft_name": (peft_name),
            "peft_config": (peft_config),
            "freeze_backbone": (config["freeze_backbone"]),
        },
    )


def main():
    args = parse_args_json(_build_parser())

    model_path = Path(args.model_path)

    checkpoint_dirs = _get_checkpoint_dirs(
        model_path,
        args.ckpt_name,
    )

    print(
        "Found checkpoints: "
        + ", ".join(checkpoint_dir.name for checkpoint_dir in checkpoint_dirs)
    )

    checkpoint_signatures = []

    reference_config = None
    model = None

    for checkpoint_dir in checkpoint_dirs:
        print(
            "\n"
            "================================"
            "\n"
            f"Checkpoint: "
            f"{checkpoint_dir.name}"
            "\n"
            "================================"
        )

        (
            model,
            signatures,
            checkpoint_config,
        ) = _estimate_checkpoint(
            args,
            checkpoint_dir,
            reference_config=(reference_config),
        )

        if reference_config is None:
            reference_config = checkpoint_config

        checkpoint_signatures.append(signatures)

    # --------------------------------------------------------------
    # Reliability across independently trained checkpoints.
    # --------------------------------------------------------------

    checkpoint_reliability = model.functional_signature_reliability(
        checkpoint_signatures
    )

    _save_json(
        model_path / ("functional_signatures_" "reliability.json"),
        {
            "checkpoints": [path.name for path in checkpoint_dirs],
            **checkpoint_reliability,
        },
    )

    # --------------------------------------------------------------
    # Final dataset x method functional signature.
    # --------------------------------------------------------------

    final_signatures = model.merge_functional_signatures(checkpoint_signatures)

    _save_signatures(
        model_path / "functional_signatures.npz",
        final_signatures,
        dataset_name=(args.dataset_name),
        backbone_name=(args.backbone_name),
        peft_name=(reference_config["peft_name"]),
        peft_config=(reference_config["peft_config"]),
        checkpoints=np.array([path.name for path in checkpoint_dirs]),
        run_seeds=np.array(args.run_seeds),
        eps=args.eps,
    )

    print(
        "\nSaved final functional "
        f"signatures to "
        f"{model_path / 'functional_signatures.npz'}"
    )


if __name__ == "__main__":
    main()
