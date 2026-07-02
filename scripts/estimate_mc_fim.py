import argparse
import gc
import json
from pathlib import Path

import torch

from datasets import datasets
from models import vpt
from task_characterization import MonteCarloFIMEstimator
from utils import set_seed, seed_worker, parse_args_json


def _build_parser():
    parser = argparse.ArgumentParser(description="Monte Carlo FIM estimator.")

    parser.add_argument(
        "--load-model-paths",
        type=json.loads,
        default=None,
        help=(
            "JSON array with model directories. Each directory must contain "
            "config.json and model.pt. Example: "
            '--load-model-paths \'["path/to/model1", "path/to/model2"]\''
        ),
    )

    parser.add_argument(
        "--load-model-names",
        type=json.loads,
        default=None,
        help="JSON array with model names. If None, default names are used.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for FIM computation.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device.",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="fim_outputs/",
        help="Where to save FIM representations and plots.",
    )

    parser.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="Optional subdirectory inside save-dir.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for pseudo-label sampling. If None, uses first config run_seed or 0.",
    )

    parser.add_argument(
        "--mc-samples",
        type=int,
        default=10,
        help="Number of Monte Carlo pseudo-label samples per batch.",
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional maximum number of batches for debugging.",
    )

    parser.add_argument(
        "--plot-norm",
        type=str,
        default="percentile",
        choices=["max", "percentile"],
        help="Heatmap scale statistic.",
    )

    parser.add_argument(
        "--plot-percentile",
        type=float,
        default=99.0,
        help="Percentile used if --plot-norm percentile.",
    )

    return parser


def _parse_args():
    parser = _build_parser()
    return parse_args_json(parser)


def _load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def _load_state_dict(path: Path):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    return ckpt


def _get_model_name(config: dict):
    backbone_name = config["backbone_name"]
    prompt_length = int(config["prompt_length"])
    prompt_mode = config["prompt_mode"]
    freeze_backbone = bool(config["freeze_backbone"])

    if prompt_length > 0:
        return f"{backbone_name}_{prompt_mode}_{prompt_length}"

    if freeze_backbone:
        return f"{backbone_name}_linear_probe"

    return f"{backbone_name}_full_ft"


def _check_same_dataset(configs):
    dataset_names = [cfg["dataset_name"] for cfg in configs]
    first_dataset = dataset_names[0]

    for dataset_name in dataset_names:
        if dataset_name != first_dataset:
            raise ValueError(
                "All loaded models must use the same dataset for this script. "
                f"Found: {dataset_names}"
            )

    return first_dataset


def _load_model_from_dir(
    model_dir: Path, img_size: int, n_classes: int, model_name=None
):
    config_path = model_dir / "config.json"
    ckpt_path = model_dir / "model.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json: {config_path}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing model.pt: {ckpt_path}")

    config = _load_json(config_path)

    model = vpt[config["backbone_name"]](
        img_size=img_size,
        n_classes=n_classes,
        prompt_length=int(config["prompt_length"]),
        prompt_mode=config["prompt_mode"],
        freeze_backbone=bool(config["freeze_backbone"]),
    )

    state_dict = _load_state_dict(ckpt_path)
    model.load_state_dict(state_dict, strict=False)

    if model_name is None:
        model_name = _get_model_name(config)

    return model, model_name, config


def _load_all_models(model_dirs, img_size: int, n_classes: int, model_names=None):
    models = []
    loaded_model_names = []
    configs = []

    for i, model_dir in enumerate(model_dirs):
        model, model_name, config = _load_model_from_dir(
            model_dir=model_dir,
            img_size=img_size,
            n_classes=n_classes,
            model_name=None if model_names is None else model_names[i],
        )

        models.append(model)
        loaded_model_names.append(model_name)
        configs.append(config)

    return models, loaded_model_names, configs


def main():
    args = _parse_args()

    if args.load_model_paths is None:
        raise ValueError("--load-model-paths is required.")

    model_dirs = [Path(p) for p in args.load_model_paths]

    if len(model_dirs) == 0:
        raise ValueError("--load-model-paths must contain at least one path.")

    if args.load_model_names is not None and len(args.load_model_names) != len(
        model_dirs
    ):
        raise ValueError(
            "--load-model-names must have the same length as --load-model-paths. "
            f"Found {len(args.load_model_names)} names and {len(model_dirs)} paths."
        )

    # ----- Read configs first -----
    configs = []
    for model_dir in model_dirs:
        config_path = model_dir / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json: {config_path}")

        configs.append(_load_json(config_path))

    dataset_name = _check_same_dataset(configs)

    # ----- Seed -----
    seed = args.seed
    if seed is None:
        seed = int(configs[0].get("run_seed", 0))

    set_seed(seed)

    # ----- Dataset: train+val -----
    img_size = datasets[dataset_name].IMG_SIZE
    n_classes = datasets[dataset_name].N_CLASSES

    fim_ds = datasets[dataset_name](split="train+val")
    fim_loader = fim_ds.get_dataloader(
        batch_size=args.batch_size,
        worker_init_fn=seed_worker,
    )

    # ----- Models -----
    models, model_names, configs = _load_all_models(
        model_dirs=model_dirs,
        img_size=img_size,
        n_classes=n_classes,
        model_names=args.load_model_names,
    )

    # ----- Save dir -----
    save_dir = Path(args.save_dir)
    if args.save_name is not None:
        save_dir = save_dir / args.save_name

    save_dir.mkdir(parents=True, exist_ok=True)

    # ----- FIM estimator -----
    estimator = MonteCarloFIMEstimator(
        models=models,
        model_names=model_names,
        save_dir=save_dir,
        plot_norm=args.plot_norm,
        plot_percentile=args.plot_percentile,
    )

    estimator.compute(
        loader=fim_loader,
        device=args.device,
        max_batches=args.max_batches,
        mc_samples=args.mc_samples,
        save=True,
        plot=True,
    )

    # ----- Save run config -----
    with open(save_dir / "mc_fim_config.json", "w") as f:
        json.dump(
            {
                **vars(args),
                "seed": seed,
                "dataset_name": dataset_name,
                "img_size": img_size,
                "n_classes": n_classes,
                "model_names": model_names,
                "model_dirs": [str(p) for p in model_dirs],
            },
            f,
            indent=2,
        )

    # ----- Cleanup -----
    del fim_ds, fim_loader
    del models
    del estimator

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
