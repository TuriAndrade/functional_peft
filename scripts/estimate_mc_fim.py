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

    parser.add_argument("--load-linear-probe-path", type=str, default=None)
    parser.add_argument("--load-linear-probe-name", type=str, default=None)
    parser.add_argument("--load-prompt-model-paths", type=json.loads, default=None)
    parser.add_argument("--load-prompt-model-names", type=json.loads, default=None)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default="fim_outputs/")
    parser.add_argument("--save-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mc-samples", type=int, default=10)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--plot-norm",
        type=str,
        default="percentile",
        choices=["max", "percentile"],
    )
    parser.add_argument("--plot-percentile", type=float, default=99.0)

    return parser


def _parse_args():
    return parse_args_json(_build_parser())


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_state_dict(path: Path):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]

    return ckpt


def _check_same_dataset(configs):
    dataset_names = [cfg["dataset_name"] for cfg in configs]

    if len(set(dataset_names)) != 1:
        raise ValueError(
            "All loaded models must use the same dataset for this script. "
            f"Found: {dataset_names}"
        )

    return dataset_names[0]


def _check_model_args(args):
    if args.load_linear_probe_path is None:
        raise ValueError("--load-linear-probe-path is required.")

    if args.load_linear_probe_name is None:
        raise ValueError("--load-linear-probe-name is required.")

    if args.load_prompt_model_paths is None:
        raise ValueError("--load-prompt-model-paths is required.")

    if args.load_prompt_model_names is None:
        raise ValueError("--load-prompt-model-names is required.")

    if len(args.load_prompt_model_paths) == 0:
        raise ValueError("--load-prompt-model-paths must contain at least one path.")

    if len(args.load_prompt_model_names) != len(args.load_prompt_model_paths):
        raise ValueError(
            "--load-prompt-model-names must have the same length as "
            "--load-prompt-model-paths. "
            f"Found {len(args.load_prompt_model_names)} names and "
            f"{len(args.load_prompt_model_paths)} paths."
        )


def _load_model(model_dir: Path, img_size: int, n_classes: int):
    config = _load_json(model_dir / "config.json")
    ckpt_path = model_dir / "model.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing model.pt: {ckpt_path}")

    model = vpt[config["backbone_name"]](
        img_size=img_size,
        n_classes=n_classes,
        prompt_length=int(config["prompt_length"]),
        prompt_mode=config["prompt_mode"],
        freeze_backbone=bool(config["freeze_backbone"]),
    )

    model.load_state_dict(_load_state_dict(ckpt_path), strict=False)

    return model, config


def _load_prompt_models(model_dirs, model_names, img_size: int, n_classes: int):
    models, configs = [], []

    for model_name, model_dir in zip(model_names, model_dirs):
        model, config = _load_model(model_dir, img_size, n_classes)
        models.append((model_name, model))
        configs.append(config)

    return models, configs


def main():
    args = _parse_args()
    _check_model_args(args)

    linear_probe_dir = Path(args.load_linear_probe_path)
    prompt_model_dirs = [Path(p) for p in args.load_prompt_model_paths]

    configs = [_load_json(linear_probe_dir / "config.json")]
    configs += [
        _load_json(model_dir / "config.json") for model_dir in prompt_model_dirs
    ]

    dataset_name = _check_same_dataset(configs)

    seed = args.seed if args.seed is not None else int(configs[0].get("run_seed", 0))
    set_seed(seed)

    img_size = datasets[dataset_name].IMG_SIZE
    n_classes = datasets[dataset_name].N_CLASSES

    fim_ds = datasets[dataset_name](split="train+val")
    fim_loader = fim_ds.get_dataloader(
        batch_size=args.batch_size,
        worker_init_fn=seed_worker,
    )

    linear_probe_model, _ = _load_model(linear_probe_dir, img_size, n_classes)
    prompt_models, _ = _load_prompt_models(
        prompt_model_dirs,
        args.load_prompt_model_names,
        img_size,
        n_classes,
    )

    save_dir = Path(args.save_dir)
    if args.save_name is not None:
        save_dir = save_dir / args.save_name
    save_dir.mkdir(parents=True, exist_ok=True)

    estimator = MonteCarloFIMEstimator(
        linear_probe=(args.load_linear_probe_name, linear_probe_model),
        prompt_models=prompt_models,
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

    with (save_dir / "mc_fim_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "seed": seed,
                "dataset_name": dataset_name,
                "img_size": img_size,
                "n_classes": n_classes,
                "linear_probe_name": args.load_linear_probe_name,
                "linear_probe_dir": str(linear_probe_dir),
                "prompt_model_names": args.load_prompt_model_names,
                "prompt_model_dirs": [str(p) for p in prompt_model_dirs],
            },
            f,
            indent=2,
        )

    del fim_ds, fim_loader, linear_probe_model, prompt_models, estimator

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
