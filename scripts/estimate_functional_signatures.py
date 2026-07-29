import argparse
import json
from pathlib import Path

import numpy as np

from datasets import datasets
from models import vpt
from utils import parse_args_json, seed_worker, set_seed


def _build_parser():
    parser = argparse.ArgumentParser(description="Estimate VPT functional signatures.")

    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--backbone-name", type=str, default=None)

    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--ckpt-name", type=str, default=None)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--save-name", type=str, default=None)

    return parser


def main():
    args = parse_args_json(_build_parser())

    model_path = Path(args.model_path)

    with open(model_path / "config.json") as f:
        config = json.load(f)

    seed = config.get("run_seed", 0)
    set_seed(seed)

    # Train+val task data used to estimate the model Fisher.
    ds = datasets[args.dataset_name](split="train+val")

    loader = ds.get_dataloader(
        batch_size=args.batch_size,
        worker_init_fn=seed_worker,
    )

    # Reconstruct the exact trained VPT configuration.
    model = vpt[args.backbone_name](
        img_size=datasets[args.dataset_name].IMG_SIZE,
        n_classes=datasets[args.dataset_name].N_CLASSES,
        prompt_length=config["prompt_length"],
        prompt_mode=config["prompt_mode"],
        freeze_backbone=config["freeze_backbone"],
    )

    model.load_from_path(
        model_path / args.ckpt_name,
        map_location="cpu",
    )
    model.to(args.device)

    signatures = model.estimate_functional_signatures(
        loader,
        mc_samples=args.mc_samples,
        eps=args.eps,
    )

    # Flatten nested signature dictionaries into NPZ keys such as
    # F__attention_head and C__mlp_neuron.
    arrays = {
        f"{signature_name}__{component_name}": values.detach().cpu().numpy()
        for signature_name, signature in signatures.items()
        for component_name, values in signature.items()
    }

    save_dir = Path(args.save_dir)
    if args.save_name is not None:
        save_dir /= args.save_name
    save_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        save_dir / "functional_signatures.npz",
        dataset_name=np.array(args.dataset_name),
        backbone_name=np.array(args.backbone_name),
        prompt_mode=np.array(config["prompt_mode"]),
        prompt_length=np.array(config["prompt_length"]),
        mc_samples=np.array(args.mc_samples),
        eps=np.array(args.eps),
        **arrays,
    )


if __name__ == "__main__":
    main()
