import argparse
import json
from pathlib import Path

import numpy as np

from datasets import datasets
from models import models
from utils import parse_args_json, seed_worker, set_seed

PEFT_METHODS = ("vpt", "lora", "adaptformer", "ssf")


def _build_parser():
    parser = argparse.ArgumentParser(description="Estimate PEFT functional signatures.")

    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--backbone-name", type=str, default=None)
    parser.add_argument(
        "--peft-name",
        type=str,
        choices=PEFT_METHODS,
        default=None,
    )

    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--ckpt-name", type=str, default="model.pt")

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--run-seeds",
        type=json.loads,
        default=[0],
        help="List of estimation seeds to run. Example: --run-seeds '[0, 1, 2]'",
    )

    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--save-name", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")

    return parser


def _estimate_seed(args, run_seed: int):
    # Build output path first and skip if already completed.
    save_dir = Path(args.save_dir)

    if args.save_name is not None:
        save_dir /= args.save_name

    save_path = save_dir / f"functional_signatures_seed-{run_seed}.npz"

    if save_path.exists():
        print(f"Experiment already done: {save_path}. Skipping.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    # Set the seed for this functional-signature estimation run.
    set_seed(run_seed)

    # Load the configuration of the trained model.
    model_path = Path(args.model_path)

    with open(model_path / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    # Ensure the requested PEFT method matches the trained model.
    config_peft_name = config["peft_name"]

    if args.peft_name is not None and args.peft_name != config_peft_name:
        raise ValueError(
            f"PEFT method mismatch: requested '{args.peft_name}', "
            f"but config.json contains '{config_peft_name}'."
        )

    peft_name = args.peft_name or config_peft_name
    peft_config = config["peft_config"]

    # Train+val task data used to estimate the model Fisher.
    ds = datasets[args.dataset_name](split="train+val")
    loader = ds.get_dataloader(
        batch_size=args.batch_size,
        worker_init_fn=seed_worker,
    )

    # Reconstruct the exact trained PEFT model.
    model = models[args.backbone_name][peft_name](
        img_size=datasets[args.dataset_name].IMG_SIZE,
        n_classes=datasets[args.dataset_name].N_CLASSES,
        freeze_backbone=config["freeze_backbone"],
        **peft_config,
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
        verbose=args.verbose,
    )

    # Flatten nested signature dictionaries into NPZ keys such as
    # F_m__attention_head_output and C_global_m__mlp_neuron.
    arrays = {
        f"{signature_name}__{component_name}": values.detach().cpu().numpy()
        for signature_name, signature in signatures.items()
        for component_name, values in signature.items()
    }

    np.savez_compressed(
        save_path,
        dataset_name=np.array(args.dataset_name),
        backbone_name=np.array(args.backbone_name),
        peft_name=np.array(peft_name),
        peft_config=np.array(json.dumps(peft_config)),
        run_seed=np.array(run_seed),
        mc_samples=np.array(args.mc_samples),
        eps=np.array(args.eps),
        **arrays,
    )


def main():
    args = parse_args_json(_build_parser())

    for run_seed in args.run_seeds:
        print(f"\n===== Functional-signature seed {run_seed} =====")
        _estimate_seed(
            args,
            run_seed,
        )


if __name__ == "__main__":
    main()
