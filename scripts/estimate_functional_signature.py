import argparse
import json
from pathlib import Path

import numpy as np
import torch

from datasets import datasets
from models import models
from models.utils import functional_seed_reliability
from utils import parse_args_json, seed_worker, set_seed

PEFT_METHODS = ("vpt", "lora", "adaptformer", "ssf")
SIGNATURE_VERSION = 1
STATISTICS = ("F", "P", "S")


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Estimate exact diagonal functional signatures (F, P, S)."
    )
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--backbone-name", type=str, default=None)
    parser.add_argument("--peft-name", type=str, choices=PEFT_METHODS, default=None)
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Directory containing one subdirectory per independently trained checkpoint.",
    )
    parser.add_argument("--ckpt-name", type=str, default="model.pt")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-chunk-size", type=int, default=64)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--basis-cache-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _get_checkpoint_dirs(model_path: Path, ckpt_name: str):
    checkpoint_dirs = sorted(path for path in model_path.iterdir() if path.is_dir())
    if not checkpoint_dirs:
        raise ValueError(f"No checkpoint directories found in {model_path}.")

    for checkpoint_dir in checkpoint_dirs:
        if not (checkpoint_dir / "config.json").exists():
            raise FileNotFoundError(f"Missing config.json in {checkpoint_dir}.")
        if not (checkpoint_dir / ckpt_name).exists():
            raise FileNotFoundError(f"Missing {ckpt_name} in {checkpoint_dir}.")

    return checkpoint_dirs


def _load_config(checkpoint_dir: Path):
    with open(checkpoint_dir / "config.json", encoding="utf-8") as f:
        return json.load(f)


def _validate_checkpoint_config(config, reference_config):
    if reference_config is None:
        return
    if config["peft_name"] != reference_config["peft_name"]:
        raise ValueError("Checkpoint PEFT methods do not match.")
    if config["peft_config"] != reference_config["peft_config"]:
        raise ValueError("Checkpoint PEFT configurations do not match.")
    if config["freeze_backbone"] != reference_config["freeze_backbone"]:
        raise ValueError("Checkpoint freeze_backbone settings do not match.")


def _resolve_peft(args, config):
    config_peft_name = config["peft_name"]
    if args.peft_name is not None and args.peft_name != config_peft_name:
        raise ValueError(
            f"PEFT method mismatch: requested '{args.peft_name}', "
            f"but config.json contains '{config_peft_name}'."
        )
    return args.peft_name or config_peft_name, config["peft_config"]


def _build_model(args, checkpoint_dir: Path, config):
    peft_name, peft_config = _resolve_peft(args, config)
    model = models[args.backbone_name][peft_name](
        img_size=datasets[args.dataset_name].IMG_SIZE,
        n_classes=datasets[args.dataset_name].N_CLASSES,
        freeze_backbone=config["freeze_backbone"],
        **peft_config,
    )
    model.load_from_path(checkpoint_dir / args.ckpt_name, map_location="cpu")
    return model.to(args.device), peft_name, peft_config


def _to_numpy(value):
    return (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )


def _signature_metadata(
    args,
    checkpoint_dir: Path,
    peft_name: str,
    peft_config,
    basis_cache_path: Path,
):
    return {
        "signature_version": SIGNATURE_VERSION,
        "dataset_name": args.dataset_name,
        "backbone_name": args.backbone_name,
        "peft_name": peft_name,
        "peft_config": json.dumps(peft_config, sort_keys=True),
        "checkpoint": checkpoint_dir.name,
        "eps": float(args.eps),
        "data_seed": int(args.data_seed),
        "basis_cache_path": str(basis_cache_path.resolve()),
    }


def _save_signature(path: Path, signature, metadata):
    arrays = {
        f"{statistic}__{component}": _to_numpy(value)
        for statistic in STATISTICS
        for component, value in signature[statistic].items()
    }
    arrays.update({name: np.asarray(value) for name, value in metadata.items()})
    np.savez_compressed(path, **arrays)


def _load_signature(path: Path, metadata):
    try:
        with np.load(path, allow_pickle=False) as data:
            for key, expected in metadata.items():
                if key not in data or data[key].item() != expected:
                    return None

            signature = {}
            for statistic in STATISTICS:
                prefix = f"{statistic}__"
                signature[statistic] = {
                    key[len(prefix) :]: data[key].copy()
                    for key in data.files
                    if key.startswith(prefix)
                }
                if not signature[statistic]:
                    return None

            components = tuple(signature["F"])
            if any(tuple(signature[name]) != components for name in ("P", "S")):
                return None
            return signature
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _save_json(path: Path, value):
    def clean(obj):
        if isinstance(obj, dict):
            return {key: clean(item) for key, item in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(item) for item in obj]
        if isinstance(obj, np.generic):
            return clean(obj.item())
        if isinstance(obj, float):
            return obj if np.isfinite(obj) else None
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean(value), f, indent=2)


def _estimate_checkpoint(
    args,
    checkpoint_dir: Path,
    *,
    basis_cache_path: Path,
    reference_config=None,
):
    config = _load_config(checkpoint_dir)
    _validate_checkpoint_config(config, reference_config)
    peft_name, peft_config = _resolve_peft(args, config)
    metadata = _signature_metadata(
        args,
        checkpoint_dir,
        peft_name,
        peft_config,
        basis_cache_path,
    )

    output_path = checkpoint_dir / "functional_signature.npz"
    signature = _load_signature(output_path, metadata) if output_path.exists() else None

    if signature is not None:
        print(f"Functional signature already exists: {output_path}. Loading.")
    else:
        if output_path.exists():
            print(f"Incompatible signature cache: {output_path}. Recomputing.")

        model, _, _ = _build_model(args, checkpoint_dir, config)
        try:
            set_seed(args.data_seed)
            ds = datasets[args.dataset_name](split="train+val")
            loader = ds.get_dataloader(
                batch_size=args.batch_size,
                worker_init_fn=seed_worker,
            )

            signature = model.estimate_functional_signature(
                loader,
                basis_cache_path=str(basis_cache_path),
                eps=args.eps,
                grad_chunk_size=args.grad_chunk_size,
                verbose=args.verbose,
            )
            _save_signature(output_path, signature, metadata)
            print(f"Saved functional signature to {output_path}")
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return signature, {
        "peft_name": peft_name,
        "peft_config": peft_config,
        "freeze_backbone": config["freeze_backbone"],
    }


def main():
    args = parse_args_json(_build_parser())
    model_path = Path(args.model_path)
    checkpoint_dirs = _get_checkpoint_dirs(model_path, args.ckpt_name)
    basis_cache_path = (
        Path(args.basis_cache_path)
        if args.basis_cache_path is not None
        else Path("functional_spectral_bases") / f"{args.backbone_name}.pt"
    )

    print("Found checkpoints: " + ", ".join(path.name for path in checkpoint_dirs))
    print(f"Spectral-basis cache: {basis_cache_path}")

    signatures = []
    reference_config = None
    for checkpoint_dir in checkpoint_dirs:
        print(
            "\n================================\n"
            f"Checkpoint: {checkpoint_dir.name}\n"
            "================================"
        )
        signature, checkpoint_config = _estimate_checkpoint(
            args,
            checkpoint_dir,
            basis_cache_path=basis_cache_path,
            reference_config=reference_config,
        )
        signatures.append(signature)
        if reference_config is None:
            reference_config = checkpoint_config

    reliability_path = model_path / "functional_signature_reliability.json"
    if len(signatures) >= 2:
        reliability = functional_seed_reliability(signatures)
        _save_json(
            reliability_path,
            {
                "checkpoints": [path.name for path in checkpoint_dirs],
                "data_seed": args.data_seed,
                **reliability,
            },
        )
        print(f"Saved model-seed reliability to {reliability_path}")
    else:
        reliability_path.unlink(missing_ok=True)

    print(
        f"\nFinished functional-signature extraction for {len(checkpoint_dirs)} checkpoint(s)."
    )


if __name__ == "__main__":
    main()
