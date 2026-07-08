import argparse
import json
from pathlib import Path

import numpy as np

from task_characterization import Task2VecDistanceEstimator
from utils import parse_args_json

REP_TYPES = [
    "input_patch_channel",
    "input_pos_grid",
    "block_path",
    "attention_circuit_head",
    "mlp_memory_group",
    "residual_io_channel",
    "attention_circuit_dim",
    "mlp_memory_unit",
]


def _build_parser():
    parser = argparse.ArgumentParser(description="Estimate Task2Vec distances.")

    parser.add_argument("--representations-path", type=str, default=None)
    parser.add_argument("--dataset-names", type=json.loads, default=None)

    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--save-name", type=str, default=None)
    parser.add_argument("--save-filename", type=str, default="distance_matrix.npz")

    parser.add_argument("--rep-types", type=json.loads, default=REP_TYPES)
    parser.add_argument("--eps", type=float, default=1e-12)

    return parser


def _parse_args():
    return parse_args_json(_build_parser())


def _find_npz(dataset_dir: Path):
    path = dataset_dir / "fim_representations.npz"
    if path.exists():
        return path

    files = sorted(dataset_dir.glob("*.npz"))
    if len(files) != 1:
        raise ValueError(
            f"Expected fim_representations.npz or exactly one .npz in "
            f"{dataset_dir}, found {len(files)}."
        )

    return files[0]


def _has_npz(dataset_dir: Path):
    return (dataset_dir / "fim_representations.npz").exists() or bool(
        list(dataset_dir.glob("*.npz"))
    )


def _get_dataset_names(root: Path, dataset_names):
    if dataset_names is not None:
        return list(map(str, dataset_names))

    dirs = sorted(p for p in root.iterdir() if p.is_dir() and _has_npz(p))
    if not dirs:
        raise FileNotFoundError(
            f"No dataset directories with .npz files found in {root}"
        )

    return [p.name for p in dirs]


def _read_model_names(npz_path: Path):
    with np.load(npz_path, allow_pickle=False) as data:
        linear_probe_name = str(np.asarray(data["linear_probe_name"]).item())
        prompt_model_names = [str(x) for x in data["prompt_model_names"]]

    return linear_probe_name, prompt_model_names


def main():
    args = _parse_args()

    root = Path(args.representations_path)
    if not root.exists():
        raise FileNotFoundError(f"Missing representations path: {root}")

    dataset_names = _get_dataset_names(root, args.dataset_names)
    linear_probe_name, prompt_model_names = _read_model_names(
        _find_npz(root / dataset_names[0])
    )

    save_dir = Path(args.save_dir) if args.save_dir else root / "distance_matrices"
    if args.save_name:
        save_dir = save_dir / args.save_name

    result = Task2VecDistanceEstimator(
        representations_path=root,
        dataset_names=dataset_names,
        linear_probe_name=linear_probe_name,
        prompt_model_names=prompt_model_names,
        save_dir=save_dir,
    ).compute(
        rep_types=args.rep_types,
        eps=args.eps,
        filename=args.save_filename,
    )

    print(f"Saved distance matrices to: {result['output_path']}")
    print(f"Linear probe: {linear_probe_name}")
    print(f"Prompt models: {prompt_model_names}")
    print(f"Dataset names: {dataset_names}")

    for view, matrix in result["distance_matrices"].items():
        print(f"{view}: shape={matrix.shape}, " f"models={result['model_names'][view]}")


if __name__ == "__main__":
    main()
