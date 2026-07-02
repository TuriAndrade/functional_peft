import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from task_characterization import TaskDistanceAnalyzer
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
    parser = argparse.ArgumentParser(description="Analyze dataset distances.")

    parser.add_argument("--representations-path", type=str, default=None)
    parser.add_argument("--dataset-names", type=json.loads, default=None)
    parser.add_argument("--model-names", type=json.loads, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--save-name", type=str, default=None)
    parser.add_argument("--rep-types", type=json.loads, default=REP_TYPES)
    parser.add_argument(
        "--distance-metric", type=str, default="task2vec", choices=["task2vec"]
    )
    parser.add_argument(
        "--linkage-method",
        type=str,
        default="average",
        choices=["single", "complete", "average", "weighted"],
    )
    parser.add_argument("--cluster-ks", type=json.loads, default=[2, 3, 4, 5])
    parser.add_argument("--eps", type=float, default=1e-12)

    return parser


def _parse_args():
    return parse_args_json(_build_parser())


def _find_npz(dataset_dir: Path):
    files = sorted(dataset_dir.glob("*.npz"))
    if len(files) != 1:
        raise ValueError(
            f"Expected exactly one .npz file in {dataset_dir}, found {len(files)}."
        )
    return files[0]


def _get_dataset_files(root: Path, dataset_names: Sequence[str] | None):
    if dataset_names is None:
        dirs = sorted(p for p in root.iterdir() if p.is_dir() and list(p.glob("*.npz")))
        dataset_names = [p.name for p in dirs]
    else:
        dirs = [root / name for name in dataset_names]

    if not dirs:
        raise FileNotFoundError(
            f"No dataset directories with .npz files found in {root}"
        )

    return list(dataset_names), [_find_npz(p) for p in dirs]


def _get_model_names(npz_path: Path, model_names: Sequence[str] | None):
    with np.load(npz_path, allow_pickle=False) as data:
        available = [str(x) for x in data["model_names"]]

    if model_names is None:
        return available

    missing = [m for m in model_names if m not in available]
    if missing:
        raise ValueError(
            f"Models not found in {npz_path}: {missing}. Available: {available}"
        )

    return list(model_names)


def _load_embeddings(
    npz_paths: Sequence[Path],
    model_names: Sequence[str],
    rep_types: Sequence[str],
):
    out = []

    for path in npz_paths:
        with np.load(path, allow_pickle=False) as data:
            file_models = [str(x) for x in data["model_names"]]
            model_to_idx = {m: i for i, m in enumerate(file_models)}

            per_model = []
            for model_name in model_names:
                i = model_to_idx[model_name]
                vec = np.concatenate(
                    [
                        np.asarray(data[r][i], dtype=np.float64).reshape(-1)
                        for r in rep_types
                    ]
                )
                per_model.append(np.maximum(vec, 0.0))

            out.append(np.stack(per_model))

    return np.stack(out, axis=1)  # [n_models, n_datasets, dim]


def _task2vec_distance(a, b, eps):
    den = a + b
    a = np.divide(a, den, out=np.zeros_like(a), where=den > eps)
    b = np.divide(b, den, out=np.zeros_like(b), where=den > eps)

    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= eps and nb <= eps:
        return 0.0
    if na <= eps or nb <= eps:
        return 1.0

    return float(1.0 - np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def _build_distance_matrix(embeddings, eps):
    n_models, n_datasets, _ = embeddings.shape
    D = np.zeros((n_models, n_models, n_datasets, n_datasets), dtype=np.float64)

    for mi in range(n_models):
        for mj in range(mi, n_models):
            for di in range(n_datasets):
                for dj in range(n_datasets):
                    d = _task2vec_distance(embeddings[mi, di], embeddings[mj, dj], eps)
                    D[mi, mj, di, dj] = d
                    D[mj, mi, dj, di] = d

    for m in range(n_models):
        np.fill_diagonal(D[m, m], 0.0)

    return D


def main():
    args = _parse_args()

    if args.representations_path is None:
        raise ValueError("--representations-path is required.")

    root = Path(args.representations_path)
    if not root.exists():
        raise FileNotFoundError(f"Missing representations path: {root}")

    dataset_names, npz_paths = _get_dataset_files(root, args.dataset_names)
    model_names = _get_model_names(npz_paths[0], args.model_names)
    rep_types = args.rep_types
    cluster_ks = [int(k) for k in args.cluster_ks]

    save_dir = (
        Path(args.save_dir) if args.save_dir is not None else root / "distance_analysis"
    )
    if args.save_name:
        save_dir = save_dir / args.save_name

    embeddings = _load_embeddings(npz_paths, model_names, rep_types)
    distance_matrix = _build_distance_matrix(embeddings, args.eps)

    analyzer = TaskDistanceAnalyzer(
        distance_matrix=distance_matrix,
        model_names=model_names,
        dataset_names=dataset_names,
        save_dir=save_dir,
    )

    result = analyzer.compute(
        linkage_method=args.linkage_method,
        cluster_ks=cluster_ks,
        eps=args.eps,
    )

    output_dir = Path(result["output_dir"])

    print(f"Saved distance analysis to: {output_dir}")
    print(f"Saved clustering outputs to: {output_dir / 'clustering_outputs.npz'}")
    print(f"Saved clustering summary to: {output_dir / 'clustering_summary.json'}")
    print(f"Saved per-model plots to: {output_dir / 'per_model'}")
    print(f"Saved 2D plots to: {output_dir / 'cluster_2d'}")
    print(f"Saved cross-model plots to: {output_dir / 'cross_model'}")
    print(f"Embedding vectors shape: {embeddings.shape}")
    print(f"Distance matrix shape: {distance_matrix.shape}")


if __name__ == "__main__":
    main()
