import argparse
import json
from pathlib import Path

from task_characterization import FIMTaskGeometryAnalyzer
from utils import parse_args_json


def _parse_json_or_str(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_cluster_ks(value: str):
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = [
            item.strip()
            for chunk in value.split(",")
            for item in chunk.split()
            if item.strip()
        ]

    if isinstance(parsed, int):
        return [parsed]

    return [int(k) for k in parsed]


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Analyze MC FIM dataset distances and clustering."
    )

    parser.add_argument(
        "--representations-path",
        type=str,
        required=True,
        help=(
            "Directory containing one subdirectory per dataset. Each dataset "
            "directory must contain fim_representations.npz."
        ),
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help=(
            "Where to save distance matrices, clustered plots, and metadata. "
            "If omitted, saves under <representations-path>/distance_analysis."
        ),
    )

    parser.add_argument(
        "--rep-types",
        type=_parse_json_or_str,
        default="block",
        help=(
            "Representation types to use. "
            "string, or a JSON list. Valid names: block, block_attn, "
            "block_attn_q, block_attn_k, block_attn_v, block_attn_o, "
            "block_mlp, patch_embd."
        ),
    )

    parser.add_argument(
        "--npz-filename",
        type=str,
        default="fim_representations.npz",
        help="Name of the FIM representations file inside each dataset directory.",
    )

    parser.add_argument(
        "--save-name",
        type=str,
        default="dataset_fim_distances",
        help="Subdirectory name for the saved distance analysis.",
    )

    parser.add_argument(
        "--linkage-method",
        type=str,
        default="average",
        choices=["single", "complete", "average", "weighted"],
        help="Hierarchical clustering linkage method.",
    )

    parser.add_argument(
        "--cluster-ks",
        type=_parse_cluster_ks,
        default=[2, 3, 4, 5],
        help=(
            "Cluster counts to save from the dendrogram. "
            "Use a JSON list like '[2,3,4,5]' or a string like '2,3,4,5'."
        ),
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Numerical epsilon used in the Task2Vec distance.",
    )

    return parser


def _parse_args():
    parser = _build_parser()
    return parse_args_json(parser)


def _get_dataset_fim_paths(
    representations_path: Path,
    npz_filename: str,
):
    dataset_dirs = [
        path
        for path in sorted(representations_path.iterdir())
        if path.is_dir() and (path / npz_filename).exists()
    ]

    if len(dataset_dirs) == 0:
        raise FileNotFoundError(
            f"No dataset directories with {npz_filename} found in "
            f"{representations_path}"
        )

    dataset_names = [path.name for path in dataset_dirs]
    dataset_fim_paths = [str(path) for path in dataset_dirs]

    return dataset_names, dataset_fim_paths


def main():
    args = _parse_args()

    representations_path = Path(args.representations_path)
    if not representations_path.exists():
        raise FileNotFoundError(f"Missing representations path: {representations_path}")

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = representations_path / "distance_analysis"

    if args.save_name is not None:
        save_dir = save_dir / args.save_name

    dataset_names, dataset_fim_paths = _get_dataset_fim_paths(
        representations_path=representations_path,
        npz_filename=args.npz_filename,
    )

    analyzer = FIMTaskGeometryAnalyzer(
        dataset_fim_paths=dataset_fim_paths,
        dataset_names=dataset_names,
        save_dir=save_dir,
        npz_filename=args.npz_filename,
    )
    result = analyzer.analyze(
        rep_types=args.rep_types,
        linkage_method=args.linkage_method,
        cluster_ks=args.cluster_ks,
        eps=args.eps,
    )

    output_dir = Path(result["output_dir"])

    print(f"Saved distance analysis to: {output_dir}")
    print(f"Saved distance matrices to: {output_dir / 'distance_matrices.npz'}")
    print(f"Saved clustering summary to: {output_dir / 'clustering_summary.json'}")
    print(f"Saved per-model plots to: {output_dir / 'per_model'}")
    print(f"Saved cross-model plots to: {output_dir / 'cross_model'}")
    print(f"Embedding vectors shape: {result['embedding_vectors'].shape}")
    print(f"Per-model distance shape: {result['distance_by_model'].shape}")
    print(f"Cross-model distance shape: {result['cross_model_distance'].shape}")


if __name__ == "__main__":
    main()
