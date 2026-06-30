import argparse
import json
from pathlib import Path

from task_characterization import TaskSeparationAnalyzer
from utils import parse_args_json

VTAB_GROUPS = {
    # Natural
    "caltech101": "natural",
    "cifar100": "natural",
    "dtd": "natural",
    "flowers102": "natural",
    "pets": "natural",
    "sun397": "natural",
    "svhn": "natural",
    # Specialized
    "eurosat": "specialized",
    "resisc45": "specialized",
    "patch_camelyon": "specialized",
    "retinopathy": "specialized",
    # Structured
    "clevr_count": "structured",
    "clevr_distance": "structured",
    "dmlab": "structured",
    "dsprites_location": "structured",
    "dsprites_orientation": "structured",
    "kitti_distance": "structured",
    "smallnorb_azimuth": "structured",
    "smallnorb_elevation": "structured",
}


VTAB_NATURAL_OR_NOT = {
    name: ("natural" if group == "natural" else "not_natural")
    for name, group in VTAB_GROUPS.items()
}

VTAB_SPECIALIZED_OR_NOT = {
    name: ("specialized" if group == "specialized" else "not_specialized")
    for name, group in VTAB_GROUPS.items()
}

VTAB_STRUCTURED_OR_NOT = {
    name: ("structured" if group == "structured" else "not_structured")
    for name, group in VTAB_GROUPS.items()
}


GROUP_PRESETS = {
    "vtab_groups": VTAB_GROUPS,
    "vtab_natural_or_not": VTAB_NATURAL_OR_NOT,
    "vtab_specialized_or_not": VTAB_SPECIALIZED_OR_NOT,
    "vtab_structured_or_not": VTAB_STRUCTURED_OR_NOT,
}


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Analyze semantic group separation in task distance matrices."
    )

    parser.add_argument(
        "--distance-matrix-path",
        type=str,
        default=None,
        help="Path to distance_matrices.npz saved by TaskDistanceAnalyzer.",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help=(
            "Where to save outputs. If omitted, saves under "
            "<distance-matrix-parent>/separation_analysis."
        ),
    )

    parser.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="Optional subdirectory name for this separation analysis.",
    )

    parser.add_argument(
        "--group-preset",
        type=str,
        default="vtab_groups",
        choices=sorted(GROUP_PRESETS.keys()),
        help="Dataset grouping preset to evaluate.",
    )

    parser.add_argument(
        "--n-permutations",
        type=int,
        default=999,
        help="Number of label permutations for p-values.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for permutation tests.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k nearest neighbors used for retrieval purity.",
    )

    return parser


def _parse_args():
    return parse_args_json(_build_parser())


def main():
    args = _parse_args()

    if args.distance_matrix_path is None:
        raise ValueError("--distance-matrix-path is required.")

    distance_matrix_path = Path(args.distance_matrix_path)
    if not distance_matrix_path.exists():
        raise FileNotFoundError(f"Missing distance matrix file: {distance_matrix_path}")

    save_dir = (
        Path(args.save_dir)
        if args.save_dir is not None
        else distance_matrix_path.parent / "separation_analysis"
    )

    if args.save_name:
        save_dir = save_dir / args.save_name

    dataset_groups = GROUP_PRESETS[args.group_preset]

    analyzer = TaskSeparationAnalyzer(
        distance_matrix_path=distance_matrix_path,
        dataset_groups=dataset_groups,
        save_dir=save_dir,
    )

    result = analyzer.compute(
        n_permutations=args.n_permutations,
        random_state=args.random_state,
        top_k=args.top_k,
    )

    with open(save_dir / "separation_config.json", "w") as f:
        json.dump(
            {
                "distance_matrix_path": str(distance_matrix_path),
                "save_dir": str(save_dir),
                "group_preset": args.group_preset,
                "dataset_groups": dataset_groups,
                "n_permutations": args.n_permutations,
                "random_state": args.random_state,
                "top_k": args.top_k,
            },
            f,
            indent=2,
        )

    print(f"Saved separation analysis to: {save_dir}")
    print(f"Saved summary metrics to: {save_dir / 'summary_metrics.csv'}")
    print(f"Saved summary JSON to: {save_dir / 'summary_metrics.json'}")
    print(f"Saved pairwise deltas to: {save_dir / 'pairwise_deltas'}")
    print(f"Used datasets: {len(result['dataset_names'])}")
    print(f"Ignored datasets: {len(result['ignored_dataset_names'])}")
    print(f"Models: {result['model_names']}")


if __name__ == "__main__":
    main()
