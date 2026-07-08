import argparse
import json
from pathlib import Path

import numpy as np

from task_characterization import TaskSeparationAnalyzer
from utils import parse_args_json

VTAB_GROUPS = {
    "caltech101": "natural",
    "cifar100": "natural",
    "dtd": "natural",
    "flowers102": "natural",
    "pets": "natural",
    "sun397": "natural",
    "svhn": "natural",
    "eurosat": "specialized",
    "resisc45": "specialized",
    "patch_camelyon": "specialized",
    "retinopathy": "specialized",
    "clevr_count": "structured",
    "clevr_distance": "structured",
    "dmlab": "structured",
    "dsprites_location": "structured",
    "dsprites_orientation": "structured",
    "kitti_distance": "structured",
    "smallnorb_azimuth": "structured",
    "smallnorb_elevation": "structured",
}

GROUP_PRESETS = {
    "vtab_groups": VTAB_GROUPS,
    "vtab_natural_or_not": {
        d: "natural" if g == "natural" else "not_natural"
        for d, g in VTAB_GROUPS.items()
    },
    "vtab_specialized_or_not": {
        d: "specialized" if g == "specialized" else "not_specialized"
        for d, g in VTAB_GROUPS.items()
    },
    "vtab_structured_or_not": {
        d: "structured" if g == "structured" else "not_structured"
        for d, g in VTAB_GROUPS.items()
    },
}


def _build_parser():
    p = argparse.ArgumentParser(description="Analyze task group separation.")
    p.add_argument("--distance-matrix-path", type=str, default=None)
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--save-name", type=str, default=None)
    p.add_argument(
        "--group-preset",
        type=str,
        default="vtab_groups",
        choices=sorted(GROUP_PRESETS),
    )
    p.add_argument("--n-permutations", type=int, default=999)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--top-k", type=int, default=3)
    return p


def main():
    args = parse_args_json(_build_parser())

    path = Path(args.distance_matrix_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing distance matrix file: {path}")

    root = Path(args.save_dir) if args.save_dir else path.parent / "separation_analysis"
    if args.save_name:
        root = root / args.save_name

    dataset_groups = GROUP_PRESETS[args.group_preset]

    with np.load(path, allow_pickle=False) as data:
        dataset_names = [str(x) for x in data["dataset_names"]]
        view_names = [str(x) for x in data["view_names"]]

        for view in view_names:
            model_names = [str(x) for x in data[f"{view}__model_names"]]
            distance_matrix = data[f"{view}__distance_matrix"]
            save_dir = root / view

            result = TaskSeparationAnalyzer(
                distance_matrix=distance_matrix,
                model_names=model_names,
                dataset_names=dataset_names,
                dataset_groups=dataset_groups,
                save_dir=save_dir,
            ).compute(
                n_permutations=args.n_permutations,
                random_state=args.random_state,
                top_k=args.top_k,
            )

            save_dir.mkdir(parents=True, exist_ok=True)
            with open(save_dir / "separation_config.json", "w") as f:
                json.dump(
                    {
                        "distance_matrix_path": str(path),
                        "view": view,
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

            print(
                f"{view}: saved to {save_dir} | "
                f"shape={distance_matrix.shape} | "
                f"models={result['model_names']} | "
                f"used={len(result['dataset_names'])} | "
                f"ignored={len(result['ignored_dataset_names'])}"
            )


if __name__ == "__main__":
    main()
