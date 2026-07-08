import argparse
from pathlib import Path

import numpy as np

from task_characterization import TaskDistanceViewer
from utils import parse_args_json


def _build_parser():
    p = argparse.ArgumentParser(description="View Task2Vec distance matrices.")
    p.add_argument("--distance-matrix-path", type=str, default=None)
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--save-name", type=str, default=None)
    return p


def _parse_args():
    return parse_args_json(_build_parser())


def main():
    args = _parse_args()

    path = Path(args.distance_matrix_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing distance matrix file: {path}")

    root = Path(args.save_dir) if args.save_dir else path.parent / "distance_analysis"
    if args.save_name:
        root = root / args.save_name

    with np.load(path, allow_pickle=False) as data:
        dataset_names = [str(x) for x in data["dataset_names"]]
        view_names = [str(x) for x in data["view_names"]]

        for view in view_names:
            model_names = [str(x) for x in data[f"{view}__model_names"]]
            distance_matrix = data[f"{view}__distance_matrix"]
            save_dir = root / view

            result = TaskDistanceViewer(
                distance_matrix=distance_matrix,
                model_names=model_names,
                dataset_names=dataset_names,
                save_dir=save_dir,
            ).compute()

            out = Path(result["output_dir"])
            print(f"{view}: saved to {out} | shape={distance_matrix.shape}")


if __name__ == "__main__":
    main()