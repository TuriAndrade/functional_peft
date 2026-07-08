from pathlib import Path
from typing import Dict, List, Tuple

_SUFFIX = "view_task2vec_distances"
_DISTANCE_VISUALIZATION_DIR = "task2vec_distance_visualization"


def _find_npz(directory: Path) -> Path:
    files = sorted(directory.glob("*.npz"))
    if len(files) != 1:
        raise ValueError(
            f"Expected exactly one .npz file in {directory}, found {len(files)}."
        )
    return files[0]


def build_args(distances_path: str) -> Tuple[List[Dict], str, str]:
    distances_path = Path(distances_path)

    dirs = sorted(p for p in distances_path.iterdir() if p.is_dir())

    args = [
        {
            "distance_matrix_path": str(_find_npz(directory)),
            "save_dir": str(directory),
            "save_name": _DISTANCE_VISUALIZATION_DIR,
        }
        for directory in dirs
    ]

    output_path = str(distances_path / f"{_SUFFIX}.txt")
    exe_prefix = "python view_task2vec_distances_exe.py"

    return args, output_path, exe_prefix
