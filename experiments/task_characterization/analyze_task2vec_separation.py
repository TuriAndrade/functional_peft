from pathlib import Path
from typing import Dict, List, Tuple

_SUFFIX = "analyze_task2vec_separation"
_SEPARATION_ANALYSIS_DIR = "task2vec_separation_analysis"

_GROUP_PRESETS = (
    "vtab_groups",
    "vtab_natural_or_not",
    "vtab_specialized_or_not",
    "vtab_structured_or_not",
)


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
            "save_name": f"{group_preset}_{_SEPARATION_ANALYSIS_DIR}",
            "group_preset": group_preset,
        }
        for group_preset in _GROUP_PRESETS
        for directory in dirs
    ]

    output_path = str(distances_path / f"{_SUFFIX}.txt")
    exe_prefix = "python analyze_task2vec_separation_exe.py"

    return args, output_path, exe_prefix
