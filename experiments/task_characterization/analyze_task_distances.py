from pathlib import Path
from typing import Dict, List, Tuple

_SUFFIX = "analyze_task_distances"
_DISTANCE_ANALYSIS_DIR = "distance_analysis"

_REP_TYPE_RUNS = [
    ("input_patch_channel", ("input_patch_channel",)),
    ("input_pos_grid", ("input_pos_grid",)),
    ("block_path", ("block_path",)),
    ("attention_circuit_head", ("attention_circuit_head",)),
    ("mlp_memory_group", ("mlp_memory_group",)),
    ("residual_io_channel", ("residual_io_channel",)),
    ("attention_circuit_dim", ("attention_circuit_dim",)),
    ("mlp_memory_unit", ("mlp_memory_unit",)),
    ("input_interface", ("input_patch_channel", "input_pos_grid")),
    ("coarse_structure", ("input_patch_channel", "input_pos_grid", "block_path")),
    ("transformer_circuits", ("attention_circuit_head", "mlp_memory_group")),
    ("attention_residual", ("attention_circuit_head", "residual_io_channel")),
    ("mlp_residual", ("mlp_memory_group", "residual_io_channel")),
    (
        "internal_structure",
        ("attention_circuit_head", "mlp_memory_group", "residual_io_channel"),
    ),
    (
        "main_head_level",
        (
            "input_patch_channel",
            "input_pos_grid",
            "attention_circuit_head",
            "mlp_memory_group",
            "residual_io_channel",
        ),
    ),
    ("localization_detail", ("attention_circuit_dim", "mlp_memory_unit")),
]


def build_args(representations_path: str) -> Tuple[List[Dict], str, str]:
    representations_path = Path(representations_path)

    args = [
        {
            "representations_path": str(representations_path),
            "rep_types": list(rep_types),
            "save_name": save_name,
        }
        for save_name, rep_types in _REP_TYPE_RUNS
    ]

    output_path = str(representations_path / _DISTANCE_ANALYSIS_DIR / f"{_SUFFIX}.txt")
    exe_prefix = "python analyze_task_distances_exe.py"

    return args, output_path, exe_prefix
