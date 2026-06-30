from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_SUFFIX = "analyze_task_separation"

_REP_TYPE_RUNS = [
    ("fim_block",),
    ("fim_block_attn",),
    ("fim_block_attn_q",),
    ("fim_block_attn_k",),
    ("fim_block_attn_v",),
    ("fim_block_attn_o",),
    ("fim_block_mlp",),
    ("fim_patch_embd",),
    ("fim_block_attn", "fim_block_mlp"),
    ("fim_block_attn_q", "fim_block_attn_k"),
    ("fim_block_attn_v", "fim_block_attn_o"),
    ("fim_block", "fim_patch_embd"),
    ("fim_block_attn", "fim_block_mlp", "fim_patch_embd"),
]

_GROUP_PRESETS = (
    "vtab_groups",
    "vtab_natural_or_not",
    "vtab_specialized_or_not",
    "vtab_structured_or_not",
)


def _save_name(rep_types: Sequence[str]) -> str:
    return "__".join(rep_type.removeprefix("fim_") for rep_type in rep_types)


def build_args(distance_analysis_path: str) -> Tuple[List[Dict], str, str]:
    distance_analysis_path = Path(distance_analysis_path)

    args = [
        {
            "distance_matrix_path": str(
                distance_analysis_path / _save_name(rep_types) / "distance_matrices.npz"
            ),
            "group_preset": group_preset,
            "save_name": group_preset,
        }
        for rep_types in _REP_TYPE_RUNS
        for group_preset in _GROUP_PRESETS
    ]

    output_path = str(distance_analysis_path / f"{_SUFFIX}.txt")
    exe_prefix = "python analyze_task_separation_exe.py"

    return args, output_path, exe_prefix
