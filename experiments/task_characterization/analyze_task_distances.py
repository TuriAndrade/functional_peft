from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_SUFFIX = "analyze_task_distances"
_DISTANCE_ANALYSIS_DIR = "distance_analysis"

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


def _save_name(rep_types: Sequence[str]) -> str:
    return "__".join(rep_type.removeprefix("fim_") for rep_type in rep_types)


def build_args(representations_path: str) -> Tuple[List[Dict], str, str]:
    representations_path = Path(representations_path)

    args = [
        {
            "representations_path": str(representations_path),
            "rep_types": list(rep_types),
            "save_name": _save_name(rep_types),
        }
        for rep_types in _REP_TYPE_RUNS
    ]

    output_path = str(representations_path / _DISTANCE_ANALYSIS_DIR / f"{_SUFFIX}.txt")
    exe_prefix = "python analyze_task_distances_exe.py"

    return args, output_path, exe_prefix
