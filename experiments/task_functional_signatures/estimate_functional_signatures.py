from pathlib import Path
from typing import List, Dict, Tuple

_RUN_SEED = 0

_COMMON_ARGS = {
    "batch_size": 1,
    "mc_samples": 20,
    "eps": 1e-12,
    "device": "cuda",
}

_METHODS = ("vpt", "linear_probe")


def build_args(
    backbone_name: str,
    ckpt_name: str = "model.pt",
) -> Tuple[List[Dict], str, str]:
    experiments_dir = Path(__file__).resolve().parent.parent

    args = []

    for method in _METHODS:
        base_dir = experiments_dir / method / "top1_acc_ft" / backbone_name

        for dataset_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
            eval_dir = dataset_dir / "eval"
            if not eval_dir.is_dir():
                continue

            for model_path in sorted(
                p
                for p in eval_dir.iterdir()
                if p.is_dir() and p.name.endswith(f"_{_RUN_SEED}")
            ):
                args.append(
                    {
                        **_COMMON_ARGS,
                        "dataset_name": dataset_dir.name,
                        "backbone_name": backbone_name,
                        "model_path": str(model_path),
                        "ckpt_name": ckpt_name,
                        "save_dir": str(model_path),
                        "save_name": None,
                    }
                )

    output_path = str(
        Path(__file__).resolve().parent
        / backbone_name
        / "estimate_functional_signatures.txt"
    )
    exe_prefix = "python estimate_functional_signatures_exe.py"

    return args, output_path, exe_prefix
