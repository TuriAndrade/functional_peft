import argparse
from pathlib import Path

from task_characterization import FIMRankEstimator
from utils import parse_args_json


def _build_parser():
    p = argparse.ArgumentParser(description="Estimate FIM coordinate rankings.")
    p.add_argument("--representations-path", type=str, default=None)
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--save-name", type=str, default=None)
    return p


def main():
    args = parse_args_json(_build_parser())

    path = Path(args.representations_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing representations path: {path}")

    root = Path(args.save_dir) if args.save_dir else path.parent / "fim_rank_estimates"
    if args.save_name:
        root = root / args.save_name

    result = FIMRankEstimator(
        representations_path=path,
        save_dir=root,
    ).compute()

    print(f"Saved FIM rank estimates to: {root}")
    print(f"Saved full rankings to: {result['output_path']}")
    print(f"Saved top/bottom table to: {result['table_path']}")


if __name__ == "__main__":
    main()
