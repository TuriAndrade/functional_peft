import argparse
import json
from pathlib import Path

from task_functional_signature import FunctionalSignatureAnalyzer
from utils import parse_args_json


def _build_parser():
    parser = argparse.ArgumentParser(description="Analyze functional signatures.")

    parser.add_argument("--model-names", type=json.loads, default=None)
    parser.add_argument("--dataset-names", type=json.loads, default=None)
    parser.add_argument("--dataset-groups", type=json.loads, default=None)
    parser.add_argument("--f-sig-paths", type=json.loads, default=None)
    parser.add_argument("--f-sig-keys", type=json.loads, default=None)

    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--save-name", type=str, default=None)

    return parser


def main():
    args = parse_args_json(_build_parser())

    # Build output path first and skip if already completed.
    save_dir = Path(args.save_dir)
    if args.save_name is not None:
        save_dir /= args.save_name

    save_path = save_dir / "task_separation.csv"

    if save_path.exists():
        print(f"Experiment already done: {save_path}")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    analyzer = FunctionalSignatureAnalyzer(
        model_names=args.model_names,
        dataset_names=args.dataset_names,
        dataset_groups=args.dataset_groups,
        f_sig_paths=args.f_sig_paths,
        f_sig_keys=args.f_sig_keys,
        save_dir=save_dir,
    )

    analyzer.analyze()


if __name__ == "__main__":
    main()
