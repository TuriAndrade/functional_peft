import argparse
import json
from pathlib import Path

from task_functional_signatures import VTABFunctionalSemanticsAnalyzer
from utils import parse_args_json


def _build_parser():
    parser = argparse.ArgumentParser(description="Analyze VTAB functional semantics.")

    parser.add_argument("--model-names", type=json.loads, default=None)
    parser.add_argument("--model-groups", type=json.loads, default=None)

    parser.add_argument("--signature-paths", type=json.loads, default=None)

    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--save-name", type=str, default=None)

    parser.add_argument("--verbose", action="store_true")

    return parser


def main():
    args = parse_args_json(_build_parser())

    save_dir = Path(args.save_dir)
    if args.save_name is not None:
        save_dir /= args.save_name

    save_dir.mkdir(parents=True, exist_ok=True)

    analyzer = VTABFunctionalSemanticsAnalyzer(
        model_names=args.model_names,
        model_groups=args.model_groups,
        signature_paths=args.signature_paths,
        save_dir=save_dir,
        verbose=args.verbose,
    )

    analyzer.analyze()


if __name__ == "__main__":
    main()
