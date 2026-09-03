import argparse

from functional_signatures import FunctionalReorganizationAnalyzer


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Analyze functional reorganization for RQ1."
    )

    parser.add_argument(
        "--backbone-name",
        type=str,
        default="vit_b16",
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
    )

    return parser


def main():
    args = _build_parser().parse_args()

    analyzer = FunctionalReorganizationAnalyzer(
        backbone_name=args.backbone_name,
        eps=args.eps,
        verbose=not args.quiet,
    )

    analyzer.analyze_pair_attribution()  # SQ1
    analyzer.analyze_dataset_organization()  # SQ2
    analyzer.analyze_method_organization()  # SQ3
    analyzer.analyze_global_organization()  # SQ4
    analyzer.analyze_prediction_alignment()  # SQ5


if __name__ == "__main__":
    main()
