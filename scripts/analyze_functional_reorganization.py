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

    # SQ1 — global task / method / interaction
    analyzer.analyze_task_method_decomposition()

    # SQ2 — task-group vs individual-task hierarchy
    analyzer.analyze_task_hierarchy()

    # SQ3 — per-task / per-group consistency across methods
    analyzer.analyze_task_consistency()

    # SQ4 — per-method specificity
    analyzer.analyze_method_specificity()


if __name__ == "__main__":
    main()
