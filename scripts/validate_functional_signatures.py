import argparse

from functional_signatures import FunctionalSignatureValidator


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Validate functional signatures against predictive behavior."
    )
    parser.add_argument(
        "--backbone-name",
        type=str,
        default="vit_b16",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
    )

    return parser


def main():
    args = _build_parser().parse_args()

    validator = FunctionalSignatureValidator(
        backbone_name=args.backbone_name,
        verbose=not args.quiet,
    )

    validator.run()


if __name__ == "__main__":
    main()
