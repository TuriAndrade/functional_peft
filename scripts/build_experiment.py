from dotenv import load_dotenv

load_dotenv()

import argparse
import json
from src.experiments import experiments


def _parse_value(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_extra_args(args):
    parsed = {}

    i = 0
    while i < len(args):
        key = args[i]

        if not key.startswith("--"):
            raise ValueError(f"Expected argument name, got: {key}")

        key = key[2:].replace("-", "_")

        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            parsed[key] = _parse_value(args[i + 1])
            i += 2
        else:
            parsed[key] = True
            i += 1

    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", required=True, choices=experiments.keys())

    args, extra_args = parser.parse_known_args()
    exp_args = _parse_extra_args(extra_args)

    experiments[args.exp_name](**exp_args)


if __name__ == "__main__":
    main()
