from dotenv import load_dotenv

load_dotenv()

import json
import argparse
from typing import Dict, Any, List
from experiments import build_results


def parse_cli_value(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return value


def parse_unknown_kwargs(unknown: List[str]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    i = 0

    while i < len(unknown):
        token = unknown[i]

        if not token.startswith("--"):
            raise ValueError(f"Unexpected argument: {token}")

        # Support --key=value
        if "=" in token:
            key, value = token[2:].split("=", 1)
            kwargs[key.replace("-", "_")] = parse_cli_value(value)
            i += 1
            continue

        key = token[2:].replace("-", "_")

        # Support flag-style args: --some-flag
        if i + 1 >= len(unknown) or unknown[i + 1].startswith("--"):
            kwargs[key] = True
            i += 1
            continue

        value = unknown[i + 1]
        kwargs[key] = parse_cli_value(value)
        i += 2

    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)

    args, unknown = parser.parse_known_args()
    kwargs = parse_unknown_kwargs(unknown)
    build_results[args.exp_name](**kwargs)


if __name__ == "__main__":
    main()
