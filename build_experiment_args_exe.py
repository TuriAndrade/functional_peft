from dotenv import load_dotenv

load_dotenv()

import os
import json
import shlex
import argparse
from typing import Dict, Any, List
from experiments import build_args


def format_arg_value(value: Any) -> str:
    if (
        isinstance(value, (dict, list, tuple))
        or value is None
        or isinstance(value, bool)
    ):
        return shlex.quote(json.dumps(value))
    return shlex.quote(str(value))


def build_command(exe_prefix: str, args_dict: Dict[str, Any]) -> str:
    parts: List[str] = [exe_prefix]

    args_json = json.dumps(args_dict)
    parts.append("--args-json")
    parts.append(shlex.quote(args_json))

    return " ".join(parts)


def generate_args(
    exp_name: str,
    max_exp_per_file: int | None = None,
    **kwargs: Any,
) -> None:
    args_list, output_path, exe_prefix = build_args[exp_name](**kwargs)

    lines = []

    for config in args_list:
        lines.append(build_command(exe_prefix, config))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if (
        max_exp_per_file is None
        or max_exp_per_file <= 0
        or len(lines) <= max_exp_per_file
    ):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            f.write("\n")
        return

    base, ext = os.path.splitext(output_path)

    for part_idx, start in enumerate(range(0, len(lines), max_exp_per_file), start=1):
        part_output_path = f"{base}_part_{part_idx}{ext}"
        part_lines = lines[start : start + max_exp_per_file]

        with open(part_output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(part_lines))
            f.write("\n")


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
    parser.add_argument("--max-exp-per-file", type=int, default=None)

    args, unknown = parser.parse_known_args()
    kwargs = parse_unknown_kwargs(unknown)
    generate_args(
        exp_name=args.exp_name,
        max_exp_per_file=args.max_exp_per_file,
        **kwargs,
    )


if __name__ == "__main__":
    main()
