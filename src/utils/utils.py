import json, dataclasses, pathlib, argparse, inspect
from pathlib import Path
from typing import Any
import numpy as np
import torch
import random
import os


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o: Any):
        # --- common structured types ---
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, argparse.Namespace):
            return vars(o)
        if isinstance(o, (pathlib.Path,)):
            return str(o)
        if isinstance(o, (set, frozenset)):
            return list(o)

        # --- numpy ---
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()

        # --- torch small helpers (avoid tensors) ---
        if isinstance(o, torch.Size):
            return list(o)
        if isinstance(o, torch.device):
            return str(o)
        if isinstance(o, torch.dtype):
            return str(o)
        if isinstance(o, torch.nn.Module):
            return o.__class__.__name__

        # --- callables / classes / built-ins ---
        # Don't try inspect.getfile; many built-ins have no file.
        if inspect.isclass(o) or callable(o):
            mod = getattr(o, "__module__", None)
            name = getattr(o, "__qualname__", getattr(o, "__name__", type(o).__name__))
            return f"{mod}.{name}" if mod else name

        # --- last resort: readable string ---
        try:
            return super().default(o)
        except TypeError:
            return repr(o)


def save_config(config: dict, path: str | Path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(config, indent=2, cls=CustomJSONEncoder))


def parse_args_json(parser: argparse.ArgumentParser):
    # First parse only --args-json, without touching the rest yet
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--args-json", type=str, default=None)
    pre_args, remaining_argv = pre_parser.parse_known_args()

    # Mode 1: JSON mode
    if pre_args.args_json is not None:
        json_args = json.loads(pre_args.args_json)

        if not isinstance(json_args, dict):
            raise ValueError("--args-json must decode to a JSON object/dict.")

        # JSON keys must match argparse dest names.
        parser.set_defaults(**json_args)

        # Parse remaining CLI args too, so CLI can still override JSON if desired
        args = parser.parse_args(remaining_argv)

    # Mode 2: normal CLI mode
    else:
        args = parser.parse_args()

    return args
