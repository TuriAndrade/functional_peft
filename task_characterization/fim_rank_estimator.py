from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


class FIMRankEstimator:
    """
    Ranks every coordinate of every FIM representation.

    Saved arrays use the original flattened coordinate order:
        ranks[model, coordinate] = rank

    Rank 1 corresponds to:
        - the largest value for ordinary rankings;
        - the largest absolute value for absolute delta rankings.

    Exact ties are broken deterministically by flattened coordinate order, producing
    strict ranks from 1 to n as required by the weighted rank-distance estimator.

    Saves:
        fim_rankings.npz
        top_bottom_rankings.csv
    """

    VIEWS = ("raw", "adapt_dir", "adapt_mag", "prompt_dir", "prompt_mag")
    DELTA_VIEWS = ("adapt_dir", "adapt_mag", "prompt_dir", "prompt_mag")

    REP_TYPES = (
        "input_patch_channel",
        "input_pos_grid",
        "block_path",
        "attention_circuit_head",
        "mlp_memory_group",
        "residual_io_channel",
        "attention_circuit_dim",
        "mlp_memory_unit",
    )

    def __init__(self, representations_path: str | Path, save_dir: str | Path):
        self.path = self._resolve_npz(Path(representations_path))
        self.save_dir = Path(save_dir)

        with np.load(self.path, allow_pickle=False) as data:
            self.linear_probe_name = str(data["linear_probe_name"].item())
            self.prompt_model_names = [str(x) for x in data["prompt_model_names"]]

    @staticmethod
    def _resolve_npz(path: Path) -> Path:
        if path.is_file():
            return path

        preferred = path / "fim_representations.npz"
        if preferred.exists():
            return preferred

        files = sorted(path.glob("*.npz"))
        if len(files) != 1:
            raise ValueError(
                f"Expected fim_representations.npz or exactly one .npz in "
                f"{path}, found {len(files)}."
            )
        return files[0]

    def compute(self, top_k: int = 20, filename: str = "fim_rankings.npz"):
        arrays = {
            "linear_probe_name": np.array(self.linear_probe_name),
            "prompt_model_names": np.array(self.prompt_model_names),
            "view_names": np.array(self.VIEWS),
            "rep_types": np.array(self.REP_TYPES),
            "rank_method": np.array("ordinal_stable"),
            "rank_order": np.array("descending"),
        }
        rows = []

        with np.load(self.path, allow_pickle=False) as data:
            for view in self.VIEWS:
                names = (
                    [self.linear_probe_name, *self.prompt_model_names]
                    if view == "raw"
                    else self.prompt_model_names
                )
                arrays[f"{view}__model_names"] = np.array(names)

                for rep in self.REP_TYPES:
                    x = np.asarray(data[f"{view}__{rep}"], dtype=np.float32)
                    shape = x.shape[1:]
                    flat = x.reshape(len(names), -1)
                    prefix = f"{view}__{rep}"

                    arrays[f"{prefix}__ranks"] = self._rank(-flat)
                    rows.extend(self._rows(view, rep, names, flat, shape, top_k))

                    if view in self.DELTA_VIEWS:
                        arrays[f"{prefix}__abs_ranks"] = self._rank(-np.abs(flat))

        self.save_dir.mkdir(parents=True, exist_ok=True)

        output_path = self.save_dir / filename
        table_path = self.save_dir / "top_bottom_rankings.csv"

        np.savez_compressed(output_path, **arrays)
        self._write_csv(table_path, rows)

        return {
            "output_path": output_path,
            "table_path": table_path,
        }

    @staticmethod
    def _rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, axis=1, kind="stable")
        ranks = np.empty_like(order, dtype=np.int32)
        positions = np.broadcast_to(
            np.arange(1, x.shape[1] + 1, dtype=np.int32),
            order.shape,
        )
        np.put_along_axis(ranks, order, positions, axis=1)
        return ranks

    def _rows(
        self,
        view: str,
        rep: str,
        names: list[str],
        flat: np.ndarray,
        shape: tuple[int, ...],
        top_k: int,
    ):
        rows = []
        k = min(top_k, flat.shape[1])

        for model_idx, model in enumerate(names):
            values = flat[model_idx]
            order = np.argsort(-values, kind="stable")

            selections = [
                ("value", "top", order[:k]),
                ("value", "bottom", order[-1 : -k - 1 : -1]),
            ]

            if view in self.DELTA_VIEWS:
                abs_order = np.argsort(-np.abs(values), kind="stable")
                selections.append(("absolute", "top", abs_order[:k]))

            for ranking, side, indices in selections:
                for rank, index in enumerate(indices, 1):
                    rows.append(
                        {
                            "view": view,
                            "representation": rep,
                            "model": model,
                            "ranking": ranking,
                            "side": side,
                            "rank": rank,
                            "value": float(values[index]),
                            "coordinate": self._coordinate(rep, shape, int(index)),
                        }
                    )

        return rows

    @staticmethod
    def _coordinate(rep: str, shape: tuple[int, ...], index: int) -> str:
        idx = np.unravel_index(index, shape)

        if rep == "input_patch_channel":
            return f"channel={idx[0]}"

        if rep == "input_pos_grid":
            return f"row={idx[0]}, col={idx[1]}"

        if rep == "block_path":
            path = ("LN1", "QK", "OV", "LN2", "MLP key", "MLP value")[idx[1]]
            return f"block={idx[0]}, path={path}"

        if rep == "attention_circuit_head":
            circuit = ("QK", "OV")[idx[2]]
            return f"block={idx[0]}, head={idx[1]}, circuit={circuit}"

        if rep == "mlp_memory_group":
            role = ("key", "value")[idx[2]]
            return f"block={idx[0]}, group={idx[1]}, role={role}"

        if rep == "residual_io_channel":
            role = ("read", "write")[idx[1]]
            return f"block={idx[0]}, role={role}, channel={idx[2]}"

        if rep == "attention_circuit_dim":
            circuit = ("QK", "OV")[idx[2]]
            return f"block={idx[0]}, head={idx[1]}, " f"circuit={circuit}, dim={idx[3]}"

        if rep == "mlp_memory_unit":
            role = ("key", "value")[idx[2]]
            return f"block={idx[0]}, unit={idx[1]}, role={role}"

        raise ValueError(f"Unknown representation: {rep}")

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]):
        fields = (
            "view",
            "representation",
            "model",
            "ranking",
            "side",
            "rank",
            "value",
            "coordinate",
        )

        with open(path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
