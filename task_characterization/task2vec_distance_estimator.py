from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


class Task2VecDistanceEstimator:
    """
    Computes Task2Vec-style distance matrices from FIM representation files saved by
    MonteCarloFIMEstimator.

    Expected dataset directory structure:
        representations_path/
            dataset_1/
                fim_representations.npz
            dataset_2/
                fim_representations.npz
            ...

    Expected .npz structure per dataset:
        linear_probe_name
        prompt_model_names

        raw__{group}
            shape: [1 + n_prompt_models, ...]

        adapt_mag__{group}
        adapt_dir__{group}
        prompt_mag__{group}
        prompt_dir__{group}
            shape: [n_prompt_models, ...]

    Model selection:
        linear_probe_name:
            the baseline model name used only in the raw view.

        prompt_model_names:
            prompt models used in all views.
            raw uses [linear_probe_name, *prompt_model_names].
            delta views use prompt_model_names only.

    Signed delta views are converted to nonnegative vectors before Task2Vec distance:
        x -> concat(max(x, 0), max(-x, 0))

    This preserves both increases and decreases in FIM mass instead of clipping
    negative deltas away.

    Saved output:
        distance_matrix.npz containing:

            dataset_names
            view_names
            rep_types
            linear_probe_name
            prompt_model_names

            raw__model_names
            raw__distance_matrix

            adapt_mag__model_names
            adapt_mag__distance_matrix

            adapt_dir__model_names
            adapt_dir__distance_matrix

            prompt_mag__model_names
            prompt_mag__distance_matrix

            prompt_dir__model_names
            prompt_dir__distance_matrix

        Each distance matrix has shape:
            [n_models_for_view, n_models_for_view, n_datasets, n_datasets]
    """

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

    VIEWS = ("raw", "adapt_mag", "adapt_dir", "prompt_mag", "prompt_dir")

    def __init__(
        self,
        representations_path: str | Path,
        dataset_names: Sequence[str],
        linear_probe_name: str,
        prompt_model_names: Sequence[str],
        save_dir: str | Path,
    ):
        self.root = Path(representations_path)
        self.dataset_names = list(map(str, dataset_names))
        self.linear_probe_name = str(linear_probe_name)
        self.prompt_model_names = list(map(str, prompt_model_names))
        self.save_dir = Path(save_dir)

    def compute(
        self,
        rep_types: Sequence[str] = REP_TYPES,
        views: Sequence[str] = VIEWS,
        eps: float = 1e-12,
        filename: str = "distance_matrix.npz",
    ):
        paths = [self._find_npz(self.root / d) for d in self.dataset_names]
        rep_types = tuple(map(str, rep_types))
        views = tuple(map(str, views))

        names = {
            view: (
                [self.linear_probe_name, *self.prompt_model_names]
                if view == "raw"
                else self.prompt_model_names
            )
            for view in views
        }

        matrices = {
            view: self._build_distance_matrix(
                self._load_embeddings(paths, view, names[view], rep_types),
                eps,
            )
            for view in views
        }

        self.save_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.save_dir / filename

        arrays = {
            "dataset_names": np.array(self.dataset_names),
            "view_names": np.array(views),
            "rep_types": np.array(rep_types),
            "linear_probe_name": np.array(self.linear_probe_name),
            "prompt_model_names": np.array(self.prompt_model_names),
            "distance_method": np.array("task2vec_cosine"),
            "signed_delta_encoding": np.array("positive_negative_split"),
        }

        for view in views:
            arrays[f"{view}__model_names"] = np.array(names[view])
            arrays[f"{view}__distance_matrix"] = matrices[view]

        np.savez_compressed(output_path, **arrays)

        return {
            "output_path": output_path,
            "dataset_names": self.dataset_names,
            "model_names": names,
            "distance_matrices": matrices,
        }

    @staticmethod
    def _find_npz(dataset_dir: Path) -> Path:
        preferred = dataset_dir / "fim_representations.npz"
        if preferred.exists():
            return preferred

        files = sorted(dataset_dir.glob("*.npz"))
        if len(files) != 1:
            raise ValueError(
                f"Expected fim_representations.npz or exactly one .npz in "
                f"{dataset_dir}, found {len(files)}."
            )
        return files[0]

    @staticmethod
    def _meta(data) -> tuple[str, list[str]]:
        return (
            str(np.asarray(data["linear_probe_name"]).item()),
            [str(x) for x in data["prompt_model_names"]],
        )

    def _validate_meta(self, path: Path, lp: str, prompts: Sequence[str]):
        if lp != self.linear_probe_name:
            raise ValueError(
                f"Linear probe mismatch in {path}: expected "
                f"{self.linear_probe_name}, found {lp}."
            )

        missing = [m for m in self.prompt_model_names if m not in prompts]
        if missing:
            raise ValueError(f"Prompt models not found in {path}: {missing}")

    def _load_embeddings(
        self,
        paths: Sequence[Path],
        view: str,
        names: Sequence[str],
        rep_types: Sequence[str],
    ) -> np.ndarray:
        per_dataset = []

        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                lp, prompts = self._meta(data)
                self._validate_meta(path, lp, prompts)

                available = [lp, *prompts] if view == "raw" else prompts
                idx = {name: i for i, name in enumerate(available)}

                missing_keys = [
                    f"{view}__{rep}"
                    for rep in rep_types
                    if f"{view}__{rep}" not in data
                ]
                if missing_keys:
                    raise ValueError(
                        f"Missing representations in {path}: {missing_keys}"
                    )

                per_dataset.append(
                    np.stack(
                        [
                            self._encode(
                                np.concatenate(
                                    [
                                        np.asarray(
                                            data[f"{view}__{rep}"][idx[name]],
                                            dtype=np.float64,
                                        ).reshape(-1)
                                        for rep in rep_types
                                    ]
                                ),
                                signed=view != "raw",
                            )
                            for name in names
                        ],
                        axis=0,
                    )
                )

        return np.stack(per_dataset, axis=1)  # [n_models, n_datasets, dim]

    @staticmethod
    def _encode(x: np.ndarray, signed: bool) -> np.ndarray:
        if signed:
            return np.concatenate([np.maximum(x, 0.0), np.maximum(-x, 0.0)])
        return np.maximum(x, 0.0)

    @classmethod
    def _build_distance_matrix(cls, X: np.ndarray, eps: float) -> np.ndarray:
        n_models, n_datasets, _ = X.shape
        D = np.zeros((n_models, n_models, n_datasets, n_datasets), dtype=np.float64)

        for mi in range(n_models):
            for mj in range(mi, n_models):
                for di in range(n_datasets):
                    for dj in range(n_datasets):
                        d = cls._task2vec_distance(X[mi, di], X[mj, dj], eps)
                        D[mi, mj, di, dj] = d
                        D[mj, mi, dj, di] = d

        for m in range(n_models):
            np.fill_diagonal(D[m, m], 0.0)

        return D

    @staticmethod
    def _task2vec_distance(a: np.ndarray, b: np.ndarray, eps: float) -> float:
        den = a + b

        a = np.divide(a, den, out=np.zeros_like(a), where=den > eps)
        b = np.divide(b, den, out=np.zeros_like(b), where=den > eps)

        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)

        if na <= eps and nb <= eps:
            return 0.0
        if na <= eps or nb <= eps:
            return 1.0

        return float(1.0 - np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
