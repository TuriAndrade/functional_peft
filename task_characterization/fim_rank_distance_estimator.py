from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


class FIMRankDistanceEstimator:
    """
    Computes weighted Spearman distances between FIM coordinate rankings saved by
    FIMRankEstimator.

    Expected dataset directory structure:
        rankings_path/
            dataset_1/
                fim_rankings.npz
            dataset_2/
                fim_rankings.npz
            ...

    Expected .npz structure per dataset:
        linear_probe_name
        prompt_model_names

        {view}__model_names
        {view}__{rep}__ranks

        For delta views:
            {view}__{rep}__abs_ranks

    Ranking comparison:
        raw and *_abs:
            emphasize agreement among the highest-ranked coordinates.

        signed delta views:
            emphasize both extremes:
                top    = strongest positive changes
                bottom = strongest negative changes.

    rank_weight_power:
        1:
            ordinary Spearman distance.

        > 1:
            increasingly emphasizes the relevant ranking extremes.

    For multiple representations, distances are computed independently and averaged
    equally so that large representations do not dominate smaller ones.

    Distance:
        d = (1 - correlation) / 2

        0.0 = identical rankings
        0.5 = unrelated rankings
        1.0 = reversed rankings

    Saved output:
        distance_matrix.npz containing:

            dataset_names
            view_names
            rep_types
            linear_probe_name
            prompt_model_names
            rank_weight_power
            distance_method

            {view}__model_names
            {view}__distance_matrix

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

    VIEWS = (
        "raw",
        "adapt_mag",
        "adapt_mag_abs",
        "adapt_dir",
        "adapt_dir_abs",
        "prompt_mag",
        "prompt_mag_abs",
        "prompt_dir",
        "prompt_dir_abs",
    )

    def __init__(
        self,
        rankings_path: str | Path,
        dataset_names: Sequence[str],
        linear_probe_name: str,
        prompt_model_names: Sequence[str],
        save_dir: str | Path,
    ):
        self.root = Path(rankings_path)
        self.dataset_names = list(map(str, dataset_names))
        self.linear_probe_name = str(linear_probe_name)
        self.prompt_model_names = list(map(str, prompt_model_names))
        self.save_dir = Path(save_dir)

    def compute(
        self,
        rep_types: Sequence[str] = REP_TYPES,
        views: Sequence[str] = VIEWS,
        rank_weight_power: int = 1,
        filename: str = "distance_matrix.npz",
    ):
        if rank_weight_power < 1 or int(rank_weight_power) != rank_weight_power:
            raise ValueError("rank_weight_power must be an integer >= 1.")

        paths = [self._find_npz(self.root / d) for d in self.dataset_names]
        rep_types = tuple(map(str, rep_types))
        views = tuple(map(str, views))
        rank_weight_power = int(rank_weight_power)

        if not rep_types:
            raise ValueError("At least one representation is required.")

        names = {
            view: (
                [self.linear_probe_name, *self.prompt_model_names]
                if view == "raw"
                else self.prompt_model_names
            )
            for view in views
        }

        matrices = {
            view: np.mean(
                [
                    self._distance_matrix(
                        self._load_ranks(paths, view, rep, names[view]),
                        rank_weight_power,
                        two_sided=view != "raw" and not view.endswith("_abs"),
                    )
                    for rep in rep_types
                ],
                axis=0,
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
            "rank_weight_power": np.array(rank_weight_power),
            "distance_method": np.array("weighted_spearman"),
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
        preferred = dataset_dir / "fim_rankings.npz"
        if preferred.exists():
            return preferred

        files = sorted(dataset_dir.glob("*.npz"))
        if len(files) != 1:
            raise ValueError(
                f"Expected fim_rankings.npz or exactly one .npz in "
                f"{dataset_dir}, found {len(files)}."
            )

        return files[0]

    def _load_ranks(
        self,
        paths: Sequence[Path],
        view: str,
        rep: str,
        names: Sequence[str],
    ) -> np.ndarray:
        source_view, suffix = self._source(view)
        key = f"{source_view}__{rep}__{suffix}"
        datasets = []

        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                lp = str(data["linear_probe_name"].item())
                prompts = [str(x) for x in data["prompt_model_names"]]

                if lp != self.linear_probe_name:
                    raise ValueError(
                        f"Linear probe mismatch in {path}: expected "
                        f"{self.linear_probe_name}, found {lp}."
                    )

                missing = [
                    name for name in self.prompt_model_names if name not in prompts
                ]
                if missing:
                    raise ValueError(f"Prompt models not found in {path}: {missing}")

                available = [str(x) for x in data[f"{source_view}__model_names"]]
                index = {name: i for i, name in enumerate(available)}

                datasets.append(
                    np.asarray(
                        data[key][[index[name] for name in names]],
                        dtype=np.float64,
                    )
                )

        return np.stack(datasets, axis=1)

    @staticmethod
    def _source(view: str) -> tuple[str, str]:
        if view.endswith("_abs"):
            return view.removesuffix("_abs"), "abs_ranks"
        return view, "ranks"

    @classmethod
    def _distance_matrix(
        cls,
        ranks: np.ndarray,
        power: int,
        two_sided: bool,
    ) -> np.ndarray:
        n_models, n_datasets, n_coordinates = ranks.shape
        x = ranks.reshape(-1, n_coordinates)

        correlation = (
            cls._spearman(x)
            if power == 1
            else cls._weighted_spearman(x, power, two_sided)
        )

        np.fill_diagonal(correlation, 1.0)
        distance = (1.0 - np.clip(correlation, -1.0, 1.0)) / 2.0

        return distance.reshape(
            n_models,
            n_datasets,
            n_models,
            n_datasets,
        ).transpose(0, 2, 1, 3)

    @staticmethod
    def _spearman(ranks: np.ndarray) -> np.ndarray:
        x = ranks - ranks.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)
        return x @ x.T

    @staticmethod
    def _weighted_spearman(
        ranks: np.ndarray,
        power: int,
        two_sided: bool,
    ) -> np.ndarray:
        n = ranks.shape[1]
        if n < 2:
            return np.ones((len(ranks), len(ranks)))

        i = np.arange(1, n + 1, dtype=np.float64)
        k_p = np.sum(i**power)
        k_p1 = np.sum(i ** (power + 1))
        denominator = 2 * k_p1 - (n + 1) * k_p

        weights = (n + 1 - ranks) ** power
        cross = weights @ ranks.T
        top = ((n + 1) * k_p - cross - cross.T) / denominator

        if not two_sided:
            return top

        i = np.arange(1, n, dtype=np.float64)
        k_p = np.sum(i**power)
        k_p1 = np.sum(i ** (power + 1))
        denominator = 2 * k_p1 - (n - 1) * k_p

        weights = (ranks - 1) ** power
        cross = weights @ ranks.T
        bottom = (-(n + 1) * k_p + cross + cross.T) / denominator

        return (top + bottom) / 2
