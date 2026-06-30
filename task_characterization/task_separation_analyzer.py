import csv
import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr


class TaskSeparationAnalyzer:
    DELTA_METRICS = [
        "within_mean",
        "between_mean",
        "separation_gap",
        "separation_ratio",
        "normalized_separation",
        "silhouette_micro",
        "silhouette_macro",
        "anosim_R",
        "permanova_R2",
        "permanova_F",
        "permdisp_F",
        "nn_top1_accuracy",
        "nn_topk_purity",
        "mean_average_precision",
        "mantel_spearman",
    ]

    def __init__(
        self,
        distance_matrix_path: str | Path,
        dataset_groups: Mapping[str, str],
        save_dir: str | Path,
    ):
        self.distance_matrix_path = Path(distance_matrix_path)
        self.dataset_groups = {str(k): str(v) for k, v in dataset_groups.items()}
        self.save_dir = Path(save_dir)

    def compute(
        self,
        n_permutations: int = 999,
        random_state: int = 0,
        top_k: int = 3,
    ):
        rng = np.random.default_rng(random_state)

        distance_by_model, model_names, dataset_names = self._load_distance_file()
        indices, labels, used_dataset_names, ignored_dataset_names = (
            self._resolve_dataset_groups(dataset_names)
        )

        self.save_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        for model_idx, model_name in enumerate(model_names):
            D = self._sanitize_distance_matrix(
                distance_by_model[model_idx][np.ix_(indices, indices)]
            )

            result = self._compute_metrics(
                D=D,
                labels=labels,
                dataset_names=used_dataset_names,
                rng=rng,
                n_permutations=n_permutations,
                top_k=top_k,
            )

            results[model_name] = result["metrics"]

            model_dir = self.save_dir / self._safe_name(model_name)
            self._save_model_outputs(
                model_dir=model_dir,
                metrics=result["metrics"],
                per_dataset_rows=result["per_dataset_rows"],
                pair_rows=result["pair_rows"],
                group_names=result["group_names"],
                group_distance_matrix=result["group_distance_matrix"],
            )

            self._plot_group_distance_matrix(
                matrix=result["group_distance_matrix"],
                labels=result["group_names"],
                title=f"Mean group distances | {model_name}",
                path=model_dir / "group_distance_matrix.png",
            )

            self._plot_within_between(
                pair_rows=result["pair_rows"],
                title=f"Within vs between distances | {model_name}",
                path=model_dir / "within_between_distances.png",
            )

        self._save_summary(
            dataset_names=used_dataset_names,
            ignored_dataset_names=ignored_dataset_names,
            labels=labels,
            model_names=model_names,
            results=results,
        )
        self._save_pairwise_deltas(model_names=model_names, results=results)

        return {
            "save_dir": self.save_dir,
            "dataset_names": used_dataset_names,
            "ignored_dataset_names": ignored_dataset_names,
            "group_labels": labels.tolist(),
            "model_names": model_names,
            "metrics": results,
            "distance_by_model": distance_by_model,
        }

    # ------------------------------------------------------------------
    # Loading / groups
    # ------------------------------------------------------------------
    def _load_distance_file(self):
        path = self.distance_matrix_path
        if not path.exists():
            raise FileNotFoundError(f"Missing distance matrix file: {path}")

        if path.suffix == ".npy":
            D = self._sanitize_distance_matrix(np.load(path))
            return D[None], [path.stem], [f"dataset_{i}" for i in range(D.shape[0])]

        if path.suffix != ".npz":
            raise ValueError(f"Expected .npz or .npy, got: {path}")

        with np.load(path, allow_pickle=False) as data:
            if "distance_by_model" in data.files:
                distances = data["distance_by_model"]
            elif "distance_matrix" in data.files:
                distances = data["distance_matrix"]
            elif "D" in data.files:
                distances = data["D"]
            else:
                raise KeyError(
                    f"No distance matrix found. Available keys: {data.files}"
                )

            if distances.ndim == 2:
                distances = distances[None]
            elif distances.ndim != 3:
                raise ValueError(f"Expected 2D or 3D distances, got {distances.shape}")

            n_models, n_datasets, _ = distances.shape

            model_names = (
                [str(x) for x in data["model_names"]]
                if "model_names" in data.files
                else [f"model_{i}" for i in range(n_models)]
            )

            dataset_names = (
                [str(x) for x in data["dataset_names"]]
                if "dataset_names" in data.files
                else [f"dataset_{i}" for i in range(n_datasets)]
            )

        distances = np.stack(
            [self._sanitize_distance_matrix(D) for D in distances],
            axis=0,
        )

        return distances, model_names, dataset_names

    def _resolve_dataset_groups(self, dataset_names: list[str]):
        indices, labels, used_names, ignored_names = [], [], [], []

        for i, name in enumerate(dataset_names):
            group = self.dataset_groups.get(name)

            if group is None:
                group = self.dataset_groups.get(self._normalize_dataset_name(name))

            if group is None:
                ignored_names.append(name)
                continue

            indices.append(i)
            labels.append(group)
            used_names.append(name)

        if len(indices) < 2 or len(set(labels)) < 2:
            raise ValueError(
                "Need at least two datasets and two groups after matching "
                "dataset_names against dataset_groups."
            )

        return (
            np.array(indices, dtype=int),
            np.array(labels, dtype=object),
            used_names,
            ignored_names,
        )

    @staticmethod
    def _normalize_dataset_name(name: str):
        name = str(name)
        return name[5:] if name.startswith("vtab_") else name

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def _compute_metrics(
        self,
        D: np.ndarray,
        labels: np.ndarray,
        dataset_names: list[str],
        rng: np.random.Generator,
        n_permutations: int,
        top_k: int,
    ):
        pair_rows = self._pair_rows(D, labels, dataset_names)

        metrics = {}
        metrics.update(self._within_between(D, labels))
        metrics.update(self._silhouette(D, labels))
        metrics.update(self._anosim(D, labels))
        metrics.update(self._permanova(D, labels))
        metrics.update(self._permdisp(D, labels))
        metrics.update(self._retrieval(D, labels, top_k))
        metrics.update(self._mantel(D, labels))
        metrics.update(
            self._permutation_p_values(
                D=D,
                labels=labels,
                observed=metrics,
                rng=rng,
                n_permutations=n_permutations,
                top_k=top_k,
            )
        )

        metrics["n_datasets"] = int(len(labels))
        metrics["n_groups"] = int(len(set(labels)))
        metrics["top_k"] = int(min(top_k, len(labels) - 1))
        metrics["n_permutations"] = int(n_permutations)

        group_names, group_matrix = self._group_distance_matrix(D, labels)

        return {
            "metrics": metrics,
            "per_dataset_rows": self._per_dataset_rows(D, labels, dataset_names, top_k),
            "pair_rows": pair_rows,
            "group_names": group_names,
            "group_distance_matrix": group_matrix,
        }

    @staticmethod
    def _within_between(D, labels):
        i, j = np.triu_indices(len(labels), k=1)
        distances = D[i, j]
        same = labels[i] == labels[j]

        within = distances[same]
        between = distances[~same]

        within_mean = TaskSeparationAnalyzer._mean(within)
        between_mean = TaskSeparationAnalyzer._mean(between)
        gap = between_mean - within_mean

        return {
            "within_mean": within_mean,
            "within_median": TaskSeparationAnalyzer._median(within),
            "between_mean": between_mean,
            "between_median": TaskSeparationAnalyzer._median(between),
            "separation_gap": gap,
            "separation_ratio": (
                between_mean / within_mean if within_mean > 0 else np.nan
            ),
            "normalized_separation": gap / between_mean if between_mean > 0 else np.nan,
        }

    @staticmethod
    def _silhouette(D, labels):
        scores = TaskSeparationAnalyzer._silhouette_samples(D, labels)
        groups = TaskSeparationAnalyzer._unique(labels)

        return {
            "silhouette_micro": float(np.mean(scores)),
            "silhouette_macro": float(
                np.mean([scores[labels == g].mean() for g in groups])
            ),
        }

    @staticmethod
    def _silhouette_samples(D, labels):
        groups = TaskSeparationAnalyzer._unique(labels)
        scores = []

        for i in range(len(labels)):
            same = np.where(labels == labels[i])[0]
            same = same[same != i]

            if len(same) == 0:
                scores.append(0.0)
                continue

            a = D[i, same].mean()
            b = min(
                D[i, np.where(labels == g)[0]].mean() for g in groups if g != labels[i]
            )

            scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)

        return np.array(scores, dtype=np.float64)

    @staticmethod
    def _anosim(D, labels):
        n = len(labels)
        i, j = np.triu_indices(n, k=1)
        same = labels[i] == labels[j]

        if same.all() or (~same).all():
            return {"anosim_R": np.nan}

        ranks = rankdata(D[i, j])
        r_within = ranks[same].mean()
        r_between = ranks[~same].mean()
        R = (r_between - r_within) / (n * (n - 1) / 4.0)

        return {"anosim_R": float(R)}

    @classmethod
    def _permanova(cls, D, labels):
        labels = np.asarray(labels)
        groups = cls._unique(labels)
        n, g = len(labels), len(groups)

        if g < 2 or n <= g:
            return {"permanova_F": np.nan, "permanova_R2": np.nan}

        B = cls._gower_center(D)
        X = np.column_stack([(labels == group).astype(float) for group in groups])
        H = X @ np.linalg.pinv(X.T @ X) @ X.T

        ss_total = np.trace(B)
        ss_between = np.trace(H @ B)
        ss_within = ss_total - ss_between

        if ss_total <= 0 or ss_within <= 0:
            return {"permanova_F": np.nan, "permanova_R2": np.nan}

        F = (ss_between / (g - 1)) / (ss_within / (n - g))

        return {
            "permanova_F": float(F),
            "permanova_R2": float(ss_between / ss_total),
        }

    @classmethod
    def _permdisp(cls, D, labels):
        coords = cls._pcoa(D)
        distances = np.zeros(len(labels), dtype=np.float64)

        for group in cls._unique(labels):
            idx = np.where(labels == group)[0]
            centroid = coords[idx].mean(axis=0)
            distances[idx] = np.linalg.norm(coords[idx] - centroid, axis=1)

        return {"permdisp_F": cls._one_way_f(distances, labels)}

    @staticmethod
    def _retrieval(D, labels, top_k):
        n = len(labels)
        k = min(top_k, n - 1)

        top1, topk, aps = [], [], []
        for i in range(n):
            order = np.argsort(D[i])
            order = order[order != i]

            same = labels[order] == labels[i]
            top1.append(float(same[0]))
            topk.append(float(same[:k].mean()))

            relevant = np.where(same)[0]
            aps.append(
                np.nan
                if len(relevant) == 0
                else float(np.mean([same[: r + 1].mean() for r in relevant]))
            )

        return {
            "nn_top1_accuracy": float(np.mean(top1)),
            "nn_topk_purity": float(np.mean(topk)),
            "mean_average_precision": TaskSeparationAnalyzer._mean(aps),
        }

    @staticmethod
    def _mantel(D, labels):
        i, j = np.triu_indices(len(labels), k=1)
        semantic = (labels[i] != labels[j]).astype(float)

        if len(np.unique(semantic)) < 2:
            return {"mantel_spearman": np.nan}

        return {"mantel_spearman": float(spearmanr(D[i, j], semantic).correlation)}

    def _permutation_p_values(
        self,
        D,
        labels,
        observed,
        rng,
        n_permutations,
        top_k,
    ):
        tests = {
            "separation_gap": lambda y: self._within_between(D, y)["separation_gap"],
            "silhouette_micro": lambda y: self._silhouette(D, y)["silhouette_micro"],
            "silhouette_macro": lambda y: self._silhouette(D, y)["silhouette_macro"],
            "anosim_R": lambda y: self._anosim(D, y)["anosim_R"],
            "permanova_F": lambda y: self._permanova(D, y)["permanova_F"],
            "permanova_R2": lambda y: self._permanova(D, y)["permanova_R2"],
            "permdisp_F": lambda y: self._permdisp(D, y)["permdisp_F"],
            "nn_top1_accuracy": lambda y: self._retrieval(D, y, top_k)[
                "nn_top1_accuracy"
            ],
            "nn_topk_purity": lambda y: self._retrieval(D, y, top_k)["nn_topk_purity"],
            "mean_average_precision": lambda y: self._retrieval(D, y, top_k)[
                "mean_average_precision"
            ],
            "mantel_spearman": lambda y: self._mantel(D, y)["mantel_spearman"],
        }

        counts = {k: 0 for k in tests}
        totals = {k: 0 for k in tests}

        for _ in range(n_permutations):
            permuted = rng.permutation(labels)

            for name, fn in tests.items():
                obs = observed.get(name, np.nan)
                val = fn(permuted)

                if np.isfinite(obs) and np.isfinite(val):
                    totals[name] += 1
                    counts[name] += int(val >= obs)

        return {
            f"{name}_p": (
                (counts[name] + 1) / (totals[name] + 1) if totals[name] > 0 else np.nan
            )
            for name in tests
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _per_dataset_rows(self, D, labels, dataset_names, top_k):
        k = min(top_k, len(labels) - 1)
        silhouettes = self._silhouette_samples(D, labels)
        rows = []

        for i, name in enumerate(dataset_names):
            order = np.argsort(D[i])
            order = order[order != i]
            same = labels[order] == labels[i]
            nearest = order[0]
            relevant = np.where(same)[0]

            rows.append(
                {
                    "dataset": name,
                    "group": labels[i],
                    "silhouette": silhouettes[i],
                    "nearest_dataset": dataset_names[nearest],
                    "nearest_group": labels[nearest],
                    "nearest_distance": D[i, nearest],
                    "nearest_same_group": labels[nearest] == labels[i],
                    f"top{k}_purity": same[:k].mean(),
                    "average_precision": (
                        np.nan
                        if len(relevant) == 0
                        else np.mean([same[: r + 1].mean() for r in relevant])
                    ),
                }
            )

        return rows

    @staticmethod
    def _pair_rows(D, labels, dataset_names):
        return [
            {
                "dataset_a": dataset_names[i],
                "dataset_b": dataset_names[j],
                "group_a": labels[i],
                "group_b": labels[j],
                "same_group": labels[i] == labels[j],
                "distance": D[i, j],
            }
            for i in range(len(labels))
            for j in range(i + 1, len(labels))
        ]

    @classmethod
    def _group_distance_matrix(cls, D, labels):
        groups = cls._unique(labels)
        M = np.full((len(groups), len(groups)), np.nan, dtype=np.float64)

        for i, ga in enumerate(groups):
            ia = np.where(labels == ga)[0]

            for j, gb in enumerate(groups):
                ib = np.where(labels == gb)[0]

                if ga == gb:
                    if len(ia) >= 2:
                        sub = D[np.ix_(ia, ia)]
                        r, c = np.triu_indices(len(ia), k=1)
                        M[i, j] = sub[r, c].mean()
                else:
                    M[i, j] = D[np.ix_(ia, ib)].mean()

        return groups, M

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def _save_model_outputs(
        self,
        model_dir,
        metrics,
        per_dataset_rows,
        pair_rows,
        group_names,
        group_distance_matrix,
    ):
        self._write_json(model_dir / "metrics.json", metrics)
        self._write_csv(model_dir / "per_dataset_diagnostics.csv", per_dataset_rows)
        self._write_csv(model_dir / "pairwise_dataset_distances.csv", pair_rows)

        rows = []
        for i, group in enumerate(group_names):
            row = {"group": group}
            row.update(
                {g: group_distance_matrix[i, j] for j, g in enumerate(group_names)}
            )
            rows.append(row)

        self._write_csv(model_dir / "group_distance_matrix.csv", rows)

    def _save_summary(
        self,
        dataset_names,
        ignored_dataset_names,
        labels,
        model_names,
        results,
    ):
        rows = [{"model": model, **metrics} for model, metrics in results.items()]

        self._write_csv(self.save_dir / "summary_metrics.csv", rows)
        self._write_json(
            self.save_dir / "summary_metrics.json",
            {
                "dataset_names": dataset_names,
                "ignored_dataset_names": ignored_dataset_names,
                "dataset_groups": {
                    name: group for name, group in zip(dataset_names, labels)
                },
                "model_names": model_names,
                "metrics": results,
            },
        )

    def _save_pairwise_deltas(self, model_names, results):
        out = self.save_dir / "pairwise_deltas"
        out.mkdir(parents=True, exist_ok=True)

        delta_data = {}
        for metric in self.DELTA_METRICS:
            values = np.array(
                [results[model].get(metric, np.nan) for model in model_names],
                dtype=np.float64,
            )

            if not np.all(np.isfinite(values)):
                continue

            delta = values[:, None] - values[None, :]
            delta_data[metric] = delta

            rows = []
            for i, row_model in enumerate(model_names):
                rows.append(
                    {
                        "row_minus_column": row_model,
                        **{
                            col_model: delta[i, j]
                            for j, col_model in enumerate(model_names)
                        },
                    }
                )

            self._write_csv(out / f"{metric}_delta.csv", rows)
            self._plot_delta_matrix(
                delta=delta,
                labels=model_names,
                metric=metric,
                path=out / f"{metric}_delta.png",
            )

        np.savez_compressed(
            out / "pairwise_delta_matrices.npz",
            model_names=np.array(model_names),
            **delta_data,
        )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    @staticmethod
    def _plot_group_distance_matrix(matrix, labels, title, path):
        fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(labels) + 3), 6))
        im = ax.imshow(matrix, aspect="auto")

        ax.set_title(title)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

        for i in range(len(labels)):
            for j in range(len(labels)):
                value = matrix[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8)

        fig.colorbar(im, ax=ax, label="Mean distance")
        fig.tight_layout()

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _plot_within_between(pair_rows, title, path):
        within = [row["distance"] for row in pair_rows if row["same_group"]]
        between = [row["distance"] for row in pair_rows if not row["same_group"]]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.boxplot([within, between], labels=["Within", "Between"])
        ax.set_title(title)
        ax.set_ylabel("Distance")
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _plot_delta_matrix(delta, labels, metric, path):
        fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(labels) + 3), 6))
        im = ax.imshow(delta, aspect="auto")

        ax.set_title(f"Pairwise metric delta | {metric}")
        ax.set_xlabel("Column model")
        ax.set_ylabel("Row model")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(
                    j, i, f"{delta[i, j]:.3f}", ha="center", va="center", fontsize=8
                )

        fig.colorbar(im, ax=ax, label="Row - column")
        fig.tight_layout()

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Math / IO helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_distance_matrix(D):
        D = np.asarray(D, dtype=np.float64)

        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError(f"Expected square distance matrix, got {D.shape}")

        if not np.all(np.isfinite(D)):
            raise ValueError("Distance matrix contains non-finite values.")

        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)
        return D

    @staticmethod
    def _gower_center(D):
        n = D.shape[0]
        J = np.eye(n) - np.ones((n, n)) / n
        return -0.5 * J @ (D**2) @ J

    @classmethod
    def _pcoa(cls, D):
        B = cls._gower_center(cls._sanitize_distance_matrix(D))
        eigvals, eigvecs = np.linalg.eigh(B)

        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        keep = eigvals > 1e-12
        if not np.any(keep):
            return np.zeros((D.shape[0], 1), dtype=np.float64)

        return eigvecs[:, keep] * np.sqrt(eigvals[keep])

    @classmethod
    def _one_way_f(cls, values, labels):
        groups = cls._unique(labels)

        if len(groups) < 2:
            return np.nan

        grand = values.mean()
        ss_between = 0.0
        ss_within = 0.0

        for group in groups:
            vals = values[labels == group]
            mean = vals.mean()
            ss_between += len(vals) * (mean - grand) ** 2
            ss_within += np.sum((vals - mean) ** 2)

        df_between = len(groups) - 1
        df_within = len(values) - len(groups)

        if df_within <= 0 or ss_within <= 0:
            return np.nan

        return float((ss_between / df_between) / (ss_within / df_within))

    @staticmethod
    def _unique(values):
        return list(dict.fromkeys([str(v) for v in values]))

    @staticmethod
    def _mean(values):
        values = np.asarray(values, dtype=np.float64)
        return float(np.nanmean(values)) if np.any(np.isfinite(values)) else np.nan

    @staticmethod
    def _median(values):
        values = np.asarray(values, dtype=np.float64)
        return float(np.nanmedian(values)) if np.any(np.isfinite(values)) else np.nan

    @staticmethod
    def _safe_name(name):
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))

    @classmethod
    def _jsonable(cls, value):
        if isinstance(value, dict):
            return {str(k): cls._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._jsonable(v) for v in value]
        if isinstance(value, np.ndarray):
            return cls._jsonable(value.tolist())
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            value = float(value)
            return value if np.isfinite(value) else None
        return value

    @classmethod
    def _write_json(cls, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(cls._jsonable(data), f, indent=2)

    @classmethod
    def _write_csv(cls, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)

        if len(rows) == 0:
            path.write_text("")
            return

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(cls._jsonable(row))
