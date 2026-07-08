import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr


class TaskSeparationAnalyzer:
    def __init__(
        self,
        distance_matrix: np.ndarray,
        model_names: Sequence[str],
        dataset_names: Sequence[str],
        dataset_groups: Mapping[str, str],
        save_dir: str | Path,
    ):
        self.D = np.asarray(distance_matrix, dtype=np.float64)
        self.model_names = list(map(str, model_names))
        self.dataset_names = list(map(str, dataset_names))
        self.dataset_groups = {str(k): str(v) for k, v in dataset_groups.items()}
        self.save_dir = Path(save_dir)

        expected_shape = (
            len(self.model_names),
            len(self.model_names),
            len(self.dataset_names),
            len(self.dataset_names),
        )
        if self.D.shape != expected_shape:
            raise ValueError(
                f"distance_matrix has shape {self.D.shape}, expected {expected_shape}"
            )

    def compute(self, n_permutations: int = 999, random_state: int = 0, top_k: int = 3):
        rng = np.random.default_rng(random_state)
        idx, labels, used_names, ignored_names = self._resolve_groups()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        for i, model_name in enumerate(self.model_names):
            D = self._sanitize(self.D[i, i][np.ix_(idx, idx)])
            out = self._metrics(D, labels, used_names, rng, n_permutations, top_k)
            results[model_name] = out["metrics"]

            model_dir = self.save_dir / self._safe_name(model_name)
            self._save_model(model_dir, out)
            self._plot_group_matrix(
                out["group_distance_matrix"],
                out["group_names"],
                f"Mean group distances | {model_name}",
                model_dir / "group_distance_matrix.png",
            )
            self._plot_within_between(
                out["pair_rows"],
                f"Within vs between distances | {model_name}",
                model_dir / "within_between_distances.png",
            )

        self._save_summary(used_names, ignored_names, labels, results)

        return {
            "save_dir": self.save_dir,
            "dataset_names": used_names,
            "ignored_dataset_names": ignored_names,
            "group_labels": labels.tolist(),
            "model_names": self.model_names,
            "metrics": results,
        }

    def _resolve_groups(self):
        idx, labels, used, ignored = [], [], [], []

        for i, name in enumerate(self.dataset_names):
            group = self.dataset_groups.get(name)
            if group is None:
                group = self.dataset_groups.get(self._normalize(name))

            if group is None:
                ignored.append(name)
            else:
                idx.append(i)
                labels.append(group)
                used.append(name)

        if len(idx) < 2 or len(set(labels)) < 2:
            raise ValueError("Need at least two datasets and two groups.")

        return np.array(idx), np.array(labels, dtype=object), used, ignored

    def _metrics(self, D, labels, dataset_names, rng, n_permutations, top_k):
        metrics = {}
        metrics.update(self._within_between(D, labels))
        metrics.update(self._silhouette(D, labels))
        metrics.update(self._anosim(D, labels))
        metrics.update(self._permanova(D, labels))
        metrics.update(self._permdisp(D, labels))
        metrics.update(self._retrieval(D, labels, top_k))
        metrics.update(self._mantel(D, labels))
        metrics.update(self._pvalues(D, labels, metrics, rng, n_permutations, top_k))
        metrics.update(
            {
                "n_datasets": int(len(labels)),
                "n_groups": int(len(set(labels))),
                "top_k": int(min(top_k, len(labels) - 1)),
                "n_permutations": int(n_permutations),
            }
        )

        group_names, group_matrix = self._group_matrix(D, labels)

        return {
            "metrics": metrics,
            "per_dataset_rows": self._per_dataset_rows(D, labels, dataset_names, top_k),
            "pair_rows": self._pair_rows(D, labels, dataset_names),
            "group_names": group_names,
            "group_distance_matrix": group_matrix,
        }

    @staticmethod
    def _within_between(D, labels):
        i, j = np.triu_indices(len(labels), 1)
        d = D[i, j]
        same = labels[i] == labels[j]
        within, between = d[same], d[~same]

        wm = TaskSeparationAnalyzer._mean(within)
        bm = TaskSeparationAnalyzer._mean(between)
        gap = bm - wm

        return {
            "within_mean": wm,
            "within_median": TaskSeparationAnalyzer._median(within),
            "between_mean": bm,
            "between_median": TaskSeparationAnalyzer._median(between),
            "separation_gap": gap,
            "separation_ratio": bm / wm if wm > 0 else np.nan,
            "normalized_separation": gap / bm if bm > 0 else np.nan,
        }

    @classmethod
    def _silhouette(cls, D, labels):
        scores = cls._silhouette_samples(D, labels)
        groups = cls._unique(labels)
        return {
            "silhouette_micro": float(scores.mean()),
            "silhouette_macro": float(
                np.mean([scores[labels == g].mean() for g in groups])
            ),
        }

    @classmethod
    def _silhouette_samples(cls, D, labels):
        groups = cls._unique(labels)
        scores = []

        for i in range(len(labels)):
            same = np.where(labels == labels[i])[0]
            same = same[same != i]

            if len(same) == 0:
                scores.append(0.0)
                continue

            a = D[i, same].mean()
            b = min(D[i, labels == g].mean() for g in groups if g != labels[i])
            scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)

        return np.array(scores, dtype=np.float64)

    @staticmethod
    def _anosim(D, labels):
        n = len(labels)
        i, j = np.triu_indices(n, 1)
        same = labels[i] == labels[j]

        if same.all() or (~same).all():
            return {"anosim_R": np.nan}

        ranks = rankdata(D[i, j])
        r_within = ranks[same].mean()
        r_between = ranks[~same].mean()

        return {"anosim_R": float((r_between - r_within) / (n * (n - 1) / 4.0))}

    @classmethod
    def _permanova(cls, D, labels):
        groups = cls._unique(labels)
        n, g = len(labels), len(groups)

        if g < 2 or n <= g:
            return {"permanova_F": np.nan, "permanova_R2": np.nan}

        B = cls._gower(D)
        X = np.column_stack([(labels == group).astype(float) for group in groups])
        H = X @ np.linalg.pinv(X.T @ X) @ X.T

        ss_total = np.trace(B)
        ss_between = np.trace(H @ B)
        ss_within = ss_total - ss_between

        if ss_total <= 0 or ss_within <= 0:
            return {"permanova_F": np.nan, "permanova_R2": np.nan}

        return {
            "permanova_F": float((ss_between / (g - 1)) / (ss_within / (n - g))),
            "permanova_R2": float(ss_between / ss_total),
        }

    @classmethod
    def _permdisp(cls, D, labels):
        X = cls._pcoa(D)
        z = np.zeros(len(labels), dtype=np.float64)

        for group in cls._unique(labels):
            idx = np.where(labels == group)[0]
            z[idx] = np.linalg.norm(X[idx] - X[idx].mean(axis=0), axis=1)

        return {"permdisp_F": cls._one_way_f(z, labels)}

    @staticmethod
    def _retrieval(D, labels, top_k):
        n = len(labels)
        k = min(top_k, n - 1)
        top1, topk, aps = [], [], []

        for i in range(n):
            order = np.argsort(D[i])
            order = order[order != i]
            same = labels[order] == labels[i]
            relevant = np.where(same)[0]

            top1.append(float(same[0]))
            topk.append(float(same[:k].mean()))
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
        i, j = np.triu_indices(len(labels), 1)
        target = (labels[i] != labels[j]).astype(float)

        if len(np.unique(target)) < 2:
            return {"mantel_spearman": np.nan}

        return {"mantel_spearman": float(spearmanr(D[i, j], target).correlation)}

    def _pvalues(self, D, labels, observed, rng, n_permutations, top_k):
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
            y = rng.permutation(labels)

            for name, fn in tests.items():
                obs, val = observed.get(name, np.nan), fn(y)
                if np.isfinite(obs) and np.isfinite(val):
                    totals[name] += 1
                    counts[name] += int(val >= obs)

        return {
            f"{name}_p": (
                (counts[name] + 1) / (totals[name] + 1) if totals[name] > 0 else np.nan
            )
            for name in tests
        }

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
    def _group_matrix(cls, D, labels):
        groups = cls._unique(labels)
        M = np.full((len(groups), len(groups)), np.nan, dtype=np.float64)

        for i, a in enumerate(groups):
            ia = np.where(labels == a)[0]

            for j, b in enumerate(groups):
                ib = np.where(labels == b)[0]

                if a == b and len(ia) >= 2:
                    sub = D[np.ix_(ia, ia)]
                    r, c = np.triu_indices(len(ia), 1)
                    M[i, j] = sub[r, c].mean()
                elif a != b:
                    M[i, j] = D[np.ix_(ia, ib)].mean()

        return groups, M

    def _save_model(self, model_dir, out):
        self._write_json(model_dir / "metrics.json", out["metrics"])
        self._write_csv(
            model_dir / "per_dataset_diagnostics.csv", out["per_dataset_rows"]
        )
        self._write_csv(model_dir / "pairwise_dataset_distances.csv", out["pair_rows"])

        rows = []
        for i, group in enumerate(out["group_names"]):
            rows.append(
                {
                    "group": group,
                    **{
                        g: out["group_distance_matrix"][i, j]
                        for j, g in enumerate(out["group_names"])
                    },
                }
            )

        self._write_csv(model_dir / "group_distance_matrix.csv", rows)

    def _save_summary(self, dataset_names, ignored_names, labels, results):
        self._write_csv(
            self.save_dir / "summary_metrics.csv",
            [{"model": model, **metrics} for model, metrics in results.items()],
        )
        self._write_json(
            self.save_dir / "summary_metrics.json",
            {
                "dataset_names": dataset_names,
                "ignored_dataset_names": ignored_names,
                "dataset_groups": dict(zip(dataset_names, labels)),
                "model_names": self.model_names,
                "metrics": results,
            },
        )

    @staticmethod
    def _plot_group_matrix(matrix, labels, title, path):
        fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(labels) + 3), 6))
        im = ax.imshow(matrix, aspect="auto")

        ax.set_title(title)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

        for i in range(len(labels)):
            for j in range(len(labels)):
                if np.isfinite(matrix[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.3f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

        fig.colorbar(im, ax=ax, label="Mean distance")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _plot_within_between(pair_rows, title, path):
        within = [r["distance"] for r in pair_rows if r["same_group"]]
        between = [r["distance"] for r in pair_rows if not r["same_group"]]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.boxplot(
            [within if within else [np.nan], between if between else [np.nan]],
            labels=["Within", "Between"],
        )
        ax.set_title(title)
        ax.set_ylabel("Distance")
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _sanitize(D):
        D = np.asarray(D, dtype=np.float64)

        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError(f"Expected square distance matrix, got {D.shape}")
        if not np.all(np.isfinite(D)):
            raise ValueError("Distance matrix contains non-finite values.")

        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)
        return D

    @staticmethod
    def _gower(D):
        n = D.shape[0]
        J = np.eye(n) - np.ones((n, n)) / n
        return -0.5 * J @ (D**2) @ J

    @classmethod
    def _pcoa(cls, D):
        vals, vecs = np.linalg.eigh(cls._gower(cls._sanitize(D)))
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        keep = vals > 1e-12

        if not np.any(keep):
            return np.zeros((D.shape[0], 1), dtype=np.float64)

        return vecs[:, keep] * np.sqrt(vals[keep])

    @classmethod
    def _one_way_f(cls, values, labels):
        groups = cls._unique(labels)

        if len(groups) < 2:
            return np.nan

        grand = values.mean()
        ss_between = 0.0
        ss_within = 0.0

        for group in groups:
            x = values[labels == group]
            ss_between += len(x) * (x.mean() - grand) ** 2
            ss_within += np.sum((x - x.mean()) ** 2)

        df_between = len(groups) - 1
        df_within = len(values) - len(groups)

        if df_within <= 0 or ss_within <= 0:
            return np.nan

        return float((ss_between / df_between) / (ss_within / df_within))

    @staticmethod
    def _normalize(name: str):
        name = str(name)
        return name[5:] if name.startswith("vtab_") else name

    @staticmethod
    def _unique(values):
        return list(dict.fromkeys(map(str, values)))

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
    def _jsonable(cls, x):
        if isinstance(x, dict):
            return {str(k): cls._jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [cls._jsonable(v) for v in x]
        if isinstance(x, np.ndarray):
            return cls._jsonable(x.tolist())
        if isinstance(x, (np.bool_, bool)):
            return bool(x)
        if isinstance(x, (np.integer, int)):
            return int(x)
        if isinstance(x, (np.floating, float)):
            x = float(x)
            return x if np.isfinite(x) else None
        return x

    @classmethod
    def _write_json(cls, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(cls._jsonable(data), f, indent=2)

    @classmethod
    def _write_csv(cls, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)

        if not rows:
            path.write_text("")
            return

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(cls._jsonable(row) for row in rows)
