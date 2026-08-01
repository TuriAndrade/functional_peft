import csv
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS


class FisherSignatureAnalyzer:
    def __init__(
        self,
        model_names,
        model_groups,
        dataset_names,
        dataset_groups,
        f_sig_paths,
        f_sig_keys,
        save_dir,
    ):
        self.model_names = model_names
        self.model_groups = model_groups
        self.dataset_names = dataset_names
        self.dataset_groups = dataset_groups
        self.f_sig_paths = np.asarray(f_sig_paths)
        self.save_dir = Path(save_dir)

        self.keys = {
            key.replace(",", "+"): tuple(x.strip() for x in key.split(","))
            for key in f_sig_keys
        }

    @staticmethod
    def _task2vec_distance(a, b, eps=1e-12):
        """Symmetric Task2Vec distance between two Fisher signatures."""
        d = a + b
        x = np.divide(a, d, out=np.zeros_like(a), where=d != 0)
        y = np.divide(b, d, out=np.zeros_like(b), where=d != 0)

        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        if nx == 0 and ny == 0:
            return 0.0

        return 1.0 - np.dot(x, y) / (nx * ny + eps)

    def _load_signatures(self):
        """Load all required Fisher families for all models."""
        families = set(f for fs in self.keys.values() for f in fs)
        signatures = {}

        for model_idx, model_name in enumerate(self.model_names):
            signatures[model_name] = {f: [] for f in families}

            for path in self.f_sig_paths[model_idx]:
                with np.load(path) as data:
                    for family in families:
                        signatures[model_name][family].append(data[f"F__{family}"])

        return signatures

    def _distance_matrix(self, signatures, layer=None):
        """Compute the global or single-layer Task2Vec distance matrix."""
        vectors = [(x if layer is None else x[layer]).reshape(-1) for x in signatures]

        D = np.zeros((len(vectors), len(vectors)))

        for i in range(len(vectors)):
            for j in range(i):
                D[i, j] = D[j, i] = self._task2vec_distance(
                    vectors[i],
                    vectors[j],
                )

        return D

    def _compute_distances(self, signatures):
        """Compute global and layerwise matrices for every model and key."""
        results = {}

        for model in self.model_names:
            sigs = signatures[model]
            results[model] = {}

            for key, families in self.keys.items():
                n_layers = sigs[families[0]][0].shape[0]

                results[model][key] = {
                    "global": np.mean(
                        [self._distance_matrix(sigs[f]) for f in families],
                        axis=0,
                    ),
                    "layers": np.stack(
                        [
                            np.mean(
                                [
                                    self._distance_matrix(sigs[f], layer)
                                    for f in families
                                ],
                                axis=0,
                            )
                            for layer in range(n_layers)
                        ]
                    ),
                }

        return results

    def _plot(self, D, path):
        """Project a distance matrix to 2D with MDS and label every dataset."""
        xy = (
            np.zeros((len(D), 2))
            if np.allclose(D, 0)
            else MDS(
                n_components=2,
                dissimilarity="precomputed",
                random_state=0,
            ).fit_transform(D)
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(xy[:, 0], xy[:, 1])

        for i, name in enumerate(self.dataset_names):
            ax.annotate(name, xy[i], xytext=(4, 4), textcoords="offset points")

        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def _save_plots(self, distances):
        """Save global and layerwise MDS plots."""
        for model, keys in distances.items():
            for key, matrices in keys.items():
                out = self.save_dir / model / key

                self._plot(matrices["global"], out / "global.png")

                for layer, D in enumerate(matrices["layers"]):
                    self._plot(D, out / f"layer_{layer:02d}.png")

    @staticmethod
    def _pair_concordance(D, anchor, closer, farther):
        """Concordance that closer-tier tasks rank before farther-tier tasks."""
        scores = [
            (
                1.0
                if D[anchor, a] < D[anchor, b]
                else 0.5 if D[anchor, a] == D[anchor, b] else 0.0
            )
            for a in closer
            for b in farther
        ]

        return float(np.mean(scores)) if scores else np.nan

    def _group_separation(self, D, group):
        """Compute ordinal ranking concordance for one dataset hierarchy."""
        index = {name: i for i, name in enumerate(self.dataset_names)}
        hierarchy = [
            [index[name] for name in tier] for tier in self.dataset_groups[group]
        ]

        positive = hierarchy[0]
        anchor_scores = []

        for anchor in positive:
            tiers = [
                [i for i in positive if i != anchor],
                *hierarchy[1:],
            ]

            scores = [
                self._pair_concordance(D, anchor, tiers[i], tiers[j])
                for i in range(len(tiers))
                for j in range(i + 1, len(tiers))
                if tiers[i] and tiers[j]
            ]

            if scores:
                anchor_scores.append(np.mean(scores))

        return float(np.mean(anchor_scores)) if anchor_scores else np.nan

    def _save_separation(self, distances):
        """Save layerwise separation results and global summary."""
        groups = list(self.dataset_groups)
        global_rows = []

        for model, keys in distances.items():
            for key, matrices in keys.items():
                variants = [("global", matrices["global"])] + [
                    (f"layer_{layer:02d}", D)
                    for layer, D in enumerate(matrices["layers"])
                ]

                scores = {
                    group: {
                        name: self._group_separation(D, group) for name, D in variants
                    }
                    for group in groups
                }

                out = self.save_dir / model / key
                out.mkdir(parents=True, exist_ok=True)

                with open(out / "functional_separation.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["group_name"] + [name for name, _ in variants])

                    for group in groups:
                        writer.writerow(
                            [group] + [scores[group][name] for name, _ in variants]
                        )

                global_rows.append(
                    [model, key] + [scores[group]["global"] for group in groups]
                )

        self.save_dir.mkdir(parents=True, exist_ok=True)

        with open(self.save_dir / "functional_separation.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model_name", "signature_key"] + groups)
            writer.writerows(global_rows)

    def _model_distance(self, signatures, models, dataset_idx, families):
        """Mean Fisher distance across model pairs for one dataset."""
        return np.mean(
            [
                np.mean(
                    [
                        self._task2vec_distance(
                            signatures[a][family][dataset_idx].reshape(-1),
                            signatures[b][family][dataset_idx].reshape(-1),
                        )
                        for family in families
                    ]
                )
                for a, b in combinations(models, 2)
            ]
        )

    def _save_r_statistics(self, signatures, distances):
        """
        Save model-group R statistics.

        For dataset t:

            R_t =
                mean distance across models for t
                ----------------------------------
                mean distance from t to other datasets
                under the same models

        Only the first tier of each model group is used.
        """
        rows = []

        for group, hierarchy in self.model_groups.items():
            models = list(hierarchy[0])

            if len(models) < 2:
                raise ValueError(
                    f"Model group '{group}' needs at least two positive models."
                )

            unknown = set(models) - set(self.model_names)
            if unknown:
                raise ValueError(
                    f"Unknown models in group '{group}': {sorted(unknown)}"
                )

            for key, families in self.keys.items():
                values = []

                for dataset_idx in range(len(self.dataset_names)):
                    numerator = self._model_distance(
                        signatures,
                        models,
                        dataset_idx,
                        families,
                    )

                    denominator = np.mean(
                        [
                            np.mean(
                                np.delete(
                                    distances[model][key]["global"][dataset_idx],
                                    dataset_idx,
                                )
                            )
                            for model in models
                        ]
                    )

                    values.append(
                        numerator / denominator if denominator > 0 else np.nan
                    )

                rows.append([group, key, np.nanmean(values), *values])

        self.save_dir.mkdir(parents=True, exist_ok=True)

        with open(self.save_dir / "r_statistic.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "model_group",
                    "signature_key",
                    "global",
                    *self.dataset_names,
                ]
            )
            writer.writerows(rows)

    def analyze(self):
        """Compute distances, plots, task separation, and model-group R."""
        signatures = self._load_signatures()
        distances = self._compute_distances(signatures)

        self._save_plots(distances)
        self._save_separation(distances)
        self._save_r_statistics(signatures, distances)
