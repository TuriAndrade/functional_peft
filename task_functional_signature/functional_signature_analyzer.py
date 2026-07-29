import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS


class FunctionalSignatureAnalyzer:
    def __init__(
        self,
        model_names,
        dataset_names,
        dataset_groups,
        f_sig_paths,
        f_sig_keys,
        save_dir,
    ):
        self.model_names = model_names
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

    def _load_model(self, model_idx):
        """Load all required Fisher families for one model."""
        families = set(f for fs in self.keys.values() for f in fs)
        signatures = {f: [] for f in families}

        for path in self.f_sig_paths[model_idx]:
            with np.load(path) as data:
                for family in families:
                    signatures[family].append(data[f"F__{family}"])

        return signatures

    def _distance_matrix(self, signatures, layer=None):
        """Compute the global or single-layer Task2Vec distance matrix."""
        vectors = [(x if layer is None else x[layer]).reshape(-1) for x in signatures]

        n = len(vectors)
        D = np.zeros((n, n))

        for i in range(n):
            for j in range(i):
                D[i, j] = D[j, i] = self._task2vec_distance(vectors[i], vectors[j])

        return D

    def _compute_distances(self):
        """Compute global and layerwise matrices for every model and key."""
        results = {}

        for model_idx, model_name in enumerate(self.model_names):
            signatures = self._load_model(model_idx)
            results[model_name] = {}

            for key, families in self.keys.items():
                n_layers = signatures[families[0]][0].shape[0]

                results[model_name][key] = {
                    "global": np.mean(
                        [self._distance_matrix(signatures[f]) for f in families],
                        axis=0,
                    ),
                    "layers": np.stack(
                        [
                            np.mean(
                                [
                                    self._distance_matrix(signatures[f], layer)
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
        if np.allclose(D, 0):
            xy = np.zeros((len(D), 2))
        else:
            xy = MDS(
                n_components=2,
                dissimilarity="precomputed",
                random_state=0,
            ).fit_transform(D)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(xy[:, 0], xy[:, 1])

        for i, name in enumerate(self.dataset_names):
            ax.annotate(
                name,
                xy[i],
                xytext=(4, 4),
                textcoords="offset points",
            )

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
        comparisons = [
            (
                1.0
                if D[anchor, a] < D[anchor, b]
                else 0.5 if D[anchor, a] == D[anchor, b] else 0.0
            )
            for a in closer
            for b in farther
        ]

        return float(np.mean(comparisons)) if comparisons else np.nan

    def _group_separation(self, D, group):
        """
        Compute ordinal ranking concordance for one hierarchy.

        dataset_groups[group] is:
            (positive, neg1, neg2, ...)

        For each positive anchor, expected distance ordering is:
            positive < neg1 < neg2 < ...

        All ordered tier pairs receive equal weight.
        """
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

            pair_scores = [
                self._pair_concordance(
                    D,
                    anchor,
                    tiers[i],
                    tiers[j],
                )
                for i in range(len(tiers))
                for j in range(i + 1, len(tiers))
                if tiers[i] and tiers[j]
            ]

            if pair_scores:
                anchor_scores.append(np.mean(pair_scores))

        return float(np.mean(anchor_scores)) if anchor_scores else np.nan

    def _save_separation(self, distances):
        """Save per-signature layerwise results and a global summary."""
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

                with open(out / "separation.csv", "w", newline="") as f:
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

        with open(
            self.save_dir / "separation_global.csv",
            "w",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["model_name", "signature_key"] + groups)
            writer.writerows(global_rows)

    def analyze(self):
        """Compute distances, save MDS plots, and evaluate task separation."""
        distances = self._compute_distances()
        self._save_plots(distances)
        self._save_separation(distances)
