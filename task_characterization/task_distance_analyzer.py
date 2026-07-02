import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import (
    linkage,
    optimal_leaf_ordering,
    leaves_list,
    fcluster,
)
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS


class TaskDistanceAnalyzer:
    def __init__(
        self,
        distance_matrix: np.ndarray,
        model_names: Sequence[str],
        dataset_names: Sequence[str],
        save_dir: str | Path,
    ):
        self.D = np.asarray(distance_matrix, dtype=np.float64)
        self.model_names = list(map(str, model_names))
        self.dataset_names = list(map(str, dataset_names))
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

    def compute(
        self,
        linkage_method: str = "average",
        cluster_ks: Sequence[int] = (2, 3, 4, 5),
        eps: float = 1e-12,
    ):
        self.save_dir.mkdir(parents=True, exist_ok=True)

        within_model_D = np.stack(
            [self._sanitize(self.D[i, i], eps) for i in range(len(self.model_names))]
        )

        clustering = self._cluster_all(within_model_D, linkage_method, cluster_ks, eps)
        mds_coords = np.stack([self._mds(D, eps) for D in within_model_D])

        self._save_outputs(clustering, mds_coords, linkage_method, cluster_ks)
        self._plot_per_model_heatmaps(within_model_D, clustering["leaf_orders"])
        self._plot_2d_maps(mds_coords, clustering["labels_by_model"], cluster_ks)
        self._plot_cross_model_heatmaps(clustering["leaf_orders"])

        return {
            "output_dir": self.save_dir,
            "model_names": self.model_names,
            "dataset_names": self.dataset_names,
            "linkage_by_model": clustering["linkages"],
            "leaf_order_by_model": clustering["leaf_orders"],
            "cluster_labels_by_model": clustering["labels_by_model"],
            "mds_coordinates_by_model": mds_coords,
        }

    def _cluster_all(self, distance_by_model, linkage_method, cluster_ks, eps):
        linkages, leaf_orders, labels_by_model, summary = [], [], [], {}

        for name, D in zip(self.model_names, distance_by_model):
            Z, order, labels = self._cluster(D, linkage_method, cluster_ks, eps)
            linkages.append(Z)
            leaf_orders.append(order)
            labels_by_model.append(labels)

            summary[name] = {
                "leaf_order": order.astype(int).tolist(),
                "ordered_dataset_names": [self.dataset_names[i] for i in order],
                "cluster_labels_by_k": labels,
            }

        return {
            "linkages": np.stack(linkages),
            "leaf_orders": np.stack(leaf_orders),
            "labels_by_model": labels_by_model,
            "summary": summary,
        }

    @classmethod
    def _cluster(cls, D, linkage_method, cluster_ks, eps):
        D = cls._sanitize(D, eps)
        condensed = squareform(D, checks=False)
        Z = optimal_leaf_ordering(linkage(condensed, method=linkage_method), condensed)
        order = leaves_list(Z)

        labels = {
            str(k): fcluster(Z, t=k, criterion="maxclust").astype(int).tolist()
            for k in cluster_ks
        }

        return Z, order, labels

    @classmethod
    def _mds(cls, D, eps):
        return MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=0,
            n_init=8,
        ).fit_transform(cls._sanitize(D, eps))

    @staticmethod
    def _sanitize(D, eps=1e-12):
        D = np.asarray(D, dtype=np.float64)
        D = 0.5 * (D + D.T)
        D[np.abs(D) < eps] = 0.0
        D[D < 0.0] = 0.0
        np.fill_diagonal(D, 0.0)
        return D

    def _save_outputs(self, clustering, mds_coords, linkage_method, cluster_ks):
        np.savez_compressed(
            self.save_dir / "clustering_outputs.npz",
            dataset_names=np.array(self.dataset_names),
            model_names=np.array(self.model_names),
            linkage_by_model=clustering["linkages"],
            leaf_order_by_model=clustering["leaf_orders"],
            mds_coordinates_by_model=mds_coords,
        )

        with open(self.save_dir / "clustering_summary.json", "w") as f:
            json.dump(
                {
                    "distance_matrix": "precomputed",
                    "distance_matrix_shape": list(self.D.shape),
                    "linkage_method": linkage_method,
                    "cluster_ks": list(cluster_ks),
                    "dataset_names": self.dataset_names,
                    "model_names": self.model_names,
                    "mds_coordinates_by_model_shape": list(mds_coords.shape),
                    "per_model_clustering": clustering["summary"],
                },
                f,
                indent=2,
            )

    def _plot_per_model_heatmaps(self, distance_by_model, leaf_orders):
        out = self.save_dir / "per_model"

        for i, name in enumerate(self.model_names):
            order = leaf_orders[i]
            labels = [self.dataset_names[j] for j in order]
            D = distance_by_model[i][np.ix_(order, order)]

            self._plot_heatmap(
                D,
                labels,
                labels,
                f"Dataset distances | {name}",
                out / f"{self._safe_name(name)}_clustered_distance_matrix.png",
            )

    def _plot_2d_maps(self, mds_coords, labels_by_model, cluster_ks):
        out = self.save_dir / "cluster_2d"

        for i, name in enumerate(self.model_names):
            safe = self._safe_name(name)

            self._plot_2d_points(
                mds_coords[i],
                self.dataset_names,
                f"2D task map | {name}",
                out / f"{safe}_points_2d.png",
            )

            for k in cluster_ks:
                self._plot_2d_points(
                    mds_coords[i],
                    self.dataset_names,
                    f"2D task map | {name} | k={k}",
                    out / f"{safe}_k{k}_cluster_2d.png",
                    cluster_labels=labels_by_model[i][str(k)],
                )

    def _plot_cross_model_heatmaps(self, leaf_orders):
        out = self.save_dir / "cross_model"

        for i, row_name in enumerate(self.model_names):
            row_order = leaf_orders[i]
            row_labels = [self.dataset_names[j] for j in row_order]

            for j, col_name in enumerate(self.model_names):
                if i == j:
                    continue

                col_order = leaf_orders[j]
                col_labels = [self.dataset_names[k] for k in col_order]
                raw = np.asarray(self.D[i, j], dtype=np.float64)

                self._plot_heatmap(
                    raw[np.ix_(row_order, col_order)],
                    row_labels,
                    col_labels,
                    f"Cross-model dataset distances | {row_name} vs {col_name}",
                    out
                    / f"{self._safe_name(row_name)}_vs_{self._safe_name(col_name)}.png",
                    side_values=np.diag(raw),
                    side_labels=self.dataset_names,
                )

    @staticmethod
    def _plot_heatmap(
        matrix,
        row_labels,
        col_labels,
        title,
        path,
        side_values=None,
        side_labels=None,
    ):
        has_side = side_values is not None and side_labels is not None
        fig_w = max(8.0, 0.45 * len(col_labels) + 3.0)
        fig_h = max(6.0, 0.35 * len(row_labels) + 2.5)

        if has_side:
            fig, (ax, side_ax) = plt.subplots(
                1,
                2,
                figsize=(fig_w + 4.0, fig_h),
                gridspec_kw={"width_ratios": [3.2, 1.4]},
            )
        else:
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            side_ax = None

        im = ax.imshow(matrix, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels, rotation=90)
        ax.set_yticklabels(row_labels)
        fig.colorbar(im, ax=ax, label="Distance", fraction=0.046, pad=0.04)

        if has_side:
            side_values = np.asarray(side_values, dtype=np.float64)
            order = np.argsort(side_values)
            y = np.arange(len(order))

            side_ax.barh(y, side_values[order])
            side_ax.set_title("Same-dataset\ncross-model distance")
            side_ax.set_xlabel("Distance")
            side_ax.set_yticks(y)
            side_ax.set_yticklabels([side_labels[i] for i in order])
            side_ax.invert_yaxis()
            side_ax.grid(axis="x", alpha=0.3)

        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _plot_2d_points(coords, point_labels, title, path, cluster_labels=None):
        coords = np.asarray(coords, dtype=np.float64)
        fig, ax = plt.subplots(figsize=(9.5, 7.5))

        if cluster_labels is None:
            ax.scatter(coords[:, 0], coords[:, 1], s=70)
        else:
            cluster_labels = np.asarray(cluster_labels)
            for c in sorted(np.unique(cluster_labels)):
                mask = cluster_labels == c
                ax.scatter(coords[mask, 0], coords[mask, 1], s=70, label=f"Cluster {c}")
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)

        for x, y, label in zip(coords[:, 0], coords[:, 1], point_labels):
            ax.annotate(
                label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=9
            )

        ax.set_title(title)
        ax.set_xlabel("MDS 1")
        ax.set_ylabel("MDS 2")
        ax.grid(alpha=0.3)

        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _safe_name(name):
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))
