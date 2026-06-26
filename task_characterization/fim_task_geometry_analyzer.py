import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from scipy.cluster.hierarchy import (
    linkage,
    optimal_leaf_ordering,
    leaves_list,
    fcluster,
)
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS


class FIMTaskGeometryAnalyzer:
    REP_SPECS = {
        "block": ("fim_by_transformer_block", None),
        "block_attn": ("fim_by_attention_block_head_projection", None),
        "block_attn_q": ("fim_by_attention_block_head_projection", 0),
        "block_attn_k": ("fim_by_attention_block_head_projection", 1),
        "block_attn_v": ("fim_by_attention_block_head_projection", 2),
        "block_attn_o": ("fim_by_attention_block_head_projection", 3),
        "block_mlp": ("fim_by_mlp_block", None),
        "patch_embd": ("fim_by_patch_embedding_channel", None),
    }

    def __init__(
        self,
        dataset_fim_paths: Sequence[str | Path],
        dataset_names: Sequence[str],
        save_dir: str | Path,
        npz_filename: str = "fim_representations.npz",
    ):
        self.dataset_fim_paths = [Path(path) for path in dataset_fim_paths]
        self.dataset_names = [str(name) for name in dataset_names]
        self.save_dir = Path(save_dir)
        self.npz_filename = npz_filename

    def analyze(
        self,
        rep_types: str | Sequence[str] = "all",
        linkage_method: str = "average",
        cluster_ks: Sequence[int] = (2, 3, 4, 5),
        eps: float = 1e-12,
    ):
        rep_keys = self._parse_rep_types(rep_types)

        embedding_vectors, model_names = self._load_embeddings(rep_keys)
        distance_by_model, cross_model_distance = self._compute_distances(
            embedding_vectors,
            eps=eps,
        )

        (
            linkage_by_model,
            leaf_order_by_model,
            cluster_labels_by_model,
            clustering_summary,
        ) = self._cluster_models(
            distance_by_model,
            model_names,
            linkage_method=linkage_method,
            cluster_ks=cluster_ks,
        )

        mds_coordinates_by_model = self._compute_mds_by_model(distance_by_model)

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._save_outputs(
            rep_keys=rep_keys,
            model_names=model_names,
            embedding_vectors=embedding_vectors,
            distance_by_model=distance_by_model,
            cross_model_distance=cross_model_distance,
            linkage_by_model=linkage_by_model,
            leaf_order_by_model=leaf_order_by_model,
            mds_coordinates_by_model=mds_coordinates_by_model,
            linkage_method=linkage_method,
            clustering_summary=clustering_summary,
        )

        self._plot_per_model_heatmaps(
            model_names=model_names,
            distance_by_model=distance_by_model,
            leaf_order_by_model=leaf_order_by_model,
        )

        self._plot_cluster_2d(
            model_names=model_names,
            mds_coordinates_by_model=mds_coordinates_by_model,
            cluster_labels_by_model=cluster_labels_by_model,
            cluster_ks=cluster_ks,
        )

        self._plot_cross_model_heatmaps(
            model_names=model_names,
            cross_model_distance=cross_model_distance,
            leaf_order_by_model=leaf_order_by_model,
        )

        return {
            "output_dir": self.save_dir,
            "dataset_names": self.dataset_names,
            "model_names": model_names,
            "rep_keys": rep_keys,
            "embedding_vectors": embedding_vectors,
            "distance_by_model": distance_by_model,
            "cross_model_distance": cross_model_distance,
            "linkage_by_model": linkage_by_model,
            "leaf_order_by_model": leaf_order_by_model,
            "mds_coordinates_by_model": mds_coordinates_by_model,
        }

    def _parse_rep_types(self, value):
        if isinstance(value, str):
            value = value.strip()

            if value == "all":
                keys = list(self.REP_SPECS.keys())
            else:
                keys = [
                    item.strip()
                    for chunk in value.split(",")
                    for item in chunk.split()
                    if item.strip()
                ]
        else:
            keys = list(value)

        rep_keys = []
        for key in keys:
            key = str(key).strip()

            if key == "all":
                rep_keys.extend(self.REP_SPECS.keys())
                continue

            if key not in self.REP_SPECS:
                raise ValueError(
                    f"Unknown rep_type '{key}'. "
                    f"Valid values: {list(self.REP_SPECS.keys()) + ['all']}"
                )

            rep_keys.append(key)

        rep_keys = list(dict.fromkeys(rep_keys))

        if len(rep_keys) == 0:
            raise ValueError("At least one rep_type must be selected.")

        return rep_keys

    @staticmethod
    def _safe_name(name: str) -> str:
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))

    def _find_npz_path(self, path: str | Path) -> Path:
        path = Path(path)

        if path.is_file():
            return path

        direct_path = path / self.npz_filename
        if direct_path.exists():
            return direct_path

        matches = sorted(path.glob("*representations*.npz"))
        if len(matches) == 1:
            return matches[0]

        raise FileNotFoundError(f"Could not find representation npz in {path}")

    def _load_dataset_vectors(self, npz_path: Path, rep_keys: Sequence[str]):
        parts = []

        with np.load(npz_path) as data:
            model_names = [str(name) for name in data["model_names"]]

            for rep_key in rep_keys:
                array_name, projection_idx = self.REP_SPECS[rep_key]
                arr = data[array_name]

                if projection_idx is not None:
                    arr = arr[..., projection_idx]

                arr = arr.reshape(arr.shape[0], -1)
                parts.append(arr.astype(np.float64))

        return np.concatenate(parts, axis=1), model_names

    def _load_embeddings(self, rep_keys: Sequence[str]):
        vectors_per_dataset = []
        model_names = None

        for path in self.dataset_fim_paths:
            npz_path = self._find_npz_path(path)
            vectors, file_model_names = self._load_dataset_vectors(npz_path, rep_keys)

            if model_names is None:
                model_names = file_model_names

            vectors_per_dataset.append(vectors)

        return np.stack(vectors_per_dataset, axis=0), model_names

    @staticmethod
    def _task2vec_distance_matrix(A: np.ndarray, B: np.ndarray, eps: float):
        A = np.maximum(A.astype(np.float64), 0.0)
        B = np.maximum(B.astype(np.float64), 0.0)

        denom = A[:, None, :] + B[None, :, :] + eps
        A_norm = A[:, None, :] / denom
        B_norm = B[None, :, :] / denom

        dot = np.sum(A_norm * B_norm, axis=-1)
        norm = np.linalg.norm(A_norm, axis=-1) * np.linalg.norm(B_norm, axis=-1) + eps

        D = 1.0 - dot / norm
        return np.clip(D, 0.0, 2.0)

    def _compute_distances(self, embedding_vectors: np.ndarray, eps: float):
        n_datasets, n_models, _ = embedding_vectors.shape

        distance_by_model = np.zeros(
            (n_models, n_datasets, n_datasets),
            dtype=np.float64,
        )

        cross_model_distance = np.zeros(
            (n_models, n_models, n_datasets, n_datasets),
            dtype=np.float64,
        )

        for m1 in range(n_models):
            X1 = embedding_vectors[:, m1, :]

            for m2 in range(n_models):
                X2 = embedding_vectors[:, m2, :]
                D = self._task2vec_distance_matrix(X1, X2, eps=eps)

                if m1 == m2:
                    D = 0.5 * (D + D.T)
                    np.fill_diagonal(D, 0.0)
                    distance_by_model[m1] = D

                cross_model_distance[m1, m2] = D

        return distance_by_model, cross_model_distance

    @staticmethod
    def _cluster_distance_matrix(
        D: np.ndarray,
        linkage_method: str,
        cluster_ks: Sequence[int],
    ):

        D = np.asarray(D, dtype=np.float64)
        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)

        condensed = squareform(D, checks=False)
        Z = linkage(condensed, method=linkage_method)
        Z = optimal_leaf_ordering(Z, condensed)

        order = leaves_list(Z)

        labels_by_k = {
            str(k): fcluster(Z, t=k, criterion="maxclust").astype(int).tolist()
            for k in cluster_ks
        }

        return Z, order, labels_by_k

    def _cluster_models(
        self,
        distance_by_model: np.ndarray,
        model_names: Sequence[str],
        linkage_method: str,
        cluster_ks: Sequence[int],
    ):
        linkage_by_model = []
        leaf_order_by_model = []
        cluster_labels_by_model = []
        clustering_summary = {}

        for model_idx, model_name in enumerate(model_names):
            Z, order, labels_by_k = self._cluster_distance_matrix(
                distance_by_model[model_idx],
                linkage_method=linkage_method,
                cluster_ks=cluster_ks,
            )

            linkage_by_model.append(Z)
            leaf_order_by_model.append(order)
            cluster_labels_by_model.append(labels_by_k)

            clustering_summary[model_name] = {
                "leaf_order": order.astype(int).tolist(),
                "ordered_dataset_names": [self.dataset_names[i] for i in order],
                "cluster_labels_by_k": labels_by_k,
            }

        return (
            np.stack(linkage_by_model, axis=0),
            np.stack(leaf_order_by_model, axis=0),
            cluster_labels_by_model,
            clustering_summary,
        )

    @staticmethod
    def _compute_mds(D: np.ndarray):

        D = np.asarray(D, dtype=np.float64)
        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)

        return MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=0,
            n_init=8,
        ).fit_transform(D)

    def _compute_mds_by_model(self, distance_by_model: np.ndarray):
        return np.stack(
            [self._compute_mds(D) for D in distance_by_model],
            axis=0,
        )

    def _save_outputs(
        self,
        rep_keys,
        model_names,
        embedding_vectors,
        distance_by_model,
        cross_model_distance,
        linkage_by_model,
        leaf_order_by_model,
        mds_coordinates_by_model,
        linkage_method,
        clustering_summary,
    ):
        np.savez_compressed(
            self.save_dir / "distance_matrices.npz",
            dataset_names=np.array(self.dataset_names),
            model_names=np.array(model_names),
            rep_keys=np.array(rep_keys),
            embedding_vectors=embedding_vectors,
            distance_by_model=distance_by_model,
            cross_model_distance=cross_model_distance,
            linkage_by_model=linkage_by_model,
            leaf_order_by_model=leaf_order_by_model,
            mds_coordinates_by_model=mds_coordinates_by_model,
        )

        with open(self.save_dir / "clustering_summary.json", "w") as f:
            json.dump(
                {
                    "distance_metric": "task2vec_symmetric_cosine",
                    "distance_formula": (
                        "cosine_distance(Fa / (Fa + Fb), Fb / (Fa + Fb))"
                    ),
                    "linkage_method": linkage_method,
                    "rep_keys": rep_keys,
                    "dataset_names": self.dataset_names,
                    "model_names": model_names,
                    "embedding_shape": list(embedding_vectors.shape),
                    "distance_by_model_shape": list(distance_by_model.shape),
                    "cross_model_distance_shape": list(cross_model_distance.shape),
                    "mds_coordinates_by_model_shape": list(
                        mds_coordinates_by_model.shape
                    ),
                    "per_model_clustering": clustering_summary,
                },
                f,
                indent=2,
            )

    @staticmethod
    def _plot_heatmap(
        matrix: np.ndarray,
        row_labels: Sequence[str],
        col_labels: Sequence[str],
        title: str,
        path: Path,
        diagonal_values: np.ndarray | None = None,
        diagonal_labels: Sequence[str] | None = None,
    ):
        matrix = np.asarray(matrix)
        has_diagonal_panel = diagonal_values is not None and diagonal_labels is not None

        fig_w = max(8.0, 0.45 * len(col_labels) + 3.0)
        fig_h = max(6.0, 0.35 * len(row_labels) + 2.5)

        if has_diagonal_panel:
            fig_w += max(4.0, 0.18 * max(len(str(x)) for x in diagonal_labels))
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(fig_w, fig_h),
                gridspec_kw={"width_ratios": [3.2, 1.4]},
            )
            ax, diag_ax = axes
        else:
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            diag_ax = None

        im = ax.imshow(matrix, aspect="auto")

        ax.set_title(title)
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels, rotation=90)
        ax.set_yticklabels(row_labels)

        fig.colorbar(im, ax=ax, label="Task2Vec distance", fraction=0.046, pad=0.04)

        if has_diagonal_panel:
            diagonal_values = np.asarray(diagonal_values, dtype=np.float64)
            order = np.argsort(diagonal_values)

            sorted_values = diagonal_values[order]
            sorted_labels = [diagonal_labels[i] for i in order]

            y = np.arange(len(sorted_values))
            diag_ax.barh(y, sorted_values)
            diag_ax.set_title("Same-dataset distance\nsorted")
            diag_ax.set_xlabel("Distance")
            diag_ax.set_yticks(y)
            diag_ax.set_yticklabels(sorted_labels)
            diag_ax.invert_yaxis()
            diag_ax.grid(axis="x", alpha=0.3)

        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _plot_cluster_2d(
        coords: np.ndarray,
        cluster_labels: Sequence[int],
        point_labels: Sequence[str],
        title: str,
        path: Path,
    ):
        coords = np.asarray(coords, dtype=np.float64)
        cluster_labels = np.asarray(cluster_labels)

        fig, ax = plt.subplots(figsize=(9.5, 7.5))

        for cluster_id in sorted(np.unique(cluster_labels)):
            mask = cluster_labels == cluster_id
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=70,
                label=f"Cluster {cluster_id}",
            )

        for x, y, label in zip(coords[:, 0], coords[:, 1], point_labels):
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=9,
            )

        ax.set_title(title)
        ax.set_xlabel("MDS 1")
        ax.set_ylabel("MDS 2")
        ax.grid(alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)

        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    def _plot_per_model_heatmaps(
        self,
        model_names,
        distance_by_model,
        leaf_order_by_model,
    ):
        output_dir = self.save_dir / "per_model"

        for model_idx, model_name in enumerate(model_names):
            order = leaf_order_by_model[model_idx]
            ordered_names = [self.dataset_names[i] for i in order]
            D_ordered = distance_by_model[model_idx][np.ix_(order, order)]

            self._plot_heatmap(
                matrix=D_ordered,
                row_labels=ordered_names,
                col_labels=ordered_names,
                title=f"Dataset distances | {model_name}",
                path=output_dir
                / f"{self._safe_name(model_name)}_clustered_distance_matrix.png",
            )

    def _plot_cluster_2d(
        self,
        model_names,
        mds_coordinates_by_model,
        cluster_labels_by_model,
        cluster_ks,
    ):
        output_dir = self.save_dir / "cluster_2d"

        for model_idx, model_name in enumerate(model_names):
            coords = mds_coordinates_by_model[model_idx]
            labels_by_k = cluster_labels_by_model[model_idx]

            for k in cluster_ks:
                k = str(k)

                self._plot_cluster_2d_single(
                    coords=coords,
                    cluster_labels=labels_by_k[k],
                    point_labels=self.dataset_names,
                    title=f"2D task map | {model_name} | k={k}",
                    path=output_dir
                    / f"{self._safe_name(model_name)}_k{k}_cluster_2d.png",
                )

    @staticmethod
    def _plot_cluster_2d_single(
        coords: np.ndarray,
        cluster_labels: Sequence[int],
        point_labels: Sequence[str],
        title: str,
        path: Path,
    ):
        coords = np.asarray(coords, dtype=np.float64)
        cluster_labels = np.asarray(cluster_labels)

        fig, ax = plt.subplots(figsize=(9.5, 7.5))

        for cluster_id in sorted(np.unique(cluster_labels)):
            mask = cluster_labels == cluster_id
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=70,
                label=f"Cluster {cluster_id}",
            )

        for x, y, label in zip(coords[:, 0], coords[:, 1], point_labels):
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=9,
            )

        ax.set_title(title)
        ax.set_xlabel("MDS 1")
        ax.set_ylabel("MDS 2")
        ax.grid(alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)

        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    def _plot_cross_model_heatmaps(
        self,
        model_names,
        cross_model_distance,
        leaf_order_by_model,
    ):
        output_dir = self.save_dir / "cross_model"

        for row_model_idx, row_model_name in enumerate(model_names):
            row_order = leaf_order_by_model[row_model_idx]
            row_labels = [self.dataset_names[i] for i in row_order]

            for col_model_idx, col_model_name in enumerate(model_names):
                if row_model_idx == col_model_idx:
                    continue

                col_order = leaf_order_by_model[col_model_idx]
                col_labels = [self.dataset_names[i] for i in col_order]

                D_cross_raw = cross_model_distance[row_model_idx, col_model_idx]
                D_cross = D_cross_raw[np.ix_(row_order, col_order)]

                self._plot_heatmap(
                    matrix=D_cross,
                    row_labels=row_labels,
                    col_labels=col_labels,
                    title=(
                        "Cross-model dataset distances | "
                        f"{row_model_name} vs {col_model_name}"
                    ),
                    path=output_dir
                    / (
                        f"{self._safe_name(row_model_name)}_vs_"
                        f"{self._safe_name(col_model_name)}_distance_matrix.png"
                    ),
                    diagonal_values=np.diag(D_cross_raw),
                    diagonal_labels=self.dataset_names,
                )
