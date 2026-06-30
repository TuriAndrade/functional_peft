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
    REP_SPECS = {
        "fim_block": ("fim_by_transformer_block", None),
        "fim_block_attn": ("fim_by_attention_block_head_projection", None),
        "fim_block_attn_q": ("fim_by_attention_block_head_projection", 0),
        "fim_block_attn_k": ("fim_by_attention_block_head_projection", 1),
        "fim_block_attn_v": ("fim_by_attention_block_head_projection", 2),
        "fim_block_attn_o": ("fim_by_attention_block_head_projection", 3),
        "fim_block_mlp": ("fim_by_mlp_block", None),
        "fim_patch_embd": ("fim_by_patch_embedding_channel", None),
    }

    def __init__(
        self,
        dataset_paths: Sequence[str | Path],
        dataset_names: Sequence[str],
        save_dir: str | Path,
    ):
        self.dataset_paths = [Path(path) for path in dataset_paths]
        self.dataset_names = [str(name) for name in dataset_names]
        self.save_dir = Path(save_dir)

    def compute(
        self,
        rep_types: str | Sequence[str] = "all",
        linkage_method: str = "average",
        cluster_ks: Sequence[int] = (2, 3, 4, 5),
        eps: float = 1e-12,
    ):
        rep_keys = self._parse_rep_types(rep_types)
        embeddings, model_names = self._load_embeddings(rep_keys)

        distance_by_model, cross_model_distance = self._compute_distances(
            embeddings,
            eps=eps,
        )

        clustering = self._cluster_all_models(
            distance_by_model=distance_by_model,
            model_names=model_names,
            linkage_method=linkage_method,
            cluster_ks=cluster_ks,
        )

        mds_coords = self._compute_all_mds(distance_by_model)

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._save_outputs(
            rep_keys=rep_keys,
            model_names=model_names,
            embeddings=embeddings,
            distance_by_model=distance_by_model,
            cross_model_distance=cross_model_distance,
            clustering=clustering,
            mds_coords=mds_coords,
            linkage_method=linkage_method,
        )

        self._plot_per_model_heatmaps(
            model_names=model_names,
            distance_by_model=distance_by_model,
            leaf_orders=clustering["leaf_orders"],
        )

        self._plot_2d_maps(
            model_names=model_names,
            mds_coords=mds_coords,
            labels_by_model=clustering["labels_by_model"],
            cluster_ks=cluster_ks,
        )

        self._plot_cross_model_heatmaps(
            model_names=model_names,
            cross_model_distance=cross_model_distance,
            leaf_orders=clustering["leaf_orders"],
        )

        return {
            "output_dir": self.save_dir,
            "dataset_names": self.dataset_names,
            "model_names": model_names,
            "rep_keys": rep_keys,
            "embedding_vectors": embeddings,
            "distance_by_model": distance_by_model,
            "cross_model_distance": cross_model_distance,
            "linkage_by_model": clustering["linkages"],
            "leaf_order_by_model": clustering["leaf_orders"],
            "mds_coordinates_by_model": mds_coords,
        }

    def _parse_rep_types(self, rep_types):
        if isinstance(rep_types, str):
            rep_types = rep_types.strip()
            keys = (
                list(self.REP_SPECS)
                if rep_types == "all"
                else [
                    item.strip()
                    for chunk in rep_types.split(",")
                    for item in chunk.split()
                    if item.strip()
                ]
            )
        else:
            keys = list(rep_types)

        rep_keys = []
        for key in keys:
            key = str(key).strip()

            if key == "all":
                rep_keys.extend(self.REP_SPECS)
            elif key in self.REP_SPECS:
                rep_keys.append(key)
            else:
                valid = list(self.REP_SPECS) + ["all"]
                raise ValueError(f"Unknown rep_type '{key}'. Valid values: {valid}")

        rep_keys = list(dict.fromkeys(rep_keys))

        if len(rep_keys) == 0:
            raise ValueError("At least one rep_type must be selected.")

        return rep_keys

    def _load_embeddings(self, rep_keys: Sequence[str]):
        vectors_per_dataset = []
        model_names = None

        for path in self.dataset_paths:
            vectors, names = self._load_dataset_vectors(
                npz_path=self._find_npz_path(path),
                rep_keys=rep_keys,
            )

            if model_names is None:
                model_names = names
            elif model_names != names:
                raise ValueError(
                    "All representation files must use the same model_names."
                )

            vectors_per_dataset.append(vectors)

        return np.stack(vectors_per_dataset, axis=0), model_names

    def _find_npz_path(self, path: str | Path) -> Path:
        path = Path(path)

        if path.is_file():
            if path.suffix != ".npz":
                raise ValueError(f"Expected a .npz file, got: {path}")
            return path

        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if not path.is_dir():
            raise ValueError(f"Expected a .npz file or directory, got: {path}")

        matches = sorted(path.glob("*.npz"))

        if len(matches) == 0:
            raise FileNotFoundError(f"No .npz file found in {path}")

        if len(matches) > 1:
            raise ValueError(
                f"Expected exactly one .npz file in {path}, found {len(matches)}: "
                f"{[match.name for match in matches]}"
            )

        return matches[0]

    def _load_dataset_vectors(self, npz_path: Path, rep_keys: Sequence[str]):
        parts = []

        with np.load(npz_path) as data:
            available_keys = set(data.files)

            if "model_names" not in available_keys:
                raise KeyError(
                    f"Missing key 'model_names' in {npz_path}. "
                    f"Available keys: {sorted(available_keys)}"
                )

            model_names = [str(name) for name in data["model_names"]]

            for rep_key in rep_keys:
                array_name, projection_idx = self.REP_SPECS[rep_key]

                if array_name not in available_keys:
                    raise KeyError(
                        f"Missing key '{array_name}' for rep_type '{rep_key}' in {npz_path}. "
                        f"Available keys: {sorted(available_keys)}"
                    )

                arr = data[array_name]

                if arr.shape[0] != len(model_names):
                    raise ValueError(
                        f"Array '{array_name}' in {npz_path} has first dimension "
                        f"{arr.shape[0]}, but model_names has length {len(model_names)}."
                    )

                if projection_idx is not None:
                    if arr.shape[-1] <= projection_idx:
                        raise ValueError(
                            f"Array '{array_name}' in {npz_path} has last dimension "
                            f"{arr.shape[-1]}, but projection index {projection_idx} was requested."
                        )

                    arr = arr[..., projection_idx]

                parts.append(arr.reshape(arr.shape[0], -1).astype(np.float64))

        return np.concatenate(parts, axis=1), model_names

    def _compute_distances(self, embeddings: np.ndarray, eps: float):
        n_datasets, n_models, _ = embeddings.shape

        distance_by_model = np.zeros(
            (n_models, n_datasets, n_datasets),
            dtype=np.float64,
        )

        cross_model_distance = np.zeros(
            (n_models, n_models, n_datasets, n_datasets),
            dtype=np.float64,
        )

        for m1 in range(n_models):
            for m2 in range(n_models):
                D = self._task2vec_distance(
                    embeddings[:, m1, :],
                    embeddings[:, m2, :],
                    eps=eps,
                )

                if m1 == m2:
                    D = self._sanitize_distance_matrix(D)
                    distance_by_model[m1] = D

                cross_model_distance[m1, m2] = D

        return distance_by_model, cross_model_distance

    @staticmethod
    def _task2vec_distance(A: np.ndarray, B: np.ndarray, eps: float):
        A = np.maximum(A.astype(np.float64), 0.0)
        B = np.maximum(B.astype(np.float64), 0.0)

        denom = A[:, None, :] + B[None, :, :] + eps
        A_norm = A[:, None, :] / denom
        B_norm = B[None, :, :] / denom

        dot = np.sum(A_norm * B_norm, axis=-1)
        norm = np.linalg.norm(A_norm, axis=-1) * np.linalg.norm(B_norm, axis=-1) + eps

        return np.clip(1.0 - dot / norm, 0.0, 2.0)

    @staticmethod
    def _sanitize_distance_matrix(D: np.ndarray):
        D = np.asarray(D, dtype=np.float64)
        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)
        return D

    def _cluster_all_models(
        self,
        distance_by_model: np.ndarray,
        model_names: Sequence[str],
        linkage_method: str,
        cluster_ks: Sequence[int],
    ):
        linkages = []
        leaf_orders = []
        labels_by_model = []
        summary = {}

        for model_idx, model_name in enumerate(model_names):
            Z, order, labels_by_k = self._cluster_distance_matrix(
                distance_by_model[model_idx],
                linkage_method=linkage_method,
                cluster_ks=cluster_ks,
            )

            linkages.append(Z)
            leaf_orders.append(order)
            labels_by_model.append(labels_by_k)

            summary[model_name] = {
                "leaf_order": order.astype(int).tolist(),
                "ordered_dataset_names": [self.dataset_names[i] for i in order],
                "cluster_labels_by_k": labels_by_k,
            }

        return {
            "linkages": np.stack(linkages, axis=0),
            "leaf_orders": np.stack(leaf_orders, axis=0),
            "labels_by_model": labels_by_model,
            "summary": summary,
        }

    @classmethod
    def _cluster_distance_matrix(
        cls,
        D: np.ndarray,
        linkage_method: str,
        cluster_ks: Sequence[int],
    ):
        D = cls._sanitize_distance_matrix(D)
        condensed = squareform(D, checks=False)

        Z = linkage(condensed, method=linkage_method)
        Z = optimal_leaf_ordering(Z, condensed)

        order = leaves_list(Z)
        labels_by_k = {
            str(k): fcluster(Z, t=k, criterion="maxclust").astype(int).tolist()
            for k in cluster_ks
        }

        return Z, order, labels_by_k

    @classmethod
    def _compute_mds(cls, D: np.ndarray):
        D = cls._sanitize_distance_matrix(D)

        return MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=0,
            n_init=8,
        ).fit_transform(D)

    def _compute_all_mds(self, distance_by_model: np.ndarray):
        return np.stack(
            [self._compute_mds(D) for D in distance_by_model],
            axis=0,
        )

    def _save_outputs(
        self,
        rep_keys,
        model_names,
        embeddings,
        distance_by_model,
        cross_model_distance,
        clustering,
        mds_coords,
        linkage_method,
    ):
        np.savez_compressed(
            self.save_dir / "distance_matrices.npz",
            dataset_names=np.array(self.dataset_names),
            model_names=np.array(model_names),
            rep_keys=np.array(rep_keys),
            embedding_vectors=embeddings,
            distance_by_model=distance_by_model,
            cross_model_distance=cross_model_distance,
            linkage_by_model=clustering["linkages"],
            leaf_order_by_model=clustering["leaf_orders"],
            mds_coordinates_by_model=mds_coords,
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
                    "embedding_shape": list(embeddings.shape),
                    "distance_by_model_shape": list(distance_by_model.shape),
                    "cross_model_distance_shape": list(cross_model_distance.shape),
                    "mds_coordinates_by_model_shape": list(mds_coords.shape),
                    "per_model_clustering": clustering["summary"],
                },
                f,
                indent=2,
            )

    def _plot_per_model_heatmaps(
        self,
        model_names,
        distance_by_model,
        leaf_orders,
    ):
        output_dir = self.save_dir / "per_model"

        for model_idx, model_name in enumerate(model_names):
            order = leaf_orders[model_idx]
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

    def _plot_2d_maps(
        self,
        model_names,
        mds_coords,
        labels_by_model,
        cluster_ks,
    ):
        output_dir = self.save_dir / "cluster_2d"

        for model_idx, model_name in enumerate(model_names):
            coords = mds_coords[model_idx]
            model_safe_name = self._safe_name(model_name)

            self._plot_2d_points(
                coords=coords,
                point_labels=self.dataset_names,
                title=f"2D task map | {model_name}",
                path=output_dir / f"{model_safe_name}_points_2d.png",
            )

            for k in cluster_ks:
                self._plot_2d_points(
                    coords=coords,
                    point_labels=self.dataset_names,
                    title=f"2D task map | {model_name} | k={k}",
                    path=output_dir / f"{model_safe_name}_k{k}_cluster_2d.png",
                    cluster_labels=labels_by_model[model_idx][str(k)],
                )

    def _plot_cross_model_heatmaps(
        self,
        model_names,
        cross_model_distance,
        leaf_orders,
    ):
        output_dir = self.save_dir / "cross_model"

        for row_idx, row_name in enumerate(model_names):
            row_order = leaf_orders[row_idx]
            row_labels = [self.dataset_names[i] for i in row_order]

            for col_idx, col_name in enumerate(model_names):
                if row_idx == col_idx:
                    continue

                col_order = leaf_orders[col_idx]
                col_labels = [self.dataset_names[i] for i in col_order]

                D_raw = cross_model_distance[row_idx, col_idx]
                D_ordered = D_raw[np.ix_(row_order, col_order)]

                self._plot_heatmap(
                    matrix=D_ordered,
                    row_labels=row_labels,
                    col_labels=col_labels,
                    title=f"Cross-model dataset distances | {row_name} vs {col_name}",
                    path=output_dir
                    / (
                        f"{self._safe_name(row_name)}_vs_"
                        f"{self._safe_name(col_name)}_distance_matrix.png"
                    ),
                    diagonal_values=np.diag(D_raw),
                    diagonal_labels=self.dataset_names,
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
        has_diagonal = diagonal_values is not None and diagonal_labels is not None

        fig_w = max(8.0, 0.45 * len(col_labels) + 3.0)
        fig_h = max(6.0, 0.35 * len(row_labels) + 2.5)

        if has_diagonal:
            fig_w += max(4.0, 0.18 * max(len(str(x)) for x in diagonal_labels))
            fig, (ax, diag_ax) = plt.subplots(
                1,
                2,
                figsize=(fig_w, fig_h),
                gridspec_kw={"width_ratios": [3.2, 1.4]},
            )
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

        if has_diagonal:
            diagonal_values = np.asarray(diagonal_values, dtype=np.float64)
            order = np.argsort(diagonal_values)

            y = np.arange(len(order))
            diag_ax.barh(y, diagonal_values[order])
            diag_ax.set_title("Same-dataset distance\nsorted")
            diag_ax.set_xlabel("Distance")
            diag_ax.set_yticks(y)
            diag_ax.set_yticklabels([diagonal_labels[i] for i in order])
            diag_ax.invert_yaxis()
            diag_ax.grid(axis="x", alpha=0.3)

        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _plot_2d_points(
        coords: np.ndarray,
        point_labels: Sequence[str],
        title: str,
        path: Path,
        cluster_labels: Sequence[int] | None = None,
    ):
        coords = np.asarray(coords, dtype=np.float64)

        fig, ax = plt.subplots(figsize=(9.5, 7.5))

        if cluster_labels is None:
            ax.scatter(coords[:, 0], coords[:, 1], s=70)
        else:
            cluster_labels = np.asarray(cluster_labels)

            for cluster_id in sorted(np.unique(cluster_labels)):
                mask = cluster_labels == cluster_id
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=70,
                    label=f"Cluster {cluster_id}",
                )

            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)

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

        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _safe_name(name: str) -> str:
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))
