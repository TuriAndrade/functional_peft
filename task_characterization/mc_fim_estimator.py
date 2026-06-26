import json
from pathlib import Path
from typing import Dict, Any, Sequence

from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from scipy.cluster.hierarchy import (
    linkage,
    optimal_leaf_ordering,
    leaves_list,
    fcluster,
)
from scipy.spatial.distance import squareform


# ------- MonteCarloFIMEstimator -------
class MonteCarloFIMEstimator:
    def __init__(
        self,
        models: Sequence[nn.Module],
        save_dir: str | Path,
        model_names: Sequence[str] | None = None,
        plot_norm: str = "max",  # "max" or "percentile"
        plot_percentile: float = 99.0,
    ):
        self.models = list(models)

        if model_names is None:
            self.model_names = [f"model_{i}" for i in range(len(self.models))]
        else:
            self.model_names = list(model_names)

        if len(self.models) != len(self.model_names):
            raise ValueError("models and model_names must have the same length.")

        if plot_norm not in {"max", "percentile"}:
            raise ValueError("plot_norm must be either 'max' or 'percentile'.")

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.plot_norm = plot_norm
        self.plot_percentile = plot_percentile

        self.plot_norm_types = [
            "global",
            "by_model",
            "by_row",
            "by_column",
        ]

        self.group_names = [
            "block",
            "attention_qkvo",
            "mlp_layer",
            "patch_embed_channel",
        ]

        self.attention_projection_names = ["query", "key", "value", "output"]
        self.attention_projection_labels = ["Q", "K", "V", "O"]

    def _save_json(self, obj: Dict[str, Any], filename: str):
        with open(self.save_dir / filename, "w") as f:
            json.dump(obj, f, indent=2)

    def _get_x(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch
        return batch[0]

    def _get_logits(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        out = model(x)

        # Some ViT implementations return (logits, attn_weights)
        if isinstance(out, tuple):
            return out[0]

        return out

    def _get_backbone(self, model: nn.Module):
        """
        Supports either:
          1. model.get_param_group(...)
          2. model.backbone.get_param_group(...)
        """
        if hasattr(model, "get_param_group"):
            return model

        if hasattr(model, "backbone") and hasattr(model.backbone, "get_param_group"):
            return model.backbone

        raise AttributeError(
            "Could not find get_param_group(...). "
            "Expected either model.get_param_group or model.backbone.get_param_group."
        )

    def _snapshot_requires_grad(self, model: nn.Module):
        return {p: p.requires_grad for p in model.parameters()}

    def _restore_requires_grad(self, snapshot):
        for p, requires_grad in snapshot.items():
            p.requires_grad = requires_grad

    def _collect_spec_params(self, specs: Dict[str, Dict[str, Any]]):
        params = set()

        for spec in specs.values():
            for group in spec["groups"]:
                for item in group["items"]:
                    params.add(item["param"])

        return params

    def _prepare_for_fim(self, model: nn.Module, specs: Dict[str, Dict[str, Any]]):
        """
        Disable gradients for everything, then enable gradients only for the
        backbone params used by the FIM groups.

        The head/probe still participates in the forward pass, but its params
        do not need gradients for Fisher over the backbone.
        """
        for p in model.parameters():
            p.requires_grad = False

        for p in self._collect_spec_params(specs):
            p.requires_grad = True

    def _aggregate_grad2(self, spec: Dict[str, Any], device: str):
        out = torch.zeros(spec["shape"], device=device)

        for group in spec["groups"]:
            total = torch.tensor(0.0, device=device)
            count = 0

            for item in group["items"]:
                grad = item["param"].grad

                if grad is None:
                    continue

                if item["index"] is not None:
                    grad = grad[item["index"]]

                grad2 = grad.detach().pow(2)
                total = total + grad2.sum()
                count += grad2.numel()

            if count > 0:
                out[group["index"]] = total / count

        return out

    def _compute_one_model(
        self,
        model: nn.Module,
        loader,
        device: str,
        model_name: str,
        max_batches: int | None,
        mc_samples: int,
    ):
        model.to(device)
        model.eval()

        backbone = self._get_backbone(model)
        specs = {
            group_name: backbone.get_param_group(group_name)
            for group_name in self.group_names
        }

        requires_grad_snapshot = self._snapshot_requires_grad(model)
        self._prepare_for_fim(model, specs)

        fim_sums = {
            group_name: torch.zeros(spec["shape"], device=device)
            for group_name, spec in specs.items()
        }

        n_steps = 0
        n_samples = 0

        pbar = tqdm(
            loader,
            desc=f"[FIM] {model_name}",
            unit="batch",
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        try:
            with torch.enable_grad():
                for batch_idx, batch in enumerate(pbar):
                    if max_batches is not None and batch_idx >= max_batches:
                        break

                    x = self._get_x(batch).to(device, non_blocking=True)
                    bs = x.size(0)
                    n_samples += bs

                    for _ in range(mc_samples):
                        model.zero_grad(set_to_none=True)

                        logits = self._get_logits(model, x)

                        pseudo_y = torch.distributions.Categorical(
                            logits=logits
                        ).sample()

                        loss = F.cross_entropy(logits, pseudo_y)
                        loss.backward()

                        for group_name, spec in specs.items():
                            fim_sums[group_name] += self._aggregate_grad2(
                                spec,
                                device=device,
                            )

                        n_steps += 1

                    running = {
                        k: float(v.mean().detach().cpu()) for k, v in fim_sums.items()
                    }

                    pbar.set_postfix_str(
                        " • ".join(
                            f"{k}={v / max(n_steps, 1):.3e}" for k, v in running.items()
                        )
                    )

        finally:
            self._restore_requires_grad(requires_grad_snapshot)

        pbar.close()

        if n_steps == 0:
            raise RuntimeError("No FIM steps were computed. Check loader/max_batches.")

        reps = {
            group_name: (fim_sums[group_name] / n_steps).detach().cpu().numpy()
            for group_name in self.group_names
        }

        summary = {
            "model_name": model_name,
            "n_samples_seen": n_samples,
            "n_fim_steps": n_steps,
            "mc_samples": mc_samples,
        }

        for group_name, arr in reps.items():
            summary[group_name] = {
                "shape": list(arr.shape),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "l1": float(np.abs(arr).sum()),
                "l2": float(np.sqrt((arr**2).sum())),
            }

        tqdm.write(
            f"FIM done • {model_name} • " f"steps={n_steps} • samples={n_samples}"
        )

        return reps, summary

    def _get_plot_scale(self, arr: np.ndarray):
        flat = arr.reshape(-1)

        if self.plot_norm == "max":
            scale = float(flat.max())
        elif self.plot_norm == "percentile":
            scale = float(np.percentile(flat, self.plot_percentile))
        else:
            raise ValueError("plot_norm must be either 'max' or 'percentile'.")

        if scale <= 0:
            scale = 1.0

        return scale

    def _get_axis_plot_scale(self, arr: np.ndarray, axis: int):
        if self.plot_norm == "max":
            scale = arr.max(axis=axis, keepdims=True)
        elif self.plot_norm == "percentile":
            scale = np.percentile(
                arr,
                self.plot_percentile,
                axis=axis,
                keepdims=True,
            )
        else:
            raise ValueError("plot_norm must be either 'max' or 'percentile'.")

        scale = np.asarray(scale, dtype=np.float64)
        scale = np.where(scale <= 0, 1.0, scale)
        return scale

    def _normalize_for_plot(
        self,
        arr: np.ndarray,
        norm_type: str,
        global_scale: float | None = None,
    ):
        if norm_type == "global":
            if global_scale is None:
                raise ValueError(
                    "global_scale must be provided for global normalization."
                )
            scale = global_scale
            arr = arr / scale

        elif norm_type == "by_model":
            scale = self._get_plot_scale(arr)
            arr = arr / scale

        elif norm_type == "by_row":
            if arr.ndim != 2:
                raise ValueError("by_row normalization only applies to 2D arrays.")
            scale = self._get_axis_plot_scale(arr, axis=1)
            arr = arr / scale

        elif norm_type == "by_column":
            if arr.ndim != 2:
                raise ValueError("by_column normalization only applies to 2D arrays.")
            scale = self._get_axis_plot_scale(arr, axis=0)
            arr = arr / scale

        else:
            raise ValueError(
                "norm_type must be one of: "
                "'global', 'by_model', 'by_row', 'by_column'."
            )

        arr = np.clip(arr, 0.0, 1.0)
        return arr, scale

    def _get_global_plot_scale(self, arrays: Sequence[np.ndarray]):
        flat = np.concatenate([arr.reshape(-1) for arr in arrays], axis=0)
        return self._get_plot_scale(flat)

    def _plot_arrays(
        self,
        arrays: Sequence[np.ndarray],
        title: str,
        filename: str,
        xlabel: str,
        norm_type: str,
        ylabel: str | None = None,
        yticklabels: Sequence[str] | None = None,
    ):
        if norm_type in {"by_row", "by_column"}:
            if any(arr.ndim != 2 for arr in arrays):
                return

        global_scale = None
        if norm_type == "global":
            global_scale = self._get_global_plot_scale(arrays)

        n_models = len(self.model_names)
        ncols = 2 if n_models <= 4 else 3
        nrows = math.ceil(n_models / ncols)

        first = arrays[0]
        first_2d = first if first.ndim == 2 else first.reshape(1, -1)
        subplot_height = 3.5 if first_2d.shape[0] > 1 else 2.2

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.5 * ncols + 0.7, subplot_height * nrows),
            squeeze=False,
        )

        im = None

        for i, (model_name, arr) in enumerate(zip(self.model_names, arrays)):
            ax = axes[i // ncols, i % ncols]

            data = arr if arr.ndim == 2 else arr.reshape(1, -1)
            data, scale = self._normalize_for_plot(
                data,
                norm_type=norm_type,
                global_scale=global_scale,
            )

            im = ax.imshow(
                data,
                aspect="auto",
                vmin=0.0,
                vmax=1.0,
            )

            ax.set_title(model_name)
            ax.set_xlabel(xlabel)

            if ylabel is not None:
                ax.set_ylabel(ylabel)

            if yticklabels is not None:
                ax.set_yticks(np.arange(len(yticklabels)))
                ax.set_yticklabels(yticklabels)
            elif data.shape[0] == 1:
                ax.set_yticks([])

        for j in range(n_models, nrows * ncols):
            axes[j // ncols, j % ncols].axis("off")

        fig.suptitle(
            f"{title} | {norm_type} normalization",
            y=0.98,
        )

        fig.subplots_adjust(
            left=0.08,
            right=0.88,
            top=0.88,
            bottom=0.10,
            wspace=0.35,
            hspace=0.55,
        )

        cbar_ax = fig.add_axes([0.91, 0.18, 0.025, 0.64])
        fig.colorbar(
            im,
            cax=cbar_ax,
            label=f"Normalized FIM ({norm_type})",
        )

        plot_dir = self.save_dir / "plots" / norm_type
        plot_dir.mkdir(parents=True, exist_ok=True)

        plt.savefig(plot_dir / filename, bbox_inches="tight")
        plt.close(fig)

    def _plot_arrays_all_norms(
        self,
        arrays: Sequence[np.ndarray],
        title: str,
        filename: str,
        xlabel: str,
        ylabel: str | None = None,
        yticklabels: Sequence[str] | None = None,
    ):
        for norm_type in self.plot_norm_types:
            self._plot_arrays(
                arrays=arrays,
                title=title,
                filename=filename,
                xlabel=xlabel,
                norm_type=norm_type,
                ylabel=ylabel,
                yticklabels=yticklabels,
            )

    def _plot_all(self, all_reps: Dict[str, Dict[str, np.ndarray]]):
        self._plot_arrays_all_norms(
            arrays=[all_reps[name]["block"] for name in self.model_names],
            title="FIM by transformer block",
            filename="fim_by_transformer_block_heatmaps.png",
            xlabel="Transformer block",
        )

        for projection_idx, projection_name in enumerate(
            self.attention_projection_names
        ):
            self._plot_arrays_all_norms(
                arrays=[
                    all_reps[name]["attention_qkvo"][:, :, projection_idx]
                    for name in self.model_names
                ],
                title=f"FIM by attention {projection_name} projection, block and head",
                filename=(
                    f"fim_by_attention_{projection_name}_projection_"
                    "block_head_heatmaps.png"
                ),
                xlabel="Attention head",
                ylabel="Transformer block",
            )

        self._plot_arrays_all_norms(
            arrays=[
                all_reps[name]["attention_qkvo"].mean(axis=1).T
                for name in self.model_names
            ],
            title="FIM by attention projection and block, averaged over heads",
            filename="fim_by_attention_projection_block_mean_over_heads_heatmaps.png",
            xlabel="Transformer block",
            ylabel="Attention projection",
            yticklabels=self.attention_projection_labels,
        )

        self._plot_arrays_all_norms(
            arrays=[
                all_reps[name]["attention_qkvo"].mean(axis=(1, 2))
                for name in self.model_names
            ],
            title="FIM by attention block, averaged over heads and projections",
            filename="fim_by_attention_block_mean_over_heads_and_projections_heatmaps.png",
            xlabel="Transformer block",
        )

        self._plot_arrays_all_norms(
            arrays=[all_reps[name]["mlp_layer"] for name in self.model_names],
            title="FIM by MLP block",
            filename="fim_by_mlp_block_heatmaps.png",
            xlabel="Transformer block",
        )

        self._plot_arrays_all_norms(
            arrays=[all_reps[name]["patch_embed_channel"] for name in self.model_names],
            title="FIM by patch embedding channel",
            filename="fim_by_patch_embedding_channel_heatmaps.png",
            xlabel="Patch embedding output channel",
        )

    def _save_npz(self, all_reps: Dict[str, Dict[str, np.ndarray]]):
        fim_by_transformer_block = np.stack(
            [all_reps[name]["block"] for name in self.model_names],
            axis=0,
        )

        fim_by_attention_block_head_projection = np.stack(
            [all_reps[name]["attention_qkvo"] for name in self.model_names],
            axis=0,
        )

        fim_by_mlp_block = np.stack(
            [all_reps[name]["mlp_layer"] for name in self.model_names],
            axis=0,
        )

        fim_by_patch_embedding_channel = np.stack(
            [all_reps[name]["patch_embed_channel"] for name in self.model_names],
            axis=0,
        )

        np.savez_compressed(
            self.save_dir / "fim_representations.npz",
            model_names=np.array(self.model_names),
            attention_projection_names=np.array(self.attention_projection_names),
            attention_projection_labels=np.array(self.attention_projection_labels),
            fim_by_transformer_block=fim_by_transformer_block,
            fim_by_attention_block_head_projection=(
                fim_by_attention_block_head_projection
            ),
            fim_by_mlp_block=fim_by_mlp_block,
            fim_by_patch_embedding_channel=fim_by_patch_embedding_channel,
        )

    def compute(
        self,
        loader,
        device: str = "cuda",
        max_batches: int | None = None,
        mc_samples: int = 1,
        save: bool = True,
        plot: bool = True,
    ):
        all_reps = {}
        summaries = []

        for model, model_name in zip(self.models, self.model_names):
            reps, summary = self._compute_one_model(
                model=model,
                loader=loader,
                device=device,
                model_name=model_name,
                max_batches=max_batches,
                mc_samples=mc_samples,
            )

            all_reps[model_name] = reps
            summaries.append(summary)

        if save:
            self._save_npz(all_reps)
            self._save_json(
                {
                    "estimator": "monte_carlo_diagonal_fisher",
                    "plot_normalizations": self.plot_norm_types,
                    "plot_norm": self.plot_norm,
                    "plot_percentile": self.plot_percentile,
                    "plot_dirs": {
                        norm_type: str(Path("plots") / norm_type)
                        for norm_type in self.plot_norm_types
                    },
                    "backbone_param_groups": self.group_names,
                    "attention_projection_names": self.attention_projection_names,
                    "attention_projection_labels": self.attention_projection_labels,
                    "saved_arrays": {
                        "fim_by_transformer_block": ("[n_models, n_blocks]"),
                        "fim_by_attention_block_head_projection": (
                            "[n_models, n_blocks, n_heads, 4] "
                            "where the last axis is query/key/value/output"
                        ),
                        "fim_by_mlp_block": ("[n_models, n_blocks]"),
                        "fim_by_patch_embedding_channel": (
                            "[n_models, n_patch_embedding_channels]"
                        ),
                    },
                    "models": summaries,
                },
                "fim_summary.json",
            )

        if plot:
            self._plot_all(all_reps)

        return all_reps

    @staticmethod
    def analyze_dataset_fim_distances(
        dataset_fim_paths: Sequence[str | Path],
        dataset_names: Sequence[str],
        save_dir: str | Path,
        rep_types: str | Sequence[str] = "all",
        npz_filename: str = "fim_representations.npz",
        save_name: str = "dataset_fim_distances",
        linkage_method: str = "average",
        cluster_ks: Sequence[int] = (2, 3, 4, 5),
        eps: float = 1e-12,
    ):
        from scipy.cluster.hierarchy import (
            linkage,
            optimal_leaf_ordering,
            leaves_list,
            fcluster,
        )
        from scipy.spatial.distance import squareform

        rep_specs = {
            "block": ("fim_by_transformer_block", None),
            "block_attn": ("fim_by_attention_block_head_projection", None),
            "block_attn_q": ("fim_by_attention_block_head_projection", 0),
            "block_attn_k": ("fim_by_attention_block_head_projection", 1),
            "block_attn_v": ("fim_by_attention_block_head_projection", 2),
            "block_attn_o": ("fim_by_attention_block_head_projection", 3),
            "block_mlp": ("fim_by_mlp_block", None),
            "patch_embd": ("fim_by_patch_embedding_channel", None),
        }

        def safe_name(name: str) -> str:
            return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))

        def parse_rep_types(value):
            if isinstance(value, str):
                value = value.strip()

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

                if key not in rep_specs:
                    raise ValueError(
                        f"Unknown rep_type '{key}'. Valid values: {rep_specs.keys()}"
                    )

                rep_keys.append(key)

            rep_keys = list(dict.fromkeys(rep_keys))

            if len(rep_keys) == 0:
                raise ValueError("At least one rep_type must be selected.")

            return rep_keys

        def find_npz_path(path: str | Path) -> Path:
            path = Path(path)

            if path.is_file():
                return path

            direct_path = path / npz_filename
            if direct_path.exists():
                return direct_path

            matches = sorted(path.glob("*representations*.npz"))
            if len(matches) == 1:
                return matches[0]

            raise FileNotFoundError(f"Could not find representation npz in {path}")

        def load_dataset_vectors(npz_path: Path, rep_keys: Sequence[str]):
            parts = []

            with np.load(npz_path) as data:
                model_names = [str(name) for name in data["model_names"]]

                for rep_key in rep_keys:
                    array_name, projection_idx = rep_specs[rep_key]
                    arr = data[array_name]

                    if projection_idx is not None:
                        arr = arr[..., projection_idx]

                    arr = arr.reshape(arr.shape[0], -1)
                    parts.append(arr.astype(np.float64))

            return np.concatenate(parts, axis=1), model_names

        def task2vec_distance_matrix(A: np.ndarray, B: np.ndarray):
            A = np.maximum(A.astype(np.float64), 0.0)
            B = np.maximum(B.astype(np.float64), 0.0)

            denom = A[:, None, :] + B[None, :, :] + eps
            A_norm = A[:, None, :] / denom
            B_norm = B[None, :, :] / denom

            dot = np.sum(A_norm * B_norm, axis=-1)
            norm = (
                np.linalg.norm(A_norm, axis=-1) * np.linalg.norm(B_norm, axis=-1) + eps
            )

            D = 1.0 - dot / norm
            return np.clip(D, 0.0, 2.0)

        def cluster_distance_matrix(D: np.ndarray):
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

        def plot_heatmap(
            matrix: np.ndarray,
            row_labels: Sequence[str],
            col_labels: Sequence[str],
            title: str,
            path: Path,
            diagonal_values: np.ndarray | None = None,
            diagonal_labels: Sequence[str] | None = None,
        ):
            matrix = np.asarray(matrix)

            has_diagonal_panel = (
                diagonal_values is not None and diagonal_labels is not None
            )

            fig_w = max(8.0, 0.45 * len(col_labels) + 3.0)
            fig_h = max(6.0, 0.35 * len(row_labels) + 2.5)

            if has_diagonal_panel:
                fig_w = fig_w + max(
                    4.0, 0.18 * max(len(str(x)) for x in diagonal_labels)
                )
                fig, axes = plt.subplots(
                    1,
                    2,
                    figsize=(fig_w, fig_h),
                    gridspec_kw={"width_ratios": [3.2, 1.4]},
                )
                ax = axes[0]
                diag_ax = axes[1]
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

        # ------------------------------------------------------------------
        # Load FIM vectors
        # ------------------------------------------------------------------
        rep_keys = parse_rep_types(rep_types)
        dataset_fim_paths = [Path(path) for path in dataset_fim_paths]
        dataset_names = [str(name) for name in dataset_names]

        vectors_per_dataset = []
        model_names = None

        for path in dataset_fim_paths:
            npz_path = find_npz_path(path)
            vectors, file_model_names = load_dataset_vectors(npz_path, rep_keys)

            if model_names is None:
                model_names = file_model_names

            vectors_per_dataset.append(vectors)

        embedding_vectors = np.stack(vectors_per_dataset, axis=0)

        n_datasets, n_models, embedding_dim = embedding_vectors.shape

        # ------------------------------------------------------------------
        # Compute per-model and cross-model distances
        # ------------------------------------------------------------------
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
                D = task2vec_distance_matrix(X1, X2)

                if m1 == m2:
                    D = 0.5 * (D + D.T)
                    np.fill_diagonal(D, 0.0)
                    distance_by_model[m1] = D

                cross_model_distance[m1, m2] = D

        # ------------------------------------------------------------------
        # Cluster per-model matrices
        # ------------------------------------------------------------------
        linkage_by_model = []
        leaf_order_by_model = []
        clustering_summary = {}

        for model_idx, model_name in enumerate(model_names):
            Z, order, labels_by_k = cluster_distance_matrix(
                distance_by_model[model_idx]
            )

            linkage_by_model.append(Z)
            leaf_order_by_model.append(order)

            ordered_dataset_names = [dataset_names[i] for i in order]

            clustering_summary[model_name] = {
                "leaf_order": order.astype(int).tolist(),
                "ordered_dataset_names": ordered_dataset_names,
                "cluster_labels_by_k": labels_by_k,
            }

        linkage_by_model = np.stack(linkage_by_model, axis=0)
        leaf_order_by_model = np.stack(leaf_order_by_model, axis=0)

        # ------------------------------------------------------------------
        # Save matrices
        # ------------------------------------------------------------------
        output_dir = Path(save_dir) / save_name
        output_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_dir / "distance_matrices.npz",
            dataset_names=np.array(dataset_names),
            model_names=np.array(model_names),
            rep_keys=np.array(rep_keys),
            embedding_vectors=embedding_vectors,
            distance_by_model=distance_by_model,
            cross_model_distance=cross_model_distance,
            linkage_by_model=linkage_by_model,
            leaf_order_by_model=leaf_order_by_model,
        )

        with open(output_dir / "clustering_summary.json", "w") as f:
            json.dump(
                {
                    "distance_metric": "task2vec_symmetric_cosine",
                    "distance_formula": (
                        "cosine_distance(Fa / (Fa + Fb), Fb / (Fa + Fb))"
                    ),
                    "linkage_method": linkage_method,
                    "rep_keys": rep_keys,
                    "dataset_names": dataset_names,
                    "model_names": model_names,
                    "embedding_shape": list(embedding_vectors.shape),
                    "distance_by_model_shape": list(distance_by_model.shape),
                    "cross_model_distance_shape": list(cross_model_distance.shape),
                    "per_model_clustering": clustering_summary,
                },
                f,
                indent=2,
            )

        # ------------------------------------------------------------------
        # Plot per-model clustered distance matrices
        # ------------------------------------------------------------------
        per_model_dir = output_dir / "per_model"

        for model_idx, model_name in enumerate(model_names):
            order = leaf_order_by_model[model_idx]
            ordered_names = [dataset_names[i] for i in order]

            D_ordered = distance_by_model[model_idx][np.ix_(order, order)]

            plot_heatmap(
                matrix=D_ordered,
                row_labels=ordered_names,
                col_labels=ordered_names,
                title=f"Dataset distances | {model_name}",
                path=per_model_dir
                / f"{safe_name(model_name)}_clustered_distance_matrix.png",
            )

        # ------------------------------------------------------------------
        # Plot cross-model alignment matrices
        # ------------------------------------------------------------------
        cross_model_dir = output_dir / "cross_model"

        for row_model_idx, row_model_name in enumerate(model_names):
            row_order = leaf_order_by_model[row_model_idx]
            row_labels = [dataset_names[i] for i in row_order]

            for col_model_idx, col_model_name in enumerate(model_names):
                if row_model_idx == col_model_idx:
                    continue

                col_order = leaf_order_by_model[col_model_idx]
                col_labels = [dataset_names[i] for i in col_order]

                D_cross_raw = cross_model_distance[row_model_idx, col_model_idx]
                D_cross = D_cross_raw[np.ix_(row_order, col_order)]

                plot_heatmap(
                    matrix=D_cross,
                    row_labels=row_labels,
                    col_labels=col_labels,
                    title=(
                        "Cross-model dataset distances | "
                        f"{row_model_name} vs {col_model_name}"
                    ),
                    path=cross_model_dir
                    / (
                        f"{safe_name(row_model_name)}_vs_"
                        f"{safe_name(col_model_name)}_distance_matrix.png"
                    ),
                    diagonal_values=np.diag(D_cross_raw),
                    diagonal_labels=dataset_names,
                )

        return {
            "output_dir": output_dir,
            "dataset_names": dataset_names,
            "model_names": model_names,
            "rep_keys": rep_keys,
            "embedding_vectors": embedding_vectors,
            "distance_by_model": distance_by_model,
            "cross_model_distance": cross_model_distance,
            "linkage_by_model": linkage_by_model,
            "leaf_order_by_model": leaf_order_by_model,
        }
