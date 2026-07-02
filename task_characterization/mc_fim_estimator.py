import json
import math
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class MonteCarloFIMEstimator:
    """
    Assumed backbone interface:

        backbone.aggregate_fim_grads() -> dict[str, torch.Tensor]

    Returned FIM groups:

        input_patch_channel: [D]
            FIM per patch-embedding output channel.

        input_pos_grid: [G, G]
            FIM per learned patch positional embedding, arranged as the image patch grid.

        block_path: [B, 6]
            FIM per transformer block and computational path.
            Path axis: [LN1, QK, OV, LN2, MLP key, MLP value].

        attention_circuit_head: [B, H, 2]
            FIM per block, attention head, and attention circuit.
            Circuit axis: [QK, OV].

        mlp_memory_group: [B, K, 2]
            FIM per block, grouped MLP hidden units, and MLP memory role.
            Role axis: [key, value].

        residual_io_channel: [B, 2, D]
            FIM per block, residual-stream role, and residual channel.
            Role axis: [read, write].

        attention_circuit_dim: [B, H, 2, Dh]
            Fine-grained attention FIM per block, head, circuit, and head dimension.
            Circuit axis: [QK, OV].

        mlp_memory_unit: [B, M, 2]
            Fine-grained MLP FIM per block, MLP hidden unit, and memory role.
            Role axis: [key, value].
    """

    def __init__(
        self,
        models: Sequence[nn.Module],
        save_dir: str | Path,
        model_names: Sequence[str] | None = None,
        plot_norm: str = "max",
        plot_percentile: float = 99.0,
    ):
        self.models = list(models)
        self.model_names = (
            list(model_names)
            if model_names is not None
            else [f"model_{i}" for i in range(len(self.models))]
        )

        if len(self.models) != len(self.model_names):
            raise ValueError("models and model_names must have the same length.")
        if plot_norm not in {"max", "percentile"}:
            raise ValueError("plot_norm must be 'max' or 'percentile'.")

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.plot_norm = plot_norm
        self.plot_percentile = plot_percentile
        self.plot_norms = ["global", "by_model", "by_row", "by_column"]

        self.group_names = [
            "input_patch_channel",
            "input_pos_grid",
            "block_path",
            "attention_circuit_head",
            "mlp_memory_group",
            "residual_io_channel",
            "attention_circuit_dim",
            "mlp_memory_unit",
        ]

        self.labels = {
            "block_path": ["LN1", "QK", "OV", "LN2", "MLP key", "MLP value"],
            "attention_circuit": ["QK", "OV"],
            "mlp_memory": ["key", "value"],
            "residual_io": ["read", "write"],
        }

    @staticmethod
    def _get_x(batch):
        if isinstance(batch, torch.Tensor):
            return batch
        if isinstance(batch, dict):
            for k in ("x", "image", "images", "pixel_values"):
                if k in batch:
                    return batch[k]
            return next(iter(batch.values()))
        return batch[0]

    @staticmethod
    def _get_logits(model: nn.Module, x: torch.Tensor):
        out = model(x)
        if isinstance(out, dict):
            return out["logits"]
        return out[0] if isinstance(out, (tuple, list)) else out

    @staticmethod
    def _get_backbone(model: nn.Module):
        for m in (model, getattr(model, "module", None)):
            if m is not None and hasattr(m, "aggregate_fim_grads"):
                return m
            b = getattr(m, "backbone", None) if m is not None else None
            if b is not None and hasattr(b, "aggregate_fim_grads"):
                return b
        raise AttributeError(
            "Expected model or model.backbone to implement aggregate_fim_grads()."
        )

    @staticmethod
    def _snapshot_requires_grad(model: nn.Module):
        return {p: p.requires_grad for p in model.parameters()}

    @staticmethod
    def _restore_requires_grad(snapshot):
        for p, v in snapshot.items():
            p.requires_grad_(v)

    @staticmethod
    def _prepare_for_fim(model: nn.Module, backbone: nn.Module):
        for p in model.parameters():
            p.requires_grad_(False)
        for p in backbone.parameters():
            p.requires_grad_(True)

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

        mode_snapshot = {m: m.training for m in model.modules()}
        model.eval()

        backbone = self._get_backbone(model)

        grad_snapshot = self._snapshot_requires_grad(model)
        self._prepare_for_fim(model, backbone)

        sums = None
        n_steps = 0
        n_samples = 0
        fim_den = 0

        pbar = tqdm(loader, desc=f"[FIM] {model_name}", unit="batch", leave=False)

        try:
            with torch.enable_grad():
                for batch_idx, batch in enumerate(pbar):
                    if max_batches is not None and batch_idx >= max_batches:
                        break

                    x = self._get_x(batch).to(device, non_blocking=True)
                    batch_size = x.size(0)
                    n_samples += batch_size

                    for _ in range(mc_samples):
                        model.zero_grad(set_to_none=True)

                        logits = self._get_logits(model, x)

                        with torch.no_grad():
                            y = torch.distributions.Categorical(
                                logits=logits.detach()
                            ).sample()

                        loss = F.cross_entropy(logits, y, reduction="sum")
                        loss.backward()

                        reps = backbone.aggregate_fim_grads()

                        if sums is None:
                            sums = {
                                k: torch.zeros_like(reps[k]) for k in self.group_names
                            }

                        for k in self.group_names:
                            sums[k].add_(reps[k].detach())

                        n_steps += 1
                        fim_den += batch_size

                    pbar.set_postfix(
                        steps=n_steps,
                        samples=n_samples,
                        effective=fim_den,
                    )

        finally:
            model.zero_grad(set_to_none=True)
            self._restore_requires_grad(grad_snapshot)

            for module, was_training in mode_snapshot.items():
                module.train(was_training)

            pbar.close()

        if n_steps == 0:
            raise RuntimeError("No FIM steps were computed. Check loader/max_batches.")

        reps = {k: (sums[k] / fim_den).cpu().numpy() for k in self.group_names}

        summary = {
            "model_name": model_name,
            "n_samples_seen": n_samples,
            "n_effective_fim_samples": fim_den,
            "n_fim_steps": n_steps,
            "mc_samples": mc_samples,
        }

        for k, a in reps.items():
            summary[k] = {
                "shape": list(a.shape),
                "min": float(a.min()),
                "max": float(a.max()),
                "mean": float(a.mean()),
                "l1": float(np.abs(a).sum()),
                "l2": float(np.sqrt((a**2).sum())),
            }

        tqdm.write(
            f"FIM done • {model_name} • steps={n_steps} • "
            f"samples={n_samples} • effective={fim_den}"
        )

        return reps, summary

    def _scale(self, x: np.ndarray):
        x = x.reshape(-1)
        s = (
            float(x.max())
            if self.plot_norm == "max"
            else float(np.percentile(x, self.plot_percentile))
        )
        return s if s > 0 else 1.0

    def _normalize(self, x: np.ndarray, norm: str, global_scale: float | None):
        if norm == "global":
            s = global_scale
        elif norm == "by_model":
            s = self._scale(x)
        elif norm == "by_row":
            s = (
                x.max(1, keepdims=True)
                if self.plot_norm == "max"
                else np.percentile(x, self.plot_percentile, axis=1, keepdims=True)
            )
        elif norm == "by_column":
            s = (
                x.max(0, keepdims=True)
                if self.plot_norm == "max"
                else np.percentile(x, self.plot_percentile, axis=0, keepdims=True)
            )
        else:
            raise ValueError(f"Unknown normalization: {norm}")

        return np.clip(x / np.where(np.asarray(s) <= 0, 1.0, s), 0.0, 1.0)

    @staticmethod
    def _safe(x: str):
        return x.lower().replace(" ", "_").replace("/", "_")

    def _plot_arrays(
        self,
        arrays: Sequence[np.ndarray],
        title: str,
        filename: str,
        xlabel: str,
        norm: str,
        ylabel: str | None = None,
        yticklabels: Sequence[str] | None = None,
    ):
        arrays = [a if a.ndim == 2 else a.reshape(1, -1) for a in arrays]
        global_scale = (
            self._scale(np.concatenate([a.reshape(-1) for a in arrays]))
            if norm == "global"
            else None
        )

        ncols = 2 if len(self.model_names) <= 4 else 3
        nrows = math.ceil(len(self.model_names) / ncols)
        height = 3.5 if arrays[0].shape[0] > 1 else 2.2

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4.5 * ncols + 0.7, height * nrows), squeeze=False
        )
        im = None

        for i, (model_name, arr) in enumerate(zip(self.model_names, arrays)):
            ax = axes[i // ncols, i % ncols]
            data = self._normalize(arr.astype(np.float64), norm, global_scale)

            im = ax.imshow(data, aspect="auto", vmin=0.0, vmax=1.0)
            ax.set_title(model_name)
            ax.set_xlabel(xlabel)

            if ylabel:
                ax.set_ylabel(ylabel)
            if yticklabels is not None:
                ax.set_yticks(np.arange(len(yticklabels)))
                ax.set_yticklabels(yticklabels)
            elif data.shape[0] == 1:
                ax.set_yticks([])

        for j in range(len(self.model_names), nrows * ncols):
            axes[j // ncols, j % ncols].axis("off")

        fig.suptitle(f"{title} | {norm}", y=0.98)
        fig.subplots_adjust(
            left=0.08, right=0.88, top=0.88, bottom=0.10, wspace=0.35, hspace=0.55
        )

        cbar_ax = fig.add_axes([0.91, 0.18, 0.025, 0.64])
        fig.colorbar(im, cax=cbar_ax, label=f"Normalized FIM ({norm})")

        out_dir = self.save_dir / "plots" / norm
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / filename, bbox_inches="tight")
        plt.close(fig)

    def _plot_all_norms(
        self, arrays, title, filename, xlabel, ylabel=None, yticklabels=None
    ):
        for norm in self.plot_norms:
            self._plot_arrays(
                arrays, title, filename, xlabel, norm, ylabel, yticklabels
            )

    def _plot_all(self, reps: Dict[str, Dict[str, np.ndarray]]):
        r = lambda k: [reps[m][k] for m in self.model_names]

        self._plot_all_norms(
            r("input_patch_channel"),
            "FIM by input patch channel",
            "input_patch_channel.png",
            "Patch embedding channel",
        )
        self._plot_all_norms(
            r("input_pos_grid"),
            "FIM by input positional grid",
            "input_pos_grid.png",
            "Patch column",
            "Patch row",
        )

        self._plot_all_norms(
            [reps[m]["block_path"].T for m in self.model_names],
            "FIM by block path",
            "block_path.png",
            "Transformer block",
            "Path",
            self.labels["block_path"],
        )

        for i, label in enumerate(self.labels["attention_circuit"]):
            self._plot_all_norms(
                [reps[m]["attention_circuit_head"][:, :, i] for m in self.model_names],
                f"FIM by attention {label} circuit",
                f"attention_circuit_head_{self._safe(label)}.png",
                "Attention head",
                "Transformer block",
            )

        self._plot_all_norms(
            [reps[m]["attention_circuit_head"].mean(1).T for m in self.model_names],
            "FIM by attention circuit and block",
            "attention_circuit_head_block_mean.png",
            "Transformer block",
            "Circuit",
            self.labels["attention_circuit"],
        )

        for i, label in enumerate(self.labels["mlp_memory"]):
            self._plot_all_norms(
                [reps[m]["mlp_memory_group"][:, :, i] for m in self.model_names],
                f"FIM by MLP memory {label}",
                f"mlp_memory_group_{self._safe(label)}.png",
                "MLP group",
                "Transformer block",
            )

        self._plot_all_norms(
            [reps[m]["mlp_memory_group"].mean(1).T for m in self.model_names],
            "FIM by MLP memory role and block",
            "mlp_memory_group_block_mean.png",
            "Transformer block",
            "MLP role",
            self.labels["mlp_memory"],
        )

        self._plot_all_norms(
            [reps[m]["residual_io_channel"].mean(2).T for m in self.model_names],
            "FIM by residual read/write and block",
            "residual_io_channel_block_mean.png",
            "Transformer block",
            "Residual role",
            self.labels["residual_io"],
        )

    def _stack(self, reps: Dict[str, Dict[str, np.ndarray]], name: str):
        return np.stack([reps[m][name] for m in self.model_names], axis=0)

    def _save_json(self, obj: Dict[str, Any], filename: str):
        with open(self.save_dir / filename, "w") as f:
            json.dump(obj, f, indent=2)

    def _save_npz(self, reps: Dict[str, Dict[str, np.ndarray]]):
        np.savez_compressed(
            self.save_dir / "fim_representations.npz",
            model_names=np.array(self.model_names),
            group_names=np.array(self.group_names),
            block_path_labels=np.array(self.labels["block_path"]),
            attention_circuit_labels=np.array(self.labels["attention_circuit"]),
            mlp_memory_labels=np.array(self.labels["mlp_memory"]),
            residual_io_labels=np.array(self.labels["residual_io"]),
            **{k: self._stack(reps, k) for k in self.group_names},
        )

    def _save_summary(self, reps, summaries):
        self._save_json(
            {
                "estimator": "monte_carlo_diagonal_fisher",
                "groups": self.group_names,
                "labels": self.labels,
                "plot_normalizations": self.plot_norms,
                "plot_norm": self.plot_norm,
                "plot_percentile": self.plot_percentile,
                "saved_arrays": {
                    k: list(self._stack(reps, k).shape) for k in self.group_names
                },
                "models": summaries,
            },
            "fim_summary.json",
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
        reps, summaries = {}, []

        for model, model_name in zip(self.models, self.model_names):
            model_reps, summary = self._compute_one_model(
                model=model,
                loader=loader,
                device=device,
                model_name=model_name,
                max_batches=max_batches,
                mc_samples=mc_samples,
            )
            reps[model_name] = model_reps
            summaries.append(summary)

        if save:
            self._save_npz(reps)
            self._save_summary(reps, summaries)

        if plot:
            self._plot_all(reps)

        return reps
