from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class MonteCarloFIMEstimator:
    """
    Assumes:
        linear_probe = (name, model)
        prompt_models = [(name, model), ...]

        backbone.aggregate_fim_grads() -> dict[str, torch.Tensor]
        prompt_models[i].init_prompts() -> None

    Computes:
        raw:
            [linear probe, learned prompt models]

        adapt_mag:
            learned prompt FIM - linear probe FIM

        adapt_dir:
            L1-normalized learned prompt FIM - L1-normalized linear probe FIM

        prompt_mag:
            learned prompt FIM - prompt-initialized FIM, same VPT model/head

        prompt_dir:
            L1-normalized learned prompt FIM - L1-normalized prompt-initialized FIM

    Computed FIM groups:
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

    Direction normalization is always within each primitive FIM group.
    """

    group_names = [
        "input_patch_channel",
        "input_pos_grid",
        "block_path",
        "attention_circuit_head",
        "mlp_memory_group",
        "residual_io_channel",
        "attention_circuit_dim",
        "mlp_memory_unit",
    ]

    view_names = ["raw", "adapt_mag", "adapt_dir", "prompt_mag", "prompt_dir"]
    plot_norms = ["global", "by_model", "by_row", "by_column"]

    def __init__(
        self,
        linear_probe: tuple[str, nn.Module],
        prompt_models: Sequence[tuple[str, nn.Module]],
        save_dir: str | Path,
        plot_norm: str = "max",
        plot_percentile: float = 99.0,
        eps: float = 1e-12,
    ):
        self.lp_name, self.lp_model = linear_probe
        self.prompt_models = list(prompt_models)
        self.prompt_names = [n for n, _ in self.prompt_models]
        self.raw_names = [self.lp_name] + self.prompt_names

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.plot_norm = plot_norm
        self.plot_percentile = plot_percentile
        self.eps = eps

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
        model = getattr(model, "module", model)

        if not hasattr(model, "backbone"):
            raise AttributeError("model/module must have a backbone attribute")

        backbone = model.backbone

        if not hasattr(backbone, "aggregate_fim_grads"):
            raise AttributeError("model.backbone must implement aggregate_fim_grads()")

        return backbone

    @staticmethod
    def _set_fim_grads(model: nn.Module, backbone: nn.Module):
        old = {p: p.requires_grad for p in model.parameters()}
        for p in model.parameters():
            p.requires_grad_(False)
        for p in backbone.parameters():
            p.requires_grad_(True)
        return old

    @staticmethod
    def _restore_grads(old):
        for p, v in old.items():
            p.requires_grad_(v)

    @staticmethod
    def _init_prompt(model: nn.Module):
        model = getattr(model, "module", model)
        if not hasattr(model, "init_prompt"):
            raise AttributeError("model/module must implement init_prompt()")
        model.init_prompt()

    def _compute_one(
        self,
        model: nn.Module,
        name: str,
        loader,
        device: str,
        max_batches: int | None,
        mc_samples: int,
    ):
        model.to(device)
        train_modes = {m: m.training for m in model.modules()}
        model.eval()

        backbone = self._get_backbone(model)
        grad_modes = self._set_fim_grads(model, backbone)

        sums, n_steps, n_seen, fim_den = None, 0, 0, 0
        pbar = tqdm(loader, desc=f"[FIM] {name}", unit="batch", leave=False)

        try:
            with torch.enable_grad():
                for batch_idx, batch in enumerate(pbar):
                    if max_batches is not None and batch_idx >= max_batches:
                        break

                    x = self._get_x(batch).to(device, non_blocking=True)
                    bsz = x.size(0)
                    n_seen += bsz

                    for _ in range(mc_samples):
                        model.zero_grad(set_to_none=True)
                        logits = self._get_logits(model, x)

                        with torch.no_grad():
                            y = torch.distributions.Categorical(
                                logits=logits.detach()
                            ).sample()

                        F.cross_entropy(logits, y, reduction="sum").backward()
                        reps = backbone.aggregate_fim_grads()

                        if sums is None:
                            sums = {
                                k: torch.zeros_like(reps[k]) for k in self.group_names
                            }

                        for k in self.group_names:
                            sums[k].add_(reps[k].detach())

                        n_steps += 1
                        fim_den += bsz

                    pbar.set_postfix(steps=n_steps, samples=n_seen, effective=fim_den)

        finally:
            model.zero_grad(set_to_none=True)
            self._restore_grads(grad_modes)
            for m, was_training in train_modes.items():
                m.train(was_training)
            pbar.close()

        if sums is None:
            raise RuntimeError("No FIM steps were computed.")

        tqdm.write(
            f"FIM done • {name} • steps={n_steps} • samples={n_seen} • effective={fim_den}"
        )
        return {k: (v / fim_den).cpu().numpy() for k, v in sums.items()}

    def _dir(self, x: np.ndarray):
        return x / (np.abs(x).sum() + self.eps)

    def _make_views(self, lp_raw, prompt_raw, pinit_raw):
        views = {v: {} for v in self.view_names}

        for k in self.group_names:
            lp = lp_raw[k]
            prompts = [prompt_raw[n][k] for n in self.prompt_names]
            pinits = [pinit_raw[n][k] for n in self.prompt_names]

            views["raw"][k] = np.stack([lp] + prompts)
            views["adapt_mag"][k] = np.stack([p - lp for p in prompts])
            views["adapt_dir"][k] = np.stack(
                [self._dir(p) - self._dir(lp) for p in prompts]
            )
            views["prompt_mag"][k] = np.stack([p - q for p, q in zip(prompts, pinits)])
            views["prompt_dir"][k] = np.stack(
                [self._dir(p) - self._dir(q) for p, q in zip(prompts, pinits)]
            )

        return views

    def _scale(self, x: np.ndarray, signed: bool):
        x = np.abs(x).reshape(-1) if signed else x.reshape(-1)
        s = (
            np.percentile(x, self.plot_percentile)
            if self.plot_norm == "percentile"
            else x.max()
        )
        return float(s) if s > 0 else 1.0

    def _norm_plot(
        self, x: np.ndarray, mode: str, signed: bool, global_scale: float | None
    ):
        base = np.abs(x) if signed else x

        if mode == "global":
            s = global_scale
        elif mode == "by_model":
            s = self._scale(x, signed)
        elif mode == "by_row":
            s = (
                np.percentile(base, self.plot_percentile, axis=1, keepdims=True)
                if self.plot_norm == "percentile"
                else base.max(1, keepdims=True)
            )
        elif mode == "by_column":
            s = (
                np.percentile(base, self.plot_percentile, axis=0, keepdims=True)
                if self.plot_norm == "percentile"
                else base.max(0, keepdims=True)
            )
        else:
            s = 1.0

        y = x / np.where(np.asarray(s) <= 0, 1.0, s)
        return np.clip(y, -1.0, 1.0) if signed else np.clip(y, 0.0, 1.0)

    @staticmethod
    def _safe(x: str):
        return x.lower().replace(" ", "_").replace("/", "_")

    def _plot_arrays(
        self,
        arrays: np.ndarray,
        names: Sequence[str],
        view: str,
        title: str,
        filename: str,
        xlabel: str,
        ylabel: str | None = None,
        yticklabels: Sequence[str] | None = None,
    ):
        arrays = [a if a.ndim == 2 else a.reshape(1, -1) for a in arrays]
        signed = view != "raw"

        for norm in self.plot_norms:
            global_scale = (
                self._scale(np.concatenate([a.reshape(-1) for a in arrays]), signed)
                if norm == "global"
                else None
            )
            ncols = 2 if len(names) <= 4 else 3
            nrows = math.ceil(len(names) / ncols)
            height = 3.5 if arrays[0].shape[0] > 1 else 2.2

            fig, axes = plt.subplots(
                nrows, ncols, figsize=(4.5 * ncols + 0.7, height * nrows), squeeze=False
            )
            im = None

            for i, (name, arr) in enumerate(zip(names, arrays)):
                ax = axes[i // ncols, i % ncols]
                data = self._norm_plot(
                    arr.astype(np.float64), norm, signed, global_scale
                )

                im = ax.imshow(
                    data,
                    aspect="auto",
                    vmin=-1.0 if signed else 0.0,
                    vmax=1.0,
                    cmap="coolwarm" if signed else None,
                )
                ax.set_title(name)
                ax.set_xlabel(xlabel)

                if ylabel:
                    ax.set_ylabel(ylabel)
                if yticklabels is not None:
                    ax.set_yticks(np.arange(len(yticklabels)))
                    ax.set_yticklabels(yticklabels)
                elif data.shape[0] == 1:
                    ax.set_yticks([])

            for j in range(len(names), nrows * ncols):
                axes[j // ncols, j % ncols].axis("off")

            fig.suptitle(f"{view}: {title} | {norm}", y=0.98)
            fig.subplots_adjust(
                left=0.08, right=0.88, top=0.88, bottom=0.10, wspace=0.35, hspace=0.55
            )

            cbar_ax = fig.add_axes([0.91, 0.18, 0.025, 0.64])
            fig.colorbar(
                im,
                cax=cbar_ax,
                label="Signed normalized delta" if signed else "Normalized FIM",
            )

            out_dir = self.save_dir / "plots" / view / norm
            out_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_dir / filename, bbox_inches="tight")
            plt.close(fig)

    def _plot_view(self, view: str, reps: dict[str, np.ndarray]):
        names = self.raw_names if view == "raw" else self.prompt_names

        self._plot_arrays(
            reps["input_patch_channel"],
            names,
            view,
            "FIM by input patch channel",
            "input_patch_channel.png",
            "Patch embedding channel",
        )
        self._plot_arrays(
            reps["input_pos_grid"],
            names,
            view,
            "FIM by input positional grid",
            "input_pos_grid.png",
            "Patch column",
            "Patch row",
        )

        self._plot_arrays(
            reps["block_path"].transpose(0, 2, 1),
            names,
            view,
            "FIM by block path",
            "block_path.png",
            "Transformer block",
            "Path",
            ["LN1", "QK", "OV", "LN2", "MLP key", "MLP value"],
        )

        for i, label in enumerate(("QK", "OV")):
            self._plot_arrays(
                reps["attention_circuit_head"][:, :, :, i],
                names,
                view,
                f"FIM by attention {label} circuit",
                f"attention_circuit_head_{self._safe(label)}.png",
                "Attention head",
                "Transformer block",
            )

        self._plot_arrays(
            reps["attention_circuit_head"].mean(2).transpose(0, 2, 1),
            names,
            view,
            "FIM by attention circuit and block",
            "attention_circuit_head_block_mean.png",
            "Transformer block",
            "Circuit",
            ["QK", "OV"],
        )

        for i, label in enumerate(("key", "value")):
            self._plot_arrays(
                reps["mlp_memory_group"][:, :, :, i],
                names,
                view,
                f"FIM by MLP memory {label}",
                f"mlp_memory_group_{self._safe(label)}.png",
                "MLP group",
                "Transformer block",
            )

        self._plot_arrays(
            reps["mlp_memory_group"].mean(2).transpose(0, 2, 1),
            names,
            view,
            "FIM by MLP memory role and block",
            "mlp_memory_group_block_mean.png",
            "Transformer block",
            "MLP role",
            ["key", "value"],
        )

        self._plot_arrays(
            reps["residual_io_channel"].mean(3).transpose(0, 2, 1),
            names,
            view,
            "FIM by residual read/write and block",
            "residual_io_channel_block_mean.png",
            "Transformer block",
            "Residual role",
            ["read", "write"],
        )

    def _save_npz(self, views):
        arrays = {
            "linear_probe_name": np.array(self.lp_name),
            "prompt_model_names": np.array(self.prompt_names),
            "view_names": np.array(self.view_names),
            "group_names": np.array(self.group_names),
        }

        for view in self.view_names:
            for group in self.group_names:
                arrays[f"{view}__{group}"] = views[view][group]

        np.savez_compressed(self.save_dir / "fim_representations.npz", **arrays)

    def compute(
        self,
        loader,
        device: str = "cuda",
        max_batches: int | None = None,
        mc_samples: int = 1,
        save: bool = True,
        plot: bool = True,
    ):
        lp_raw = self._compute_one(
            self.lp_model, self.lp_name, loader, device, max_batches, mc_samples
        )

        prompt_raw = {
            name: self._compute_one(
                model, name, loader, device, max_batches, mc_samples
            )
            for name, model in self.prompt_models
        }

        pinit_raw = {}
        for name, model in self.prompt_models:
            m = copy.deepcopy(model)
            self._init_prompt(m)
            pinit_raw[name] = self._compute_one(
                m, f"{name}__pinit", loader, device, max_batches, mc_samples
            )
            del m
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

        views = self._make_views(lp_raw, prompt_raw, pinit_raw)

        if save:
            self._save_npz(views)

        if plot:
            for view in self.view_names:
                self._plot_view(view, views[view])

        return views
