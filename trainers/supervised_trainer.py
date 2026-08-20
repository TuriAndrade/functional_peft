import json
from pathlib import Path
from typing import Sequence, Dict, Any
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np


# ------- SupervisedTrainer -------
class SupervisedTrainer:
    def __init__(
        self,
        model: nn.Module,
        save_dir: str | Path,
    ):
        self.model: nn.Module = model

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _snapshot_state_dict(self) -> Dict[str, torch.Tensor]:
        # Detach, move to CPU, and clone so it won't be mutated by further training
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def _save_json(self, obj: Dict[str, Any], filename: str):
        with open(self.save_dir / filename, "w") as f:
            json.dump(obj, f, indent=2)

    def _run_epoch(
        self,
        loader,
        optimizer,
        lr_scheduler,
        wd_scheduler,
        device,
        epoch,
        accum_steps,
        train,
    ):
        phase = "Train" if train else "Val"
        self.model.train(mode=train)

        with torch.set_grad_enabled(train):
            total_loss, total_loss_components, n = 0.0, {}, 0
            correct = 0

            pbar = tqdm(
                loader,
                desc=f"[{phase}] Epoch {epoch}",
                unit="batch",
                leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

            new_lr = 0
            new_wd = 0

            if train:
                optimizer.zero_grad(set_to_none=True)

            n_batches = len(loader)
            for i, batch in enumerate(pbar):
                x, y = batch[0].to(device, non_blocking=True), batch[1].to(
                    device, non_blocking=True
                )

                logits, loss, loss_components = self.model.compute_loss(x, y)
                raw_loss = loss.item()

                pred = logits.argmax(dim=1)
                correct += pred.eq(y).sum().item()

                if train:
                    loss = loss / accum_steps
                    loss.backward()

                    if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                        if lr_scheduler is not None:
                            new_lr = lr_scheduler.step()
                        if wd_scheduler is not None:
                            new_wd = wd_scheduler.step()

                bs = y.size(0)
                total_loss += raw_loss * bs
                for k, v in loss_components.items():
                    total_loss_components.setdefault(k, []).append(v * bs)
                n += bs

                running = total_loss / max(n, 1)
                running_acc = (correct / max(n, 1)) * 100.0
                pbar.set_postfix_str(f"loss={running:.4f}, acc={running_acc:.2f}")

            avg_acc = (correct / max(n, 1)) * 100.0

            components = {
                k: sum(v) / max(n, 1) for k, v in total_loss_components.items()
            }
            components["acc"] = avg_acc

            if train:
                if lr_scheduler is not None:
                    components.update({"lr": new_lr})
                if wd_scheduler is not None:
                    components.update({"wd": new_wd})

            pbar.close()
            tqdm.write(f"Epoch {epoch} • {phase:<5} • acc={avg_acc:.2f}")

            return avg_acc, components

    def _plot_hist(self, hist: list):
        if not hist:
            return

        keys = sorted(
            {k for rec in hist for k in rec.get("train_hist", {}).keys()}
            | {k for rec in hist for k in rec.get("val_hist", {}).keys()}
        )
        if not keys:
            return

        # Subplot grid
        n = len(keys)
        ncols = 2 if n <= 4 else 3
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False
        )

        for i, key in enumerate(keys):
            ax = axes[i // ncols, i % ncols]

            train_epochs, train_values = [], []
            val_epochs, val_values = [], []

            for rec in hist:
                epoch = rec["epoch"]

                train_rec = rec.get("train_hist", {})
                val_rec = rec.get("val_hist", {})

                if key in train_rec:
                    train_epochs.append(epoch)
                    train_values.append(train_rec[key])

                if key in val_rec:
                    val_epochs.append(epoch)
                    val_values.append(val_rec[key])

            if train_values:
                ax.plot(train_epochs, train_values, label="train")
            if val_values:
                ax.plot(val_epochs, val_values, label="val")

            ax.set_title(key.replace("_", " ").title())
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide any unused subplots
        for j in range(i + 1, nrows * ncols):
            axes[j // ncols, j % ncols].axis("off")

        plt.tight_layout()
        plt.savefig(self.save_dir / "loss_curves.png")
        plt.close(fig)

    def train(
        self,
        train_loader,
        val_loader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        wd_scheduler: Any,
        epochs: int = 10,
        min_epochs: int = 0,
        accum_steps: int = 1,
        early_stopping: bool = True,
        patience: int = 5,
        plot_interval: int = 5,
        device: str = "cuda",
        save_ckpt: bool = False,
        save_ckpt_interval: int = 0,
    ):
        hist = []
        wait = 0
        best_val_acc = -float("inf")
        best_epoch = -1
        use_validation = val_loader is not None
        best_state_dict = None

        self.model.to(device)
        if save_ckpt and (save_ckpt_interval > 0):
            self.model.save(self.save_dir / "model_0.pt")

        for epoch in range(1, epochs + 1):
            train_acc, train_components = self._run_epoch(
                train_loader,
                optimizer,
                lr_scheduler,
                wd_scheduler,
                device,
                epoch,
                accum_steps=accum_steps,
                train=True,
            )

            if use_validation:
                val_acc, val_components = self._run_epoch(
                    val_loader,
                    optimizer,
                    lr_scheduler,
                    wd_scheduler,
                    device,
                    epoch,
                    accum_steps=accum_steps,
                    train=False,
                )
            else:
                val_acc, val_components = None, {}

            # log & persist history
            rec = {
                "epoch": epoch,
                "train_hist": train_components,
                "val_hist": val_components,
            }
            hist.append(rec)
            self._save_json({"history": hist}, "history.json")

            if epoch % plot_interval == 0:
                self._plot_hist(hist)

            if (
                save_ckpt
                and (save_ckpt_interval > 0)
                and (epoch % save_ckpt_interval == 0)
            ):
                self.model.save(self.save_dir / f"model_{epoch}.pt")

            if use_validation:
                if val_acc > best_val_acc:
                    wait = 0
                    best_val_acc = val_acc
                    best_epoch = epoch
                    best_state_dict = self._snapshot_state_dict()
                else:
                    wait += 1

                if early_stopping and wait >= patience and epoch > min_epochs:
                    tqdm.write(
                        f"Early stopping at epoch {epoch} "
                        f"(no val improvement for {patience} epochs). "
                        f"Best: epoch {best_epoch} with "
                        f"{'val_acc'}={best_val_acc:.4f}"
                    )
                    break

        self._plot_hist(hist)

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        if save_ckpt:
            self.model.save(self.save_dir / "model.pt")

        self._save_json(
            {
                "epochs": epoch,
                "use_validation": use_validation,
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
            },
            "train_summary.json",
        )

    def evaluate_acc(
        self,
        loader,
        save_name: str,
        topk: Sequence[int] = (1,),
        device: str = "cuda",
        bootstrap: bool = False,
        bootstrap_num_samples: int = 1000,
        bootstrap_ci: float = 95.0,
        bootstrap_seed: int = 0,
    ):
        ks = sorted(set(int(k) for k in topk if k >= 1))
        correct = {k: 0 for k in ks}
        n = 0

        # store per-sample correctness for bootstrap
        bootstrap_hits = {k: [] for k in ks} if bootstrap else None

        self.model.to(device).eval()
        pbar = tqdm(
            loader,
            desc="[AccEval]",
            unit="batch",
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = self.model(x)  # (B, C)
            max_k = max(ks)
            max_k = min(max_k, logits.size(1))  # clamp k to num classes
            _, pred = logits.topk(max_k, dim=1)  # (B, max_k)
            pred = pred.t()  # (max_k, B)
            correct_mat = pred.eq(y.view(1, -1))  # (max_k, B)

            bs = y.size(0)
            n += bs
            for k in ks:
                k_eff = min(k, logits.size(1))
                hits = correct_mat[:k_eff].any(dim=0).float()  # (B,)
                correct[k] += hits.sum().item()

                if bootstrap:
                    bootstrap_hits[k].append(hits.detach().cpu())

        acc = {f"top{k}": (correct[k] / max(n, 1)) * 100.0 for k in ks}
        parts = [f"top{k}={acc[f'top{k}']:.2f}%" for k in ks]
        tqdm.write(f"Accuracy eval • n={n} • " + " • ".join(parts))

        out = {"n_samples": n, "accuracy": acc}

        if bootstrap and n > 0:
            rng = np.random.default_rng(bootstrap_seed)
            alpha = 100.0 - bootstrap_ci
            lower_q = alpha / 2.0
            upper_q = 100.0 - alpha / 2.0

            bootstrap_out = {}
            for k in ks:
                hits = torch.cat(bootstrap_hits[k], dim=0).numpy()  # (N,)
                samples = np.empty(bootstrap_num_samples, dtype=np.float64)

                bootstrap_pbar = tqdm(
                    range(bootstrap_num_samples),
                    desc=f"[Bootstrap top{k}]",
                    unit="sample",
                    leave=False,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                )
                for i in bootstrap_pbar:
                    idx = rng.integers(0, len(hits), size=len(hits))
                    samples[i] = hits[idx].mean() * 100.0

                bootstrap_out[f"top{k}"] = {
                    "mean": float(samples.mean()),
                    "ci_lower": float(np.percentile(samples, lower_q)),
                    "ci_upper": float(np.percentile(samples, upper_q)),
                    "ci_level": bootstrap_ci,
                    "num_samples": bootstrap_num_samples,
                }

            out["bootstrap"] = bootstrap_out

        self._save_json(out, save_name)
        return out
