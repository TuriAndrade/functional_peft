import csv
import gc
import itertools
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
from scipy.stats import rankdata
from tqdm.auto import tqdm

from datasets import datasets
from models import models
from utils import seed_worker

# -----------------------------------------------------------------------------
# Numerical helpers
# -----------------------------------------------------------------------------


def _upper(x):
    x = np.asarray(x, float)
    return x[np.triu_indices(x.shape[0], 1)]


def _finite_mean(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else np.nan


def _rank_vector(values):
    values = np.asarray(values, float)
    out = np.full(values.shape, np.nan)
    keep = np.isfinite(values)
    if keep.any():
        out[keep] = rankdata(values[keep])
    return out


def _spearman(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def _row_corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x = x - x.mean(1, keepdims=True)
    y = y - y.mean()
    den = np.linalg.norm(x, axis=1) * np.linalg.norm(y)
    return np.divide(x @ y, den, out=np.full(x.shape[0], np.nan), where=den > 1e-12)


def _nn_match(a, b):
    """Tie-aware expected exact nearest-neighbor match, averaged over anchors."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    scores = []
    for i in range(a.shape[0]):
        valid = np.isfinite(a[i]) & np.isfinite(b[i])
        valid[i] = False
        if not valid.any():
            continue
        amin, bmin = np.min(a[i, valid]), np.min(b[i, valid])
        an = np.flatnonzero(valid & np.isclose(a[i], amin, rtol=1e-10, atol=1e-12))
        bn = np.flatnonzero(valid & np.isclose(b[i], bmin, rtol=1e-10, atol=1e-12))
        scores.append(len(np.intersect1d(an, bn)) / (len(an) * len(bn)))
    return _finite_mean(scores)


def _triplet_agreement(a, b):
    """Fraction of strict anchored distance orderings shared by two RDMs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    hits, total = 0, 0
    for i in range(a.shape[0]):
        others = [j for j in range(a.shape[0]) if j != i]
        for j, k in itertools.combinations(others, 2):
            av = (a[i, j], a[i, k])
            bv = (b[i, j], b[i, k])
            if not np.all(np.isfinite((*av, *bv))):
                continue
            if np.isclose(av[0], av[1], rtol=1e-10, atol=1e-12) or np.isclose(
                bv[0], bv[1], rtol=1e-10, atol=1e-12
            ):
                continue
            hits += (av[0] < av[1]) == (bv[0] < bv[1])
            total += 1
    return float(hits / total) if total else np.nan


def _bh(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    if not idx.size:
        return out
    order = idx[np.argsort(p[idx])]
    q = p[order] * len(order) / np.arange(1, len(order) + 1)
    out[order] = np.minimum(np.minimum.accumulate(q[::-1])[::-1], 1.0)
    return out


def _write_csv(path, rows, fieldnames=None):
    if not rows:
        raise ValueError(f"No rows for {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: (
                        "NA"
                        if isinstance(v, (float, np.floating)) and not np.isfinite(v)
                        else v
                    )
                    for k, v in row.items()
                }
            )
    return path


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _flatten(x):
    x = np.asarray(x, np.float64)
    return x.reshape(*x.shape[:3], -1)


def _rdms_from_vectors(x):
    """Cross-seed central RDM and same-seed replicate RDMs."""
    x = np.asarray(x, np.float64)
    if x.ndim != 4:
        raise ValueError(f"Expected [task, method, seed, coordinate], got {x.shape}")
    t, m, s, d = x.shape
    if d < 1:
        raise ValueError("Functional signature has no coordinates")

    mean_rdm = np.zeros((t, m, m), np.float64)
    seed_rdms = np.zeros((t, s, m, m), np.float64)
    for i, j in itertools.combinations(range(m), 2):
        diff = x[:, i, :, None] - x[:, j, None, :]
        dist = np.sqrt(np.mean(np.sum(diff * diff, axis=-1) / d, axis=(1, 2)))
        mean_rdm[:, i, j] = mean_rdm[:, j, i] = dist

        same = x[:, i] - x[:, j]
        dist_s = np.sqrt(np.sum(same * same, axis=-1) / d)
        seed_rdms[:, :, i, j] = dist_s
        seed_rdms[:, :, j, i] = dist_s
    return mean_rdm, seed_rdms


def _rank_pool(rdms):
    vectors = np.stack([_upper(rdm) for rdm in np.asarray(rdms, float)])
    keep = np.all(np.isfinite(vectors), axis=0)
    out = np.full(vectors.shape[1], np.nan)
    if keep.any():
        out[keep] = np.stack([rankdata(row[keep]) for row in vectors]).mean(0)
    return out


def _noise_ceiling(replicate_rdms):
    r = np.asarray(replicate_rdms, float)
    if r.ndim != 3 or r.shape[0] < 2:
        raise ValueError("replicate_rdms must be [replicate, method, method]")
    pooled = _rank_pool(r)
    low, high = [], []
    for s in range(r.shape[0]):
        low.append(_spearman(_upper(r[s]), _rank_pool(np.delete(r, s, axis=0))))
        high.append(_spearman(_upper(r[s]), pooled))
    return _finite_mean(low), _finite_mean(high)


def _bootstrap_mean(values, groups, n_bootstrap, rng, stratified):
    values, groups = np.asarray(values, float), np.asarray(groups)
    keep = np.isfinite(values)
    values, groups = values[keep], groups[keep]
    if not values.size or n_bootstrap <= 0:
        return np.nan, np.nan

    if stratified:
        draws = []
        for g in np.unique(groups):
            pool = values[groups == g]
            idx = rng.integers(0, len(pool), size=(n_bootstrap, len(pool)))
            draws.append(pool[idx])
        samples = np.concatenate(draws, axis=1).mean(1)
    else:
        idx = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
        samples = values[idx].mean(1)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def _format_rho(rho, low, high):
    if not np.isfinite(rho):
        return "NA"
    lo = "NA" if not np.isfinite(low) else f"{low:.2f}"
    hi = "NA" if not np.isfinite(high) else f"{high:.2f}"
    return f"{rho:.2f} [{lo}, {hi}]"


def _format_score(value):
    return "NA" if not np.isfinite(value) else f"{value:.3f}"


# -----------------------------------------------------------------------------
# Validator
# -----------------------------------------------------------------------------


class FunctionalSignaturesValidator:
    """Validate functional-signature geometry against predictive behavior and CKA."""

    NATURAL = (
        "vtab_caltech101",
        "vtab_cifar100",
        "vtab_dtd",
        "vtab_flowers102",
        "vtab_pets",
        "vtab_sun397",
        "vtab_svhn",
    )
    SPECIALIZED = (
        "vtab_eurosat",
        "vtab_resisc45",
        "vtab_patch_camelyon",
        "vtab_retinopathy",
    )
    STRUCTURED = (
        "vtab_clevr_count",
        "vtab_clevr_distance",
        "vtab_dmlab",
        "vtab_kitti_distance",
        "vtab_dsprites_location",
        "vtab_dsprites_orientation",
        "vtab_smallnorb_azimuth",
        "vtab_smallnorb_elevation",
    )
    DATASETS = (*NATURAL, *SPECIALIZED, *STRUCTURED)
    TASK_GROUPS = {
        "natural": NATURAL,
        "specialized": SPECIALIZED,
        "structured": STRUCTURED,
    }

    REFERENCE_METHOD = "linear_probe"
    PEFT_METHODS = ("vpt_shallow", "vpt_deep", "lora", "ssf", "adaptformer")
    ADAPTATION_METHODS = ("full_ft", *PEFT_METHODS)
    SCOPES = {"peft_only": PEFT_METHODS, "peft+ft": ADAPTATION_METHODS}
    SCOPE_DIRS = {"peft_only": "peft_only", "peft+ft": "peft_plus_ft"}
    RESOLUTIONS = ("overall", "layerwise")
    SEEDS = (0, 1, 2)

    RAW_SIGNATURES = ("F_m", "P")
    SIGNATURES = ("F", "S", "P")
    BEHAVIOR_METRICS = ("jsd", "top1_disagreement", "error_dissimilarity")
    ALIGNMENT_SCORES = ("rho", "nn_match", "triplet_agreement")
    PLOT_METRIC = "jsd"
    CKA_FINAL = "CKA final"
    CKA_MEAN = "CKA mean"
    CKA_OBJECTS = (CKA_FINAL, CKA_MEAN)
    CKA_BASELINE = CKA_FINAL

    BASE_COMPONENTS = (
        "residual_carry",
        "sublayer_update",
        "attention_selection_route",
        "attention_transport_route",
        "sublayer_input_channel",
        "attention_head_selection_route",
        "attention_head_transport_route",
        "mlp_neuron",
        "final_cls_readout_channel",
        "residual_token_carry",
        "sublayer_token_update",
        "attention_head_output",
    )
    DERIVED_COMPONENTS = {
        "residual_cls_carry": ("residual_token_carry", 0),
        "residual_patch_carry": ("residual_token_carry", 1),
        "sublayer_cls_update": ("sublayer_token_update", 0),
        "sublayer_patch_update": ("sublayer_token_update", 1),
    }
    CORE_COMPONENTS = (
        "residual_carry",
        "residual_cls_carry",
        "residual_patch_carry",
        "sublayer_update",
        "sublayer_cls_update",
        "sublayer_patch_update",
        "attention_selection_route",
        "attention_transport_route",
    )
    ALL_COMPONENTS = (
        "residual_carry",
        "residual_cls_carry",
        "residual_patch_carry",
        "sublayer_update",
        "sublayer_cls_update",
        "sublayer_patch_update",
        "attention_selection_route",
        "attention_transport_route",
        "sublayer_input_channel",
        "attention_head_selection_route",
        "attention_head_transport_route",
        "mlp_neuron",
        "residual_token_carry",
        "sublayer_token_update",
        "attention_head_output",
        "final_cls_readout_channel",
    )
    NON_LAYERWISE = {"final_cls_readout_channel"}

    EVAL_EXPERIMENT_NAME = "top_1_acc_eval"
    SIGNATURE_FILENAME = "functional_signatures.npz"

    def __init__(
        self,
        backbone_name="vit_b16",
        eps=1e-12,
        n_bootstrap=2000,
        n_global_permutations=10000,
        random_seed=0,
        verbose=True,
    ):
        self.backbone_name = str(backbone_name)
        self.experiments_dir = Path(os.environ.get("EXPERIMENTS_DIR", "./experiments"))
        self.save_dir = (
            self.experiments_dir / "functional_reorganization" / self.backbone_name
        )
        self.alignment_dir = self.save_dir / "alignment"
        self.eps = float(eps)
        self.n_bootstrap = int(n_bootstrap)
        self.n_global_permutations = int(n_global_permutations)
        self.random_seed = int(random_seed)
        self.verbose = bool(verbose)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._task_group = {
            t: g for g, tasks in self.TASK_GROUPS.items() for t in tasks
        }
        self._runs = {}
        self._perm_pairs = {}
        for n in {len(v) for v in self.SCOPES.values()}:
            perms = np.asarray(list(itertools.permutations(range(n))), dtype=int)
            iu, ju = np.triu_indices(n, 1)
            self._perm_pairs[n] = (perms[:, iu], perms[:, ju])

    # ---- progress / runs ------------------------------------------------------

    def _phase(self, index, label):
        if self.verbose:
            tqdm.write(f"[{index}/6] {label}")

    def _progress(self, x, desc, total=None):
        return tqdm(
            x,
            desc=desc,
            total=total,
            disable=not self.verbose,
            leave=False,
            dynamic_ncols=True,
        )

    def _method_dir(self, task, method):
        return (
            self.experiments_dir
            / self.EVAL_EXPERIMENT_NAME
            / self.backbone_name
            / task
            / method
        )

    def _get_runs(self, task, method):
        key = (task, method)
        if key in self._runs:
            return self._runs[key]
        root = self._method_dir(task, method)
        if not root.is_dir():
            raise FileNotFoundError(root)

        runs = []
        for path in root.iterdir():
            cfg, model, acc = (
                path / "config.json",
                path / "model.pt",
                path / "test_acc_eval.json",
            )
            if not path.is_dir() or not (
                cfg.is_file() and model.is_file() and acc.is_file()
            ):
                continue
            with cfg.open() as f:
                config = json.load(f)
            runs.append(
                {"seed": int(config["run_seed"]), "path": path, "config": config}
            )

        runs.sort(key=lambda r: r["seed"])
        found = tuple(r["seed"] for r in runs)
        if found != self.SEEDS:
            raise RuntimeError(
                f"Expected seeds {self.SEEDS} for {task}/{method}, found {found}"
            )
        self._runs[key] = runs
        return runs

    def _components(self, scope):
        return self.ALL_COMPONENTS if scope == "peft_only" else self.CORE_COMPONENTS

    def _empty_sources(self):
        return {
            resolution: {scope: {} for scope in self.SCOPES}
            for resolution in self.RESOLUTIONS
        }

    # ---- functional sources --------------------------------------------------

    def _load_component(self, component):
        """Load one primitive component, preserving its native shape."""
        methods = (self.REFERENCE_METHOD, *self.ADAPTATION_METHODS)
        out = {k: [] for k in self.RAW_SIGNATURES}
        native_shape = None

        for task in self.DATASETS:
            by_sig = {k: {} for k in self.RAW_SIGNATURES}
            for method in methods:
                values = {k: [] for k in self.RAW_SIGNATURES}
                for run in self._get_runs(task, method):
                    path = run["path"] / self.SIGNATURE_FILENAME
                    with np.load(path, allow_pickle=False) as data:
                        for sig in self.RAW_SIGNATURES:
                            key = f"{sig}__{component}"
                            if key not in data:
                                raise KeyError(f"Missing {key!r} in {path}")
                            x = np.asarray(data[key], np.float64)
                            if np.any(x < -self.eps):
                                raise ValueError(f"Negative values in {key!r}: {path}")
                            if native_shape is None:
                                native_shape = x.shape
                            elif x.shape != native_shape:
                                raise ValueError(
                                    f"Shape mismatch for {component}: expected {native_shape}, got {x.shape} in {path}"
                                )
                            values[sig].append(np.log(np.maximum(x, 0.0) + self.eps))
                for sig in self.RAW_SIGNATURES:
                    by_sig[sig][method] = np.stack(values[sig])

            for sig in self.RAW_SIGNATURES:
                lp = by_sig[sig][self.REFERENCE_METHOD].mean(0)
                out[sig].append(
                    np.stack([by_sig[sig][m] - lp for m in self.ADAPTATION_METHODS])
                )

        raw = {k: np.stack(v) for k, v in out.items()}
        signatures = {"F": raw["F_m"], "P": raw["P"]}
        signatures["S"] = signatures["F"] - signatures["P"]
        if not np.allclose(
            signatures["F"], signatures["P"] + signatures["S"], rtol=1e-8, atol=1e-9
        ):
            raise RuntimeError(f"F=P+S identity failed for {component}")
        return signatures, tuple(native_shape)

    def _component_views(self, component, signatures, shape):
        yield component, signatures, shape, None
        derived = [
            (name, token)
            for name, (source, token) in self.DERIVED_COMPONENTS.items()
            if source == component
        ]
        if not derived:
            return
        if not shape or shape[-1] != 2:
            raise RuntimeError(
                f"Expected final CLS/patch axis of size 2 for {component}, got {shape}"
            )
        for name, token in derived:
            yield name, {sig: x[..., token] for sig, x in signatures.items()}, shape[
                :-1
            ], component

    def _reliability_rows(self, scope, object_type, name, seed_rdms, layer=None):
        rows = []
        for ti, task in enumerate(self.DATASETS):
            lo, hi = _noise_ceiling(seed_rdms[ti])
            rows.append(
                {
                    "scope": scope,
                    "object_type": object_type,
                    "distance_object": name,
                    "layer": layer,
                    "task": task,
                    "task_group": self._task_group[task],
                    "reliability_low": lo,
                    "reliability_high": hi,
                }
            )
        return rows

    def functional_sources(self):
        sources = self._empty_sources()
        reliability = {resolution: [] for resolution in self.RESOLUTIONS}
        metadata, n_layers = {}, None
        all_idx = {m: i for i, m in enumerate(self.ADAPTATION_METHODS)}

        bar = self._progress(self.BASE_COMPONENTS, "Functional components")
        for base in bar:
            bar.set_postfix_str(base, refresh=True)
            signatures, shape = self._load_component(base)
            for component, view, view_shape, source in self._component_views(
                base, signatures, shape
            ):
                layerwise = component not in self.NON_LAYERWISE
                if layerwise:
                    if not view_shape:
                        raise RuntimeError(
                            f"Layerwise component {component} has scalar shape"
                        )
                    if n_layers is None:
                        n_layers = int(view_shape[0])
                    elif int(view_shape[0]) != n_layers:
                        raise RuntimeError(
                            f"Layer count mismatch for {component}: expected {n_layers}, got {view_shape[0]}"
                        )

                metadata[component] = {
                    "native_shape": list(view_shape),
                    "n_coordinates": int(np.prod(view_shape)),
                    "layerwise": layerwise,
                    **({"derived_from": source} if source else {}),
                }

                for scope, methods in self.SCOPES.items():
                    if component not in self._components(scope):
                        continue
                    idx = [all_idx[m] for m in methods]
                    for sig in self.SIGNATURES:
                        x = view[sig][:, idx]
                        name = f"{sig}/{component}"
                        mean_rdm, seed_rdms = _rdms_from_vectors(_flatten(x))
                        sources["overall"][scope][name] = mean_rdm
                        reliability["overall"].extend(
                            self._reliability_rows(scope, "functional", name, seed_rdms)
                        )

                        if not layerwise:
                            continue
                        layer_rdms = []
                        for li in range(n_layers):
                            layer_mean, layer_seed = _rdms_from_vectors(
                                _flatten(x[:, :, :, li])
                            )
                            layer_rdms.append(layer_mean)
                            reliability["layerwise"].extend(
                                self._reliability_rows(
                                    scope, "functional", name, layer_seed, layer=li + 1
                                )
                            )
                        sources["layerwise"][scope][name] = np.stack(layer_rdms, axis=1)

            del signatures
            gc.collect()

        if n_layers is None:
            raise RuntimeError("No layerwise functional components were found")
        return sources, reliability, metadata, n_layers

    # ---- prediction targets ---------------------------------------------------

    def _prediction_cache_path(self, task, method, seed):
        path = (
            self.save_dir
            / "prediction_cache"
            / "test"
            / task
            / method
            / f"seed-{seed}.npz"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _read_prediction_cache(path):
        try:
            with np.load(path, allow_pickle=False) as data:
                logits = np.asarray(data["logits"])
                labels = np.asarray(data["labels"])
            if logits.ndim != 2 or labels.ndim != 1 or len(logits) != len(labels):
                raise ValueError("invalid cached prediction shapes")
            return logits, labels
        except Exception:
            path.unlink(missing_ok=True)
            return None

    @staticmethod
    def _atomic_npz(path, **arrays):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as f:
            np.savez(f, **arrays)
        os.replace(tmp, path)

    def _seed_logits(self, task, method, run):
        cache = self._prediction_cache_path(task, method, run["seed"])
        if cache.is_file():
            cached = self._read_prediction_cache(cache)
            if cached is not None:
                return cached

        cfg = run["config"]
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        dataset = datasets[task](split="test")
        loader = dataset.get_dataloader(
            batch_size=int(cfg["gpu_batch_size"]), worker_init_fn=seed_worker
        )
        model = models[self.backbone_name][cfg["peft_name"]](
            img_size=datasets[task].IMG_SIZE,
            n_classes=datasets[task].N_CLASSES,
            freeze_backbone=bool(cfg["freeze_backbone"]),
            **cfg["peft_config"],
        )
        model.load_from_path(str(run["path"] / "model.pt"), map_location="cpu")
        model.to(self.device).eval()

        try:
            logits, labels = [], []
            with torch.inference_mode():
                iterator = self._progress(
                    loader, f"Predict test {task}/{method}/s{run['seed']}", len(loader)
                )
                for x, y in iterator:
                    logits.append(model(x.to(self.device)).cpu())
                    labels.append(y.cpu())
            logits = torch.cat(logits).numpy().astype(np.float32)
            labels = torch.cat(labels).numpy()
            self._atomic_npz(cache, logits=logits, labels=labels)
        finally:
            del model, dataset, loader
            _cleanup_cuda()

        cached = self._read_prediction_cache(cache)
        if cached is None:
            raise RuntimeError(f"Failed to create prediction cache: {cache}")
        return cached

    def _predictions(self, task, method):
        probs, labels = [], None
        for run in self._get_runs(task, method):
            logits, y = self._seed_logits(task, method, run)
            if labels is None:
                labels = y
            elif not np.array_equal(labels, y):
                raise RuntimeError(
                    f"Test ordering mismatch across seeds for {task}/{method}"
                )
            z = logits.astype(np.float64)
            z -= z.max(1, keepdims=True)
            p = np.exp(z)
            probs.append(p / p.sum(1, keepdims=True))
        probs = np.stack(probs)
        top1 = probs.argmax(-1)
        return {
            "probabilities": probs,
            "top1": top1,
            "correct": top1 == labels[None],
            "labels": labels,
        }

    def _jsd(self, p, q):
        p, q = np.clip(p, self.eps, 1.0), np.clip(q, self.eps, 1.0)
        p, q = p / p.sum(-1, keepdims=True), q / q.sum(-1, keepdims=True)
        m = 0.5 * (p + q)
        return float(
            (
                0.5 * np.sum(p * np.log2(p / m), -1)
                + 0.5 * np.sum(q * np.log2(q / m), -1)
            ).mean()
        )

    @staticmethod
    def _error_dissimilarity(a, b):
        observed = np.mean(a == b)
        pa, pb = a.mean(), b.mean()
        expected = pa * pb + (1 - pa) * (1 - pb)
        return (
            np.nan
            if 1 - expected <= np.finfo(float).eps
            else float(1 - (observed - expected) / (1 - expected))
        )

    def _prediction_distance(self, a, b, seed, metric):
        if metric == "jsd":
            return self._jsd(a["probabilities"][seed], b["probabilities"][seed])
        if metric == "top1_disagreement":
            return float(np.mean(a["top1"][seed] != b["top1"][seed]))
        if metric == "error_dissimilarity":
            return self._error_dissimilarity(a["correct"][seed], b["correct"][seed])
        raise KeyError(metric)

    def _prediction_rdms(self, pred, methods, metric):
        m = len(methods)
        rdms = np.zeros((len(self.SEEDS), m, m), float)
        for i, j in itertools.combinations(range(m), 2):
            a, b = pred[methods[i]], pred[methods[j]]
            for s in range(len(self.SEEDS)):
                v = self._prediction_distance(a, b, s, metric)
                rdms[s, i, j] = rdms[s, j, i] = v
        return rdms

    @staticmethod
    def _check_prediction_order(pred, methods, task):
        labels = pred[methods[0]]["labels"]
        if any(not np.array_equal(labels, pred[m]["labels"]) for m in methods[1:]):
            raise RuntimeError(f"Test ordering mismatch across methods for {task}")

    def prediction_targets(self):
        targets = {
            scope: {metric: [] for metric in self.BEHAVIOR_METRICS}
            for scope in self.SCOPES
        }
        reliability = []

        for task in self._progress(self.DATASETS, "Prediction tasks"):
            pred = {m: self._predictions(task, m) for m in self.ADAPTATION_METHODS}
            self._check_prediction_order(pred, self.ADAPTATION_METHODS, task)

            for scope, methods in self.SCOPES.items():
                for metric in self.BEHAVIOR_METRICS:
                    rdms = self._prediction_rdms(pred, methods, metric)
                    targets[scope][metric].append(rdms)
                    lo, hi = _noise_ceiling(rdms)
                    reliability.append(
                        {
                            "scope": scope,
                            "object_type": "prediction",
                            "distance_object": f"prediction/{metric}",
                            "layer": None,
                            "task": task,
                            "task_group": self._task_group[task],
                            "reliability_low": lo,
                            "reliability_high": hi,
                        }
                    )
            del pred
            gc.collect()

        for scope in self.SCOPES:
            for metric in self.BEHAVIOR_METRICS:
                targets[scope][metric] = np.stack(targets[scope][metric])
        return targets, reliability

    # ---- CKA sources ----------------------------------------------------------

    def _cka_cache_path(self, task):
        path = self.save_dir / "cka_cache" / f"{task}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _task_cka(self, task):
        cache = self._cka_cache_path(task)
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as data:
                methods = tuple(data["methods"].tolist())
                seeds = tuple(int(x) for x in data["seeds"])
                distances = data["layerwise_distance"]
            if methods == self.ADAPTATION_METHODS and seeds == self.SEEDS:
                return distances

        dataset = datasets[task](split="train+val")
        features, labels_ref = [], None
        for method in self.ADAPTATION_METHODS:
            for run in self._get_runs(task, method):
                cfg = run["config"]
                loader = dataset.get_dataloader(
                    batch_size=int(cfg["gpu_batch_size"]), num_workers=0
                )
                model = models[self.backbone_name][cfg["peft_name"]](
                    img_size=datasets[task].IMG_SIZE,
                    n_classes=datasets[task].N_CLASSES,
                    freeze_backbone=bool(cfg["freeze_backbone"]),
                    **cfg["peft_config"],
                )
                model.load_from_path(str(run["path"] / "model.pt"), map_location="cpu")
                model.to(self.device).eval()

                cls_batches, labels = [], []
                with torch.inference_mode():
                    iterator = self._progress(
                        loader, f"CKA {task}/{method}/s{run['seed']}", len(loader)
                    )
                    for x, y in iterator:
                        cls_batches.append(
                            model.get_layerwise_features(x.to(self.device))["cls"].cpu()
                        )
                        labels.append(y.cpu())
                cls, labels = (
                    torch.cat(cls_batches, dim=1).float(),
                    torch.cat(labels).numpy(),
                )
                if labels_ref is None:
                    labels_ref = labels
                elif not np.array_equal(labels_ref, labels):
                    raise RuntimeError(f"Train+val ordering mismatch for CKA on {task}")
                features.append(cls)
                del model, loader, cls_batches
                _cleanup_cuda()

        shape = features[0].shape
        if any(f.shape != shape for f in features[1:]):
            raise RuntimeError(f"Layerwise feature shape mismatch for {task}")

        n_ckpt, n_layers = len(features), shape[0]
        sim = np.empty((n_ckpt, n_ckpt, n_layers), np.float32)
        for layer in range(n_layers):
            kernels = []
            for feat in features:
                x = feat[layer].to(self.device)
                x = x - x.mean(0, keepdim=True)
                k = x @ x.T
                norm = torch.linalg.vector_norm(k)
                if norm <= self.eps:
                    raise RuntimeError(
                        f"Degenerate CKA representation for {task}, layer {layer}"
                    )
                kernels.append((k / norm).reshape(-1))
            kernels = torch.stack(kernels)
            sim[:, :, layer] = (kernels @ kernels.T).clamp(0, 1).cpu().numpy()

        nm, ns = len(self.ADAPTATION_METHODS), len(self.SEEDS)
        distances = (1 - sim).reshape(nm, ns, nm, ns, n_layers).transpose(0, 2, 1, 3, 4)
        np.savez_compressed(
            cache,
            methods=np.array(self.ADAPTATION_METHODS),
            seeds=np.array(self.SEEDS),
            layerwise_distance=distances,
        )
        del features, dataset
        _cleanup_cuda()
        return distances

    @staticmethod
    def _match_cka_layers(distances, expected_layers):
        n = distances.shape[-1]
        if n == expected_layers:
            return distances
        if n == expected_layers + 1:
            return distances[..., -expected_layers:]
        raise RuntimeError(
            f"CKA exposes {n} layers but functional signatures expose {expected_layers}; "
            "expected the same count or one extra input/embedding representation."
        )

    def cka_sources(self, n_layers):
        sources = self._empty_sources()
        reliability = {resolution: [] for resolution in self.RESOLUTIONS}
        all_idx = {m: i for i, m in enumerate(self.ADAPTATION_METHODS)}

        overall = {
            name: {scope: [] for scope in self.SCOPES} for name in self.CKA_OBJECTS
        }
        overall_seed = {
            name: {scope: [] for scope in self.SCOPES} for name in self.CKA_OBJECTS
        }
        layerwise = {scope: [] for scope in self.SCOPES}
        layer_seed = {scope: [[] for _ in range(n_layers)] for scope in self.SCOPES}

        for task in self._progress(self.DATASETS, "CKA tasks"):
            distances = self._match_cka_layers(self._task_cka(task), n_layers)
            for scope, methods in self.SCOPES.items():
                idx = [all_idx[m] for m in methods]
                sub = distances[
                    np.ix_(
                        idx,
                        idx,
                        range(len(self.SEEDS)),
                        range(len(self.SEEDS)),
                        range(n_layers),
                    )
                ]

                views = {
                    self.CKA_FINAL: sub[..., -1],
                    self.CKA_MEAN: sub.mean(axis=-1),
                }
                for name, view in views.items():
                    overall[name][scope].append(view.mean(axis=(2, 3)))
                    overall_seed[name][scope].append(
                        np.stack([view[:, :, s, s] for s in range(len(self.SEEDS))])
                    )

                layerwise[scope].append(sub.mean(axis=(2, 3)).transpose(2, 0, 1))
                for li in range(n_layers):
                    layer_seed[scope][li].append(
                        np.stack([sub[:, :, s, s, li] for s in range(len(self.SEEDS))])
                    )

        for scope in self.SCOPES:
            layer_stack = np.stack(layerwise[scope])
            for name in self.CKA_OBJECTS:
                sources["overall"][scope][name] = np.stack(overall[name][scope])
                sources["layerwise"][scope][name] = layer_stack
                reliability["overall"].extend(
                    self._reliability_rows(
                        scope, "cka", name, np.stack(overall_seed[name][scope])
                    )
                )
                for li in range(n_layers):
                    reliability["layerwise"].extend(
                        self._reliability_rows(
                            scope,
                            "cka",
                            name,
                            np.stack(layer_seed[scope][li]),
                            layer=li + 1,
                        )
                    )
        return sources, reliability

    # ---- RSA / summaries ------------------------------------------------------

    def _rsa(self, source, target_rdms, target_vectors, target_ranks):
        x = _upper(np.asarray(source, float))
        rho = _finite_mean([_spearman(x, row) for row in target_vectors])
        nn_match = _finite_mean([_nn_match(source, target) for target in target_rdms])
        triplet = _finite_mean(
            [_triplet_agreement(source, target) for target in target_rdms]
        )
        if not np.isfinite(rho):
            return rho, np.nan, np.empty(0), nn_match, triplet

        pi, pj = self._perm_pairs[source.shape[0]]
        xp = source[pi, pj]
        if np.all(np.isfinite(xp)) and np.all(np.isfinite(target_vectors)):
            rx = rankdata(xp, axis=1)
            null = np.mean(
                np.stack([_row_corr(rx, row) for row in target_ranks], axis=1), axis=1
            )
        else:
            null = np.asarray(
                [
                    _finite_mean([_spearman(row, target) for target in target_vectors])
                    for row in xp
                ]
            )
        null = null[np.isfinite(null)]
        p = (
            np.nan
            if not null.size
            else float(np.mean(np.abs(null) >= abs(rho) - 1e-15))
        )
        return rho, p, null, nn_match, triplet

    def _global_p(self, nulls, observed, rng):
        if not np.isfinite(observed):
            return np.nan
        nulls = [x for x in nulls if len(x)]
        if not nulls or self.n_global_permutations <= 0:
            return np.nan
        sim = np.zeros(self.n_global_permutations, float)
        for null in nulls:
            sim += null[rng.integers(0, len(null), self.n_global_permutations)]
        sim /= len(nulls)
        return float(
            (1 + np.sum(np.abs(sim) >= abs(observed)))
            / (self.n_global_permutations + 1)
        )

    def _target_cache(self, prediction):
        cache = {}
        for scope in self.SCOPES:
            for metric in self.BEHAVIOR_METRICS:
                for ti, replicates in enumerate(prediction[scope][metric]):
                    vectors = np.stack([_upper(rdm) for rdm in replicates])
                    cache[(scope, metric, ti)] = (
                        replicates,
                        vectors,
                        np.stack([_rank_vector(row) for row in vectors]),
                    )
        return cache

    def align(self, sources, prediction, prediction_reliability, resolution):
        ceiling = {
            (r["scope"], r["task"], r["distance_object"].split("/", 1)[1]): (
                r["reliability_low"],
                r["reliability_high"],
            )
            for r in prediction_reliability
        }
        target_cache = self._target_cache(prediction)
        n_layers = (
            next(iter(next(iter(sources.values())).values())).shape[1]
            if resolution == "layerwise"
            else 1
        )
        total = sum(
            len(objects) * len(self.BEHAVIOR_METRICS) * len(self.DATASETS) * n_layers
            for objects in sources.values()
        )

        records, undefined = [], 0
        bar = self._progress(range(total), f"RSA {resolution}", total)
        iterator = iter(bar)
        for scope, objects in sources.items():
            for obj, matrices in objects.items():
                for metric in self.BEHAVIOR_METRICS:
                    for ti, task in enumerate(self.DATASETS):
                        target_rdms, target_vectors, target_ranks = target_cache[
                            (scope, metric, ti)
                        ]
                        layer_items = (
                            enumerate(matrices[ti], 1)
                            if resolution == "layerwise"
                            else ((None, matrices[ti]),)
                        )
                        for layer, source in layer_items:
                            next(iterator)
                            rho, p, null, nn_match, triplet = self._rsa(
                                source, target_rdms, target_vectors, target_ranks
                            )
                            undefined += int(not np.isfinite(rho))
                            lo, hi = ceiling[(scope, task, metric)]
                            records.append(
                                {
                                    "resolution": resolution,
                                    "scope": scope,
                                    "distance_object": obj,
                                    "prediction_metric": metric,
                                    "layer": layer,
                                    "task": task,
                                    "task_group": self._task_group[task],
                                    "rho": rho,
                                    "nn_match": nn_match,
                                    "triplet_agreement": triplet,
                                    "noise_ceiling_low": lo,
                                    "noise_ceiling_high": hi,
                                    "p_permutation": p,
                                    "null": null,
                                }
                            )
        bar.close()
        if undefined and self.verbose:
            tqdm.write(f"  {resolution}: {undefined} undefined RSA cells kept as NA")
        return records

    def summarize(self, records, resolution):
        rng = np.random.default_rng(self.random_seed)
        grouped = defaultdict(list)
        for r in records:
            grouped[
                (
                    r["scope"],
                    r["distance_object"],
                    r["prediction_metric"],
                    r["layer"],
                )
            ].append(r)

        stats = []
        for key, rows in self._progress(
            grouped.items(), f"Summaries {resolution}", len(grouped)
        ):
            scope, obj, metric, layer = key
            by_task = {r["task"]: r for r in rows}
            selections = [
                ("avg", rows),
                *[
                    (g, [r for r in rows if r["task_group"] == g])
                    for g in self.TASK_GROUPS
                ],
                *[(task, [by_task[task]]) for task in self.DATASETS],
            ]
            for aggregation, use in selections:
                groups = np.asarray([r["task_group"] for r in use])
                values = {
                    score: np.asarray([r[score] for r in use], float)
                    for score in self.ALIGNMENT_SCORES
                }
                summary = {}
                for score, score_values in values.items():
                    summary[score] = _finite_mean(score_values)
                    if len(use) > 1:
                        lo, hi = _bootstrap_mean(
                            score_values,
                            groups,
                            self.n_bootstrap,
                            rng,
                            stratified=aggregation == "avg",
                        )
                    else:
                        lo = hi = np.nan
                    summary[f"{score}_ci_low"] = lo
                    summary[f"{score}_ci_high"] = hi

                p = (
                    self._global_p([r["null"] for r in use], summary["rho"], rng)
                    if len(use) > 1
                    else use[0]["p_permutation"]
                )
                stats.append(
                    {
                        "resolution": resolution,
                        "scope": scope,
                        "distance_object": obj,
                        "prediction_metric": metric,
                        "layer": layer,
                        "aggregation": aggregation,
                        **summary,
                        "noise_ceiling_low": _finite_mean(
                            [r["noise_ceiling_low"] for r in use]
                        ),
                        "noise_ceiling_high": _finite_mean(
                            [r["noise_ceiling_high"] for r in use]
                        ),
                        "p_permutation": p,
                        "q_bh": np.nan,
                        "n_tasks": len(use),
                        "delta_vs_cka": np.nan,
                        "delta_vs_cka_ci_low": np.nan,
                        "delta_vs_cka_ci_high": np.nan,
                    }
                )

        for scope in self.SCOPES:
            for metric in self.BEHAVIOR_METRICS:
                idx = [
                    i
                    for i, r in enumerate(stats)
                    if r["scope"] == scope
                    and r["prediction_metric"] == metric
                    and r["aggregation"] == "avg"
                ]
                for i, q in zip(idx, _bh([stats[i]["p_permutation"] for i in idx])):
                    stats[i]["q_bh"] = q

        task_rho = {
            (
                r["scope"],
                r["distance_object"],
                r["prediction_metric"],
                r["layer"],
                r["task"],
            ): r["rho"]
            for r in records
        }
        for row in stats:
            if row["aggregation"] not in {"avg", *self.TASK_GROUPS}:
                continue
            tasks = (
                self.DATASETS
                if row["aggregation"] == "avg"
                else self.TASK_GROUPS[row["aggregation"]]
            )
            delta = np.asarray(
                [
                    task_rho[
                        (
                            row["scope"],
                            row["distance_object"],
                            row["prediction_metric"],
                            row["layer"],
                            task,
                        )
                    ]
                    - task_rho[
                        (
                            row["scope"],
                            self.CKA_BASELINE,
                            row["prediction_metric"],
                            row["layer"],
                            task,
                        )
                    ]
                    for task in tasks
                ],
                float,
            )
            row["delta_vs_cka"] = _finite_mean(delta)
            if row["distance_object"] == self.CKA_BASELINE:
                row["delta_vs_cka_ci_low"] = row["delta_vs_cka_ci_high"] = 0.0
            else:
                groups = np.asarray([self._task_group[t] for t in tasks])
                row["delta_vs_cka_ci_low"], row["delta_vs_cka_ci_high"] = (
                    _bootstrap_mean(
                        delta,
                        groups,
                        self.n_bootstrap,
                        rng,
                        stratified=row["aggregation"] == "avg",
                    )
                )
        return stats

    # ---- output ---------------------------------------------------------------

    def _wide_rows(self, stats, scope, resolution):
        stats = [r for r in stats if r["scope"] == scope]
        lookup = {
            (
                r["distance_object"],
                r["prediction_metric"],
                r["layer"],
                r["aggregation"],
            ): r
            for r in stats
        }
        objects = {
            (r["distance_object"], r["prediction_metric"], r["layer"])
            for r in stats
            if r["aggregation"] == "avg"
        }
        aggs = ("avg", *self.TASK_GROUPS, *self.DATASETS)
        rows = []
        metric_order = {m: i for i, m in enumerate(self.BEHAVIOR_METRICS)}
        for obj, metric, layer in sorted(
            objects,
            key=lambda x: (x[0], metric_order[x[1]], -1 if x[2] is None else x[2]),
        ):
            row = {
                "scope": scope,
                "distance_object": obj,
                "prediction_metric": metric,
                **({"layer": layer} if resolution == "layerwise" else {}),
            }
            for agg in aggs:
                r = lookup[(obj, metric, layer, agg)]
                row[f"rho_{agg}"] = _format_rho(
                    r["rho"], r["noise_ceiling_low"], r["noise_ceiling_high"]
                )
                row[f"nn_match_{agg}"] = _format_score(r["nn_match"])
                row[f"triplet_agreement_{agg}"] = _format_score(r["triplet_agreement"])
            rows.append(row)
        return rows

    def _manifest(self, metadata, n_layers):
        data = {
            "backbone": self.backbone_name,
            "datasets": list(self.DATASETS),
            "task_groups": {k: list(v) for k, v in self.TASK_GROUPS.items()},
            "reference_method": self.REFERENCE_METHOD,
            "scopes": {k: list(v) for k, v in self.SCOPES.items()},
            "scope_directories": self.SCOPE_DIRS,
            "seeds": list(self.SEEDS),
            "signatures": list(self.SIGNATURES),
            "base_components": list(self.BASE_COMPONENTS),
            "derived_components": {
                k: {"source": v[0], "token_index": v[1]}
                for k, v in self.DERIVED_COMPONENTS.items()
            },
            "components": metadata,
            "peft_plus_ft_components": list(self.CORE_COMPONENTS),
            "behavior_metrics": list(self.BEHAVIOR_METRICS),
            "alignment_scores": {
                "rho": "mean Spearman RSA over the 3 same-seed prediction RDMs",
                "nn_match": "mean tie-aware expected exact nearest-neighbor match over the 3 same-seed prediction RDMs",
                "triplet_agreement": "mean strict anchored triplet-order agreement over the 3 same-seed prediction RDMs; tied orderings omitted",
            },
            "plot_metric": self.PLOT_METRIC,
            "n_layers": n_layers,
            "functional_object": "log(signature+eps) minus mean LP log profile over 3 LP checkpoints",
            "sensitivity": "S=F-P in LP-referenced log coordinates",
            "functional_distance": "cross-seed RMS Euclidean distance per coordinate over all 3x3 seed pairs",
            "layerwise_distance": "same functional distance after slicing one transformer layer before flattening",
            "cka": {
                "final": "final matched transformer-layer CLS linear CKA distance (1-CKA)",
                "mean": "mean matched transformer-layer CLS linear CKA distance; equivalent to 1 minus mean CKA",
                "layerwise": "matched transformer-layer CLS linear CKA distance, shared by the Final CKA and Mean CKA heatmap rows",
            },
            "delta_vs_cka": f"paired taskwise rho difference from {self.CKA_BASELINE}",
            "rsa": "fixed source RDM evaluated by mean Spearman correlation to 3 same-seed test prediction-RDM replicates; exact method-label permutations",
            "prediction_noise_ceiling": "same-seed test prediction RDM replicates; finite-rank pooled leave-one-seed-out lower and inclusive upper bounds",
            "n_bootstrap": self.n_bootstrap,
            "n_global_permutations": self.n_global_permutations,
            "random_seed": self.random_seed,
            "prediction_cache_dir": str(self.save_dir / "prediction_cache"),
            "cka_cache_dir": str(self.save_dir / "cka_cache"),
        }
        path = self.alignment_dir / "manifest.json"
        path.write_text(json.dumps(data, indent=2))
        return path

    def _write_resolution(
        self,
        scope,
        resolution,
        stats,
        source_rel,
        prediction_rel,
        n_layers,
        overall_stats=None,
    ):
        out = self.alignment_dir / self.SCOPE_DIRS[scope] / resolution
        out.mkdir(parents=True, exist_ok=True)
        metric_order = {m: i for i, m in enumerate(self.BEHAVIOR_METRICS)}
        aggregation_order = {
            a: i for i, a in enumerate(("avg", *self.TASK_GROUPS, *self.DATASETS))
        }
        scope_stats = sorted(
            (r for r in stats if r["scope"] == scope),
            key=lambda r: (
                r["distance_object"],
                metric_order[r["prediction_metric"]],
                -1 if r["layer"] is None else r["layer"],
                aggregation_order[r["aggregation"]],
            ),
        )
        rel = [r for r in source_rel if r["scope"] == scope] + [
            r for r in prediction_rel if r["scope"] == scope
        ]
        aggs = ("avg", *self.TASK_GROUPS, *self.DATASETS)
        fields = [
            "scope",
            "distance_object",
            "prediction_metric",
            *(["layer"] if resolution == "layerwise" else []),
            *[f"{score}_{agg}" for agg in aggs for score in self.ALIGNMENT_SCORES],
        ]
        files = {
            "alignment": _write_csv(
                out / "alignment.csv",
                self._wide_rows(stats, scope, resolution),
                fields,
            ),
            "stats": _write_csv(out / "stats.csv", scope_stats),
            "reliability": _write_csv(out / "reliability.csv", rel),
        }
        files["figures"] = make_alignment_figures(
            self,
            scope,
            resolution,
            scope_stats,
            n_layers,
            out / "figures",
            overall_stats=overall_stats,
        )
        return files

    def run(self):
        self.alignment_dir.mkdir(parents=True, exist_ok=True)

        self._phase(1, "Functional geometry")
        sources, source_rel, metadata, n_layers = self.functional_sources()

        self._phase(2, "Prediction targets")
        prediction, prediction_rel = self.prediction_targets()

        self._phase(3, "CKA geometry")
        cka_sources, cka_rel = self.cka_sources(n_layers)
        for resolution in self.RESOLUTIONS:
            for scope in self.SCOPES:
                sources[resolution][scope].update(cka_sources[resolution][scope])
            source_rel[resolution].extend(cka_rel[resolution])

        self._phase(4, "Behavior alignment")
        records = {
            resolution: self.align(
                sources[resolution], prediction, prediction_rel, resolution
            )
            for resolution in self.RESOLUTIONS
        }

        self._phase(5, "Statistical summaries")
        stats = {
            resolution: self.summarize(records[resolution], resolution)
            for resolution in self.RESOLUTIONS
        }
        del records
        gc.collect()

        self._phase(6, "Outputs and figures")
        outputs = {"manifest": self._manifest(metadata, n_layers)}
        for scope in self._progress(self.SCOPES, "Output scopes"):
            outputs[scope] = {}
            for resolution in self.RESOLUTIONS:
                outputs[scope][resolution] = self._write_resolution(
                    scope,
                    resolution,
                    stats[resolution],
                    source_rel[resolution],
                    prediction_rel,
                    n_layers,
                    overall_stats=(
                        stats["overall"] if resolution == "layerwise" else None
                    ),
                )
        return outputs


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------


LABELS = {
    "vtab_caltech101": "Caltech-101",
    "vtab_cifar100": "CIFAR-100",
    "vtab_dtd": "DTD",
    "vtab_flowers102": "Flowers-102",
    "vtab_pets": "Pets",
    "vtab_sun397": "SUN397",
    "vtab_svhn": "SVHN",
    "vtab_eurosat": "EuroSAT",
    "vtab_resisc45": "RESISC45",
    "vtab_patch_camelyon": "Patch Camelyon",
    "vtab_retinopathy": "Retinopathy",
    "vtab_clevr_count": "CLEVR count",
    "vtab_clevr_distance": "CLEVR distance",
    "vtab_dmlab": "DMLab",
    "vtab_kitti_distance": "KITTI distance",
    "vtab_dsprites_location": "dSprites location",
    "vtab_dsprites_orientation": "dSprites orientation",
    "vtab_smallnorb_azimuth": "SmallNORB azimuth",
    "vtab_smallnorb_elevation": "SmallNORB elevation",
    "residual_carry": "Residual carry",
    "residual_cls_carry": "Residual CLS carry",
    "residual_patch_carry": "Residual patch carry",
    "sublayer_update": "Sublayer update",
    "sublayer_cls_update": "Sublayer CLS update",
    "sublayer_patch_update": "Sublayer patch update",
    "attention_selection_route": "Attention selection route",
    "attention_transport_route": "Attention transport route",
    "sublayer_input_channel": "Sublayer input channel",
    "attention_head_selection_route": "Head selection route",
    "attention_head_transport_route": "Head transport route",
    "mlp_neuron": "MLP neuron",
    "final_cls_readout_channel": "Final CLS readout",
    "residual_token_carry": "Residual token carry",
    "sublayer_token_update": "Sublayer token update",
    "attention_head_output": "Attention head output",
    "F": r"$F$",
    "S": r"$S$",
    "P": r"$P$",
    "CKA final": "CKA final",
    "CKA mean": "CKA mean",
}
SCOPE_LABELS = {"peft_only": "PEFT only", "peft+ft": "PEFT + full FT"}
POINT_COLOR = "#4477AA"
SCORE_LABELS = {
    "rho": "Spearman ρ",
    "nn_match": "NN match",
    "triplet_agreement": "Triplet agreement",
}


def _pretty(value):
    if value in LABELS:
        return LABELS[value]
    if "/" in value:
        sig, comp = value.split("/", 1)
        return f"{LABELS.get(sig, sig)} · {LABELS.get(comp, comp)}"
    return str(value).replace("_", " ")


def _header(fig, left, scope, title):
    fig.text(
        left,
        0.955,
        SCOPE_LABELS[scope],
        ha="left",
        va="top",
        fontsize=10,
        color="#666666",
    )
    fig.text(
        left, 0.912, title, ha="left", va="top", fontsize=15.5, fontweight="semibold"
    )


def _score_style(ax, score, scope):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.8, zorder=0)
    ax.tick_params(axis="y", length=0)
    if score == "rho":
        ax.set_xlim(-1, 1)
        ax.set_xticks((-1, -0.5, 0, 0.5, 1))
        ax.axvline(0, color="#888888", linewidth=0.9, zorder=1)
    else:
        ax.set_xlim(0, 1)
        ax.set_xticks((0, 0.25, 0.5, 0.75, 1))
        chance = (
            1 / (len(FunctionalSignaturesValidator.SCOPES[scope]) - 1)
            if score == "nn_match"
            else 0.5
        )
        ax.axvline(
            chance, color="#888888", linewidth=0.9, linestyle=(0, (2, 2)), zorder=1
        )


def _forest_rows(
    ax,
    rows,
    labels,
    score,
    scope,
    *,
    ceiling_band=False,
    group_start=None,
    show_labels=True,
):
    y = np.arange(len(rows))
    if score == "rho" and ceiling_band:
        lo = _finite_mean([r["noise_ceiling_low"] for r in rows if r is not None])
        hi = _finite_mean([r["noise_ceiling_high"] for r in rows if r is not None])
        if np.isfinite(lo) and np.isfinite(hi):
            ax.axvspan(
                min(lo, hi),
                max(lo, hi),
                color="#D9D9D9",
                alpha=0.55,
                linewidth=0,
                zorder=0,
            )

    for yi, row in zip(y, rows):
        if row is None:
            continue
        value = row[score]
        if score == "rho" and not ceiling_band:
            lo, hi = row["noise_ceiling_low"], row["noise_ceiling_high"]
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot(
                    [min(lo, hi), max(lo, hi)],
                    [yi, yi],
                    color="#CFCFCF",
                    linewidth=5,
                    solid_capstyle="round",
                    zorder=1,
                )
        if not np.isfinite(value):
            continue
        ci_lo, ci_hi = row.get(f"{score}_ci_low", np.nan), row.get(
            f"{score}_ci_high", np.nan
        )
        if np.isfinite(ci_lo) and np.isfinite(ci_hi):
            ax.errorbar(
                value,
                yi,
                xerr=np.asarray([[max(0.0, value - ci_lo)], [max(0.0, ci_hi - value)]]),
                fmt="none",
                ecolor=POINT_COLOR,
                elinewidth=1.2,
                capsize=2.8,
                zorder=2,
            )
        ax.scatter(
            value,
            yi,
            s=32,
            marker="o",
            color=POINT_COLOR,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

    if group_start is not None:
        ax.axhline(group_start - 0.5, color="#A8A8A8", linewidth=0.9, zorder=0)
    ax.set_yticks(y)
    if show_labels:
        ax.set_yticklabels(labels, fontsize=9)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    _score_style(ax, score, scope)


def _cka_lines(ax, lookup, score, a):
    handles = []
    for name, color, linestyle in (
        (a.CKA_FINAL, "#222222", "-"),
        (a.CKA_MEAN, "#666666", "--"),
    ):
        row = lookup.get((name, "avg"))
        if row is not None and np.isfinite(row[score]):
            ax.axvline(
                row[score], color=color, linestyle=linestyle, linewidth=1.7, zorder=4
            )
            (handle,) = ax.plot(
                [], [], color=color, linestyle=linestyle, linewidth=1.7, label=name
            )
            handles.append(handle)
    return handles


def _overall_detail_page(a, scope, obj, lookup):
    groups = tuple(a.TASK_GROUPS)
    rows = (
        [lookup[(obj, task)] for task in a.DATASETS]
        + [None]
        + [lookup[(obj, g)] for g in groups]
    )
    labels = [_pretty(t) for t in a.DATASETS] + [""] + [g.title() for g in groups]

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 9.2), sharey=True)
    fig.subplots_adjust(left=0.19, right=0.985, top=0.83, bottom=0.12, wspace=0.18)
    for ai, (ax, score) in enumerate(zip(axes, a.ALIGNMENT_SCORES)):
        _forest_rows(
            ax,
            rows,
            labels,
            score,
            scope,
            group_start=len(a.DATASETS) + 1,
            show_labels=ai == 0,
        )
        for boundary in (
            len(a.NATURAL) - 0.5,
            len(a.NATURAL) + len(a.SPECIALIZED) - 0.5,
        ):
            ax.axhline(
                boundary,
                color="#BDBDBD",
                linewidth=0.8,
                linestyle=(0, (2, 2)),
                zorder=0,
            )
        ax.set_xlabel(SCORE_LABELS[score], fontsize=10)
        ax.set_title(f"{SCORE_LABELS[score]} vs JSD", fontsize=10.5)
    _header(fig, 0.19, scope, _pretty(obj))
    fig.text(
        0.19,
        0.038,
        "ρ: thick interval = prediction-RDM noise ceiling   •   Group whiskers = 95% task-bootstrap CI   •   NN/triplet dotted line = chance",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#666666",
    )
    return fig


def _overall_summary_page(a, scope, signature, lookup, components):
    objects = [
        f"{signature}/{c}" for c in components if (f"{signature}/{c}", "avg") in lookup
    ]
    if not objects:
        return None
    rows = [lookup[(obj, "avg")] for obj in objects]
    labels = [_pretty(obj.split("/", 1)[1]) for obj in objects]

    fig, axes = plt.subplots(
        1, 3, figsize=(17.2, max(6.0, 0.48 * len(rows) + 2.4)), sharey=True
    )
    fig.subplots_adjust(left=0.22, right=0.985, top=0.83, bottom=0.12, wspace=0.18)
    legend = []
    for ai, (ax, score) in enumerate(zip(axes, a.ALIGNMENT_SCORES)):
        _forest_rows(
            ax, rows, labels, score, scope, ceiling_band=True, show_labels=ai == 0
        )
        handles = _cka_lines(ax, lookup, score, a)
        if ai == 2:
            legend = handles
        ax.set_xlabel(f"Mean task-level {SCORE_LABELS[score]}", fontsize=10)
        ax.set_title(f"{SCORE_LABELS[score]} vs JSD", fontsize=10.5)
    if legend:
        axes[-1].legend(handles=legend, loc="lower right", frameon=False, fontsize=8.5)
    _header(
        fig,
        0.22,
        scope,
        f"{_pretty(signature)} functional signatures vs predictive behavior",
    )
    fig.text(
        0.22,
        0.038,
        "ρ band = mean prediction-RDM noise ceiling   •   Whiskers = 95% task-bootstrap CI   •   Lines = CKA baselines   •   Dotted NN/triplet line = chance",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#666666",
    )
    return fig


def _heatmap_values(a, scope, obj, lookup, n_layers, score):
    groups = tuple(a.TASK_GROUPS)
    task_rows = np.asarray(
        [
            [lookup[(obj, li, task)][score] for li in range(1, n_layers + 1)]
            for task in a.DATASETS
        ]
    )
    group_rows = np.asarray(
        [
            [lookup[(obj, li, group)][score] for li in range(1, n_layers + 1)]
            for group in groups
        ]
    )
    return np.vstack([task_rows, np.full((1, n_layers), np.nan), group_rows])


def _layer_detail_page(a, scope, obj, lookup, n_layers):
    labels = (
        [_pretty(t) for t in a.DATASETS] + [""] + [g.title() for g in a.TASK_GROUPS]
    )
    fig, axes = plt.subplots(1, 3, figsize=(18.2, 9.3), sharey=True)
    fig.subplots_adjust(left=0.18, right=0.975, top=0.83, bottom=0.12, wspace=0.27)
    for ai, (ax, score) in enumerate(zip(axes, a.ALIGNMENT_SCORES)):
        matrix = _heatmap_values(a, scope, obj, lookup, n_layers, score)
        vmin, vmax = (-1, 1) if score == "rho" else (0, 1)
        im = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(n_layers), np.arange(1, n_layers + 1))
        ax.set_yticks(np.arange(len(labels)))
        if ai == 0:
            ax.set_yticklabels(labels, fontsize=8.5)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlabel("Transformer block")
        ax.set_title(f"{SCORE_LABELS[score]} vs JSD", fontsize=10.5)
        for boundary in (
            len(a.NATURAL) - 0.5,
            len(a.NATURAL) + len(a.SPECIALIZED) - 0.5,
            len(a.DATASETS) - 0.5,
        ):
            ax.axhline(boundary, color="white", linewidth=0.9, alpha=0.8)
        fig.colorbar(im, ax=ax, label=SCORE_LABELS[score], fraction=0.045, pad=0.03)
    _header(fig, 0.18, scope, f"{_pretty(obj)} - layerwise alignment")
    fig.text(
        0.18,
        0.038,
        "Each cell uses only that transformer-layer slice   •   Bottom rows = VTAB-group means",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#666666",
    )
    return fig


def _layer_summary_page(
    a, scope, signature, overall_lookup, layer_lookup, components, n_layers
):
    objects = [
        f"{signature}/{c}"
        for c in components
        if (f"{signature}/{c}", 1, "avg") in layer_lookup
    ]
    cka_start = len(objects)
    objects += [name for name in a.CKA_OBJECTS if (name, 1, "avg") in layer_lookup]
    if not objects:
        return None
    labels = [
        _pretty(obj) if obj in a.CKA_OBJECTS else _pretty(obj.split("/", 1)[1])
        for obj in objects
    ]

    fig, axes = plt.subplots(
        1, 3, figsize=(18.6, max(6.4, 0.46 * len(objects) + 2.5)), sharey=True
    )
    fig.subplots_adjust(left=0.22, right=0.975, top=0.83, bottom=0.14, wspace=0.27)
    for ai, (ax, score) in enumerate(zip(axes, a.ALIGNMENT_SCORES)):
        matrix = []
        for obj in objects:
            matrix.append(
                [layer_lookup[(obj, li, "avg")][score] for li in range(1, n_layers + 1)]
                + [overall_lookup[(obj, "avg")][score]]
            )
        vmin, vmax = (-1, 1) if score == "rho" else (0, 1)
        im = ax.imshow(np.asarray(matrix, float), aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(
            np.arange(n_layers + 1), [*map(str, range(1, n_layers + 1)), "All"]
        )
        ax.set_yticks(np.arange(len(labels)))
        if ai == 0:
            ax.set_yticklabels(labels, fontsize=8.7)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlabel("Transformer block")
        ax.set_title(f"{SCORE_LABELS[score]} vs JSD", fontsize=10.5)
        ax.axvline(n_layers - 0.5, color="white", linewidth=1.1, alpha=0.9)
        if cka_start:
            ax.axhline(cka_start - 0.5, color="white", linewidth=0.9, alpha=0.8)
        fig.colorbar(im, ax=ax, label=SCORE_LABELS[score], fraction=0.045, pad=0.03)
    _header(fig, 0.22, scope, f"{_pretty(signature)} layerwise behavioral alignment")
    fig.text(
        0.22,
        0.038,
        "Layers use layer-sliced alignment   •   All = whole-component alignment   •   CKA rows share layerwise values; All uses final vs mean CKA",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#666666",
    )
    return fig


def _save_pdf(path, items, page_fn, desc, verbose):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for item in tqdm(
            items, desc=desc, disable=not verbose, leave=False, dynamic_ncols=True
        ):
            fig = page_fn(item)
            if fig is not None:
                pdf.savefig(fig)
                plt.close(fig)
    return path


def make_alignment_figures(
    a, scope, resolution, stats, n_layers, out, overall_stats=None
):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stats = [r for r in stats if r["prediction_metric"] == a.PLOT_METRIC]
    components = a._components(scope)
    paths = {}

    if resolution == "overall":
        lookup = {(r["distance_object"], r["aggregation"]): r for r in stats}
        available = {r["distance_object"] for r in stats if r["aggregation"] == "avg"}
        objects = [
            f"{sig}/{c}"
            for sig in a.SIGNATURES
            for c in components
            if f"{sig}/{c}" in available
        ] + [name for name in a.CKA_OBJECTS if name in available]
        paths["detail"] = _save_pdf(
            out / "detail.pdf",
            objects,
            lambda obj: _overall_detail_page(a, scope, obj, lookup),
            f"Figures {scope}/overall detail",
            a.verbose,
        )
        paths["summary"] = _save_pdf(
            out / "summary.pdf",
            a.SIGNATURES,
            lambda sig: _overall_summary_page(a, scope, sig, lookup, components),
            f"Figures {scope}/overall summary",
            a.verbose,
        )
        return paths

    layer_lookup = {
        (r["distance_object"], r["layer"], r["aggregation"]): r for r in stats
    }
    available = {r["distance_object"] for r in stats if r["aggregation"] == "avg"}
    objects = [
        f"{sig}/{c}"
        for sig in a.SIGNATURES
        for c in components
        if f"{sig}/{c}" in available
    ] + [name for name in a.CKA_OBJECTS if name in available]

    if overall_stats is None:
        raise ValueError("overall_stats is required for layerwise summary figures")
    overall_lookup = {
        (r["distance_object"], "avg"): r
        for r in overall_stats
        if r["scope"] == scope
        and r["prediction_metric"] == a.PLOT_METRIC
        and r["aggregation"] == "avg"
    }
    paths["detail"] = _save_pdf(
        out / "detail.pdf",
        objects,
        lambda obj: _layer_detail_page(a, scope, obj, layer_lookup, n_layers),
        f"Figures {scope}/layerwise detail",
        a.verbose,
    )
    paths["summary"] = _save_pdf(
        out / "summary.pdf",
        a.SIGNATURES,
        lambda sig: _layer_summary_page(
            a, scope, sig, overall_lookup, layer_lookup, components, n_layers
        ),
        f"Figures {scope}/layerwise summary",
        a.verbose,
    )
    return paths
