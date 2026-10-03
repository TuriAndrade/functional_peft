import csv
import gc
import itertools
import json
import os
import random
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


def _upper(x):
    x = np.asarray(x, float)
    return x[np.triu_indices(x.shape[-1], 1)]


def _mean(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else np.nan


def _spearman(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 2 or not np.ptp(a) or not np.ptp(b):
        return np.nan
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def _bh(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    if not idx.size:
        return out
    order = idx[np.argsort(p[idx])]
    q = p[order] * len(order) / np.arange(1, len(order) + 1)
    out[order] = np.minimum(np.minimum.accumulate(q[::-1])[::-1], 1)
    return out


def _bootstrap(values, groups, n, rng):
    values, groups = np.asarray(values, float), np.asarray(groups)
    keep = np.isfinite(values)
    values, groups = values[keep], groups[keep]
    if not len(values) or n <= 0:
        return np.nan, np.nan
    draws = [
        pool[rng.integers(0, len(pool), (n, len(pool)))]
        for group in np.unique(groups)
        for pool in [values[groups == group]]
    ]
    samples = np.concatenate(draws, axis=1).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def _write_csv(path, rows):
    if not rows:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        "NA"
                        if isinstance(value, (float, np.floating))
                        and not np.isfinite(value)
                        else value
                    )
                    for key, value in row.items()
                }
            )
    return path


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class FunctionalSignatureValidator:
    """Validate functional-signature geometry against predictive behavior and CLS baselines."""

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
    GROUPS = {"natural": NATURAL, "specialized": SPECIALIZED, "structured": STRUCTURED}

    LP = "linear_probe"
    METHODS = ("vpt_shallow", "vpt_deep", "lora", "ssf", "adaptformer")
    SEEDS = (0, 1, 2)
    COMPONENTS = ("attention_qk", "attention_ov", "mlp_in", "mlp_out")
    SIGNATURES = ("P", "F", "S")
    COORDINATES = ("absolute", "lp_relative")
    BASELINES = ("cka", "procrustes")

    EXPERIMENT = "top_1_acc_eval"
    SIGNATURE_FILE = "functional_signature.npz"
    PREDICTION_SPLIT = "test"
    BASELINE_SPLIT = "train+val"

    EPS = 1e-12
    N_BOOTSTRAP = 2000
    N_GLOBAL_PERMUTATIONS = 10000
    RANDOM_SEED = 0

    def __init__(self, backbone_name="vit_b16", verbose=True):
        self.backbone_name = str(backbone_name)
        self.verbose = bool(verbose)
        self.experiments_dir = Path(os.environ.get("EXPERIMENTS_DIR", "./experiments"))

        # Keep this root unchanged so existing prediction / CKA / Procrustes caches are reused.
        self.save_dir = (
            self.experiments_dir / "functional_reorganization" / self.backbone_name
        )
        self.out_dir = self.save_dir / "functional_signature_validation"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._runs = {}
        self._perm_cache = {}
        self._group = {
            dataset: group
            for group, values in self.GROUPS.items()
            for dataset in values
        }

    def _phase(self, text):
        if self.verbose:
            tqdm.write(text)

    def _progress(self, it, desc, total=None):
        return tqdm(
            it,
            desc=desc,
            total=total,
            disable=not self.verbose,
            leave=False,
            dynamic_ncols=True,
        )

    # ------------------------------------------------------------------
    # Runs / models / unchanged caches
    # ------------------------------------------------------------------

    def _run_dir(self, dataset, method):
        return (
            self.experiments_dir
            / self.EXPERIMENT
            / self.backbone_name
            / dataset
            / method
        )

    def _get_runs(self, dataset, method):
        key = dataset, method
        if key in self._runs:
            return self._runs[key]

        root = self._run_dir(dataset, method)
        if not root.is_dir():
            raise FileNotFoundError(root)

        by_seed = {}
        for path in root.iterdir():
            config, model, acc = (
                path / "config.json",
                path / "model.pt",
                path / "test_acc_eval.json",
            )
            if path.is_dir() and config.is_file() and model.is_file() and acc.is_file():
                with config.open(encoding="utf-8") as f:
                    cfg = json.load(f)
                seed = int(cfg["run_seed"])
                if seed in by_seed:
                    raise RuntimeError(f"Duplicate seed {seed} for {dataset}/{method}")
                by_seed[seed] = {"seed": seed, "path": path, "config": cfg}

        missing = [seed for seed in self.SEEDS if seed not in by_seed]
        if missing:
            raise RuntimeError(
                f"Missing seeds {missing} for {dataset}/{method}; found {sorted(by_seed)}"
            )

        self._runs[key] = [by_seed[seed] for seed in self.SEEDS]
        return self._runs[key]

    @staticmethod
    def _atomic_npz(path, **arrays):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)

    def _load_model(self, dataset, run):
        cfg = run["config"]
        model = models[self.backbone_name][cfg["peft_name"]](
            img_size=datasets[dataset].IMG_SIZE,
            n_classes=datasets[dataset].N_CLASSES,
            freeze_backbone=bool(cfg["freeze_backbone"]),
            **cfg["peft_config"],
        )
        model.load_from_path(str(run["path"] / "model.pt"), map_location="cpu")
        return model.to(self.device).eval()

    def _prediction_path(self, dataset, method, seed):
        return (
            self.save_dir
            / "prediction_cache"
            / self.PREDICTION_SPLIT
            / dataset
            / method
            / f"seed-{seed}.npz"
        )

    @staticmethod
    def _read_prediction(path):
        try:
            with np.load(path, allow_pickle=False) as data:
                logits, labels = np.asarray(data["logits"]), np.asarray(data["labels"])
            if logits.ndim != 2 or labels.ndim != 1 or len(logits) != len(labels):
                raise ValueError
            return logits, labels
        except Exception:
            Path(path).unlink(missing_ok=True)
            return None

    def _ensure_prediction(self, dataset, method, run):
        path = self._prediction_path(dataset, method, run["seed"])
        if path.is_file() and self._read_prediction(path) is not None:
            return

        random.seed(self.RANDOM_SEED)
        np.random.seed(self.RANDOM_SEED)
        torch.manual_seed(self.RANDOM_SEED)
        model = self._load_model(dataset, run)
        try:
            ds = datasets[dataset](split=self.PREDICTION_SPLIT)
            loader = ds.get_dataloader(
                batch_size=int(run["config"]["gpu_batch_size"]),
                worker_init_fn=seed_worker,
            )
            logits, labels = [], []
            with torch.inference_mode():
                for x, y in self._progress(
                    loader,
                    f"Predict {self.PREDICTION_SPLIT} {dataset}/{method}/s{run['seed']}",
                    len(loader),
                ):
                    logits.append(model(x.to(self.device)).cpu())
                    labels.append(y.cpu())
            self._atomic_npz(
                path,
                logits=torch.cat(logits).numpy().astype(np.float32),
                labels=torch.cat(labels).numpy(),
            )
        finally:
            del model
            _cleanup()

    def _predictions(self, dataset, method):
        logits, labels = [], None
        for run in self._get_runs(dataset, method):
            cached = self._read_prediction(
                self._prediction_path(dataset, method, run["seed"])
            )
            if cached is None:
                raise RuntimeError(
                    f"Missing prediction cache: {dataset}/{method}/{self.PREDICTION_SPLIT}"
                )
            z, y = cached
            if labels is None:
                labels = y
            elif not np.array_equal(labels, y):
                raise RuntimeError(
                    f"Prediction ordering mismatch across seeds: {dataset}/{method}"
                )
            logits.append(z.astype(np.float64))

        logits = np.stack(logits)
        centered = logits - logits.mean(-1, keepdims=True)
        stable = logits - logits.max(-1, keepdims=True)
        probs = np.exp(stable)
        probs /= probs.sum(-1, keepdims=True)
        return {"centered_logits": centered, "p": probs, "labels": labels}

    def _jsd(self, p, q):
        p, q = np.clip(p, self.EPS, 1), np.clip(q, self.EPS, 1)
        m = 0.5 * (p + q)
        return float(
            (
                0.5 * np.sum(p * np.log2(p / m), axis=-1)
                + 0.5 * np.sum(q * np.log2(q / m), axis=-1)
            ).mean()
        )

    # ------------------------------------------------------------------
    # Functional signatures
    # ------------------------------------------------------------------

    def _read_component(self, run, component):
        path = run["path"] / self.SIGNATURE_FILE
        with np.load(path, allow_pickle=False) as data:
            out = {}
            for signature in self.SIGNATURES:
                key = f"{signature}__{component}"
                if key not in data:
                    raise KeyError(
                        f"Missing '{key}' in {path}; available={list(data.files)}"
                    )
                value = np.asarray(data[key], np.float64)
                if not np.all(np.isfinite(value)) or np.any(value < -self.EPS):
                    raise ValueError(f"Invalid {signature}/{component} in {path}")
                out[signature] = np.sqrt(np.maximum(value, 0))

        shapes = {value.shape for value in out.values()}
        if len(shapes) != 1 or not next(iter(shapes)):
            raise ValueError(f"Incompatible F/P/S shapes for {component} in {path}")
        return out

    @staticmethod
    def _vector_rdm(x, metric):
        x = np.asarray(x, np.float64)
        methods, seeds = x.shape[:2]
        flat = x.reshape(methods * seeds, -1)
        gram = flat @ flat.T
        norm2 = np.einsum("ij,ij->i", flat, flat)

        if metric == "euclidean":
            dist = np.sqrt(np.maximum(norm2[:, None] + norm2[None, :] - 2 * gram, 0))
        elif metric == "angular":
            norm = np.sqrt(norm2)
            den = norm[:, None] * norm[None, :]
            cosine = np.divide(gram, den, out=np.zeros_like(gram), where=den > 0)
            dist = np.arccos(np.clip(cosine, -1, 1)) / np.pi
            zero = norm == 0
            dist[np.logical_and(zero[:, None], zero[None, :])] = 0
            dist[np.logical_xor(zero[:, None], zero[None, :])] = 0.5
        else:
            raise ValueError(metric)

        central = dist.reshape(methods, seeds, methods, seeds).mean(axis=(1, 3))
        np.fill_diagonal(central, 0)
        return central

    def functional_geometry(self):
        full = {
            (coord, sig, comp): []
            for coord in self.COORDINATES
            for sig in self.SIGNATURES
            for comp in self.COMPONENTS
        }
        layers = {
            (coord, sig, comp): []
            for coord in self.COORDINATES
            for sig in self.SIGNATURES
            for comp in self.COMPONENTS
        }

        for dataset in self._progress(self.DATASETS, "Functional geometry"):
            for component in self.COMPONENTS:
                by_method = {}
                for method in (self.LP, *self.METHODS):
                    values = {signature: [] for signature in self.SIGNATURES}
                    for run in self._get_runs(dataset, method):
                        rep = self._read_component(run, component)
                        for signature in self.SIGNATURES:
                            values[signature].append(rep[signature])
                    by_method[method] = {
                        key: np.stack(value) for key, value in values.items()
                    }

                for signature in self.SIGNATURES:
                    x = np.stack(
                        [by_method[method][signature] for method in self.METHODS]
                    )
                    ref = by_method[self.LP][signature].mean(axis=0)
                    relative = x - ref[None, None]

                    full[("absolute", signature, component)].append(
                        self._vector_rdm(x, "euclidean")
                    )
                    full[("lp_relative", signature, component)].append(
                        self._vector_rdm(relative, "angular")
                    )

                    n_layers = x.shape[2]
                    layers[("absolute", signature, component)].append(
                        np.stack(
                            [
                                self._vector_rdm(x[:, :, layer], "euclidean")
                                for layer in range(n_layers)
                            ]
                        )
                    )
                    layers[("lp_relative", signature, component)].append(
                        np.stack(
                            [
                                self._vector_rdm(relative[:, :, layer], "angular")
                                for layer in range(n_layers)
                            ]
                        )
                    )

                del by_method
                _cleanup()

        return (
            {key: np.stack(value) for key, value in full.items()},
            {key: np.stack(value) for key, value in layers.items()},
        )

    # ------------------------------------------------------------------
    # Prediction geometry
    # ------------------------------------------------------------------

    def behavior_geometry(self):
        absolute, relative = [], []
        methods = (self.LP, *self.METHODS)

        for dataset in self._progress(self.DATASETS, "Prediction geometry"):
            for method in methods:
                for run in self._get_runs(dataset, method):
                    self._ensure_prediction(dataset, method, run)

            pred = {method: self._predictions(dataset, method) for method in methods}
            labels = pred[self.LP]["labels"]
            if any(
                not np.array_equal(labels, pred[method]["labels"])
                for method in self.METHODS
            ):
                raise RuntimeError(
                    f"Prediction ordering mismatch across methods: {dataset}"
                )

            m, s = len(self.METHODS), len(self.SEEDS)
            raw = np.zeros((m, m), float)
            for i, j in itertools.combinations(range(m), 2):
                a, b = pred[self.METHODS[i]]["p"], pred[self.METHODS[j]]["p"]
                raw[i, j] = raw[j, i] = _mean(
                    [self._jsd(a[sa], b[sb]) for sa in range(s) for sb in range(s)]
                )
            absolute.append(raw)

            lp_ref = pred[self.LP]["centered_logits"].mean(axis=0)
            delta = np.stack(
                [
                    pred[method]["centered_logits"] - lp_ref[None]
                    for method in self.METHODS
                ]
            )
            relative.append(self._vector_rdm(delta, "angular"))
            del pred, delta
            _cleanup()

        return {"absolute": np.stack(absolute), "lp_relative": np.stack(relative)}

    # ------------------------------------------------------------------
    # Existing final-CLS CKA / Procrustes caches
    # ------------------------------------------------------------------

    def _baseline_path(self, name, dataset, coordinate):
        suffix = "" if coordinate == "absolute" else "_delta"
        return self.save_dir / f"{name}{suffix}_cache" / f"{dataset}.npz"

    def _read_baseline(self, name, dataset, coordinate):
        path = self._baseline_path(name, dataset, coordinate)
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                cached_methods = tuple(data["methods"].tolist())
                seeds = tuple(int(x) for x in data["seeds"])
                distance = np.asarray(data["distance"], float)

            if seeds != self.SEEDS or any(
                method not in cached_methods for method in self.METHODS
            ):
                raise ValueError
            expected = (
                len(cached_methods),
                len(cached_methods),
                len(self.SEEDS),
                len(self.SEEDS),
            )
            if distance.shape != expected:
                raise ValueError

            idx = [cached_methods.index(method) for method in self.METHODS]
            distance = np.take(np.take(distance, idx, axis=0), idx, axis=1)
            return distance
        except Exception:
            return None

    def _cls_compare(self, x, y, need):
        x, y = x.to(self.device), y.to(self.device)
        x, y = x - x.mean(0, keepdim=True), y - y.mean(0, keepdim=True)
        out = {}
        if "cka" in need:
            xy, xx, yy = x.T @ y, x.T @ x, y.T @ y
            sim = (
                xy.square().sum() / torch.sqrt(xx.square().sum() * yy.square().sum())
            ).clamp(0, 1)
            out["cka"] = float((1 - sim).cpu())
        if "procrustes" in need:
            xn = x / torch.linalg.vector_norm(x).clamp_min(self.EPS)
            yn = y / torch.linalg.vector_norm(y).clamp_min(self.EPS)
            sim = torch.linalg.svdvals(xn.T @ yn).sum().clamp(0, 1)
            out["procrustes"] = float(torch.sqrt(torch.clamp(2 - 2 * sim, min=0)).cpu())
        return out

    def _extract_cls(self, dataset, method, ds):
        values, labels_ref = [], None
        for run in self._get_runs(dataset, method):
            model = self._load_model(dataset, run)
            random.seed(self.RANDOM_SEED)
            np.random.seed(self.RANDOM_SEED)
            torch.manual_seed(self.RANDOM_SEED)
            loader = ds.get_dataloader(
                batch_size=int(run["config"]["gpu_batch_size"]),
                num_workers=0,
            )
            xs, ys = [], []
            with torch.inference_mode():
                for x, y in self._progress(
                    loader, f"CLS {dataset}/{method}/s{run['seed']}", len(loader)
                ):
                    xs.append(
                        model.forward_features(x.to(self.device))["cls"].detach().cpu()
                    )
                    ys.append(y.cpu())
            features, labels = torch.cat(xs).float(), torch.cat(ys).numpy()
            if features.ndim != 2:
                raise ValueError(
                    f"forward_features()['cls'] must be [N,D], got {tuple(features.shape)}"
                )
            if labels_ref is None:
                labels_ref = labels
            elif not np.array_equal(labels_ref, labels):
                raise RuntimeError(
                    f"{self.BASELINE_SPLIT} ordering mismatch: {dataset}/{method}"
                )
            values.append(features)
            del model, loader, xs, ys
            _cleanup()
        return torch.stack(values), labels_ref

    def _compute_baseline_distances(self, features, need):
        m, s = len(self.METHODS), len(self.SEEDS)
        shape = (m, m, s, s)
        out = {name: np.zeros(shape, np.float32) for name in need}
        items = [(mi, si, features[mi, si]) for mi in range(m) for si in range(s)]
        for a, b in itertools.combinations(range(len(items)), 2):
            mi, si, x = items[a]
            mj, sj, y = items[b]
            for name, value in self._cls_compare(x, y, need).items():
                out[name][mi, mj, si, sj] = value
                out[name][mj, mi, sj, si] = value
        return out

    def _task_baselines(self, dataset):
        cached = {
            (coordinate, name): self._read_baseline(name, dataset, coordinate)
            for coordinate in self.COORDINATES
            for name in self.BASELINES
        }
        need = {
            coordinate: {
                name for name in self.BASELINES if cached[(coordinate, name)] is None
            }
            for coordinate in self.COORDINATES
        }
        if not any(need.values()):
            return cached

        features, labels = {}, None
        ds = datasets[dataset](split=self.BASELINE_SPLIT)
        methods = (self.LP, *self.METHODS) if need["lp_relative"] else self.METHODS
        for method in methods:
            feat, y = self._extract_cls(dataset, method, ds)
            if labels is None:
                labels = y
            elif not np.array_equal(labels, y):
                raise RuntimeError(
                    f"{self.BASELINE_SPLIT} ordering mismatch across methods: {dataset}"
                )
            features[method] = feat

        absolute = torch.stack([features[method] for method in self.METHODS])
        relative = None
        if need["lp_relative"]:
            relative = absolute - features[self.LP].mean(0)[None, None]

        for coordinate, values in (("absolute", absolute), ("lp_relative", relative)):
            if not need[coordinate]:
                continue
            computed = self._compute_baseline_distances(values, need[coordinate])
            for name, distance in computed.items():
                self._atomic_npz(
                    self._baseline_path(name, dataset, coordinate),
                    methods=np.array(self.METHODS),
                    seeds=np.array(self.SEEDS),
                    distance=distance,
                )
                cached[(coordinate, name)] = distance

        del features, absolute, relative, ds
        _cleanup()
        return cached

    def baseline_geometry(self):
        out = {
            (coordinate, name): []
            for coordinate in self.COORDINATES
            for name in self.BASELINES
        }
        for dataset in self._progress(self.DATASETS, "Final-CLS baselines"):
            for key, distance in self._task_baselines(dataset).items():
                out[key].append(distance.mean(axis=(2, 3)))
        return {key: np.stack(value) for key, value in out.items()}

    # ------------------------------------------------------------------
    # RSA / aggregation
    # ------------------------------------------------------------------

    def _perm_edges(self):
        if "full" not in self._perm_cache:
            perms = np.asarray(list(itertools.permutations(range(len(self.METHODS)))))
            i, j = np.triu_indices(len(self.METHODS), 1)
            self._perm_cache["full"] = perms[:, i], perms[:, j]
        return self._perm_cache["full"]

    def _rsa(self, source, target):
        x, y = _upper(source), _upper(target)
        rho = _spearman(x, y)
        if not np.isfinite(rho):
            return rho, np.nan, np.empty(0)

        pi, pj = self._perm_edges()
        xp, ry = source[pi, pj], rankdata(y)
        rx = rankdata(xp, axis=1)
        rx -= rx.mean(1, keepdims=True)
        ry -= ry.mean()
        den = np.linalg.norm(rx, axis=1) * np.linalg.norm(ry)
        null = np.divide(rx @ ry, den, out=np.full(len(rx), np.nan), where=den > 0)
        null = null[np.isfinite(null)]
        p = float(np.mean(np.abs(null) >= abs(rho) - 1e-15)) if len(null) else np.nan
        return rho, p, null

    def _anchor_rsa(self, source, target, anchor):
        mask = np.arange(len(self.METHODS)) != anchor
        x, y = source[anchor, mask], target[anchor, mask]
        rho = _spearman(x, y)
        if not np.isfinite(rho):
            return rho, np.nan
        perms = self._perm_cache.setdefault(
            "anchor",
            np.asarray(list(itertools.permutations(range(len(self.METHODS) - 1)))),
        )
        null = np.asarray([_spearman(x, y[p]) for p in perms])
        null = null[np.isfinite(null)]
        p = float(np.mean(np.abs(null) >= abs(rho) - 1e-15)) if len(null) else np.nan
        return rho, p

    def _global_p(self, rows, observed, rng):
        nulls = [
            row["null"]
            for row in rows
            if len(row.get("null", ())) and np.isfinite(row["rho"])
        ]
        if not nulls or not np.isfinite(observed) or self.N_GLOBAL_PERMUTATIONS <= 0:
            return np.nan
        sim = sum(
            null[rng.integers(0, len(null), self.N_GLOBAL_PERMUTATIONS)]
            for null in nulls
        ) / len(nulls)
        return float(
            (1 + np.sum(np.abs(sim) >= abs(observed)))
            / (self.N_GLOBAL_PERMUTATIONS + 1)
        )

    def _sources(self, functional, baselines, coordinate):
        for signature in self.SIGNATURES:
            for component in self.COMPONENTS:
                yield {
                    "source_type": "functional",
                    "signature": signature,
                    "component": component,
                    "source_metric": (
                        "euclidean" if coordinate == "absolute" else "angular"
                    ),
                    "matrices": functional[(coordinate, signature, component)],
                }
        for name in self.BASELINES:
            yield {
                "source_type": "baseline",
                "signature": "",
                "component": "",
                "source_metric": name,
                "matrices": baselines[(coordinate, name)],
            }

    def alignment(self, functional, behavior, baselines):
        dataset_rows, overall_rows = [], []
        rng = np.random.default_rng(self.RANDOM_SEED)

        for coordinate in self.COORDINATES:
            target = behavior[coordinate]
            behavior_metric = (
                "jsd" if coordinate == "absolute" else "angular_centered_logit_delta"
            )
            for spec in self._sources(functional, baselines, coordinate):
                current = []
                for ti, dataset in enumerate(self.DATASETS):
                    rho, p, null = self._rsa(spec["matrices"][ti], target[ti])
                    row = {
                        "coordinate_space": coordinate,
                        "source_type": spec["source_type"],
                        "signature": spec["signature"],
                        "component": spec["component"],
                        "source_metric": spec["source_metric"],
                        "behavior_metric": behavior_metric,
                        "dataset": dataset,
                        "dataset_group": self._group[dataset],
                        "rho": rho,
                        "p_permutation": p,
                        "n_methods": len(self.METHODS),
                        "n_edges": len(self.METHODS) * (len(self.METHODS) - 1) // 2,
                        "null": null,
                    }
                    dataset_rows.append(row)
                    current.append(row)

                values = [row["rho"] for row in current]
                groups = [row["dataset_group"] for row in current]
                rho = _mean(values)
                lo, hi = _bootstrap(values, groups, self.N_BOOTSTRAP, rng)
                overall_rows.append(
                    {
                        **{
                            key: current[0][key]
                            for key in (
                                "coordinate_space",
                                "source_type",
                                "signature",
                                "component",
                                "source_metric",
                                "behavior_metric",
                                "n_methods",
                                "n_edges",
                            )
                        },
                        "rho": rho,
                        "ci_low": lo,
                        "ci_high": hi,
                        "p_permutation": self._global_p(current, rho, rng),
                        "q_bh": np.nan,
                        "n_datasets": len(current),
                    }
                )

        for coordinate in self.COORDINATES:
            idx = [
                i
                for i, row in enumerate(overall_rows)
                if row["coordinate_space"] == coordinate
            ]
            for i, q in zip(idx, _bh([overall_rows[i]["p_permutation"] for i in idx])):
                overall_rows[i]["q_bh"] = q
        return dataset_rows, overall_rows

    def anchor_alignment(self, functional, behavior, baselines):
        dataset_method_rows, method_rows = [], []
        rng = np.random.default_rng(self.RANDOM_SEED)

        for coordinate in self.COORDINATES:
            target = behavior[coordinate]
            behavior_metric = (
                "jsd" if coordinate == "absolute" else "angular_centered_logit_delta"
            )
            for spec in self._sources(functional, baselines, coordinate):
                for anchor, method in enumerate(self.METHODS):
                    current = []
                    for ti, dataset in enumerate(self.DATASETS):
                        rho, p = self._anchor_rsa(
                            spec["matrices"][ti], target[ti], anchor
                        )
                        row = {
                            "coordinate_space": coordinate,
                            "source_type": spec["source_type"],
                            "signature": spec["signature"],
                            "component": spec["component"],
                            "source_metric": spec["source_metric"],
                            "behavior_metric": behavior_metric,
                            "dataset": dataset,
                            "dataset_group": self._group[dataset],
                            "anchor_method": method,
                            "rho": rho,
                            "p_permutation": p,
                            "n_comparisons": len(self.METHODS) - 1,
                        }
                        dataset_method_rows.append(row)
                        current.append(row)

                    values = [row["rho"] for row in current]
                    groups = [row["dataset_group"] for row in current]
                    rho = _mean(values)
                    lo, hi = _bootstrap(values, groups, self.N_BOOTSTRAP, rng)
                    method_rows.append(
                        {
                            **{
                                key: current[0][key]
                                for key in (
                                    "coordinate_space",
                                    "source_type",
                                    "signature",
                                    "component",
                                    "source_metric",
                                    "behavior_metric",
                                    "anchor_method",
                                    "n_comparisons",
                                )
                            },
                            "rho": rho,
                            "ci_low": lo,
                            "ci_high": hi,
                            "n_datasets": len(current),
                        }
                    )
        return dataset_method_rows, method_rows

    def layerwise_alignment(self, layerwise, behavior):
        rows = []
        rng = np.random.default_rng(self.RANDOM_SEED)

        for coordinate in self.COORDINATES:
            target = behavior[coordinate]
            for signature in self.SIGNATURES:
                for component in self.COMPONENTS:
                    source = layerwise[
                        (coordinate, signature, component)
                    ]  # [dataset, layer, method, method]
                    for layer in range(source.shape[1]):
                        current = []
                        for ti, dataset in enumerate(self.DATASETS):
                            rho, p, _ = self._rsa(source[ti, layer], target[ti])
                            current.append((dataset, rho, p))
                            rows.append(
                                {
                                    "coordinate_space": coordinate,
                                    "signature": signature,
                                    "component": component,
                                    "layer": layer + 1,
                                    "aggregation": "dataset",
                                    "dataset": dataset,
                                    "dataset_group": self._group[dataset],
                                    "rho": rho,
                                    "ci_low": np.nan,
                                    "ci_high": np.nan,
                                    "p_permutation": p,
                                }
                            )

                        values = [rho for _, rho, _ in current]
                        groups = [self._group[dataset] for dataset, _, _ in current]
                        rho = _mean(values)
                        lo, hi = _bootstrap(values, groups, self.N_BOOTSTRAP, rng)
                        rows.append(
                            {
                                "coordinate_space": coordinate,
                                "signature": signature,
                                "component": component,
                                "layer": layer + 1,
                                "aggregation": "all",
                                "dataset": "",
                                "dataset_group": "",
                                "rho": rho,
                                "ci_low": lo,
                                "ci_high": hi,
                                "p_permutation": np.nan,
                            }
                        )
        return rows

    def component_correlations(self, functional):
        rows = []
        for coordinate in self.COORDINATES:
            for signature in self.SIGNATURES:
                for a, b in itertools.combinations(self.COMPONENTS, 2):
                    values = []
                    for ti, dataset in enumerate(self.DATASETS):
                        rho = _spearman(
                            _upper(functional[(coordinate, signature, a)][ti]),
                            _upper(functional[(coordinate, signature, b)][ti]),
                        )
                        values.append(rho)
                        rows.append(
                            {
                                "coordinate_space": coordinate,
                                "signature": signature,
                                "component_a": a,
                                "component_b": b,
                                "aggregation": "dataset",
                                "dataset": dataset,
                                "dataset_group": self._group[dataset],
                                "rho": rho,
                            }
                        )
                    rows.append(
                        {
                            "coordinate_space": coordinate,
                            "signature": signature,
                            "component_a": a,
                            "component_b": b,
                            "aggregation": "all",
                            "dataset": "",
                            "dataset_group": "",
                            "rho": _mean(values),
                        }
                    )
        return rows

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def pairwise_rows(self, functional, behavior, baselines):
        rows = []
        for ti, dataset in enumerate(self.DATASETS):
            for i, j in itertools.combinations(range(len(self.METHODS)), 2):
                row = {
                    "dataset": dataset,
                    "dataset_group": self._group[dataset],
                    "method_a": self.METHODS[i],
                    "method_b": self.METHODS[j],
                    "prediction__absolute__jsd": behavior["absolute"][ti, i, j],
                    "prediction__lp_relative__angular_centered_logit_delta": behavior[
                        "lp_relative"
                    ][ti, i, j],
                }
                for coordinate in self.COORDINATES:
                    for name in self.BASELINES:
                        row[f"baseline__{coordinate}__{name}"] = baselines[
                            (coordinate, name)
                        ][ti, i, j]
                    for signature in self.SIGNATURES:
                        for component in self.COMPONENTS:
                            row[
                                f"functional__{coordinate}__sqrt__{signature}__{component}"
                            ] = functional[(coordinate, signature, component)][ti, i, j]
                rows.append(row)
        return rows

    def _manifest(self):
        data = {
            "backbone": self.backbone_name,
            "datasets": list(self.DATASETS),
            "dataset_groups": {key: list(value) for key, value in self.GROUPS.items()},
            "methods": list(self.METHODS),
            "reference_method": self.LP,
            "seeds": list(self.SEEDS),
            "components": list(self.COMPONENTS),
            "signatures": list(self.SIGNATURES),
            "functional_inputs": "functional_signature.npz with exact diagonal F, P, and sample-normalized S",
            "representation": "elementwise sqrt of each non-negative F/P/S signature",
            "absolute_functional_distance": "Euclidean distance on sqrt signatures",
            "lp_relative_functional_distance": "angular distance after subtracting the per-dataset mean LP sqrt signature over seeds",
            "absolute_prediction_distance": "mean per-example Jensen-Shannon divergence over all cross-seed prediction pairs",
            "lp_relative_prediction": "per-example class-centered logits minus the per-dataset mean LP centered logits",
            "lp_relative_prediction_distance": "angular distance after flattening centered-logit deltas",
            "prediction_cache": "unchanged: prediction_cache/test/<dataset>/<method>/seed-<seed>.npz",
            "activation_baseline_caches": "unchanged: cka_cache, procrustes_cache, cka_delta_cache, procrustes_delta_cache; caches containing full_ft are sliced to PEFT methods",
            "central_rdms": "mean of all K^2 cross-seed distances for each PEFT-method pair",
            "full_rsa": "dataset-wise Spearman correlation of the 10 upper-triangle PEFT-pair distances with exact method-label permutations",
            "anchor_rsa": "dataset x anchor-method Spearman correlation over the four distances from that anchor to the other PEFT methods",
            "overall_aggregation": "mean dataset-wise rho with bootstrap stratified by VTAB group",
            "component_correlation": "Spearman correlation between component-induced PEFT RDM upper triangles, computed per dataset then averaged",
            "n_bootstrap": self.N_BOOTSTRAP,
            "n_global_permutations": self.N_GLOBAL_PERMUTATIONS,
            "random_seed": self.RANDOM_SEED,
        }
        path = self.out_dir / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def _component_label(name):
        return (
            name.replace("attention_", "attn ")
            .replace("mlp_", "mlp ")
            .replace("_", " ")
        )

    def _summary_page(self, pdf, rows, title):
        fig, axes = plt.subplots(
            1, len(self.SIGNATURES), figsize=(14, 5.6), sharex=True
        )
        y = np.arange(len(self.COMPONENTS))

        for ax, signature in zip(axes, self.SIGNATURES):
            functional = {
                row["component"]: row
                for row in rows
                if row["source_type"] == "functional" and row["signature"] == signature
            }
            for yi, component in enumerate(self.COMPONENTS):
                row = functional.get(component)
                if row is None or not np.isfinite(row["rho"]):
                    continue
                xerr = None
                if np.isfinite(row.get("ci_low", np.nan)) and np.isfinite(
                    row.get("ci_high", np.nan)
                ):
                    xerr = np.array(
                        [
                            [max(0, row["rho"] - row["ci_low"])],
                            [max(0, row["ci_high"] - row["rho"])],
                        ]
                    )
                ax.errorbar(row["rho"], yi, xerr=xerr, fmt="o", capsize=2)

            for name, style in zip(self.BASELINES, ("--", ":")):
                row = next(
                    (
                        r
                        for r in rows
                        if r["source_type"] == "baseline" and r["source_metric"] == name
                    ),
                    None,
                )
                if row is not None and np.isfinite(row["rho"]):
                    ax.axvline(
                        row["rho"], linestyle=style, linewidth=1.2, label=name.upper()
                    )

            ax.set_yticks(
                y, [self._component_label(component) for component in self.COMPONENTS]
            )
            ax.set_ylim(len(y) - 0.5, -0.5)
            ax.set_xlim(-1, 1)
            ax.axvline(0, linewidth=0.8)
            ax.grid(axis="x", linewidth=0.4)
            ax.set_title(signature)
            ax.set_xlabel("Spearman ρ")
            ax.legend(frameon=False, fontsize=8)

        fig.suptitle(title)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    @staticmethod
    def _heatmap(ax, matrix, rows, cols, title):
        image = ax.imshow(matrix, aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(cols)), cols, rotation=35, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(rows)), rows, fontsize=7)
        ax.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                    )
        return image

    def _figures(
        self,
        overall,
        by_dataset,
        by_method,
        by_dataset_method,
        layerwise,
        component_corr,
    ):
        path = self.out_dir / "functional_signature_validation.pdf"
        with PdfPages(path) as pdf:
            # Overall
            for coordinate in self.COORDINATES:
                rows = [row for row in overall if row["coordinate_space"] == coordinate]
                self._summary_page(
                    pdf, rows, f"Overall · {coordinate.replace('_', ' ')}"
                )

            # Per dataset, same summary format with CKA / Procrustes reference lines.
            for dataset in self.DATASETS:
                for coordinate in self.COORDINATES:
                    rows = [
                        row
                        for row in by_dataset
                        if row["dataset"] == dataset
                        and row["coordinate_space"] == coordinate
                    ]
                    self._summary_page(
                        pdf,
                        rows,
                        f"{dataset} · {coordinate.replace('_', ' ')}",
                    )

            # Per anchor method, aggregated across datasets.
            for method in self.METHODS:
                for coordinate in self.COORDINATES:
                    rows = [
                        row
                        for row in by_method
                        if row["anchor_method"] == method
                        and row["coordinate_space"] == coordinate
                    ]
                    self._summary_page(
                        pdf,
                        rows,
                        f"Anchor: {method} · {coordinate.replace('_', ' ')}",
                    )

            # Dataset x anchor method heatmaps: two coordinates x three signatures per dataset.
            component_labels = [
                self._component_label(component) for component in self.COMPONENTS
            ]
            for dataset in self.DATASETS:
                fig, axes = plt.subplots(2, len(self.SIGNATURES), figsize=(15, 8))
                image = None
                for ri, coordinate in enumerate(self.COORDINATES):
                    for ci, signature in enumerate(self.SIGNATURES):
                        matrix = np.full(
                            (len(self.COMPONENTS), len(self.METHODS)), np.nan
                        )
                        for i, component in enumerate(self.COMPONENTS):
                            for j, method in enumerate(self.METHODS):
                                row = next(
                                    (
                                        r
                                        for r in by_dataset_method
                                        if r["source_type"] == "functional"
                                        and r["dataset"] == dataset
                                        and r["coordinate_space"] == coordinate
                                        and r["signature"] == signature
                                        and r["component"] == component
                                        and r["anchor_method"] == method
                                    ),
                                    None,
                                )
                                if row is not None:
                                    matrix[i, j] = row["rho"]
                        image = self._heatmap(
                            axes[ri, ci],
                            matrix,
                            component_labels,
                            list(self.METHODS),
                            f"{coordinate.replace('_', ' ')} · {signature}",
                        )
                fig.suptitle(f"Dataset × anchor-method alignment · {dataset}")
                fig.tight_layout()
                if image is not None:
                    fig.colorbar(
                        image, ax=axes.ravel().tolist(), shrink=0.7, label="Spearman ρ"
                    )
                pdf.savefig(fig)
                plt.close(fig)

            # Layerwise alignment.
            for coordinate in self.COORDINATES:
                fig, axes = plt.subplots(
                    1, len(self.SIGNATURES), figsize=(13, 6), sharey=True
                )
                image = None
                for ax, signature in zip(axes, self.SIGNATURES):
                    rows = [
                        row
                        for row in layerwise
                        if row["aggregation"] == "all"
                        and row["coordinate_space"] == coordinate
                        and row["signature"] == signature
                    ]
                    n_layers = max(row["layer"] for row in rows)
                    matrix = np.full((n_layers, len(self.COMPONENTS)), np.nan)
                    for row in rows:
                        matrix[
                            row["layer"] - 1, self.COMPONENTS.index(row["component"])
                        ] = row["rho"]
                    image = self._heatmap(
                        ax,
                        matrix,
                        [str(i) for i in range(1, n_layers + 1)],
                        component_labels,
                        signature,
                    )
                    ax.set_ylabel("Layer")
                fig.suptitle(
                    f"Layerwise prediction alignment · {coordinate.replace('_', ' ')}"
                )
                fig.tight_layout()
                if image is not None:
                    fig.colorbar(
                        image, ax=axes.ravel().tolist(), shrink=0.75, label="Spearman ρ"
                    )
                pdf.savefig(fig)
                plt.close(fig)

            # Correlation between component-induced method geometries.
            for coordinate in self.COORDINATES:
                fig, axes = plt.subplots(1, len(self.SIGNATURES), figsize=(13, 4.5))
                image = None
                for ax, signature in zip(axes, self.SIGNATURES):
                    matrix = np.eye(len(self.COMPONENTS))
                    for row in component_corr:
                        if (
                            row["aggregation"] == "all"
                            and row["coordinate_space"] == coordinate
                            and row["signature"] == signature
                        ):
                            i = self.COMPONENTS.index(row["component_a"])
                            j = self.COMPONENTS.index(row["component_b"])
                            matrix[i, j] = matrix[j, i] = row["rho"]
                    image = self._heatmap(
                        ax, matrix, component_labels, component_labels, signature
                    )
                fig.suptitle(
                    f"Component geometry correlations · {coordinate.replace('_', ' ')}"
                )
                fig.tight_layout()
                if image is not None:
                    fig.colorbar(
                        image, ax=axes.ravel().tolist(), shrink=0.8, label="Spearman ρ"
                    )
                pdf.savefig(fig)
                plt.close(fig)

        return path

    def run(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._phase("[1/5] Functional signature geometry")
        functional, layerwise_geometry = self.functional_geometry()

        self._phase("[2/5] Prediction geometry")
        behavior = self.behavior_geometry()

        self._phase("[3/5] Final-CLS CKA / Procrustes baselines")
        baselines = self.baseline_geometry()

        self._phase("[4/5] Signature validation analyses")
        by_dataset, overall = self.alignment(functional, behavior, baselines)
        by_dataset_method, by_method = self.anchor_alignment(
            functional, behavior, baselines
        )
        layerwise = self.layerwise_alignment(layerwise_geometry, behavior)
        component_corr = self.component_correlations(functional)
        pairwise = self.pairwise_rows(functional, behavior, baselines)

        self._phase("[5/5] Tables and figures")
        clean = lambda rows: [
            {key: value for key, value in row.items() if key != "null"} for row in rows
        ]
        outputs = {
            "manifest": self._manifest(),
            "pairwise_distances": _write_csv(
                self.out_dir / "pairwise_distances.csv", pairwise
            ),
            "alignment_overall": _write_csv(
                self.out_dir / "alignment_overall.csv", clean(overall)
            ),
            "alignment_by_dataset": _write_csv(
                self.out_dir / "alignment_by_dataset.csv", clean(by_dataset)
            ),
            "alignment_by_method": _write_csv(
                self.out_dir / "alignment_by_method.csv", by_method
            ),
            "alignment_by_dataset_method": _write_csv(
                self.out_dir / "alignment_by_dataset_method.csv", by_dataset_method
            ),
            "alignment_layerwise": _write_csv(
                self.out_dir / "alignment_layerwise.csv", layerwise
            ),
            "component_correlations": _write_csv(
                self.out_dir / "component_correlations.csv", component_corr
            ),
        }
        outputs["figure"] = self._figures(
            overall,
            by_dataset,
            by_method,
            by_dataset_method,
            layerwise,
            component_corr,
        )
        return outputs
