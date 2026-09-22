import csv
import gc
import itertools
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import rankdata
from tqdm.auto import tqdm

from datasets import datasets
from models import models
from utils import seed_worker

# -----------------------------------------------------------------------------
# Small numerical / I/O helpers
# -----------------------------------------------------------------------------


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
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


def _read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _assert_close(a, b, label, rtol=1e-8, atol=1e-9):
    if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
        err = float(np.nanmax(np.abs(np.asarray(a) - np.asarray(b))))
        raise RuntimeError(f"{label} failed; max error={err}")


def _safe_ratio(a, b):
    return np.nan if abs(float(b)) <= np.finfo(float).eps else float(a) / float(b)


def _cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return np.nan if d <= np.finfo(float).eps else float(np.dot(a, b) / d)


def _upper(x):
    return x[np.triu_indices(x.shape[0], 1)]


def _spearman(a, b):
    a, b = np.asarray(a, float).reshape(-1), np.asarray(b, float).reshape(-1)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def _row_corr(x, y):
    x, y = x - x.mean(1, keepdims=True), y - y.mean()
    d = np.linalg.norm(x, axis=1) * np.linalg.norm(y)
    return np.divide(x @ y, d, out=np.full(x.shape[0], np.nan), where=d > 1e-12)


def _bh(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    if not idx.size:
        return out
    order = idx[np.argsort(p[idx])]
    q = p[order] * len(order) / np.arange(1, len(order) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out[order] = np.minimum(q, 1.0)
    return out


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class FunctionalReorganizationAnalyzer:
    """RQ1 analysis: functional reorganization, behavioral grounding, and CKA baseline."""

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
    METHODS = (REFERENCE_METHOD, *ADAPTATION_METHODS)
    METHOD_SCOPES = {"peft_only": PEFT_METHODS, "peft_plus_ft": ADAPTATION_METHODS}
    SEEDS = (0, 1, 2)

    SOURCE_SIGNATURES = ("F_m", "F_e", "P")
    BEHAVIOR_METRICS = ("jsd", "top1_disagreement", "error_dissimilarity")
    PREDICTION_METRICS = (*BEHAVIOR_METRICS, "accuracy_gap")

    CORE_COMPONENTS = (
        "residual_carry",
        "sublayer_update",
        "attention_selection_route",
        "attention_transport_route",
    )
    ALL_COMPONENTS = (
        *CORE_COMPONENTS,
        "sublayer_input_channel",
        "attention_head_selection_route",
        "attention_head_transport_route",
        "mlp_neuron",
        "final_cls_readout_channel",
        "residual_token_carry",
        "sublayer_token_update",
        "attention_head_output",
    )

    EVAL_EXPERIMENT_NAME = "top_1_acc_eval"
    SIGNATURE_FILENAME = "functional_signatures.npz"
    LEVELS = ("dataset_method", "dataset", "method", "global")

    def __init__(
        self,
        backbone_name="vit_b16",
        eps=1e-12,
        verbose=True,
        n_bootstrap=2000,
        n_global_permutations=10000,
        random_seed=0,
    ):
        self.backbone_name = str(backbone_name)
        self.experiments_dir = Path(os.environ.get("EXPERIMENTS_DIR", "./experiments"))
        self.save_dir = (
            self.experiments_dir / "functional_reorganization" / self.backbone_name
        )
        self.eps = float(eps)
        self.verbose = bool(verbose)
        self.n_bootstrap = int(n_bootstrap)
        self.n_global_permutations = int(n_global_permutations)
        self.random_seed = int(random_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._task_group = {
            t: g for g, tasks in self.TASK_GROUPS.items() for t in tasks
        }
        self._group_indices = {
            g: np.array([self.DATASETS.index(t) for t in tasks], dtype=int)
            for g, tasks in self.TASK_GROUPS.items()
        }
        self._perms = {
            n: np.asarray(list(itertools.permutations(range(n))), dtype=int)
            for n in {len(x) for x in self.METHOD_SCOPES.values()}
        }
        self._runs_cache = {}

    # ------------------------------------------------------------------
    # Paths / experiment loading
    # ------------------------------------------------------------------

    def _progress(self, x, desc, total=None):
        return tqdm(
            x,
            desc=desc,
            total=total,
            disable=not self.verbose,
            leave=False,
            dynamic_ncols=True,
        )

    def _dir(self, *parts):
        path = self.save_dir.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _scope_methods(self, scope):
        return self.METHOD_SCOPES[scope]

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
        if key in self._runs_cache:
            return self._runs_cache[key]
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
        seeds = tuple(r["seed"] for r in runs)
        if seeds != self.SEEDS:
            raise RuntimeError(
                f"Expected seeds {self.SEEDS} for {task}/{method}, found {seeds}"
            )
        self._runs_cache[key] = runs
        return runs

    # ------------------------------------------------------------------
    # Functional signatures
    # ------------------------------------------------------------------

    def _load_component(self, scope, component):
        methods = self._scope_methods(scope)
        tasks = {sig: [] for sig in self.SOURCE_SIGNATURES}
        native_shape = None

        for task in self._progress(self.DATASETS, f"Signatures {scope}/{component}"):
            by_method = {sig: {} for sig in self.SOURCE_SIGNATURES}
            for method in (self.REFERENCE_METHOD, *methods):
                per_seed = {sig: [] for sig in self.SOURCE_SIGNATURES}
                for run in self._get_runs(task, method):
                    path = run["path"] / self.SIGNATURE_FILENAME
                    with np.load(path, allow_pickle=False) as data:
                        for sig in self.SOURCE_SIGNATURES:
                            key = f"{sig}__{component}"
                            if key not in data:
                                raise KeyError(f"Missing {key!r} in {path}")
                            value = np.asarray(data[key], np.float64)
                            if np.any(value < -self.eps):
                                raise ValueError(f"Negative values in {key!r}: {path}")
                            if native_shape is None:
                                native_shape = value.shape
                            elif value.shape != native_shape:
                                raise ValueError(
                                    f"Shape mismatch for {component}: expected {native_shape}, "
                                    f"got {value.shape} in {path}"
                                )
                            per_seed[sig].append(
                                np.log(np.maximum(value, 0.0) + self.eps).reshape(-1)
                            )
                for sig in self.SOURCE_SIGNATURES:
                    by_method[sig][method] = np.stack(per_seed[sig])

            for sig in self.SOURCE_SIGNATURES:
                lp = by_method[sig][self.REFERENCE_METHOD].mean(0)
                tasks[sig].append(np.stack([by_method[sig][m] - lp for m in methods]))

        out = {sig: np.stack(values) for sig, values in tasks.items()}
        out["S_m"] = out["F_m"] - out["P"]
        out["S_e"] = out["F_e"] - out["P"]
        _assert_close(out["F_m"], out["P"] + out["S_m"], (component, "model F=P+S"))
        _assert_close(out["F_e"], out["P"] + out["S_e"], (component, "observed F=P+S"))
        return out, native_shape

    @staticmethod
    def _normalize_component(x):
        return x / np.sqrt(x.shape[-1])

    @staticmethod
    def _balanced_core(arrays):
        c = len(arrays)
        return np.concatenate([x / np.sqrt(c * x.shape[-1]) for x in arrays], axis=-1)

    # ------------------------------------------------------------------
    # Four-level decomposition
    # ------------------------------------------------------------------

    def _effects(self, x, group_labels=None):
        cell = x.mean(2)
        grand = cell.mean((0, 1))
        y = cell - grand
        task = y.mean(1)
        method = y.mean(0)
        interaction = y - task[:, None] - method[None]
        labels = np.asarray(
            group_labels
            if group_labels is not None
            else [self._task_group[t] for t in self.DATASETS]
        )
        group = np.empty_like(task)
        for g in np.unique(labels):
            idx = np.flatnonzero(labels == g)
            group[idx] = task[idx].mean(0)
        dataset = task - group
        seed = x - cell[:, :, None]
        centered = x - grand
        T, M, S = x.shape[:3]
        ss = {
            "group": M * S * np.sum(group**2),
            "dataset": M * S * np.sum(dataset**2),
            "method": T * S * np.sum(method**2),
            "interaction": S * np.sum(interaction**2),
            "seed": np.sum(seed**2),
        }
        total = np.sum(centered**2)
        _assert_close(total, sum(ss.values()), "SS decomposition")
        return {
            "grand": grand,
            "cell": cell,
            "y": y,
            "task": task,
            "group": group,
            "dataset": dataset,
            "method": method,
            "interaction": interaction,
            "seed": seed,
            "centered": centered,
            "ss": ss,
            "total": total,
        }

    def _bootstrap_decomposition(self, x):
        keys = ("group", "dataset", "method", "interaction", "seed")
        if self.n_bootstrap <= 0:
            return {f"r_{k}_ci_low": np.nan for k in keys} | {
                f"r_{k}_ci_high": np.nan for k in keys
            }

        rng = np.random.default_rng(self.random_seed)
        vals = {k: [] for k in keys}
        for _ in range(self.n_bootstrap):
            idx, labels = [], []
            for g, original in self._group_indices.items():
                sampled = rng.choice(original, size=len(original), replace=True)
                idx.extend(sampled.tolist())
                labels.extend([g] * len(original))
            xb = x[np.asarray(idx)]
            T, M, S = xb.shape[:3]
            seed_idx = rng.integers(0, S, size=(T, M, S))
            xb = xb[np.arange(T)[:, None, None], np.arange(M)[None, :, None], seed_idx]
            e = self._effects(xb, labels)
            for k in keys:
                vals[k].append(_safe_ratio(e["ss"][k], e["total"]))

        out = {}
        for k in keys:
            out[f"r_{k}_ci_low"], out[f"r_{k}_ci_high"] = np.nanquantile(
                vals[k], [0.025, 0.975]
            )
        task = np.asarray(vals["group"]) + np.asarray(vals["dataset"])
        out["r_task_ci_low"], out["r_task_ci_high"] = np.nanquantile(
            task, [0.025, 0.975]
        )
        return out

    @staticmethod
    def _ps_row(meta, effect, f, p, s, weight=1.0):
        ss_f, ss_p, ss_s = (weight * np.sum(x**2) for x in (f, p, s))
        cross = 2 * weight * np.sum(p * s)
        _assert_close(ss_f, ss_p + ss_s + cross, (*meta.values(), effect, "PS"))
        return {
            **meta,
            "effect": effect,
            "ss_f": float(ss_f),
            "ss_p": float(ss_p),
            "ss_s": float(ss_s),
            "cross_ps": float(cross),
            "p_over_f": _safe_ratio(ss_p, ss_f),
            "s_over_f": _safe_ratio(ss_s, ss_f),
            "cross_over_f": _safe_ratio(cross, ss_f),
            "cos_ps": _cosine(p.reshape(-1), s.reshape(-1)),
        }

    def _method_task_parts(self, e, mi):
        h = e["task"] + e["interaction"][:, mi]
        group = np.empty_like(h)
        for idx in self._group_indices.values():
            group[idx] = h[idx].mean(0)
        return group, h - group

    def _decomposition_rows(self, scope, channel, signature, component, x, e):
        methods = self._scope_methods(scope)
        T, M, S = x.shape[:3]
        base = {
            "scope": scope,
            "channel": channel,
            "signature": signature,
            "component": component,
        }
        out = {level: [] for level in self.LEVELS}

        for ti, task in enumerate(self.DATASETS):
            task_ss = M * S * np.sum(e["task"][ti] ** 2)
            conditioned = e["method"] + e["interaction"][ti]
            method_ss = S * np.sum(conditioned**2)
            seed_ss = np.sum(e["seed"][ti] ** 2)
            total = np.sum(e["centered"][ti] ** 2)
            _assert_close(total, task_ss + method_ss + seed_ss, (task, signature))
            out["dataset"].append(
                {
                    **base,
                    "task": task,
                    "task_group": self._task_group[task],
                    "ss_task_shared": float(task_ss),
                    "ss_method_conditioned": float(method_ss),
                    "ss_seed": float(seed_ss),
                    "ss_total": float(total),
                    "r_task_shared": _safe_ratio(task_ss, total),
                    "r_method_conditioned": _safe_ratio(method_ss, total),
                    "r_seed": _safe_ratio(seed_ss, total),
                }
            )

            for mi, method in enumerate(methods):
                y = e["y"][ti, mi]
                stable = S * np.sum(y**2)
                parts = {
                    "group": e["group"][ti],
                    "dataset": e["dataset"][ti],
                    "method": e["method"][mi],
                    "interaction": e["interaction"][ti, mi],
                }
                contrib = {k: float(S * np.dot(y, v)) for k, v in parts.items()}
                _assert_close(stable, sum(contrib.values()), (task, method, signature))
                seed = float(np.sum(e["seed"][ti, mi] ** 2))
                total_cell = stable + seed
                gross = sum(abs(v) for v in contrib.values())
                row = {
                    **base,
                    "task": task,
                    "task_group": self._task_group[task],
                    "method": method,
                    "ss_stable": float(stable),
                    "ss_seed": seed,
                    "ss_total": float(total_cell),
                    "r_stable": _safe_ratio(stable, total_cell),
                    "r_seed": _safe_ratio(seed, total_cell),
                }
                for k, v in contrib.items():
                    row[f"c_{k}"] = v
                    row[f"r_{k}_signed"] = _safe_ratio(v, stable)
                    row[f"r_{k}_gross"] = _safe_ratio(abs(v), gross)
                out["dataset_method"].append(row)

        for mi, method in enumerate(methods):
            group, dataset = self._method_task_parts(e, mi)
            ss_method = T * S * np.sum(e["method"][mi] ** 2)
            ss_group = S * np.sum(group**2)
            ss_dataset = S * np.sum(dataset**2)
            ss_seed = np.sum(e["seed"][:, mi] ** 2)
            total = np.sum(e["centered"][:, mi] ** 2)
            _assert_close(
                total, ss_method + ss_group + ss_dataset + ss_seed, (method, signature)
            )
            out["method"].append(
                {
                    **base,
                    "method": method,
                    "ss_method_general": float(ss_method),
                    "ss_task_group": float(ss_group),
                    "ss_dataset_specific": float(ss_dataset),
                    "ss_task_conditioned": float(ss_group + ss_dataset),
                    "ss_seed": float(ss_seed),
                    "ss_total": float(total),
                    "r_method_general": _safe_ratio(ss_method, total),
                    "r_task_group": _safe_ratio(ss_group, total),
                    "r_dataset_specific": _safe_ratio(ss_dataset, total),
                    "r_task_conditioned": _safe_ratio(ss_group + ss_dataset, total),
                    "r_seed": _safe_ratio(ss_seed, total),
                }
            )

        ss = e["ss"]
        row = {
            **base,
            **{f"ss_{k}": float(v) for k, v in ss.items()},
            "ss_task": float(ss["group"] + ss["dataset"]),
            "ss_total": float(e["total"]),
            **{f"r_{k}": _safe_ratio(v, e["total"]) for k, v in ss.items()},
            "r_task": _safe_ratio(ss["group"] + ss["dataset"], e["total"]),
        }
        if component == "balanced_core":
            row |= self._bootstrap_decomposition(x)
        out["global"].append(row)
        return out

    def _ps_rows(self, scope, channel, component, ef, ep, es):
        methods = self._scope_methods(scope)
        T, M, S = ef["centered"].shape[:3]
        base = {"scope": scope, "channel": channel, "component": component}
        out = {level: [] for level in self.LEVELS}

        for ti, task in enumerate(self.DATASETS):
            meta = {**base, "task": task, "task_group": self._task_group[task]}
            for effect, key, weight in (
                ("task_shared", "task", M * S),
                ("seed", "seed", 1),
                ("total", "centered", 1),
            ):
                out["dataset"].append(
                    self._ps_row(
                        meta, effect, ef[key][ti], ep[key][ti], es[key][ti], weight
                    )
                )
            out["dataset"].append(
                self._ps_row(
                    meta,
                    "method_conditioned",
                    ef["method"] + ef["interaction"][ti],
                    ep["method"] + ep["interaction"][ti],
                    es["method"] + es["interaction"][ti],
                    S,
                )
            )
            for mi, method in enumerate(methods):
                cell_meta = {**meta, "method": method}
                for effect, key in (
                    ("stable", "y"),
                    ("seed", "seed"),
                    ("total", "centered"),
                ):
                    out["dataset_method"].append(
                        self._ps_row(
                            cell_meta,
                            effect,
                            ef[key][ti, mi],
                            ep[key][ti, mi],
                            es[key][ti, mi],
                            S if effect == "stable" else 1,
                        )
                    )

        for mi, method in enumerate(methods):
            meta = {**base, "method": method}
            gf, df = self._method_task_parts(ef, mi)
            gp, dp = self._method_task_parts(ep, mi)
            gs, ds = self._method_task_parts(es, mi)
            rows = (
                (
                    "method_general",
                    ef["method"][mi],
                    ep["method"][mi],
                    es["method"][mi],
                    T * S,
                ),
                ("task_group", gf, gp, gs, S),
                ("dataset_specific", df, dp, ds, S),
                ("task_conditioned", gf + df, gp + dp, gs + ds, S),
                ("seed", ef["seed"][:, mi], ep["seed"][:, mi], es["seed"][:, mi], 1),
                (
                    "total",
                    ef["centered"][:, mi],
                    ep["centered"][:, mi],
                    es["centered"][:, mi],
                    1,
                ),
            )
            out["method"].extend(self._ps_row(meta, *row) for row in rows)

        rows = (
            ("group", ef["group"], ep["group"], es["group"], M * S),
            ("dataset", ef["dataset"], ep["dataset"], es["dataset"], M * S),
            ("task", ef["task"], ep["task"], es["task"], M * S),
            ("method", ef["method"], ep["method"], es["method"], T * S),
            ("interaction", ef["interaction"], ep["interaction"], es["interaction"], S),
            ("seed", ef["seed"], ep["seed"], es["seed"], 1),
            ("total", ef["centered"], ep["centered"], es["centered"], 1),
        )
        out["global"].extend(self._ps_row(base, *row) for row in rows)
        return out

    # ------------------------------------------------------------------
    # Functional pairwise geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _within_seed(x):
        centroid = x.mean(0)
        variance = float(np.mean(np.sum((x - centroid) ** 2, axis=-1)))
        pairs = list(itertools.combinations(range(len(x)), 2))
        pair_sq = [np.sum((x[i] - x[j]) ** 2) for i, j in pairs]
        return (
            variance,
            float(np.sqrt(np.mean(pair_sq))),
            [float(np.sqrt(v)) for v in pair_sq],
        )

    def _functional_distances(self, scope, channel, component, f, p, s):
        methods = self._scope_methods(scope)
        T, M = len(self.DATASETS), len(methods)
        matrices = {name: np.zeros((T, M, M), float) for name in ("F", "P", "S")}
        pair_rows, seed_rows, within = [], [], {}

        for ti, task in enumerate(self.DATASETS):
            for mi, method in enumerate(methods):
                values = {}
                for name, x in (("F", f), ("P", p), ("S", s)):
                    var, rms, pairs = self._within_seed(x[ti, mi])
                    values[f"within_variance_{name.lower()}"] = var
                    values[f"within_pair_rms_{name.lower()}"] = rms
                    for pi, value in enumerate(pairs):
                        values[f"pair_{pi}_{name.lower()}"] = value
                    within[(ti, mi, name)] = (var, rms)
                seed_rows.append(
                    {
                        "scope": scope,
                        "task": task,
                        "task_group": self._task_group[task],
                        "method": method,
                        "channel": channel,
                        "component": component,
                        **values,
                    }
                )

            for i, j in itertools.combinations(range(M), 2):
                row = {
                    "scope": scope,
                    "task": task,
                    "task_group": self._task_group[task],
                    "method_a": methods[i],
                    "method_b": methods[j],
                    "channel": channel,
                    "component": component,
                    "n_seed_pairs": len(self.SEEDS) ** 2,
                }
                diffs = {}
                for name, x in (("F", f), ("P", p), ("S", s)):
                    d = x[ti, i, :, None, :] - x[ti, j, None, :, :]
                    sq = np.sum(d**2, axis=-1)
                    dist = float(np.sqrt(np.mean(sq)))
                    centroid = float(
                        np.linalg.norm(x[ti, i].mean(0) - x[ti, j].mean(0))
                    )
                    va, wa = within[(ti, i, name)]
                    vb, wb = within[(ti, j, name)]
                    _assert_close(
                        dist**2,
                        centroid**2 + va + vb,
                        (task, methods[i], methods[j], name, "seed identity"),
                    )
                    row |= {
                        f"d_{name.lower()}": dist,
                        f"d2_{name.lower()}": dist**2,
                        f"centroid_{name.lower()}": centroid,
                        f"seed_pair_sd_{name.lower()}": float(np.std(np.sqrt(sq))),
                        f"within_variance_a_{name.lower()}": va,
                        f"within_variance_b_{name.lower()}": vb,
                        f"within_pair_rms_a_{name.lower()}": wa,
                        f"within_pair_rms_b_{name.lower()}": wb,
                    }
                    matrices[name][ti, i, j] = matrices[name][ti, j, i] = dist
                    diffs[name] = d

                dot = float(np.mean(np.sum(diffs["P"] * diffs["S"], axis=-1)))
                cross = 2 * dot
                _assert_close(
                    row["d2_f"],
                    row["d2_p"] + row["d2_s"] + cross,
                    (task, methods[i], methods[j], "pair PS"),
                )
                row["cross_ps"] = cross
                row["cos_ps"] = (
                    np.nan
                    if row["d_p"] * row["d_s"] <= self.eps
                    else dot / (row["d_p"] * row["d_s"])
                )
                pair_rows.append(row)

        return matrices, pair_rows, seed_rows

    def analyze_signatures(self):
        pairwise_dir = self._dir("pairwise")
        metadata, core_raw, core_norm = {}, {}, {}

        for component in self.CORE_COMPONENTS:
            raw, shape = self._load_component("peft_plus_ft", component)
            metadata[component] = {
                "native_shape": list(shape),
                "n_coordinates": int(np.prod(shape)),
                "core": True,
                "analysis_scale": float(1 / np.sqrt(np.prod(shape))),
            }
            core_raw[component] = raw
            core_norm[component] = {
                k: self._normalize_component(v) for k, v in raw.items()
            }

        balanced = {
            sig: self._balanced_core([core_raw[c][sig] for c in self.CORE_COMPONENTS])
            for sig in ("F_m", "F_e", "P", "S_m", "S_e")
        }
        core_norm["balanced_core"] = balanced
        metadata["balanced_core"] = {
            "core": True,
            "balanced": True,
            "components": list(self.CORE_COMPONENTS),
            "definition": "concat(delta_c/sqrt(C*d_c)); Euclidean distance equals sqrt(mean_c RMS_c^2)",
            "n_coordinates": int(
                sum(core_raw[c]["P"].shape[-1] for c in self.CORE_COMPONENTS)
            ),
        }

        peft_idx = [self.ADAPTATION_METHODS.index(m) for m in self.PEFT_METHODS]
        functional = {scope: {} for scope in self.METHOD_SCOPES}
        pair_rows, seed_rows = [], []
        decomp = {level: [] for level in self.LEVELS}
        ps = {level: [] for level in self.LEVELS}

        def process(scope, component, data):
            channels = {
                "model": (data["F_m"], data["P"], data["S_m"]),
                "observed": (data["F_e"], data["P"], data["S_e"]),
            }
            for channel, (f, p, s) in channels.items():
                matrices, rows, seeds = self._functional_distances(
                    scope, channel, component, f, p, s
                )
                pair_rows.extend(rows)
                seed_rows.extend(seeds)
                for source, matrix in matrices.items():
                    functional[scope][(channel, component, source)] = matrix

                _assert_close(f, p + s, (scope, channel, component, "profile PS"))
                effects = {
                    "F": self._effects(f),
                    "P": self._effects(p),
                    "S": self._effects(s),
                }
                for source, x in (("F", f), ("P", p), ("S", s)):
                    rows = self._decomposition_rows(
                        scope, channel, source, component, x, effects[source]
                    )
                    for level in self.LEVELS:
                        decomp[level].extend(rows[level])
                rows = self._ps_rows(
                    scope, channel, component, effects["F"], effects["P"], effects["S"]
                )
                for level in self.LEVELS:
                    ps[level].extend(rows[level])

        for component in (*self.CORE_COMPONENTS, "balanced_core"):
            process("peft_plus_ft", component, core_norm[component])
            process(
                "peft_only",
                component,
                {k: v[:, peft_idx] for k, v in core_norm[component].items()},
            )

        for component in self.ALL_COMPONENTS:
            if component in self.CORE_COMPONENTS:
                continue
            raw, shape = self._load_component("peft_only", component)
            metadata[component] = {
                "native_shape": list(shape),
                "n_coordinates": int(np.prod(shape)),
                "core": False,
                "analysis_scale": float(1 / np.sqrt(np.prod(shape))),
            }
            process(
                "peft_only",
                component,
                {k: self._normalize_component(v) for k, v in raw.items()},
            )
            del raw
            gc.collect()

        decomposition_paths, ps_paths = {}, {}
        for level in self.LEVELS:
            out = self._dir("decomposition", level)
            decomposition_paths[level] = _write_csv(
                out / "decomposition.csv", decomp[level]
            )
            ps_paths[level] = _write_csv(out / "fisher_ps.csv", ps[level])

        return functional, {
            "decomposition": decomposition_paths,
            "fisher_ps": ps_paths,
            "functional_pairwise": _write_csv(
                pairwise_dir / "functional_pairwise.csv", pair_rows
            ),
            "functional_seed_variability": _write_csv(
                pairwise_dir / "functional_seed_variability.csv", seed_rows
            ),
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # Prediction behavior + unchanged prediction cache path
    # ------------------------------------------------------------------

    def _prediction_cache_path(self, task, method, seed):
        path = self.save_dir / "prediction_cache" / task / method / f"seed-{seed}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _get_seed_logits(self, task, method, run):
        cache = self._prediction_cache_path(task, method, run["seed"])
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as data:
                return data["logits"], data["labels"]

        cfg = run["config"]
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

        logits, labels = [], []
        with torch.inference_mode():
            for x, y in self._progress(
                loader, f"Predict {task}/{method}/seed-{run['seed']}", len(loader)
            ):
                logits.append(model(x.to(self.device)).cpu())
                labels.append(y.cpu())
        logits = torch.cat(logits).numpy().astype(np.float32)
        labels = torch.cat(labels).numpy()
        np.savez(cache, logits=logits, labels=labels)
        del model, dataset, loader
        _cleanup_cuda()
        return logits, labels

    def _method_predictions(self, task, method):
        probs, labels = [], None
        for run in self._get_runs(task, method):
            logits, y = self._get_seed_logits(task, method, run)
            if labels is None:
                labels = y
            elif not np.array_equal(labels, y):
                raise RuntimeError(f"Test ordering mismatch for {task}/{method}")
            z = logits.astype(np.float64)
            z -= z.max(1, keepdims=True)
            p = np.exp(z)
            probs.append(p / p.sum(1, keepdims=True))
        probs = np.stack(probs)
        top1 = probs.argmax(-1)
        correct = top1 == labels[None]
        return {
            "probabilities": probs,
            "top1": top1,
            "correct": correct,
            "accuracy": correct.mean(1),
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
    def _error_consistency(a, b):
        observed = np.mean(a == b)
        pa, pb = a.mean(), b.mean()
        expected = pa * pb + (1 - pa) * (1 - pb)
        return (
            np.nan
            if 1 - expected <= np.finfo(float).eps
            else float((observed - expected) / (1 - expected))
        )

    def _behavior(self, a, b, pairs):
        pairs = list(pairs)
        jsd, top1, kappa, acc = [], [], [], []
        for i, j in pairs:
            jsd.append(self._jsd(a["probabilities"][i], b["probabilities"][j]))
            top1.append(np.mean(a["top1"][i] != b["top1"][j]))
            kappa.append(self._error_consistency(a["correct"][i], b["correct"][j]))
            acc.append(abs(a["accuracy"][i] - b["accuracy"][j]))
        k = np.asarray(kappa, float)
        kmean = float(np.nanmean(k)) if np.any(np.isfinite(k)) else np.nan
        return {
            "jsd": float(np.mean(jsd)),
            "top1_disagreement": float(np.mean(top1)),
            "error_consistency": kmean,
            "error_dissimilarity": 1 - kmean if np.isfinite(kmean) else np.nan,
            "accuracy_gap": float(np.mean(acc)),
            "n_seed_pairs": len(pairs),
        }

    def analyze_predictions(self):
        pairwise_dir = self._dir("pairwise")
        matrices = {
            scope: {metric: [] for metric in self.PREDICTION_METRICS}
            for scope in self.METHOD_SCOPES
        }
        pair_rows, seed_rows = [], []
        within_pairs = list(itertools.combinations(range(len(self.SEEDS)), 2))
        cross_pairs = list(itertools.product(range(len(self.SEEDS)), repeat=2))

        for task in self._progress(self.DATASETS, "Prediction behavior"):
            pred = {m: self._method_predictions(task, m) for m in self.METHODS}
            labels = pred[self.METHODS[0]]["labels"]
            if any(
                not np.array_equal(labels, pred[m]["labels"]) for m in self.METHODS[1:]
            ):
                raise RuntimeError(f"Test ordering mismatch across methods for {task}")

            for method in self.METHODS:
                v = self._behavior(pred[method], pred[method], within_pairs)
                seed_rows.append(
                    {
                        "task": task,
                        "task_group": self._task_group[task],
                        "method": method,
                        "mean_accuracy": float(np.mean(pred[method]["accuracy"])),
                        "within_jsd": v["jsd"],
                        "within_top1_disagreement": v["top1_disagreement"],
                        "within_error_consistency": v["error_consistency"],
                        "within_error_dissimilarity": v["error_dissimilarity"],
                        "n_seed_pairs": len(within_pairs),
                    }
                )

            for scope, methods in self.METHOD_SCOPES.items():
                mats = {
                    metric: np.zeros((len(methods), len(methods)))
                    for metric in self.PREDICTION_METRICS
                }
                for i, j in itertools.combinations(range(len(methods)), 2):
                    v = self._behavior(pred[methods[i]], pred[methods[j]], cross_pairs)
                    pair_rows.append(
                        {
                            "scope": scope,
                            "task": task,
                            "task_group": self._task_group[task],
                            "method_a": methods[i],
                            "method_b": methods[j],
                            **v,
                        }
                    )
                    for metric in self.PREDICTION_METRICS:
                        mats[metric][i, j] = mats[metric][j, i] = v[metric]
                for metric in self.PREDICTION_METRICS:
                    matrices[scope][metric].append(mats[metric])
            del pred
            gc.collect()

        matrices = {
            scope: {metric: np.stack(values) for metric, values in bundle.items()}
            for scope, bundle in matrices.items()
        }
        return matrices, {
            "pairwise": _write_csv(pairwise_dir / "prediction_pairwise.csv", pair_rows),
            "seed_variability": _write_csv(
                pairwise_dir / "prediction_seed_variability.csv", seed_rows
            ),
        }

    # ------------------------------------------------------------------
    # CKA baseline + unchanged CKA cache path
    # ------------------------------------------------------------------

    def _cka_cache_path(self, task):
        path = self.save_dir / "cka_cache" / f"{task}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _get_task_cka(self, task):
        cache = self._cka_cache_path(task)
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as data:
                methods = tuple(data["methods"].tolist())
                seeds = tuple(int(x) for x in data["seeds"])
                distances = data["layerwise_distance"]
            if methods == self.ADAPTATION_METHODS and seeds == self.SEEDS:
                return distances

        runs = {m: self._get_runs(task, m) for m in self.ADAPTATION_METHODS}
        dataset = datasets[task](split="train+val")
        features, labels_ref = [], None
        for method in self.ADAPTATION_METHODS:
            for run in runs[method]:
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
                    for x, y in self._progress(
                        loader, f"CKA {task}/{method}/seed-{run['seed']}", len(loader)
                    ):
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

        n_m, n_s = len(self.ADAPTATION_METHODS), len(self.SEEDS)
        distances = (
            (1 - sim).reshape(n_m, n_s, n_m, n_s, n_layers).transpose(0, 2, 1, 3, 4)
        )
        np.savez_compressed(
            cache,
            methods=np.array(self.ADAPTATION_METHODS),
            seeds=np.array(self.SEEDS),
            layerwise_distance=distances,
        )
        del features, dataset
        _cleanup_cuda()
        return distances

    def analyze_cka(self):
        pairwise_dir = self._dir("pairwise")
        matrices = {scope: [] for scope in self.METHOD_SCOPES}
        pair_rows, seed_rows = [], []
        indices = {
            scope: [self.ADAPTATION_METHODS.index(m) for m in methods]
            for scope, methods in self.METHOD_SCOPES.items()
        }
        within_pairs = list(itertools.combinations(range(len(self.SEEDS)), 2))

        for task in self._progress(self.DATASETS, "CKA baseline"):
            final = self._get_task_cka(task)[..., -1]  # [method, method, seed, seed]

            for mi, method in enumerate(self.ADAPTATION_METHODS):
                values = np.asarray(
                    [final[mi, mi, i, j] for i, j in within_pairs], float
                )
                d = float(values.mean())
                seed_rows.append(
                    {
                        "task": task,
                        "task_group": self._task_group[task],
                        "method": method,
                        "within_cka": 1 - d,
                        "within_cka_distance": d,
                        "seed_pair_sd": float(values.std()),
                        "n_seed_pairs": len(values),
                    }
                )

            for scope, idx in indices.items():
                methods = self._scope_methods(scope)
                matrix = np.zeros((len(methods), len(methods)))
                for i, j in itertools.combinations(range(len(methods)), 2):
                    values = final[idx[i], idx[j]].reshape(-1)
                    d = float(values.mean())
                    matrix[i, j] = matrix[j, i] = d
                    pair_rows.append(
                        {
                            "scope": scope,
                            "task": task,
                            "task_group": self._task_group[task],
                            "method_a": methods[i],
                            "method_b": methods[j],
                            "cka": 1 - d,
                            "cka_distance": d,
                            "seed_pair_sd": float(values.std()),
                            "n_seed_pairs": len(values),
                        }
                    )
                matrices[scope].append(matrix)

        matrices = {scope: np.stack(values) for scope, values in matrices.items()}
        return matrices, {
            "pairwise": _write_csv(pairwise_dir / "cka_pairwise.csv", pair_rows),
            "seed_variability": _write_csv(
                pairwise_dir / "cka_seed_variability.csv", seed_rows
            ),
        }

    # ------------------------------------------------------------------
    # Behavioral grounding / RSA
    # ------------------------------------------------------------------

    def _rsa(self, source, target):
        x, y = _upper(np.asarray(source, float)), _upper(np.asarray(target, float))
        observed = _spearman(x, y)
        if not np.isfinite(observed):
            return np.nan, np.nan, np.empty(0)

        n = source.shape[0]
        perms = self._perms[n]
        iu, ju = np.triu_indices(n, 1)
        xp = source[perms[:, iu], perms[:, ju]]
        null = (
            _row_corr(rankdata(xp, axis=1), rankdata(y))
            if np.all(np.isfinite(xp)) and np.all(np.isfinite(y))
            else np.asarray([_spearman(row, y) for row in xp])
        )
        null = null[np.isfinite(null)]
        p = (
            np.nan
            if not null.size
            else float(np.mean(np.abs(null) >= abs(observed) - 1e-15))
        )
        return observed, p, null

    def _bootstrap_mean_ci(self, values, groups, stratified=True):
        values, groups = np.asarray(values, float), np.asarray(groups)
        keep = np.isfinite(values)
        values, groups = values[keep], groups[keep]
        if not len(values) or self.n_bootstrap <= 0:
            return np.nan, np.nan

        rng = np.random.default_rng(self.random_seed)
        samples = np.empty(self.n_bootstrap)
        unique = np.unique(groups) if stratified else ("all",)
        for b in range(self.n_bootstrap):
            chosen = []
            if stratified:
                for g in unique:
                    x = values[groups == g]
                    chosen.extend(rng.choice(x, len(x), replace=True))
            else:
                chosen = rng.choice(values, len(values), replace=True)
            samples[b] = np.mean(chosen)
        return tuple(np.quantile(samples, [0.025, 0.975]))

    def _alignment_summary(self, records):
        keys = (
            "analysis_level",
            "scope",
            "source",
            "channel",
            "component",
            "behavior_metric",
        )
        grouped = defaultdict(list)
        for rec in records:
            grouped[tuple(rec["row"][k] for k in keys)].append(rec)

        rows = []
        rng = np.random.default_rng(self.random_seed)
        for key, items in grouped.items():
            base = dict(zip(keys, key))
            for aggregation in ("all", *self.TASK_GROUPS):
                use = [
                    x
                    for x in items
                    if (aggregation == "all" or x["row"]["task_group"] == aggregation)
                    and np.isfinite(x["row"]["rho"])
                    and len(x["null"])
                ]
                if not use:
                    continue
                rho = np.asarray([x["row"]["rho"] for x in use])
                groups = np.asarray([x["row"]["task_group"] for x in use])
                lo, hi = self._bootstrap_mean_ci(
                    rho, groups, stratified=aggregation == "all"
                )

                null_mean = np.zeros(self.n_global_permutations)
                for x in use:
                    null = x["null"]
                    null_mean += null[
                        rng.integers(0, len(null), size=self.n_global_permutations)
                    ]
                null_mean /= len(use)
                obs = float(rho.mean())
                p = float(
                    (1 + np.sum(np.abs(null_mean) >= abs(obs)))
                    / (self.n_global_permutations + 1)
                )
                rows.append(
                    {
                        **base,
                        "aggregation": aggregation,
                        "mean_rho": obs,
                        "median_rho": float(np.median(rho)),
                        "sd_rho": float(np.std(rho)),
                        "ci_low": lo,
                        "ci_high": hi,
                        "p_global_permutation": p,
                        "n_tasks": len(rho),
                    }
                )

        q = _bh([r["p_global_permutation"] for r in rows])
        for row, value in zip(rows, q):
            row["q_global_bh"] = value
        return rows

    def _method_pair_summary(self, functional, prediction, cka):
        rows = []
        groups = np.asarray([self._task_group[t] for t in self.DATASETS])
        for scope, methods in self.METHOD_SCOPES.items():
            for channel in ("model", "observed"):
                f = functional[scope][(channel, "balanced_core", "F")]
                p = functional[scope][(channel, "balanced_core", "P")]
                s = functional[scope][(channel, "balanced_core", "S")]
                for i, j in itertools.combinations(range(len(methods)), 2):
                    values = {
                        "d_f": f[:, i, j],
                        "d_p": p[:, i, j],
                        "d_s": s[:, i, j],
                        "jsd": prediction[scope]["jsd"][:, i, j],
                        "top1_disagreement": prediction[scope]["top1_disagreement"][
                            :, i, j
                        ],
                        "error_dissimilarity": prediction[scope]["error_dissimilarity"][
                            :, i, j
                        ],
                        "accuracy_gap": prediction[scope]["accuracy_gap"][:, i, j],
                        "cka_distance": cka[scope][:, i, j],
                    }
                    row = {
                        "scope": scope,
                        "channel": channel,
                        "method_a": methods[i],
                        "method_b": methods[j],
                    }
                    for name, x in values.items():
                        lo, hi = self._bootstrap_mean_ci(x, groups, stratified=True)
                        row[name] = float(np.nanmean(x))
                        row[f"{name}_ci_low"] = lo
                        row[f"{name}_ci_high"] = hi
                    rows.append(row)
        return rows

    def analyze_alignment(self, functional, prediction, cka):
        out = self._dir("alignment")
        records, configs = [], []

        for channel in ("model", "observed"):
            for source in ("F", "P", "S"):
                configs.append(
                    (
                        "core",
                        "peft_plus_ft",
                        source,
                        channel,
                        "balanced_core",
                        self.BEHAVIOR_METRICS,
                    )
                )
        configs.append(
            (
                "core",
                "peft_plus_ft",
                "CKA",
                "representation",
                "final_cls",
                self.BEHAVIOR_METRICS,
            )
        )
        for component in self.CORE_COMPONENTS:
            for source in ("F", "P", "S"):
                configs.append(
                    (
                        "core_localization",
                        "peft_plus_ft",
                        source,
                        "model",
                        component,
                        ("jsd",),
                    )
                )
        for component in self.ALL_COMPONENTS:
            for source in ("F", "P", "S"):
                configs.append(
                    (
                        "peft_localization",
                        "peft_only",
                        source,
                        "model",
                        component,
                        ("jsd",),
                    )
                )

        for level, scope, source, channel, component, behaviors in configs:
            matrices = (
                cka[scope]
                if source == "CKA"
                else functional[scope][(channel, component, source)]
            )
            for ti, task in enumerate(self.DATASETS):
                for behavior in behaviors:
                    rho, p, null = self._rsa(
                        matrices[ti], prediction[scope][behavior][ti]
                    )
                    row = {
                        "analysis_level": level,
                        "scope": scope,
                        "task": task,
                        "task_group": self._task_group[task],
                        "source": source,
                        "channel": channel,
                        "component": component,
                        "behavior_metric": behavior,
                        "rho": rho,
                        "p_permutation": p,
                        "n_method_pairs": len(_upper(matrices[ti])),
                    }
                    records.append({"row": row, "null": null})

        summary = self._alignment_summary(records)
        return {
            "by_task": _write_csv(
                out / "alignment_by_task.csv", [x["row"] for x in records]
            ),
            "summary": _write_csv(out / "alignment_summary.csv", summary),
            "method_pair_summary": _write_csv(
                out / "method_pair_summary.csv",
                self._method_pair_summary(functional, prediction, cka),
            ),
        }, summary

    # ------------------------------------------------------------------
    # Manifests / orchestration
    # ------------------------------------------------------------------

    def _write_analysis_manifest(self, component_metadata):
        manifest = {
            "backbone": self.backbone_name,
            "datasets": list(self.DATASETS),
            "task_groups": {k: list(v) for k, v in self.TASK_GROUPS.items()},
            "reference_method": self.REFERENCE_METHOD,
            "scopes": {k: list(v) for k, v in self.METHOD_SCOPES.items()},
            "seeds": list(self.SEEDS),
            "epsilon": self.eps,
            "functional_object": "LP-referenced log delta only",
            "lp_reference": "mean log profile over the three LP checkpoints",
            "channels": {
                "model": "model Fisher (F_m, P, S_m)",
                "observed": "observed-label (F_e, P, S_e)",
            },
            "sensitivity_definition": "delta_S = delta_F - delta_P",
            "distance": "cross-seed RMS: sqrt(mean over 9 seed pairs of squared normalized Euclidean distance)",
            "components": component_metadata,
            "balanced_core": list(self.CORE_COMPONENTS),
            "balanced_core_weighting": "each component contributes equally; coordinates within a component contribute equally",
            "behavior_metrics": list(self.BEHAVIOR_METRICS),
            "cka_baseline": "final-layer CLS linear CKA distance (1-CKA)",
            "primary_result_dirs": ["alignment", "decomposition", "figures"],
            "supporting_result_dir": "pairwise",
            "prediction_cache_dir": str(self.save_dir / "prediction_cache"),
            "cka_cache_dir": str(self.save_dir / "cka_cache"),
            "n_bootstrap": self.n_bootstrap,
            "n_global_permutations": self.n_global_permutations,
            "random_seed": self.random_seed,
        }
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / "analysis_manifest.json"
        path.write_text(json.dumps(manifest, indent=2))
        return path

    def _write_cache_manifest(self):
        runs = []
        for task in self.DATASETS:
            for method in self.METHODS:
                for run in self._get_runs(task, method):
                    model = run["path"] / "model.pt"
                    sig = run["path"] / self.SIGNATURE_FILENAME
                    pred = self._prediction_cache_path(task, method, run["seed"])
                    runs.append(
                        {
                            "task": task,
                            "method": method,
                            "seed": run["seed"],
                            "run_dir": str(run["path"]),
                            "model_path": str(model),
                            "model_size": model.stat().st_size,
                            "model_mtime_ns": model.stat().st_mtime_ns,
                            "signature_path": str(sig),
                            "signature_exists": sig.is_file(),
                            "prediction_cache": str(pred),
                            "prediction_cache_exists": pred.is_file(),
                        }
                    )
        cache = {
            "prediction_cache_dir": str(self.save_dir / "prediction_cache"),
            "cka_cache_dir": str(self.save_dir / "cka_cache"),
            "runs": runs,
            "cka": [
                {
                    "task": task,
                    "path": str(self._cka_cache_path(task)),
                    "exists": self._cka_cache_path(task).is_file(),
                }
                for task in self.DATASETS
            ],
        }
        path = self.save_dir / "cache_manifest.json"
        path.write_text(json.dumps(cache, indent=2))
        return path

    def analyze_all(self):
        functional, signature_outputs = self.analyze_signatures()
        prediction, prediction_outputs = self.analyze_predictions()
        cka, cka_outputs = self.analyze_cka()
        alignment_outputs, alignment_summary = self.analyze_alignment(
            functional, prediction, cka
        )
        figures = make_figures(
            self, signature_outputs, alignment_summary, functional, prediction, cka
        )
        manifest = self._write_analysis_manifest(signature_outputs["metadata"])
        cache_manifest = self._write_cache_manifest()
        return {
            "manifest": manifest,
            "cache_manifest": cache_manifest,
            "signatures": signature_outputs,
            "predictions": prediction_outputs,
            "cka": cka_outputs,
            "alignment": alignment_outputs,
            "figures": figures,
        }


# -----------------------------------------------------------------------------
# Paper figures
# -----------------------------------------------------------------------------


COLORS = {
    "blue": "#4477AA",
    "cyan": "#66CCEE",
    "green": "#228833",
    "yellow": "#CCBB44",
    "red": "#EE6677",
    "purple": "#AA3377",
    "gray": "#BBBBBB",
}
DECOMP_COLORS = {
    "group": COLORS["blue"],
    "dataset": COLORS["cyan"],
    "method": COLORS["yellow"],
    "interaction": COLORS["purple"],
    "seed": COLORS["gray"],
    "method_general": COLORS["blue"],
    "task_group": COLORS["cyan"],
    "dataset_specific": COLORS["yellow"],
    "task_shared": COLORS["blue"],
    "method_conditioned": COLORS["yellow"],
}
SOURCE_COLORS = {
    "F": COLORS["blue"],
    "P": COLORS["green"],
    "S": COLORS["red"],
    "CKA": COLORS["purple"],
}
PRETTY_METHOD = {
    "full_ft": "Full FT",
    "vpt_shallow": "VPT-Shallow",
    "vpt_deep": "VPT-Deep",
    "lora": "LoRA",
    "ssf": "SSF",
    "adaptformer": "AdaptFormer",
}
PRETTY_EFFECT = {
    "group": "Task group",
    "dataset": "Dataset",
    "method": "Method",
    "interaction": "Interaction",
    "seed": "Seed",
    "method_general": "Method-general",
    "task_group": "Task group",
    "dataset_specific": "Dataset-specific",
    "task_shared": "Task-shared",
    "method_conditioned": "Method-conditioned",
}
PRETTY_BEHAVIOR = {
    "jsd": "Jensen–Shannon divergence",
    "top1_disagreement": "Top-1 disagreement",
    "error_dissimilarity": "Error dissimilarity",
}


def _style_ax(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B8B8B8")
    ax.spines["bottom"].set_color("#B8B8B8")
    ax.tick_params(length=3, color="#777777")
    if grid_axis is not None:
        ax.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.65, alpha=0.55, zorder=0)
        ax.set_axisbelow(True)


def _heatmap(fig, ax, matrix, title, methods, ylabels=None, aspect="equal"):
    im = ax.imshow(matrix, aspect=aspect, cmap="viridis", interpolation="nearest")
    labels = [PRETTY_METHOD[m] for m in methods]
    ax.set_title(title, pad=7)
    ax.set_xticks(range(len(methods)), labels, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(ylabels or methods)), ylabels or labels, fontsize=7.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.outline.set_visible(False)


def _save(fig, out, name):
    path = out / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def make_figures(a, signature_outputs, alignment_summary, functional, prediction, cka):
    out = a._dir("figures")
    paths = {}
    order = ("F", "P", "S")
    short_tasks = [t.removeprefix("vtab_").replace("_", " ") for t in a.DATASETS]

    # Global decomposition
    rows = [
        r
        for r in _read_csv(signature_outputs["decomposition"]["global"])
        if r["scope"] == "peft_plus_ft"
        and r["channel"] == "model"
        and r["component"] == "balanced_core"
    ]
    rows.sort(key=lambda r: order.index(r["signature"]))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bottom = np.zeros(len(order))
    for key in ("group", "dataset", "method", "interaction", "seed"):
        values = np.array([float(r[f"r_{key}"]) for r in rows])
        ax.bar(
            order,
            values,
            bottom=bottom,
            width=0.62,
            color=DECOMP_COLORS[key],
            edgecolor="white",
            linewidth=0.8,
            label=PRETTY_EFFECT[key],
            zorder=3,
        )
        bottom += values
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of centered reorganization energy")
    ax.set_title("Global functional reorganization", pad=10)
    _style_ax(ax)
    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    fig.tight_layout()
    paths["decomposition_global"] = _save(fig, out, "decomposition_global.pdf")

    # Per-method decomposition
    rows = [
        r
        for r in _read_csv(signature_outputs["decomposition"]["method"])
        if r["scope"] == "peft_plus_ft"
        and r["channel"] == "model"
        and r["component"] == "balanced_core"
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2), sharey=True)
    for ax, sig in zip(axes, order):
        by = {r["method"]: r for r in rows if r["signature"] == sig}
        y, left = np.arange(len(a.ADAPTATION_METHODS)), np.zeros(
            len(a.ADAPTATION_METHODS)
        )
        for key in ("method_general", "task_group", "dataset_specific", "seed"):
            values = np.array([float(by[m][f"r_{key}"]) for m in a.ADAPTATION_METHODS])
            ax.barh(
                y,
                values,
                left=left,
                height=0.68,
                color=DECOMP_COLORS[key],
                edgecolor="white",
                linewidth=0.6,
                label=PRETTY_EFFECT[key],
                zorder=3,
            )
            left += values
        ax.set(title=sig, xlim=(0, 1), xlabel="Fraction of energy")
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(y, [PRETTY_METHOD[m] for m in a.ADAPTATION_METHODS])
        ax.invert_yaxis()
        _style_ax(ax, "x")
    axes[0].legend(
        frameon=False, fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(0, -0.14)
    )
    fig.tight_layout()
    paths["decomposition_by_method"] = _save(fig, out, "decomposition_by_method.pdf")

    # Per-dataset decomposition
    rows = [
        r
        for r in _read_csv(signature_outputs["decomposition"]["dataset"])
        if r["scope"] == "peft_plus_ft"
        and r["channel"] == "model"
        and r["component"] == "balanced_core"
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 8.2), sharey=True)
    for ax, sig in zip(axes, order):
        by = {r["task"]: r for r in rows if r["signature"] == sig}
        y, left = np.arange(len(a.DATASETS)), np.zeros(len(a.DATASETS))
        for key in ("task_shared", "method_conditioned", "seed"):
            values = np.array([float(by[t][f"r_{key}"]) for t in a.DATASETS])
            ax.barh(
                y,
                values,
                left=left,
                height=0.70,
                color=DECOMP_COLORS[key],
                edgecolor="white",
                linewidth=0.55,
                label=PRETTY_EFFECT[key],
                zorder=3,
            )
            left += values
        for boundary in (
            len(a.NATURAL) - 0.5,
            len(a.NATURAL) + len(a.SPECIALIZED) - 0.5,
        ):
            ax.axhline(
                boundary, color="#888888", linewidth=0.7, linestyle="--", alpha=0.55
            )
        ax.set(title=sig, xlim=(0, 1), xlabel="Fraction of energy")
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(y, short_tasks, fontsize=7.5)
        ax.invert_yaxis()
        _style_ax(ax, "x")
    axes[0].legend(
        frameon=False, fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.08)
    )
    fig.tight_layout()
    paths["decomposition_by_dataset"] = _save(fig, out, "decomposition_by_dataset.pdf")

    # Dataset x method attribution
    rows = [
        r
        for r in _read_csv(signature_outputs["decomposition"]["dataset_method"])
        if r["scope"] == "peft_plus_ft"
        and r["channel"] == "model"
        and r["signature"] == "F"
        and r["component"] == "balanced_core"
    ]
    lookup = {(r["task"], r["method"]): r for r in rows}
    panels = (
        ("r_group_signed", "Task group"),
        ("r_dataset_signed", "Dataset"),
        ("r_method_signed", "Method"),
        ("r_interaction_signed", "Interaction"),
        ("r_seed", "Seed"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 10.2))
    for ax, (key, title) in zip(axes.ravel(), panels):
        matrix = np.array(
            [
                [float(lookup[t, m][key]) for m in a.ADAPTATION_METHODS]
                for t in a.DATASETS
            ]
        )
        _heatmap(
            fig, ax, matrix, title, a.ADAPTATION_METHODS, short_tasks, aspect="auto"
        )
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    paths["decomposition_dataset_method"] = _save(
        fig, out, "decomposition_dataset_method.pdf"
    )

    # Global F=P+S decomposition
    rows = [
        r
        for r in _read_csv(signature_outputs["fisher_ps"]["global"])
        if r["scope"] == "peft_plus_ft"
        and r["channel"] == "model"
        and r["component"] == "balanced_core"
        and r["effect"] not in {"task", "total"}
    ]
    effect_order = ("group", "dataset", "method", "interaction", "seed")
    rows.sort(key=lambda r: effect_order.index(r["effect"]))
    x, width = np.arange(len(rows)), 0.24
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for offset, key, label, color in (
        (-width, "p_over_f", "P", COLORS["blue"]),
        (0, "s_over_f", "S", COLORS["green"]),
        (width, "cross_over_f", "P × S", COLORS["purple"]),
    ):
        ax.bar(
            x + offset,
            [np.nan if r[key] == "NA" else float(r[key]) for r in rows],
            width,
            color=color,
            edgecolor="white",
            linewidth=0.65,
            label=label,
            zorder=3,
        )
    ax.axhline(0, color="#555555", linewidth=0.9)
    ax.set_xticks(x, [PRETTY_EFFECT[r["effect"]] for r in rows])
    ax.set_ylabel("Term / Fisher energy")
    ax.set_title("Participation–sensitivity decomposition", pad=10)
    _style_ax(ax)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    paths["fisher_ps"] = _save(fig, out, "fisher_ps_decomposition.pdf")

    # Behavioral grounding
    core = [
        r
        for r in alignment_summary
        if r["analysis_level"] == "core"
        and r["scope"] == "peft_plus_ft"
        and r["aggregation"] == "all"
        and (
            (
                r["channel"] == "model"
                and r["component"] == "balanced_core"
                and r["source"] in {"F", "P", "S"}
            )
            or r["source"] == "CKA"
        )
    ]
    source_order = ("F", "P", "S", "CKA")
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.0), sharey=True)
    for ax, behavior in zip(axes, a.BEHAVIOR_METRICS):
        by = {r["source"]: r for r in core if r["behavior_metric"] == behavior}
        y = np.array([float(by[s]["mean_rho"]) for s in source_order])
        lo = np.array([float(by[s]["ci_low"]) for s in source_order])
        hi = np.array([float(by[s]["ci_high"]) for s in source_order])
        for i, source in enumerate(source_order):
            ax.errorbar(
                i,
                y[i],
                yerr=np.array([[max(0.0, y[i] - lo[i])], [max(0.0, hi[i] - y[i])]]),
                fmt="o",
                markersize=7,
                color=SOURCE_COLORS[source],
                ecolor=SOURCE_COLORS[source],
                markeredgecolor="white",
                markeredgewidth=0.8,
                elinewidth=1.4,
                capsize=4,
                capthick=1.2,
                zorder=3,
            )
        ax.set_xticks(range(len(source_order)), source_order)
        ax.set_title(PRETTY_BEHAVIOR[behavior], pad=8, fontsize=10)
        ax.set_ylim(-1, 1)
        ax.set_yticks(np.linspace(-1, 1, 9))
        _style_ax(ax)
    axes[0].set_ylabel("Mean task-level Spearman ρ")
    fig.tight_layout()
    paths["alignment"] = _save(fig, out, "behavior_alignment.pdf")

    # Method geometry: model / representation distances vs prediction distances.
    methods = a.ADAPTATION_METHODS
    model_mats = (
        ("$F_m$", functional["peft_plus_ft"][("model", "balanced_core", "F")].mean(0)),
        ("$P$", functional["peft_plus_ft"][("model", "balanced_core", "P")].mean(0)),
        ("$S_m$", functional["peft_plus_ft"][("model", "balanced_core", "S")].mean(0)),
        (
            "$F_e$",
            functional["peft_plus_ft"][("observed", "balanced_core", "F")].mean(0),
        ),
        (
            "$S_e$",
            functional["peft_plus_ft"][("observed", "balanced_core", "S")].mean(0),
        ),
        ("CKA", cka["peft_plus_ft"].mean(0)),
    )
    prediction_mats = (
        ("JSD", prediction["peft_plus_ft"]["jsd"].mean(0)),
        ("Top-1 disagreement", prediction["peft_plus_ft"]["top1_disagreement"].mean(0)),
        (
            "Error dissimilarity",
            prediction["peft_plus_ft"]["error_dissimilarity"].mean(0),
        ),
    )
    fig = plt.figure(figsize=(12.2, 11.0))
    gs = fig.add_gridspec(3, 3, hspace=0.62, wspace=0.38)
    model_axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    pred_axes = [fig.add_subplot(gs[2, c]) for c in range(3)]
    for ax, (title, matrix) in zip(model_axes, model_mats):
        _heatmap(fig, ax, matrix, title, methods)
    for ax, (title, matrix) in zip(pred_axes, prediction_mats):
        _heatmap(fig, ax, matrix, title, methods)
    fig.text(
        0.5,
        0.965,
        "Model / representation distance",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.335,
        "Prediction distance",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    paths["geometry"] = _save(fig, out, "method_geometry.pdf")

    return paths
