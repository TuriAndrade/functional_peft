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
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from tqdm.auto import tqdm

from datasets import datasets
from models import models
from utils import seed_worker


class FunctionalReorganizationAnalyzer:
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
    OFFICIAL_TASK_GROUPS = {
        "natural": NATURAL,
        "specialized": SPECIALIZED,
        "structured": STRUCTURED,
    }

    REFERENCE_METHOD = "linear_probe"
    PEFT_METHODS = ("vpt_shallow", "vpt_deep", "lora", "ssf", "adaptformer")
    ADAPTATION_METHODS = ("full_ft", *PEFT_METHODS)
    METHODS = (REFERENCE_METHOD, *ADAPTATION_METHODS)
    METHOD_SCOPES = {"peft_only": PEFT_METHODS, "peft_plus_ft": ADAPTATION_METHODS}

    REPRESENTATIONS = ("raw", "delta")
    SOURCE_SIGNATURES = ("F_m", "F_e", "P")
    SIGNATURES = ("F_m", "F_e", "P", "S_m", "S_e")
    BEHAVIOR_METRICS = ("jsd", "top1_disagreement", "error_dissimilarity")
    CKA_SUMMARIES = ("mean_layers", "final_layer")

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
    SCOPE_COMPONENTS = {
        "peft_only": ALL_COMPONENTS,
        "peft_plus_ft": CORE_COMPONENTS,
    }

    EVAL_EXPERIMENT_NAME = "top_1_acc_eval"
    SIGNATURE_FILENAME = "functional_signatures.npz"

    def __init__(self, backbone_name="vit_b16", eps=1e-12, verbose=True):
        self.backbone_name = str(backbone_name)
        self.experiments_dir = Path(os.environ.get("EXPERIMENTS_DIR", "./experiments"))
        self.save_dir = (
            self.experiments_dir / "functional_reorganization" / self.backbone_name
        )

        self.eps = float(eps)
        self.verbose = bool(verbose)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._component_cache = {}
        self._component_shapes = {}
        self._effects_cache = {}
        self._signature_distance_cache = {}
        self._runs_cache = {}
        self._permutation_cache = {}
        self._behavior_cache = None

        self._dataset_index = {name: i for i, name in enumerate(self.DATASETS)}
        self._task_group = {
            task: group
            for group, tasks in self.OFFICIAL_TASK_GROUPS.items()
            for task in tasks
        }
        self._official_group_indices = {
            group: np.array([self._dataset_index[d] for d in tasks], dtype=int)
            for group, tasks in self.OFFICIAL_TASK_GROUPS.items()
        }


    def _log(self, message):
        if self.verbose:
            print(message, flush=True)

    def _progress(self, iterable, desc, total=None, leave=False):
        return tqdm(
            iterable,
            desc=desc,
            total=total,
            disable=not self.verbose,
            leave=leave,
            dynamic_ncols=True,
        )

    def _method_dir(self, dataset_name, method_name):
        return (
            self.experiments_dir
            / self.EVAL_EXPERIMENT_NAME
            / self.backbone_name
            / dataset_name
            / method_name
        )

    def _signature_path(self, dataset_name, method_name):
        return self._method_dir(dataset_name, method_name) / self.SIGNATURE_FILENAME

    def _analysis_dir(self, *parts):
        path = self.save_dir.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _scope_methods(self, scope):
        try:
            return self.METHOD_SCOPES[scope]
        except KeyError as exc:
            raise ValueError(f"Unknown scope {scope!r}") from exc

    def _scope_components(self, scope):
        try:
            return self.SCOPE_COMPONENTS[scope]
        except KeyError as exc:
            raise ValueError(f"Unknown scope {scope!r}") from exc

    @staticmethod
    def _safe_ratio(a, b):
        b = float(b)
        return np.nan if abs(b) <= np.finfo(float).eps else float(a) / b

    @staticmethod
    def _assert_close(a, b, label):
        if not np.allclose(a, b, rtol=1e-9, atol=1e-10, equal_nan=True):
            error = float(np.nanmax(np.abs(np.asarray(a) - np.asarray(b))))
            raise RuntimeError(f"{label} failed; max error={error}")

    @staticmethod
    def _write_csv(path, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not records:
            raise ValueError(f"No records for {path}")
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=tuple(records[0]))
            writer.writeheader()
            for row in records:
                writer.writerow(
                    {
                        k: (
                            "NA"
                            if isinstance(v, (float, np.floating))
                            and not np.isfinite(v)
                            else v
                        )
                        for k, v in row.items()
                    }
                )
        return path

    @staticmethod
    def _upper(matrix):
        return matrix[np.triu_indices(matrix.shape[0], 1)]

    @staticmethod
    def _finite_arrays(*arrays):
        arrays = [np.asarray(x, dtype=float).reshape(-1) for x in arrays]
        valid = np.logical_and.reduce([np.isfinite(x) for x in arrays])
        return [x[valid] for x in arrays]

    @classmethod
    def _spearman(cls, x, y):
        x, y = cls._finite_arrays(x, y)
        if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
            return np.nan
        return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])

    @classmethod
    def _partial_spearman(cls, x, y, z):
        x, y, z = cls._finite_arrays(x, y, z)
        if x.size < 3:
            return np.nan
        rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
        design = np.column_stack((np.ones(rz.size), rz))
        ex = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
        ey = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
        if np.linalg.norm(ex) <= 1e-12 or np.linalg.norm(ey) <= 1e-12:
            return np.nan
        return float(np.corrcoef(ex, ey)[0, 1])

    @staticmethod
    def _mean_correlation(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else np.nan

    def _permutations(self, n):
        if n not in self._permutation_cache:
            self._permutation_cache[n] = np.asarray(
                list(itertools.permutations(range(n))), dtype=int
            )
        return self._permutation_cache[n]

    @staticmethod
    def _row_correlations(x, y):
        x = x - x.mean(axis=1, keepdims=True)
        y = y - y.mean()
        d = np.linalg.norm(x, axis=1) * np.linalg.norm(y)
        return np.divide(x @ y, d, out=np.full(x.shape[0], np.nan), where=d > 1e-12)

    def _rsa_permutation_p(self, functional, predictive, accuracy=None):
        functional = np.asarray(functional, dtype=float)
        predictive = np.asarray(predictive, dtype=float)
        accuracy = None if accuracy is None else np.asarray(accuracy, dtype=float)
        observed = (
            self._spearman(self._upper(functional), self._upper(predictive))
            if accuracy is None
            else self._partial_spearman(
                self._upper(functional), self._upper(predictive), self._upper(accuracy)
            )
        )
        if not np.isfinite(observed):
            return np.nan

        n = functional.shape[0]
        perms = self._permutations(n)
        iu, ju = np.triu_indices(n, 1)
        x = functional[perms[:, iu], perms[:, ju]]
        y = predictive[iu, ju]
        z = None if accuracy is None else accuracy[iu, ju]
        finite = np.all(np.isfinite(x)) and np.all(np.isfinite(y))
        finite &= z is None or np.all(np.isfinite(z))

        if finite:
            rx, ry = rankdata(x, axis=1), rankdata(y)
            if z is None:
                values = self._row_correlations(rx, ry)
            else:
                rz = rankdata(z)
                design = np.column_stack((np.ones(rz.size), rz))
                residualizer = np.eye(rz.size) - design @ np.linalg.pinv(design)
                values = self._row_correlations(rx @ residualizer, ry @ residualizer)
        else:
            values = np.array(
                [
                    (
                        self._spearman(row, y)
                        if z is None
                        else self._partial_spearman(row, y, z)
                    )
                    for row in x
                ]
            )

        values = values[np.isfinite(values)]
        return (
            np.nan
            if not values.size
            else float(np.mean(np.abs(values) >= abs(observed) - 1e-15))
        )

    @staticmethod
    def _pairwise_rms_distance(x):
        x = np.asarray(x, dtype=float)
        return np.sqrt(np.mean((x[:, None] - x[None, :]) ** 2, axis=-1))

    @staticmethod
    def _pairwise_angular_distance(x):
        x = np.asarray(x, dtype=float)
        norms = np.linalg.norm(x, axis=1)
        d = norms[:, None] * norms[None]
        cosine = np.divide(
            x @ x.T,
            d,
            out=np.full((len(x), len(x)), np.nan),
            where=d > np.finfo(float).eps,
        )
        out = np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi
        np.fill_diagonal(out, 0.0)
        return out

    @staticmethod
    def _pairwise_tv_distance(x):
        x = np.asarray(x, dtype=float)
        return 0.5 * np.sum(np.abs(x[:, None] - x[None, :]), axis=-1)

    @staticmethod
    def _mean_matrices(matrices):
        x = np.asarray(matrices, dtype=float)
        valid = np.isfinite(x)
        count = valid.sum(axis=0)
        total = np.where(valid, x, 0.0).sum(axis=0)
        return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)

    def _load_component(self, scope, component):
        cache_key = (scope, component)
        if cache_key in self._component_cache:
            return self._component_cache[cache_key]

        required_methods = (self.REFERENCE_METHOD, *self._scope_methods(scope))
        loaded = {
            signature: {method: [] for method in required_methods}
            for signature in self.SOURCE_SIGNATURES
        }
        expected_shape = self._component_shapes.get(component)
        for method in self._progress(
            required_methods, f"Load {scope}/{component}", len(required_methods)
        ):
            for task in self.DATASETS:
                path = self._signature_path(task, method)
                if not path.is_file():
                    raise FileNotFoundError(path)
                with np.load(path, allow_pickle=False) as data:
                    for signature in self.SOURCE_SIGNATURES:
                        npz_key = f"{signature}__{component}"
                        if npz_key not in data:
                            raise KeyError(f"Missing {npz_key!r} in {path}")
                        value = np.asarray(data[npz_key], dtype=np.float64)
                        if np.any(value < -self.eps):
                            raise ValueError(f"Negative values in {npz_key!r}: {path}")
                        if expected_shape is None:
                            expected_shape = value.shape
                        elif value.shape != expected_shape:
                            raise ValueError(
                                f"Inconsistent shape for {component}: expected "
                                f"{expected_shape}, got {value.shape} in {path}"
                            )
                        loaded[signature][method].append(
                            np.log(np.maximum(value, 0.0) + self.eps).reshape(-1)
                        )

        for signature in self.SOURCE_SIGNATURES:
            for method in required_methods:
                loaded[signature][method] = np.stack(loaded[signature][method])
        self._component_shapes[component] = expected_shape
        self._component_cache[cache_key] = loaded
        return loaded

    def _log_profiles(self, scope, signature, component):
        source = self._load_component(scope, component)
        if signature in self.SOURCE_SIGNATURES:
            return source[signature]
        if signature == "S_m":
            return {m: source["F_m"][m] - source["P"][m] for m in source["P"]}
        if signature == "S_e":
            return {m: source["F_e"][m] - source["P"][m] for m in source["P"]}
        raise ValueError(f"Unknown signature {signature!r}")

    def _task_method_values(self, scope, representation, signature, component):
        methods = self._scope_methods(scope)
        profiles = self._log_profiles(scope, signature, component)
        x = np.stack([profiles[m] for m in methods], axis=1)
        if representation == "delta":
            x = x - profiles[self.REFERENCE_METHOD][:, None]
        elif representation != "raw":
            raise ValueError(f"Unknown representation {representation!r}")
        return x

    def _effects(self, scope, representation, signature, component):
        key = (scope, representation, signature, component)
        if key in self._effects_cache:
            return self._effects_cache[key]

        x = self._task_method_values(scope, representation, signature, component)
        u = x.mean(axis=(0, 1))
        y = x - u
        task = y.mean(axis=1)
        method = y.mean(axis=0)
        interaction = y - task[:, None] - method[None]
        group = np.empty_like(task)
        for indices in self._official_group_indices.values():
            group[indices] = task[indices].mean(axis=0)
        dataset = task - group

        self._assert_close(y, task[:, None] + method[None] + interaction, key)
        self._assert_close(task, group + dataset, key)
        self._assert_close(task.sum(axis=0), 0.0, f"{key}:task")
        self._assert_close(method.sum(axis=0), 0.0, f"{key}:method")
        self._assert_close(interaction.sum(axis=1), 0.0, f"{key}:interaction-method")
        self._assert_close(interaction.sum(axis=0), 0.0, f"{key}:interaction-task")

        effects = {
            "x": x,
            "u": u,
            "y": y,
            "task": task,
            "method": method,
            "interaction": interaction,
            "group": group,
            "dataset": dataset,
        }
        self._effects_cache[key] = effects
        return effects

    def _attributions(self, effects, label):
        y = effects["y"]
        energy = np.sum(y**2, axis=-1)
        contributions = np.stack(
            (
                np.sum(y * effects["group"][:, None], axis=-1),
                np.sum(y * effects["dataset"][:, None], axis=-1),
                np.sum(y * effects["method"][None], axis=-1),
                np.sum(y * effects["interaction"], axis=-1),
            ),
            axis=-1,
        )
        self._assert_close(energy, contributions.sum(axis=-1), label)
        gross = np.abs(contributions).sum(axis=-1, keepdims=True)
        r = np.divide(
            np.abs(contributions),
            gross,
            out=np.full_like(contributions, np.nan),
            where=gross > np.finfo(float).eps,
        )
        return {
            "energy": energy,
            "c_group": contributions[..., 0],
            "c_dataset": contributions[..., 1],
            "c_method": contributions[..., 2],
            "c_interaction": contributions[..., 3],
            "r": r,
        }

    def _dataset_organization(self, effects, n_methods, label):
        rows = []
        for ti, task in enumerate(self.DATASETS):
            ss_task = n_methods * np.sum(effects["task"][ti] ** 2)
            ss_method = np.sum((effects["method"] + effects["interaction"][ti]) ** 2)
            energy = np.sum(effects["y"][ti] ** 2)
            self._assert_close(energy, ss_task + ss_method, (label, task))
            rows.append(
                {
                    "task": task,
                    "ss_task_shared": float(ss_task),
                    "ss_method_conditioned": float(ss_method),
                    "r_task_shared": self._safe_ratio(ss_task, energy),
                    "r_method_conditioned": self._safe_ratio(ss_method, energy),
                }
            )
        return rows


    def analyze_pair_attribution(self):
        records, effect_paths = [], {}
        out = self._analysis_dir("decomposition", "cell")
        for scope, methods in self.METHOD_SCOPES.items():
            for representation in self.REPRESENTATIONS:
                self._log(f"Cell attribution [{scope}] [{representation}]")
                payload = {}
                for signature in self.SIGNATURES:
                    for component in self._scope_components(scope):
                        effects = self._effects(
                            scope, representation, signature, component
                        )
                        attrs = self._attributions(
                            effects, (scope, representation, signature, component)
                        )
                        for ti, task in enumerate(self.DATASETS):
                            for mi, method in enumerate(methods):
                                r = attrs["r"][ti, mi]
                                records.append(
                                    {
                                        "representation": representation,
                                        "scope": scope,
                                        "signature": signature,
                                        "component": component,
                                        "task": task,
                                        "task_group": self._task_group[task],
                                        "method": method,
                                        "energy_centered": attrs["energy"][ti, mi],
                                        "c_group": attrs["c_group"][ti, mi],
                                        "c_dataset": attrs["c_dataset"][ti, mi],
                                        "c_method": attrs["c_method"][ti, mi],
                                        "c_interaction": attrs["c_interaction"][ti, mi],
                                        "r_group": r[0],
                                        "r_dataset": r[1],
                                        "r_method": r[2],
                                        "r_interaction": r[3],
                                    }
                                )
                        prefix = f"{signature}__{component}"
                        for name in (
                            "u",
                            "task",
                            "method",
                            "interaction",
                            "group",
                            "dataset",
                        ):
                            payload[f"{prefix}__{name}"] = effects[name]

                npz_path = out / f"effects_{scope}_{representation}.npz"
                np.savez_compressed(
                    npz_path,
                    datasets=np.array(self.DATASETS),
                    methods=np.array(methods),
                    signatures=np.array(self.SIGNATURES),
                    components=np.array(self._scope_components(scope)),
                    representation=np.array(representation),
                    scope=np.array(scope),
                    **payload,
                )
                effect_paths[(scope, representation)] = npz_path

        return {
            "cell_attribution": self._write_csv(out / "cell_attribution.csv", records),
            "effects": effect_paths,
        }

    def analyze_dataset_organization(self):
        records = []
        for scope, methods in self.METHOD_SCOPES.items():
            for representation in self.REPRESENTATIONS:
                for signature in self.SIGNATURES:
                    for component in self._scope_components(scope):
                        effects = self._effects(
                            scope, representation, signature, component
                        )
                        for row in self._dataset_organization(
                            effects,
                            len(methods),
                            (scope, representation, signature, component),
                        ):
                            records.append(
                                {
                                    "representation": representation,
                                    "scope": scope,
                                    "signature": signature,
                                    "component": component,
                                    "task": row["task"],
                                    "task_group": self._task_group[row["task"]],
                                    **{k: v for k, v in row.items() if k != "task"},
                                }
                            )
        out = self._analysis_dir("decomposition", "dataset")
        return self._write_csv(out / "dataset_organization.csv", records)

    def analyze_method_organization(self):
        records, n_tasks = [], len(self.DATASETS)
        for scope, methods in self.METHOD_SCOPES.items():
            for representation in self.REPRESENTATIONS:
                for signature in self.SIGNATURES:
                    for component in self._scope_components(scope):
                        effects = self._effects(
                            scope, representation, signature, component
                        )
                        for mi, method_name in enumerate(methods):
                            h = effects["task"] + effects["interaction"][:, mi]
                            ss_method = n_tasks * np.sum(effects["method"][mi] ** 2)
                            ss_group = ss_dataset = 0.0
                            for indices in self._official_group_indices.values():
                                mean = h[indices].mean(axis=0)
                                ss_group += len(indices) * np.sum(mean**2)
                                ss_dataset += np.sum((h[indices] - mean) ** 2)
                            energy = np.sum(effects["y"][:, mi] ** 2)
                            self._assert_close(
                                energy,
                                ss_method + ss_group + ss_dataset,
                                (
                                    scope,
                                    representation,
                                    signature,
                                    component,
                                    method_name,
                                ),
                            )
                            records.append(
                                {
                                    "representation": representation,
                                    "scope": scope,
                                    "signature": signature,
                                    "component": component,
                                    "method": method_name,
                                    "ss_method_general": float(ss_method),
                                    "ss_task_group": float(ss_group),
                                    "ss_dataset_specific": float(ss_dataset),
                                    "r_method_general": self._safe_ratio(
                                        ss_method, energy
                                    ),
                                    "r_task_group": self._safe_ratio(ss_group, energy),
                                    "r_dataset_specific": self._safe_ratio(
                                        ss_dataset, energy
                                    ),
                                    "r_task_conditioned": self._safe_ratio(
                                        ss_group + ss_dataset, energy
                                    ),
                                }
                            )
        out = self._analysis_dir("decomposition", "method")
        return self._write_csv(out / "method_organization.csv", records)

    def analyze_global_organization(self):
        records, n_tasks = [], len(self.DATASETS)
        for scope, methods in self.METHOD_SCOPES.items():
            for representation in self.REPRESENTATIONS:
                for signature in self.SIGNATURES:
                    rows = []
                    for component in self._scope_components(scope):
                        effects = self._effects(
                            scope, representation, signature, component
                        )
                        energy = np.sum(effects["y"] ** 2)
                        ss_method = n_tasks * np.sum(effects["method"] ** 2)
                        ss_interaction = np.sum(effects["interaction"] ** 2)
                        ss_group = len(methods) * np.sum(effects["group"] ** 2)
                        ss_dataset = len(methods) * np.sum(effects["dataset"] ** 2)
                        self._assert_close(
                            energy,
                            ss_group + ss_dataset + ss_method + ss_interaction,
                            (scope, representation, signature, component),
                        )
                        row = {
                            "representation": representation,
                            "scope": scope,
                            "signature": signature,
                            "component": component,
                            "ss_task_group": float(ss_group),
                            "ss_dataset_specific": float(ss_dataset),
                            "ss_method": float(ss_method),
                            "ss_interaction": float(ss_interaction),
                            "r_task_group": self._safe_ratio(ss_group, energy),
                            "r_dataset_specific": self._safe_ratio(ss_dataset, energy),
                            "r_method": self._safe_ratio(ss_method, energy),
                            "r_interaction": self._safe_ratio(ss_interaction, energy),
                            "r_task": self._safe_ratio(ss_group + ss_dataset, energy),
                        }
                        rows.append(row)
                        records.append(row)

                    core = [r for r in rows if r["component"] in self.CORE_COMPONENTS]
                    records.append(
                        {
                            "representation": representation,
                            "scope": scope,
                            "signature": signature,
                            "component": "mean_core",
                            "ss_task_group": np.nan,
                            "ss_dataset_specific": np.nan,
                            "ss_method": np.nan,
                            "ss_interaction": np.nan,
                            **{
                                key: float(np.mean([r[key] for r in core]))
                                for key in (
                                    "r_task_group",
                                    "r_dataset_specific",
                                    "r_method",
                                    "r_interaction",
                                    "r_task",
                                )
                            },
                        }
                    )
        out = self._analysis_dir("decomposition", "global")
        return self._write_csv(out / "global_organization.csv", records)


    def _get_runs(self, dataset_name, method_name):
        key = (dataset_name, method_name)
        if key in self._runs_cache:
            return self._runs_cache[key]
        method_dir = self._method_dir(dataset_name, method_name)
        if not method_dir.is_dir():
            raise FileNotFoundError(method_dir)
        runs = []
        for path in method_dir.iterdir():
            if not path.is_dir():
                continue
            config_path, model_path = path / "config.json", path / "model.pt"
            accuracy_path = path / "test_acc_eval.json"
            if not (
                config_path.is_file()
                and model_path.is_file()
                and accuracy_path.is_file()
            ):
                continue
            with config_path.open() as f:
                config = json.load(f)
            runs.append(
                {"seed": int(config["run_seed"]), "path": path, "config": config}
            )
        if not runs:
            raise FileNotFoundError(
                f"No completed runs for {dataset_name}/{method_name}"
            )
        runs.sort(key=lambda x: x["seed"])
        seeds = [r["seed"] for r in runs]
        if len(seeds) != len(set(seeds)):
            raise RuntimeError(f"Duplicate seeds for {dataset_name}/{method_name}")
        self._runs_cache[key] = runs
        return runs

    def _prediction_cache_path(self, dataset_name, method_name, seed):
        path = (
            self.save_dir
            / "prediction_cache"
            / dataset_name
            / method_name
            / f"seed-{seed}.npz"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _get_seed_logits(self, dataset_name, method_name, run):
        cache = self._prediction_cache_path(dataset_name, method_name, run["seed"])
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as data:
                return data["logits"], data["labels"]

        config = run["config"]
        dataset = datasets[dataset_name](split="test")
        loader = dataset.get_dataloader(
            batch_size=int(config["gpu_batch_size"]),
            worker_init_fn=seed_worker,
        )
        model = models[self.backbone_name][config["peft_name"]](
            img_size=datasets[dataset_name].IMG_SIZE,
            n_classes=datasets[dataset_name].N_CLASSES,
            freeze_backbone=bool(config["freeze_backbone"]),
            **config["peft_config"],
        )
        model.load_from_path(str(run["path"] / "model.pt"), map_location="cpu")
        model.to(self.device).eval()

        logits, labels = [], []
        with torch.inference_mode():
            for x, y in self._progress(
                loader,
                f"Predict {dataset_name}/{method_name}/seed-{run['seed']}",
                total=len(loader),
            ):
                logits.append(model(x.to(self.device)).cpu())
                labels.append(y.cpu())

        logits = torch.cat(logits).numpy().astype(np.float32)
        labels = torch.cat(labels).numpy()
        np.savez(cache, logits=logits, labels=labels)

        del model, dataset, loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return logits, labels

    def _get_method_predictions(self, dataset_name, method_name):
        probabilities, seeds, labels = [], [], None
        for run in self._get_runs(dataset_name, method_name):
            logits, y = self._get_seed_logits(dataset_name, method_name, run)
            if labels is None:
                labels = y
            elif not np.array_equal(labels, y):
                raise RuntimeError(
                    f"Test ordering mismatch for {dataset_name}/{method_name}"
                )
            values = logits.astype(np.float64)
            values -= values.max(axis=1, keepdims=True)
            probs = np.exp(values)
            probs /= probs.sum(axis=1, keepdims=True)
            probabilities.append(probs)
            seeds.append(run["seed"])
        probabilities = np.stack(probabilities)
        top1 = probabilities.argmax(axis=-1)
        correct = top1 == labels[None]
        return {
            "seeds": np.asarray(seeds),
            "probabilities": probabilities,
            "top1": top1,
            "correct": correct,
            "accuracy": correct.mean(axis=1),
            "labels": labels,
        }

    def _jsd(self, p, q):
        p = np.clip(np.asarray(p, dtype=np.float64), self.eps, 1.0)
        q = np.clip(np.asarray(q, dtype=np.float64), self.eps, 1.0)
        p /= p.sum(axis=-1, keepdims=True)
        q /= q.sum(axis=-1, keepdims=True)
        m = 0.5 * (p + q)
        return float(
            (
                0.5 * np.sum(p * np.log2(p / m), axis=-1)
                + 0.5 * np.sum(q * np.log2(q / m), axis=-1)
            ).mean()
        )

    @staticmethod
    def _error_consistency(correct_a, correct_b):
        a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
        observed = np.mean(a == b)
        pa, pb = a.mean(), b.mean()
        expected = pa * pb + (1 - pa) * (1 - pb)
        d = 1 - expected
        return np.nan if d <= np.finfo(float).eps else float((observed - expected) / d)

    def _behavior_metrics(self, a, b, seed_pairs):
        jsd, top1, consistency, accuracy = [], [], [], []
        for i, j in seed_pairs:
            jsd.append(self._jsd(a["probabilities"][i], b["probabilities"][j]))
            top1.append(float(np.mean(a["top1"][i] != b["top1"][j])))
            consistency.append(
                self._error_consistency(a["correct"][i], b["correct"][j])
            )
            accuracy.append(abs(a["accuracy"][i] - b["accuracy"][j]))
        consistency = np.asarray(consistency, dtype=float)
        kappa = (
            float(np.nanmean(consistency))
            if np.any(np.isfinite(consistency))
            else np.nan
        )
        return {
            "jsd": float(np.mean(jsd)),
            "top1_disagreement": float(np.mean(top1)),
            "error_consistency": kappa,
            "error_dissimilarity": 1 - kappa if np.isfinite(kappa) else np.nan,
            "accuracy_gap": float(np.mean(accuracy)),
            "n_seed_pairs": len(seed_pairs),
        }

    def _build_behavior_outputs(self):
        if self._behavior_cache is not None:
            return self._behavior_cache

        matrices, records, seed_records = {}, [], []
        for task in self._progress(
            self.DATASETS, "Prediction behavior", len(self.DATASETS)
        ):
            predictions = {
                m: self._get_method_predictions(task, m) for m in self.METHODS
            }
            labels = predictions[self.METHODS[0]]["labels"]
            if any(
                not np.array_equal(labels, predictions[m]["labels"])
                for m in self.METHODS[1:]
            ):
                raise RuntimeError(f"Test ordering mismatch across methods for {task}")

            for method, pred in predictions.items():
                pairs = list(itertools.combinations(range(len(pred["seeds"])), 2))
                metrics = (
                    self._behavior_metrics(pred, pred, pairs)
                    if pairs
                    else {
                        "jsd": np.nan,
                        "top1_disagreement": np.nan,
                        "error_consistency": np.nan,
                        "error_dissimilarity": np.nan,
                        "accuracy_gap": np.nan,
                        "n_seed_pairs": 0,
                    }
                )
                seed_records.append(
                    {
                        "task": task,
                        "task_group": self._task_group[task],
                        "method": method,
                        "mean_accuracy": float(np.mean(pred["accuracy"])),
                        "mean_within_jsd": metrics["jsd"],
                        "mean_within_top1_disagreement": metrics["top1_disagreement"],
                        "mean_within_error_consistency": metrics["error_consistency"],
                        "mean_within_error_dissimilarity": metrics[
                            "error_dissimilarity"
                        ],
                        "n_seed_pairs": metrics["n_seed_pairs"],
                    }
                )

            for scope, methods in self.METHOD_SCOPES.items():
                n = len(methods)
                task_matrices = {
                    name: np.zeros((n, n), dtype=float)
                    for name in (
                        *self.BEHAVIOR_METRICS,
                        "error_consistency",
                        "accuracy_gap",
                    )
                }
                for i, j in itertools.combinations(range(n), 2):
                    a, b = predictions[methods[i]], predictions[methods[j]]
                    pairs = list(
                        itertools.product(
                            range(len(a["seeds"])), range(len(b["seeds"]))
                        )
                    )
                    metrics = self._behavior_metrics(a, b, pairs)
                    for name in task_matrices:
                        task_matrices[name][i, j] = task_matrices[name][j, i] = metrics[
                            name
                        ]
                    records.append(
                        {
                            "scope": scope,
                            "task": task,
                            "task_group": self._task_group[task],
                            "method_a": methods[i],
                            "method_b": methods[j],
                            **metrics,
                        }
                    )
                matrices[(scope, task)] = task_matrices

            del predictions
            gc.collect()

        self._behavior_cache = (matrices, records, seed_records)
        return self._behavior_cache


    def _signature_distance_bundle(self, scope, signature, representation, component):
        key = (scope, signature, representation, component)
        if key in self._signature_distance_cache:
            return self._signature_distance_cache[key]

        methods = self._scope_methods(scope)
        profiles = self._log_profiles(scope, signature, component)
        values = np.stack([profiles[m] for m in methods], axis=1)
        if representation == "delta":
            values = values - profiles[self.REFERENCE_METHOD][:, None]
            fn = self._pairwise_angular_distance
        elif representation == "raw":
            fn = self._pairwise_rms_distance
        else:
            raise ValueError(representation)

        matrices = np.stack([fn(values[t]) for t in range(len(self.DATASETS))])
        self._signature_distance_cache[key] = matrices
        return matrices

    def _summarize_alignment(self, records):
        keys = (
            "scope",
            "signature",
            "representation",
            "metric",
            "component",
            "behavior_metric",
        )
        grouped = defaultdict(list)
        for row in records:
            grouped[tuple(row[k] for k in keys)].append(row)
        out = []
        for key, rows in grouped.items():
            summary = dict(zip(keys, key))
            summary.update(
                {
                    "mean_rho": self._mean_correlation([r["rho"] for r in rows]),
                    "mean_partial_rho_accuracy": self._mean_correlation(
                        [r["partial_rho_accuracy"] for r in rows]
                    ),
                    "n_tasks": sum(np.isfinite(r["rho"]) for r in rows),
                    "n_tasks_partial": sum(
                        np.isfinite(r["partial_rho_accuracy"]) for r in rows
                    ),
                }
            )
            for group in self.OFFICIAL_TASK_GROUPS:
                g = [r for r in rows if r["task_group"] == group]
                summary[f"rho_{group}"] = self._mean_correlation([r["rho"] for r in g])
                summary[f"partial_rho_{group}"] = self._mean_correlation(
                    [r["partial_rho_accuracy"] for r in g]
                )
                summary[f"n_tasks_{group}"] = sum(np.isfinite(r["rho"]) for r in g)
                summary[f"n_tasks_partial_{group}"] = sum(
                    np.isfinite(r["partial_rho_accuracy"]) for r in g
                )
            out.append(summary)
        return out

    def _record_signature_bundle(
        self,
        scope,
        signature,
        representation,
        component,
        matrices,
        behavior,
        distance_records,
        alignment_records,
    ):
        methods = self._scope_methods(scope)
        metric = "rms_log" if representation == "raw" else "angular"
        for ti, task in enumerate(self.DATASETS):
            matrix = matrices[ti]
            for i, j in itertools.combinations(range(len(methods)), 2):
                distance_records.append(
                    {
                        "scope": scope,
                        "task": task,
                        "task_group": self._task_group[task],
                        "signature": signature,
                        "representation": representation,
                        "metric": metric,
                        "component": component,
                        "method_a": methods[i],
                        "method_b": methods[j],
                        "distance": matrix[i, j],
                    }
                )

            accuracy = behavior[(scope, task)]["accuracy_gap"]
            x, z = self._upper(matrix), self._upper(accuracy)
            for behavior_metric in self.BEHAVIOR_METRICS:
                predictive = behavior[(scope, task)][behavior_metric]
                y = self._upper(predictive)
                valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                alignment_records.append(
                    {
                        "scope": scope,
                        "task": task,
                        "task_group": self._task_group[task],
                        "signature": signature,
                        "representation": representation,
                        "metric": metric,
                        "component": component,
                        "behavior_metric": behavior_metric,
                        "rho": self._spearman(x[valid], y[valid]),
                        "p_permutation": self._rsa_permutation_p(matrix, predictive),
                        "partial_rho_accuracy": self._partial_spearman(
                            x[valid], y[valid], z[valid]
                        ),
                        "partial_p_permutation_accuracy": self._rsa_permutation_p(
                            matrix, predictive, accuracy
                        ),
                        "n_method_pairs": int(valid.sum()),
                    }
                )

    def analyze_prediction_alignment(self):
        behavior, prediction_records, seed_records = self._build_behavior_outputs()
        distance_records, alignment_records = [], []
        matrix_paths, prediction_matrix_paths = {}, {}

        out = self._analysis_dir("signature_alignment")
        pred_out = self._analysis_dir("prediction_behavior")

        for scope in self.METHOD_SCOPES:
            self._log(f"Signature alignment [{scope}]")
            payload = {}
            for signature in self.SIGNATURES:
                for representation in self.REPRESENTATIONS:
                    core = []
                    for component in self._scope_components(scope):
                        matrices = self._signature_distance_bundle(
                            scope, signature, representation, component
                        )
                        payload[f"{signature}__{representation}__{component}"] = matrices
                        self._record_signature_bundle(
                            scope,
                            signature,
                            representation,
                            component,
                            matrices,
                            behavior,
                            distance_records,
                            alignment_records,
                        )
                        if component in self.CORE_COMPONENTS:
                            core.append(matrices)

                    mean_core = np.mean(core, axis=0)
                    payload[f"{signature}__{representation}__mean_core"] = mean_core
                    self._record_signature_bundle(
                        scope,
                        signature,
                        representation,
                        "mean_core",
                        mean_core,
                        behavior,
                        distance_records,
                        alignment_records,
                    )

            matrix_path = out / f"signature_distance_matrices_{scope}.npz"
            np.savez_compressed(
                matrix_path,
                datasets=np.array(self.DATASETS),
                methods=np.array(self._scope_methods(scope)),
                **payload,
            )
            matrix_paths[scope] = matrix_path

            prediction_matrix_path = pred_out / f"prediction_matrices_{scope}.npz"
            pred_payload = {
                f"{task}__{metric}": behavior[(scope, task)][metric]
                for task in self.DATASETS
                for metric in (
                    *self.BEHAVIOR_METRICS,
                    "error_consistency",
                    "accuracy_gap",
                )
            }
            np.savez_compressed(
                prediction_matrix_path,
                datasets=np.array(self.DATASETS),
                methods=np.array(self._scope_methods(scope)),
                **pred_payload,
            )
            prediction_matrix_paths[scope] = prediction_matrix_path

        return {
            "prediction_distances": self._write_csv(
                pred_out / "prediction_distances.csv", prediction_records
            ),
            "prediction_matrices": prediction_matrix_paths,
            "seed_behavior_variability": self._write_csv(
                pred_out / "seed_behavior_variability.csv", seed_records
            ),
            "signature_distances": self._write_csv(
                out / "signature_distances.csv", distance_records
            ),
            "signature_distance_matrices": matrix_paths,
            "alignment_by_task": self._write_csv(
                out / "alignment_by_task.csv", alignment_records
            ),
            "alignment_summary": self._write_csv(
                out / "alignment_summary.csv",
                self._summarize_alignment(alignment_records),
            ),
        }

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
            if methods == self.ADAPTATION_METHODS and seeds == (0, 1, 2):
                return distances

        runs = {m: self._get_runs(task, m) for m in self.ADAPTATION_METHODS}
        seeds = tuple(run["seed"] for run in runs[self.ADAPTATION_METHODS[0]])
        if seeds != (0, 1, 2) or any(
            tuple(run["seed"] for run in runs[m]) != seeds
            for m in self.ADAPTATION_METHODS[1:]
        ):
            raise RuntimeError(f"CKA requires seeds (0, 1, 2) for every method in {task}")

        dataset = datasets[task](split="train+val")
        features, labels_ref = [], None

        for method in self.ADAPTATION_METHODS:
            for run in runs[method]:
                config = run["config"]
                loader = dataset.get_dataloader(
                    batch_size=int(config["gpu_batch_size"]),
                    num_workers=0,
                )
                model = models[self.backbone_name][config["peft_name"]](
                    img_size=datasets[task].IMG_SIZE,
                    n_classes=datasets[task].N_CLASSES,
                    freeze_backbone=bool(config["freeze_backbone"]),
                    **config["peft_config"],
                )
                model.load_from_path(str(run["path"] / "model.pt"), map_location="cpu")
                model.to(self.device).eval()

                cls_batches, labels = [], []
                with torch.inference_mode():
                    for x, y in self._progress(
                        loader,
                        f"CKA {task}/{method}/seed-{run['seed']}",
                        total=len(loader),
                    ):
                        cls_batches.append(
                            model.get_layerwise_features(x.to(self.device))["cls"].cpu()
                        )
                        labels.append(y.cpu())

                cls = torch.cat(cls_batches, dim=1).float()
                labels = torch.cat(labels).numpy()
                if labels_ref is None:
                    labels_ref = labels
                elif not np.array_equal(labels_ref, labels):
                    raise RuntimeError(f"Train+val ordering mismatch for CKA on {task}")
                features.append(cls)

                del model, loader, cls_batches
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        shape = features[0].shape
        if any(f.shape != shape for f in features[1:]):
            raise RuntimeError(f"Layerwise feature shape mismatch for CKA on {task}")

        n_checkpoints, n_layers = len(features), shape[0]
        similarities = np.empty((n_checkpoints, n_checkpoints, n_layers), dtype=np.float32)

        for layer in range(n_layers):
            kernels = []
            for feature in features:
                x = feature[layer].to(self.device)
                x = x - x.mean(dim=0, keepdim=True)
                kernel = x @ x.T
                norm = torch.linalg.vector_norm(kernel)
                if norm <= self.eps:
                    raise RuntimeError(f"Degenerate CKA representation for {task}, layer {layer}")
                kernels.append((kernel / norm).reshape(-1))

            kernels = torch.stack(kernels)
            similarities[:, :, layer] = (
                (kernels @ kernels.T).clamp(0.0, 1.0).cpu().numpy()
            )

        n_methods, n_seeds = len(self.ADAPTATION_METHODS), len(seeds)
        distances = (1.0 - similarities).reshape(
            n_methods, n_seeds, n_methods, n_seeds, n_layers
        ).transpose(0, 2, 1, 3, 4)

        np.savez_compressed(
            cache,
            methods=np.array(self.ADAPTATION_METHODS),
            seeds=np.array(seeds),
            layerwise_distance=distances,
        )

        del features, dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return distances

    def _cka_scope_matrices(self, scope):
        methods = self._scope_methods(scope)
        out = self._analysis_dir("cka_baseline")
        path = out / f"cka_distance_matrices_{scope}.npz"

        indices = [self.ADAPTATION_METHODS.index(m) for m in methods]
        matrices = {name: [] for name in self.CKA_SUMMARIES}

        for task in self.DATASETS:
            distances = self._get_task_cka(task)
            layerwise = distances.mean(axis=(2, 3))[np.ix_(indices, indices)]

            mean_layers = layerwise.mean(axis=-1)
            final_layer = layerwise[..., -1]

            np.fill_diagonal(mean_layers, 0.0)
            np.fill_diagonal(final_layer, 0.0)

            matrices["mean_layers"].append(mean_layers)
            matrices["final_layer"].append(final_layer)

        matrices = {k: np.stack(v) for k, v in matrices.items()}

        np.savez_compressed(
            path,
            datasets=np.array(self.DATASETS),
            methods=np.array(methods),
            **matrices,
        )

        return matrices, path

    def _summarize_cka_alignment(self, records):
        grouped = defaultdict(list)
        for row in records:
            grouped[(row["scope"], row["cka_summary"], row["behavior_metric"])].append(row)

        summaries = []
        for (scope, cka_summary, behavior_metric), rows in grouped.items():
            summary = {
                "scope": scope,
                "cka_summary": cka_summary,
                "behavior_metric": behavior_metric,
                "mean_rho": self._mean_correlation([r["rho"] for r in rows]),
                "n_tasks": sum(np.isfinite(r["rho"]) for r in rows),
            }
            for group in self.OFFICIAL_TASK_GROUPS:
                values = [r["rho"] for r in rows if r["task_group"] == group]
                summary[f"rho_{group}"] = self._mean_correlation(values)
                summary[f"n_tasks_{group}"] = sum(np.isfinite(v) for v in values)
            summaries.append(summary)
        return summaries

    def analyze_cka_alignment(self):
        behavior, _, _ = self._build_behavior_outputs()
        records, matrix_paths = [], {}

        for scope in self.METHOD_SCOPES:
            matrices, matrix_paths[scope] = self._cka_scope_matrices(scope)
            for ti, task in enumerate(self.DATASETS):
                for cka_summary, task_matrices in matrices.items():
                    x = self._upper(task_matrices[ti])
                    for behavior_metric in self.BEHAVIOR_METRICS:
                        y = self._upper(behavior[(scope, task)][behavior_metric])
                        valid = np.isfinite(x) & np.isfinite(y)
                        records.append(
                            {
                                "scope": scope,
                                "task": task,
                                "task_group": self._task_group[task],
                                "cka_summary": cka_summary,
                                "behavior_metric": behavior_metric,
                                "rho": self._spearman(x[valid], y[valid]),
                                "n_method_pairs": int(valid.sum()),
                            }
                        )

        out = self._analysis_dir("cka_baseline")
        return {
            "cka_alignment_by_task": self._write_csv(
                out / "cka_alignment_by_task.csv", records
            ),
            "cka_alignment_summary": self._write_csv(
                out / "cka_alignment_summary.csv",
                self._summarize_cka_alignment(records),
            ),
            "cka_distance_matrices": matrix_paths,
        }


    def analyze_r_distances(self):
        records, matrix_paths = [], {}
        out = self._analysis_dir("r_distances")
        for scope, methods in self.METHOD_SCOPES.items():
            payload = {}
            for signature in self.SIGNATURES:
                for representation in self.REPRESENTATIONS:
                    core = []
                    for component in self._scope_components(scope):
                        effects = self._effects(
                            scope, representation, signature, component
                        )
                        attrs = self._attributions(
                            effects,
                            ("r-distance", scope, representation, signature, component),
                        )
                        matrices = np.stack(
                            [
                                self._pairwise_tv_distance(attrs["r"][ti])
                                for ti in range(len(self.DATASETS))
                            ]
                        )
                        payload[f"{signature}__{representation}__{component}"] = matrices
                        if component in self.CORE_COMPONENTS:
                            core.append(matrices)
                        for ti, task in enumerate(self.DATASETS):
                            for i, j in itertools.combinations(range(len(methods)), 2):
                                records.append(
                                    {
                                        "scope": scope,
                                        "task": task,
                                        "task_group": self._task_group[task],
                                        "signature": signature,
                                        "representation": representation,
                                        "metric": "total_variation",
                                        "component": component,
                                        "method_a": methods[i],
                                        "method_b": methods[j],
                                        "distance": matrices[ti, i, j],
                                    }
                                )

                    mean_core = np.mean(core, axis=0)
                    payload[f"{signature}__{representation}__mean_core"] = mean_core
                    for ti, task in enumerate(self.DATASETS):
                        for i, j in itertools.combinations(range(len(methods)), 2):
                            records.append(
                                {
                                    "scope": scope,
                                    "task": task,
                                    "task_group": self._task_group[task],
                                    "signature": signature,
                                    "representation": representation,
                                    "metric": "total_variation",
                                    "component": "mean_core",
                                    "method_a": methods[i],
                                    "method_b": methods[j],
                                    "distance": mean_core[ti, i, j],
                                }
                            )

            npz_path = out / f"r_distance_matrices_{scope}.npz"
            np.savez_compressed(
                npz_path,
                datasets=np.array(self.DATASETS),
                methods=np.array(methods),
                **payload,
            )
            matrix_paths[scope] = npz_path

        return {
            "r_distances": self._write_csv(out / "r_distances.csv", records),
            "r_distance_matrices": matrix_paths,
        }


    def _cross_task_summary(self, x, y, z):
        x, y, z = map(lambda v: np.asarray(v, dtype=float), (x, y, z))
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        out = {
            "rho": self._spearman(x[valid], y[valid]),
            "partial_rho_accuracy": self._partial_spearman(
                x[valid], y[valid], z[valid]
            ),
            "n_tasks": int(valid.sum()),
        }
        for group, indices in self._official_group_indices.items():
            g = valid[indices]
            idx = indices[g]
            out[f"rho_{group}"] = self._spearman(x[idx], y[idx])
            out[f"partial_rho_{group}"] = self._partial_spearman(x[idx], y[idx], z[idx])
            out[f"n_tasks_{group}"] = int(len(idx))
        return out

    def analyze_r_prediction_diversity(self):
        behavior, _, _ = self._build_behavior_outputs()
        diversity, points, summary = [], [], []

        for scope, methods in self.METHOD_SCOPES.items():
            diversity_arrays = {
                m: [] for m in (*self.BEHAVIOR_METRICS, "accuracy_gap")
            }
            for task in self.DATASETS:
                row = {
                    "scope": scope,
                    "task": task,
                    "task_group": self._task_group[task],
                }
                for metric in diversity_arrays:
                    value = float(
                        np.nanmean(self._upper(behavior[(scope, task)][metric]))
                    )
                    row[f"mean_{metric}"] = value
                    diversity_arrays[metric].append(value)
                diversity.append(row)
            diversity_arrays = {
                k: np.asarray(v) for k, v in diversity_arrays.items()
            }

            for signature in self.SIGNATURES:
                for representation in self.REPRESENTATIONS:
                    r_by_component = {}
                    for component in self._scope_components(scope):
                        effects = self._effects(
                            scope, representation, signature, component
                        )
                        org = self._dataset_organization(
                            effects,
                            len(methods),
                            (
                                "r-diversity",
                                scope,
                                representation,
                                signature,
                                component,
                            ),
                        )
                        r_by_component[component] = np.asarray(
                            [row["r_method_conditioned"] for row in org], dtype=float
                        )
                    r_by_component["mean_core"] = np.mean(
                        [r_by_component[c] for c in self.CORE_COMPONENTS], axis=0
                    )

                    for component, r_values in r_by_component.items():
                        for ti, task in enumerate(self.DATASETS):
                            points.append(
                                {
                                    "scope": scope,
                                    "signature": signature,
                                    "representation": representation,
                                    "component": component,
                                    "task": task,
                                    "task_group": self._task_group[task],
                                    "r_method_conditioned": r_values[ti],
                                    **{
                                        f"mean_{metric}": values[ti]
                                        for metric, values in diversity_arrays.items()
                                    },
                                }
                            )
                        for behavior_metric in self.BEHAVIOR_METRICS:
                            summary.append(
                                {
                                    "scope": scope,
                                    "signature": signature,
                                    "representation": representation,
                                    "component": component,
                                    "behavior_metric": behavior_metric,
                                    **self._cross_task_summary(
                                        r_values,
                                        diversity_arrays[behavior_metric],
                                        diversity_arrays["accuracy_gap"],
                                    ),
                                }
                            )

        out = self._analysis_dir("r_prediction_diversity")
        return {
            "prediction_diversity_by_task": self._write_csv(
                out / "prediction_diversity_by_task.csv", diversity
            ),
            "r_prediction_diversity_points": self._write_csv(
                out / "r_prediction_diversity_points.csv", points
            ),
            "r_prediction_diversity_summary": self._write_csv(
                out / "r_prediction_diversity_summary.csv", summary
            ),
        }



    @staticmethod
    def _cluster_order(matrix):
        matrix = np.asarray(matrix, dtype=float)
        m = 0.5 * (matrix + matrix.T)
        np.fill_diagonal(m, 0.0)
        off = m[np.triu_indices_from(m, 1)]
        finite = off[np.isfinite(off)]
        fill = float(np.max(finite)) if finite.size else 0.0
        m = np.where(np.isfinite(m), m, fill)
        if len(m) < 3:
            return np.arange(len(m))
        condensed = squareform(m, checks=False)
        return leaves_list(linkage(condensed, method="average", optimal_ordering=True))

    @staticmethod
    def _plot_matrix(fig, ax, matrix, labels, title):
        image = ax.imshow(matrix)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = matrix[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=6)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    def analyze_heatmaps(self):
        behavior, _, _ = self._build_behavior_outputs()
        paths = {}
        for scope, methods in self.METHOD_SCOPES.items():
            prediction_mean = {
                metric: self._mean_matrices(
                    [behavior[(scope, task)][metric] for task in self.DATASETS]
                )
                for metric in self.BEHAVIOR_METRICS
            }
            pdf_path = (
                self._analysis_dir("figures")
                / f"{scope}_signature_prediction_heatmaps.pdf"
            )
            with PdfPages(pdf_path) as pdf:
                for signature in self.SIGNATURES:
                    for representation in self.REPRESENTATIONS:
                        bundles = {}
                        core = []
                        for component in self._scope_components(scope):
                            matrices = self._signature_distance_bundle(
                                scope, signature, representation, component
                            )
                            bundles[component] = matrices
                            if component in self.CORE_COMPONENTS:
                                core.append(matrices)
                        bundles["mean_core"] = np.mean(core, axis=0)

                        for component, matrices in bundles.items():
                            signature_mean = self._mean_matrices(matrices)
                            order = self._cluster_order(signature_mean)
                            labels = [methods[i].replace("_", " ") for i in order]
                            s = signature_mean[np.ix_(order, order)]

                            fig, axes = plt.subplots(3, 2, figsize=(11, 13))
                            metric_name = (
                                "RMS log" if representation == "raw" else "angular"
                            )
                            fig.suptitle(
                                f"{scope} | {signature} | {representation} | {component} | "
                                f"mean over VTAB-19",
                                fontsize=12,
                            )
                            for row, behavior_metric in enumerate(
                                self.BEHAVIOR_METRICS
                            ):
                                p = prediction_mean[behavior_metric][
                                    np.ix_(order, order)
                                ]
                                rhos, partials = [], []
                                for ti, task in enumerate(self.DATASETS):
                                    x = self._upper(matrices[ti])
                                    y = self._upper(
                                        behavior[(scope, task)][behavior_metric]
                                    )
                                    z = self._upper(
                                        behavior[(scope, task)]["accuracy_gap"]
                                    )
                                    valid = (
                                        np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                                    )
                                    rhos.append(self._spearman(x[valid], y[valid]))
                                    partials.append(
                                        self._partial_spearman(
                                            x[valid], y[valid], z[valid]
                                        )
                                    )
                                rho = self._mean_correlation(rhos)
                                partial = self._mean_correlation(partials)
                                self._plot_matrix(
                                    fig,
                                    axes[row, 0],
                                    s,
                                    labels,
                                    f"Signature distance ({metric_name})",
                                )
                                self._plot_matrix(
                                    fig,
                                    axes[row, 1],
                                    p,
                                    labels,
                                    f"{behavior_metric} | mean rho={rho:.3f}, partial={partial:.3f}",
                                )
                            fig.tight_layout(rect=(0, 0, 1, 0.97))
                            pdf.savefig(fig, bbox_inches="tight")
                            plt.close(fig)
            paths[scope] = pdf_path
        return paths


    def analyze_all(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "backbone": self.backbone_name,
            "prediction_cache_dir": str(self.save_dir / "prediction_cache"),
            "cka_cache_dir": str(self.save_dir / "cka_cache"),
            "scopes": {k: list(v) for k, v in self.METHOD_SCOPES.items()},
            "scope_components": {k: list(v) for k, v in self.SCOPE_COMPONENTS.items()},
            "core_components": list(self.CORE_COMPONENTS),
            "signatures": list(self.SIGNATURES),
            "representations": list(self.REPRESENTATIONS),
            "behavior_metrics": list(self.BEHAVIOR_METRICS),
            "correlation_summary": "arithmetic_mean",
            "cka": {
                "split": "train+val",
                "token": "cls",
                "metric": "linear_cka_distance",
                "seeds": [0, 1, 2],
                "seed_aggregation": "mean_all_9_cross_seed_pairs",
                "summaries": list(self.CKA_SUMMARIES),
            },
        }
        manifest_path = self.save_dir / "analysis_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return {
            "manifest": manifest_path,
            "cell": self.analyze_pair_attribution(),
            "dataset": self.analyze_dataset_organization(),
            "method": self.analyze_method_organization(),
            "global": self.analyze_global_organization(),
            "signature_alignment": self.analyze_prediction_alignment(),
            "cka_baseline": self.analyze_cka_alignment(),
            "r_distances": self.analyze_r_distances(),
            "r_prediction_diversity": self.analyze_r_prediction_diversity(),
            "heatmaps": self.analyze_heatmaps(),
        }
