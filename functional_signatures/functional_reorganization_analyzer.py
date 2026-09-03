import csv
import gc
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
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

    DATASETS = (
        *NATURAL,
        *SPECIALIZED,
        *STRUCTURED,
    )

    OFFICIAL_TASK_GROUPS = {
        "natural": NATURAL,
        "specialized": SPECIALIZED,
        "structured": STRUCTURED,
    }

    REFERENCE_METHOD = "linear_probe"

    PEFT_METHODS = (
        "vpt_shallow",
        "vpt_deep",
        "lora",
        "ssf",
        "adaptformer",
    )

    ADAPTATION_METHODS = (
        "full_ft",
        *PEFT_METHODS,
    )

    METHODS = (
        REFERENCE_METHOD,
        *ADAPTATION_METHODS,
    )

    METHOD_SCOPES = {
        "peft_only": PEFT_METHODS,
        "peft_plus_ft": ADAPTATION_METHODS,
    }

    REPRESENTATIONS = (
        "raw",
        "delta",
    )

    SOURCE_SIGNATURES = (
        "F_m",
        "P",
    )

    RQ1_SIGNATURES = (
        "F_m",
        "P",
        "S_m",
    )

    COMPONENTS = (
        "sublayer_input_channel",
        "attention_head_selection_route",
        "attention_head_transport_route",
        "mlp_neuron",
        "residual_token_carry",
        "sublayer_token_update",
        "final_cls_readout_channel",
        "residual_carry",
        "sublayer_update",
        "attention_selection_route",
        "attention_transport_route",
        "attention_head_output",
    )

    CORE_COMPONENTS = (
        "residual_token_carry",
        "sublayer_token_update",
        "attention_selection_route",
        "attention_transport_route",
    )

    EVAL_EXPERIMENT_NAME = "top_1_acc_eval"
    SIGNATURE_FILENAME = "functional_signatures.npz"

    def __init__(
        self,
        backbone_name="vit_b16",
        eps=1e-12,
        verbose=True,
    ):
        self.backbone_name = str(backbone_name)

        self.experiments_dir = Path(
            os.environ.get(
                "EXPERIMENTS_DIR",
                "./experiments",
            )
        )

        self.save_dir = (
            self.experiments_dir / "functional_reorganization" / self.backbone_name
        )

        self.eps = float(eps)
        self.verbose = bool(verbose)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._component_shapes = {}

        self._profile_cache = {}
        self._log_source_cache = {}
        self._derived_log_cache = {}

        self._values_cache = {}
        self._effects_cache = {}
        self._attribution_cache = {}
        self._dataset_org_cache = {}

        self._runs_cache = {}
        self._full_behavior_cache = {}
        self._behavior_cache = {}

        self._dataset_index = {name: i for i, name in enumerate(self.DATASETS)}

        self._official_group_indices = {
            name: np.array(
                [self._dataset_index[d] for d in group],
                dtype=int,
            )
            for name, group in self.OFFICIAL_TASK_GROUPS.items()
        }

    # ==================================================================
    # GENERIC HELPERS
    # ==================================================================

    def _log(
        self,
        message,
    ):
        if self.verbose:
            print(
                message,
                flush=True,
            )

    def _progress(
        self,
        iterable,
        desc,
        total=None,
        leave=False,
    ):
        return tqdm(
            iterable,
            desc=desc,
            total=total,
            disable=not self.verbose,
            leave=leave,
            dynamic_ncols=True,
        )

    def _method_dir(
        self,
        dataset_name,
        method_name,
    ):
        return (
            self.experiments_dir
            / self.EVAL_EXPERIMENT_NAME
            / self.backbone_name
            / dataset_name
            / method_name
        )

    def _signature_path(
        self,
        dataset_name,
        method_name,
    ):
        return (
            self._method_dir(
                dataset_name,
                method_name,
            )
            / self.SIGNATURE_FILENAME
        )

    def _analysis_save_dir(
        self,
        analysis_name,
        *parts,
    ):
        path = self.save_dir / analysis_name

        for part in parts:
            path /= part

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path

    def _scope_methods(
        self,
        scope,
    ):
        if scope not in self.METHOD_SCOPES:
            raise ValueError(f"Unknown scope {scope!r}")

        return self.METHOD_SCOPES[scope]

    @staticmethod
    def _safe_ratio(
        a,
        b,
    ):
        b = float(b)

        if abs(b) <= np.finfo(float).eps:
            return np.nan

        return float(a) / b

    @staticmethod
    def _safe_abs_shares(
        values,
    ):
        values = np.asarray(
            values,
            dtype=float,
        )

        denominator = np.abs(values).sum()

        if denominator <= np.finfo(float).eps:
            return np.full_like(
                values,
                np.nan,
            )

        return np.abs(values) / denominator

    @staticmethod
    def _rms(
        ss,
        n_cells,
        n_coordinates,
    ):
        return float(
            np.sqrt(
                max(
                    float(ss),
                    0.0,
                )
                / (n_cells * n_coordinates)
            )
        )

    @staticmethod
    def _vector_rms(
        x,
    ):
        x = np.asarray(
            x,
            dtype=float,
        ).reshape(-1)

        return float(np.linalg.norm(x) / np.sqrt(x.size))

    @staticmethod
    def _assert_close(
        a,
        b,
        label,
    ):
        if not np.allclose(
            a,
            b,
            rtol=1e-9,
            atol=1e-10,
        ):
            error = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

            raise RuntimeError(f"{label} failed; " f"max error={error}")

    @staticmethod
    def _write_csv(
        path,
        fieldnames,
        records,
    ):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for record in records:
                row = {
                    k: (
                        "NA"
                        if isinstance(
                            v,
                            (
                                float,
                                np.floating,
                            ),
                        )
                        and not np.isfinite(v)
                        else v
                    )
                    for k, v in record.items()
                }

                writer.writerow(row)

        return path

    @staticmethod
    def _spearman(
        x,
        y,
    ):
        value = spearmanr(
            x,
            y,
        ).statistic

        return float(value) if np.isfinite(value) else np.nan

    @staticmethod
    def _partial_spearman(
        x,
        y,
        z,
    ):
        rx, ry, rz = map(
            rankdata,
            (
                x,
                y,
                z,
            ),
        )

        design = np.column_stack(
            (
                np.ones(len(rz)),
                rz,
            )
        )

        residual_x = (
            rx
            - design
            @ np.linalg.lstsq(
                design,
                rx,
                rcond=None,
            )[0]
        )

        residual_y = (
            ry
            - design
            @ np.linalg.lstsq(
                design,
                ry,
                rcond=None,
            )[0]
        )

        if np.std(residual_x) == 0 or np.std(residual_y) == 0:
            return np.nan

        return float(
            np.corrcoef(
                residual_x,
                residual_y,
            )[0, 1]
        )

    def _rsa_permutation_p(
        self,
        functional,
        predictive,
        accuracy=None,
    ):
        if accuracy is None:
            observed = self._spearman(
                self._upper(functional),
                self._upper(predictive),
            )

        else:
            observed = self._partial_spearman(
                self._upper(functional),
                self._upper(predictive),
                self._upper(accuracy),
            )

        if not np.isfinite(observed):
            return np.nan

        count = 0
        total = 0

        for permutation in itertools.permutations(range(functional.shape[0])):
            permuted = functional[
                np.ix_(
                    permutation,
                    permutation,
                )
            ]

            if accuracy is None:
                value = self._spearman(
                    self._upper(permuted),
                    self._upper(predictive),
                )

            else:
                value = self._partial_spearman(
                    self._upper(permuted),
                    self._upper(predictive),
                    self._upper(accuracy),
                )

            if np.isfinite(value):
                total += 1

                count += abs(value) >= abs(observed) - 1e-15

        return count / total if total else np.nan

    @staticmethod
    def _upper(
        matrix,
    ):
        indices = np.triu_indices(
            matrix.shape[0],
            1,
        )

        return matrix[indices]

    @staticmethod
    def _pairwise_rms_distance(
        x,
    ):
        x = np.asarray(
            x,
            dtype=float,
        )

        differences = x[:, None, :] - x[None, :, :]

        return np.sqrt(
            np.mean(
                differences**2,
                axis=-1,
            )
        )

    def _jsd(
        self,
        p,
        q,
    ):
        p = np.clip(
            np.asarray(
                p,
                dtype=np.float64,
            ),
            self.eps,
            1.0,
        )

        q = np.clip(
            np.asarray(
                q,
                dtype=np.float64,
            ),
            self.eps,
            1.0,
        )

        p /= p.sum(
            axis=-1,
            keepdims=True,
        )

        q /= q.sum(
            axis=-1,
            keepdims=True,
        )

        middle = 0.5 * (p + q)

        jsd = 0.5 * np.sum(
            p * np.log2(p / middle),
            axis=-1,
        )

        jsd += 0.5 * np.sum(
            q * np.log2(q / middle),
            axis=-1,
        )

        return float(jsd.mean())

    # ==================================================================
    # SIGNATURE LOADING
    # ==================================================================

    def _load_source_profiles(
        self,
    ):
        loaded = {
            signature: {
                method: {component: [] for component in self.COMPONENTS}
                for method in self.METHODS
            }
            for signature in self.SOURCE_SIGNATURES
        }

        for method in self._progress(
            self.METHODS,
            desc="Loading signatures",
            total=len(self.METHODS),
        ):
            self._log(f"Loading signatures: " f"{method}")

            for dataset in self.DATASETS:
                path = self._signature_path(
                    dataset,
                    method,
                )

                if not path.is_file():
                    raise FileNotFoundError(path)

                with np.load(
                    path,
                    allow_pickle=False,
                ) as data:
                    for signature in self.SOURCE_SIGNATURES:
                        for component in self.COMPONENTS:
                            key = f"{signature}" f"__{component}"

                            value = np.asarray(
                                data[key],
                                dtype=np.float64,
                            )

                            shape = self._component_shapes.setdefault(
                                component,
                                value.shape,
                            )

                            if value.shape != shape:
                                raise ValueError(
                                    "Inconsistent shape " f"for {component}"
                                )

                            loaded[signature][method][component].append(value)

        for signature in self.SOURCE_SIGNATURES:
            for method in self.METHODS:
                for component in self.COMPONENTS:
                    loaded[signature][method][component] = np.stack(
                        loaded[signature][method][component]
                    )

        return loaded

    def _get_source_profiles(
        self,
    ):
        if not self._profile_cache:
            self._profile_cache = self._load_source_profiles()

        return self._profile_cache

    def _get_log_signature_profiles(
        self,
        signature,
    ):
        if signature in self.SOURCE_SIGNATURES:
            if signature not in self._log_source_cache:
                profiles = self._get_source_profiles()[signature]

                self._log_source_cache[signature] = {
                    method: {
                        component: np.log(profiles[method][component] + self.eps)
                        for component in self.COMPONENTS
                    }
                    for method in self.METHODS
                }

            return self._log_source_cache[signature]

        if signature != "S_m":
            raise ValueError(f"Unknown signature " f"{signature!r}")

        if signature not in self._derived_log_cache:
            fisher = self._get_log_signature_profiles("F_m")

            participation = self._get_log_signature_profiles("P")

            self._derived_log_cache[signature] = {
                method: {
                    component: (
                        fisher[method][component] - participation[method][component]
                    )
                    for component in self.COMPONENTS
                }
                for method in self.METHODS
            }

        return self._derived_log_cache[signature]

    def _get_task_method_values(
        self,
        scope,
        representation,
        signature,
        component,
    ):
        key = (
            scope,
            representation,
            signature,
            component,
        )

        if key in self._values_cache:
            return self._values_cache[key]

        methods = self._scope_methods(scope)

        profiles = self._get_log_signature_profiles(signature)

        reference = profiles[self.REFERENCE_METHOD][component]

        values = [
            (
                profiles[method][component] - reference
                if representation == "delta"
                else profiles[method][component]
            )
            for method in methods
        ]

        values = np.stack(
            values,
            axis=1,
        )

        values = values.reshape(
            values.shape[0],
            values.shape[1],
            -1,
        )

        self._values_cache[key] = values

        return values

    # ==================================================================
    # SHARED FACTORIAL STATE
    # ==================================================================

    def _get_effects(
        self,
        scope,
        representation,
        signature,
        component,
    ):
        key = (
            scope,
            representation,
            signature,
            component,
        )

        if key in self._effects_cache:
            return self._effects_cache[key]

        x = self._get_task_method_values(
            scope,
            representation,
            signature,
            component,
        )

        u = x.mean(axis=(0, 1))

        y = x - u

        tau = y.mean(axis=1)

        method = y.mean(axis=0)

        interaction = y - tau[:, None, :] - method[None, :, :]

        group = np.empty_like(tau)

        for indices in self._official_group_indices.values():
            group[indices] = tau[indices].mean(axis=0)

        task_specific = tau - group

        self._assert_close(
            y,
            (tau[:, None] + method[None] + interaction),
            key,
        )

        self._assert_close(
            tau,
            group + task_specific,
            key,
        )

        result = {
            "x": x,
            "u": u,
            "y": y,
            "tau": tau,
            "method": method,
            "interaction": interaction,
            "group": group,
            "task_specific": task_specific,
        }

        self._effects_cache[key] = result

        return result

    def _get_attributions(
        self,
        scope,
        representation,
        signature,
        component,
    ):
        key = (
            scope,
            representation,
            signature,
            component,
        )

        if key in self._attribution_cache:
            return self._attribution_cache[key]

        effects = self._get_effects(
            scope,
            representation,
            signature,
            component,
        )

        y = effects["y"]

        energy = np.sum(
            y**2,
            axis=-1,
        )

        c_task = np.sum(
            y * effects["tau"][:, None],
            axis=-1,
        )

        c_method = np.sum(
            y * effects["method"][None],
            axis=-1,
        )

        c_interaction = np.sum(
            y * effects["interaction"],
            axis=-1,
        )

        c_group = np.sum(
            y * effects["group"][:, None],
            axis=-1,
        )

        c_specific = np.sum(
            y * effects["task_specific"][:, None],
            axis=-1,
        )

        self._assert_close(
            energy,
            (c_task + c_method + c_interaction),
            key,
        )

        self._assert_close(
            c_task,
            c_group + c_specific,
            key,
        )

        gross = np.abs(c_task) + np.abs(c_method) + np.abs(c_interaction)

        task_gross = np.abs(c_group) + np.abs(c_specific)

        valid = gross > np.finfo(float).eps

        task_valid = task_gross > np.finfo(float).eps

        result = {
            "energy": energy,
            "c_task": c_task,
            "c_method": c_method,
            "c_interaction": c_interaction,
            "c_group": c_group,
            "c_task_specific": c_specific,
            "share_task": np.divide(
                np.abs(c_task),
                gross,
                out=np.full_like(
                    gross,
                    np.nan,
                ),
                where=valid,
            ),
            "share_method": np.divide(
                np.abs(c_method),
                gross,
                out=np.full_like(
                    gross,
                    np.nan,
                ),
                where=valid,
            ),
            "share_interaction": np.divide(
                np.abs(c_interaction),
                gross,
                out=np.full_like(
                    gross,
                    np.nan,
                ),
                where=valid,
            ),
            "share_group_within_task": np.divide(
                np.abs(c_group),
                task_gross,
                out=np.full_like(
                    task_gross,
                    np.nan,
                ),
                where=task_valid,
            ),
            "share_specific_within_task": np.divide(
                np.abs(c_specific),
                task_gross,
                out=np.full_like(
                    task_gross,
                    np.nan,
                ),
                where=task_valid,
            ),
        }

        self._attribution_cache[key] = result

        return result

    def _get_dataset_organization(
        self,
        scope,
        representation,
        signature,
        component,
    ):
        key = (
            scope,
            representation,
            signature,
            component,
        )

        if key in self._dataset_org_cache:
            return self._dataset_org_cache[key]

        methods = self._scope_methods(scope)

        effects = self._get_effects(
            scope,
            representation,
            signature,
            component,
        )

        attrs = self._get_attributions(
            scope,
            representation,
            signature,
            component,
        )

        n_methods = len(methods)

        rows = []

        for task_index, task in enumerate(self.DATASETS):
            ss_task = n_methods * np.sum(effects["tau"][task_index] ** 2)

            ss_method_conditioned = np.sum(
                (effects["method"] + effects["interaction"][task_index]) ** 2
            )

            energy = attrs["energy"][task_index].sum()

            rows.append(
                {
                    "task": task,
                    "r_method_conditioned": (
                        self._safe_ratio(
                            ss_method_conditioned,
                            energy,
                        )
                    ),
                    "ss_task_shared": float(ss_task),
                    "ss_method_conditioned": float(ss_method_conditioned),
                    "energy": float(energy),
                }
            )

        self._dataset_org_cache[key] = rows

        return rows

    # ==================================================================
    # SQ1
    # ==================================================================

    def analyze_pair_attribution(
        self,
    ):
        paths = {}

        for representation in self.REPRESENTATIONS:
            paths[representation] = {}

            for (
                scope,
                methods,
            ) in self.METHOD_SCOPES.items():
                self._log(f"\nSQ1 [{representation}] [{scope}]")

                output_dir = self._analysis_save_dir(
                    "sq1_pair_attribution",
                    representation,
                    scope,
                )

                records = []
                payload = {}

                for signature in self.RQ1_SIGNATURES:
                    for component in self._progress(
                        self.COMPONENTS,
                        desc=f"SQ1 {representation}/{scope} {signature}",
                        total=len(self.COMPONENTS),
                    ):
                        effects = self._get_effects(
                            scope,
                            representation,
                            signature,
                            component,
                        )

                        attrs = self._get_attributions(
                            scope,
                            representation,
                            signature,
                            component,
                        )

                        n_coordinates = effects["y"].shape[-1]

                        rms_universal = self._vector_rms(effects["u"])

                        for (
                            task_index,
                            task,
                        ) in enumerate(self.DATASETS):
                            for (
                                method_index,
                                method,
                            ) in enumerate(methods):
                                energy = attrs["energy"][
                                    task_index,
                                    method_index,
                                ]

                                records.append(
                                    {
                                        "signature": signature,
                                        "task": task,
                                        "method": method,
                                        "component": component,
                                        "n_coordinates": n_coordinates,
                                        "rms_universal": rms_universal,
                                        "energy_centered": energy,
                                        "rms_centered": self._rms(
                                            energy,
                                            1,
                                            n_coordinates,
                                        ),
                                        "c_task": attrs["c_task"][
                                            task_index,
                                            method_index,
                                        ],
                                        "c_method": attrs["c_method"][
                                            task_index,
                                            method_index,
                                        ],
                                        "c_interaction": attrs["c_interaction"][
                                            task_index,
                                            method_index,
                                        ],
                                        "share_task": attrs["share_task"][
                                            task_index,
                                            method_index,
                                        ],
                                        "share_method": attrs["share_method"][
                                            task_index,
                                            method_index,
                                        ],
                                        "share_interaction": attrs["share_interaction"][
                                            task_index,
                                            method_index,
                                        ],
                                        "c_task_group": attrs["c_group"][
                                            task_index,
                                            method_index,
                                        ],
                                        "c_individual_task": attrs["c_task_specific"][
                                            task_index,
                                            method_index,
                                        ],
                                        "share_task_group_within_task": (
                                            attrs["share_group_within_task"][
                                                task_index,
                                                method_index,
                                            ]
                                        ),
                                        "share_individual_task_within_task": (
                                            attrs["share_specific_within_task"][
                                                task_index,
                                                method_index,
                                            ]
                                        ),
                                    }
                                )

                        prefix = f"{signature}" f"__{component}"

                        for name in (
                            "u",
                            "tau",
                            "method",
                            "interaction",
                            "group",
                            "task_specific",
                        ):
                            payload[f"{prefix}" f"__{name}"] = effects[name]

                pairs_path = self._write_csv(
                    output_dir / "pairs.csv",
                    tuple(records[0]),
                    records,
                )

                effects_path = output_dir / "effects.npz"

                np.savez_compressed(
                    effects_path,
                    datasets=np.array(self.DATASETS),
                    methods=np.array(methods),
                    signatures=np.array(self.RQ1_SIGNATURES),
                    components=np.array(self.COMPONENTS),
                    **payload,
                )

                paths[representation][scope] = {
                    "pairs": pairs_path,
                    "effects": effects_path,
                }

        return paths

    # ==================================================================
    # SQ2
    # ==================================================================

    def analyze_dataset_organization(
        self,
    ):
        paths = {}

        for representation in self.REPRESENTATIONS:
            paths[representation] = {}

            for (
                scope,
                methods,
            ) in self.METHOD_SCOPES.items():
                self._log(f"\nSQ2 [{representation}] [{scope}]")

                records = []

                for signature in self.RQ1_SIGNATURES:
                    for component in self._progress(
                        self.COMPONENTS,
                        desc=f"SQ2 {representation}/{scope} {signature}",
                        total=len(self.COMPONENTS),
                    ):
                        effects = self._get_effects(
                            scope,
                            representation,
                            signature,
                            component,
                        )

                        attrs = self._get_attributions(
                            scope,
                            representation,
                            signature,
                            component,
                        )

                        organization = self._get_dataset_organization(
                            scope,
                            representation,
                            signature,
                            component,
                        )

                        n_coordinates = effects["y"].shape[-1]

                        for (
                            task_index,
                            task,
                        ) in enumerate(self.DATASETS):
                            c_task = attrs["c_task"][task_index].sum()

                            c_method = attrs["c_method"][task_index].sum()

                            c_interaction = attrs["c_interaction"][task_index].sum()

                            c_group = attrs["c_group"][task_index].sum()

                            c_specific = attrs["c_task_specific"][task_index].sum()

                            top_shares = self._safe_abs_shares(
                                (
                                    c_task,
                                    c_method,
                                    c_interaction,
                                )
                            )

                            nested_shares = self._safe_abs_shares(
                                (
                                    c_group,
                                    c_specific,
                                )
                            )

                            energy = organization[task_index]["energy"]

                            ss_task = organization[task_index]["ss_task_shared"]

                            ss_method_conditioned = organization[task_index][
                                "ss_method_conditioned"
                            ]

                            records.append(
                                {
                                    "signature": signature,
                                    "task": task,
                                    "component": component,
                                    "n_coordinates": n_coordinates,
                                    "rms_universal": self._vector_rms(effects["u"]),
                                    "energy_centered": energy,
                                    "c_task": c_task,
                                    "c_method": c_method,
                                    "c_interaction": c_interaction,
                                    "share_task": top_shares[0],
                                    "share_method": top_shares[1],
                                    "share_interaction": top_shares[2],
                                    "c_task_group": c_group,
                                    "c_individual_task": c_specific,
                                    "share_task_group_within_task": nested_shares[0],
                                    "share_individual_task_within_task": nested_shares[
                                        1
                                    ],
                                    "ss_task_shared": ss_task,
                                    "ss_method_conditioned": ss_method_conditioned,
                                    "r_task_shared": self._safe_ratio(
                                        ss_task,
                                        energy,
                                    ),
                                    "r_method_conditioned": self._safe_ratio(
                                        ss_method_conditioned,
                                        energy,
                                    ),
                                    "rms_centered": self._rms(
                                        energy,
                                        len(methods),
                                        n_coordinates,
                                    ),
                                    "rms_task_shared": self._rms(
                                        ss_task,
                                        len(methods),
                                        n_coordinates,
                                    ),
                                    "rms_method_conditioned": self._rms(
                                        ss_method_conditioned,
                                        len(methods),
                                        n_coordinates,
                                    ),
                                }
                            )

                output = self._write_csv(
                    self._analysis_save_dir(
                        "sq2_dataset_organization",
                        representation,
                        scope,
                    )
                    / "datasets.csv",
                    tuple(records[0]),
                    records,
                )

                paths[representation][scope] = {"datasets": output}

        return paths

    # ==================================================================
    # SQ3
    # ==================================================================

    def analyze_method_organization(
        self,
    ):
        paths = {}
        n_tasks = len(self.DATASETS)

        for representation in self.REPRESENTATIONS:
            paths[representation] = {}

            for (
                scope,
                methods,
            ) in self.METHOD_SCOPES.items():
                self._log(f"\nSQ3 [{representation}] [{scope}]")

                records = []

                for signature in self.RQ1_SIGNATURES:
                    for component in self._progress(
                        self.COMPONENTS,
                        desc=f"SQ3 {representation}/{scope} {signature}",
                        total=len(self.COMPONENTS),
                    ):
                        effects = self._get_effects(
                            scope,
                            representation,
                            signature,
                            component,
                        )

                        attrs = self._get_attributions(
                            scope,
                            representation,
                            signature,
                            component,
                        )

                        n_coordinates = effects["y"].shape[-1]

                        for (
                            method_index,
                            method_name,
                        ) in enumerate(methods):
                            task_conditioned = (
                                effects["tau"]
                                + effects["interaction"][
                                    :,
                                    method_index,
                                ]
                            )

                            ss_method = n_tasks * np.sum(
                                effects["method"][method_index] ** 2
                            )

                            ss_group = 0.0
                            ss_individual = 0.0

                            for indices in self._official_group_indices.values():
                                group_mean = task_conditioned[indices].mean(axis=0)

                                ss_group += len(indices) * np.sum(group_mean**2)

                                ss_individual += np.sum(
                                    (task_conditioned[indices] - group_mean) ** 2
                                )

                            energy = attrs["energy"][
                                :,
                                method_index,
                            ].sum()

                            c_task = attrs["c_task"][
                                :,
                                method_index,
                            ].sum()

                            c_method = attrs["c_method"][
                                :,
                                method_index,
                            ].sum()

                            c_interaction = attrs["c_interaction"][
                                :,
                                method_index,
                            ].sum()

                            c_group = attrs["c_group"][
                                :,
                                method_index,
                            ].sum()

                            c_specific = attrs["c_task_specific"][
                                :,
                                method_index,
                            ].sum()

                            top_shares = self._safe_abs_shares(
                                (
                                    c_task,
                                    c_method,
                                    c_interaction,
                                )
                            )

                            nested_shares = self._safe_abs_shares(
                                (
                                    c_group,
                                    c_specific,
                                )
                            )

                            ss_task_conditioned = ss_group + ss_individual

                            records.append(
                                {
                                    "signature": signature,
                                    "method": method_name,
                                    "component": component,
                                    "n_coordinates": n_coordinates,
                                    "rms_universal": self._vector_rms(effects["u"]),
                                    "energy_centered": energy,
                                    "c_task": c_task,
                                    "c_method": c_method,
                                    "c_interaction": c_interaction,
                                    "share_task": top_shares[0],
                                    "share_method": top_shares[1],
                                    "share_interaction": top_shares[2],
                                    "c_task_group": c_group,
                                    "c_individual_task": c_specific,
                                    "share_task_group_within_task": nested_shares[0],
                                    "share_individual_task_within_task": nested_shares[
                                        1
                                    ],
                                    "ss_method_general": ss_method,
                                    "ss_task_group": ss_group,
                                    "ss_individual_task": ss_individual,
                                    "ss_task_conditioned": ss_task_conditioned,
                                    "r_method_general": self._safe_ratio(
                                        ss_method,
                                        energy,
                                    ),
                                    "r_task_group": self._safe_ratio(
                                        ss_group,
                                        energy,
                                    ),
                                    "r_individual_task": self._safe_ratio(
                                        ss_individual,
                                        energy,
                                    ),
                                    "r_task_conditioned": self._safe_ratio(
                                        ss_task_conditioned,
                                        energy,
                                    ),
                                    "rms_centered": self._rms(
                                        energy,
                                        n_tasks,
                                        n_coordinates,
                                    ),
                                    "rms_method_general": self._rms(
                                        ss_method,
                                        n_tasks,
                                        n_coordinates,
                                    ),
                                    "rms_task_group": self._rms(
                                        ss_group,
                                        n_tasks,
                                        n_coordinates,
                                    ),
                                    "rms_individual_task": self._rms(
                                        ss_individual,
                                        n_tasks,
                                        n_coordinates,
                                    ),
                                }
                            )

                output = self._write_csv(
                    self._analysis_save_dir(
                        "sq3_method_organization",
                        representation,
                        scope,
                    )
                    / "methods.csv",
                    tuple(records[0]),
                    records,
                )

                paths[representation][scope] = {"methods": output}

        return paths

    # ==================================================================
    # SQ4
    # ==================================================================

    def analyze_global_organization(
        self,
    ):
        paths = {}

        n_tasks = len(self.DATASETS)

        for representation in self.REPRESENTATIONS:
            paths[representation] = {}

            for (
                scope,
                methods,
            ) in self.METHOD_SCOPES.items():
                self._log(f"\nSQ4 [{representation}] [{scope}]")

                records = []

                n_methods = len(methods)

                n_cells = n_tasks * n_methods

                for signature in self.RQ1_SIGNATURES:
                    for component in self._progress(
                        self.COMPONENTS,
                        desc=f"SQ4 {representation}/{scope} {signature}",
                        total=len(self.COMPONENTS),
                    ):
                        effects = self._get_effects(
                            scope,
                            representation,
                            signature,
                            component,
                        )

                        attrs = self._get_attributions(
                            scope,
                            representation,
                            signature,
                            component,
                        )

                        n_coordinates = effects["y"].shape[-1]

                        energy = attrs["energy"].sum()

                        ss_task = n_methods * np.sum(effects["tau"] ** 2)

                        ss_method = n_tasks * np.sum(effects["method"] ** 2)

                        ss_interaction = np.sum(effects["interaction"] ** 2)

                        ss_group = n_methods * np.sum(effects["group"] ** 2)

                        ss_individual = n_methods * np.sum(
                            effects["task_specific"] ** 2
                        )

                        c_task = attrs["c_task"].sum()

                        c_method = attrs["c_method"].sum()

                        c_interaction = attrs["c_interaction"].sum()

                        c_group = attrs["c_group"].sum()

                        c_specific = attrs["c_task_specific"].sum()

                        top_shares = self._safe_abs_shares(
                            (
                                c_task,
                                c_method,
                                c_interaction,
                            )
                        )

                        nested_shares = self._safe_abs_shares(
                            (
                                c_group,
                                c_specific,
                            )
                        )

                        self._assert_close(
                            energy,
                            (ss_task + ss_method + ss_interaction),
                            component,
                        )

                        self._assert_close(
                            ss_task,
                            (ss_group + ss_individual),
                            component,
                        )

                        records.append(
                            {
                                "signature": signature,
                                "component": component,
                                "n_coordinates": n_coordinates,
                                "universal_energy": (
                                    n_cells * np.sum(effects["u"] ** 2)
                                ),
                                "rms_universal": self._vector_rms(effects["u"]),
                                "energy_centered": energy,
                                "c_task": c_task,
                                "c_method": c_method,
                                "c_interaction": c_interaction,
                                "share_task": top_shares[0],
                                "share_method": top_shares[1],
                                "share_interaction": top_shares[2],
                                "c_task_group": c_group,
                                "c_individual_task": c_specific,
                                "share_task_group_within_task": nested_shares[0],
                                "share_individual_task_within_task": nested_shares[1],
                                "ss_task": ss_task,
                                "ss_method": ss_method,
                                "ss_interaction": ss_interaction,
                                "ss_task_group": ss_group,
                                "ss_individual_task": ss_individual,
                                "r_task": self._safe_ratio(
                                    ss_task,
                                    energy,
                                ),
                                "r_method": self._safe_ratio(
                                    ss_method,
                                    energy,
                                ),
                                "r_interaction": self._safe_ratio(
                                    ss_interaction,
                                    energy,
                                ),
                                "r_task_group": self._safe_ratio(
                                    ss_group,
                                    energy,
                                ),
                                "r_individual_task": self._safe_ratio(
                                    ss_individual,
                                    energy,
                                ),
                                "r_task_group_within_task": self._safe_ratio(
                                    ss_group,
                                    ss_task,
                                ),
                                "r_individual_task_within_task": self._safe_ratio(
                                    ss_individual,
                                    ss_task,
                                ),
                                "rms_centered": self._rms(
                                    energy,
                                    n_cells,
                                    n_coordinates,
                                ),
                                "rms_task": self._rms(
                                    ss_task,
                                    n_cells,
                                    n_coordinates,
                                ),
                                "rms_method": self._rms(
                                    ss_method,
                                    n_cells,
                                    n_coordinates,
                                ),
                                "rms_interaction": self._rms(
                                    ss_interaction,
                                    n_cells,
                                    n_coordinates,
                                ),
                            }
                        )

                output = self._write_csv(
                    self._analysis_save_dir(
                        "sq4_global_organization",
                        representation,
                        scope,
                    )
                    / "global.csv",
                    tuple(records[0]),
                    records,
                )

                paths[representation][scope] = {"global": output}

        return paths

    # ==================================================================
    # SQ5: CHECKPOINT / PREDICTION HELPERS
    # ==================================================================

    def _get_runs(
        self,
        dataset_name,
        method_name,
    ):
        key = (
            dataset_name,
            method_name,
        )

        if key in self._runs_cache:
            return self._runs_cache[key]

        runs = []

        for path in self._method_dir(
            dataset_name,
            method_name,
        ).iterdir():
            if not path.is_dir():
                continue

            config_path = path / "config.json"

            model_path = path / "model.pt"

            accuracy_path = path / "test_acc_eval.json"

            if not (
                config_path.is_file()
                and model_path.is_file()
                and accuracy_path.is_file()
            ):
                continue

            with config_path.open() as f:
                config = json.load(f)

            with accuracy_path.open() as f:
                accuracy = float(json.load(f)["accuracy"]["top1"])

            runs.append(
                {
                    "seed": int(config["run_seed"]),
                    "path": path,
                    "config": config,
                    "accuracy": accuracy,
                }
            )

        if not runs:
            raise FileNotFoundError(
                "No completed runs found for " f"{dataset_name}/" f"{method_name}"
            )

        runs.sort(key=lambda x: x["seed"])

        self._runs_cache[key] = runs

        return runs

    def _prediction_cache_path(
        self,
        dataset_name,
        method_name,
        seed,
    ):
        path = (
            self.save_dir
            / "prediction_cache"
            / dataset_name
            / method_name
            / f"seed-{seed}.npz"
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path

    def _get_seed_logits(
        self,
        dataset_name,
        method_name,
        run,
    ):
        cache = self._prediction_cache_path(
            dataset_name,
            method_name,
            run["seed"],
        )

        if cache.is_file():
            with np.load(
                cache,
                allow_pickle=False,
            ) as data:
                return (
                    data["logits"],
                    data["labels"],
                )

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

        model.load_from_path(
            str(run["path"] / "model.pt"),
            map_location="cpu",
        )

        model.to(self.device).eval()

        logits = []
        labels = []

        with torch.inference_mode():
            for x, y in self._progress(
                loader,
                desc=(f"Predict {dataset_name}/{method_name}/" f"seed-{run['seed']}"),
                total=len(loader),
            ):
                logits.append(model(x.to(self.device)).cpu())

                labels.append(y.cpu())

        logits = torch.cat(logits).numpy().astype(np.float32)

        labels = torch.cat(labels).numpy()

        np.savez(
            cache,
            logits=logits,
            labels=labels,
        )

        del model
        del dataset
        del loader

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (
            logits,
            labels,
        )

    def _get_method_prediction(
        self,
        dataset_name,
        method_name,
    ):
        probabilities = []
        accuracies = []
        labels = None

        for run in self._get_runs(
            dataset_name,
            method_name,
        ):
            logits, y = self._get_seed_logits(
                dataset_name,
                method_name,
                run,
            )

            if labels is None:
                labels = y

            elif not np.array_equal(
                labels,
                y,
            ):
                raise RuntimeError("Test ordering mismatch " f"for {dataset_name}")

            values = logits.astype(np.float64)

            values -= values.max(
                axis=1,
                keepdims=True,
            )

            probs = np.exp(values)

            probs /= probs.sum(
                axis=1,
                keepdims=True,
            )

            probabilities.append(probs)

            accuracies.append(run["accuracy"])

        if len(probabilities) > 1:
            seed_noise = float(
                np.mean(
                    [
                        self._jsd(
                            probabilities[i],
                            probabilities[j],
                        )
                        for i, j in itertools.combinations(
                            range(len(probabilities)),
                            2,
                        )
                    ]
                )
            )

        else:
            seed_noise = np.nan

        return {
            "probabilities": np.mean(
                probabilities,
                axis=0,
            ),
            "accuracy": float(np.mean(accuracies)),
            "seed_noise": seed_noise,
            "n_seeds": len(probabilities),
            "labels": labels,
        }

    def _get_full_behavior(
        self,
        dataset_name,
    ):
        if dataset_name in self._full_behavior_cache:
            return self._full_behavior_cache[dataset_name]

        methods = self.METHODS

        predictions = {
            method: (
                self._get_method_prediction(
                    dataset_name,
                    method,
                )
            )
            for method in methods
        }

        jsd = np.zeros(
            (
                len(methods),
                len(methods),
            ),
            dtype=float,
        )

        for i, j in itertools.combinations(
            range(len(methods)),
            2,
        ):
            value = self._jsd(
                predictions[methods[i]]["probabilities"],
                predictions[methods[j]]["probabilities"],
            )

            jsd[
                i,
                j,
            ] = value

            jsd[
                j,
                i,
            ] = value

        result = {
            "methods": methods,
            "jsd": jsd,
            "accuracy": np.array(
                [predictions[method]["accuracy"] for method in methods]
            ),
            "seed_noise": np.array(
                [predictions[method]["seed_noise"] for method in methods]
            ),
            "n_seeds": np.array([predictions[method]["n_seeds"] for method in methods]),
        }

        self._full_behavior_cache[dataset_name] = result

        return result

    def _get_behavior(
        self,
        dataset_name,
        scope,
    ):
        key = (
            dataset_name,
            scope,
        )

        if key in self._behavior_cache:
            return self._behavior_cache[key]

        full = self._get_full_behavior(dataset_name)

        wanted = (
            self.REFERENCE_METHOD,
            *self._scope_methods(scope),
        )

        indices = [full["methods"].index(method) for method in wanted]

        result = {
            "methods": wanted,
            "jsd": full["jsd"][
                np.ix_(
                    indices,
                    indices,
                )
            ],
            "accuracy": full["accuracy"][indices],
            "seed_noise": full["seed_noise"][indices],
            "n_seeds": full["n_seeds"][indices],
        }

        self._behavior_cache[key] = result

        return result

    # ==================================================================
    # SQ5
    # ==================================================================

    def analyze_prediction_alignment(
        self,
    ):
        """
        SQ5

        A) Functional method geometry vs predictive JSD geometry.
        B) Task method-conditioning vs predictive diversity.
        C) LP-referenced functional movement vs predictive movement.

        Pairwise functional distance is computed only once in raw log
        space because raw and LP-delta pairwise distances are identical.
        """

        all_paths = {}

        for (
            scope,
            methods,
        ) in self.METHOD_SCOPES.items():
            self._log(f"\nSQ5 [{scope}]")

            output_dir = self._analysis_save_dir(
                "sq5_prediction_alignment",
                scope,
            )

            pair_records = []
            rsa_records = []

            diversity_records = []
            diversity_corr_records = []

            delta_records = []
            delta_rsa_records = []

            noise_records = []

            # ----------------------------------------------------------
            # SQ5-A + SQ5-C
            # ----------------------------------------------------------

            for (
                task_index,
                task,
            ) in self._progress(
                enumerate(self.DATASETS),
                desc=f"SQ5-A/C {scope}",
                total=len(self.DATASETS),
            ):
                behavior = self._get_behavior(
                    task,
                    scope,
                )

                # Index 0 is LP.
                prediction_matrix = behavior["jsd"][
                    1:,
                    1:,
                ]

                accuracies = behavior["accuracy"][1:]

                prediction_vector = self._upper(prediction_matrix)

                accuracy_matrix = np.abs(accuracies[:, None] - accuracies[None, :])

                accuracy_vector = self._upper(accuracy_matrix)

                for index, method in enumerate(behavior["methods"]):
                    noise_records.append(
                        {
                            "task": task,
                            "method": method,
                            "n_seeds": int(behavior["n_seeds"][index]),
                            "mean_seed_jsd": behavior["seed_noise"][index],
                            "mean_accuracy": behavior["accuracy"][index],
                        }
                    )

                for signature in self.RQ1_SIGNATURES:
                    profiles = self._get_log_signature_profiles(signature)

                    for component in self.COMPONENTS:
                        # ----------------------------------------------
                        # Pairwise functional geometry
                        # ----------------------------------------------

                        x = np.stack(
                            [
                                profiles[method][component][task_index].reshape(-1)
                                for method in methods
                            ]
                        )

                        functional_matrix = self._pairwise_rms_distance(x)

                        functional_vector = self._upper(functional_matrix)

                        for i, j in itertools.combinations(
                            range(len(methods)),
                            2,
                        ):
                            pair_records.append(
                                {
                                    "task": task,
                                    "signature": signature,
                                    "component": component,
                                    "method_a": methods[i],
                                    "method_b": methods[j],
                                    "functional_distance": functional_matrix[
                                        i,
                                        j,
                                    ],
                                    "prediction_jsd": prediction_matrix[
                                        i,
                                        j,
                                    ],
                                    "accuracy_difference": abs(
                                        accuracies[i] - accuracies[j]
                                    ),
                                }
                            )

                        rsa_records.append(
                            {
                                "task": task,
                                "signature": signature,
                                "component": component,
                                "is_core": (component in self.CORE_COMPONENTS),
                                "rho": self._spearman(
                                    functional_vector,
                                    prediction_vector,
                                ),
                                "p_permutation": (
                                    self._rsa_permutation_p(
                                        functional_matrix,
                                        prediction_matrix,
                                    )
                                ),
                                "partial_rho_accuracy": (
                                    self._partial_spearman(
                                        functional_vector,
                                        prediction_vector,
                                        accuracy_vector,
                                    )
                                ),
                                "partial_p_permutation_accuracy": (
                                    self._rsa_permutation_p(
                                        functional_matrix,
                                        prediction_matrix,
                                        accuracy_matrix,
                                    )
                                ),
                            }
                        )

                        # ----------------------------------------------
                        # LP-referenced movement
                        # ----------------------------------------------

                        reference = profiles[self.REFERENCE_METHOD][component][
                            task_index
                        ].reshape(-1)

                        functional_to_lp = np.sqrt(
                            np.mean(
                                (x - reference) ** 2,
                                axis=1,
                            )
                        )

                        prediction_to_lp = behavior["jsd"][
                            0,
                            1:,
                        ]

                        for (
                            method_index,
                            method,
                        ) in enumerate(methods):
                            delta_records.append(
                                {
                                    "task": task,
                                    "method": method,
                                    "signature": signature,
                                    "component": component,
                                    "functional_distance_to_lp": functional_to_lp[
                                        method_index
                                    ],
                                    "prediction_jsd_to_lp": prediction_to_lp[
                                        method_index
                                    ],
                                    "accuracy_gain_over_lp": (
                                        accuracies[method_index]
                                        - behavior["accuracy"][0]
                                    ),
                                }
                            )

                        delta_rsa_records.append(
                            {
                                "task": task,
                                "signature": signature,
                                "component": component,
                                "is_core": (component in self.CORE_COMPONENTS),
                                "rho": self._spearman(
                                    functional_to_lp,
                                    prediction_to_lp,
                                ),
                            }
                        )

            # ----------------------------------------------------------
            # SQ5-B
            # ----------------------------------------------------------

            for representation in self.REPRESENTATIONS:
                for signature in self.RQ1_SIGNATURES:
                    for component in self._progress(
                        self.COMPONENTS,
                        desc=f"SQ5-B {scope} {representation} {signature}",
                        total=len(self.COMPONENTS),
                    ):
                        organization = self._get_dataset_organization(
                            scope,
                            representation,
                            signature,
                            component,
                        )

                        method_conditioning = []
                        prediction_diversity = []

                        for task in self.DATASETS:
                            task_index = self._dataset_index[task]

                            behavior = self._get_behavior(
                                task,
                                scope,
                            )

                            prediction_matrix = behavior["jsd"][
                                1:,
                                1:,
                            ]

                            mean_jsd = float(self._upper(prediction_matrix).mean())

                            r_method_conditioned = organization[task_index][
                                "r_method_conditioned"
                            ]

                            method_conditioning.append(r_method_conditioned)

                            prediction_diversity.append(mean_jsd)

                            diversity_records.append(
                                {
                                    "representation": representation,
                                    "task": task,
                                    "signature": signature,
                                    "component": component,
                                    "is_core": (component in self.CORE_COMPONENTS),
                                    "r_method_conditioned": r_method_conditioned,
                                    "mean_prediction_jsd": mean_jsd,
                                }
                            )

                        correlation = spearmanr(
                            method_conditioning,
                            prediction_diversity,
                        )

                        diversity_corr_records.append(
                            {
                                "representation": representation,
                                "signature": signature,
                                "component": component,
                                "is_core": (component in self.CORE_COMPONENTS),
                                "rho": correlation.statistic,
                                "p_value": correlation.pvalue,
                            }
                        )

            files = {
                "pairwise_distances": self._write_csv(
                    output_dir / "pairwise_distances.csv",
                    tuple(pair_records[0]),
                    pair_records,
                ),
                "rsa_by_task": self._write_csv(
                    output_dir / "rsa_by_task.csv",
                    tuple(rsa_records[0]),
                    rsa_records,
                ),
                "dataset_prediction_diversity": self._write_csv(
                    output_dir / "dataset_prediction_diversity.csv",
                    tuple(diversity_records[0]),
                    diversity_records,
                ),
                "dataset_diversity_correlations": self._write_csv(
                    output_dir / "dataset_diversity_correlations.csv",
                    tuple(diversity_corr_records[0]),
                    diversity_corr_records,
                ),
                "delta_to_lp": self._write_csv(
                    output_dir / "delta_to_lp.csv",
                    tuple(delta_records[0]),
                    delta_records,
                ),
                "delta_rsa_by_task": self._write_csv(
                    output_dir / "delta_rsa_by_task.csv",
                    tuple(delta_rsa_records[0]),
                    delta_rsa_records,
                ),
                "seed_prediction_noise": self._write_csv(
                    output_dir / "seed_prediction_noise.csv",
                    tuple(noise_records[0]),
                    noise_records,
                ),
            }

            all_paths[scope] = files

        return all_paths
