import csv
import os
from pathlib import Path

import numpy as np


class FunctionalReorganizationAnalyzer:
    """
    Analyze task- and adaptation-dependent functional organization.

    RQ1 is formulated as a hierarchy of exact Euclidean sums-of-squares
    decompositions.

    The same analyses are performed for two representations:

        raw:
            z_tj = log(F_tj + eps)

        delta:
            d_tj = log(F_tj + eps) - log(F_t,LP + eps)

    Both representations use the same adaptive methods and the same
    task x method decomposition.

    Public methods correspond one-to-one with the four RQ1 subquestions.
    Each public analysis is independently runnable and may reuse privately
    cached intermediate representations.
    """

    # ======================================================================
    # DATASETS
    # ======================================================================

    NATURAL = (
        "vtab_caltech101",
        "vtab_cifar100",
        "vtab_dtd",
        "vtab_flowers102",
        "vtab_pets",
        "vtab_sun397",
        "vtab_svhn",
    )

    REMOTE_SENSING = (
        "vtab_eurosat",
        "vtab_resisc45",
    )

    MEDICAL = (
        "vtab_patch_camelyon",
        "vtab_retinopathy",
    )

    SPECIALIZED = (
        *REMOTE_SENSING,
        *MEDICAL,
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

    # ======================================================================
    # TASK HIERARCHY
    # ======================================================================

    OFFICIAL_TASK_GROUPS = {
        "natural": NATURAL,
        "specialized": SPECIALIZED,
        "structured": STRUCTURED,
    }

    SPECIALIZED_SUBGROUPS = {
        "remote_sensing": REMOTE_SENSING,
        "medical": MEDICAL,
    }

    # Used by SQ3 for independent conditional analyses.
    # remote_sensing and medical are nested inside specialized, but this
    # is not a problem because each task set is analyzed independently.
    TASK_SETS = {
        "natural": NATURAL,
        "specialized": SPECIALIZED,
        "structured": STRUCTURED,
        "remote_sensing": REMOTE_SENSING,
        "medical": MEDICAL,
    }

    # ======================================================================
    # METHODS
    # ======================================================================

    REFERENCE_METHOD = "linear_probe"

    ADAPTATION_METHODS = (
        "full_ft",
        "vpt_shallow",
        "vpt_deep",
        "lora",
        "ssf",
        "adaptformer",
    )

    METHODS = (
        REFERENCE_METHOD,
        *ADAPTATION_METHODS,
    )

    # ======================================================================
    # REPRESENTATIONS
    # ======================================================================

    REPRESENTATIONS = (
        "raw",
        "delta",
    )

    # ======================================================================
    # SIGNATURES
    # ======================================================================

    PRIMARY_SIGNATURES = (
        "F_m",
        "F_e",
    )

    PARTICIPATION_SIGNATURE = "P"

    AVAILABLE_SIGNATURES = (
        *PRIMARY_SIGNATURES,
        PARTICIPATION_SIGNATURE,
    )

    # ======================================================================
    # COMMON BACKBONE COMPONENTS
    # ======================================================================

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

    # ======================================================================
    # EXPERIMENT LAYOUT
    # ======================================================================

    EVAL_EXPERIMENT_NAME = "top_1_acc_eval"
    SIGNATURE_FILENAME = "functional_signatures.npz"

    # ======================================================================
    # INITIALIZATION
    # ======================================================================

    def __init__(
        self,
        backbone_name: str = "vit_b16",
        eps: float = 1e-12,
        verbose: bool = True,
    ):
        if eps <= 0:
            raise ValueError("eps must be positive")

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

        # Populated lazily from the NPZ files.
        self._component_shapes = {}

        # signature -> method -> component -> [T, *component_shape]
        self._profile_cache = {}

        # signature -> method -> component -> [T, *component_shape]
        self._log_profile_cache = {}

        # signature -> adaptive_method -> component -> [T, *component_shape]
        self._delta_profile_cache = {}

        # (representation, signature, component)
        #     -> global task x method decomposition
        self._decomposition_cache = {}

    # ======================================================================
    # COMMON HELPERS
    # ======================================================================

    def _log(self, message):
        if self.verbose:
            print(message, flush=True)

    def _signature_path(
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
            / self.SIGNATURE_FILENAME
        )

    def _analysis_save_dir(
        self,
        analysis_name,
        representation,
    ):
        if representation not in self.REPRESENTATIONS:
            raise ValueError(
                f"Unknown representation '{representation}'. "
                f"Expected one of {self.REPRESENTATIONS}."
            )

        return self.save_dir / analysis_name / representation

    @staticmethod
    def _npz_scalar_string(value):
        value = np.asarray(value)

        if value.ndim == 0:
            return str(value.item())

        return str(value)

    def _load_profiles(
        self,
        signature_names,
    ):
        """
        Load final dataset x method signatures.

        Returned structure:
            signature
                -> method
                    -> component
                        -> [T, *component_shape]
        """
        signature_names = tuple(signature_names)

        unknown = set(signature_names) - set(self.AVAILABLE_SIGNATURES)

        if unknown:
            raise ValueError(f"Unknown signatures: {sorted(unknown)}")

        loaded = {
            signature_name: {
                method_name: {component: [] for component in self.COMPONENTS}
                for method_name in self.METHODS
            }
            for signature_name in signature_names
        }

        for method_name in self.METHODS:
            self._log(f"Loading {method_name}...")

            for dataset_name in self.DATASETS:
                path = self._signature_path(
                    dataset_name,
                    method_name,
                )

                if not path.is_file():
                    raise FileNotFoundError(f"Functional signatures not found: {path}")

                with np.load(
                    path,
                    allow_pickle=False,
                ) as data:
                    if "dataset_name" in data:
                        stored_dataset = self._npz_scalar_string(data["dataset_name"])

                        if stored_dataset != dataset_name:
                            raise ValueError(
                                f"Dataset mismatch in {path}: "
                                f"expected '{dataset_name}', "
                                f"found '{stored_dataset}'."
                            )

                    if "backbone_name" in data:
                        stored_backbone = self._npz_scalar_string(data["backbone_name"])

                        if stored_backbone != self.backbone_name:
                            raise ValueError(
                                f"Backbone mismatch in {path}: "
                                f"expected '{self.backbone_name}', "
                                f"found '{stored_backbone}'."
                            )

                    for signature_name in signature_names:
                        for component in self.COMPONENTS:
                            key = f"{signature_name}" f"__{component}"

                            if key not in data:
                                raise KeyError(f"Missing '{key}' in {path}.")

                            value = np.asarray(
                                data[key],
                                dtype=np.float64,
                            )

                            if not np.all(np.isfinite(value)):
                                raise ValueError(
                                    f"Non-finite values in " f"'{key}' from {path}."
                                )

                            if signature_name in (
                                "F_m",
                                "F_e",
                                "P",
                            ):
                                if np.any(value < 0):
                                    raise ValueError(
                                        f"Negative values in "
                                        f"nonnegative signature "
                                        f"'{key}' from {path}."
                                    )

                            expected_shape = self._component_shapes.get(component)

                            if expected_shape is None:
                                self._component_shapes[component] = value.shape

                            elif value.shape != expected_shape:
                                raise ValueError(
                                    f"Inconsistent shape for "
                                    f"component '{component}': "
                                    f"{value.shape} in {path}, "
                                    f"expected {expected_shape}."
                                )

                            loaded[signature_name][method_name][component].append(value)

        for signature_name in signature_names:
            for method_name in self.METHODS:
                for component in self.COMPONENTS:
                    loaded[signature_name][method_name][component] = np.stack(
                        loaded[signature_name][method_name][component],
                        axis=0,
                    )

        return loaded

    def _get_profiles(
        self,
        signature_names,
    ):
        """
        Retrieve signatures, loading only missing signatures.
        """
        signature_names = tuple(signature_names)

        missing = [
            signature_name
            for signature_name in signature_names
            if signature_name not in self._profile_cache
        ]

        if missing:
            loaded = self._load_profiles(missing)

            self._profile_cache.update(loaded)

        return {
            signature_name: (self._profile_cache[signature_name])
            for signature_name in signature_names
        }

    def _get_log_profiles(
        self,
        signature_name,
    ):
        """
        Log-transform one nonnegative signature.

        z_tj = log(F_tj + eps)
        """
        if signature_name not in self._log_profile_cache:
            profiles = self._get_profiles((signature_name,))[signature_name]

            self._log_profile_cache[signature_name] = {
                method_name: {
                    component: np.log(profiles[method_name][component] + self.eps)
                    for component in self.COMPONENTS
                }
                for method_name in self.METHODS
            }

        return self._log_profile_cache[signature_name]

    def _get_delta_profiles(
        self,
        signature_name,
    ):
        """
        Construct LP-referenced log reorganization profiles.

        d_tj
            = log(F_tj + eps)
            - log(F_t,LP + eps)
        """
        if signature_name not in self._delta_profile_cache:
            log_profiles = self._get_log_profiles(signature_name)

            reference = log_profiles[self.REFERENCE_METHOD]

            self._delta_profile_cache[signature_name] = {
                method_name: {
                    component: (
                        log_profiles[method_name][component] - reference[component]
                    )
                    for component in self.COMPONENTS
                }
                for method_name in self.ADAPTATION_METHODS
            }

        return self._delta_profile_cache[signature_name]

    def _get_task_method_values(
        self,
        representation,
        signature_name,
        component,
    ):
        """
        Return values with shape:

            [T, J, *component_shape]

        where J contains only the adaptive methods.

        raw:
            log(F_tj + eps)

        delta:
            log(F_tj + eps)
            - log(F_t,LP + eps)
        """
        if representation not in self.REPRESENTATIONS:
            raise ValueError(f"Unknown representation " f"'{representation}'.")

        if signature_name not in self.PRIMARY_SIGNATURES:
            raise ValueError(
                f"RQ1 decomposition only supports "
                f"primary signatures, found "
                f"'{signature_name}'."
            )

        if component not in self.COMPONENTS:
            raise ValueError(f"Unknown component '{component}'.")

        if representation == "raw":
            profiles = self._get_log_profiles(signature_name)

        else:
            profiles = self._get_delta_profiles(signature_name)

        return np.stack(
            [
                profiles[method_name][component]
                for method_name in self.ADAPTATION_METHODS
            ],
            axis=1,
        )

    @staticmethod
    def _task_method_decomposition(
        values,
    ):
        """
        Exact balanced task x method decomposition.

        values has shape:

            [T, J, *component_shape]

        x_tj = mu + alpha_t + beta_j + gamma_tj
        """
        values = np.asarray(
            values,
            dtype=np.float64,
        )

        if values.ndim < 3:
            raise ValueError("values must have shape " "[T, J, *component_shape]")

        if not np.all(np.isfinite(values)):
            raise ValueError("values contain non-finite entries")

        n_tasks = values.shape[0]
        n_methods = values.shape[1]

        if n_tasks < 2:
            raise ValueError("At least two tasks are required")

        if n_methods < 2:
            raise ValueError("At least two methods are required")

        mu = values.mean(axis=(0, 1))

        alpha = values.mean(axis=1) - mu

        beta = values.mean(axis=0) - mu

        gamma = values - mu - alpha[:, None] - beta[None, :]

        centered = values - mu

        ss_task = n_methods * np.square(alpha).sum()

        ss_method = n_tasks * np.square(beta).sum()

        ss_interaction = np.square(gamma).sum()

        ss_total = np.square(centered).sum()

        FunctionalReorganizationAnalyzer._check_ss_identity(
            ss_total,
            (
                ss_task,
                ss_method,
                ss_interaction,
            ),
            "global task-method decomposition",
        )

        r_task = FunctionalReorganizationAnalyzer._safe_ratio(
            ss_task,
            ss_total,
        )

        r_method = FunctionalReorganizationAnalyzer._safe_ratio(
            ss_method,
            ss_total,
        )

        r_interaction = FunctionalReorganizationAnalyzer._safe_ratio(
            ss_interaction,
            ss_total,
        )

        return {
            "mu": mu,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "ss_task": float(ss_task),
            "ss_method": float(ss_method),
            "ss_interaction": float(ss_interaction),
            "ss_total": float(ss_total),
            "r_task": float(r_task),
            "r_method": float(r_method),
            "r_interaction": float(r_interaction),
        }

    def _get_decomposition(
        self,
        representation,
        signature_name,
        component,
    ):
        key = (
            representation,
            signature_name,
            component,
        )

        if key not in self._decomposition_cache:
            values = self._get_task_method_values(
                representation,
                signature_name,
                component,
            )

            self._decomposition_cache[key] = self._task_method_decomposition(values)

        return self._decomposition_cache[key]

    @staticmethod
    def _safe_ratio(
        numerator,
        denominator,
    ):
        numerator = float(numerator)

        denominator = float(denominator)

        if denominator <= np.finfo(np.float64).eps:
            return np.nan

        return numerator / denominator

    @staticmethod
    def _rms_from_ss(
        ss,
        n_cells,
        n_coordinates,
    ):
        """
        RMS contribution per functional coordinate over
        the cells entering the decomposition.
        """
        ss = float(ss)

        denominator = int(n_cells) * int(n_coordinates)

        if denominator <= 0:
            raise ValueError("RMS denominator must be positive")

        return float(np.sqrt(max(ss, 0.0) / denominator))

    @staticmethod
    def _check_ss_identity(
        total,
        parts,
        label,
    ):
        total = float(total)

        reconstructed = float(sum(float(part) for part in parts))

        if not np.isclose(
            total,
            reconstructed,
            rtol=1e-10,
            atol=1e-12
            * max(
                1.0,
                abs(total),
            ),
        ):
            raise RuntimeError(
                f"{label} failed SS identity: "
                f"total={total}, "
                f"decomposed={reconstructed}."
            )

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
                row = dict(record)

                for key, value in row.items():
                    if isinstance(
                        value,
                        (
                            float,
                            np.floating,
                        ),
                    ):
                        row[key] = value if np.isfinite(value) else "NA"

                writer.writerow(row)

        return path

    def _dataset_indices(
        self,
        dataset_names,
    ):
        index = {name: i for i, name in enumerate(self.DATASETS)}

        return np.array(
            [index[name] for name in dataset_names],
            dtype=np.int64,
        )

    # ======================================================================
    # SQ1: GLOBAL DETERMINANTS
    # ======================================================================

    def analyze_task_method_decomposition(
        self,
    ):
        """
        SQ1
        ---
        Across all tasks and adaptive methods, how much functional
        heterogeneity is attributable to task, method, and the
        task x method interaction?

        The exact same analysis is run for raw and delta profiles.
        """
        all_paths = {}

        n_tasks = len(self.DATASETS)

        n_methods = len(self.ADAPTATION_METHODS)

        for representation in self.REPRESENTATIONS:
            output_dir = self._analysis_save_dir(
                "sq1_global_determinants",
                representation,
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            records = []
            effects = {}

            for signature_name in self.PRIMARY_SIGNATURES:
                self._log(
                    "\n"
                    "================================\n"
                    f"SQ1 [{representation}]: "
                    f"{signature_name}\n"
                    "================================"
                )

                for component in self.COMPONENTS:
                    self._log(f"  {component}")

                    result = self._get_decomposition(
                        representation,
                        signature_name,
                        component,
                    )

                    n_coordinates = int(np.prod(result["mu"].shape))

                    common_rms = float(
                        np.linalg.norm(result["mu"].reshape(-1))
                        / np.sqrt(n_coordinates)
                    )

                    heterogeneity_rms = self._rms_from_ss(
                        result["ss_total"],
                        n_tasks * n_methods,
                        n_coordinates,
                    )

                    records.append(
                        {
                            "signature": signature_name,
                            "component": component,
                            "n_coordinates": n_coordinates,
                            "ss_task": result["ss_task"],
                            "ss_method": result["ss_method"],
                            "ss_interaction": result["ss_interaction"],
                            "ss_total": result["ss_total"],
                            "r_task": result["r_task"],
                            "r_method": result["r_method"],
                            "r_interaction": result["r_interaction"],
                            "rms_common_shift": (common_rms),
                            "rms_heterogeneity": (heterogeneity_rms),
                        }
                    )

                    prefix = f"{signature_name}" f"__{component}"

                    effects[f"{prefix}__mu"] = result["mu"]

                    effects[f"{prefix}__alpha"] = result["alpha"]

                    effects[f"{prefix}__beta"] = result["beta"]

                    effects[f"{prefix}__gamma"] = result["gamma"]

            csv_path = self._write_csv(
                output_dir / "decomposition.csv",
                (
                    "signature",
                    "component",
                    "n_coordinates",
                    "ss_task",
                    "ss_method",
                    "ss_interaction",
                    "ss_total",
                    "r_task",
                    "r_method",
                    "r_interaction",
                    "rms_common_shift",
                    "rms_heterogeneity",
                ),
                records,
            )

            effects_path = output_dir / "effects.npz"

            np.savez_compressed(
                effects_path,
                representation=np.array(representation),
                backbone_name=np.array(self.backbone_name),
                reference_method=np.array(self.REFERENCE_METHOD),
                datasets=np.array(self.DATASETS),
                methods=np.array(self.ADAPTATION_METHODS),
                signatures=np.array(self.PRIMARY_SIGNATURES),
                components=np.array(self.COMPONENTS),
                eps=np.array(self.eps),
                **effects,
            )

            self._log(
                f"\nSaved SQ1 "
                f"[{representation}] results:"
                f"\n  {csv_path}"
                f"\n  {effects_path}"
            )

            all_paths[representation] = {
                "decomposition": csv_path,
                "effects": effects_path,
            }

        return all_paths

    # ======================================================================
    # SQ2: HIERARCHICAL ORGANIZATION OF TASK EFFECTS
    # ======================================================================

    def analyze_task_hierarchy(
        self,
    ):
        """
        SQ2
        ---
        When the task main effect matters, how much is organized at the
        official VTAB group level versus the individual-task level?

        Also decomposes Specialized within-group task variation into
        Remote Sensing vs Medical variation and residual within-subdomain
        variation.

        The same analysis is run for raw and delta profiles.
        """
        all_paths = {}

        n_methods = len(self.ADAPTATION_METHODS)

        n_tasks = len(self.DATASETS)

        official_group_indices = {
            name: self._dataset_indices(datasets)
            for name, datasets in self.OFFICIAL_TASK_GROUPS.items()
        }

        specialized_indices = self._dataset_indices(self.SPECIALIZED)

        specialized_subgroup_indices = {
            name: self._dataset_indices(datasets)
            for name, datasets in self.SPECIALIZED_SUBGROUPS.items()
        }

        for representation in self.REPRESENTATIONS:
            output_dir = self._analysis_save_dir(
                "sq2_task_hierarchy",
                representation,
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            hierarchy_records = []
            specialized_records = []

            for signature_name in self.PRIMARY_SIGNATURES:
                self._log(
                    "\n"
                    "================================\n"
                    f"SQ2 [{representation}]: "
                    f"{signature_name}\n"
                    "================================"
                )

                for component in self.COMPONENTS:
                    self._log(f"  {component}")

                    result = self._get_decomposition(
                        representation,
                        signature_name,
                        component,
                    )

                    alpha = result["alpha"]

                    n_coordinates = int(np.prod(alpha.shape[1:]))

                    # --------------------------------------------------
                    # Official VTAB hierarchy:
                    #
                    # SS_task
                    #   = SS_task_group
                    #   + SS_individual_task
                    # --------------------------------------------------

                    ss_group = 0.0
                    ss_individual = 0.0

                    for indices in official_group_indices.values():
                        group_alpha = alpha[indices].mean(axis=0)

                        ss_group += (
                            n_methods
                            * len(indices)
                            * float(np.square(group_alpha).sum())
                        )

                        deviations = alpha[indices] - group_alpha

                        ss_individual += n_methods * float(np.square(deviations).sum())

                    self._check_ss_identity(
                        result["ss_task"],
                        (
                            ss_group,
                            ss_individual,
                        ),
                        (
                            f"SQ2 task hierarchy "
                            f"{representation}/"
                            f"{signature_name}/"
                            f"{component}"
                        ),
                    )

                    hierarchy_records.append(
                        {
                            "signature": signature_name,
                            "component": component,
                            "n_coordinates": (n_coordinates),
                            "ss_task": result["ss_task"],
                            "ss_task_group": (ss_group),
                            "ss_individual_task": (ss_individual),
                            "r_task_group_within_task": (
                                self._safe_ratio(
                                    ss_group,
                                    result["ss_task"],
                                )
                            ),
                            "r_individual_task_within_task": (
                                self._safe_ratio(
                                    ss_individual,
                                    result["ss_task"],
                                )
                            ),
                            "rms_task": self._rms_from_ss(
                                result["ss_task"],
                                n_tasks * n_methods,
                                n_coordinates,
                            ),
                            "rms_task_group": (
                                self._rms_from_ss(
                                    ss_group,
                                    n_tasks * n_methods,
                                    n_coordinates,
                                )
                            ),
                            "rms_individual_task": (
                                self._rms_from_ss(
                                    ss_individual,
                                    n_tasks * n_methods,
                                    n_coordinates,
                                )
                            ),
                        }
                    )

                    # --------------------------------------------------
                    # Specialized hierarchy:
                    #
                    # SS_within_specialized
                    #   = SS_remote_vs_medical
                    #   + SS_within_subdomain
                    # --------------------------------------------------

                    alpha_specialized = alpha[specialized_indices]

                    specialized_mean = alpha_specialized.mean(axis=0)

                    ss_specialized = n_methods * float(
                        np.square(alpha_specialized - specialized_mean).sum()
                    )

                    ss_subdomain = 0.0
                    ss_within_subdomain = 0.0

                    for indices in specialized_subgroup_indices.values():
                        subgroup_alpha = alpha[indices]

                        subgroup_mean = subgroup_alpha.mean(axis=0)

                        ss_subdomain += (
                            n_methods
                            * len(indices)
                            * float(np.square(subgroup_mean - specialized_mean).sum())
                        )

                        ss_within_subdomain += n_methods * float(
                            np.square(subgroup_alpha - subgroup_mean).sum()
                        )

                    self._check_ss_identity(
                        ss_specialized,
                        (
                            ss_subdomain,
                            ss_within_subdomain,
                        ),
                        (
                            f"SQ2 specialized "
                            f"hierarchy "
                            f"{representation}/"
                            f"{signature_name}/"
                            f"{component}"
                        ),
                    )

                    n_specialized = len(self.SPECIALIZED)

                    specialized_records.append(
                        {
                            "signature": signature_name,
                            "component": component,
                            "n_coordinates": (n_coordinates),
                            "ss_within_specialized": (ss_specialized),
                            "ss_remote_vs_medical": (ss_subdomain),
                            "ss_within_subdomain": (ss_within_subdomain),
                            "r_remote_vs_medical_within_specialized": (
                                self._safe_ratio(
                                    ss_subdomain,
                                    ss_specialized,
                                )
                            ),
                            "r_within_subdomain_within_specialized": (
                                self._safe_ratio(
                                    ss_within_subdomain,
                                    ss_specialized,
                                )
                            ),
                            "rms_within_specialized": (
                                self._rms_from_ss(
                                    ss_specialized,
                                    n_specialized * n_methods,
                                    n_coordinates,
                                )
                            ),
                            "rms_remote_vs_medical": (
                                self._rms_from_ss(
                                    ss_subdomain,
                                    n_specialized * n_methods,
                                    n_coordinates,
                                )
                            ),
                            "rms_within_subdomain": (
                                self._rms_from_ss(
                                    ss_within_subdomain,
                                    n_specialized * n_methods,
                                    n_coordinates,
                                )
                            ),
                        }
                    )

            hierarchy_path = self._write_csv(
                output_dir / "task_hierarchy.csv",
                (
                    "signature",
                    "component",
                    "n_coordinates",
                    "ss_task",
                    "ss_task_group",
                    "ss_individual_task",
                    "r_task_group_within_task",
                    "r_individual_task_within_task",
                    "rms_task",
                    "rms_task_group",
                    "rms_individual_task",
                ),
                hierarchy_records,
            )

            specialized_path = self._write_csv(
                output_dir / "specialized_hierarchy.csv",
                (
                    "signature",
                    "component",
                    "n_coordinates",
                    "ss_within_specialized",
                    "ss_remote_vs_medical",
                    "ss_within_subdomain",
                    "r_remote_vs_medical_within_specialized",
                    "r_within_subdomain_within_specialized",
                    "rms_within_specialized",
                    "rms_remote_vs_medical",
                    "rms_within_subdomain",
                ),
                specialized_records,
            )

            self._log(
                f"\nSaved SQ2 "
                f"[{representation}] results:"
                f"\n  {hierarchy_path}"
                f"\n  {specialized_path}"
            )

            all_paths[representation] = {
                "task_hierarchy": (hierarchy_path),
                "specialized_hierarchy": (specialized_path),
            }

        return all_paths

    # ======================================================================
    # SQ3: TASK-CENTERED CONSISTENCY ACROSS METHODS
    # ======================================================================

    def analyze_task_consistency(
        self,
    ):
        """
        SQ3
        ---
        For each individual task or predefined task set, how much of its
        deviation from the grand pattern is shared across methods, and
        how much is method-conditioned?

        For a task set G:

            x_tj - mu
              = alpha_G
              + (alpha_t - alpha_G)
              + (beta_j + gamma_tj)

        For a single task, the within-set task term is identically zero.

        The same analysis is run for raw and delta profiles.
        """
        all_paths = {}

        n_methods = len(self.ADAPTATION_METHODS)

        dataset_index = {name: i for i, name in enumerate(self.DATASETS)}

        for representation in self.REPRESENTATIONS:
            output_dir = self._analysis_save_dir(
                "sq3_task_consistency",
                representation,
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            task_records = []
            group_records = []

            for signature_name in self.PRIMARY_SIGNATURES:
                self._log(
                    "\n"
                    "================================\n"
                    f"SQ3 [{representation}]: "
                    f"{signature_name}\n"
                    "================================"
                )

                # ------------------------------------------------------
                # Individual tasks.
                #
                # Ordering:
                # task -> component
                # ------------------------------------------------------

                for task_name in self.DATASETS:
                    t = dataset_index[task_name]

                    for component in self.COMPONENTS:
                        self._log(f"  {task_name}: " f"{component}")

                        result = self._get_decomposition(
                            representation,
                            signature_name,
                            component,
                        )

                        values = self._get_task_method_values(
                            representation,
                            signature_name,
                            component,
                        )

                        alpha = result["alpha"]

                        beta = result["beta"]

                        gamma = result["gamma"]

                        n_coordinates = int(np.prod(alpha.shape[1:]))

                        shared = alpha[t]

                        method_conditioned = beta + gamma[t]

                        ss_shared = n_methods * float(np.square(shared).sum())

                        ss_method_conditioned = float(
                            np.square(method_conditioned).sum()
                        )

                        ss_total = ss_shared + ss_method_conditioned

                        direct_total = float(np.square(values[t] - result["mu"]).sum())

                        self._check_ss_identity(
                            direct_total,
                            (
                                ss_shared,
                                ss_method_conditioned,
                            ),
                            (
                                f"SQ3 task "
                                f"{task_name} "
                                f"{representation}/"
                                f"{signature_name}/"
                                f"{component}"
                            ),
                        )

                        n_cells = n_methods

                        task_records.append(
                            {
                                "signature": signature_name,
                                "task": task_name,
                                "component": component,
                                "n_coordinates": (n_coordinates),
                                "ss_shared_task": (ss_shared),
                                "ss_method_conditioned": (ss_method_conditioned),
                                "ss_total": ss_total,
                                "r_shared_task": (
                                    self._safe_ratio(
                                        ss_shared,
                                        ss_total,
                                    )
                                ),
                                "r_method_conditioned": (
                                    self._safe_ratio(
                                        ss_method_conditioned,
                                        ss_total,
                                    )
                                ),
                                "rms_total": (
                                    self._rms_from_ss(
                                        ss_total,
                                        n_cells,
                                        n_coordinates,
                                    )
                                ),
                                "rms_shared_task": (
                                    self._rms_from_ss(
                                        ss_shared,
                                        n_cells,
                                        n_coordinates,
                                    )
                                ),
                                "rms_method_conditioned": (
                                    self._rms_from_ss(
                                        ss_method_conditioned,
                                        n_cells,
                                        n_coordinates,
                                    )
                                ),
                            }
                        )

                # ------------------------------------------------------
                # Task sets.
                #
                # Ordering:
                # group -> component
                #
                # Each set is analyzed independently.
                # ------------------------------------------------------

                for (
                    group_name,
                    task_names,
                ) in self.TASK_SETS.items():
                    indices = self._dataset_indices(task_names)

                    for component in self.COMPONENTS:
                        self._log(f"  {group_name}: " f"{component}")

                        result = self._get_decomposition(
                            representation,
                            signature_name,
                            component,
                        )

                        values = self._get_task_method_values(
                            representation,
                            signature_name,
                            component,
                        )

                        alpha = result["alpha"]

                        beta = result["beta"]

                        gamma = result["gamma"]

                        n_coordinates = int(np.prod(alpha.shape[1:]))

                        group_alpha = alpha[indices].mean(axis=0)

                        within = alpha[indices] - group_alpha

                        method_conditioned = beta[None, ...] + gamma[indices]

                        n_group_tasks = len(indices)

                        n_cells = n_group_tasks * n_methods

                        ss_shared = (
                            n_group_tasks
                            * n_methods
                            * float(np.square(group_alpha).sum())
                        )

                        ss_within = n_methods * float(np.square(within).sum())

                        ss_method_conditioned = float(
                            np.square(method_conditioned).sum()
                        )

                        ss_total = ss_shared + ss_within + ss_method_conditioned

                        direct_total = float(
                            np.square(values[indices] - result["mu"]).sum()
                        )

                        self._check_ss_identity(
                            direct_total,
                            (
                                ss_shared,
                                ss_within,
                                ss_method_conditioned,
                            ),
                            (
                                f"SQ3 group "
                                f"{group_name} "
                                f"{representation}/"
                                f"{signature_name}/"
                                f"{component}"
                            ),
                        )

                        group_records.append(
                            {
                                "signature": signature_name,
                                "group": group_name,
                                "component": component,
                                "n_tasks": (n_group_tasks),
                                "n_coordinates": (n_coordinates),
                                "ss_shared_group": (ss_shared),
                                "ss_within_group_task": (ss_within),
                                "ss_method_conditioned": (ss_method_conditioned),
                                "ss_total": ss_total,
                                "r_shared_group": (
                                    self._safe_ratio(
                                        ss_shared,
                                        ss_total,
                                    )
                                ),
                                "r_within_group_task": (
                                    self._safe_ratio(
                                        ss_within,
                                        ss_total,
                                    )
                                ),
                                "r_method_conditioned": (
                                    self._safe_ratio(
                                        ss_method_conditioned,
                                        ss_total,
                                    )
                                ),
                                "rms_total": (
                                    self._rms_from_ss(
                                        ss_total,
                                        n_cells,
                                        n_coordinates,
                                    )
                                ),
                                "rms_shared_group": (
                                    self._rms_from_ss(
                                        ss_shared,
                                        n_cells,
                                        n_coordinates,
                                    )
                                ),
                                "rms_within_group_task": (
                                    self._rms_from_ss(
                                        ss_within,
                                        n_cells,
                                        n_coordinates,
                                    )
                                ),
                                "rms_method_conditioned": (
                                    self._rms_from_ss(
                                        ss_method_conditioned,
                                        n_cells,
                                        n_coordinates,
                                    )
                                ),
                            }
                        )

            tasks_path = self._write_csv(
                output_dir / "tasks.csv",
                (
                    "signature",
                    "task",
                    "component",
                    "n_coordinates",
                    "ss_shared_task",
                    "ss_method_conditioned",
                    "ss_total",
                    "r_shared_task",
                    "r_method_conditioned",
                    "rms_total",
                    "rms_shared_task",
                    "rms_method_conditioned",
                ),
                task_records,
            )

            groups_path = self._write_csv(
                output_dir / "groups.csv",
                (
                    "signature",
                    "group",
                    "component",
                    "n_tasks",
                    "n_coordinates",
                    "ss_shared_group",
                    "ss_within_group_task",
                    "ss_method_conditioned",
                    "ss_total",
                    "r_shared_group",
                    "r_within_group_task",
                    "r_method_conditioned",
                    "rms_total",
                    "rms_shared_group",
                    "rms_within_group_task",
                    "rms_method_conditioned",
                ),
                group_records,
            )

            self._log(
                f"\nSaved SQ3 "
                f"[{representation}] results:"
                f"\n  {tasks_path}"
                f"\n  {groups_path}"
            )

            all_paths[representation] = {
                "tasks": tasks_path,
                "groups": groups_path,
            }

        return all_paths

    # ======================================================================
    # SQ4: METHOD-CENTERED SPECIFICITY
    # ======================================================================

    def analyze_method_specificity(
        self,
    ):
        """
        SQ4
        ---
        For each adaptation method, how much of its deviation from the
        grand pattern is method-general versus organized by official
        VTAB task group or individual task identity within those groups?

        For fixed method j:

            x_tj - mu
                = beta_j + h_tj

            h_tj
                = alpha_t + gamma_tj

        and:

            SS_j
                = SS_method_general,j
                + SS_task_group,j
                + SS_individual_task,j

        The same analysis is run for raw and delta profiles.
        """
        all_paths = {}

        n_tasks = len(self.DATASETS)

        official_group_indices = {
            name: self._dataset_indices(datasets)
            for name, datasets in self.OFFICIAL_TASK_GROUPS.items()
        }

        for representation in self.REPRESENTATIONS:
            output_dir = self._analysis_save_dir(
                "sq4_method_specificity",
                representation,
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            records = []

            for signature_name in self.PRIMARY_SIGNATURES:
                self._log(
                    "\n"
                    "================================\n"
                    f"SQ4 [{representation}]: "
                    f"{signature_name}\n"
                    "================================"
                )

                # ------------------------------------------------------
                # Ordering:
                # method -> component
                # ------------------------------------------------------

                for (
                    j,
                    method_name,
                ) in enumerate(self.ADAPTATION_METHODS):
                    for component in self.COMPONENTS:
                        self._log(f"  {method_name}: " f"{component}")

                        result = self._get_decomposition(
                            representation,
                            signature_name,
                            component,
                        )

                        alpha = result["alpha"]

                        beta = result["beta"]

                        gamma = result["gamma"]

                        values = self._get_task_method_values(
                            representation,
                            signature_name,
                            component,
                        )

                        n_coordinates = int(np.prod(alpha.shape[1:]))

                        h = alpha + gamma[:, j]

                        ss_method_general = n_tasks * float(np.square(beta[j]).sum())

                        ss_task_group = 0.0
                        ss_individual_task = 0.0

                        for indices in official_group_indices.values():
                            group_h = h[indices].mean(axis=0)

                            ss_task_group += len(indices) * float(
                                np.square(group_h).sum()
                            )

                            ss_individual_task += float(
                                np.square(h[indices] - group_h).sum()
                            )

                        ss_total = (
                            ss_method_general + ss_task_group + ss_individual_task
                        )

                        direct_total = float(
                            np.square(values[:, j] - result["mu"]).sum()
                        )

                        self._check_ss_identity(
                            direct_total,
                            (
                                ss_method_general,
                                ss_task_group,
                                ss_individual_task,
                            ),
                            (
                                f"SQ4 method "
                                f"{method_name} "
                                f"{representation}/"
                                f"{signature_name}/"
                                f"{component}"
                            ),
                        )

                        records.append(
                            {
                                "signature": signature_name,
                                "method": method_name,
                                "component": component,
                                "n_coordinates": (n_coordinates),
                                "ss_method_general": (ss_method_general),
                                "ss_task_group": (ss_task_group),
                                "ss_individual_task": (ss_individual_task),
                                "ss_total": (ss_total),
                                "r_method_general": (
                                    self._safe_ratio(
                                        ss_method_general,
                                        ss_total,
                                    )
                                ),
                                "r_task_group": (
                                    self._safe_ratio(
                                        ss_task_group,
                                        ss_total,
                                    )
                                ),
                                "r_individual_task": (
                                    self._safe_ratio(
                                        ss_individual_task,
                                        ss_total,
                                    )
                                ),
                                "rms_total": (
                                    self._rms_from_ss(
                                        ss_total,
                                        n_tasks,
                                        n_coordinates,
                                    )
                                ),
                                "rms_method_general": (
                                    self._rms_from_ss(
                                        ss_method_general,
                                        n_tasks,
                                        n_coordinates,
                                    )
                                ),
                                "rms_task_group": (
                                    self._rms_from_ss(
                                        ss_task_group,
                                        n_tasks,
                                        n_coordinates,
                                    )
                                ),
                                "rms_individual_task": (
                                    self._rms_from_ss(
                                        ss_individual_task,
                                        n_tasks,
                                        n_coordinates,
                                    )
                                ),
                            }
                        )

            methods_path = self._write_csv(
                output_dir / "methods.csv",
                (
                    "signature",
                    "method",
                    "component",
                    "n_coordinates",
                    "ss_method_general",
                    "ss_task_group",
                    "ss_individual_task",
                    "ss_total",
                    "r_method_general",
                    "r_task_group",
                    "r_individual_task",
                    "rms_total",
                    "rms_method_general",
                    "rms_task_group",
                    "rms_individual_task",
                ),
                records,
            )

            self._log(
                f"\nSaved SQ4 " f"[{representation}] results:" f"\n  {methods_path}"
            )

            all_paths[representation] = {
                "methods": methods_path,
            }

        return all_paths
