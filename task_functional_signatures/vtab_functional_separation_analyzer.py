import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS
from tqdm.auto import tqdm


class VTABFunctionalSeparationAnalyzer:
    """Analyze raw and adaptation-delta VTAB functional signatures."""

    PROFILES = ("F", "P", "S")
    REFERENCE_MODEL = "linear_probe"

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
        "vtab_dsprites_location",
        "vtab_dsprites_orientation",
        "vtab_kitti_distance",
        "vtab_smallnorb_azimuth",
        "vtab_smallnorb_elevation",
    )
    DATASET_NAMES = NATURAL + SPECIALIZED + STRUCTURED + ("imagenet_val",)

    GROUPS = {
        "vtab_natural": {
            "anchors": NATURAL,
            "positives": NATURAL,
            "negatives": SPECIALIZED + STRUCTURED,
        },
        "vtab_specialized": {
            "anchors": SPECIALIZED,
            "positives": SPECIALIZED,
            "negatives": NATURAL + STRUCTURED,
        },
        "vtab_structured": {
            "anchors": STRUCTURED,
            "positives": STRUCTURED,
            "negatives": NATURAL + SPECIALIZED,
        },
        "domain_remote_sensing": {
            "anchors": ("vtab_eurosat", "vtab_resisc45"),
            "positives": ("vtab_eurosat", "vtab_resisc45"),
            "negatives": ("vtab_patch_camelyon", "vtab_retinopathy"),
        },
        "domain_medical": {
            "anchors": ("vtab_patch_camelyon", "vtab_retinopathy"),
            "positives": ("vtab_patch_camelyon", "vtab_retinopathy"),
            "negatives": ("vtab_eurosat", "vtab_resisc45"),
        },
        "source_clevr": {
            "anchors": ("vtab_clevr_count", "vtab_clevr_distance"),
            "positives": ("vtab_clevr_count", "vtab_clevr_distance"),
            "negatives": (
                "vtab_dmlab",
                "vtab_dsprites_location",
                "vtab_dsprites_orientation",
                "vtab_kitti_distance",
                "vtab_smallnorb_azimuth",
                "vtab_smallnorb_elevation",
            ),
        },
        "source_dsprites": {
            "anchors": ("vtab_dsprites_location", "vtab_dsprites_orientation"),
            "positives": ("vtab_dsprites_location", "vtab_dsprites_orientation"),
            "negatives": (
                "vtab_clevr_count",
                "vtab_clevr_distance",
                "vtab_dmlab",
                "vtab_kitti_distance",
                "vtab_smallnorb_azimuth",
                "vtab_smallnorb_elevation",
            ),
        },
        "source_smallnorb": {
            "anchors": ("vtab_smallnorb_azimuth", "vtab_smallnorb_elevation"),
            "positives": ("vtab_smallnorb_azimuth", "vtab_smallnorb_elevation"),
            "negatives": (
                "vtab_clevr_count",
                "vtab_clevr_distance",
                "vtab_dmlab",
                "vtab_dsprites_location",
                "vtab_dsprites_orientation",
                "vtab_kitti_distance",
            ),
        },
        "task_object_entity_recognition": {
            "anchors": (
                "vtab_caltech101",
                "vtab_cifar100",
                "vtab_flowers102",
                "vtab_pets",
            ),
            "positives": (
                "vtab_caltech101",
                "vtab_cifar100",
                "vtab_flowers102",
                "vtab_pets",
            ),
            "negatives": ("vtab_dtd", "vtab_sun397", "vtab_svhn"),
        },
        "task_direct_depth_prediction": {
            "anchors": ("vtab_clevr_distance", "vtab_kitti_distance"),
            "positives": ("vtab_clevr_distance", "vtab_kitti_distance"),
            "negatives": (
                "vtab_clevr_count",
                "vtab_dsprites_location",
                "vtab_dsprites_orientation",
                "vtab_smallnorb_azimuth",
                "vtab_smallnorb_elevation",
            ),
        },
        "task_viewpoint_orientation": {
            "anchors": (
                "vtab_dsprites_orientation",
                "vtab_smallnorb_azimuth",
                "vtab_smallnorb_elevation",
            ),
            "positives": (
                "vtab_dsprites_orientation",
                "vtab_smallnorb_azimuth",
                "vtab_smallnorb_elevation",
            ),
            "negatives": (
                "vtab_clevr_count",
                "vtab_clevr_distance",
                "vtab_dsprites_location",
                "vtab_kitti_distance",
            ),
        },
        "pretraining_domain_alignment": {
            "anchors": ("imagenet_val",),
            "positives": NATURAL,
            "negatives": SPECIALIZED + STRUCTURED,
        },
    }

    METAGROUPS = {
        "vtab": ("vtab_natural", "vtab_specialized", "vtab_structured"),
        "specialized_domain": ("domain_remote_sensing", "domain_medical"),
        "source": ("source_clevr", "source_dsprites", "source_smallnorb"),
    }
    FUNCTIONAL_GROUPS = (
        "task_object_entity_recognition",
        "task_direct_depth_prediction",
        "task_viewpoint_orientation",
    )
    INDIVIDUAL_GROUPS = (
        "vtab_natural",
        "vtab_specialized",
        "vtab_structured",
        "domain_remote_sensing",
        "domain_medical",
        "source_clevr",
        "source_dsprites",
        "source_smallnorb",
        "pretraining_domain_alignment",
    )
    REPORT_GROUPS = tuple(METAGROUPS) + FUNCTIONAL_GROUPS + INDIVIDUAL_GROUPS

    RAW_METRICS = {
        "F": ("hellinger", "pairwise_normalized_cosine"),
        "P": ("hellinger",),
        "S": ("hellinger",),
    }
    DELTA_METRICS = {profile: ("angular",) for profile in PROFILES}

    def __init__(
        self,
        model_names,
        model_groups,
        signature_paths,
        signature_keys,
        save_dir,
        create_plots=False,
        verbose=True,
    ):
        self.model_names = tuple(model_names)
        self.model_groups = {
            name: tuple(models) for name, models in model_groups.items()
        }
        self.signature_paths = tuple(tuple(paths) for paths in signature_paths)
        self.save_dir = Path(save_dir)
        self.create_plots = bool(create_plots)
        self.verbose = bool(verbose)

        if len(set(self.model_names)) != len(self.model_names):
            raise ValueError("model_names must be unique.")
        if self.REFERENCE_MODEL not in self.model_names:
            raise ValueError(f"model_names must contain '{self.REFERENCE_MODEL}'.")
        if len(self.signature_paths) != len(self.model_names):
            raise ValueError("signature_paths must contain one sequence per model.")
        if any(len(paths) != len(self.DATASET_NAMES) for paths in self.signature_paths):
            raise ValueError(
                f"Each model must have {len(self.DATASET_NAMES)} signature paths "
                "ordered according to DATASET_NAMES."
            )

        average_names = set()
        for name, models in self.model_groups.items():
            if not models:
                raise ValueError(f"Model group '{name}' is empty.")
            if len(set(models)) != len(models):
                raise ValueError(f"Model group '{name}' contains duplicate models.")
            unknown = set(models) - set(self.model_names)
            if unknown:
                raise ValueError(f"Unknown models in '{name}': {sorted(unknown)}.")
            average_name = f"{name}_average"
            if average_name in self.model_names or average_name in average_names:
                raise ValueError(
                    f"Conflicting model-group average name: {average_name}."
                )
            average_names.add(average_name)

        if isinstance(signature_keys, str):
            signature_keys = (signature_keys,)
        self.keys = {}
        for key in signature_keys:
            families = tuple(x.strip() for x in key.split(",") if x.strip())
            name = "+".join(families)
            if not families:
                raise ValueError(f"Invalid signature key: {key!r}.")
            if name in self.keys:
                raise ValueError(f"Duplicate signature key: {name!r}.")
            self.keys[name] = families
        if not self.keys:
            raise ValueError("At least one signature key is required.")

        self.families = tuple(
            dict.fromkeys(
                family for families in self.keys.values() for family in families
            )
        )
        self.dataset_index = {
            name: index for index, name in enumerate(self.DATASET_NAMES)
        }

    def _log(self, message):
        if self.verbose:
            print(message, flush=True)

    def _load_profiles(self):
        profiles = {profile: {} for profile in self.PROFILES}
        total = len(self.model_names) * len(self.DATASET_NAMES)

        with tqdm(
            total=total, desc="Loading profiles", disable=not self.verbose
        ) as progress:
            for model_index, model in enumerate(self.model_names):
                loaded = {
                    profile: {family: [] for family in self.families}
                    for profile in self.PROFILES
                }
                for path in self.signature_paths[model_index]:
                    with np.load(path) as data:
                        for family in self.families:
                            values = {}
                            for profile in self.PROFILES:
                                name = f"{profile}__{family}"
                                if name not in data:
                                    raise KeyError(f"Missing '{name}' in {path}.")
                                x = np.asarray(data[name], dtype=np.float64)
                                if not np.all(np.isfinite(x)):
                                    raise ValueError(
                                        f"Non-finite values in '{name}' from {path}."
                                    )
                                values[profile] = x
                            if len({x.shape for x in values.values()}) != 1:
                                shapes = ", ".join(
                                    f"{profile}{x.shape}"
                                    for profile, x in values.items()
                                )
                                raise ValueError(
                                    f"Shape mismatch for family '{family}' in {path}: "
                                    f"{shapes}."
                                )
                            for profile, x in values.items():
                                loaded[profile][family].append(x)
                    progress.update()

                for profile in self.PROFILES:
                    profiles[profile][model] = {
                        family: np.stack(values)
                        for family, values in loaded[profile].items()
                    }
        return profiles

    @staticmethod
    def _magnitude_stats(x):
        values = np.abs(np.asarray(x, dtype=np.float64).reshape(-1))
        if not values.size:
            raise ValueError("Cannot analyze an empty profile.")

        l1_norm = float(values.sum())
        if l1_norm == 0:
            support = 0.0
        else:
            p = values[values > 0] / l1_norm
            support = float(np.exp(-np.sum(p * np.log(p))) / values.size)

        return {
            "l1_norm": l1_norm,
            "mean_absolute_value": float(values.mean()),
            "effective_support_fraction": support,
        }

    @staticmethod
    def _profile_vectors(signatures, model, dataset, families):
        arrays = [signatures[model][family][dataset] for family in families]
        n_layers = arrays[0].shape[0]
        if any(array.shape[0] != n_layers for array in arrays):
            raise ValueError("Families in a signature key have different layer counts.")

        vectors = {"global": np.concatenate([array.reshape(-1) for array in arrays])}
        vectors.update(
            {
                f"layer_{layer:02d}": np.concatenate(
                    [array[layer].reshape(-1) for array in arrays]
                )
                for layer in range(n_layers)
            }
        )
        return vectors

    @staticmethod
    def _mean_stats(dataset_stats):
        result = {}
        for variant, metrics in next(iter(dataset_stats.values())).items():
            result[variant] = {}
            for metric in metrics:
                values = np.asarray(
                    [stats[variant][metric] for stats in dataset_stats.values()],
                    dtype=np.float64,
                )
                values = values[np.isfinite(values)]
                result[variant][metric] = (
                    float(values.mean()) if values.size else np.nan
                )
        return result

    def _compute_magnitude(self, representations):
        results = {}

        for representation, profiles in representations.items():
            results[representation] = {}

            for profile, signatures in profiles.items():
                results[representation][profile] = {}
                reference = representations["raw"][profile][self.REFERENCE_MODEL]

                for model in signatures:
                    results[representation][profile][model] = {}

                    for key, families in self.keys.items():
                        dataset_stats = {}

                        for dataset, dataset_name in enumerate(self.DATASET_NAMES):
                            vectors = self._profile_vectors(
                                signatures, model, dataset, families
                            )
                            stats = {
                                variant: self._magnitude_stats(vector)
                                for variant, vector in vectors.items()
                            }

                            if representation == "delta":
                                reference_vectors = self._profile_vectors(
                                    {self.REFERENCE_MODEL: reference},
                                    self.REFERENCE_MODEL,
                                    dataset,
                                    families,
                                )
                                for variant, values in stats.items():
                                    reference_l1 = float(
                                        np.abs(reference_vectors[variant]).sum()
                                    )
                                    values["relative_l1_change"] = (
                                        values["l1_norm"] / reference_l1
                                        if reference_l1 > 0
                                        else np.nan
                                    )

                            dataset_stats[dataset_name] = stats

                        results[representation][profile][model][key] = {
                            "global": self._mean_stats(dataset_stats),
                            "datasets": dataset_stats,
                        }

        return results

    def _save_magnitude(self, magnitude):
        base_columns = (
            "l1_norm",
            "mean_absolute_value",
            "effective_support_fraction",
        )

        for representation, profiles in magnitude.items():
            columns = base_columns + (
                ("relative_l1_change",) if representation == "delta" else ()
            )

            for profile, models in profiles.items():
                root = self.save_dir / "magnitude_analysis" / representation / profile
                summary = []

                for model in self.model_names:
                    if model not in models:
                        continue

                    for key, values in models[model].items():
                        out = root / model / key
                        out.mkdir(parents=True, exist_ok=True)

                        with open(out / "magnitude.csv", "w", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow(["dataset_name", "variant", *columns])

                            for variant, stats in values["global"].items():
                                writer.writerow(
                                    [
                                        "global",
                                        variant,
                                        *[stats[column] for column in columns],
                                    ]
                                )

                            for dataset_name in self.DATASET_NAMES:
                                for variant, stats in values["datasets"][
                                    dataset_name
                                ].items():
                                    writer.writerow(
                                        [
                                            dataset_name,
                                            variant,
                                            *[stats[column] for column in columns],
                                        ]
                                    )

                        stats = values["global"]["global"]
                        summary.append(
                            [
                                model,
                                key,
                                *[stats[column] for column in columns],
                            ]
                        )

                root.mkdir(parents=True, exist_ok=True)
                with open(root / "magnitude.csv", "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["model_name", "signature_key", *columns])
                    writer.writerows(summary)

    def _make_delta_profiles(self, profiles):
        delta = {profile: {} for profile in self.PROFILES}
        for profile in self.PROFILES:
            reference = profiles[profile][self.REFERENCE_MODEL]
            for model in self.model_names:
                if model == self.REFERENCE_MODEL:
                    continue
                delta[profile][model] = {}
                for family in self.families:
                    x = profiles[profile][model][family]
                    ref = reference[family]
                    if x.shape != ref.shape:
                        raise ValueError(
                            f"Cannot compute {profile} delta for model '{model}', "
                            f"family '{family}': {x.shape} vs {ref.shape}."
                        )
                    delta[profile][model][family] = x - ref
        return delta

    @staticmethod
    def _zero_case(a, b):
        a_zero = not np.any(a)
        b_zero = not np.any(b)
        if a_zero and b_zero:
            return 0.0
        if a_zero or b_zero:
            return 1.0
        return None

    def _hellinger(self, a, b):
        if np.any(a < 0) or np.any(b < 0):
            raise ValueError("Hellinger distance requires nonnegative profiles.")
        zero = self._zero_case(a, b)
        if zero is not None:
            return zero
        p, q = a / a.sum(), b / b.sum()
        return float(np.sqrt(0.5 * np.square(np.sqrt(p) - np.sqrt(q)).sum()))

    def _pairwise_normalized_cosine(self, a, b):
        if np.any(a < 0) or np.any(b < 0):
            raise ValueError(
                "Task2Vec pairwise-normalized cosine requires nonnegative profiles."
            )
        zero = self._zero_case(a, b)
        if zero is not None:
            return zero
        total = a + b
        x = np.divide(a, total, out=np.zeros_like(a), where=total > 0)
        y = np.divide(b, total, out=np.zeros_like(b), where=total > 0)
        similarity = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        return float(1.0 - np.clip(similarity, -1.0, 1.0))

    def _angular(self, a, b):
        zero = self._zero_case(a, b)
        if zero is not None:
            return zero
        similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.arccos(np.clip(similarity, -1.0, 1.0)) / np.pi)

    def _distance(self, metric, a, b):
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        return getattr(self, f"_{metric}")(a, b)

    def _combined_distance(
        self, signatures, metric, model, dataset_a, dataset_b, families, layer=None
    ):
        values = []
        for family in families:
            a = signatures[model][family][dataset_a]
            b = signatures[model][family][dataset_b]
            if layer is not None:
                a, b = a[layer], b[layer]
            values.append(self._distance(metric, a, b))
        return float(np.sqrt(np.mean(np.square(values))))

    def _distance_matrix(self, signatures, metric, model, families, layer=None):
        n = len(self.DATASET_NAMES)
        matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i):
                matrix[i, j] = matrix[j, i] = self._combined_distance(
                    signatures, metric, model, i, j, families, layer
                )
        return matrix

    def _compute_distances(self, representations):
        metric_policy = {"raw": self.RAW_METRICS, "delta": self.DELTA_METRICS}
        total = sum(
            len(metrics)
            * len(representations[representation][profile])
            * len(self.keys)
            for representation, policies in metric_policy.items()
            for profile, metrics in policies.items()
        )
        results = {"raw": {}, "delta": {}}

        with tqdm(
            total=total, desc="Computing distances", disable=not self.verbose
        ) as progress:
            for representation, policies in metric_policy.items():
                for profile, metrics in policies.items():
                    signatures = representations[representation][profile]
                    for metric in metrics:
                        results[representation].setdefault(metric, {})[profile] = {}
                        for model in signatures:
                            results[representation][metric][profile][model] = {}
                            for key, families in self.keys.items():
                                n_layers = signatures[model][families[0]].shape[1]
                                if any(
                                    signatures[model][family].shape[1] != n_layers
                                    for family in families
                                ):
                                    raise ValueError(
                                        f"Families in '{key}' have different layer counts "
                                        f"for model '{model}'."
                                    )
                                results[representation][metric][profile][model][key] = {
                                    "global": self._distance_matrix(
                                        signatures, metric, model, families
                                    ),
                                    "layers": np.stack(
                                        [
                                            self._distance_matrix(
                                                signatures,
                                                metric,
                                                model,
                                                families,
                                                layer,
                                            )
                                            for layer in range(n_layers)
                                        ]
                                    ),
                                }
                                progress.set_postfix_str(
                                    f"{representation} | {metric} | {profile} | "
                                    f"{model} | {key}",
                                    refresh=False,
                                )
                                progress.update()
        return results

    @staticmethod
    def _pair_concordance(matrix, anchor, positives, negatives):
        scores = []
        for positive in positives:
            for negative in negatives:
                near, far = matrix[anchor, positive], matrix[anchor, negative]
                scores.append(
                    1.0 if near < far else 0.5 if np.isclose(near, far) else 0.0
                )
        return float(np.mean(scores))

    def _group_auc(self, matrix, group):
        spec = self.GROUPS[group]
        anchors = [self.dataset_index[name] for name in spec["anchors"]]
        positives = [self.dataset_index[name] for name in spec["positives"]]
        negatives = [self.dataset_index[name] for name in spec["negatives"]]
        scores = []
        for anchor in anchors:
            peers = [index for index in positives if index != anchor]
            if not peers or not negatives:
                raise ValueError(f"Group '{group}' has no valid comparisons.")
            scores.append(self._pair_concordance(matrix, anchor, peers, negatives))
        return float(np.mean(scores))

    def _separation_scores(self, matrix):
        individual = {group: self._group_auc(matrix, group) for group in self.GROUPS}
        meta = {
            name: float(np.mean([individual[group] for group in groups]))
            for name, groups in self.METAGROUPS.items()
        }
        values = {**individual, **meta}
        return {group: values[group] for group in self.REPORT_GROUPS}

    def _group_averages(self, model_scores):
        averages = {}
        for group_name, group_models in self.model_groups.items():
            models = [model for model in group_models if model in model_scores]
            if not models:
                continue
            average_name = f"{group_name}_average"
            averages[average_name] = {}
            for key in self.keys:
                averages[average_name][key] = {
                    group: float(
                        np.mean(
                            [
                                model_scores[model][key]["global"][group]
                                for model in models
                            ]
                        )
                    )
                    for group in self.REPORT_GROUPS
                }
        return averages

    def _compute_separation(self, distances):
        results = {"raw": {}, "delta": {}}
        total = sum(
            len(keys)
            for representation in distances.values()
            for profiles in representation.values()
            for models in profiles.values()
            for keys in models.values()
        )

        with tqdm(
            total=total, desc="Computing separation", disable=not self.verbose
        ) as progress:
            for representation, metrics in distances.items():
                for metric, profiles in metrics.items():
                    results[representation].setdefault(metric, {})
                    for profile, models in profiles.items():
                        model_scores = {}
                        for model, keys in models.items():
                            model_scores[model] = {}
                            for key, matrices in keys.items():
                                model_scores[model][key] = {
                                    "global": self._separation_scores(
                                        matrices["global"]
                                    ),
                                    "layers": [
                                        self._separation_scores(matrix)
                                        for matrix in matrices["layers"]
                                    ],
                                }
                                progress.update()
                        results[representation][metric][profile] = {
                            "models": model_scores,
                            "group_averages": self._group_averages(model_scores),
                        }
        return results

    def _plot_distance(self, matrix, path, title):
        xy = (
            np.zeros((len(matrix), 2))
            if np.allclose(matrix, 0)
            else MDS(
                n_components=2,
                dissimilarity="precomputed",
                random_state=0,
                n_init=4,
            ).fit_transform(matrix)
        )
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(xy[:, 0], xy[:, 1])
        for index, name in enumerate(self.DATASET_NAMES):
            ax.annotate(name, xy[index], xytext=(4, 4), textcoords="offset points")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def _save_distance_plots(self, distances):
        if not self.create_plots:
            return
        total = sum(
            1 + len(matrices["layers"])
            for representation in distances.values()
            for profiles in representation.values()
            for models in profiles.values()
            for keys in models.values()
            for matrices in keys.values()
        )
        with tqdm(
            total=total, desc="Saving distance plots", disable=not self.verbose
        ) as progress:
            for representation, metrics in distances.items():
                for metric, profiles in metrics.items():
                    for profile, models in profiles.items():
                        for model, keys in models.items():
                            for key, matrices in keys.items():
                                out = (
                                    self.save_dir
                                    / "distance_analysis"
                                    / representation
                                    / metric
                                    / profile
                                    / model
                                    / key
                                    / "plots"
                                )
                                title = (
                                    f"{representation} | {metric} | {profile} | "
                                    f"{model} | {key}"
                                )
                                self._plot_distance(
                                    matrices["global"],
                                    out / "global.png",
                                    f"{title} | global",
                                )
                                progress.update()
                                for layer, matrix in enumerate(matrices["layers"]):
                                    self._plot_distance(
                                        matrix,
                                        out / f"layer_{layer:02d}.png",
                                        f"{title} | layer {layer:02d}",
                                    )
                                    progress.update()

    def _save_separation(self, separation):
        for representation, metrics in separation.items():
            for metric, profiles in metrics.items():
                for profile, result in profiles.items():
                    root = (
                        self.save_dir
                        / "distance_analysis"
                        / representation
                        / metric
                        / profile
                    )
                    summary = []
                    for model in self.model_names:
                        if model not in result["models"]:
                            continue
                        for key, scores in result["models"][model].items():
                            out = root / model / key
                            out.mkdir(parents=True, exist_ok=True)
                            with open(out / "separation.csv", "w", newline="") as file:
                                writer = csv.writer(file)
                                writer.writerow(["variant", *self.REPORT_GROUPS])
                                writer.writerow(
                                    [
                                        "global",
                                        *[
                                            scores["global"][group]
                                            for group in self.REPORT_GROUPS
                                        ],
                                    ]
                                )
                                for layer, layer_scores in enumerate(scores["layers"]):
                                    writer.writerow(
                                        [
                                            f"layer_{layer:02d}",
                                            *[
                                                layer_scores[group]
                                                for group in self.REPORT_GROUPS
                                            ],
                                        ]
                                    )
                            summary.append(
                                [
                                    model,
                                    key,
                                    *[
                                        scores["global"][group]
                                        for group in self.REPORT_GROUPS
                                    ],
                                ]
                            )

                    for average_name, keys in result["group_averages"].items():
                        for key, scores in keys.items():
                            summary.append(
                                [
                                    average_name,
                                    key,
                                    *[scores[group] for group in self.REPORT_GROUPS],
                                ]
                            )

                    root.mkdir(parents=True, exist_ok=True)
                    with open(root / "separation.csv", "w", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(
                            ["model_name", "signature_key", *self.REPORT_GROUPS]
                        )
                        writer.writerows(summary)

    def analyze(self):
        self._log("Starting VTAB functional-separation analysis.")
        raw = self._load_profiles()
        representations = {"raw": raw, "delta": self._make_delta_profiles(raw)}
        magnitude = self._compute_magnitude(representations)
        distances = self._compute_distances(representations)
        separation = self._compute_separation(distances)
        self._save_magnitude(magnitude)
        self._save_distance_plots(distances)
        self._save_separation(separation)
        self._log(f"Analysis complete. Results saved to: {self.save_dir}")
        return {
            "magnitude": magnitude,
            "distances": distances,
            "separation": separation,
        }
