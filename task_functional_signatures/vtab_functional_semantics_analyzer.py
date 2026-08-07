import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS
from tqdm.auto import tqdm


class VTABFunctionalSemanticsAnalyzer:
    """Analyze semantic structure in raw and adaptation-delta VTAB signatures."""

    SIGNATURES = ("F", "P", "S")
    REFERENCE_MODEL = "linear_probe"

    # ------------------------------------------------------------------
    # Functional-component schema
    # ------------------------------------------------------------------

    L, H, D, K, MLP_DIM = 12, 12, 768, 64, 3072

    AXIS_VALUES = {
        "site": (
            "attention_read",
            "attention_write",
            "mlp_read",
            "mlp_write",
        ),
        "token_group": ("cls", "patch"),
        "query_group": ("cls", "patch"),
        "destination_group": ("cls", "patch"),
        "route": (
            "cls<-cls",
            "cls<-patch",
            "patch<-cls",
            "patch<-patch_self",
            "patch<-patch_local",
            "patch<-patch_medium",
            "patch<-patch_global",
        ),
        "sublayer": ("attention", "mlp"),
    }

    COMPONENTS = {
        "residual_stream_channel": (
            (L, 4, 2, D),
            ("layer", "site", "token_group", "residual_channel"),
        ),
        "final_cls_readout": (
            (D,),
            ("residual_channel",),
        ),
        "attention_qk_route": (
            (L, H, 7),
            ("layer", "head", "route"),
        ),
        "attention_ov_route": (
            (L, H, 7),
            ("layer", "head", "route"),
        ),
        "attention_qk_channel": (
            (L, H, 2, K),
            ("layer", "head", "query_group", "head_channel"),
        ),
        "attention_ov_channel": (
            (L, H, 2, K),
            ("layer", "head", "destination_group", "head_channel"),
        ),
        "mlp_neuron": (
            (L, 2, MLP_DIM),
            ("layer", "token_group", "mlp_neuron"),
        ),
        "prompt_to_backbone_qk_route": (
            (L, H, 2),
            ("layer", "head", "destination_group"),
        ),
        "prompt_to_backbone_ov_route": (
            (L, H, 2),
            ("layer", "head", "destination_group"),
        ),
        "sublayer_update": (
            (L, 2),
            ("layer", "sublayer"),
        ),
        "sublayer_token_update": (
            (L, 2, 2),
            ("layer", "sublayer", "token_group"),
        ),
        "attention_head_update": (
            (L, H),
            ("layer", "head"),
        ),
    }

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
            "anchors": (
                "vtab_dsprites_location",
                "vtab_dsprites_orientation",
            ),
            "positives": (
                "vtab_dsprites_location",
                "vtab_dsprites_orientation",
            ),
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
            "anchors": (
                "vtab_smallnorb_azimuth",
                "vtab_smallnorb_elevation",
            ),
            "positives": (
                "vtab_smallnorb_azimuth",
                "vtab_smallnorb_elevation",
            ),
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
            "negatives": (
                "vtab_dtd",
                "vtab_sun397",
                "vtab_svhn",
            ),
        },
        "task_direct_depth_prediction": {
            "anchors": (
                "vtab_clevr_distance",
                "vtab_kitti_distance",
            ),
            "positives": (
                "vtab_clevr_distance",
                "vtab_kitti_distance",
            ),
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
        "vtab": (
            "vtab_natural",
            "vtab_specialized",
            "vtab_structured",
        ),
        "specialized_domain": (
            "domain_remote_sensing",
            "domain_medical",
        ),
        "source": (
            "source_clevr",
            "source_dsprites",
            "source_smallnorb",
        ),
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

    DELTA_METRICS = {signature: ("angular",) for signature in SIGNATURES}

    AGREEMENT_METRICS = ("ranking_agreement", "geometric_agreement")

    def __init__(
        self,
        model_names,
        model_groups,
        signature_paths,
        save_dir,
        save_distances=False,
        create_plots=False,
        verbose=True,
    ):
        self.model_names = tuple(model_names)
        self.model_groups = {
            name: tuple(models) for name, models in model_groups.items()
        }
        self.signature_paths = tuple(tuple(paths) for paths in signature_paths)
        self.save_dir = Path(save_dir)
        self.save_distances = bool(save_distances)
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

        self.components = tuple(self.COMPONENTS)
        self.dataset_index = {name: i for i, name in enumerate(self.DATASET_NAMES)}

    def _log(self, message):
        if self.verbose:
            print(message, flush=True)

    @classmethod
    def _layer_count(cls, component):
        shape, axes = cls.COMPONENTS[component]
        return shape[0] if axes[0] == "layer" else 0

    def _load_signatures(self):
        signatures = {signature: {} for signature in self.SIGNATURES}
        total = len(self.model_names) * len(self.DATASET_NAMES)

        with tqdm(
            total=total, desc="Loading signatures", disable=not self.verbose
        ) as pbar:
            for model_index, model in enumerate(self.model_names):
                loaded = {
                    signature: {component: [] for component in self.components}
                    for signature in self.SIGNATURES
                }

                for path in self.signature_paths[model_index]:
                    with np.load(path) as data:
                        for component in self.components:
                            expected_shape = self.COMPONENTS[component][0]
                            for signature in self.SIGNATURES:
                                name = f"{signature}__{component}"
                                if name not in data:
                                    raise KeyError(f"Missing '{name}' in {path}.")
                                x = np.asarray(data[name], dtype=np.float64)
                                if x.shape != expected_shape:
                                    raise ValueError(
                                        f"Invalid shape for '{name}' in {path}: "
                                        f"{x.shape}, expected {expected_shape}."
                                    )
                                if not np.all(np.isfinite(x)):
                                    raise ValueError(
                                        f"Non-finite values in '{name}' from {path}."
                                    )
                                loaded[signature][component].append(x)
                    pbar.update()

                for signature in self.SIGNATURES:
                    signatures[signature][model] = {
                        component: np.stack(values)
                        for component, values in loaded[signature].items()
                    }

        return signatures

    @staticmethod
    def _magnitude_stats(x):
        x = np.abs(np.asarray(x, dtype=np.float64).reshape(-1))
        if not x.size:
            raise ValueError("Cannot analyze an empty signature.")

        l1 = float(x.sum())
        if l1:
            p = x[x > 0] / l1
            support = float(np.exp(-np.sum(p * np.log(p))) / x.size)
        else:
            support = 0.0

        return {
            "l1_norm": l1,
            "mean_absolute_value": float(x.mean()),
            "effective_support_fraction": support,
        }

    def _signature_vectors(self, signatures, model, dataset, component):
        x = signatures[model][component][dataset]
        vectors = {"global": x.reshape(-1)}
        vectors.update(
            {
                f"layer_{layer:02d}": x[layer].reshape(-1)
                for layer in range(self._layer_count(component))
            }
        )
        return vectors

    @staticmethod
    def _mean_stats(dataset_stats):
        first = next(iter(dataset_stats.values()))
        return {
            variant: {
                metric: float(values.mean()) if values.size else np.nan
                for metric in metrics
                for values in [
                    np.asarray(
                        [
                            stats[variant][metric]
                            for stats in dataset_stats.values()
                            if np.isfinite(stats[variant][metric])
                        ],
                        dtype=np.float64,
                    )
                ]
            }
            for variant, metrics in first.items()
        }

    def _compute_magnitude(self, representations):
        results = {}

        for representation, signatures_by_type in representations.items():
            results[representation] = {}
            for signature, signatures in signatures_by_type.items():
                results[representation][signature] = {}
                reference = representations["raw"][signature][self.REFERENCE_MODEL]

                for model in signatures:
                    results[representation][signature][model] = {}
                    for component in self.components:
                        dataset_stats = {}

                        for dataset, dataset_name in enumerate(self.DATASET_NAMES):
                            vectors = self._signature_vectors(
                                signatures, model, dataset, component
                            )
                            stats = {
                                variant: self._magnitude_stats(vector)
                                for variant, vector in vectors.items()
                            }

                            if representation == "delta":
                                reference_vectors = self._signature_vectors(
                                    {self.REFERENCE_MODEL: reference},
                                    self.REFERENCE_MODEL,
                                    dataset,
                                    component,
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

                        results[representation][signature][model][component] = {
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

        for representation, signatures in magnitude.items():
            columns = base_columns + (
                ("relative_l1_change",) if representation == "delta" else ()
            )

            for signature, models in signatures.items():
                root = self.save_dir / "magnitude_analysis" / representation / signature
                summary = []

                for model in self.model_names:
                    if model not in models:
                        continue

                    for component, values in models[model].items():
                        out = root / model / component
                        out.mkdir(parents=True, exist_ok=True)

                        with open(out / "magnitude.csv", "w", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow(["dataset_name", "variant", *columns])

                            for variant, stats in values["global"].items():
                                writer.writerow(
                                    ["global", variant, *[stats[c] for c in columns]]
                                )

                            for dataset_name in self.DATASET_NAMES:
                                for variant, stats in values["datasets"][
                                    dataset_name
                                ].items():
                                    writer.writerow(
                                        [
                                            dataset_name,
                                            variant,
                                            *[stats[c] for c in columns],
                                        ]
                                    )

                        stats = values["global"]["global"]
                        summary.append([model, component, *[stats[c] for c in columns]])

                root.mkdir(parents=True, exist_ok=True)
                with open(root / "magnitude.csv", "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["model_name", "component", *columns])
                    writer.writerows(summary)

    def _make_delta_signatures(self, signatures):
        return {
            signature: {
                model: {
                    component: (
                        signatures[signature][model][component]
                        - signatures[signature][self.REFERENCE_MODEL][component]
                    )
                    for component in self.components
                }
                for model in self.model_names
                if model != self.REFERENCE_MODEL
            }
            for signature in self.SIGNATURES
        }

    @staticmethod
    def _zero_case(a, b):
        a_zero, b_zero = not np.any(a), not np.any(b)
        return 0.0 if a_zero and b_zero else 1.0 if a_zero or b_zero else None

    def _hellinger(self, a, b):
        if np.any(a < 0) or np.any(b < 0):
            raise ValueError("Hellinger distance requires nonnegative signatures.")
        zero = self._zero_case(a, b)
        if zero is not None:
            return zero
        p, q = a / a.sum(), b / b.sum()
        return float(np.sqrt(0.5 * np.square(np.sqrt(p) - np.sqrt(q)).sum()))

    def _pairwise_normalized_cosine(self, a, b):
        if np.any(a < 0) or np.any(b < 0):
            raise ValueError(
                "Task2Vec pairwise-normalized cosine requires nonnegative signatures."
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

    def _distance_matrix(self, signatures, metric, model, component, layer=None):
        n = len(self.DATASET_NAMES)
        matrix = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            a = signatures[model][component][i]
            if layer is not None:
                a = a[layer]

            for j in range(i):
                b = signatures[model][component][j]
                if layer is not None:
                    b = b[layer]
                matrix[i, j] = matrix[j, i] = self._distance(metric, a, b)

        return matrix

    def _compute_distances(self, representations):
        policy = {"raw": self.RAW_METRICS, "delta": self.DELTA_METRICS}
        results = {"raw": {}, "delta": {}}
        total = sum(
            len(metrics)
            * len(representations[representation][signature])
            * len(self.components)
            for representation, signatures in policy.items()
            for signature, metrics in signatures.items()
        )

        with tqdm(
            total=total, desc="Computing distances", disable=not self.verbose
        ) as pbar:
            for representation, signatures in policy.items():
                for signature, metrics in signatures.items():
                    values = representations[representation][signature]

                    for metric in metrics:
                        results[representation].setdefault(metric, {})[signature] = {}

                        for model in values:
                            results[representation][metric][signature][model] = {}

                            for component in self.components:
                                n_layers = self._layer_count(component)
                                layers = (
                                    np.stack(
                                        [
                                            self._distance_matrix(
                                                values, metric, model, component, layer
                                            )
                                            for layer in range(n_layers)
                                        ]
                                    )
                                    if n_layers
                                    else np.empty(
                                        (
                                            0,
                                            len(self.DATASET_NAMES),
                                            len(self.DATASET_NAMES),
                                        ),
                                        dtype=np.float64,
                                    )
                                )

                                results[representation][metric][signature][model][
                                    component
                                ] = {
                                    "global": self._distance_matrix(
                                        values, metric, model, component
                                    ),
                                    "layers": layers,
                                }

                                pbar.set_postfix_str(
                                    f"{representation} | {metric} | {signature} | "
                                    f"{model} | {component}",
                                    refresh=False,
                                )
                                pbar.update()

        return results

    def _group_scores(self, matrix, group):
        spec = self.GROUPS[group]
        anchors = [self.dataset_index[x] for x in spec["anchors"]]
        positives = [self.dataset_index[x] for x in spec["positives"]]
        negatives = [self.dataset_index[x] for x in spec["negatives"]]
        ranking, geometric = [], []

        for anchor in anchors:
            peers = [x for x in positives if x != anchor]
            if not peers or not negatives:
                raise ValueError(f"Group '{group}' has no valid comparisons.")

            positive = matrix[anchor, peers]
            negative = matrix[anchor, negatives]

            # ROC-AUC / concordance:
            # probability that a positive is closer than a negative.
            p = positive[:, None]
            n = negative[None, :]
            ranking.append(
                float(((p < n).sum() + 0.5 * (p == n).sum()) / p.size / n.size)
            )

            # Silhouette-style normalized semantic distance margin.
            d_pos, d_neg = float(positive.mean()), float(negative.mean())
            denominator = max(d_pos, d_neg)
            geometric.append((d_neg - d_pos) / denominator if denominator else 0.0)

        return {
            "ranking_agreement": float(np.mean(ranking)),
            "geometric_agreement": float(np.mean(geometric)),
        }

    def _separation_scores(self, matrix):
        individual = {group: self._group_scores(matrix, group) for group in self.GROUPS}

        for name, groups in self.METAGROUPS.items():
            individual[name] = {
                metric: float(np.mean([individual[group][metric] for group in groups]))
                for metric in self.AGREEMENT_METRICS
            }

        return {
            metric: {group: individual[group][metric] for group in self.REPORT_GROUPS}
            for metric in self.AGREEMENT_METRICS
        }

    def _group_averages(self, model_scores):
        averages = {}

        for group_name, group_models in self.model_groups.items():
            models = [model for model in group_models if model in model_scores]
            if not models:
                continue

            averages[f"{group_name}_average"] = {
                component: {
                    metric: {
                        group: float(
                            np.mean(
                                [
                                    model_scores[model][component]["global"][metric][
                                        group
                                    ]
                                    for model in models
                                ]
                            )
                        )
                        for group in self.REPORT_GROUPS
                    }
                    for metric in self.AGREEMENT_METRICS
                }
                for component in self.components
            }

        return averages

    def _compute_separation(self, distances):
        results = {"raw": {}, "delta": {}}
        total = sum(
            len(components)
            for metrics in distances.values()
            for signatures in metrics.values()
            for models in signatures.values()
            for components in models.values()
        )

        with tqdm(
            total=total, desc="Computing separation", disable=not self.verbose
        ) as pbar:
            for representation, metrics in distances.items():
                for metric, signatures in metrics.items():
                    results[representation].setdefault(metric, {})

                    for signature, models in signatures.items():
                        model_scores = {
                            model: {
                                component: {
                                    "global": self._separation_scores(
                                        matrices["global"]
                                    ),
                                    "layers": [
                                        self._separation_scores(matrix)
                                        for matrix in matrices["layers"]
                                    ],
                                }
                                for component, matrices in components.items()
                            }
                            for model, components in models.items()
                        }

                        pbar.update(
                            sum(len(components) for components in models.values())
                        )

                        results[representation][metric][signature] = {
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

        for i, name in enumerate(self.DATASET_NAMES):
            ax.annotate(name, xy[i], xytext=(4, 4), textcoords="offset points")

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def _save_distances(self, distances):
        if not (self.save_distances or self.create_plots):
            return

        total = sum(
            1 + len(matrices["layers"])
            for metrics in distances.values()
            for signatures in metrics.values()
            for models in signatures.values()
            for components in models.values()
            for matrices in components.values()
        )

        with tqdm(
            total=total, desc="Saving distance outputs", disable=not self.verbose
        ) as pbar:
            for representation, metrics in distances.items():
                for metric, signatures in metrics.items():
                    for signature, models in signatures.items():
                        for model, components in models.items():
                            for component, matrices in components.items():
                                out = (
                                    self.save_dir
                                    / "distance_analysis"
                                    / representation
                                    / metric
                                    / signature
                                    / model
                                    / component
                                )
                                items = [("global", matrices["global"])] + [
                                    (f"layer_{layer:02d}", matrix)
                                    for layer, matrix in enumerate(matrices["layers"])
                                ]
                                title = (
                                    f"{representation} | {metric} | {signature} | "
                                    f"{model} | {component}"
                                )

                                for variant, matrix in items:
                                    if self.save_distances:
                                        out.mkdir(parents=True, exist_ok=True)
                                        np.save(out / f"{variant}.npy", matrix)

                                    if self.create_plots:
                                        self._plot_distance(
                                            matrix,
                                            out / "plots" / f"{variant}.png",
                                            f"{title} | {variant}",
                                        )

                                    pbar.update()

    def _save_separation(self, separation):
        columns = [
            f"{group}__{metric}"
            for group in self.REPORT_GROUPS
            for metric in self.AGREEMENT_METRICS
        ]

        def row(scores):
            return [
                scores[metric][group]
                for group in self.REPORT_GROUPS
                for metric in self.AGREEMENT_METRICS
            ]

        for representation, metrics in separation.items():
            for metric, signatures in metrics.items():
                for signature, result in signatures.items():
                    root = (
                        self.save_dir
                        / "separation_analysis"
                        / representation
                        / metric
                        / signature
                    )
                    summary = []

                    for model in self.model_names:
                        if model not in result["models"]:
                            continue

                        for component, scores in result["models"][model].items():
                            out = root / model / component
                            out.mkdir(parents=True, exist_ok=True)

                            with open(out / "separation.csv", "w", newline="") as file:
                                writer = csv.writer(file)
                                writer.writerow(["variant", *columns])
                                writer.writerow(["global", *row(scores["global"])])
                                for layer, layer_scores in enumerate(scores["layers"]):
                                    writer.writerow(
                                        [f"layer_{layer:02d}", *row(layer_scores)]
                                    )

                            summary.append([model, component, *row(scores["global"])])

                    for average_name, components in result["group_averages"].items():
                        for component, scores in components.items():
                            summary.append([average_name, component, *row(scores)])

                    root.mkdir(parents=True, exist_ok=True)
                    with open(root / "separation.csv", "w", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(["model_name", "component", *columns])
                        writer.writerows(summary)

    def analyze(self):
        self._log("Starting VTAB functional-semantic analysis.")

        raw = self._load_signatures()
        representations = {
            "raw": raw,
            "delta": self._make_delta_signatures(raw),
        }

        magnitude = self._compute_magnitude(representations)
        distances = self._compute_distances(representations)
        separation = self._compute_separation(distances)

        self._save_magnitude(magnitude)
        self._save_distances(distances)
        self._save_separation(separation)

        self._log(f"Analysis complete. Results saved to: {self.save_dir}")

        return {
            "magnitude": magnitude,
            "distances": distances,
            "separation": separation,
        }
