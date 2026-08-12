import base64
import gzip
import json
import math
from itertools import product
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


class VTABFunctionalSemanticsAnalyzer:
    """Interpretable semantic analysis of raw and adaptation-delta VTAB signatures."""

    SIGNATURES = ("F", "P", "S")
    REFERENCE_MODEL = "linear_probe"
    L, H, D, K, MLP_DIM = 12, 12, 768, 64, 3072

    AXIS_VALUES = {
        "site": ("attention_read", "attention_write", "mlp_read", "mlp_write"),
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
        "final_cls_readout": ((D,), ("residual_channel",)),
        "attention_qk_route": ((L, H, 7), ("layer", "head", "route")),
        "attention_ov_route": ((L, H, 7), ("layer", "head", "route")),
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
        "sublayer_update": ((L, 2), ("layer", "sublayer")),
        "sublayer_token_update": (
            (L, 2, 2),
            ("layer", "sublayer", "token_group"),
        ),
        "attention_head_update": ((L, H), ("layer", "head")),
    }

    COMPONENT_FAMILY = {
        "sublayer_update": "backbone_updates",
        "sublayer_token_update": "backbone_updates",
        "attention_head_update": "backbone_updates",
        "residual_stream_channel": "residual_flow",
        "attention_qk_route": "attention_routing",
        "attention_ov_route": "attention_routing",
        "attention_qk_channel": "feature_content",
        "attention_ov_channel": "feature_content",
        "mlp_neuron": "feature_content",
        "prompt_to_backbone_qk_route": "prompt_influence",
        "prompt_to_backbone_ov_route": "prompt_influence",
        "final_cls_readout": "final_readout",
    }

    VIEW_AXES = {
        "sublayer_update": (
            ("sublayer",),
            ("layer",),
        ),
        "sublayer_token_update": (
            ("sublayer",),
            ("token_group",),
            ("sublayer", "token_group"),
            ("layer",),
            ("layer", "sublayer"),
            ("layer", "token_group"),
        ),
        "attention_head_update": (("layer",),),
        "residual_stream_channel": (
            ("site",),
            ("token_group",),
            ("site", "token_group"),
            ("layer",),
            ("layer", "site", "token_group"),
        ),
        "attention_qk_route": (
            ("route",),
            ("layer",),
            ("layer", "route"),
        ),
        "attention_ov_route": (
            ("route",),
            ("layer",),
            ("layer", "route"),
        ),
        "attention_qk_channel": (
            ("query_group",),
            ("layer",),
            ("layer", "query_group"),
        ),
        "attention_ov_channel": (
            ("destination_group",),
            ("layer",),
            ("layer", "destination_group"),
        ),
        "mlp_neuron": (
            ("token_group",),
            ("layer",),
            ("layer", "token_group"),
        ),
        "prompt_to_backbone_qk_route": (
            ("destination_group",),
            ("layer",),
            ("layer", "destination_group"),
        ),
        "prompt_to_backbone_ov_route": (
            ("destination_group",),
            ("layer",),
            ("layer", "destination_group"),
        ),
        "final_cls_readout": (),
    }

    PROMPT_COMPONENTS = (
        "prompt_to_backbone_qk_route",
        "prompt_to_backbone_ov_route",
    )

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
    AGREEMENT_METRICS = ("ranking_agreement", "geometric_agreement")

    RAW_METRICS = {
        "F": ("hellinger", "pairwise_normalized_cosine"),
        "P": ("hellinger",),
        "S": ("hellinger",),
    }

    DELTA_METRICS = {signature: ("angular",) for signature in SIGNATURES}

    MAGNITUDE_METRICS = (
        "l1_norm",
        "mean_absolute_value",
        "effective_support_fraction",
        "relative_l1_change",
    )
    AVERAGE_KEY = "__average__"

    def __init__(
        self,
        model_names,
        model_groups,
        signature_paths,
        save_dir,
        verbose=True,
        decimal_places=3,
    ):
        self.model_names = tuple(model_names)
        self.model_groups = {
            name: tuple(models) for name, models in model_groups.items()
        }
        self.signature_paths = tuple(tuple(paths) for paths in signature_paths)
        self.save_dir = Path(save_dir)
        self.verbose = bool(verbose)
        self.decimal_places = int(decimal_places)

        if self.decimal_places != decimal_places or self.decimal_places < 0:
            raise ValueError("decimal_places must be a nonnegative integer.")

        if len(set(self.model_names)) != len(self.model_names):
            raise ValueError("model_names must be unique.")

        if self.REFERENCE_MODEL not in self.model_names:
            raise ValueError(f"model_names must contain '{self.REFERENCE_MODEL}'.")

        if len(self.signature_paths) != len(self.model_names):
            raise ValueError("signature_paths must contain one sequence per model.")

        if any(len(paths) != len(self.DATASET_NAMES) for paths in self.signature_paths):
            raise ValueError(
                f"Each model must have {len(self.DATASET_NAMES)} paths "
                "ordered as DATASET_NAMES."
            )

        average_names = set()
        known = set(self.model_names)

        for name, models in self.model_groups.items():
            if not models or len(set(models)) != len(models):
                raise ValueError(
                    f"Model group '{name}' must be non-empty "
                    "and contain unique models."
                )

            unknown = set(models) - known

            if unknown:
                raise ValueError(f"Unknown models in '{name}': {sorted(unknown)}.")

            average = f"{name}_average"

            if average in known or average in average_names:
                raise ValueError(f"Conflicting model-group average name: {average}.")

            average_names.add(average)

        self.components = tuple(self.COMPONENTS)
        self.dataset_index = {name: i for i, name in enumerate(self.DATASET_NAMES)}
        self.views = self._build_views()

        self.separation_slots = tuple(
            (group, anchor)
            for group, spec in self.GROUPS.items()
            for anchor in (*spec["anchors"], self.AVERAGE_KEY)
        )

    def _log(self, message):
        if self.verbose:
            print(message, flush=True)

    @classmethod
    def _axis_values(cls, component, axis):
        shape, axes = cls.COMPONENTS[component]

        if axis == "layer":
            return range(shape[axes.index(axis)])

        return cls.AXIS_VALUES[axis]

    @classmethod
    def _axis_index(cls, axis, value):
        if axis == "layer":
            return int(value)

        return cls.AXIS_VALUES[axis].index(value)

    @staticmethod
    def _view_description(selectors, vector_axes):
        if not selectors:
            return "Full component signature."

        fixed = ", ".join(f"{axis}={value}" for axis, value in selectors.items())

        return f"Fix {fixed}; compare tasks over " f"{' x '.join(vector_axes)}."

    def _build_views(self):
        views = []

        for component in self.components:
            shape, axes = self.COMPONENTS[component]
            family = self.COMPONENT_FAMILY[component]

            views.append(
                {
                    "family": family,
                    "component": component,
                    "view": "full",
                    "view_id": f"{component}/full",
                    "selectors": {},
                    "vector_axes": axes,
                    "vector_dim": math.prod(shape),
                    "description": "Full component signature.",
                }
            )

            for fixed_axes in self.VIEW_AXES.get(component, ()):
                for values in product(
                    *(self._axis_values(component, axis) for axis in fixed_axes)
                ):
                    selectors = dict(zip(fixed_axes, values))

                    vector_axes = tuple(axis for axis in axes if axis not in selectors)

                    vector_dim = math.prod(
                        size for size, axis in zip(shape, axes) if axis not in selectors
                    )

                    if vector_dim < 2:
                        continue

                    view = "_".join(fixed_axes)

                    suffix = "/".join(
                        f"{axis}={value}" for axis, value in selectors.items()
                    )

                    views.append(
                        {
                            "family": family,
                            "component": component,
                            "view": view,
                            "view_id": f"{component}/{view}/{suffix}",
                            "selectors": selectors,
                            "vector_axes": vector_axes,
                            "vector_dim": vector_dim,
                            "description": self._view_description(
                                selectors,
                                vector_axes,
                            ),
                        }
                    )

        ids = [view["view_id"] for view in views]

        if len(ids) != len(set(ids)):
            raise RuntimeError("Duplicate view IDs.")

        return tuple(views)

    def _load_signatures(self):
        signatures = {signature: {} for signature in self.SIGNATURES}

        total = len(self.model_names) * len(self.DATASET_NAMES)

        with tqdm(
            total=total,
            desc="Loading signatures",
            disable=not self.verbose,
        ) as pbar:
            for model_idx, model in enumerate(self.model_names):
                loaded = {
                    signature: {component: [] for component in self.components}
                    for signature in self.SIGNATURES
                }

                for path in self.signature_paths[model_idx]:
                    with np.load(path) as data:
                        for component in self.components:
                            expected = self.COMPONENTS[component][0]

                            for signature in self.SIGNATURES:
                                key = f"{signature}__{component}"

                                if key not in data:
                                    raise KeyError(f"Missing '{key}' in {path}.")

                                x = np.asarray(
                                    data[key],
                                    dtype=np.float64,
                                )

                                if x.shape != expected:
                                    raise ValueError(
                                        f"Invalid shape for '{key}' "
                                        f"in {path}: {x.shape}, "
                                        f"expected {expected}."
                                    )

                                if not np.all(np.isfinite(x)):
                                    raise ValueError(
                                        f"Non-finite values in " f"'{key}' from {path}."
                                    )

                                loaded[signature][component].append(x)

                    pbar.update()

                for signature in self.SIGNATURES:
                    signatures[signature][model] = {
                        component: np.stack(values)
                        for component, values in loaded[signature].items()
                    }

        return signatures

    def _make_delta_signatures(self, raw):
        return {
            signature: {
                model: {
                    component: (
                        raw[signature][model][component]
                        - raw[signature][self.REFERENCE_MODEL][component]
                    )
                    for component in self.components
                }
                for model in self.model_names
                if model != self.REFERENCE_MODEL
            }
            for signature in self.SIGNATURES
        }

    def _has_prompts(self, raw, model):
        return any(
            np.any(raw["P"][model][component]) for component in self.PROMPT_COMPONENTS
        )

    def _extract_vectors(
        self,
        signatures,
        model,
        view,
    ):
        component = view["component"]
        x = signatures[model][component]
        axes = self.COMPONENTS[component][1]

        index = [slice(None)] + [
            (
                slice(None)
                if axis not in view["selectors"]
                else self._axis_index(
                    axis,
                    view["selectors"][axis],
                )
            )
            for axis in axes
        ]

        vectors = x[tuple(index)].reshape(
            len(self.DATASET_NAMES),
            -1,
        )

        if vectors.shape[1] != view["vector_dim"]:
            raise RuntimeError(
                f"View '{view['view_id']}' "
                f"has dimension {vectors.shape[1]}, "
                f"expected {view['vector_dim']}."
            )

        return vectors

    @staticmethod
    def _hellinger_matrix(vectors):
        if np.any(vectors < 0):
            raise ValueError("Hellinger distance requires nonnegative signatures.")

        sums = vectors.sum(axis=1)
        valid = sums > 0

        roots = np.zeros_like(vectors)

        roots[valid] = np.sqrt(vectors[valid] / sums[valid, None])

        similarity = np.clip(
            roots @ roots.T,
            0.0,
            1.0,
        )

        matrix = np.sqrt(
            np.maximum(
                0.0,
                1.0 - similarity,
            )
        )

        matrix[~valid, :] = 1.0
        matrix[:, ~valid] = 1.0
        matrix[np.ix_(~valid, ~valid)] = 0.0

        np.fill_diagonal(
            matrix,
            0.0,
        )

        return matrix

    @staticmethod
    def _angular_matrix(vectors):
        norms = np.linalg.norm(
            vectors,
            axis=1,
        )

        valid = norms > 0

        normalized = np.zeros_like(vectors)

        normalized[valid] = vectors[valid] / norms[valid, None]

        similarity = np.clip(
            normalized @ normalized.T,
            -1.0,
            1.0,
        )

        matrix = np.arccos(similarity) / np.pi

        matrix[~valid, :] = 1.0
        matrix[:, ~valid] = 1.0
        matrix[np.ix_(~valid, ~valid)] = 0.0

        np.fill_diagonal(
            matrix,
            0.0,
        )

        return matrix

    @staticmethod
    def _pairwise_normalized_cosine_matrix(vectors):
        if np.any(vectors < 0):
            raise ValueError(
                "Task2Vec pairwise-normalized cosine "
                "requires nonnegative signatures."
            )

        n = len(vectors)

        matrix = np.zeros(
            (n, n),
            dtype=np.float64,
        )

        for i in range(n):
            for j in range(i):
                a = vectors[i]
                b = vectors[j]

                a_zero = not np.any(a)
                b_zero = not np.any(b)

                if a_zero or b_zero:
                    distance = 0.0 if a_zero and b_zero else 1.0
                else:
                    total = a + b

                    x = np.divide(
                        a,
                        total,
                        out=np.zeros_like(a),
                        where=total > 0,
                    )

                    y = np.divide(
                        b,
                        total,
                        out=np.zeros_like(b),
                        where=total > 0,
                    )

                    similarity = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

                    distance = 1.0 - np.clip(
                        similarity,
                        -1.0,
                        1.0,
                    )

                matrix[i, j] = distance
                matrix[j, i] = distance

        return matrix

    def _distance_matrix(
        self,
        metric,
        vectors,
    ):
        return getattr(
            self,
            f"_{metric}_matrix",
        )(vectors)

    def _group_scores(
        self,
        matrix,
        group,
    ):
        spec = self.GROUPS[group]

        positives = [self.dataset_index[x] for x in spec["positives"]]

        negatives = [self.dataset_index[x] for x in spec["negatives"]]

        scores = {}

        for anchor_name in spec["anchors"]:
            anchor = self.dataset_index[anchor_name]

            peers = [x for x in positives if x != anchor]

            if not peers or not negatives:
                raise ValueError(f"Group '{group}' has no valid comparisons.")

            positive = matrix[
                anchor,
                peers,
            ]

            negative = matrix[
                anchor,
                negatives,
            ]

            p = positive[:, None]
            n = negative[None, :]

            ranking = float(((p < n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size))

            d_pos = float(positive.mean())
            d_neg = float(negative.mean())

            denominator = max(
                d_pos,
                d_neg,
            )

            geometric = (d_neg - d_pos) / denominator if denominator else 0.0

            scores[anchor_name] = (
                ranking,
                float(geometric),
            )

        anchor_scores = list(scores.values())

        scores[self.AVERAGE_KEY] = tuple(
            float(np.mean([score[i] for score in anchor_scores])) for i in range(2)
        )

        return scores

    def _compute_separation(
        self,
        representations,
        raw,
    ):
        policy = {
            "raw": self.RAW_METRICS,
            "delta": self.DELTA_METRICS,
        }

        records = []

        total = sum(
            len(metrics)
            * sum(
                sum(
                    (view["component"] not in self.PROMPT_COMPONENTS)
                    or self._has_prompts(
                        raw,
                        model,
                    )
                    for view in self.views
                )
                for model in representations[representation][signature]
            )
            for representation, signature_metrics in policy.items()
            for signature, metrics in signature_metrics.items()
        )

        with tqdm(
            total=total,
            desc="Computing separation",
            disable=not self.verbose,
        ) as pbar:
            for (
                representation,
                signature_metrics,
            ) in policy.items():
                for signature, metrics in signature_metrics.items():
                    signatures = representations[representation][signature]

                    for metric in metrics:
                        for model in signatures:
                            has_prompts = self._has_prompts(
                                raw,
                                model,
                            )

                            for view in self.views:
                                if (
                                    view["component"] in self.PROMPT_COMPONENTS
                                    and not has_prompts
                                ):
                                    continue

                                vectors = self._extract_vectors(
                                    signatures,
                                    model,
                                    view,
                                )

                                matrix = self._distance_matrix(
                                    metric,
                                    vectors,
                                )

                                group_scores = {
                                    group: self._group_scores(
                                        matrix,
                                        group,
                                    )
                                    for group in self.GROUPS
                                }

                                scores = []

                                for group, anchor in self.separation_slots:
                                    scores.extend(group_scores[group][anchor])

                                records.append(
                                    [
                                        representation,
                                        signature,
                                        metric,
                                        model,
                                        view["view_id"],
                                        scores,
                                    ]
                                )

                                pbar.update()

        return records

    @staticmethod
    def _magnitude_matrix(vectors):
        x = np.abs(
            np.asarray(
                vectors,
                dtype=np.float64,
            )
        )

        l1 = x.sum(axis=1)
        mean = x.mean(axis=1)

        p = np.divide(
            x,
            l1[:, None],
            out=np.zeros_like(x),
            where=l1[:, None] > 0,
        )

        logp = np.zeros_like(p)

        np.log(
            p,
            out=logp,
            where=p > 0,
        )

        support = np.exp(-(p * logp).sum(axis=1)) / x.shape[1]

        support[l1 == 0] = 0.0

        return l1, mean, support

    def _compute_magnitude(
        self,
        representations,
        raw,
    ):
        records = []

        total = sum(
            sum(
                sum(
                    (view["component"] not in self.PROMPT_COMPONENTS)
                    or self._has_prompts(
                        raw,
                        model,
                    )
                    for view in self.views
                )
                for model in representations[representation][signature]
            )
            for representation in representations
            for signature in self.SIGNATURES
        )

        with tqdm(
            total=total,
            desc="Computing magnitude",
            disable=not self.verbose,
        ) as pbar:
            for (
                representation,
                signature_types,
            ) in representations.items():
                for signature, models in signature_types.items():
                    reference_l1 = None

                    if representation == "delta":
                        reference_l1 = {
                            view["view_id"]: np.abs(
                                self._extract_vectors(
                                    raw[signature],
                                    self.REFERENCE_MODEL,
                                    view,
                                )
                            ).sum(axis=1)
                            for view in self.views
                        }

                    for model in models:
                        has_prompts = self._has_prompts(
                            raw,
                            model,
                        )

                        for view in self.views:
                            if (
                                view["component"] in self.PROMPT_COMPONENTS
                                and not has_prompts
                            ):
                                continue

                            vectors = self._extract_vectors(
                                models,
                                model,
                                view,
                            )

                            l1, mean, support = self._magnitude_matrix(vectors)

                            relative = np.full(
                                len(self.DATASET_NAMES),
                                np.nan,
                            )

                            if representation == "delta":
                                ref_l1 = reference_l1[view["view_id"]]

                                relative = np.divide(
                                    l1,
                                    ref_l1,
                                    out=relative,
                                    where=ref_l1 > 0,
                                )

                            values = []

                            for i in range(len(self.DATASET_NAMES)):
                                values.extend(
                                    (
                                        float(l1[i]),
                                        float(mean[i]),
                                        float(support[i]),
                                        (
                                            float(relative[i])
                                            if np.isfinite(relative[i])
                                            else None
                                        ),
                                    )
                                )

                            records.append(
                                [
                                    representation,
                                    signature,
                                    model,
                                    view["view_id"],
                                    values,
                                ]
                            )

                            pbar.update()

        return records

    def _payload(
        self,
        separation,
        magnitude,
    ):
        models = [
            {
                "name": model,
                "type": "individual",
                "members": [],
            }
            for model in self.model_names
        ] + [
            {
                "name": f"{name}_average",
                "type": "group_average",
                "members": list(members),
            }
            for name, members in self.model_groups.items()
        ]

        components = [
            {
                "name": component,
                "family": self.COMPONENT_FAMILY[component],
                "shape": list(shape),
                "axes": list(axes),
            }
            for component, (
                shape,
                axes,
            ) in self.COMPONENTS.items()
        ]

        relations = []

        for name in self.REPORT_GROUPS:
            if name in self.GROUPS:
                spec = self.GROUPS[name]

                relations.append(
                    {
                        "name": name,
                        "type": "relation",
                        "anchors": list(spec["anchors"]),
                        "positives": list(spec["positives"]),
                        "negatives": list(spec["negatives"]),
                        "members": [],
                    }
                )
            else:
                relations.append(
                    {
                        "name": name,
                        "type": "metagroup",
                        "anchors": [],
                        "positives": [],
                        "negatives": [],
                        "members": list(self.METAGROUPS[name]),
                    }
                )

        return {
            "datasets": list(self.DATASET_NAMES),
            "reference_model": self.REFERENCE_MODEL,
            "models": models,
            "components": components,
            "views": list(self.views),
            "relations": relations,
            "relation_order": list(self.REPORT_GROUPS),
            "separation_slots": [list(slot) for slot in self.separation_slots],
            "separation_metrics": list(self.AGREEMENT_METRICS),
            "raw_metrics": {
                key: list(value) for key, value in self.RAW_METRICS.items()
            },
            "delta_metrics": {
                key: list(value) for key, value in self.DELTA_METRICS.items()
            },
            "magnitude_metrics": list(self.MAGNITUDE_METRICS),
            "decimal_places": self.decimal_places,
            "separation": separation,
            "magnitude": magnitude,
        }

    @staticmethod
    def _html_template():
        return r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta
    name="viewport"
    content="width=device-width,initial-scale=1"
>
<title>VTAB Functional Semantics</title>
<style>
:root {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system,
        BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #18181b;
    background: #fafafa;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
}

main {
    max-width: 1800px;
    margin: auto;
    padding: 24px;
}

h1 {
    margin: 0 0 18px;
    font-size: 22px;
}

.controls {
    display: grid;
    grid-template-columns: repeat(
        auto-fit,
        minmax(170px, 1fr)
    );
    gap: 10px;
    margin-bottom: 14px;
}

.field {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.field label {
    font-size: 11px;
    font-weight: 700;
    color: #71717a;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

select,
input {
    width: 100%;
    padding: 8px 9px;
    border: 1px solid #d4d4d8;
    border-radius: 6px;
    background: white;
    color: #18181b;
}

.hidden {
    display: none !important;
}

.status {
    margin: 8px 0;
    font-size: 12px;
    color: #71717a;
}

.table-wrap {
    overflow: auto;
    max-height: 72vh;
    border: 1px solid #e4e4e7;
    border-radius: 8px;
    background: white;
}

table {
    min-width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 13px;
}

th,
td {
    padding: 8px 10px;
    border-right: 1px solid #f0f0f2;
    border-bottom: 1px solid #f0f0f2;
    white-space: nowrap;
    text-align: right;
}

th {
    position: sticky;
    top: 0;
    z-index: 2;
    background: #f4f4f5;
    font-weight: 650;
}

th:first-child,
td:first-child {
    position: sticky;
    left: 0;
    z-index: 1;
    background: white;
    text-align: left;
}

th:first-child {
    z-index: 3;
    background: #f4f4f5;
}

td.value {
    cursor: pointer;
    font-variant-numeric: tabular-nums;
}

td.value:hover {
    background: #f4f4f5;
}

td.na {
    color: #a1a1aa;
}

.average td {
    background: #fafafa;
    font-weight: 650;
}

.average td:first-child {
    background: #fafafa;
}

dialog {
    width: min(650px, 92vw);
    max-width: 650px;
    padding: 0;
    border: 0;
    border-radius: 10px;
    box-shadow: 0 20px 60px #0003;
}

dialog::backdrop {
    background: #0005;
}

.dialog-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 18px;
    border-bottom: 1px solid #e4e4e7;
}

.dialog-head h2 {
    margin: 0;
    font-size: 16px;
}

.dialog-head button {
    border: 0;
    background: transparent;
    font-size: 20px;
    cursor: pointer;
}

.details {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 7px 14px;
    padding: 16px 18px;
    font-size: 13px;
}

.details dt {
    font-weight: 650;
    color: #52525b;
}

.details dd {
    margin: 0;
    overflow-wrap: anywhere;
}

.loading {
    padding: 40px;
    color: #71717a;
    text-align: center;
}

@media (max-width: 700px) {
    main {
        padding: 14px;
    }

    .controls {
        grid-template-columns: 1fr 1fr;
    }
}
</style>
</head>

<body>
<main>
<h1>VTAB Functional Semantics</h1>

<div id="loading" class="loading">
    Loading embedded results…
</div>

<div id="app" class="hidden">
    <div class="controls">
        <div class="field">
            <label>Analysis</label>
            <select id="analysis">
                <option>Separation</option>
                <option>Magnitude</option>
            </select>
        </div>

        <div class="field">
            <label>Compare</label>
            <select id="compare">
                <option>Datasets</option>
                <option>Models</option>
            </select>
        </div>

        <div class="field">
            <label>Signature</label>
            <select id="signature">
                <option>F</option>
                <option>P</option>
                <option>S</option>
            </select>
        </div>

        <div class="field">
            <label>Representation</label>
            <select id="representation">
                <option>raw</option>
                <option>delta</option>
            </select>
        </div>

        <div class="field separation-only">
            <label>Distance</label>
            <input
                id="distance"
                list="distance-list"
                autocomplete="off"
            >
            <datalist id="distance-list"></datalist>
        </div>

        <div class="field separation-only">
            <label>Agreement</label>
            <select id="agreement"></select>
        </div>

        <div class="field magnitude-only">
            <label>Magnitude metric</label>
            <input
                id="magnitudeMetric"
                list="magnitude-list"
                autocomplete="off"
            >
            <datalist id="magnitude-list"></datalist>
        </div>

        <div class="field">
            <label>Relation</label>
            <input
                id="relation"
                list="relation-list"
                autocomplete="off"
            >
            <datalist id="relation-list"></datalist>
        </div>

        <div class="field model-control">
            <label>Model</label>
            <input
                id="model"
                list="model-list"
                autocomplete="off"
            >
            <datalist id="model-list"></datalist>
        </div>

        <div class="field dataset-control hidden">
            <label>Dataset</label>
            <input
                id="dataset"
                list="dataset-list"
                autocomplete="off"
            >
            <datalist id="dataset-list"></datalist>
        </div>

        <div class="field">
            <label>Component</label>
            <input
                id="component"
                list="component-list"
                autocomplete="off"
            >
            <datalist id="component-list"></datalist>
        </div>

        <div class="field view-control">
            <label>View</label>
            <input
                id="view"
                list="view-list"
                autocomplete="off"
            >
            <datalist id="view-list"></datalist>
        </div>

        <div class="field layer-control hidden">
            <label>Layer</label>
            <select id="layer"></select>
        </div>
    </div>

    <div id="status" class="status"></div>

    <div class="table-wrap">
        <table id="matrix"></table>
    </div>
</div>
</main>

<dialog id="dialog">
    <div class="dialog-head">
        <h2>Result details</h2>
        <button
            id="closeDialog"
            aria-label="Close"
        >×</button>
    </div>

    <dl id="details" class="details"></dl>
</dialog>

<script
    id="payload"
    type="application/octet-stream"
>__PAYLOAD__</script>

<script>
(async () => {
    const $ = id => document.getElementById(id);

    const encoded = $("payload").textContent.trim();

    if (!("DecompressionStream" in window)) {
        $("loading").textContent =
            "This browser does not support DecompressionStream. " +
            "Use a current Chromium, Firefox, or Safari browser.";
        return;
    }

    const bytes = Uint8Array.from(
        atob(encoded),
        c => c.charCodeAt(0)
    );

    const stream = new Blob([bytes])
        .stream()
        .pipeThrough(
            new DecompressionStream("gzip")
        );

    const D = JSON.parse(
        await new Response(stream).text()
    );

    const models = new Map(
        D.models.map(x => [x.name, x])
    );

    const relations = new Map(
        D.relations.map(x => [x.name, x])
    );

    const views = new Map(
        D.views.map(x => [x.view_id, x])
    );

    const separationSlots = new Map(
        D.separation_slots.map(
            (x, i) => [x[0] + "|" + x[1], i]
        )
    );

    const separationIndex = new Map(
        D.separation.map(
            row => [
                [
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                ].join("|"),
                row[5],
            ]
        )
    );

    const magnitudeIndex = new Map(
        D.magnitude.map(
            row => [
                [
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                ].join("|"),
                row[4],
            ]
        )
    );

    const separationCache = new Map();
    const magnitudeCache = new Map();

    const AVG = "__average__";
    const ALL = "Full components";

    const analysis = $("analysis");
    const compare = $("compare");
    const signature = $("signature");
    const representation = $("representation");
    const distance = $("distance");
    const agreement = $("agreement");
    const magnitudeMetric = $("magnitudeMetric");
    const relation = $("relation");
    const model = $("model");
    const dataset = $("dataset");
    const component = $("component");
    const view = $("view");
    const layer = $("layer");

    function setChoices(
        input,
        listId,
        values,
        preferred,
    ) {
        const datalist = $(listId);

        datalist.replaceChildren(
            ...values.map(
                value => Object.assign(
                    document.createElement("option"),
                    {value},
                )
            )
        );

        input.value = values.includes(input.value)
            ? input.value
            : (
                values.includes(preferred)
                    ? preferred
                    : values[0] || ""
            );
    }

    function setSelect(
        select,
        values,
        preferred,
    ) {
        select.replaceChildren(
            ...values.map(
                value => Object.assign(
                    document.createElement("option"),
                    {
                        value,
                        textContent: value,
                    },
                )
            )
        );

        select.value = values.includes(preferred)
            ? preferred
            : values[0] || "";
    }

    function validateInput(
        input,
        listId,
    ) {
        const values = [
            ...$(listId).options,
        ].map(
            option => option.value
        );

        if (!values.includes(input.value)) {
            input.value = values[0] || "";
        }
    }

    function relationAnchors(name) {
        const item = relations.get(name);

        if (!item) {
            return [];
        }

        if (item.type === "relation") {
            return item.anchors;
        }

        return [
            ...new Set(
                item.members.flatMap(
                    relationAnchors
                )
            ),
        ];
    }

    function baseRelation(
        name,
        anchor,
    ) {
        const item = relations.get(name);

        if (!item) {
            return null;
        }

        if (item.type === "relation") {
            return item.anchors.includes(anchor)
                ? name
                : null;
        }

        return (
            item.members.find(
                member => relationAnchors(
                    member
                ).includes(anchor)
            )
            || null
        );
    }

    function mean(values) {
        const valid = values.filter(
            value => (
                value !== null
                && Number.isFinite(value)
            )
        );

        if (!valid.length) {
            return null;
        }

        return (
            valid.reduce(
                (a, b) => a + b,
                0,
            )
            / valid.length
        );
    }

    function separationArray(
        representationName,
        signatureName,
        distanceName,
        modelName,
        viewId,
    ) {
        const key = [
            representationName,
            signatureName,
            distanceName,
            modelName,
            viewId,
        ].join("|");

        if (separationCache.has(key)) {
            return separationCache.get(key);
        }

        const modelSpec = models.get(modelName);
        let result = null;

        if (modelSpec?.type === "individual") {
            result = (
                separationIndex.get(key)
                || null
            );
        } else if (modelSpec) {
            const arrays = modelSpec.members.map(
                member => separationArray(
                    representationName,
                    signatureName,
                    distanceName,
                    member,
                    viewId,
                )
            );

            if (arrays.every(Boolean)) {
                result = arrays[0].map(
                    (_, i) => mean(
                        arrays.map(
                            array => array[i]
                        )
                    )
                );
            }
        }

        separationCache.set(
            key,
            result,
        );

        return result;
    }

    function separationValue(
        representationName,
        signatureName,
        distanceName,
        modelName,
        viewId,
        relationName,
        anchor,
        agreementName,
    ) {
        const values = separationArray(
            representationName,
            signatureName,
            distanceName,
            modelName,
            viewId,
        );

        if (!values) {
            return null;
        }

        const agreementIndex = (
            D.separation_metrics.indexOf(
                agreementName
            )
        );

        const relationSpec = relations.get(
            relationName
        );

        if (
            !relationSpec
            || agreementIndex < 0
        ) {
            return null;
        }

        if (relationSpec.type === "relation") {
            const slot = separationSlots.get(
                relationName + "|" + anchor
            );

            if (slot === undefined) {
                return null;
            }

            return values[
                slot
                * D.separation_metrics.length
                + agreementIndex
            ];
        }

        if (anchor === AVG) {
            return mean(
                relationSpec.members.map(
                    member => separationValue(
                        representationName,
                        signatureName,
                        distanceName,
                        modelName,
                        viewId,
                        member,
                        AVG,
                        agreementName,
                    )
                )
            );
        }

        const member = baseRelation(
            relationName,
            anchor,
        );

        return member
            ? separationValue(
                representationName,
                signatureName,
                distanceName,
                modelName,
                viewId,
                member,
                anchor,
                agreementName,
            )
            : null;
    }

    function magnitudeArray(
        representationName,
        signatureName,
        modelName,
        viewId,
    ) {
        const key = [
            representationName,
            signatureName,
            modelName,
            viewId,
        ].join("|");

        if (magnitudeCache.has(key)) {
            return magnitudeCache.get(key);
        }

        const modelSpec = models.get(modelName);
        let result = null;

        if (modelSpec?.type === "individual") {
            result = (
                magnitudeIndex.get(key)
                || null
            );
        } else if (modelSpec) {
            const arrays = modelSpec.members.map(
                member => magnitudeArray(
                    representationName,
                    signatureName,
                    member,
                    viewId,
                )
            );

            if (arrays.every(Boolean)) {
                result = arrays[0].map(
                    (_, i) => mean(
                        arrays.map(
                            array => array[i]
                        )
                    )
                );
            }
        }

        magnitudeCache.set(
            key,
            result,
        );

        return result;
    }

    function magnitudeValue(
        representationName,
        signatureName,
        modelName,
        viewId,
        relationName,
        anchor,
        metricName,
    ) {
        const values = magnitudeArray(
            representationName,
            signatureName,
            modelName,
            viewId,
        );

        if (!values) {
            return null;
        }

        const metricIndex = (
            D.magnitude_metrics.indexOf(
                metricName
            )
        );

        if (metricIndex < 0) {
            return null;
        }

        function datasetValue(name) {
            const datasetIndex = (
                D.datasets.indexOf(name)
            );

            if (datasetIndex < 0) {
                return null;
            }

            return values[
                datasetIndex
                * D.magnitude_metrics.length
                + metricIndex
            ];
        }

        if (anchor === AVG) {
            return mean(
                relationAnchors(
                    relationName
                ).map(
                    datasetValue
                )
            );
        }

        return datasetValue(anchor);
    }

    function distanceOptions() {
        const policy = (
            representation.value === "raw"
                ? D.raw_metrics
                : D.delta_metrics
        );

        return (
            policy[signature.value]
            || []
        );
    }

    function modelOptions() {
        const individual = D.models
            .filter(
                item => item.type === "individual"
            )
            .map(
                item => item.name
            );

        const averages = D.models.filter(
            item => item.type === "group_average"
        );

        if (representation.value === "delta") {
            return [
                ...individual.filter(
                    name => (
                        name !== D.reference_model
                    )
                ),
                ...averages
                    .filter(
                        item => (
                            !item.members.includes(
                                D.reference_model
                            )
                        )
                    )
                    .map(
                        item => item.name
                    ),
            ];
        }

        return [
            ...individual,
            ...averages.map(
                item => item.name
            ),
        ];
    }

    function componentViews(name) {
        return [
            ...new Set(
                D.views
                    .filter(
                        item => (
                            item.component === name
                        )
                    )
                    .map(
                        item => item.view
                    )
            ),
        ];
    }

    function viewAxes(
        name,
        viewName,
    ) {
        return [
            ...new Set(
                D.views
                    .filter(
                        item => (
                            item.component === name
                            && item.view === viewName
                        )
                    )
                    .flatMap(
                        item => (
                            Object.keys(
                                item.selectors
                            )
                        )
                    )
            ),
        ];
    }

    function shortValue(value) {
        const names = {
            "attention_read": "attn read",
            "attention_write": "attn write",
            "mlp_read": "MLP read",
            "mlp_write": "MLP write",
            "attention": "attention",
            "mlp": "MLP",
            "cls<-cls": "CLS ← CLS",
            "cls<-patch": "CLS ← patch",
            "patch<-cls": "patch ← CLS",
            "patch<-patch_self":
                "patch ← patch self",
            "patch<-patch_local":
                "patch ← patch local",
            "patch<-patch_medium":
                "patch ← patch medium",
            "patch<-patch_global":
                "patch ← patch global",
        };

        return (
            names[String(value)]
            || String(value)
        );
    }

    function columns() {
        if (component.value === ALL) {
            return D.components.map(
                item => ({
                    label: item.name,
                    viewId:
                        item.name + "/full",
                    view: views.get(
                        item.name + "/full"
                    ),
                })
            );
        }

        const full = views.get(
            component.value + "/full"
        );

        if (view.value === "full") {
            return [
                {
                    label: "Full",
                    viewId: full.view_id,
                    view: full,
                },
            ];
        }

        let items = D.views.filter(
            item => (
                item.component
                    === component.value
                && item.view
                    === view.value
            )
        );

        const axes = viewAxes(
            component.value,
            view.value,
        );

        if (
            axes.includes("layer")
            && axes.length > 1
        ) {
            items = items.filter(
                item => (
                    Number(
                        item.selectors.layer
                    )
                    === Number(layer.value)
                )
            );
        }

        function label(item) {
            const selectors = (
                Object.entries(
                    item.selectors
                ).filter(
                    ([axis]) => !(
                        axis === "layer"
                        && axes.length > 1
                    )
                )
            );

            if (!selectors.length) {
                return item.view;
            }

            return selectors
                .map(
                    ([axis, value]) => (
                        axis === "layer"
                            ? (
                                "L"
                                + String(value)
                                    .padStart(
                                        2,
                                        "0",
                                    )
                            )
                            : shortValue(value)
                    )
                )
                .join(" / ");
        }

        return [
            {
                label: "Full",
                viewId: full.view_id,
                view: full,
            },
            ...items.map(
                item => ({
                    label: label(item),
                    viewId: item.view_id,
                    view: item,
                })
            ),
        ];
    }

    function valueFor(
        modelName,
        anchor,
        column,
    ) {
        if (analysis.value === "Separation") {
            return separationValue(
                representation.value,
                signature.value,
                distance.value,
                modelName,
                column.viewId,
                relation.value,
                anchor,
                agreement.value,
            );
        }

        return magnitudeValue(
            representation.value,
            signature.value,
            modelName,
            column.viewId,
            relation.value,
            anchor,
            magnitudeMetric.value,
        );
    }

    function format(value) {
        if (
            value === null
            || !Number.isFinite(value)
        ) {
            return "—";
        }

        return Number(value).toFixed(
            D.decimal_places
        );
    }

    function availableModel(
        name,
        columns,
        anchor,
    ) {
        return columns.some(
            column => (
                valueFor(
                    name,
                    anchor,
                    column,
                ) !== null
            )
        );
    }

    function updateControls() {
        document
            .querySelectorAll(
                ".separation-only"
            )
            .forEach(
                item => item.classList.toggle(
                    "hidden",
                    analysis.value
                        !== "Separation",
                )
            );

        document
            .querySelectorAll(
                ".magnitude-only"
            )
            .forEach(
                item => item.classList.toggle(
                    "hidden",
                    analysis.value
                        !== "Magnitude",
                )
            );

        document
            .querySelectorAll(
                ".model-control"
            )
            .forEach(
                item => item.classList.toggle(
                    "hidden",
                    compare.value
                        !== "Datasets",
                )
            );

        document
            .querySelectorAll(
                ".dataset-control"
            )
            .forEach(
                item => item.classList.toggle(
                    "hidden",
                    compare.value
                        !== "Models",
                )
            );

        setChoices(
            relation,
            "relation-list",
            D.relation_order,
            relation.value
                || D.relation_order[0],
        );

        setChoices(
            component,
            "component-list",
            [
                ALL,
                ...D.components.map(
                    item => item.name
                ),
            ],
            component.value || ALL,
        );

        if (analysis.value === "Separation") {
            setChoices(
                distance,
                "distance-list",
                distanceOptions(),
                distance.value,
            );

            setSelect(
                agreement,
                D.separation_metrics,
                agreement.value,
            );
        } else {
            const metrics = (
                D.magnitude_metrics.filter(
                    name => (
                        representation.value
                            === "delta"
                        || name
                            !== "relative_l1_change"
                    )
                )
            );

            setChoices(
                magnitudeMetric,
                "magnitude-list",
                metrics,
                magnitudeMetric.value,
            );
        }

        const availableModels = modelOptions();

        setChoices(
            model,
            "model-list",
            availableModels,
            model.value
                || availableModels[0],
        );

        const datasets = [
            ...relationAnchors(
                relation.value
            ),
            AVG,
        ];

        setChoices(
            dataset,
            "dataset-list",
            datasets,
            dataset.value || datasets[0],
        );

        if (component.value === ALL) {
            document
                .querySelector(
                    ".view-control"
                )
                .classList.add("hidden");

            document
                .querySelector(
                    ".layer-control"
                )
                .classList.add("hidden");

            return;
        }

        document
            .querySelector(
                ".view-control"
            )
            .classList.remove("hidden");

        const availableViews = componentViews(
            component.value
        );

        setChoices(
            view,
            "view-list",
            availableViews,
            view.value
                || availableViews[0],
        );

        const axes = viewAxes(
            component.value,
            view.value,
        );

        const layered = (
            axes.includes("layer")
            && axes.length > 1
        );

        document
            .querySelector(
                ".layer-control"
            )
            .classList.toggle(
                "hidden",
                !layered,
            );

        if (layered) {
            setSelect(
                layer,
                [
                    ...Array(12).keys(),
                ].map(String),
                layer.value || "0",
            );
        }
    }

    function detailRows(
        modelName,
        anchor,
        column,
        value,
    ) {
        const rows = [
            [
                "Value",
                value === null
                    ? "N/A"
                    : String(value),
            ],
            ["Analysis", analysis.value],
            ["Compare", compare.value],
            ["Signature", signature.value],
            [
                "Representation",
                representation.value,
            ],
            ["Relation", relation.value],
            [
                "Dataset / aggregate",
                anchor === AVG
                    ? (
                        relation.value
                        + " average"
                    )
                    : anchor,
            ],
            ["Model", modelName],
            ["Family", column.view.family],
            [
                "Component",
                column.view.component,
            ],
            ["View", column.view.view],
            [
                "Selectors",
                (
                    Object.entries(
                        column.view.selectors
                    )
                    .map(
                        ([key, item]) => (
                            `${key}=${item}`
                        )
                    )
                    .join(", ")
                    || "none"
                ),
            ],
            [
                "Vector axes",
                column.view.vector_axes.join(
                    " × "
                ),
            ],
            [
                "Vector dimension",
                column.view.vector_dim,
            ],
        ];

        if (analysis.value === "Separation") {
            rows.splice(
                6,
                0,
                [
                    "Distance",
                    distance.value,
                ],
                [
                    "Agreement",
                    agreement.value,
                ],
            );

            let relationSpec = relations.get(
                relation.value
            );

            if (
                relationSpec?.type
                    === "metagroup"
                && anchor !== AVG
            ) {
                relationSpec = relations.get(
                    baseRelation(
                        relation.value,
                        anchor,
                    )
                );
            }

            if (
                relationSpec?.type
                    === "relation"
            ) {
                rows.push(
                    [
                        "Positive set",
                        relationSpec.positives.join(
                            ", "
                        ),
                    ],
                    [
                        "Negative set",
                        relationSpec.negatives.join(
                            ", "
                        ),
                    ],
                );
            } else if (
                relationSpec?.type
                    === "metagroup"
            ) {
                rows.push(
                    [
                        "Member relations",
                        relationSpec.members.join(
                            ", "
                        ),
                    ]
                );
            }
        } else {
            rows.splice(
                6,
                0,
                [
                    "Magnitude metric",
                    magnitudeMetric.value,
                ],
            );
        }

        const modelSpec = models.get(
            modelName
        );

        if (
            modelSpec?.type
                === "group_average"
        ) {
            rows.push(
                [
                    "Model-group members",
                    modelSpec.members.join(
                        ", "
                    ),
                ]
            );
        }

        return rows;
    }

    function openDetails(
        modelName,
        anchor,
        column,
        value,
    ) {
        const details = $("details");

        details.replaceChildren();

        for (
            const [key, item]
            of detailRows(
                modelName,
                anchor,
                column,
                value,
            )
        ) {
            const dt = document.createElement(
                "dt"
            );

            const dd = document.createElement(
                "dd"
            );

            dt.textContent = key;
            dd.textContent = item;

            details.append(
                dt,
                dd,
            );
        }

        $("dialog").showModal();
    }

    function render() {
        updateControls();

        const currentColumns = columns();
        let rows;

        if (compare.value === "Datasets") {
            rows = [
                ...relationAnchors(
                    relation.value
                ),
                AVG,
            ].map(
                anchor => ({
                    label: (
                        anchor === AVG
                            ? (
                                relation.value
                                + " average"
                            )
                            : anchor
                    ),
                    model: model.value,
                    anchor,
                    average: anchor === AVG,
                })
            );
        } else {
            const anchor = dataset.value;

            rows = modelOptions()
                .filter(
                    name => availableModel(
                        name,
                        currentColumns,
                        anchor,
                    )
                )
                .map(
                    name => ({
                        label: name,
                        model: name,
                        anchor,
                        average: (
                            models.get(name)?.type
                            === "group_average"
                        ),
                    })
                );
        }

        const table = $("matrix");

        table.replaceChildren();

        const thead = document.createElement(
            "thead"
        );

        const header = document.createElement(
            "tr"
        );

        const first = document.createElement(
            "th"
        );

        first.textContent = (
            compare.value === "Datasets"
                ? "Dataset"
                : "Model"
        );

        header.append(first);

        for (const column of currentColumns) {
            const th = document.createElement(
                "th"
            );

            th.textContent = column.label;
            th.title = column.view.view_id;

            header.append(th);
        }

        thead.append(header);
        table.append(thead);

        const tbody = document.createElement(
            "tbody"
        );

        for (const row of rows) {
            const tr = document.createElement(
                "tr"
            );

            if (row.average) {
                tr.classList.add("average");
            }

            const name = document.createElement(
                "td"
            );

            name.textContent = row.label;
            tr.append(name);

            for (
                const column
                of currentColumns
            ) {
                const value = valueFor(
                    row.model,
                    row.anchor,
                    column,
                );

                const td = document.createElement(
                    "td"
                );

                td.textContent = format(value);

                td.className = (
                    value === null
                        ? "na"
                        : "value"
                );

                if (value !== null) {
                    td.tabIndex = 0;

                    td.addEventListener(
                        "click",
                        () => openDetails(
                            row.model,
                            row.anchor,
                            column,
                            value,
                        ),
                    );

                    td.addEventListener(
                        "keydown",
                        event => {
                            if (
                                event.key
                                === "Enter"
                            ) {
                                openDetails(
                                    row.model,
                                    row.anchor,
                                    column,
                                    value,
                                );
                            }
                        },
                    );
                }

                tr.append(td);
            }

            tbody.append(tr);
        }

        table.append(tbody);

        $("status").textContent = (
            `${rows.length} rows × `
            + `${currentColumns.length} columns`
            + ` · values shown to `
            + `${D.decimal_places} decimals; `
            + `full precision is retained `
            + `in the embedded data.`
        );
    }

    setSelect(
        agreement,
        D.separation_metrics,
        D.separation_metrics[0],
    );

    setSelect(
        layer,
        [
            ...Array(12).keys(),
        ].map(String),
        "0",
    );

    setChoices(
        relation,
        "relation-list",
        D.relation_order,
        D.relation_order[0],
    );

    setChoices(
        component,
        "component-list",
        [
            ALL,
            ...D.components.map(
                item => item.name
            ),
        ],
        ALL,
    );

    setChoices(
        model,
        "model-list",
        modelOptions(),
        modelOptions()[0],
    );

    setChoices(
        dataset,
        "dataset-list",
        [
            ...relationAnchors(
                relation.value
            ),
            AVG,
        ],
        relationAnchors(
            relation.value
        )[0],
    );

    setChoices(
        distance,
        "distance-list",
        distanceOptions(),
        distanceOptions()[0],
    );

    setChoices(
        magnitudeMetric,
        "magnitude-list",
        D.magnitude_metrics,
        D.magnitude_metrics[0],
    );

    for (
        const element
        of [
            analysis,
            compare,
            signature,
            representation,
            distance,
            agreement,
            magnitudeMetric,
            relation,
            model,
            dataset,
            component,
            view,
            layer,
        ]
    ) {
        element.addEventListener(
            "change",
            () => {
                if (
                    element.tagName === "INPUT"
                ) {
                    const list = (
                        element.getAttribute(
                            "list"
                        )
                    );

                    if (list) {
                        validateInput(
                            element,
                            list,
                        );
                    }
                }

                render();
            },
        );
    }

    $("closeDialog").onclick = () => (
        $("dialog").close()
    );

    $("dialog").addEventListener(
        "click",
        event => {
            if (
                event.target
                === $("dialog")
            ) {
                $("dialog").close();
            }
        },
    );

    $("loading").classList.add("hidden");
    $("app").classList.remove("hidden");

    render();
})().catch(
    error => {
        console.error(error);

        document.getElementById(
            "loading"
        ).textContent = (
            "Failed to load analysis: "
            + error.message
        );
    }
);
</script>
</body>
</html>"""

    def _save_html(
        self,
        separation,
        magnitude,
    ):
        payload = json.dumps(
            self._payload(
                separation,
                magnitude,
            ),
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

        encoded = base64.b64encode(
            gzip.compress(
                payload,
                compresslevel=9,
            )
        ).decode("ascii")

        html = self._html_template().replace(
            "__PAYLOAD__",
            encoded,
        )

        self.save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = self.save_dir / "analysis.html"

        path.write_text(
            html,
            encoding="utf-8",
        )

        return path

    def analyze(self):
        self._log("Starting VTAB functional-semantic analysis.")

        raw = self._load_signatures()

        representations = {
            "raw": raw,
            "delta": self._make_delta_signatures(raw),
        }

        separation = self._compute_separation(
            representations,
            raw,
        )

        magnitude = self._compute_magnitude(
            representations,
            raw,
        )

        path = self._save_html(
            separation,
            magnitude,
        )

        self._log(f"Analysis complete. " f"Results saved to: {path}")

        return {
            "html": path,
            "magnitude": magnitude,
            "separation": separation,
        }
