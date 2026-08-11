import csv
import math
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
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

    AGREEMENT_METRICS = (
        "ranking_agreement",
        "geometric_agreement",
    )

    RAW_METRICS = {
        "F": (
            "hellinger",
            "pairwise_normalized_cosine",
        ),
        "P": ("hellinger",),
        "S": ("hellinger",),
    }

    DELTA_METRICS = {signature: ("angular",) for signature in SIGNATURES}

    RESULT_META_COLUMNS = (
        "representation",
        "signature",
        "distance_metric",
        "model_name",
        "family",
        "component",
        "view",
        "view_id",
        "layer",
        "site",
        "sublayer",
        "token_group",
        "query_group",
        "destination_group",
        "route",
        "vector_axes",
        "vector_dim",
    )

    DEPTH_REPORTS = (
        (
            "sublayer_update",
            "layer",
            (),
            "Sublayer balance by layer",
        ),
        (
            "sublayer_token_update",
            "layer_sublayer",
            ("sublayer",),
            "CLS/patch balance by layer and sublayer",
        ),
        (
            "sublayer_token_update",
            "layer_token_group",
            ("token_group",),
            "Attention/MLP balance by layer and token group",
        ),
        (
            "attention_head_update",
            "layer",
            (),
            "Attention-head population by layer",
        ),
        (
            "residual_stream_channel",
            "layer_site_token_group",
            ("site", "token_group"),
            "Residual flow by layer, site, and token group",
        ),
        (
            "attention_qk_route",
            "layer_route",
            ("route",),
            "QK routing by layer and route",
        ),
        (
            "attention_ov_route",
            "layer_route",
            ("route",),
            "OV transport by layer and route",
        ),
        (
            "attention_qk_channel",
            "layer_query_group",
            ("query_group",),
            "QK content by layer and query group",
        ),
        (
            "attention_ov_channel",
            "layer_destination_group",
            ("destination_group",),
            "OV content by layer and destination",
        ),
        (
            "mlp_neuron",
            "layer_token_group",
            ("token_group",),
            "MLP population by layer and token group",
        ),
        (
            "prompt_to_backbone_qk_route",
            "layer_destination_group",
            ("destination_group",),
            "Prompt QK influence by layer and destination",
        ),
        (
            "prompt_to_backbone_ov_route",
            "layer_destination_group",
            ("destination_group",),
            "Prompt OV influence by layer and destination",
        ),
    )

    def __init__(
        self,
        model_names,
        model_groups,
        signature_paths,
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
                f"Each model must have {len(self.DATASET_NAMES)} paths "
                "ordered as DATASET_NAMES."
            )

        average_names = set()

        for name, models in self.model_groups.items():
            if not models or len(set(models)) != len(models):
                raise ValueError(
                    f"Model group '{name}' must be non-empty "
                    "and contain unique models."
                )

            unknown = set(models) - set(self.model_names)

            if unknown:
                raise ValueError(f"Unknown models in '{name}': {sorted(unknown)}.")

            average = f"{name}_average"

            if average in self.model_names or average in average_names:
                raise ValueError(f"Conflicting model-group average name: {average}.")

            average_names.add(average)

        self.components = tuple(self.COMPONENTS)
        self.dataset_index = {name: i for i, name in enumerate(self.DATASET_NAMES)}
        self.views = self._build_views()

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
                value_lists = (
                    self._axis_values(component, axis) for axis in fixed_axes
                )

                for values in product(*value_lists):
                    selectors = dict(
                        zip(
                            fixed_axes,
                            values,
                        )
                    )

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
                            "view_id": (f"{component}/" f"{view}/" f"{suffix}"),
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
                                key = f"{signature}__" f"{component}"

                                if key not in data:
                                    raise KeyError(f"Missing '{key}' in {path}.")

                                x = np.asarray(
                                    data[key],
                                    dtype=np.float64,
                                )

                                if x.shape != expected:
                                    raise ValueError(
                                        f"Invalid shape for "
                                        f"'{key}' in {path}: "
                                        f"{x.shape}, expected "
                                        f"{expected}."
                                    )

                                if not np.all(np.isfinite(x)):
                                    raise ValueError(
                                        f"Non-finite values "
                                        f"in '{key}' from "
                                        f"{path}."
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

    @staticmethod
    def _magnitude_stats(x):
        x = np.abs(
            np.asarray(
                x,
                dtype=np.float64,
            ).reshape(-1)
        )

        l1 = float(x.sum())

        if not l1:
            return (
                0.0,
                float(x.mean()),
                0.0,
            )

        p = x[x > 0] / l1

        support = float(np.exp(-np.sum(p * np.log(p))) / x.size)

        return (
            l1,
            float(x.mean()),
            support,
        )

    def _compute_magnitude(self, representations, raw):
        scopes = {
            "vtab_mean": np.arange(len(self.DATASET_NAMES) - 1),
            "imagenet_val": np.asarray([len(self.DATASET_NAMES) - 1]),
        }

        rows = []

        for representation, signature_types in representations.items():
            for signature, models in signature_types.items():
                reference = raw[signature][self.REFERENCE_MODEL]

                for model, components in models.items():
                    has_prompts = self._has_prompts(
                        raw,
                        model,
                    )

                    for component, values in components.items():
                        if component in self.PROMPT_COMPONENTS and not has_prompts:
                            continue

                        shape, axes = self.COMPONENTS[component]

                        variants = [
                            (
                                "full",
                                "",
                                values.reshape(
                                    len(values),
                                    -1,
                                ),
                            )
                        ]

                        if axes[0] == "layer":
                            variants += [
                                (
                                    "layer",
                                    layer,
                                    values[
                                        :,
                                        layer,
                                    ].reshape(
                                        len(values),
                                        -1,
                                    ),
                                )
                                for layer in range(shape[0])
                            ]

                        for view, layer, vectors in variants:
                            ref = reference[component]

                            if view == "layer":
                                ref = ref[:, layer]

                            ref_l1 = (
                                np.abs(ref)
                                .reshape(
                                    len(ref),
                                    -1,
                                )
                                .sum(axis=1)
                            )

                            delta_l1 = np.abs(vectors).sum(axis=1)

                            for scope, indices in scopes.items():
                                stats = np.asarray(
                                    [
                                        self._magnitude_stats(vectors[i])
                                        for i in indices
                                    ],
                                    dtype=np.float64,
                                )

                                ratio = np.divide(
                                    delta_l1[indices],
                                    ref_l1[indices],
                                    out=np.full(
                                        len(indices),
                                        np.nan,
                                        dtype=np.float64,
                                    ),
                                    where=(ref_l1[indices] > 0),
                                )

                                finite = ratio[np.isfinite(ratio)]

                                rows.append(
                                    {
                                        "representation": representation,
                                        "signature": signature,
                                        "model_name": model,
                                        "component": component,
                                        "view": view,
                                        "layer": layer,
                                        "dataset_scope": scope,
                                        "l1_norm": float(
                                            stats[
                                                :,
                                                0,
                                            ].mean()
                                        ),
                                        "mean_absolute_value": float(
                                            stats[
                                                :,
                                                1,
                                            ].mean()
                                        ),
                                        "effective_support_fraction": float(
                                            stats[
                                                :,
                                                2,
                                            ].mean()
                                        ),
                                        "relative_l1_change": (
                                            float(finite.mean())
                                            if (
                                                representation == "delta"
                                                and finite.size
                                            )
                                            else ""
                                        ),
                                    }
                                )

        return rows

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
                f"has dimension "
                f"{vectors.shape[1]}, expected "
                f"{view['vector_dim']}."
            )

        return vectors

    @staticmethod
    def _hellinger_matrix(vectors):
        if np.any(vectors < 0):
            raise ValueError("Hellinger distance requires " "nonnegative signatures.")

        sums = vectors.sum(axis=1)
        valid = sums > 0

        roots = np.zeros_like(vectors)

        roots[valid] = np.sqrt(
            vectors[valid]
            / sums[
                valid,
                None,
            ]
        )

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

        matrix[
            ~valid,
            :,
        ] = 1.0

        matrix[
            :,
            ~valid,
        ] = 1.0

        matrix[
            np.ix_(
                ~valid,
                ~valid,
            )
        ] = 0.0

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

        normalized[valid] = (
            vectors[valid]
            / norms[
                valid,
                None,
            ]
        )

        similarity = np.clip(
            normalized @ normalized.T,
            -1.0,
            1.0,
        )

        matrix = np.arccos(similarity) / np.pi

        matrix[
            ~valid,
            :,
        ] = 1.0

        matrix[
            :,
            ~valid,
        ] = 1.0

        matrix[
            np.ix_(
                ~valid,
                ~valid,
            )
        ] = 0.0

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
                    distance = 0.0 if (a_zero and b_zero) else 1.0
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

                matrix[
                    i,
                    j,
                ] = distance

                matrix[
                    j,
                    i,
                ] = distance

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

        anchors = [self.dataset_index[x] for x in spec["anchors"]]

        positives = [self.dataset_index[x] for x in spec["positives"]]

        negatives = [self.dataset_index[x] for x in spec["negatives"]]

        ranking = []
        geometric = []

        for anchor in anchors:
            peers = [x for x in positives if x != anchor]

            if not peers or not negatives:
                raise ValueError(f"Group '{group}' has no " "valid comparisons.")

            positive = matrix[
                anchor,
                peers,
            ]

            negative = matrix[
                anchor,
                negatives,
            ]

            p = positive[
                :,
                None,
            ]

            n = negative[
                None,
                :,
            ]

            ranking.append(
                float(((p < n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size))
            )

            d_pos = float(positive.mean())

            d_neg = float(negative.mean())

            denominator = max(
                d_pos,
                d_neg,
            )

            geometric.append((d_neg - d_pos) / denominator if denominator else 0.0)

        return (
            float(np.mean(ranking)),
            float(np.mean(geometric)),
        )

    def _separation_scores(
        self,
        matrix,
    ):
        scores = {
            group: self._group_scores(
                matrix,
                group,
            )
            for group in self.GROUPS
        }

        for name, groups in self.METAGROUPS.items():
            scores[name] = tuple(
                float(np.mean([scores[group][i] for group in groups])) for i in range(2)
            )

        return {
            f"{group}__{metric}": scores[group][i]
            for group in self.REPORT_GROUPS
            for i, metric in enumerate(self.AGREEMENT_METRICS)
        }

    def _result_row(
        self,
        representation,
        signature,
        metric,
        model,
        view,
        scores,
    ):
        selectors = view["selectors"]

        row = {
            "representation": representation,
            "signature": signature,
            "distance_metric": metric,
            "model_name": model,
            "family": view["family"],
            "component": view["component"],
            "view": view["view"],
            "view_id": view["view_id"],
            **{
                axis: selectors.get(
                    axis,
                    "",
                )
                for axis in (
                    "layer",
                    "site",
                    "sublayer",
                    "token_group",
                    "query_group",
                    "destination_group",
                    "route",
                )
            },
            "vector_axes": "|".join(view["vector_axes"]),
            "vector_dim": view["vector_dim"],
        }

        row.update(scores)

        return row

    def _compute_separation(
        self,
        representations,
        raw,
    ):
        policy = {
            "raw": self.RAW_METRICS,
            "delta": self.DELTA_METRICS,
        }

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

        rows = []

        with tqdm(
            total=total,
            desc="Computing semantic views",
            disable=not self.verbose,
        ) as pbar:

            for representation, signature_metrics in policy.items():
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

                                scores = self._separation_scores(matrix)

                                rows.append(
                                    self._result_row(
                                        representation,
                                        signature,
                                        metric,
                                        model,
                                        view,
                                        scores,
                                    )
                                )

                                pbar.update()

        return rows + self._model_group_averages(rows)

    def _score_columns(self):
        return tuple(
            f"{group}__{metric}"
            for group in self.REPORT_GROUPS
            for metric in self.AGREEMENT_METRICS
        )

    def _model_group_averages(
        self,
        rows,
    ):
        score_columns = self._score_columns()

        row_map = {
            (
                row["representation"],
                row["signature"],
                row["distance_metric"],
                row["model_name"],
                row["view_id"],
            ): row
            for row in rows
        }

        configs = sorted(
            {
                (
                    row["representation"],
                    row["signature"],
                    row["distance_metric"],
                )
                for row in rows
            }
        )

        averages = []

        for representation, signature, metric in configs:
            for group_name, models in self.model_groups.items():
                view_sets = [
                    {
                        row["view_id"]
                        for row in rows
                        if (
                            row["representation"] == representation
                            and row["signature"] == signature
                            and row["distance_metric"] == metric
                            and row["model_name"] == model
                        )
                    }
                    for model in models
                ]

                common = set.intersection(*view_sets) if view_sets else set()

                for view_id in sorted(common):
                    members = [
                        row_map[
                            (
                                representation,
                                signature,
                                metric,
                                model,
                                view_id,
                            )
                        ]
                        for model in models
                    ]

                    row = dict(members[0])

                    row["model_name"] = f"{group_name}_average"

                    for column in score_columns:
                        row[column] = float(
                            np.mean([member[column] for member in members])
                        )

                    averages.append(row)

        return averages

    @staticmethod
    def _write_csv(
        path,
        rows,
        columns,
    ):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Remove only columns unused by every row of this specific CSV.
        if rows:
            columns = tuple(
                column
                for column in columns
                if any(
                    row.get(
                        column,
                        "",
                    )
                    not in (
                        "",
                        None,
                    )
                    for row in rows
                )
            )

        with open(
            path,
            "w",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=columns,
                extrasaction="ignore",
            )

            writer.writeheader()
            writer.writerows(rows)

    def _save_metadata(self):
        root = self.save_dir / "metadata"

        component_rows = [
            {
                "component": component,
                "family": self.COMPONENT_FAMILY[component],
                "shape": "x".join(
                    map(
                        str,
                        shape,
                    )
                ),
                "axes": "|".join(axes),
            }
            for component, (
                shape,
                axes,
            ) in self.COMPONENTS.items()
        ]

        self._write_csv(
            root / "component_schema.csv",
            component_rows,
            (
                "component",
                "family",
                "shape",
                "axes",
            ),
        )

        selector_axes = (
            "layer",
            "site",
            "sublayer",
            "token_group",
            "query_group",
            "destination_group",
            "route",
        )

        view_rows = []

        for view in self.views:
            view_rows.append(
                {
                    "family": view["family"],
                    "component": view["component"],
                    "view": view["view"],
                    "view_id": view["view_id"],
                    **{
                        axis: view["selectors"].get(
                            axis,
                            "",
                        )
                        for axis in selector_axes
                    },
                    "vector_axes": "|".join(view["vector_axes"]),
                    "vector_dim": view["vector_dim"],
                    "description": view["description"],
                }
            )

        self._write_csv(
            root / "view_schema.csv",
            view_rows,
            (
                "family",
                "component",
                "view",
                "view_id",
                *selector_axes,
                "vector_axes",
                "vector_dim",
                "description",
            ),
        )

        semantic_rows = [
            {
                "group": group,
                "type": "relation",
                "anchors": "|".join(spec["anchors"]),
                "positives": "|".join(spec["positives"]),
                "negatives": "|".join(spec["negatives"]),
                "members": "",
            }
            for group, spec in self.GROUPS.items()
        ]

        semantic_rows += [
            {
                "group": name,
                "type": "metagroup",
                "anchors": "",
                "positives": "",
                "negatives": "",
                "members": "|".join(groups),
            }
            for name, groups in self.METAGROUPS.items()
        ]

        self._write_csv(
            root / "semantic_groups.csv",
            semantic_rows,
            (
                "group",
                "type",
                "anchors",
                "positives",
                "negatives",
                "members",
            ),
        )

        representation_rows = [
            {
                "representation": "raw",
                "definition": (
                    "Absolute functional " "signature for each " "model and task."
                ),
                "prompt_components": (
                    "Absolute prompt-to-" "backbone signal when " "prompts exist."
                ),
            },
            {
                "representation": "delta",
                "definition": (
                    f"Model signature minus "
                    f"{self.REFERENCE_MODEL} "
                    "signature for the same "
                    "task."
                ),
                "prompt_components": (
                    "For prompt-specific "
                    "components the reference "
                    "mechanism is absent, so "
                    "delta equals the introduced "
                    "prompt signal."
                ),
            },
        ]

        self._write_csv(
            root / "representation_schema.csv",
            representation_rows,
            (
                "representation",
                "definition",
                "prompt_components",
            ),
        )

        self._write_csv(
            root / "output_schema.csv",
            [
                {
                    "analysis": "separation",
                    "path": ("separation_analysis/" "results.csv"),
                    "contents": ("All semantic views " "and configurations."),
                },
                {
                    "analysis": "separation",
                    "path": (
                        "separation_analysis/"
                        "overview/{representation}/"
                        "{distance_metric}/"
                        "{signature}.csv"
                    ),
                    "contents": ("Full-component views " "only."),
                },
                {
                    "analysis": "separation",
                    "path": ("separation_analysis/" "{family}/results.csv"),
                    "contents": ("All semantic views " "for one component family."),
                },
                {
                    "analysis": "separation",
                    "path": (
                        "separation_analysis/"
                        "{family}/overview/"
                        "{representation}/"
                        "{distance_metric}/"
                        "{signature}.csv"
                    ),
                    "contents": ("Full-component views " "for one component family."),
                },
                {
                    "analysis": "magnitude",
                    "path": ("magnitude_analysis/" "results.csv"),
                    "contents": ("All full/layer " "magnitude rows."),
                },
                {
                    "analysis": "magnitude",
                    "path": (
                        "magnitude_analysis/"
                        "overview/{representation}/"
                        "{signature}.csv"
                    ),
                    "contents": ("Full-component " "magnitude rows only."),
                },
                {
                    "analysis": "all_csv",
                    "path": "*",
                    "contents": (
                        "Columns empty for every " "row in a CSV are omitted."
                    ),
                },
            ],
            (
                "analysis",
                "path",
                "contents",
            ),
        )

    def _save_magnitude(
        self,
        rows,
    ):
        columns = (
            "representation",
            "signature",
            "model_name",
            "component",
            "view",
            "layer",
            "dataset_scope",
            "l1_norm",
            "mean_absolute_value",
            "effective_support_fraction",
            "relative_l1_change",
        )

        root = self.save_dir / "magnitude_analysis"

        self._write_csv(
            root / "results.csv",
            rows,
            columns,
        )

        full = [row for row in rows if row["view"] == "full"]

        for representation in (
            "raw",
            "delta",
        ):
            for signature in self.SIGNATURES:
                self._write_csv(
                    root / "overview" / representation / f"{signature}.csv",
                    [
                        row
                        for row in full
                        if (
                            row["representation"] == representation
                            and row["signature"] == signature
                        )
                    ],
                    columns,
                )

    def _save_separation_overviews(
        self,
        rows,
        root,
        columns,
    ):
        full = [row for row in rows if row["view"] == "full"]

        for representation, policy in (
            (
                "raw",
                self.RAW_METRICS,
            ),
            (
                "delta",
                self.DELTA_METRICS,
            ),
        ):
            for signature, metrics in policy.items():
                for metric in metrics:
                    self._write_csv(
                        root
                        / "overview"
                        / representation
                        / metric
                        / f"{signature}.csv",
                        [
                            row
                            for row in full
                            if (
                                row["representation"] == representation
                                and row["signature"] == signature
                                and row["distance_metric"] == metric
                            )
                        ],
                        columns,
                    )

    def _save_separation(
        self,
        rows,
    ):
        columns = (
            *self.RESULT_META_COLUMNS,
            *self._score_columns(),
        )

        root = self.save_dir / "separation_analysis"

        self._write_csv(
            root / "results.csv",
            rows,
            columns,
        )

        self._save_separation_overviews(
            rows,
            root,
            columns,
        )

        for family in dict.fromkeys(self.COMPONENT_FAMILY.values()):
            family_rows = [row for row in rows if row["family"] == family]

            family_root = root / family

            self._write_csv(
                family_root / "results.csv",
                family_rows,
                columns,
            )

            self._save_separation_overviews(
                family_rows,
                family_root,
                columns,
            )

    @staticmethod
    def _short_label(
        axis,
        value,
    ):
        names = {
            "attention_read": "attn-read",
            "attention_write": "attn-write",
            "mlp_read": "mlp-read",
            "mlp_write": "mlp-write",
            "attention": "attn",
            "cls<-cls": "CLS<-CLS",
            "cls<-patch": "CLS<-P",
            "patch<-cls": "P<-CLS",
            "patch<-patch_self": "P<-P self",
            "patch<-patch_local": "P<-P local",
            "patch<-patch_medium": "P<-P medium",
            "patch<-patch_global": "P<-P global",
        }

        if axis == "layer":
            return f"L{int(value):02d}"

        return names.get(
            str(value),
            str(value),
        )

    def _native_matrix(
        self,
        rows,
        component,
        view_name,
        col_axes,
        group,
        agreement,
    ):
        selected = [
            row
            for row in rows
            if (row["component"] == component and row["view"] == view_name)
        ]

        if not selected:
            return None, None

        layers = list(range(self.L))

        cols = (
            list(
                product(
                    *(
                        self._axis_values(
                            component,
                            axis,
                        )
                        for axis in col_axes
                    )
                )
            )
            if col_axes
            else [()]
        )

        lookup = {
            (
                int(row["layer"]),
                tuple(row[axis] for axis in col_axes),
            ): row[f"{group}__" f"{agreement}"]
            for row in selected
        }

        matrix = np.full(
            (
                len(layers),
                len(cols),
            ),
            np.nan,
            dtype=np.float64,
        )

        for i, layer in enumerate(layers):
            for j, values in enumerate(cols):
                matrix[
                    i,
                    j,
                ] = lookup.get(
                    (
                        layer,
                        tuple(values),
                    ),
                    np.nan,
                )

        labels = [
            (
                " / ".join(
                    self._short_label(
                        axis,
                        value,
                    )
                    for axis, value in zip(
                        col_axes,
                        values,
                    )
                )
                if col_axes
                else "profile"
            )
            for values in cols
        ]

        return matrix, labels

    @staticmethod
    def _limits(
        agreement,
    ):
        if agreement == "ranking_agreement":
            return (
                0.0,
                1.0,
                0.5,
            )

        return (
            -1.0,
            1.0,
            0.0,
        )

    def _plot_overview_page(
        self,
        pdf,
        rows,
        agreement,
        title,
    ):
        full = [row for row in rows if row["view"] == "full"]

        if not full:
            return

        components = [
            component
            for component in self.components
            if any(row["component"] == component for row in full)
        ]

        lookup = {row["component"]: row for row in full}

        matrix = np.asarray(
            [
                [
                    lookup[component][f"{group}__" f"{agreement}"]
                    for component in components
                ]
                for group in self.REPORT_GROUPS
            ],
            dtype=np.float64,
        )

        vmin, vmax, baseline = self._limits(agreement)

        fig, ax = plt.subplots(figsize=(15, 9))

        image = ax.imshow(
            matrix,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_yticks(
            range(len(self.REPORT_GROUPS)),
            self.REPORT_GROUPS,
        )

        ax.set_xticks(
            range(len(components)),
            components,
            rotation=55,
            ha="right",
        )

        ax.set_title(
            f"{title}\n" f"{agreement.replace('_', ' ')} " f"| neutral={baseline:g}"
        )

        fig.colorbar(
            image,
            ax=ax,
            fraction=0.025,
            pad=0.02,
        )

        fig.tight_layout()

        pdf.savefig(fig)
        plt.close(fig)

    def _plot_native_page(
        self,
        pdf,
        rows,
        spec,
        agreement,
    ):
        (
            component,
            view_name,
            col_axes,
            title,
        ) = spec

        if not any(
            row["component"] == component and row["view"] == view_name for row in rows
        ):
            return

        vmin, vmax, baseline = self._limits(agreement)

        fig, axes = plt.subplots(
            5,
            3,
            figsize=(18, 22),
            squeeze=False,
            constrained_layout=True,
        )

        image = None

        for ax, group in zip(
            axes.flat,
            self.REPORT_GROUPS,
        ):
            matrix, labels = self._native_matrix(
                rows,
                component,
                view_name,
                col_axes,
                group,
                agreement,
            )

            if matrix is None:
                ax.axis("off")
                continue

            if matrix.shape[1] == 1:
                ax.plot(
                    range(self.L),
                    matrix[
                        :,
                        0,
                    ],
                    marker="o",
                    markersize=2,
                )

                ax.axhline(
                    baseline,
                    linewidth=0.8,
                    linestyle="--",
                )

                ax.set_ylim(
                    vmin,
                    vmax,
                )

                ax.set_xticks(
                    range(self.L),
                    [f"L{i:02d}" for i in range(self.L)],
                    rotation=90,
                    fontsize=6,
                )

            else:
                image = ax.imshow(
                    matrix,
                    aspect="auto",
                    vmin=vmin,
                    vmax=vmax,
                )

                ax.set_yticks(
                    range(self.L),
                    [f"L{i:02d}" for i in range(self.L)],
                    fontsize=6,
                )

                ax.set_xticks(
                    range(len(labels)),
                    labels,
                    rotation=60,
                    ha="right",
                    fontsize=6,
                )

            ax.set_title(
                group,
                fontsize=9,
            )

        for ax in axes.flat[len(self.REPORT_GROUPS) :]:
            ax.axis("off")

        if image is not None:
            fig.colorbar(
                image,
                ax=axes.ravel().tolist(),
                fraction=0.012,
                pad=0.01,
            )

        fig.suptitle(
            f"{title}\n" f"{agreement.replace('_', ' ')}",
            fontsize=15,
        )

        pdf.savefig(fig)
        plt.close(fig)

    def _plot_model_overview_page(
        self,
        pdf,
        rows,
        agreement,
        title,
    ):
        models = [
            model
            for model in self.model_names
            if any(row["model_name"] == model for row in rows)
        ]

        components = [
            component
            for component in self.components
            if any(
                row["view"] == "full" and row["component"] == component for row in rows
            )
        ]

        vmin, vmax, baseline = self._limits(agreement)

        fig, axes = plt.subplots(
            5,
            3,
            figsize=(18, 22),
            squeeze=False,
            constrained_layout=True,
        )

        image = None

        for ax, group in zip(
            axes.flat,
            self.REPORT_GROUPS,
        ):
            lookup = {
                (
                    row["model_name"],
                    row["component"],
                ): row[f"{group}__" f"{agreement}"]
                for row in rows
                if row["view"] == "full"
            }

            matrix = np.asarray(
                [
                    [
                        lookup.get(
                            (
                                model,
                                component,
                            ),
                            np.nan,
                        )
                        for component in components
                    ]
                    for model in models
                ],
                dtype=np.float64,
            )

            image = ax.imshow(
                matrix,
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_yticks(
                range(len(models)),
                models,
                fontsize=7,
            )

            ax.set_xticks(
                range(len(components)),
                components,
                rotation=60,
                ha="right",
                fontsize=6,
            )

            ax.set_title(
                group,
                fontsize=9,
            )

        for ax in axes.flat[len(self.REPORT_GROUPS) :]:
            ax.axis("off")

        fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            fraction=0.012,
            pad=0.01,
        )

        fig.suptitle(
            f"{title}\n" f"{agreement.replace('_', ' ')} " f"| neutral={baseline:g}",
            fontsize=15,
        )

        pdf.savefig(fig)
        plt.close(fig)

    def _plot_model_depth_page(
        self,
        pdf,
        rows,
        component,
        agreement,
    ):
        selected = [
            row
            for row in rows
            if (row["component"] == component and row["view"] == "layer")
        ]

        if not selected:
            return

        models = [
            model
            for model in self.model_names
            if any(row["model_name"] == model for row in rows)
        ]

        vmin, vmax, baseline = self._limits(agreement)

        fig, axes = plt.subplots(
            5,
            3,
            figsize=(18, 22),
            squeeze=False,
            constrained_layout=True,
        )

        image = None

        for ax, group in zip(
            axes.flat,
            self.REPORT_GROUPS,
        ):
            lookup = {
                (
                    row["model_name"],
                    int(row["layer"]),
                ): row[f"{group}__" f"{agreement}"]
                for row in selected
            }

            matrix = np.asarray(
                [
                    [
                        lookup.get(
                            (
                                model,
                                layer,
                            ),
                            np.nan,
                        )
                        for layer in range(self.L)
                    ]
                    for model in models
                ],
                dtype=np.float64,
            )

            image = ax.imshow(
                matrix,
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_yticks(
                range(len(models)),
                models,
                fontsize=7,
            )

            ax.set_xticks(
                range(self.L),
                [f"L{i:02d}" for i in range(self.L)],
                rotation=90,
                fontsize=6,
            )

            ax.set_title(
                group,
                fontsize=9,
            )

        for ax in axes.flat[len(self.REPORT_GROUPS) :]:
            ax.axis("off")

        fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            fraction=0.012,
            pad=0.01,
        )

        fig.suptitle(
            f"Cross-model depth: "
            f"{component}\n"
            f"{agreement.replace('_', ' ')} "
            f"| neutral={baseline:g}",
            fontsize=15,
        )

        pdf.savefig(fig)
        plt.close(fig)

    def _save_reports(
        self,
        rows,
    ):
        if not self.create_plots:
            return

        real_rows = [row for row in rows if row["model_name"] in self.model_names]

        configs = sorted(
            {
                (
                    row["representation"],
                    row["distance_metric"],
                    row["signature"],
                )
                for row in real_rows
            }
        )

        for (
            representation,
            metric,
            signature,
        ) in tqdm(
            configs,
            desc=("Saving interpretability " "reports"),
            disable=not self.verbose,
        ):
            config_rows = [
                row
                for row in real_rows
                if (
                    row["representation"] == representation
                    and row["distance_metric"] == metric
                    and row["signature"] == signature
                )
            ]

            for model in self.model_names:
                model_rows = [row for row in config_rows if row["model_name"] == model]

                if not model_rows:
                    continue

                path = (
                    self.save_dir
                    / "separation_analysis"
                    / "reports"
                    / "by_model"
                    / representation
                    / metric
                    / signature
                    / f"{model}.pdf"
                )

                path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                with PdfPages(path) as pdf:
                    for agreement in self.AGREEMENT_METRICS:
                        self._plot_overview_page(
                            pdf,
                            model_rows,
                            agreement,
                            (
                                f"{model} | "
                                f"{representation} | "
                                f"{signature} | "
                                f"{metric}"
                            ),
                        )

                        for spec in self.DEPTH_REPORTS:
                            self._plot_native_page(
                                pdf,
                                model_rows,
                                spec,
                                agreement,
                            )

            path = (
                self.save_dir
                / "separation_analysis"
                / "reports"
                / "model_comparison"
                / representation
                / metric
                / f"{signature}.pdf"
            )

            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with PdfPages(path) as pdf:
                for agreement in self.AGREEMENT_METRICS:
                    self._plot_model_overview_page(
                        pdf,
                        config_rows,
                        agreement,
                        (
                            "Model comparison | "
                            f"{representation} | "
                            f"{signature} | "
                            f"{metric}"
                        ),
                    )

                    for component in self.components:
                        axes = self.COMPONENTS[component][1]

                        if axes[0] == "layer":
                            self._plot_model_depth_page(
                                pdf,
                                config_rows,
                                component,
                                agreement,
                            )

    def analyze(self):
        self._log("Starting VTAB functional-semantic analysis.")

        raw = self._load_signatures()

        representations = {
            "raw": raw,
            "delta": self._make_delta_signatures(raw),
        }

        magnitude = self._compute_magnitude(
            representations,
            raw,
        )

        separation = self._compute_separation(
            representations,
            raw,
        )

        self._save_metadata()
        self._save_magnitude(magnitude)
        self._save_separation(separation)
        self._save_reports(separation)

        self._log("Analysis complete. " f"Results saved to: " f"{self.save_dir}")

        return {
            "magnitude": magnitude,
            "separation": separation,
        }
