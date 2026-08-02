import csv
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS
from tqdm.auto import tqdm


class FunctionalSignaturesAnalyzer:
    """Analyze F, P, and derived S functional signatures."""

    PROFILES = ("F", "P", "S")
    DISTANCE_METRICS = {
        "pairwise_normalized_cosine",
        "raw_cosine",
        "sqrt_jensen_shannon",
    }
    MAGNITUDE_STATS = (
        "total_mass",
        "mean_mass_per_coordinate",
        "effective_active_coordinates",
    )

    def __init__(
        self,
        model_names,
        model_groups,
        dataset_names,
        dataset_groups,
        signature_paths,
        signature_keys,
        save_dir,
        eps=1e-12,
        create_plots=False,
        verbose=True,
    ):
        self.model_names = list(model_names)
        self.model_groups = model_groups
        self.dataset_names = list(dataset_names)
        self.dataset_groups = dataset_groups
        self.signature_paths = np.asarray(signature_paths, dtype=object)
        self.save_dir = Path(save_dir)
        self.eps = float(eps)
        self.create_plots = bool(create_plots)
        self.verbose = bool(verbose)

        if isinstance(signature_keys, str):
            signature_keys = [signature_keys]
        self.keys = {}
        for key in signature_keys:
            families = tuple(x.strip() for x in key.split(",") if x.strip())
            name = "+".join(families)
            if not families:
                raise ValueError(f"Invalid signature key: {key!r}.")
            if name in self.keys:
                raise ValueError(f"Duplicate signature key: {name!r}.")
            self.keys[name] = families

        self.families = tuple(
            dict.fromkeys(
                family for families in self.keys.values() for family in families
            )
        )

    def _log(self, message):
        if self.verbose:
            print(message, flush=True)

    def _clean(self, x, name, path):
        x = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x)):
            raise ValueError(f"Non-finite values in '{name}' from {path}.")
        if np.any(x < -self.eps):
            raise ValueError(f"Negative values in '{name}' from {path}.")
        return np.where(x > self.eps, x, 0.0)

    def _load_profiles(self):
        profiles = {profile: {} for profile in self.PROFILES}
        total = len(self.model_names) * len(self.dataset_names)

        with tqdm(
            total=total,
            desc="Loading profiles",
            disable=not self.verbose,
        ) as progress:
            for model_idx, model in enumerate(self.model_names):
                loaded = {
                    profile: {family: [] for family in self.families}
                    for profile in self.PROFILES
                }

                for path in self.signature_paths[model_idx]:
                    with np.load(path) as data:
                        for family in self.families:
                            f_key, p_key = f"F__{family}", f"P__{family}"
                            if f_key not in data or p_key not in data:
                                missing = [k for k in (f_key, p_key) if k not in data]
                                raise KeyError(f"Missing {missing} in {path}.")

                            f = self._clean(data[f_key], f_key, path)
                            p = self._clean(data[p_key], p_key, path)
                            if f.shape != p.shape:
                                raise ValueError(
                                    f"Shape mismatch for family '{family}' in {path}: "
                                    f"F{f.shape} vs P{p.shape}."
                                )

                            s = np.divide(f, p, out=np.zeros_like(f), where=p > 0)
                            if not np.all(np.isfinite(s)):
                                raise ValueError(
                                    f"Non-finite S values for family '{family}' in {path}."
                                )
                            loaded["F"][family].append(f)
                            loaded["P"][family].append(p)
                            loaded["S"][family].append(s)

                    progress.update()

                for profile in self.PROFILES:
                    profiles[profile][model] = {
                        family: np.stack(values)
                        for family, values in loaded[profile].items()
                    }

        return profiles

    @staticmethod
    def _zero_case(a, b):
        za, zb = not np.any(a), not np.any(b)
        if za and zb:
            return 0.0
        if za or zb:
            return 1.0
        return None

    @staticmethod
    def _cosine_distance(a, b):
        value = 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return 0.0 if abs(value) < 1e-15 else float(np.clip(value, 0.0, 1.0))

    @classmethod
    def _pairwise_normalized_cosine(cls, a, b):
        zero = cls._zero_case(a, b)
        if zero is not None:
            return zero
        total = a + b
        x = np.divide(a, total, out=np.zeros_like(a), where=total > 0)
        y = np.divide(b, total, out=np.zeros_like(b), where=total > 0)
        return cls._cosine_distance(x, y)

    @classmethod
    def _raw_cosine(cls, a, b):
        zero = cls._zero_case(a, b)
        return zero if zero is not None else cls._cosine_distance(a, b)

    @classmethod
    def _sqrt_jensen_shannon(cls, a, b):
        zero = cls._zero_case(a, b)
        if zero is not None:
            return zero
        p, q = a / a.sum(), b / b.sum()
        m = 0.5 * (p + q)
        js = 0.5 * (
            np.sum(p[p > 0] * np.log(p[p > 0] / m[p > 0]))
            + np.sum(q[q > 0] * np.log(q[q > 0] / m[q > 0]))
        )
        return float(np.sqrt(np.clip(js / np.log(2.0), 0.0, 1.0)))

    def _distance(self, metric, a, b):
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        return getattr(self, f"_{metric}")(a, b)

    def _combined_distance(
        self,
        signatures,
        metric,
        model_a,
        model_b,
        dataset_a,
        dataset_b,
        families,
        layer=None,
    ):
        distances = []
        for family in families:
            a = signatures[model_a][family][dataset_a]
            b = signatures[model_b][family][dataset_b]
            if layer is not None:
                a, b = a[layer], b[layer]
            distances.append(self._distance(metric, a, b))
        return float(np.sqrt(np.mean(np.square(distances))))

    def _distance_matrix(self, signatures, metric, model, families, layer=None):
        n = len(self.dataset_names)
        matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i):
                matrix[i, j] = matrix[j, i] = self._combined_distance(
                    signatures, metric, model, model, i, j, families, layer
                )
        return matrix

    def _compute_distances(self, profiles):
        results = {}
        total = (
            len(self.DISTANCE_METRICS)
            * len(self.PROFILES)
            * len(self.model_names)
            * len(self.keys)
        )

        with tqdm(
            total=total,
            desc="Computing distances",
            disable=not self.verbose,
        ) as progress:
            for metric in self.DISTANCE_METRICS:
                results[metric] = {}
                for profile in self.PROFILES:
                    signatures = profiles[profile]
                    results[metric][profile] = {}

                    for model in self.model_names:
                        results[metric][profile][model] = {}
                        for key, families in self.keys.items():
                            n_layers = signatures[model][families[0]].shape[1]
                            if any(
                                signatures[model][family].shape[1] != n_layers
                                for family in families
                            ):
                                raise ValueError(
                                    f"Families in '{key}' have different layer counts."
                                )

                            results[metric][profile][model][key] = {
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
                                f"{metric} | {profile} | {model} | {key}",
                                refresh=False,
                            )
                            progress.update()
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
        for i, name in enumerate(self.dataset_names):
            ax.annotate(name, xy[i], xytext=(4, 4), textcoords="offset points")
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
            for metrics in distances.values()
            for profiles in metrics.values()
            for keys in profiles.values()
            for matrices in keys.values()
        )
        with tqdm(
            total=total,
            desc="Saving distance plots",
            disable=not self.verbose,
        ) as progress:
            for metric, profiles in distances.items():
                for profile, models in profiles.items():
                    for model, keys in models.items():
                        for key, matrices in keys.items():
                            out = (
                                self.save_dir
                                / "distance_analysis"
                                / metric
                                / profile
                                / model
                                / key
                                / "plots"
                            )
                            self._plot_distance(
                                matrices["global"],
                                out / "global.png",
                                f"{metric} | {profile} | {model} | {key} | global",
                            )
                            progress.update()
                            for layer, matrix in enumerate(matrices["layers"]):
                                self._plot_distance(
                                    matrix,
                                    out / f"layer_{layer:02d}.png",
                                    f"{metric} | {profile} | {model} | {key} | layer {layer:02d}",
                                )
                                progress.update()

    @staticmethod
    def _pair_concordance(matrix, anchor, closer, farther):
        scores = [
            (
                1.0
                if matrix[anchor, a] < matrix[anchor, b]
                else 0.5 if matrix[anchor, a] == matrix[anchor, b] else 0.0
            )
            for a in closer
            for b in farther
        ]
        return float(np.mean(scores)) if scores else np.nan

    def _group_separation(self, matrix, group):
        index = {name: i for i, name in enumerate(self.dataset_names)}
        hierarchy = [
            [index[name] for name in tier] for tier in self.dataset_groups[group]
        ]
        positive = hierarchy[0]
        anchor_scores = []

        for anchor in positive:
            tiers = [[i for i in positive if i != anchor], *hierarchy[1:]]
            scores = [
                self._pair_concordance(matrix, anchor, tiers[i], tiers[j])
                for i in range(len(tiers))
                for j in range(i + 1, len(tiers))
                if tiers[i] and tiers[j]
            ]
            if scores:
                anchor_scores.append(np.mean(scores))

        return float(np.mean(anchor_scores)) if anchor_scores else np.nan

    def _save_separation(self, distances):
        groups = list(self.dataset_groups)
        total = (
            len(self.DISTANCE_METRICS)
            * len(self.PROFILES)
            * len(self.model_names)
            * len(self.keys)
        )

        with tqdm(
            total=total,
            desc="Computing separation",
            disable=not self.verbose,
        ) as progress:
            for metric, profiles in distances.items():
                for profile, models in profiles.items():
                    summary = []
                    root = self.save_dir / "distance_analysis" / metric / profile

                    for model, keys in models.items():
                        for key, matrices in keys.items():
                            variants = [("global", matrices["global"])] + [
                                (f"layer_{layer:02d}", matrix)
                                for layer, matrix in enumerate(matrices["layers"])
                            ]
                            scores = {
                                group: [
                                    self._group_separation(matrix, group)
                                    for _, matrix in variants
                                ]
                                for group in groups
                            }

                            out = root / model / key
                            out.mkdir(parents=True, exist_ok=True)
                            with open(out / "separation.csv", "w", newline="") as file:
                                writer = csv.writer(file)
                                writer.writerow(
                                    ["group_name", *[name for name, _ in variants]]
                                )
                                for group in groups:
                                    writer.writerow([group, *scores[group]])

                            summary.append(
                                [model, key, *[scores[group][0] for group in groups]]
                            )
                            progress.update()

                    root.mkdir(parents=True, exist_ok=True)
                    with open(root / "separation.csv", "w", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(["model_name", "signature_key", *groups])
                        writer.writerows(summary)

    def _save_distance_ratios(self, profiles, distances):
        total = (
            len(self.DISTANCE_METRICS)
            * len(self.PROFILES)
            * len(self.model_groups)
            * len(self.keys)
        )

        with tqdm(
            total=total,
            desc="Computing distance ratios",
            disable=not self.verbose,
        ) as progress:
            for metric in self.DISTANCE_METRICS:
                for profile in self.PROFILES:
                    signatures = profiles[profile]
                    rows = []

                    for group, hierarchy in self.model_groups.items():
                        models = list(hierarchy[0])
                        if len(models) < 2:
                            raise ValueError(
                                f"Model group '{group}' needs at least two models."
                            )
                        unknown = set(models) - set(self.model_names)
                        if unknown:
                            raise ValueError(
                                f"Unknown models in '{group}': {sorted(unknown)}."
                            )

                        model_pairs = list(combinations(models, 2))
                        for key, families in self.keys.items():
                            model_distances = []
                            task_distances = []
                            task_ratios = []

                            for task in range(len(self.dataset_names)):
                                model_distance = np.mean(
                                    [
                                        self._combined_distance(
                                            signatures,
                                            metric,
                                            a,
                                            b,
                                            task,
                                            task,
                                            families,
                                        )
                                        for a, b in model_pairs
                                    ]
                                )
                                task_distance = np.mean(
                                    [
                                        distances[metric][profile][model][key][
                                            "global"
                                        ][task, other]
                                        for model in models
                                        for other in range(len(self.dataset_names))
                                        if other != task
                                    ]
                                )
                                model_distances.append(model_distance)
                                task_distances.append(task_distance)
                                task_ratios.append(
                                    model_distance / task_distance
                                    if task_distance > 0
                                    else np.nan
                                )

                            mean_task_distance = np.mean(task_distances)
                            global_ratio = (
                                np.mean(model_distances) / mean_task_distance
                                if mean_task_distance > 0
                                else np.nan
                            )
                            rows.append([group, key, global_ratio, *task_ratios])
                            progress.update()

                    out = self.save_dir / "distance_analysis" / metric / profile
                    out.mkdir(parents=True, exist_ok=True)
                    with open(
                        out / "model_to_task_distance_ratio.csv", "w", newline=""
                    ) as file:
                        writer = csv.writer(file)
                        writer.writerow(
                            [
                                "model_group",
                                "signature_key",
                                "global",
                                *self.dataset_names,
                            ]
                        )
                        writer.writerows(rows)

    @staticmethod
    def _magnitude(x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        total = float(x.sum())
        squared = float(np.dot(x, x))
        return (
            total,
            total / x.size if x.size else np.nan,
            total * total / squared if squared > 0 else 0.0,
        )

    def _magnitude_values(self, signatures, model, family):
        values = {}
        for dataset_idx, dataset in enumerate(self.dataset_names):
            x = signatures[model][family][dataset_idx]
            values[dataset] = {
                "global": self._magnitude(x),
                "layers": [self._magnitude(layer) for layer in x],
            }
        return values

    def _plot_magnitude(self, values, variant, path, title):
        positions = np.arange(len(self.dataset_names))
        if variant == "global":
            rows = [values[name]["global"] for name in self.dataset_names]
        else:
            layer = int(variant.split("_")[1])
            rows = [values[name]["layers"][layer] for name in self.dataset_names]
        rows = np.asarray(rows)

        fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
        labels = (
            "Total mass",
            "Mean mass per coordinate",
            "Effective active coordinates",
        )
        for idx, (ax, label) in enumerate(zip(axes, labels)):
            ax.bar(positions, rows[:, idx])
            ax.set_ylabel(label)
        axes[0].set_title(title)
        axes[-1].set_xticks(positions, self.dataset_names, rotation=45, ha="right")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def _save_magnitude(self, profiles):
        total = len(self.PROFILES) * len(self.model_names) * len(self.families)
        with tqdm(
            total=total,
            desc="Computing magnitude",
            disable=not self.verbose,
        ) as progress:
            for profile in self.PROFILES:
                signatures = profiles[profile]
                root = self.save_dir / "magnitude_analysis" / profile
                summary = []

                for model in self.model_names:
                    for family in self.families:
                        values = self._magnitude_values(signatures, model, family)
                        n_layers = len(next(iter(values.values()))["layers"])
                        variants = ["global"] + [
                            f"layer_{layer:02d}" for layer in range(n_layers)
                        ]
                        out = root / model / family
                        out.mkdir(parents=True, exist_ok=True)

                        with open(out / "magnitude.csv", "w", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow(["dataset_name", "statistic", *variants])
                            for dataset in self.dataset_names:
                                global_values = values[dataset]["global"]
                                layer_values = values[dataset]["layers"]
                                for stat_idx, stat in enumerate(self.MAGNITUDE_STATS):
                                    writer.writerow(
                                        [
                                            dataset,
                                            stat,
                                            global_values[stat_idx],
                                            *[x[stat_idx] for x in layer_values],
                                        ]
                                    )

                                summary.append([model, dataset, family, *global_values])

                        if self.create_plots:
                            plot_dir = out / "plots"
                            self._plot_magnitude(
                                values,
                                "global",
                                plot_dir / "global.png",
                                f"{profile} | {model} | {family} | global",
                            )
                            for layer in range(n_layers):
                                variant = f"layer_{layer:02d}"
                                self._plot_magnitude(
                                    values,
                                    variant,
                                    plot_dir / f"{variant}.png",
                                    f"{profile} | {model} | {family} | layer {layer:02d}",
                                )

                        progress.update()

                root.mkdir(parents=True, exist_ok=True)
                with open(root / "magnitude.csv", "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [
                            "model_name",
                            "dataset_name",
                            "family",
                            *self.MAGNITUDE_STATS,
                        ]
                    )
                    writer.writerows(summary)

    def analyze(self):
        self._log("Starting functional-signature analysis.")
        profiles = self._load_profiles()
        distances = self._compute_distances(profiles)
        self._save_distance_plots(distances)
        self._save_separation(distances)
        self._save_distance_ratios(profiles, distances)
        self._save_magnitude(profiles)
        self._log(f"Analysis complete. Results saved to: {self.save_dir}")
        return distances
