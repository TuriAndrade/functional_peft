import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .transformer_backbone import TransformerBackbone


class PEFTModel(nn.Module, ABC):
    """Shared PEFT model and functional-signature framework."""

    def __init__(
        self,
        backbone: TransformerBackbone,
        n_classes: int,
        freeze_backbone: bool = True,
    ):
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")

        super().__init__()

        self.backbone = backbone
        self.n_classes = int(n_classes)
        self.is_backbone_frozen = bool(freeze_backbone)

        if freeze_backbone:
            self.freeze_backbone()

        self.head = nn.Linear(self.backbone.embed_dim, n_classes)
        self.init_head()

    # ======================================================================
    # Model interface
    # ======================================================================

    def freeze_backbone(self) -> None:
        self.backbone.freeze()
        self.backbone.eval()
        self.is_backbone_frozen = True

    def unfreeze_backbone(self) -> None:
        self.backbone.unfreeze()
        self.backbone.train(self.training)
        self.is_backbone_frozen = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.is_backbone_frozen:
            self.backbone.eval()

        return self

    def eval(self):
        return self.train(False)

    def init_head(
        self,
        init_from: nn.Linear | None = None,
    ) -> None:
        if init_from is None:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            return

        if (
            init_from.weight.shape != self.head.weight.shape
            or init_from.bias is None
            or init_from.bias.shape != self.head.bias.shape
        ):
            raise ValueError(
                f"Head shape mismatch: expected "
                f"weight={tuple(self.head.weight.shape)}, "
                f"bias={tuple(self.head.bias.shape)}."
            )

        with torch.no_grad():
            self.head.weight.copy_(init_from.weight)
            self.head.bias.copy_(init_from.bias)

    @abstractmethod
    def init_peft(
        self,
        init_from: Any = None,
    ) -> None: ...

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        return self.backbone.forward_features(x)

    def get_layerwise_features(
        self,
        x: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        return self.backbone.get_layerwise_features(x)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(self.forward_features(x)["cls"])

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ):
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)

        return (
            logits,
            loss,
            {"cross_entropy_loss": loss.item()},
        )

    def get_param_count(self):
        params = [
            parameter
            for name, parameter in self.named_parameters()
            if not name.startswith("functional_gates.") and name != "coupling_gates"
        ]

        total_count = sum(parameter.numel() for parameter in params)

        backbone_count = sum(
            parameter.numel() for parameter in self.backbone.parameters()
        )

        trainable_count = sum(
            parameter.numel() for parameter in params if parameter.requires_grad
        )

        return {
            "total_count": total_count,
            "backbone_count": backbone_count,
            "trainable_count": trainable_count,
            "trainable_over_total": (
                trainable_count / total_count if total_count else 0.0
            ),
            "trainable_over_backbone": (
                trainable_count / backbone_count if backbone_count else 0.0
            ),
        }

    @abstractmethod
    def save(
        self,
        path: str,
    ) -> None: ...

    def load_from_path(
        self,
        path: str,
        map_location="cpu",
    ) -> None:
        self.load_state_dict(
            torch.load(
                path,
                map_location=map_location,
            ),
            strict=False,
        )

    # ======================================================================
    # Functional-artifact construction
    # ======================================================================

    def _make_gate(
        self,
        shape: tuple[int, ...],
    ) -> nn.Parameter:
        reference = next(self.backbone.parameters())

        return nn.Parameter(
            reference.new_ones(shape),
            requires_grad=False,
        )

    def _register_functional_participation(
        self,
        name: str,
        shape: tuple[int, ...],
    ) -> None:
        self._functional_shapes[name] = tuple(shape)

        reference = next(self.backbone.parameters())

        self.register_buffer(
            f"_functional_participation_{name}",
            reference.new_zeros(shape),
            persistent=False,
        )

    def _initialize_functional_state(
        self,
        coupling_shape: tuple[int, ...],
    ) -> None:
        """Create all common functional state in place."""
        layers = self.backbone.num_layers

        gate_shapes = {
            "sublayer_input_channel": (
                layers,
                2,
                2,
                self.backbone.embed_dim,
            ),
            "attention_head_selection_route": (
                layers,
                self.backbone.num_heads,
                7,
            ),
            "attention_head_transport_route": (
                layers,
                self.backbone.num_heads,
                7,
            ),
            "mlp_neuron": (
                layers,
                2,
                self.backbone.mlp_dim,
            ),
            "residual_token_carry": (
                layers,
                2,
                2,
            ),
            "sublayer_token_update": (
                layers,
                2,
                2,
            ),
            "final_cls_readout_channel": (self.backbone.embed_dim,),
        }

        self.functional_gates = nn.ParameterDict()

        for name, shape in gate_shapes.items():
            self.functional_gates[name] = self._make_gate(shape)

        self.coupling_gates = self._make_gate(coupling_shape)

        self._functional_shapes = {}

        for name, gate in self.functional_gates.items():
            self._register_functional_participation(
                name,
                tuple(gate.shape),
            )

        for name, shape in {
            "residual_carry": (
                layers,
                2,
            ),
            "sublayer_update": (
                layers,
                2,
            ),
            "attention_selection_route": (
                layers,
                7,
            ),
            "attention_transport_route": (
                layers,
                7,
            ),
            "attention_head_output": (
                layers,
                self.backbone.num_heads,
            ),
        }.items():
            self._register_functional_participation(
                name,
                shape,
            )

        self._build_output_space_metrics()

    def _build_route_masks(
        self,
        n_tokens: int,
        patch_slice: slice,
    ) -> None:
        """Create the seven common CLS/PATCH attention routes in place."""
        reference = next(self.backbone.parameters())

        coordinates = self.backbone.get_patch_coordinates().to(
            device=reference.device,
            dtype=reference.dtype,
        )

        distances = torch.cdist(
            coordinates,
            coordinates,
        )

        positive = distances[distances > 0]

        thresholds = (
            torch.quantile(
                positive,
                distances.new_tensor([1 / 3, 2 / 3]),
            )
            if positive.numel()
            else distances.new_zeros(2)
        )

        spatial_bins = (
            torch.bucketize(
                distances,
                thresholds,
            )
            + 1
        )

        spatial_bins.fill_diagonal_(0)

        masks = reference.new_zeros(
            7,
            n_tokens,
            n_tokens,
        )

        masks[0, 0, 0] = 1
        masks[1, 0, patch_slice] = 1
        masks[2, patch_slice, 0] = 1

        for band in range(4):
            masks[
                3 + band,
                patch_slice,
                patch_slice,
            ] = (
                spatial_bins == band
            ).to(reference.dtype)

        self.register_buffer(
            "_functional_route_masks",
            masks,
            persistent=False,
        )

    def _build_output_space_metrics(
        self,
    ) -> None:
        head_metrics = []
        mlp_neuron_norms = []

        with torch.no_grad():
            for layer_idx in range(self.backbone.num_layers):
                output_weight = self.backbone.get_attention_output_weight(layer_idx)

                head_metrics.append(
                    torch.stack(
                        [
                            weight.T @ weight
                            for weight in output_weight.split(
                                self.backbone.head_dim,
                                dim=1,
                            )
                        ]
                    )
                )

                mlp_neuron_norms.append(
                    self.backbone.get_mlp_output_weight(layer_idx).square().sum(dim=0)
                )

        self.register_buffer(
            "_functional_head_metrics",
            torch.stack(head_metrics),
            persistent=False,
        )

        self.register_buffer(
            "_functional_mlp_neuron_norms",
            torch.stack(mlp_neuron_norms),
            persistent=False,
        )

    def _participation(
        self,
        name: str,
    ) -> torch.Tensor:
        return getattr(
            self,
            f"_functional_participation_{name}",
        )

    # ======================================================================
    # Functional-forward primitives
    # ======================================================================

    def _record_functional_participation(
        self,
        name: str,
        index,
        value: torch.Tensor,
    ) -> None:
        self._participation(name)[index].copy_(value.detach().sum(dim=0))

    @staticmethod
    def _grouped_channel_energy(
        value: torch.Tensor,
        group_slices: Sequence[slice],
    ) -> torch.Tensor:
        """
        [B, T, C] -> [B, G, C]
        """
        energy = value.detach().square()

        return torch.stack(
            [
                energy[
                    :,
                    group_slice,
                ].sum(dim=1)
                for group_slice in group_slices
            ],
            dim=1,
        )

    @staticmethod
    def _apply_groups(
        value: torch.Tensor,
        group_slices: Sequence[slice],
        group_gates: Sequence[torch.Tensor | None],
    ) -> torch.Tensor:
        return torch.cat(
            [
                (
                    value[:, group_slice]
                    if gate is None
                    else value[
                        :,
                        group_slice,
                    ]
                    * gate
                )
                for group_slice, gate in zip(
                    group_slices,
                    group_gates,
                    strict=True,
                )
            ],
            dim=1,
        )

    def _record_sublayer_participation(
        self,
        layer_idx: int,
        sublayer_idx: int,
        residual: torch.Tensor,
        update: torch.Tensor,
        group_slices: Sequence[slice],
    ) -> None:
        residual_energy = self._grouped_channel_energy(
            residual,
            group_slices,
        ).sum(dim=-1)

        update_energy = self._grouped_channel_energy(
            update,
            group_slices,
        ).sum(dim=-1)

        index = (
            layer_idx,
            sublayer_idx,
        )

        self._record_functional_participation(
            "residual_token_carry",
            index,
            residual_energy,
        )

        self._record_functional_participation(
            "residual_carry",
            index,
            residual_energy.sum(dim=-1),
        )

        self._record_functional_participation(
            "sublayer_token_update",
            index,
            update_energy,
        )

        self._record_functional_participation(
            "sublayer_update",
            index,
            update_energy.sum(dim=-1),
        )

    def _record_mlp_neuron_participation(
        self,
        layer_idx: int,
        value: torch.Tensor,
        group_slices: Sequence[slice],
    ) -> None:
        self._record_functional_participation(
            "mlp_neuron",
            layer_idx,
            (
                self._grouped_channel_energy(
                    value,
                    group_slices,
                )
                * self._functional_mlp_neuron_norms[layer_idx][
                    None,
                    None,
                ]
            ),
        )

    def _apply_attention_selection_gates(
        self,
        attention_scores: torch.Tensor,
        route_gates: torch.Tensor,
    ) -> torch.Tensor:
        log_gates = torch.log(
            route_gates.clamp_min(torch.finfo(route_gates.dtype).tiny)
        )

        return (
            attention_scores
            + torch.einsum(
                "hr,rij->hij",
                log_gates,
                self._functional_route_masks,
            )[None]
        )

    def _apply_attention_transport_gates(
        self,
        attention_probs: torch.Tensor,
        attention_value: torch.Tensor,
        route_gates: torch.Tensor,
    ) -> torch.Tensor:
        scale = 1 + torch.einsum(
            "hr,rij->hij",
            route_gates - 1,
            self._functional_route_masks,
        )

        return (attention_probs * scale[None]) @ attention_value

    def _record_attention_route_participation(
        self,
        attention_weights: torch.Tensor,
        attention_probs: torch.Tensor,
        attention_value: torch.Tensor,
        layer_idx: int,
    ) -> None:
        """
        Record participation for the seven common attention routes.

        Subclasses may override this method to add method-specific routes.
        """
        route_masks = self._functional_route_masks[:7]

        selection_mass = torch.einsum(
            "bhij,rij->bhr",
            attention_weights.detach(),
            route_masks,
        )

        with torch.no_grad():
            route_context = torch.einsum(
                "bhij,rij,bhjk->bhrik",
                attention_probs.detach(),
                route_masks,
                attention_value.detach(),
            )

            head_transport_energy = torch.einsum(
                "bhrik,hkl,bhril->bhr",
                route_context,
                self._functional_head_metrics[layer_idx],
                route_context,
            )

            (
                batch_size,
                n_heads,
                n_routes,
                n_tokens,
                head_dim,
            ) = route_context.shape

            route_context = route_context.permute(
                0,
                2,
                3,
                1,
                4,
            ).reshape(
                batch_size,
                n_routes,
                n_tokens,
                n_heads * head_dim,
            )

            route_output = F.linear(
                route_context,
                (self.backbone.get_attention_output_weight(layer_idx)),
                bias=None,
            )

            transport_energy = route_output.square().sum(dim=(2, 3))

        self._record_functional_participation(
            "attention_head_selection_route",
            layer_idx,
            selection_mass,
        )

        self._record_functional_participation(
            "attention_selection_route",
            layer_idx,
            selection_mass.sum(dim=1),
        )

        self._record_functional_participation(
            "attention_head_transport_route",
            layer_idx,
            head_transport_energy,
        )

        self._record_functional_participation(
            "attention_transport_route",
            layer_idx,
            transport_energy,
        )

    def _record_attention_head_output_participation(
        self,
        context: torch.Tensor,
        layer_idx: int,
    ) -> None:
        context = context.detach()

        energy = torch.einsum(
            "bhtd,hde,bhte->bh",
            context,
            self._functional_head_metrics[layer_idx],
            context,
        )

        self._record_functional_participation(
            "attention_head_output",
            layer_idx,
            energy,
        )

    def _apply_final_cls_readout_gate(
        self,
        encoded: torch.Tensor,
    ) -> torch.Tensor:
        self._participation("final_cls_readout_channel").copy_(
            encoded[
                :,
                0,
            ]
            .detach()
            .square()
            .sum(dim=0)
        )

        return self._apply_groups(
            encoded,
            (
                slice(0, 1),
                slice(1, None),
            ),
            (
                self.functional_gates["final_cls_readout_channel"],
                None,
            ),
        )

    @abstractmethod
    def _create_functional_artifacts(
        self,
    ) -> None: ...

    # ======================================================================
    # Functional scores and coupling
    # ======================================================================

    def _extend_functional_scores(
        self,
        scores: dict[
            str,
            torch.Tensor,
        ],
    ) -> None:
        pass

    def _build_functional_scores(
        self,
        raw_scores,
        gate_names,
        gates,
    ) -> torch.Tensor:
        scores = dict(
            zip(
                gate_names,
                raw_scores[: len(gates)],
            )
        )

        scores["residual_carry"] = scores["residual_token_carry"].sum(dim=-1)

        scores["sublayer_update"] = scores["sublayer_token_update"].sum(dim=-1)

        scores["attention_selection_route"] = scores[
            "attention_head_selection_route"
        ].sum(dim=2)

        scores["attention_transport_route"] = scores[
            "attention_head_transport_route"
        ].sum(dim=2)

        scores["attention_head_output"] = scores["attention_head_transport_route"].sum(
            dim=-1
        )

        self._extend_functional_scores(scores)

        return torch.cat(
            [scores[name].flatten(start_dim=1) for name in self._functional_shapes],
            dim=1,
        )

    def _build_coupling_scores(
        self,
        unit_scores: torch.Tensor,
    ) -> torch.Tensor:
        layer_scores = unit_scores.sum(dim=-1)

        global_scores = layer_scores.sum(
            dim=-1,
            keepdim=True,
        )

        return torch.cat(
            (
                unit_scores.flatten(start_dim=1),
                layer_scores,
                global_scores,
            ),
            dim=1,
        )

    def _coupling_unit_shape(
        self,
    ) -> tuple[int, ...]:
        return tuple(self.coupling_gates.shape)

    # ======================================================================
    # Functional-artifact lifecycle
    # ======================================================================

    def _reset_functional_artifacts(
        self,
    ) -> None:
        with torch.no_grad():
            for gate in self.functional_gates.values():
                gate.fill_(1)

            self.coupling_gates.fill_(1)

            for name in self._functional_shapes:
                self._participation(name).zero_()

    def _set_gate_grads(
        self,
        mode: bool = True,
    ) -> None:
        for gate in self.functional_gates.values():
            gate.grad = None
            gate.requires_grad_(mode)

        self.coupling_gates.grad = None
        self.coupling_gates.requires_grad_(mode)

    # ======================================================================
    # Signature helpers
    # ======================================================================

    def _functional_layout(self):
        sizes = tuple(math.prod(shape) for shape in self._functional_shapes.values())

        return (
            sizes,
            sum(sizes),
        )

    def _unpack_functional_vector(
        self,
        value,
        sizes,
        tail=(),
    ):
        return {
            name: part.reshape(
                *shape,
                *tail,
            )
            for (
                name,
                shape,
            ), part in zip(
                self._functional_shapes.items(),
                value.split(sizes),
                strict=True,
            )
        }

    @staticmethod
    def _make_sensitivity_signature(
        fisher_signature,
        participation_signature,
        eps,
    ):
        signature = {}

        for (
            name,
            participation,
        ) in participation_signature.items():
            fisher = fisher_signature[name]

            threshold = eps * participation.amax().clamp_min(1.0)

            signature[name] = torch.where(
                participation > threshold,
                fisher / participation.clamp_min(threshold),
                torch.zeros_like(fisher),
            )

        return signature

    def _merge_functional_statistics(
        self,
        statistics,
    ):
        """
        Pool additive sufficient statistics from repeated MC estimates
        of the SAME trained checkpoint.
        """
        if isinstance(statistics, Mapping):
            statistics = (statistics,)
        else:
            statistics = tuple(statistics)

        if not statistics:
            raise ValueError("statistics must not be empty")

        tensor_keys = (
            "participation_sum",
            "fisher_sum_m",
            "score_sum_e",
            "fisher_sum_e",
            "cross_sum_m",
            "coupling_square_sum_m",
            "cross_sum_e",
            "coupling_score_sum_e",
            "coupling_square_sum_e",
        )

        count_keys = (
            "input_count",
            "sample_count_m",
            "sample_count_e",
        )

        first = statistics[0]

        merged = {key: int(first[key]) for key in count_keys}

        merged.update(
            {
                key: (torch.as_tensor(first[key]).detach().cpu().clone())
                for key in tensor_keys
            }
        )

        for stats in statistics[1:]:
            for key in count_keys:
                merged[key] += int(stats[key])

            for key in tensor_keys:
                value = torch.as_tensor(stats[key]).detach().cpu()

                if value.shape != merged[key].shape:
                    raise ValueError(f"incompatible shape for '{key}'")

                merged[key].add_(value.to(dtype=merged[key].dtype))

        if min(merged[key] for key in count_keys) <= 0:
            raise ValueError("statistics contain an invalid sample count")

        return merged

    # ======================================================================
    # Functional-statistics estimator
    # ======================================================================

    def estimate_functional_statistics(
        self,
        loader,
        mc_samples: int = 20,
        verbose: bool = False,
    ):
        """
        Estimate additive sufficient statistics for the functional
        and coupling signatures.
        """
        if mc_samples < 1:
            raise ValueError("mc_samples must be positive")

        self._create_functional_artifacts()
        self._reset_functional_artifacts()
        self._set_gate_grads(True)
        self.eval()

        try:
            gate_names, gates = zip(*self.functional_gates.items())

            _, total_size = self._functional_layout()

            n_layers = self.backbone.num_layers

            coupling_unit_size = self.coupling_gates.numel()

            use_couplings = coupling_unit_size > 0

            coupling_size = coupling_unit_size + n_layers + 1 if use_couplings else 0

            reference = gates[0]
            device = reference.device

            participation_sum = reference.new_zeros(total_size)

            fisher_sum_m = reference.new_zeros(total_size)

            score_sum_e = reference.new_zeros(total_size)

            fisher_sum_e = reference.new_zeros(total_size)

            cross_sum_m = reference.new_zeros(
                total_size,
                coupling_size,
            )

            coupling_square_sum_m = reference.new_zeros(coupling_size)

            cross_sum_e = reference.new_zeros(
                total_size,
                coupling_size,
            )

            coupling_score_sum_e = reference.new_zeros(coupling_size)

            coupling_square_sum_e = reference.new_zeros(coupling_size)

            params = (
                (
                    *gates,
                    self.coupling_gates,
                )
                if use_couplings
                else gates
            )

            bases = {}

            input_count = 0
            sample_count_m = 0
            sample_count_e = 0

            iterator = (
                tqdm(
                    loader,
                    desc="Functional statistics",
                )
                if verbose
                else loader
            )

            for batch in iterator:
                x = batch[0].to(
                    device,
                    non_blocking=True,
                )

                y = batch[1].to(
                    device=device,
                    dtype=torch.long,
                    non_blocking=True,
                )

                with torch.no_grad():
                    for name in self._functional_shapes:
                        self._participation(name).zero_()

                log_probs = self(x).log_softmax(dim=-1)

                participation_sum.add_(
                    torch.cat(
                        [
                            self._participation(name).reshape(-1)
                            for name in self._functional_shapes
                        ]
                    )
                )

                input_count += x.shape[0]

                # Model Fisher:
                # y~ ~ p(y | x)
                labels_m = torch.multinomial(
                    log_probs.detach().exp(),
                    mc_samples,
                    replacement=True,
                )

                selected_m = log_probs.gather(
                    1,
                    labels_m,
                ).flatten()

                # Observed-label score:
                # y = y*
                selected_e = log_probs.gather(
                    1,
                    y.reshape(
                        -1,
                        1,
                    ),
                ).flatten()

                n_m = selected_m.numel()
                n_e = selected_e.numel()

                selected = torch.cat(
                    (
                        selected_m,
                        selected_e,
                    )
                )

                basis = bases.get(selected.numel())

                if basis is None:
                    basis = torch.eye(
                        selected.numel(),
                        device=device,
                        dtype=selected.dtype,
                    )

                    bases[selected.numel()] = basis

                raw_scores = torch.autograd.grad(
                    selected,
                    params,
                    grad_outputs=basis,
                    is_grads_batched=True,
                )

                functional_scores = self._build_functional_scores(
                    raw_scores,
                    gate_names,
                    gates,
                )

                coupling_scores = (
                    self._build_coupling_scores(raw_scores[-1])
                    if use_couplings
                    else None
                )

                functional_m = functional_scores[:n_m]

                functional_e = functional_scores[n_m:]

                fisher_sum_m.add_(functional_m.square().sum(dim=0))

                score_sum_e.add_(functional_e.sum(dim=0))

                fisher_sum_e.add_(functional_e.square().sum(dim=0))

                if use_couplings:
                    coupling_m = coupling_scores[:n_m]

                    coupling_e = coupling_scores[n_m:]

                    cross_sum_m.addmm_(
                        functional_m.T,
                        coupling_m,
                    )

                    coupling_square_sum_m.add_(coupling_m.square().sum(dim=0))

                    cross_sum_e.addmm_(
                        functional_e.T,
                        coupling_e,
                    )

                    coupling_score_sum_e.add_(coupling_e.sum(dim=0))

                    coupling_square_sum_e.add_(coupling_e.square().sum(dim=0))

                sample_count_m += n_m
                sample_count_e += n_e

            if input_count == 0:
                raise ValueError("loader is empty")

            return {
                "input_count": (input_count),
                "sample_count_m": (sample_count_m),
                "sample_count_e": (sample_count_e),
                "participation_sum": (participation_sum.detach().cpu()),
                "fisher_sum_m": (fisher_sum_m.detach().cpu()),
                "score_sum_e": (score_sum_e.detach().cpu()),
                "fisher_sum_e": (fisher_sum_e.detach().cpu()),
                "cross_sum_m": (cross_sum_m.detach().cpu()),
                "coupling_square_sum_m": (coupling_square_sum_m.detach().cpu()),
                "cross_sum_e": (cross_sum_e.detach().cpu()),
                "coupling_score_sum_e": (coupling_score_sum_e.detach().cpu()),
                "coupling_square_sum_e": (coupling_square_sum_e.detach().cpu()),
            }

        finally:
            self._set_gate_grads(False)

    # ======================================================================
    # Statistics -> functional signatures
    # ======================================================================

    def functional_signatures_from_statistics(
        self,
        statistics,
        eps: float = 1e-12,
    ):
        """
        Pool MC statistics from the SAME trained checkpoint and
        reconstruct one checkpoint-level functional signature.
        """
        if eps <= 0:
            raise ValueError("eps must be positive")

        self._create_functional_artifacts()

        (
            sizes,
            total_size,
        ) = self._functional_layout()

        n_layers = self.backbone.num_layers

        coupling_unit_shape = self._coupling_unit_shape()

        coupling_unit_size = self.coupling_gates.numel()

        use_couplings = coupling_unit_size > 0

        stats = self._merge_functional_statistics(statistics)

        input_count = stats["input_count"]

        sample_count_m = stats["sample_count_m"]

        sample_count_e = stats["sample_count_e"]

        participation = stats["participation_sum"] / input_count

        fisher_m = stats["fisher_sum_m"] / sample_count_m

        mean_e = stats["score_sum_e"] / sample_count_e

        fisher_e = stats["fisher_sum_e"] / sample_count_e

        variance_e = (fisher_e - mean_e.square()).clamp_min(0)

        def unpack(
            value,
            tail=(),
        ):
            return self._unpack_functional_vector(
                value,
                sizes,
                tail,
            )

        P = unpack(participation)

        F_m = unpack(fisher_m)

        S_m = self._make_sensitivity_signature(
            F_m,
            P,
            eps,
        )

        U_e = unpack(mean_e)

        F_e = unpack(fisher_e)

        S_e = self._make_sensitivity_signature(
            F_e,
            P,
            eps,
        )

        B_e = unpack(
            (mean_e.square() / fisher_e)
            .nan_to_num(
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            .clamp(
                0,
                1,
            )
        )

        reference = stats["participation_sum"]

        if use_couplings:
            coupling_second_m = stats["coupling_square_sum_m"] / sample_count_m

            C_m = (
                (stats["cross_sum_m"] / sample_count_m)
                / fisher_m.sqrt()[
                    :,
                    None,
                ]
                / coupling_second_m.sqrt()[None]
            )

            C_m = C_m.nan_to_num(
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp(
                -1,
                1,
            )

            coupling_mean_e = stats["coupling_score_sum_e"] / sample_count_e

            coupling_second_e = stats["coupling_square_sum_e"] / sample_count_e

            coupling_variance_e = (
                coupling_second_e - coupling_mean_e.square()
            ).clamp_min(0)

            C_e = (
                stats["cross_sum_e"] / sample_count_e
                - mean_e[
                    :,
                    None,
                ]
                * coupling_mean_e[None]
            )

            C_e = (
                C_e
                / variance_e.sqrt()[
                    :,
                    None,
                ]
                / coupling_variance_e.sqrt()[None]
            )

            C_e = C_e.nan_to_num(
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp(
                -1,
                1,
            )

            unit = slice(
                0,
                coupling_unit_size,
            )

            layer = slice(
                coupling_unit_size,
                coupling_unit_size + n_layers,
            )

            C_unit_m = unpack(
                C_m[
                    :,
                    unit,
                ],
                coupling_unit_shape,
            )

            C_layer_m = unpack(
                C_m[
                    :,
                    layer,
                ],
                (n_layers,),
            )

            C_global_m = unpack(
                C_m[
                    :,
                    -1,
                ]
            )

            C_unit_e = unpack(
                C_e[
                    :,
                    unit,
                ],
                coupling_unit_shape,
            )

            C_layer_e = unpack(
                C_e[
                    :,
                    layer,
                ],
                (n_layers,),
            )

            C_global_e = unpack(
                C_e[
                    :,
                    -1,
                ]
            )

        else:
            C_unit_m = unpack(
                reference.new_zeros(
                    total_size,
                    *coupling_unit_shape,
                ),
                coupling_unit_shape,
            )

            C_layer_m = unpack(
                reference.new_zeros(
                    total_size,
                    n_layers,
                ),
                (n_layers,),
            )

            C_global_m = unpack(reference.new_zeros(total_size))

            C_unit_e = unpack(
                reference.new_zeros(
                    total_size,
                    *coupling_unit_shape,
                ),
                coupling_unit_shape,
            )

            C_layer_e = unpack(
                reference.new_zeros(
                    total_size,
                    n_layers,
                ),
                (n_layers,),
            )

            C_global_e = unpack(reference.new_zeros(total_size))

        return {
            "P": P,
            "F_m": F_m,
            "S_m": S_m,
            "C_unit_m": C_unit_m,
            "C_layer_m": C_layer_m,
            "C_global_m": C_global_m,
            "U_e": U_e,
            "F_e": F_e,
            "S_e": S_e,
            "B_e": B_e,
            "C_unit_e": C_unit_e,
            "C_layer_e": C_layer_e,
            "C_global_e": C_global_e,
        }

    # ======================================================================
    # Reliability
    # ======================================================================

    @staticmethod
    def _icc_absolute_agreement(
        values,
    ):
        """
        Compute ICC(A,1) and ICC(A,k).

        Rows are functional coordinates.
        Columns are repeated estimates.
        """
        values = torch.as_tensor(values).detach().double()

        if values.ndim != 2:
            raise ValueError(
                "values must have shape " "[n_coordinates, k_measurements]"
            )

        n, k = values.shape

        if n < 2 or k < 2:
            return (
                float("nan"),
                float("nan"),
            )

        if not torch.isfinite(values).all():
            raise ValueError("ICC input contains non-finite values")

        grand_mean = values.mean()

        row_means = values.mean(dim=1)

        column_means = values.mean(dim=0)

        ss_rows = k * (row_means - grand_mean).square().sum()

        ss_columns = n * (column_means - grand_mean).square().sum()

        residual = (
            values
            - row_means[
                :,
                None,
            ]
            - column_means[
                None,
                :,
            ]
            + grand_mean
        )

        ss_error = residual.square().sum()

        ms_rows = ss_rows / (n - 1)

        ms_columns = ss_columns / (k - 1)

        ms_error = ss_error / ((n - 1) * (k - 1))

        numerator = ms_rows - ms_error

        denominator_a1 = (
            ms_rows + (k - 1) * ms_error + (k / n) * (ms_columns - ms_error)
        )

        denominator_ak = ms_rows + (ms_columns - ms_error) / n

        def ratio(
            denominator,
        ):
            if (
                not torch.isfinite(denominator)
                or denominator.abs() <= torch.finfo(values.dtype).eps
            ):
                return float("nan")

            return float((numerator / denominator).item())

        return (
            ratio(denominator_a1),
            ratio(denominator_ak),
        )

    def _functional_signatures_icc(
        self,
        signatures,
    ):
        """
        Compute ICC independently for each signature and component family.
        """
        signatures = tuple(signatures)

        if len(signatures) < 2:
            raise ValueError("at least two signatures are required")

        names = tuple(signatures[0])

        if any(tuple(signature) != names for signature in signatures[1:]):
            raise ValueError("signatures have incompatible keys")

        icc_a1 = {}
        icc_ak = {}

        for name in names:
            components = tuple(signatures[0][name])

            if any(
                tuple(signature[name]) != components for signature in signatures[1:]
            ):
                raise ValueError(
                    f"signature '{name}' has " "incompatible component keys"
                )

            icc_a1[name] = {}
            icc_ak[name] = {}

            for component in components:
                values = [
                    (torch.as_tensor(signature[name][component]).detach().cpu())
                    for signature in signatures
                ]

                shape = values[0].shape

                if any(value.shape != shape for value in values[1:]):
                    raise ValueError(f"'{name}/{component}' " "has incompatible shapes")

                matrix = torch.stack(
                    [value.reshape(-1) for value in values],
                    dim=1,
                )

                a1, ak = self._icc_absolute_agreement(matrix)

                icc_a1[name][component] = a1

                icc_ak[name][component] = ak

        return {
            "icc_a1": icc_a1,
            "icc_ak": icc_ak,
            "k": len(signatures),
        }

    def functional_statistics_reliability(
        self,
        statistics,
        eps: float = 1e-12,
    ):
        """
        Compute within-checkpoint MC reliability.

        Each MC statistics object is independently converted to a
        functional signature, and ICC(A,1) is computed across them.

        Only ICC(A,1) is returned because the actual checkpoint signature
        is obtained by pooling sufficient statistics, not by averaging
        separately finalized MC signatures.
        """
        if isinstance(
            statistics,
            Mapping,
        ):
            statistics = (statistics,)
        else:
            statistics = tuple(statistics)

        if len(statistics) < 2:
            raise ValueError("at least two MC estimates are required")

        signatures = [
            self.functional_signatures_from_statistics(
                stats,
                eps=eps,
            )
            for stats in statistics
        ]

        return self._functional_signatures_icc(signatures)["icc_a1"]

    def functional_signature_reliability(
        self,
        signatures,
    ):
        """
        Compute reliability across independently trained checkpoints.

        Returns ICC(A,1) and ICC(A,k). With three checkpoints,
        ICC(A,k) is ICC(A,3), the reliability of the three-checkpoint mean.
        """
        return self._functional_signatures_icc(signatures)

    # ======================================================================
    # Checkpoint aggregation
    # ======================================================================

    def merge_functional_signatures(
        self,
        signatures,
    ):
        """
        Arithmetic-average functional signatures from independently
        trained checkpoints to obtain the final dataset x method profile.
        """
        if isinstance(
            signatures,
            Mapping,
        ):
            signatures = (signatures,)
        else:
            signatures = tuple(signatures)

        if not signatures:
            raise ValueError("signatures must not be empty")

        names = tuple(signatures[0])

        if any(tuple(signature) != names for signature in signatures[1:]):
            raise ValueError("signatures have incompatible keys")

        merged = {}

        for name in names:
            components = tuple(signatures[0][name])

            if any(
                tuple(signature[name]) != components for signature in signatures[1:]
            ):
                raise ValueError(
                    f"signature '{name}' has " "incompatible component keys"
                )

            merged[name] = {}

            for component in components:
                values = [
                    (torch.as_tensor(signature[name][component]).detach().cpu())
                    for signature in signatures
                ]

                shape = values[0].shape

                if any(value.shape != shape for value in values[1:]):
                    raise ValueError(f"'{name}/{component}' " "has incompatible shapes")

                merged[name][component] = torch.stack(
                    values,
                    dim=0,
                ).mean(dim=0)

        return merged
