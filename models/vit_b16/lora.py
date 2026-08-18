import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from types import MethodType
from typing import Optional
from tqdm.auto import tqdm
from .backbone import ViTB16Backbone


class ViTB16LoRA(nn.Module):
    """LoRA model implemented for a ViT-B/16 backbone."""

    def __init__(
        self,
        img_size: int,
        n_classes: int,
        freeze_backbone: bool = True,
        rank: int = 8,
        lora_alpha: Optional[float] = None,
        lora_targets=("query", "value"),
    ):
        super().__init__()

        if rank < 0:
            raise ValueError("rank must be non-negative")

        lora_targets = tuple(lora_targets)

        if len(set(lora_targets)) != len(lora_targets):
            raise ValueError("lora_targets must not contain duplicates")

        if any(name not in {"query", "value"} for name in lora_targets):
            raise ValueError("lora_targets may contain only 'query' and 'value'")

        if rank > 0 and not lora_targets:
            raise ValueError("lora_targets must be non-empty when rank > 0")

        self.img_size = img_size
        self.n_classes = n_classes
        self.is_backbone_frozen = freeze_backbone
        self.rank = rank
        self.lora_targets = lora_targets
        self.lora_alpha = float(rank if lora_alpha is None else lora_alpha)
        self.lora_scaling = self.lora_alpha / rank if rank > 0 else 0.0

        if rank > 0 and self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive when rank > 0")

        # Backbone
        self.backbone = ViTB16Backbone(img_size=img_size)

        if freeze_backbone:
            self.freeze_backbone()

        # Head
        self.head = nn.Linear(self.backbone.embed_dim, n_classes)
        self.init_head()

        # LoRA
        if rank > 0:
            n_layers = len(self.backbone.blocks)
            n_targets = len(lora_targets)
            embed_dim = self.backbone.embed_dim

            reference = next(self.backbone.parameters())

            self.lora_A = nn.Parameter(
                reference.new_empty(
                    n_layers,
                    n_targets,
                    rank,
                    embed_dim,
                )
            )

            self.lora_B = nn.Parameter(
                reference.new_empty(
                    n_layers,
                    n_targets,
                    embed_dim,
                    rank,
                )
            )

            self.init_peft()
            self._patch_lora_projections()

        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def freeze_backbone(self):
        self.backbone.freeze()
        self.backbone.eval()
        self.is_backbone_frozen = True

    def unfreeze_backbone(self):
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

    def init_head(self, init_from: Optional[nn.Linear] = None):
        if init_from is not None:
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

            return

        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def init_peft(
        self,
        init_from: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        if self.rank == 0:
            return

        if init_from is not None:
            lora_A, lora_B = init_from

            if lora_A.shape != self.lora_A.shape or lora_B.shape != self.lora_B.shape:
                raise ValueError(
                    f"LoRA shape mismatch: expected "
                    f"A={tuple(self.lora_A.shape)}, B={tuple(self.lora_B.shape)}, "
                    f"got A={tuple(lora_A.shape)}, B={tuple(lora_B.shape)}."
                )

            with torch.no_grad():
                self.lora_A.copy_(lora_A)
                self.lora_B.copy_(lora_B)

            return

        for value in self.lora_A.flatten(0, 1):
            nn.init.kaiming_uniform_(value, a=math.sqrt(5))

        nn.init.zeros_(self.lora_B)

    def _lora_delta(
        self,
        x: torch.Tensor,
        layer_idx: int,
        target_idx: int,
    ) -> torch.Tensor:
        return (
            F.linear(
                F.linear(
                    x,
                    self.lora_A[layer_idx, target_idx],
                ),
                self.lora_B[layer_idx, target_idx],
            )
            * self.lora_scaling
        )

    def _patch_lora_projections(self):
        def make_forward(
            layer_idx,
            target_idx,
            original_forward,
        ):
            def forward(projection, x):
                return original_forward(x) + self._lora_delta(
                    x,
                    layer_idx,
                    target_idx,
                )

            return forward

        for layer_idx, block in enumerate(self.backbone.blocks):
            for target_idx, target_name in enumerate(self.lora_targets):
                projection = getattr(
                    block.attn,
                    target_name,
                )

                projection.forward = MethodType(
                    make_forward(
                        layer_idx,
                        target_idx,
                        projection.forward,
                    ),
                    projection,
                )

    def forward_features(self, x: torch.Tensor):
        """Standard LoRA forward through the adapted backbone."""
        return self.backbone.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.head(features["cls"])

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ):
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)

        loss_components = {
            "cross_entropy_loss": loss.item(),
        }

        return logits, loss, loss_components

    def get_param_count(self):
        total_count = sum(p.numel() for p in self.parameters())

        backbone_count = sum(p.numel() for p in self.backbone.parameters())

        trainable_count = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_count": total_count,
            "backbone_count": backbone_count,
            "trainable_count": trainable_count,
            "trainable_over_total": (
                trainable_count / total_count if total_count > 0 else 0.0
            ),
            "trainable_over_backbone": (
                trainable_count / backbone_count if backbone_count > 0 else 0.0
            ),
        }

    def save(self, path: str):
        """
        Save LoRA parameters and task head.

        If the backbone is unfrozen, also save it so rank=0 can be used
        cleanly for full fine-tuning.
        """
        state_dict = {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if (
                key.startswith("head.")
                or key in {"lora_A", "lora_B"}
                or (not self.is_backbone_frozen and key.startswith("backbone."))
            )
        }

        torch.save(
            state_dict,
            path,
        )

    def load_from_path(
        self,
        path: str,
        map_location="cpu",
    ):
        state_dict = torch.load(
            path,
            map_location=map_location,
        )

        self.load_state_dict(
            state_dict,
            strict=False,
        )

    def _create_functional_artifacts(self):
        """
        Create functional gates, LoRA coupling gates, participation
        accumulators, fixed route masks/metrics, and an instrumented ViT
        forward pass.

        Every functional and coupling gate is initialized to one, so the
        patched forward is identical to the normal LoRA forward at the
        functional evaluation point.
        """
        if hasattr(self, "functional_gates"):
            return

        blocks = self.backbone.blocks
        n_layers = len(blocks)
        n_heads = blocks[0].attn.num_attention_heads
        embed_dim = self.backbone.embed_dim
        head_dim = blocks[0].attn.attention_head_size
        mlp_dim = blocks[0].ffn.fc1.out_features
        n_patches = self.backbone.num_patches
        n_tokens = 1 + n_patches
        n_couplings = len(self.lora_targets) if self.rank > 0 else 0

        patch_slice = slice(1, None)
        reference = next(self.backbone.parameters())

        def gate(shape):
            return nn.Parameter(reference.new_ones(shape), requires_grad=False)

        self.functional_gates = nn.ParameterDict(
            {
                # [layer, sublayer, token group, residual channel]
                "sublayer_input_channel": gate((n_layers, 2, 2, embed_dim)),
                # [layer, head, route]
                #
                # route:
                #   0 = CLS   <- CLS
                #   1 = CLS   <- PATCH
                #   2 = PATCH <- CLS
                #   3 = PATCH <- PATCH self
                #   4 = PATCH <- PATCH local
                #   5 = PATCH <- PATCH medium
                #   6 = PATCH <- PATCH global
                "attention_head_selection_route": gate((n_layers, n_heads, 7)),
                "attention_head_transport_route": gate((n_layers, n_heads, 7)),
                # [layer, token group, hidden MLP neuron]
                "mlp_neuron": gate((n_layers, 2, mlp_dim)),
                # [layer, sublayer, token group]
                "residual_token_carry": gate((n_layers, 2, 2)),
                "sublayer_token_update": gate((n_layers, 2, 2)),
                # [residual channel]
                "final_cls_readout_channel": gate((embed_dim,)),
            }
        )

        # [layer, LoRA target]
        #
        # Scales the complete learned low-rank contribution:
        #
        #   W x + b + a * (alpha / r) B A x.
        #
        # Thus a=1 is trained LoRA and a=0 removes that LoRA update.
        self.coupling_gates = gate((n_layers, n_couplings))

        # Coarse component scores are exact tied-gate derivatives.
        self._functional_shapes = {
            **{
                name: tuple(value.shape)
                for name, value in self.functional_gates.items()
            },
            "residual_carry": (n_layers, 2),
            "sublayer_update": (n_layers, 2),
            "attention_selection_route": (n_layers, 7),
            "attention_transport_route": (n_layers, 7),
            "attention_head_output": (n_layers, n_heads),
        }

        self._functional_names = tuple(self._functional_shapes)

        self.functional_participation = {
            name: reference.new_zeros(shape)
            for name, shape in self._functional_shapes.items()
        }

        # ------------------------------------------------------------------
        # Fixed attention-route masks
        # ------------------------------------------------------------------

        grid = self.backbone.grid_size

        coordinates = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(grid, device=reference.device),
                    torch.arange(grid, device=reference.device),
                    indexing="ij",
                ),
                dim=-1,
            )
            .reshape(-1, 2)
            .to(reference.dtype)
        )

        distances = torch.cdist(coordinates, coordinates)

        thresholds = torch.quantile(
            distances[distances > 0],
            distances.new_tensor([1 / 3, 2 / 3]),
        )

        spatial_bins = torch.bucketize(distances, thresholds) + 1
        spatial_bins.fill_diagonal_(0)

        route_masks = reference.new_zeros(
            7,
            n_tokens,
            n_tokens,
        )

        route_masks[0, 0, 0] = 1
        route_masks[1, 0, patch_slice] = 1
        route_masks[2, patch_slice, 0] = 1

        for band in range(4):
            route_masks[
                3 + band,
                patch_slice,
                patch_slice,
            ] = (
                spatial_bins == band
            ).to(reference.dtype)

        self._functional_route_masks = route_masks

        # ------------------------------------------------------------------
        # Fixed output-space energy metrics
        # ------------------------------------------------------------------

        head_metrics = []
        mlp_neuron_norms = []

        with torch.no_grad():
            for block in blocks:
                output_weight = block.attn.out.weight
                layer_metrics = []

                for head_idx in range(n_heads):
                    head_slice = slice(
                        head_idx * head_dim,
                        (head_idx + 1) * head_dim,
                    )

                    head_weight = output_weight[:, head_slice]
                    layer_metrics.append(head_weight.T @ head_weight)

                head_metrics.append(torch.stack(layer_metrics))

                mlp_neuron_norms.append(block.ffn.fc2.weight.square().sum(dim=0))

        self._functional_head_metrics = torch.stack(head_metrics)
        self._functional_mlp_neuron_norms = torch.stack(mlp_neuron_norms)

        target_indices = (
            {name: idx for idx, name in enumerate(self.lora_targets)}
            if self.rank > 0
            else {}
        )

        # ------------------------------------------------------------------
        # Instrumented transformer-block forward
        # ------------------------------------------------------------------

        def make_forward(layer_idx):
            def forward(block, hidden):
                gates = self.functional_gates
                attention = block.attn

                def record(name, index, value):
                    self.functional_participation[name][index].copy_(
                        value.detach().sum(dim=0)
                    )

                def grouped_channel_energy(value):
                    energy = value.detach().square()

                    return torch.stack(
                        (
                            energy[:, 0],
                            energy[:, patch_slice].sum(dim=1),
                        ),
                        dim=1,
                    )

                def apply_channel_groups(value, group_gate):
                    return torch.cat(
                        (
                            value[:, :1] * group_gate[0][None, None],
                            value[:, patch_slice] * group_gate[1][None, None],
                        ),
                        dim=1,
                    )

                def apply_token_groups(value, group_gate):
                    return torch.cat(
                        (
                            value[:, :1] * group_gate[0],
                            value[:, patch_slice] * group_gate[1],
                        ),
                        dim=1,
                    )

                def projected_head_energy(context):
                    context = context.detach()

                    return torch.einsum(
                        "bhtd,hde,bhte->bh",
                        context,
                        self._functional_head_metrics[layer_idx],
                        context,
                    )

                def project(name, value):
                    """
                    W x + b + a * (alpha / r) B A x for LoRA targets.
                    """
                    projection = getattr(attention, name)

                    result = F.linear(
                        value,
                        projection.weight,
                        projection.bias,
                    )

                    target_idx = target_indices.get(name)

                    if target_idx is not None:
                        result = result + self.coupling_gates[
                            layer_idx, target_idx
                        ] * self._lora_delta(
                            value,
                            layer_idx,
                            target_idx,
                        )

                    return result

                # ==========================================================
                # Attention input
                # ==========================================================

                residual = hidden
                value = block.attention_norm(hidden)

                record(
                    "sublayer_input_channel",
                    (layer_idx, 0),
                    grouped_channel_energy(value),
                )

                value = apply_channel_groups(
                    value,
                    gates["sublayer_input_channel"][layer_idx, 0],
                )

                query = attention.transpose_for_scores(project("query", value))

                key = attention.transpose_for_scores(project("key", value))

                attention_value = attention.transpose_for_scores(
                    project("value", value)
                )

                attention_scores = query @ key.transpose(-1, -2) / math.sqrt(head_dim)

                # ==========================================================
                # Attention selection
                # ==========================================================

                selection_route_gates = gates["attention_head_selection_route"][
                    layer_idx
                ]

                selection_log_gates = torch.log(
                    selection_route_gates.clamp_min(
                        torch.finfo(selection_route_gates.dtype).tiny
                    )
                )

                attention_scores = (
                    attention_scores
                    + torch.einsum(
                        "hr,rij->hij",
                        selection_log_gates,
                        self._functional_route_masks,
                    )[None]
                )

                attention_weights = attention.softmax(attention_scores)

                selection_route_mass = torch.einsum(
                    "bhij,rij->bhr",
                    attention_weights.detach(),
                    self._functional_route_masks,
                )

                record(
                    "attention_head_selection_route",
                    layer_idx,
                    selection_route_mass,
                )

                record(
                    "attention_selection_route",
                    layer_idx,
                    selection_route_mass.sum(dim=1),
                )

                weights = attention_weights if attention.vis else None
                attention_probs = attention.attn_dropout(attention_weights)

                # ==========================================================
                # Attention transport participation
                # ==========================================================

                with torch.no_grad():
                    route_context = torch.einsum(
                        "bhij,rij,bhjk->bhrik",
                        attention_probs.detach(),
                        self._functional_route_masks,
                        attention_value.detach(),
                    )

                    transport_route_energy = torch.einsum(
                        "bhrik,hkl,bhril->bhr",
                        route_context,
                        self._functional_head_metrics[layer_idx],
                        route_context,
                    )

                    record(
                        "attention_head_transport_route",
                        layer_idx,
                        transport_route_energy,
                    )

                    route_context_concat = route_context.permute(0, 2, 3, 1, 4).reshape(
                        route_context.shape[0],
                        route_context.shape[2],
                        n_tokens,
                        n_heads * head_dim,
                    )

                    route_output = F.linear(
                        route_context_concat,
                        attention.out.weight,
                        bias=None,
                    )

                    record(
                        "attention_transport_route",
                        layer_idx,
                        route_output.square().sum(dim=(2, 3)),
                    )

                # ==========================================================
                # Attention transport intervention
                # ==========================================================

                transport_scale = 1 + torch.einsum(
                    "hr,rij->hij",
                    gates["attention_head_transport_route"][layer_idx] - 1,
                    self._functional_route_masks,
                )

                context = (attention_probs * transport_scale[None]) @ attention_value

                record(
                    "attention_head_output",
                    layer_idx,
                    projected_head_energy(context),
                )

                context = context.transpose(1, 2).reshape(
                    value.shape[0],
                    n_tokens,
                    n_heads * head_dim,
                )

                attention_output = attention.proj_dropout(attention.out(context))

                # ==========================================================
                # Attention residual
                # ==========================================================

                residual_energy = grouped_channel_energy(residual).sum(dim=-1)

                record(
                    "residual_token_carry",
                    (layer_idx, 0),
                    residual_energy,
                )

                record(
                    "residual_carry",
                    (layer_idx, 0),
                    residual_energy.sum(dim=-1),
                )

                attention_energy = grouped_channel_energy(attention_output)

                record(
                    "sublayer_token_update",
                    (layer_idx, 0),
                    attention_energy.sum(dim=-1),
                )

                record(
                    "sublayer_update",
                    (layer_idx, 0),
                    attention_energy.sum(dim=(1, 2)),
                )

                hidden = apply_token_groups(
                    residual,
                    gates["residual_token_carry"][layer_idx, 0],
                ) + apply_token_groups(
                    attention_output,
                    gates["sublayer_token_update"][layer_idx, 0],
                )

                # ==========================================================
                # MLP
                # ==========================================================

                residual = hidden
                feed_forward = block.ffn

                value = block.ffn_norm(hidden)

                record(
                    "sublayer_input_channel",
                    (layer_idx, 1),
                    grouped_channel_energy(value),
                )

                value = apply_channel_groups(
                    value,
                    gates["sublayer_input_channel"][layer_idx, 1],
                )

                value = feed_forward.dropout(
                    feed_forward.act_fn(feed_forward.fc1(value))
                )

                neuron_energy = (
                    grouped_channel_energy(value)
                    * self._functional_mlp_neuron_norms[layer_idx][None, None]
                )

                record(
                    "mlp_neuron",
                    layer_idx,
                    neuron_energy,
                )

                value = apply_channel_groups(
                    value,
                    gates["mlp_neuron"][layer_idx],
                )

                mlp_output = feed_forward.dropout(feed_forward.fc2(value))

                # ==========================================================
                # MLP residual
                # ==========================================================

                residual_energy = grouped_channel_energy(residual).sum(dim=-1)

                record(
                    "residual_token_carry",
                    (layer_idx, 1),
                    residual_energy,
                )

                record(
                    "residual_carry",
                    (layer_idx, 1),
                    residual_energy.sum(dim=-1),
                )

                mlp_energy = grouped_channel_energy(mlp_output)

                record(
                    "sublayer_token_update",
                    (layer_idx, 1),
                    mlp_energy.sum(dim=-1),
                )

                record(
                    "sublayer_update",
                    (layer_idx, 1),
                    mlp_energy.sum(dim=(1, 2)),
                )

                return (
                    apply_token_groups(
                        residual,
                        gates["residual_token_carry"][layer_idx, 1],
                    )
                    + apply_token_groups(
                        mlp_output,
                        gates["sublayer_token_update"][layer_idx, 1],
                    ),
                    weights,
                )

            return forward

        for layer_idx, block in enumerate(blocks):
            block.forward = MethodType(
                make_forward(layer_idx),
                block,
            )

        original_norm_forward = self.backbone.norm.forward

        def norm_forward(norm, hidden):
            encoded = original_norm_forward(hidden)

            self.functional_participation["final_cls_readout_channel"].copy_(
                encoded[:, 0].detach().square().sum(dim=0)
            )

            return torch.cat(
                (
                    encoded[:, :1]
                    * self.functional_gates["final_cls_readout_channel"][None, None],
                    encoded[:, 1:],
                ),
                dim=1,
            )

        self.backbone.norm.forward = MethodType(
            norm_forward,
            self.backbone.norm,
        )

    def _reset_functional_artifacts(self):
        """Reset all gates to identity and clear participation storage."""
        with torch.no_grad():
            for gate in self.functional_gates.values():
                gate.fill_(1)

            self.coupling_gates.fill_(1)

            for value in self.functional_participation.values():
                value.zero_()

    def _set_gate_grads(self, mode: bool = True):
        """Enable or disable gradients only for functional/coupling gates."""
        for gate in self.functional_gates.values():
            gate.grad = None
            gate.requires_grad_(mode)

        self.coupling_gates.grad = None
        self.coupling_gates.requires_grad_(mode)

    def estimate_functional_signatures(
        self,
        loader,
        mc_samples: int = 20,
        eps: float = 1e-12,
        verbose: bool = False,
    ):
        """
        Estimate:

        P:
            Component-specific functional participation.

        Model distribution y ~ p(y | x):
            F_m: diagonal model Fisher
            S_m: F_m / P
            C_unit_m: component <-> individual LoRA-target coupling
            C_layer_m: component <-> whole LoRA-layer coupling
            C_global_m: component <-> whole LoRA mechanism coupling

        Observed labels y = y*:
            U_e: mean functional score
            F_e: observed-label squared-score second moment
            S_e: F_e / P
            B_e: U_e^2 / F_e
            C_unit_e: centered component <-> individual LoRA-target coupling
            C_layer_e: centered component <-> whole LoRA-layer coupling
            C_global_e: centered component <-> whole LoRA mechanism coupling
        """
        if mc_samples < 1:
            raise ValueError("mc_samples must be positive")

        self._create_functional_artifacts()
        self._reset_functional_artifacts()
        self._set_gate_grads(True)
        self.eval()

        gate_names, gates = zip(*self.functional_gates.items())

        names = self._functional_names
        shapes = tuple(self._functional_shapes[name] for name in names)
        sizes = tuple(math.prod(shape) for shape in shapes)
        total_size = sum(sizes)

        n_layers, n_couplings = self.coupling_gates.shape
        use_couplings = n_couplings > 0
        device = gates[0].device

        participation_sum = gates[0].new_zeros(total_size)

        fisher_sum_m = gates[0].new_zeros(total_size)
        score_sum_e = gates[0].new_zeros(total_size)
        fisher_sum_e = gates[0].new_zeros(total_size)

        if use_couplings:
            coupling_unit_size = n_layers * n_couplings
            coupling_size = coupling_unit_size + n_layers + 1

            cross_sum_m = gates[0].new_zeros(
                total_size,
                coupling_size,
            )
            coupling_square_sum_m = gates[0].new_zeros(coupling_size)

            cross_sum_e = gates[0].new_zeros(
                total_size,
                coupling_size,
            )
            coupling_score_sum_e = gates[0].new_zeros(coupling_size)
            coupling_square_sum_e = gates[0].new_zeros(coupling_size)

        params = (*gates, self.coupling_gates) if use_couplings else gates

        def build_scores(raw_scores):
            scores = dict(
                zip(
                    gate_names,
                    raw_scores[: len(gates)],
                )
            )

            # Exact tied-gate coarse component scores.
            scores["residual_carry"] = scores["residual_token_carry"].sum(dim=-1)

            scores["sublayer_update"] = scores["sublayer_token_update"].sum(dim=-1)

            scores["attention_selection_route"] = scores[
                "attention_head_selection_route"
            ].sum(dim=2)

            scores["attention_transport_route"] = scores[
                "attention_head_transport_route"
            ].sum(dim=2)

            scores["attention_head_output"] = scores[
                "attention_head_transport_route"
            ].sum(dim=-1)

            functional_scores = torch.cat(
                [scores[name].flatten(start_dim=1) for name in names],
                dim=1,
            )

            if use_couplings:
                # Exact tied-gate coupling scores:
                #
                # unit:   one LoRA target at one layer
                # layer:  all LoRA targets at one layer
                # global: every LoRA target in every layer
                coupling_unit = raw_scores[-1]
                coupling_layer = coupling_unit.sum(dim=-1)
                coupling_global = coupling_layer.sum(
                    dim=-1,
                    keepdim=True,
                )

                coupling_scores = torch.cat(
                    (
                        coupling_unit.flatten(start_dim=1),
                        coupling_layer,
                        coupling_global,
                    ),
                    dim=1,
                )
            else:
                coupling_scores = None

            return functional_scores, coupling_scores

        bases = {}

        input_count = 0
        sample_count_m = 0
        sample_count_e = 0

        iterator = (
            tqdm(
                loader,
                desc="Functional signatures",
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
                for value in self.functional_participation.values():
                    value.zero_()

            log_probs = self(x).log_softmax(dim=-1)

            participation_sum.add_(
                torch.cat(
                    [self.functional_participation[name].reshape(-1) for name in names]
                )
            )

            input_count += x.shape[0]

            # Model Fisher: y ~ p(y | x).
            labels_m = torch.multinomial(
                log_probs.detach().exp(),
                mc_samples,
                replacement=True,
            )

            selected_m = log_probs.gather(
                1,
                labels_m,
            ).flatten()

            # Observed-label score: y = y*.
            selected_e = log_probs.gather(
                1,
                y.reshape(-1, 1),
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

            functional_scores, coupling_scores = build_scores(raw_scores)

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
            self._set_gate_grads(False)
            raise ValueError("loader is empty")

        participation = participation_sum.div_(input_count)

        fisher_m = fisher_sum_m.div_(sample_count_m)

        mean_e = score_sum_e.div_(sample_count_e)
        fisher_e = fisher_sum_e.div_(sample_count_e)
        variance_e = (fisher_e - mean_e.square()).clamp_min_(0)

        def unpack(value, tail=()):
            return {
                name: part.reshape(
                    *shape,
                    *tail,
                )
                for name, shape, part in zip(
                    names,
                    shapes,
                    value.split(sizes),
                )
            }

        def make_S(F_signature):
            signature = {}

            for name in names:
                p = P_signature[name]
                f = F_signature[name]

                threshold = eps * p.amax().clamp_min(1.0)

                signature[name] = torch.where(
                    p > threshold,
                    f / p.clamp_min(threshold),
                    torch.zeros_like(f),
                )

            return signature

        P_signature = unpack(participation)

        F_m_signature = unpack(fisher_m)
        S_m_signature = make_S(F_m_signature)

        U_e_signature = unpack(mean_e)
        F_e_signature = unpack(fisher_e)
        S_e_signature = make_S(F_e_signature)

        bias_e = mean_e.square()

        bias_e.div_(fisher_e).nan_to_num_(
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_(0, 1)

        B_e_signature = unpack(bias_e)

        if use_couplings:
            coupling_second_m = coupling_square_sum_m.div_(sample_count_m)

            # C_m = E[s r] / sqrt(E[s²] E[r²]).
            C_m = cross_sum_m.div_(sample_count_m)

            C_m.div_(fisher_m.sqrt()[:, None])

            C_m.div_(coupling_second_m.sqrt()[None])

            C_m.nan_to_num_(
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_(-1, 1)

            coupling_mean_e = coupling_score_sum_e.div_(sample_count_e)

            coupling_second_e = coupling_square_sum_e.div_(sample_count_e)

            coupling_variance_e = (
                coupling_second_e - coupling_mean_e.square()
            ).clamp_min_(0)

            # C_e = Corr(s, r).
            C_e = cross_sum_e.div_(sample_count_e)

            C_e.addmm_(
                mean_e[:, None],
                coupling_mean_e[None],
                alpha=-1.0,
            )

            C_e.div_(variance_e.sqrt()[:, None])

            C_e.div_(coupling_variance_e.sqrt()[None])

            C_e.nan_to_num_(
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_(-1, 1)

            unit = slice(
                0,
                coupling_unit_size,
            )

            layer = slice(
                coupling_unit_size,
                coupling_unit_size + n_layers,
            )

            C_unit_m_signature = unpack(
                C_m[:, unit],
                (n_layers, n_couplings),
            )

            C_layer_m_signature = unpack(
                C_m[:, layer],
                (n_layers,),
            )

            C_global_m_signature = unpack(C_m[:, -1])

            C_unit_e_signature = unpack(
                C_e[:, unit],
                (n_layers, n_couplings),
            )

            C_layer_e_signature = unpack(
                C_e[:, layer],
                (n_layers,),
            )

            C_global_e_signature = unpack(C_e[:, -1])

        else:
            C_unit_m_signature = unpack(
                fisher_m.new_zeros(
                    total_size,
                    n_layers,
                    0,
                ),
                (n_layers, 0),
            )

            C_layer_m_signature = unpack(
                fisher_m.new_zeros(
                    total_size,
                    n_layers,
                ),
                (n_layers,),
            )

            C_global_m_signature = unpack(fisher_m.new_zeros(total_size))

            C_unit_e_signature = unpack(
                fisher_e.new_zeros(
                    total_size,
                    n_layers,
                    0,
                ),
                (n_layers, 0),
            )

            C_layer_e_signature = unpack(
                fisher_e.new_zeros(
                    total_size,
                    n_layers,
                ),
                (n_layers,),
            )

            C_global_e_signature = unpack(fisher_e.new_zeros(total_size))

        self._set_gate_grads(False)

        return {
            "P": P_signature,
            "F_m": F_m_signature,
            "S_m": S_m_signature,
            "C_unit_m": C_unit_m_signature,
            "C_layer_m": C_layer_m_signature,
            "C_global_m": C_global_m_signature,
            "U_e": U_e_signature,
            "F_e": F_e_signature,
            "S_e": S_e_signature,
            "B_e": B_e_signature,
            "C_unit_e": C_unit_e_signature,
            "C_layer_e": C_layer_e_signature,
            "C_global_e": C_global_e_signature,
        }
