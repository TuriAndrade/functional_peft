import math
from types import MethodType
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .backbone import ViTB16Backbone


class AdaptFormerAdapter(nn.Module):
    """Parallel bottleneck adapter used by AdaptFormer."""

    def __init__(
        self,
        embed_dim: int,
        bottleneck_dim: int,
        scale: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim
        self.scale = float(scale)

        self.down = nn.Linear(embed_dim, bottleneck_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.down(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.forward_features(x)) * self.scale


class ViTB16AdaptFormer(nn.Module):
    """AdaptFormer implemented for a ViT-B/16 backbone."""

    def __init__(
        self,
        img_size: int,
        n_classes: int,
        freeze_backbone: bool = True,
        bottleneck_dim: int = 64,
        adapter_scale: float = 0.1,
        adapter_dropout: float = 0.1,
    ):
        super().__init__()

        if bottleneck_dim < 0:
            raise ValueError("bottleneck_dim must be non-negative")

        if bottleneck_dim > 0 and adapter_scale <= 0:
            raise ValueError("adapter_scale must be positive when bottleneck_dim > 0")

        if not 0.0 <= adapter_dropout < 1.0:
            raise ValueError("adapter_dropout must be in [0, 1)")

        self.img_size = img_size
        self.n_classes = n_classes
        self.is_backbone_frozen = freeze_backbone
        self.bottleneck_dim = bottleneck_dim
        self.adapter_scale = float(adapter_scale)
        self.adapter_dropout = float(adapter_dropout)

        # Backbone
        self.backbone = ViTB16Backbone(img_size=img_size)

        if freeze_backbone:
            self.freeze_backbone()

        # Head
        self.head = nn.Linear(
            self.backbone.embed_dim,
            n_classes,
        )
        self.init_head()

        # AdaptFormer
        if bottleneck_dim > 0:
            self.adapters = nn.ModuleList(
                [
                    AdaptFormerAdapter(
                        embed_dim=self.backbone.embed_dim,
                        bottleneck_dim=bottleneck_dim,
                        scale=self.adapter_scale,
                        dropout=self.adapter_dropout,
                    )
                    for _ in self.backbone.blocks
                ]
            )
            self.init_peft()
            self._patch_adaptformer_ffns()

        else:
            self.adapters = nn.ModuleList()

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
        init_from: Optional[nn.ModuleList] = None,
    ):
        if self.bottleneck_dim == 0:
            return

        if init_from is not None:
            try:
                self.adapters.load_state_dict(
                    init_from.state_dict(),
                    strict=True,
                )
            except RuntimeError as exc:
                raise ValueError("AdaptFormer adapter shape mismatch.") from exc

            return

        for adapter in self.adapters:
            adapter.reset_parameters()

    def _patch_adaptformer_ffns(self):
        """
        Patch each frozen FFN so that the block computes

            h + FFN(LN(h)) + AdaptFormer(LN(h)).

        The adapter therefore receives exactly the same post-LayerNorm
        input as the pretrained FFN.
        """

        def make_forward(
            layer_idx,
            original_forward,
        ):
            def forward(ffn, x):
                return original_forward(x) + self.adapters[layer_idx](x)

            return forward

        for layer_idx, block in enumerate(self.backbone.blocks):
            ffn = block.ffn

            ffn.forward = MethodType(
                make_forward(
                    layer_idx,
                    ffn.forward,
                ),
                ffn,
            )

    def forward_features(
        self,
        x: torch.Tensor,
    ):
        """Standard AdaptFormer forward through the adapted backbone."""
        return self.backbone.forward_features(x)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
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

    def save(
        self,
        path: str,
    ):
        """
        Save AdaptFormer parameters and task head.

        If the backbone is unfrozen, also save it so bottleneck_dim=0
        can be used cleanly for full fine-tuning.
        """

        state_dict = {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if (
                key.startswith("head.")
                or key.startswith("adapters.")
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
        Create common functional gates, AdaptFormer coupling gates,
        participation accumulators, fixed route masks/metrics, and an
        instrumented ViT forward pass.

        Shared functional gates have the same meaning and shapes as
        in LoRA/VPT.

        For the MLP sublayer:

            z = LN_2(h)

        is gated once and then consumed by both branches:

            FFN(z)
            AdaptFormer(z)

        while the shared sublayer update is the complete update:

            FFN(z) + AdaptFormer(z).

        AdaptFormer coupling units are the additive contributions of
        each bottleneck feature through the up projection, plus one
        unit for the up-projection bias.

        Every gate is initialized to one, so the patched forward is
        identical to native AdaptFormer at the functional evaluation
        point.
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

        n_couplings = self.bottleneck_dim + 1 if self.bottleneck_dim > 0 else 0

        patch_slice = slice(1, None)

        reference = next(self.backbone.parameters())

        def gate(shape):
            return nn.Parameter(
                reference.new_ones(shape),
                requires_grad=False,
            )

        self.functional_gates = nn.ParameterDict(
            {
                # [layer, sublayer, token group, residual channel]
                #
                # sublayer:
                #   0 = attention
                #   1 = MLP
                #
                # token group:
                #   0 = CLS
                #   1 = patches
                "sublayer_input_channel": gate(
                    (
                        n_layers,
                        2,
                        2,
                        embed_dim,
                    )
                ),
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
                "attention_head_selection_route": gate(
                    (
                        n_layers,
                        n_heads,
                        7,
                    )
                ),
                # [layer, head, route]
                "attention_head_transport_route": gate(
                    (
                        n_layers,
                        n_heads,
                        7,
                    )
                ),
                # [layer, token group, hidden pretrained-MLP neuron]
                #
                # This remains the original frozen FFN neuron family.
                # AdaptFormer bottleneck features are not merged into it.
                "mlp_neuron": gate(
                    (
                        n_layers,
                        2,
                        mlp_dim,
                    )
                ),
                # [layer, sublayer, token group]
                "residual_token_carry": gate(
                    (
                        n_layers,
                        2,
                        2,
                    )
                ),
                # [layer, sublayer, token group]
                #
                # For the MLP sublayer this gates:
                #
                #   FFN(z) + AdaptFormer(z)
                #
                # rather than the frozen FFN alone.
                "sublayer_token_update": gate(
                    (
                        n_layers,
                        2,
                        2,
                    )
                ),
                # [residual channel]
                "final_cls_readout_channel": gate((embed_dim,)),
            }
        )

        # --------------------------------------------------------------
        # AdaptFormer coupling
        # --------------------------------------------------------------
        #
        # [layer, AdaptFormer unit]
        #
        # Units 0:bottleneck_dim scale:
        #
        #   W_up[:, j] q_j
        #
        # where q_j is one post-ReLU/post-dropout bottleneck feature.
        #
        # The final unit scales:
        #
        #   b_up
        #
        # Tying every unit in a layer therefore scales the complete
        # AdaptFormer update at that layer.
        #
        # Tying every unit in every layer scales the complete
        # AdaptFormer mechanism.

        self.coupling_gates = gate(
            (
                n_layers,
                n_couplings,
            )
        )

        # --------------------------------------------------------------
        # Functional shapes
        # --------------------------------------------------------------

        self._functional_shapes = {
            **{
                name: tuple(value.shape)
                for name, value in self.functional_gates.items()
            },
            "residual_carry": (
                n_layers,
                2,
            ),
            "sublayer_update": (
                n_layers,
                2,
            ),
            "attention_selection_route": (
                n_layers,
                7,
            ),
            "attention_transport_route": (
                n_layers,
                7,
            ),
            "attention_head_output": (
                n_layers,
                n_heads,
            ),
        }

        self._functional_names = tuple(self._functional_shapes)

        self.functional_participation = {
            name: reference.new_zeros(shape)
            for name, shape in self._functional_shapes.items()
        }

        # ==============================================================
        # Fixed attention-route masks
        # ==============================================================

        grid = self.backbone.grid_size

        coordinates = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(
                        grid,
                        device=reference.device,
                    ),
                    torch.arange(
                        grid,
                        device=reference.device,
                    ),
                    indexing="ij",
                ),
                dim=-1,
            )
            .reshape(-1, 2)
            .to(reference.dtype)
        )

        distances = torch.cdist(
            coordinates,
            coordinates,
        )

        thresholds = torch.quantile(
            distances[distances > 0],
            distances.new_tensor(
                [
                    1 / 3,
                    2 / 3,
                ]
            ),
        )

        spatial_bins = (
            torch.bucketize(
                distances,
                thresholds,
            )
            + 1
        )

        spatial_bins.fill_diagonal_(0)

        route_masks = reference.new_zeros(
            7,
            n_tokens,
            n_tokens,
        )

        # CLS <- CLS
        route_masks[
            0,
            0,
            0,
        ] = 1

        # CLS <- PATCH
        route_masks[
            1,
            0,
            patch_slice,
        ] = 1

        # PATCH <- CLS
        route_masks[
            2,
            patch_slice,
            0,
        ] = 1

        # PATCH <- PATCH spatial routes
        for band in range(4):
            route_masks[
                3 + band,
                patch_slice,
                patch_slice,
            ] = (
                spatial_bins == band
            ).to(reference.dtype)

        self._functional_route_masks = route_masks

        # ==============================================================
        # Fixed output-space energy metrics
        # ==============================================================

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

                    head_weight = output_weight[
                        :,
                        head_slice,
                    ]

                    layer_metrics.append(head_weight.T @ head_weight)

                head_metrics.append(torch.stack(layer_metrics))

                mlp_neuron_norms.append(block.ffn.fc2.weight.square().sum(dim=0))

        self._functional_head_metrics = torch.stack(head_metrics)

        self._functional_mlp_neuron_norms = torch.stack(mlp_neuron_norms)

        # ==============================================================
        # Instrumented transformer-block forward
        # ==============================================================

        def make_forward(layer_idx):
            def forward(block, hidden):
                gates = self.functional_gates
                attention = block.attn

                def record(
                    name,
                    index,
                    value,
                ):
                    self.functional_participation[name][index].copy_(
                        value.detach().sum(dim=0)
                    )

                def grouped_channel_energy(value):
                    energy = value.detach().square()

                    return torch.stack(
                        (
                            energy[:, 0],
                            energy[
                                :,
                                patch_slice,
                            ].sum(dim=1),
                        ),
                        dim=1,
                    )

                def apply_channel_groups(
                    value,
                    group_gate,
                ):
                    return torch.cat(
                        (
                            value[:, :1] * group_gate[0][None, None],
                            value[
                                :,
                                patch_slice,
                            ]
                            * group_gate[1][None, None],
                        ),
                        dim=1,
                    )

                def apply_token_groups(
                    value,
                    group_gate,
                ):
                    return torch.cat(
                        (
                            value[:, :1] * group_gate[0],
                            value[
                                :,
                                patch_slice,
                            ]
                            * group_gate[1],
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

                # ======================================================
                # Attention input
                # ======================================================

                residual = hidden

                value = block.attention_norm(hidden)

                record(
                    "sublayer_input_channel",
                    (
                        layer_idx,
                        0,
                    ),
                    grouped_channel_energy(value),
                )

                value = apply_channel_groups(
                    value,
                    gates["sublayer_input_channel"][
                        layer_idx,
                        0,
                    ],
                )

                query = attention.transpose_for_scores(attention.query(value))

                key = attention.transpose_for_scores(attention.key(value))

                attention_value = attention.transpose_for_scores(attention.value(value))

                attention_scores = query @ key.transpose(-1, -2) / math.sqrt(head_dim)

                # ======================================================
                # Attention selection
                # ======================================================

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

                # ======================================================
                # Attention transport participation
                # ======================================================

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

                    route_context_concat = route_context.permute(
                        0,
                        2,
                        3,
                        1,
                        4,
                    ).reshape(
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

                # ======================================================
                # Attention transport intervention
                # ======================================================

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

                # ======================================================
                # Attention residual
                # ======================================================

                residual_energy = grouped_channel_energy(residual).sum(dim=-1)

                record(
                    "residual_token_carry",
                    (
                        layer_idx,
                        0,
                    ),
                    residual_energy,
                )

                record(
                    "residual_carry",
                    (
                        layer_idx,
                        0,
                    ),
                    residual_energy.sum(dim=-1),
                )

                attention_energy = grouped_channel_energy(attention_output)

                record(
                    "sublayer_token_update",
                    (
                        layer_idx,
                        0,
                    ),
                    attention_energy.sum(dim=-1),
                )

                record(
                    "sublayer_update",
                    (
                        layer_idx,
                        0,
                    ),
                    attention_energy.sum(dim=(1, 2)),
                )

                hidden = apply_token_groups(
                    residual,
                    gates["residual_token_carry"][
                        layer_idx,
                        0,
                    ],
                ) + apply_token_groups(
                    attention_output,
                    gates["sublayer_token_update"][
                        layer_idx,
                        0,
                    ],
                )

                # ======================================================
                # MLP + AdaptFormer input
                # ======================================================

                residual = hidden
                feed_forward = block.ffn

                # This is the single common post-LayerNorm input
                # consumed by BOTH FFN and AdaptFormer.
                value = block.ffn_norm(hidden)

                record(
                    "sublayer_input_channel",
                    (
                        layer_idx,
                        1,
                    ),
                    grouped_channel_energy(value),
                )

                # Shared MLP input-channel gate BEFORE the split.
                value = apply_channel_groups(
                    value,
                    gates["sublayer_input_channel"][
                        layer_idx,
                        1,
                    ],
                )

                # ======================================================
                # Frozen pretrained FFN branch
                # ======================================================

                ffn_hidden = feed_forward.dropout(
                    feed_forward.act_fn(feed_forward.fc1(value))
                )

                neuron_energy = (
                    grouped_channel_energy(ffn_hidden)
                    * self._functional_mlp_neuron_norms[layer_idx][None, None]
                )

                record(
                    "mlp_neuron",
                    layer_idx,
                    neuron_energy,
                )

                ffn_hidden = apply_channel_groups(
                    ffn_hidden,
                    gates["mlp_neuron"][layer_idx],
                )

                ffn_output = feed_forward.dropout(feed_forward.fc2(ffn_hidden))

                # ======================================================
                # AdaptFormer branch
                # ======================================================

                if self.bottleneck_dim > 0:
                    adapter = self.adapters[layer_idx]

                    # q = Dropout(ReLU(W_down z + b_down))
                    adapter_features = adapter.forward_features(value)

                    coupling = self.coupling_gates[layer_idx]

                    # Each bottleneck feature is one native
                    # adaptation coordinate:
                    #
                    #   a_j W_up[:, j] q_j

                    weighted_features = (
                        adapter_features * coupling[: self.bottleneck_dim][None, None]
                    )

                    # Up-projection feature contributions.
                    adapter_output = F.linear(
                        weighted_features,
                        adapter.up.weight,
                        bias=None,
                    )

                    # Last native coordinate is the complete
                    # up-projection bias contribution.
                    adapter_output = adapter_output + coupling[-1] * adapter.up.bias

                    adapter_output = adapter_output * adapter.scale

                else:
                    adapter_output = torch.zeros_like(ffn_output)

                # ======================================================
                # Complete adapted MLP update
                # ======================================================

                mlp_output = ffn_output + adapter_output

                # ======================================================
                # MLP residual
                # ======================================================

                residual_energy = grouped_channel_energy(residual).sum(dim=-1)

                record(
                    "residual_token_carry",
                    (
                        layer_idx,
                        1,
                    ),
                    residual_energy,
                )

                record(
                    "residual_carry",
                    (
                        layer_idx,
                        1,
                    ),
                    residual_energy.sum(dim=-1),
                )

                # IMPORTANT:
                #
                # Participation is the energy of the actual complete
                # update:
                #
                #   ||FFN(z) + AdaptFormer(z)||^2
                #
                # so the FFN/adapter cross-term is preserved.
                mlp_energy = grouped_channel_energy(mlp_output)

                record(
                    "sublayer_token_update",
                    (
                        layer_idx,
                        1,
                    ),
                    mlp_energy.sum(dim=-1),
                )

                record(
                    "sublayer_update",
                    (
                        layer_idx,
                        1,
                    ),
                    mlp_energy.sum(dim=(1, 2)),
                )

                return (
                    apply_token_groups(
                        residual,
                        gates["residual_token_carry"][
                            layer_idx,
                            1,
                        ],
                    )
                    + apply_token_groups(
                        mlp_output,
                        gates["sublayer_token_update"][
                            layer_idx,
                            1,
                        ],
                    ),
                    weights,
                )

            return forward

        for layer_idx, block in enumerate(blocks):
            block.forward = MethodType(
                make_forward(layer_idx),
                block,
            )

        # ==============================================================
        # Final CLS readout
        # ==============================================================

        original_norm_forward = self.backbone.norm.forward

        def norm_forward(
            norm,
            hidden,
        ):
            encoded = original_norm_forward(hidden)

            self.functional_participation["final_cls_readout_channel"].copy_(
                encoded[
                    :,
                    0,
                ]
                .detach()
                .square()
                .sum(dim=0)
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
        """
        Reset all functional/coupling gates to identity and clear
        participation storage.
        """

        with torch.no_grad():
            for gate in self.functional_gates.values():
                gate.fill_(1)

            self.coupling_gates.fill_(1)

            for value in self.functional_participation.values():
                value.zero_()

    def _set_gate_grads(
        self,
        mode: bool = True,
    ):
        """
        Enable or disable gradients only for functional/coupling gates.
        """

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
            F_m:
                diagonal model Fisher

            S_m:
                F_m / P

            C_unit_m:
                component <-> individual AdaptFormer-unit coupling

            C_layer_m:
                component <-> complete AdaptFormer-layer coupling

            C_global_m:
                component <-> complete AdaptFormer mechanism coupling

        Observed labels y = y*:
            U_e:
                mean functional score

            F_e:
                observed-label squared-score second moment

            S_e:
                F_e / P

            B_e:
                U_e^2 / F_e

            C_unit_e:
                centered component <-> individual AdaptFormer-unit
                coupling

            C_layer_e:
                centered component <-> complete AdaptFormer-layer
                coupling

            C_global_e:
                centered component <-> complete AdaptFormer mechanism
                coupling
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

        # ==============================================================
        # Accumulators
        # ==============================================================

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

        # ==============================================================
        # Raw score construction
        # ==============================================================

        def build_scores(raw_scores):
            scores = dict(
                zip(
                    gate_names,
                    raw_scores[: len(gates)],
                )
            )

            # ----------------------------------------------------------
            # Exact tied functional components
            # ----------------------------------------------------------

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

            # ----------------------------------------------------------
            # Exact tied AdaptFormer scores
            # ----------------------------------------------------------

            if use_couplings:
                # Native AdaptFormer units:
                #
                #   [layer, bottleneck feature / up bias]
                #
                coupling_unit = raw_scores[-1]

                # Exact derivative for one common gate tied across
                # every native AdaptFormer unit in a layer.
                #
                # Since:
                #
                #   AF_l =
                #       s * (
                #           sum_j W_up[:,j] q_j
                #           + b_up
                #       )
                #
                # this is exactly the derivative for:
                #
                #   FFN_l + alpha_l AF_l
                #
                coupling_layer = coupling_unit.sum(dim=-1)

                # Exact derivative for one common gate tied across
                # every AdaptFormer unit in every layer.
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

            return (
                functional_scores,
                coupling_scores,
            )

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

        # ==============================================================
        # Dataset loop
        # ==============================================================

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

            # ==========================================================
            # Model Fisher labels
            # ==========================================================

            labels_m = torch.multinomial(
                log_probs.detach().exp(),
                mc_samples,
                replacement=True,
            )

            selected_m = log_probs.gather(
                1,
                labels_m,
            ).flatten()

            # ==========================================================
            # Observed labels
            # ==========================================================

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

            (
                functional_scores,
                coupling_scores,
            ) = build_scores(raw_scores)

            functional_m = functional_scores[:n_m]

            functional_e = functional_scores[n_m:]

            fisher_sum_m.add_(functional_m.square().sum(dim=0))

            score_sum_e.add_(functional_e.sum(dim=0))

            fisher_sum_e.add_(functional_e.square().sum(dim=0))

            # ==========================================================
            # Coupling moments
            # ==========================================================

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

        # ==============================================================
        # Functional moments
        # ==============================================================

        participation = participation_sum.div_(input_count)

        fisher_m = fisher_sum_m.div_(sample_count_m)

        mean_e = score_sum_e.div_(sample_count_e)

        fisher_e = fisher_sum_e.div_(sample_count_e)

        variance_e = (fisher_e - mean_e.square()).clamp_min_(0)

        # ==============================================================
        # Helpers
        # ==============================================================

        def unpack(
            value,
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
                    part,
                ) in zip(
                    names,
                    shapes,
                    value.split(sizes),
                )
            }

        def make_S(
            F_signature,
        ):
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

        # ==============================================================
        # Functional signatures
        # ==============================================================

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
        ).clamp_(
            0,
            1,
        )

        B_e_signature = unpack(bias_e)

        # ==============================================================
        # Coupling signatures
        # ==============================================================

        if use_couplings:
            coupling_second_m = coupling_square_sum_m.div_(sample_count_m)

            # ----------------------------------------------------------
            # Model coupling
            #
            # C_m = E[s r] / sqrt(E[s^2] E[r^2])
            # ----------------------------------------------------------

            C_m = cross_sum_m.div_(sample_count_m)

            C_m.div_(
                fisher_m.sqrt()[
                    :,
                    None,
                ]
            )

            C_m.div_(coupling_second_m.sqrt()[None])

            C_m.nan_to_num_(
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_(
                -1,
                1,
            )

            # ----------------------------------------------------------
            # Observed-label coupling
            #
            # C_e = Corr_x(s, r)
            # ----------------------------------------------------------

            coupling_mean_e = coupling_score_sum_e.div_(sample_count_e)

            coupling_second_e = coupling_square_sum_e.div_(sample_count_e)

            coupling_variance_e = (
                coupling_second_e - coupling_mean_e.square()
            ).clamp_min_(0)

            C_e = cross_sum_e.div_(sample_count_e)

            C_e.addmm_(
                mean_e[:, None],
                coupling_mean_e[None],
                alpha=-1.0,
            )

            C_e.div_(
                variance_e.sqrt()[
                    :,
                    None,
                ]
            )

            C_e.div_(coupling_variance_e.sqrt()[None])

            C_e.nan_to_num_(
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_(
                -1,
                1,
            )

            # ----------------------------------------------------------
            # Split unit / layer / global coupling
            # ----------------------------------------------------------

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
                (
                    n_layers,
                    n_couplings,
                ),
            )

            C_layer_m_signature = unpack(
                C_m[:, layer],
                (n_layers,),
            )

            C_global_m_signature = unpack(C_m[:, -1])

            C_unit_e_signature = unpack(
                C_e[:, unit],
                (
                    n_layers,
                    n_couplings,
                ),
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
                (
                    n_layers,
                    0,
                ),
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
                (
                    n_layers,
                    0,
                ),
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
