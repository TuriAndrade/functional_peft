import math
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..peft_model import PEFTModel
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

        self.scale = float(scale)

        self.down = nn.Linear(embed_dim, bottleneck_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, embed_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.dropout(self.act(self.down(x)))

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.up(self.forward_features(x)) * self.scale


class ViTB16AdaptFormer(PEFTModel):
    """AdaptFormer for ViT-B/16."""

    def __init__(
        self,
        img_size: int,
        n_classes: int,
        freeze_backbone: bool = True,
        bottleneck_dim: int = 64,
        adapter_scale: float = 0.1,
        adapter_dropout: float = 0.1,
    ):
        if bottleneck_dim < 0:
            raise ValueError("bottleneck_dim must be non-negative")

        if bottleneck_dim > 0 and adapter_scale <= 0:
            raise ValueError("adapter_scale must be positive when bottleneck_dim > 0")

        if not 0.0 <= adapter_dropout < 1.0:
            raise ValueError("adapter_dropout must be in [0, 1)")

        super().__init__(
            backbone=ViTB16Backbone(img_size=img_size),
            n_classes=n_classes,
            freeze_backbone=freeze_backbone,
        )

        self.img_size = img_size
        self.bottleneck_dim = bottleneck_dim
        self.adapter_scale = float(adapter_scale)
        self.adapter_dropout = float(adapter_dropout)

        self.adapters = nn.ModuleList(
            [
                AdaptFormerAdapter(
                    self.backbone.embed_dim,
                    bottleneck_dim,
                    self.adapter_scale,
                    self.adapter_dropout,
                )
                for _ in range(self.backbone.num_layers)
            ]
            if bottleneck_dim > 0
            else []
        )

        self.init_peft()

        if bottleneck_dim > 0:
            self._patch_adaptformer_ffns()

    def init_peft(
        self,
        init_from: nn.ModuleList | None = None,
    ) -> None:
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

    def _patch_adaptformer_ffns(self) -> None:
        def make_forward(
            layer_idx,
            original_forward,
        ):
            def forward(_, x):
                return original_forward(x) + self.adapters[layer_idx](x)

            return forward

        for layer_idx, block in enumerate(self.backbone.blocks):
            block.ffn.forward = MethodType(
                make_forward(
                    layer_idx,
                    block.ffn.forward,
                ),
                block.ffn,
            )

    def save(
        self,
        path: str,
    ) -> None:
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

    # ======================================================================
    # Functional instrumentation
    # ======================================================================

    def _create_functional_artifacts(self) -> None:
        if hasattr(self, "functional_gates"):
            return

        n_layers = self.backbone.num_layers
        n_heads = self.backbone.num_heads
        head_dim = self.backbone.head_dim
        n_tokens = 1 + self.backbone.num_patches

        patch_slice = slice(1, None)
        group_slices = (
            slice(0, 1),
            patch_slice,
        )

        self._initialize_functional_state(
            (
                n_layers,
                (self.bottleneck_dim + 1 if self.bottleneck_dim > 0 else 0),
            )
        )

        self._build_route_masks(
            n_tokens,
            patch_slice,
        )

        def make_forward(layer_idx):
            def forward(block, hidden):
                gates = self.functional_gates
                attention = block.attn

                # ==========================================================
                # Attention
                # ==========================================================

                residual = hidden
                value = block.attention_norm(hidden)

                self._record_functional_participation(
                    "sublayer_input_channel",
                    (layer_idx, 0),
                    self._grouped_channel_energy(
                        value,
                        group_slices,
                    ),
                )

                value = self._apply_groups(
                    value,
                    group_slices,
                    gates["sublayer_input_channel"][
                        layer_idx,
                        0,
                    ],
                )

                query = attention.transpose_for_scores(attention.query(value))

                key = attention.transpose_for_scores(attention.key(value))

                attention_value = attention.transpose_for_scores(attention.value(value))

                attention_scores = query @ key.transpose(-1, -2) / math.sqrt(head_dim)

                attention_scores = self._apply_attention_selection_gates(
                    attention_scores,
                    gates["attention_head_selection_route"][layer_idx],
                )

                attention_weights = attention.softmax(attention_scores)

                weights = attention_weights if attention.vis else None

                attention_probs = attention.attn_dropout(attention_weights)

                self._record_attention_route_participation(
                    attention_weights,
                    attention_probs,
                    attention_value,
                    layer_idx,
                )

                context = self._apply_attention_transport_gates(
                    attention_probs,
                    attention_value,
                    gates["attention_head_transport_route"][layer_idx],
                )

                self._record_attention_head_output_participation(
                    context,
                    layer_idx,
                )

                context = context.transpose(1, 2).reshape(
                    value.shape[0],
                    n_tokens,
                    n_heads * head_dim,
                )

                attention_output = attention.proj_dropout(attention.out(context))

                self._record_sublayer_participation(
                    layer_idx,
                    0,
                    residual,
                    attention_output,
                    group_slices,
                )

                hidden = self._apply_groups(
                    residual,
                    group_slices,
                    gates["residual_token_carry"][
                        layer_idx,
                        0,
                    ],
                ) + self._apply_groups(
                    attention_output,
                    group_slices,
                    gates["sublayer_token_update"][
                        layer_idx,
                        0,
                    ],
                )

                # ==========================================================
                # MLP + AdaptFormer
                # ==========================================================

                residual = hidden
                feed_forward = block.ffn
                value = block.ffn_norm(hidden)

                self._record_functional_participation(
                    "sublayer_input_channel",
                    (layer_idx, 1),
                    self._grouped_channel_energy(
                        value,
                        group_slices,
                    ),
                )

                # Both FFN and adapter consume this same gated input.
                value = self._apply_groups(
                    value,
                    group_slices,
                    gates["sublayer_input_channel"][
                        layer_idx,
                        1,
                    ],
                )

                # ----------------------------------------------------------
                # Pretrained FFN branch
                # ----------------------------------------------------------

                ffn_hidden = feed_forward.dropout(
                    feed_forward.act_fn(feed_forward.fc1(value))
                )

                self._record_mlp_neuron_participation(
                    layer_idx,
                    ffn_hidden,
                    group_slices,
                )

                ffn_hidden = self._apply_groups(
                    ffn_hidden,
                    group_slices,
                    gates["mlp_neuron"][layer_idx],
                )

                mlp_output = feed_forward.dropout(feed_forward.fc2(ffn_hidden))

                # ----------------------------------------------------------
                # AdaptFormer branch
                # ----------------------------------------------------------

                if self.bottleneck_dim > 0:
                    adapter = self.adapters[layer_idx]
                    coupling = self.coupling_gates[layer_idx]

                    adapter_features = adapter.forward_features(value) * coupling[:-1]

                    adapter_output = (
                        F.linear(
                            adapter_features,
                            adapter.up.weight,
                            coupling[-1] * adapter.up.bias,
                        )
                        * adapter.scale
                    )

                    mlp_output = mlp_output + adapter_output

                # ----------------------------------------------------------
                # Complete MLP update
                # ----------------------------------------------------------

                self._record_sublayer_participation(
                    layer_idx,
                    1,
                    residual,
                    mlp_output,
                    group_slices,
                )

                return (
                    self._apply_groups(
                        residual,
                        group_slices,
                        gates["residual_token_carry"][
                            layer_idx,
                            1,
                        ],
                    )
                    + self._apply_groups(
                        mlp_output,
                        group_slices,
                        gates["sublayer_token_update"][
                            layer_idx,
                            1,
                        ],
                    ),
                    weights,
                )

            return forward

        for layer_idx, block in enumerate(self.backbone.blocks):
            block.forward = MethodType(
                make_forward(layer_idx),
                block,
            )

        original_norm_forward = self.backbone.norm.forward

        def norm_forward(_, hidden):
            return self._apply_final_cls_readout_gate(original_norm_forward(hidden))

        self.backbone.norm.forward = MethodType(
            norm_forward,
            self.backbone.norm,
        )
