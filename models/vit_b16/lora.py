import math
from types import MethodType
from timm.layers import DropPath

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..peft_model import PEFTModel
from .backbone import ViTB16Backbone


class ViTB16LoRA(PEFTModel):
    """LoRA for ViT-B/16."""

    def __init__(
        self,
        img_size: int,
        n_classes: int,
        freeze_backbone: bool = True,
        drop_path_rate: float = 0.1,
        rank: int = 8,
        lora_alpha: float | None = None,
        lora_targets=("query", "value"),
    ):
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")
        if rank < 0:
            raise ValueError("rank must be non-negative")

        lora_targets = tuple(lora_targets)

        if len(set(lora_targets)) != len(lora_targets):
            raise ValueError("lora_targets must not contain duplicates")

        if any(name not in {"query", "value"} for name in lora_targets):
            raise ValueError("lora_targets may contain only 'query' and 'value'")

        if rank > 0 and not lora_targets:
            raise ValueError("lora_targets must be non-empty when rank > 0")

        lora_alpha = float(rank if lora_alpha is None else lora_alpha)

        if rank > 0 and lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive when rank > 0")

        super().__init__(
            backbone=ViTB16Backbone(img_size=img_size),
            n_classes=n_classes,
            freeze_backbone=freeze_backbone,
        )

        self.drop_path_rate = float(drop_path_rate)
        drop_path_rates = torch.linspace(
            0.0,
            self.drop_path_rate,
            self.backbone.num_layers,
        ).tolist()
        self.drop_paths = nn.ModuleList(
            DropPath(rate) if rate > 0.0 else nn.Identity() for rate in drop_path_rates
        )

        self.img_size = img_size
        self.rank = rank
        self.lora_targets = lora_targets
        self.lora_alpha = lora_alpha
        self.lora_scaling = lora_alpha / rank if rank > 0 else 0.0

        if rank > 0:
            reference = next(self.backbone.parameters())

            self.lora_A = nn.Parameter(
                reference.new_empty(
                    self.backbone.num_layers,
                    len(lora_targets),
                    rank,
                    self.backbone.embed_dim,
                )
            )
            self.lora_B = nn.Parameter(
                reference.new_empty(
                    self.backbone.num_layers,
                    len(lora_targets),
                    self.backbone.embed_dim,
                    rank,
                )
            )
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.init_peft()

        if rank > 0:
            self._patch_lora_projections()

        self._patch_drop_path()

    def _patch_drop_path(self) -> None:
        def make_forward(layer_idx):
            def forward(block, hidden):
                residual = hidden

                hidden = block.attention_norm(hidden)
                hidden, weights = block.attn(hidden)
                hidden = residual + self.drop_paths[layer_idx](hidden)

                residual = hidden

                hidden = block.ffn_norm(hidden)
                hidden = block.ffn(hidden)
                hidden = residual + self.drop_paths[layer_idx](hidden)

                return hidden, weights

            return forward

        for layer_idx, block in enumerate(self.backbone.blocks):
            block.forward = MethodType(
                make_forward(layer_idx),
                block,
            )

    # ======================================================================
    # LoRA
    # ======================================================================

    def init_peft(
        self,
        init_from: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
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

    def _patch_lora_projections(self) -> None:
        def make_forward(layer_idx, target_idx, original_forward):
            def forward(_, x):
                return original_forward(x) + self._lora_delta(
                    x,
                    layer_idx,
                    target_idx,
                )

            return forward

        for layer_idx, block in enumerate(self.backbone.blocks):
            for target_idx, target_name in enumerate(self.lora_targets):
                projection = getattr(block.attn, target_name)

                projection.forward = MethodType(
                    make_forward(
                        layer_idx,
                        target_idx,
                        projection.forward,
                    ),
                    projection,
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
                or key in {"lora_A", "lora_B"}
                or (not self.is_backbone_frozen and key.startswith("backbone."))
            )
        }

        torch.save(state_dict, path)

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

        cls_slice = slice(0, 1)
        patch_slice = slice(1, None)
        group_slices = (
            cls_slice,
            patch_slice,
        )

        target_indices = (
            {name: idx for idx, name in enumerate(self.lora_targets)}
            if self.rank > 0
            else {}
        )

        self._initialize_functional_state(
            (
                n_layers,
                len(target_indices),
            )
        )

        self._build_route_masks(
            n_tokens,
            patch_slice,
        )

        def make_forward(layer_idx):
            def project(attention, name, value):
                projection = getattr(attention, name)

                result = F.linear(
                    value,
                    projection.weight,
                    projection.bias,
                )

                target_idx = target_indices.get(name)

                if target_idx is not None:
                    result = result + self.coupling_gates[
                        layer_idx,
                        target_idx,
                    ] * self._lora_delta(
                        value,
                        layer_idx,
                        target_idx,
                    )

                return result

            def forward(block, hidden):
                gates = self.functional_gates
                attention = block.attn

                # ----------------------------------------------------------
                # Attention
                # ----------------------------------------------------------

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

                query = attention.transpose_for_scores(
                    project(
                        attention,
                        "query",
                        value,
                    )
                )

                key = attention.transpose_for_scores(
                    project(
                        attention,
                        "key",
                        value,
                    )
                )

                attention_value = attention.transpose_for_scores(
                    project(
                        attention,
                        "value",
                        value,
                    )
                )

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
                attention_output = self.drop_paths[layer_idx](attention_output)

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

                # ----------------------------------------------------------
                # MLP
                # ----------------------------------------------------------

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

                value = self._apply_groups(
                    value,
                    group_slices,
                    gates["sublayer_input_channel"][
                        layer_idx,
                        1,
                    ],
                )

                value = feed_forward.dropout(
                    feed_forward.act_fn(feed_forward.fc1(value))
                )

                self._record_mlp_neuron_participation(
                    layer_idx,
                    value,
                    group_slices,
                )

                value = self._apply_groups(
                    value,
                    group_slices,
                    gates["mlp_neuron"][layer_idx],
                )

                mlp_output = feed_forward.dropout(feed_forward.fc2(value))
                mlp_output = self.drop_paths[layer_idx](mlp_output)

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
