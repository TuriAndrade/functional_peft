import math
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..peft_model import PEFTModel
from .backbone import ViTB16Backbone


class ViTB16SSF(PEFTModel):
    """Scale-and-Shift Fine-Tuning for ViT-B/16."""

    _ALL_SSF_TARGETS = (
        "patch",
        "attention_norm",
        "query",
        "key",
        "value",
        "attention_out",
        "ffn_norm",
        "fc1",
        "fc2",
        "final_norm",
    )

    def __init__(
        self,
        img_size: int,
        n_classes: int,
        freeze_backbone: bool = True,
        ssf_targets=(
            "patch",
            "attention_norm",
            "query",
            "key",
            "value",
            "attention_out",
            "ffn_norm",
            "fc1",
            "fc2",
            "final_norm",
        ),
    ):
        ssf_targets = tuple(ssf_targets)

        if len(set(ssf_targets)) != len(ssf_targets):
            raise ValueError("ssf_targets must not contain duplicates")

        if any(name not in self._ALL_SSF_TARGETS for name in ssf_targets):
            raise ValueError(
                "ssf_targets may contain only "
                + ", ".join(repr(name) for name in self._ALL_SSF_TARGETS)
            )

        super().__init__(
            backbone=ViTB16Backbone(img_size=img_size),
            n_classes=n_classes,
            freeze_backbone=freeze_backbone,
        )

        self.img_size = img_size
        self.ssf_targets = ssf_targets

        self.ssf_scale = nn.ParameterDict()
        self.ssf_shift = nn.ParameterDict()

        shapes = {
            "patch": (self.backbone.embed_dim,),
            "attention_norm": (
                self.backbone.num_layers,
                self.backbone.embed_dim,
            ),
            "query": (
                self.backbone.num_layers,
                self.backbone.embed_dim,
            ),
            "key": (
                self.backbone.num_layers,
                self.backbone.embed_dim,
            ),
            "value": (
                self.backbone.num_layers,
                self.backbone.embed_dim,
            ),
            "attention_out": (
                self.backbone.num_layers,
                self.backbone.embed_dim,
            ),
            "ffn_norm": (
                self.backbone.num_layers,
                self.backbone.embed_dim,
            ),
            "fc1": (
                self.backbone.num_layers,
                self.backbone.mlp_dim,
            ),
            "fc2": (
                self.backbone.num_layers,
                self.backbone.embed_dim,
            ),
            "final_norm": (self.backbone.embed_dim,),
        }

        reference = next(self.backbone.parameters())

        for name in ssf_targets:
            self.ssf_scale[name] = nn.Parameter(reference.new_empty(shapes[name]))
            self.ssf_shift[name] = nn.Parameter(reference.new_empty(shapes[name]))

        self.init_peft()

        if ssf_targets:
            self._patch_ssf_targets()

    # ======================================================================
    # SSF
    # ======================================================================

    def init_peft(
        self,
        init_from: tuple[nn.ParameterDict, nn.ParameterDict] | None = None,
    ) -> None:
        if not self.ssf_targets:
            return

        if init_from is not None:
            scales, shifts = init_from

            if set(scales) != set(self.ssf_scale) or set(shifts) != set(self.ssf_shift):
                raise ValueError(
                    f"SSF target mismatch: expected {self.ssf_targets}, "
                    f"got scales={tuple(scales)}, shifts={tuple(shifts)}."
                )

            for name in self.ssf_targets:
                if (
                    scales[name].shape != self.ssf_scale[name].shape
                    or shifts[name].shape != self.ssf_shift[name].shape
                ):
                    raise ValueError(
                        f"SSF shape mismatch for target {name!r}: expected "
                        f"scale={tuple(self.ssf_scale[name].shape)}, "
                        f"shift={tuple(self.ssf_shift[name].shape)}, got "
                        f"scale={tuple(scales[name].shape)}, "
                        f"shift={tuple(shifts[name].shape)}."
                    )

            with torch.no_grad():
                for name in self.ssf_targets:
                    self.ssf_scale[name].copy_(scales[name])
                    self.ssf_shift[name].copy_(shifts[name])

            return

        with torch.no_grad():
            for scale in self.ssf_scale.values():
                nn.init.normal_(scale, mean=1.0, std=0.02)

            for shift in self.ssf_shift.values():
                nn.init.normal_(shift, mean=0.0, std=0.02)

    @staticmethod
    def _apply_affine(
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        if x.shape[-1] == scale.shape[0]:
            return x * scale + shift

        shape = (
            1,
            -1,
            *(1 for _ in range(x.ndim - 2)),
        )

        return x * scale.view(shape) + shift.view(shape)

    def _apply_ssf(
        self,
        x: torch.Tensor,
        name: str,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        scale = self.ssf_scale[name]
        shift = self.ssf_shift[name]

        if layer_idx is not None:
            scale = scale[layer_idx]
            shift = shift[layer_idx]

        return self._apply_affine(
            x,
            scale,
            shift,
        )

    def _patch_ssf_targets(self) -> None:
        def patch(module, name, layer_idx=None):
            original_forward = module.forward

            def forward(_, x):
                return self._apply_ssf(
                    original_forward(x),
                    name,
                    layer_idx,
                )

            module.forward = MethodType(
                forward,
                module,
            )

        if "patch" in self.ssf_targets:
            patch(
                self.backbone.embeddings.patch_embeddings,
                "patch",
            )

        layer_targets = {
            "attention_norm": lambda block: block.attention_norm,
            "query": lambda block: block.attn.query,
            "key": lambda block: block.attn.key,
            "value": lambda block: block.attn.value,
            "attention_out": lambda block: block.attn.out,
            "ffn_norm": lambda block: block.ffn_norm,
            "fc1": lambda block: block.ffn.fc1,
            "fc2": lambda block: block.ffn.fc2,
        }

        for layer_idx, block in enumerate(self.backbone.blocks):
            for name in self.ssf_targets:
                if name in layer_targets:
                    patch(
                        layer_targets[name](block),
                        name,
                        layer_idx,
                    )

        if "final_norm" in self.ssf_targets:
            patch(
                self.backbone.norm,
                "final_norm",
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
                or key.startswith("ssf_scale.")
                or key.startswith("ssf_shift.")
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

    def _build_coupling_scores(
        self,
        unit_scores: torch.Tensor,
    ) -> torch.Tensor:
        n_block_targets = len(self._coupling_block_targets)

        if n_block_targets:
            layer_scores = (
                unit_scores[
                    :,
                    self._coupling_block_slice,
                ]
                .reshape(
                    unit_scores.shape[0],
                    self.backbone.num_layers,
                    n_block_targets,
                )
                .sum(dim=-1)
            )
        else:
            layer_scores = unit_scores.new_zeros(
                unit_scores.shape[0],
                self.backbone.num_layers,
            )

        global_scores = unit_scores.sum(
            dim=-1,
            keepdim=True,
        )

        return torch.cat(
            (
                unit_scores,
                layer_scores,
                global_scores,
            ),
            dim=1,
        )

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

        # --------------------------------------------------------------
        # SSF coupling topology
        # --------------------------------------------------------------

        self._coupling_block_targets = tuple(
            name
            for name in self._ALL_SSF_TARGETS
            if (name in self.ssf_targets and name not in {"patch", "final_norm"})
        )

        n_block_targets = len(self._coupling_block_targets)

        block_start = int("patch" in self.ssf_targets)

        block_stop = block_start + n_layers * n_block_targets

        self._coupling_block_slice = slice(
            block_start,
            block_stop,
        )

        self._coupling_unit_names = (
            (("patch",) if "patch" in self.ssf_targets else ())
            + tuple(
                f"{name}[{layer_idx}]"
                for layer_idx in range(n_layers)
                for name in self._coupling_block_targets
            )
            + (("final_norm",) if "final_norm" in self.ssf_targets else ())
        )

        self._initialize_functional_state(
            (block_stop + int("final_norm" in self.ssf_targets),)
        )

        self._build_route_masks(
            n_tokens,
            patch_slice,
        )

        block_index = {
            name: idx for idx, name in enumerate(self._coupling_block_targets)
        }

        # --------------------------------------------------------------
        # Functional SSF
        # --------------------------------------------------------------

        def apply_ssf(
            name,
            x,
            layer_idx=None,
        ):
            if name not in self.ssf_targets:
                return x

            scale = self.ssf_scale[name]
            shift = self.ssf_shift[name]

            if name == "patch":
                coupling = self.coupling_gates[0]

            elif name == "final_norm":
                coupling = self.coupling_gates[-1]

            else:
                coupling = self.coupling_gates[
                    block_start + layer_idx * n_block_targets + block_index[name]
                ]

                scale = scale[layer_idx]
                shift = shift[layer_idx]

            return self._apply_affine(
                x,
                1 + coupling * (scale - 1),
                coupling * shift,
            )

        def raw_norm(
            norm,
            x,
        ):
            return F.layer_norm(
                x,
                norm.normalized_shape,
                norm.weight,
                norm.bias,
                norm.eps,
            )

        # --------------------------------------------------------------
        # Patch embedding
        # --------------------------------------------------------------

        patch_embedding = self.backbone.embeddings.patch_embeddings

        def patch_forward(
            projection,
            x,
        ):
            return apply_ssf(
                "patch",
                projection._conv_forward(
                    x,
                    projection.weight,
                    projection.bias,
                ),
            )

        patch_embedding.forward = MethodType(
            patch_forward,
            patch_embedding,
        )

        # --------------------------------------------------------------
        # Transformer blocks
        # --------------------------------------------------------------

        def make_forward(layer_idx):
            def adapted_linear(
                name,
                module,
                x,
            ):
                return apply_ssf(
                    name,
                    F.linear(
                        x,
                        module.weight,
                        module.bias,
                    ),
                    layer_idx,
                )

            def forward(block, hidden):
                gates = self.functional_gates
                attention = block.attn

                # ======================================================
                # Attention input
                # ======================================================

                residual = hidden

                value = raw_norm(
                    block.attention_norm,
                    hidden,
                )

                value = apply_ssf(
                    "attention_norm",
                    value,
                    layer_idx,
                )

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
                    adapted_linear(
                        "query",
                        attention.query,
                        value,
                    )
                )

                key = attention.transpose_for_scores(
                    adapted_linear(
                        "key",
                        attention.key,
                        value,
                    )
                )

                attention_value = attention.transpose_for_scores(
                    adapted_linear(
                        "value",
                        attention.value,
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

                # ======================================================
                # Attention transport
                # ======================================================

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

                attention_output = adapted_linear(
                    "attention_out",
                    attention.out,
                    context,
                )

                attention_output = attention.proj_dropout(attention_output)

                # ======================================================
                # Attention residual
                # ======================================================

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

                # ======================================================
                # MLP input
                # ======================================================

                residual = hidden
                feed_forward = block.ffn

                value = raw_norm(
                    block.ffn_norm,
                    hidden,
                )

                value = apply_ssf(
                    "ffn_norm",
                    value,
                    layer_idx,
                )

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

                # ======================================================
                # MLP
                # ======================================================

                value = adapted_linear(
                    "fc1",
                    feed_forward.fc1,
                    value,
                )

                value = feed_forward.dropout(feed_forward.act_fn(value))

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

                mlp_output = adapted_linear(
                    "fc2",
                    feed_forward.fc2,
                    value,
                )

                mlp_output = feed_forward.dropout(mlp_output)

                # ======================================================
                # MLP residual
                # ======================================================

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

        # --------------------------------------------------------------
        # Final norm / CLS readout
        # --------------------------------------------------------------

        final_norm = self.backbone.norm

        def norm_forward(_, hidden):
            encoded = raw_norm(
                final_norm,
                hidden,
            )

            encoded = apply_ssf(
                "final_norm",
                encoded,
            )

            return self._apply_final_cls_readout_gate(encoded)

        final_norm.forward = MethodType(
            norm_forward,
            final_norm,
        )
