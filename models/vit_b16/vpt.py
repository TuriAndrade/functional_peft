import math
from types import MethodType

import torch
import torch.nn as nn

from ..peft_model import PEFTModel
from .backbone import ViTB16Backbone


class ViTB16VPT(PEFTModel):
    """Visual Prompt Tuning for ViT-B/16."""

    def __init__(
        self,
        img_size: int,
        n_classes: int,
        freeze_backbone: bool = True,
        prompt_length: int = 0,
        prompt_mode: str = "shallow",
        prompt_dropout: float | None = None,
    ):
        if prompt_length < 0:
            raise ValueError("prompt_length must be non-negative")
        if prompt_mode not in {"shallow", "deep"}:
            raise ValueError("prompt_mode must be 'shallow' or 'deep'")
        if prompt_dropout is not None and not 0.0 <= prompt_dropout < 1.0:
            raise ValueError("prompt_dropout must be in [0, 1)")

        super().__init__(
            backbone=ViTB16Backbone(img_size=img_size),
            n_classes=n_classes,
            freeze_backbone=freeze_backbone,
        )

        self.img_size = img_size
        self.prompt_length = prompt_length
        self.prompt_mode = prompt_mode
        self.prompt_dropout = (
            (0.0 if prompt_mode == "shallow" else 0.1)
            if prompt_dropout is None
            else float(prompt_dropout)
        )

        self.prompt = nn.Parameter(
            torch.zeros(
                self.backbone.num_layers if prompt_mode == "deep" else 1,
                prompt_length,
                self.backbone.embed_dim,
            )
        )
        self.prompt_dropout_layer = nn.Dropout(self.prompt_dropout)

        self.init_peft()

    def init_peft(
        self,
        init_from: torch.Tensor | None = None,
    ) -> None:
        if self.prompt_length == 0:
            return

        if init_from is not None:
            if init_from.shape != self.prompt.shape:
                raise ValueError(
                    f"Prompt shape mismatch: expected {tuple(self.prompt.shape)}, "
                    f"got {tuple(init_from.shape)}."
                )

            with torch.no_grad():
                self.prompt.copy_(init_from)

            return

        val = math.sqrt(
            6.0 / (3 * self.backbone.patch_size**2 + self.backbone.embed_dim)
        )
        nn.init.uniform_(self.prompt, -val, val)

    def get_prompt_tokens(
        self,
        batch_size: int,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        prompt = self.prompt[layer_idx if self.prompt_mode == "deep" else 0]

        return self.prompt_dropout_layer(
            prompt.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )
        )

    def encode_tokens(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Run the encoder with shallow or deep prompt injection."""
        if self.prompt_length == 0:
            return self.backbone.encode_tokens(tokens)

        hidden = tokens
        batch_size = hidden.shape[0]

        if self.prompt_mode == "shallow":
            hidden = torch.cat(
                (
                    hidden[:, :1],
                    self.get_prompt_tokens(batch_size),
                    hidden[:, 1:],
                ),
                dim=1,
            )

            for block in self.backbone.blocks:
                hidden, _ = block(hidden)

        else:
            for layer_idx, block in enumerate(self.backbone.blocks):
                patch_start = 1 if layer_idx == 0 else 1 + self.prompt_length

                hidden = torch.cat(
                    (
                        hidden[:, :1],
                        self.get_prompt_tokens(
                            batch_size,
                            layer_idx,
                        ),
                        hidden[:, patch_start:],
                    ),
                    dim=1,
                )

                hidden, _ = block(hidden)

        return self.backbone.norm(hidden)

    def forward_features(
        self,
        x: torch.Tensor,
    ):
        batch_size = x.shape[0]

        patch_tokens = self.backbone.get_patch_tokens(x)
        cls_token = self.backbone.get_cls_token(batch_size)
        cls_pos, patch_pos = self.backbone.get_positional_embeddings()

        tokens = torch.cat(
            (
                cls_token + cls_pos,
                patch_tokens + patch_pos,
            ),
            dim=1,
        )

        encoded = self.encode_tokens(self.backbone.input_dropout(tokens))

        prompt_end = 1 + self.prompt_length

        return {
            "cls": encoded[:, 0],
            "prompt_tokens": encoded[:, 1:prompt_end],
            "patch_tokens": encoded[:, prompt_end:],
        }

    def get_layerwise_features(
        self,
        x: torch.Tensor,
    ):
        """
        Return CLS, prompt, and patch representations after every Transformer layer.

        Returns:
            cls:
                [num_layers, B, embed_dim]
            prompt_tokens:
                [num_layers, B, prompt_length, embed_dim]
            patch_tokens:
                [num_layers, B, num_patches, embed_dim]
        """
        batch_size = x.shape[0]

        patch_tokens = self.backbone.get_patch_tokens(x)
        cls_token = self.backbone.get_cls_token(batch_size)
        cls_pos, patch_pos = self.backbone.get_positional_embeddings()

        hidden = torch.cat(
            (
                cls_token + cls_pos,
                patch_tokens + patch_pos,
            ),
            dim=1,
        )
        hidden = self.backbone.input_dropout(hidden)

        cls_features = []
        prompt_features = []
        patch_features = []

        if self.prompt_mode == "shallow":
            hidden = torch.cat(
                (
                    hidden[:, :1],
                    self.get_prompt_tokens(batch_size),
                    hidden[:, 1:],
                ),
                dim=1,
            )

            for block in self.backbone.blocks:
                hidden, _ = block(hidden)

                prompt_end = 1 + self.prompt_length

                cls_features.append(hidden[:, 0])
                prompt_features.append(hidden[:, 1:prompt_end])
                patch_features.append(hidden[:, prompt_end:])

        else:
            for layer_idx, block in enumerate(self.backbone.blocks):
                patch_start = 1 if layer_idx == 0 else 1 + self.prompt_length

                hidden = torch.cat(
                    (
                        hidden[:, :1],
                        self.get_prompt_tokens(batch_size, layer_idx),
                        hidden[:, patch_start:],
                    ),
                    dim=1,
                )

                hidden, _ = block(hidden)

                prompt_end = 1 + self.prompt_length

                cls_features.append(hidden[:, 0])
                prompt_features.append(hidden[:, 1:prompt_end])
                patch_features.append(hidden[:, prompt_end:])

        return {
            "cls": torch.stack(cls_features, dim=0),
            "prompt_tokens": torch.stack(prompt_features, dim=0),
            "patch_tokens": torch.stack(patch_features, dim=0),
        }

    def save(
        self,
        path: str,
    ) -> None:
        state_dict = {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if (
                key == "prompt"
                or key.startswith("head.")
                or (not self.is_backbone_frozen and key.startswith("backbone."))
            )
        }

        torch.save(
            state_dict,
            path,
        )

    # ======================================================================
    # VPT-specific functional extensions
    # ======================================================================

    def _extend_functional_scores(
        self,
        scores: dict[str, torch.Tensor],
    ) -> None:
        scores["attention_head_output"] += scores[
            "prompt_attention_head_transport_route"
        ].sum(dim=-1)

        scores["prompt_attention_selection_route"] = scores[
            "prompt_attention_head_selection_route"
        ].sum(dim=2)

        scores["prompt_attention_transport_route"] = scores[
            "prompt_attention_head_transport_route"
        ].sum(dim=2)

    def _record_attention_route_participation(
        self,
        attention_weights: torch.Tensor,
        attention_probs: torch.Tensor,
        attention_value: torch.Tensor,
        layer_idx: int,
    ) -> None:
        super()._record_attention_route_participation(
            attention_weights,
            attention_probs,
            attention_value,
            layer_idx,
        )

        masks = self._functional_route_masks[7:]

        selection = torch.einsum(
            "bhij,rij->bhr",
            attention_weights.detach(),
            masks,
        )

        with torch.no_grad():
            context = torch.einsum(
                "bhij,rij,bhjk->bhrik",
                attention_probs.detach(),
                masks,
                attention_value.detach(),
            )

            head_transport = torch.einsum(
                "bhrik,hkl,bhril->bhr",
                context,
                self._functional_head_metrics[layer_idx],
                context,
            )

            (
                batch_size,
                n_heads,
                n_routes,
                n_tokens,
                head_dim,
            ) = context.shape

            context = context.permute(
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

            output = context @ self.backbone.get_attention_output_weight(layer_idx).T

            transport = output.square().sum(dim=(2, 3))

        self._record_functional_participation(
            "prompt_attention_head_selection_route",
            layer_idx,
            selection,
        )

        self._record_functional_participation(
            "prompt_attention_selection_route",
            layer_idx,
            selection.sum(dim=1),
        )

        self._record_functional_participation(
            "prompt_attention_head_transport_route",
            layer_idx,
            head_transport,
        )

        self._record_functional_participation(
            "prompt_attention_transport_route",
            layer_idx,
            transport,
        )

    def _create_functional_artifacts(
        self,
    ) -> None:
        if hasattr(self, "functional_gates"):
            return

        n_layers = self.backbone.num_layers
        n_heads = self.backbone.num_heads
        head_dim = self.backbone.head_dim

        n_prompts = self.prompt_length
        n_patches = self.backbone.num_patches
        n_tokens = 1 + n_prompts + n_patches

        cls_slice = slice(0, 1)
        prompt_slice = slice(
            1,
            1 + n_prompts,
        )
        patch_slice = slice(
            1 + n_prompts,
            None,
        )

        measure_slices = (
            cls_slice,
            patch_slice,
        )

        apply_slices = (
            cls_slice,
            prompt_slice,
            patch_slice,
        )

        def prompt_passthrough(gate):
            return (
                gate[0],
                None,
                gate[1],
            )

        # ------------------------------------------------------------------
        # Common state
        # ------------------------------------------------------------------

        self._initialize_functional_state(
            (
                n_layers,
                n_prompts,
            )
        )

        # ------------------------------------------------------------------
        # Prompt-specific functional state
        # ------------------------------------------------------------------

        prompt_gate_shape = (
            n_layers,
            n_heads,
            2,
        )

        for name in (
            "prompt_attention_head_selection_route",
            "prompt_attention_head_transport_route",
        ):
            self.functional_gates[name] = self._make_gate(prompt_gate_shape)

            self._register_functional_participation(
                name,
                prompt_gate_shape,
            )

        for name in (
            "prompt_attention_selection_route",
            "prompt_attention_transport_route",
        ):
            self._register_functional_participation(
                name,
                (
                    n_layers,
                    2,
                ),
            )

        # ------------------------------------------------------------------
        # Common routes + prompt -> CLS/PATCH routes
        # ------------------------------------------------------------------

        self._build_route_masks(
            n_tokens,
            patch_slice,
        )

        prompt_masks = self._functional_route_masks.new_zeros(
            2,
            n_tokens,
            n_tokens,
        )

        prompt_masks[
            0,
            cls_slice,
            prompt_slice,
        ] = 1

        prompt_masks[
            1,
            patch_slice,
            prompt_slice,
        ] = 1

        self._functional_route_masks = torch.cat(
            (
                self._functional_route_masks,
                prompt_masks,
            )
        )

        # ------------------------------------------------------------------
        # Instrumented Transformer forward
        # ------------------------------------------------------------------

        def make_forward(layer_idx):
            def forward(block, hidden):
                gates = self.functional_gates
                attention = block.attn

                # ==========================================================
                # Attention input
                # ==========================================================

                residual = hidden
                value = block.attention_norm(hidden)

                self._record_functional_participation(
                    "sublayer_input_channel",
                    (
                        layer_idx,
                        0,
                    ),
                    self._grouped_channel_energy(
                        value,
                        measure_slices,
                    ),
                )

                value = self._apply_groups(
                    value,
                    apply_slices,
                    prompt_passthrough(
                        gates["sublayer_input_channel"][
                            layer_idx,
                            0,
                        ]
                    ),
                )

                query = attention.transpose_for_scores(attention.query(value))

                key = attention.transpose_for_scores(attention.key(value))

                attention_value = attention.transpose_for_scores(attention.value(value))

                attention_scores = (
                    query
                    @ key.transpose(
                        -1,
                        -2,
                    )
                    / math.sqrt(head_dim)
                )

                # ==========================================================
                # Prompt coupling
                # ==========================================================

                if n_prompts:
                    coupling = self.coupling_gates[layer_idx]

                    coupling_log = torch.where(
                        coupling > 0,
                        torch.log(coupling.clamp_min(torch.finfo(coupling.dtype).tiny)),
                        torch.full_like(
                            coupling,
                            -torch.inf,
                        ),
                    )

                    coupling_bias = attention_scores.new_zeros(
                        n_tokens,
                        n_tokens,
                    )

                    coupling_bias[
                        cls_slice,
                        prompt_slice,
                    ] = coupling_log

                    coupling_bias[
                        patch_slice,
                        prompt_slice,
                    ] = coupling_log

                    attention_scores = (
                        attention_scores
                        + coupling_bias[
                            None,
                            None,
                        ]
                    )

                # ==========================================================
                # Attention selection
                # ==========================================================

                selection_gates = torch.cat(
                    (
                        gates["attention_head_selection_route"][layer_idx],
                        gates["prompt_attention_head_selection_route"][layer_idx],
                    ),
                    dim=-1,
                )

                attention_scores = self._apply_attention_selection_gates(
                    attention_scores,
                    selection_gates,
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

                # ==========================================================
                # Attention transport
                # ==========================================================

                transport_gates = torch.cat(
                    (
                        gates["attention_head_transport_route"][layer_idx],
                        gates["prompt_attention_head_transport_route"][layer_idx],
                    ),
                    dim=-1,
                )

                context = self._apply_attention_transport_gates(
                    attention_probs,
                    attention_value,
                    transport_gates,
                )

                self._record_attention_head_output_participation(
                    torch.cat(
                        (
                            context[
                                :,
                                :,
                                cls_slice,
                            ],
                            context[
                                :,
                                :,
                                patch_slice,
                            ],
                        ),
                        dim=2,
                    ),
                    layer_idx,
                )

                context = context.transpose(
                    1,
                    2,
                ).reshape(
                    value.shape[0],
                    n_tokens,
                    n_heads * head_dim,
                )

                attention_output = attention.proj_dropout(attention.out(context))

                # ==========================================================
                # Attention residual
                # ==========================================================

                self._record_sublayer_participation(
                    layer_idx,
                    0,
                    residual,
                    attention_output,
                    measure_slices,
                )

                hidden = self._apply_groups(
                    residual,
                    apply_slices,
                    prompt_passthrough(
                        gates["residual_token_carry"][
                            layer_idx,
                            0,
                        ]
                    ),
                ) + self._apply_groups(
                    attention_output,
                    apply_slices,
                    prompt_passthrough(
                        gates["sublayer_token_update"][
                            layer_idx,
                            0,
                        ]
                    ),
                )

                # ==========================================================
                # MLP
                # ==========================================================

                residual = hidden
                feed_forward = block.ffn

                value = block.ffn_norm(hidden)

                self._record_functional_participation(
                    "sublayer_input_channel",
                    (
                        layer_idx,
                        1,
                    ),
                    self._grouped_channel_energy(
                        value,
                        measure_slices,
                    ),
                )

                value = self._apply_groups(
                    value,
                    apply_slices,
                    prompt_passthrough(
                        gates["sublayer_input_channel"][
                            layer_idx,
                            1,
                        ]
                    ),
                )

                value = feed_forward.dropout(
                    feed_forward.act_fn(feed_forward.fc1(value))
                )

                self._record_mlp_neuron_participation(
                    layer_idx,
                    value,
                    measure_slices,
                )

                value = self._apply_groups(
                    value,
                    apply_slices,
                    prompt_passthrough(gates["mlp_neuron"][layer_idx]),
                )

                mlp_output = feed_forward.dropout(feed_forward.fc2(value))

                # ==========================================================
                # MLP residual
                # ==========================================================

                self._record_sublayer_participation(
                    layer_idx,
                    1,
                    residual,
                    mlp_output,
                    measure_slices,
                )

                return (
                    self._apply_groups(
                        residual,
                        apply_slices,
                        prompt_passthrough(
                            gates["residual_token_carry"][
                                layer_idx,
                                1,
                            ]
                        ),
                    )
                    + self._apply_groups(
                        mlp_output,
                        apply_slices,
                        prompt_passthrough(
                            gates["sublayer_token_update"][
                                layer_idx,
                                1,
                            ]
                        ),
                    ),
                    weights,
                )

            return forward

        for layer_idx, block in enumerate(self.backbone.blocks):
            block.forward = MethodType(
                make_forward(layer_idx),
                block,
            )

        # ------------------------------------------------------------------
        # Final CLS readout
        # ------------------------------------------------------------------

        original_norm_forward = self.backbone.norm.forward

        def norm_forward(_, hidden):
            return self._apply_final_cls_readout_gate(original_norm_forward(hidden))

        self.backbone.norm.forward = MethodType(
            norm_forward,
            self.backbone.norm,
        )
