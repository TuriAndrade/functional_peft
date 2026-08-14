import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from types import MethodType
from typing import Optional
from tqdm.auto import tqdm
from .backbone import ViTB16Backbone


class ViTB16VPT(nn.Module):
    """
    VPT model implemented for a ViT-B/16 backbone.
    """

    def __init__(
        self,
        img_size: int,
        n_classes: int,
        prompt_length: int = 0,
        prompt_mode: str = "shallow",
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.img_size = img_size
        self.n_classes = n_classes
        self.prompt_length = prompt_length
        self.prompt_mode = prompt_mode
        self.is_backbone_frozen = freeze_backbone

        # Backbone
        self.backbone = ViTB16Backbone(img_size=img_size)
        if freeze_backbone:
            self.freeze_backbone()

        # Head
        self.head = nn.Linear(self.backbone.embed_dim, n_classes)
        self.init_head()

        # Prompt
        self.prompt = nn.Parameter(
            torch.zeros(
                len(self.backbone.blocks) if prompt_mode == "deep" else 1,
                self.prompt_length,
                self.backbone.embed_dim,
            )
        )
        self.prompt_dropout_layer = nn.Dropout(0.0 if prompt_mode == "shallow" else 0.1)
        self.init_prompt()

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
                or init_from.bias.shape != self.head.bias.shape
            ):
                raise ValueError(
                    f"Head shape mismatch: expected {tuple(self.head.weight.shape)}, "
                    f"got {tuple(init_from.weight.shape)}."
                )

            with torch.no_grad():
                self.head.weight.copy_(init_from.weight)
                self.head.bias.copy_(init_from.bias)
            return

        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def init_prompt(self, init_from: Optional[nn.Parameter] = None):
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

        patch_size = (self.backbone.patch_size, self.backbone.patch_size)
        prompt_dim = self.backbone.embed_dim
        val = math.sqrt(6.0 / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
        nn.init.uniform_(self.prompt.data, -val, val)

    def get_prompt_tokens(self, batch_size: int, layer_idx: int = 0) -> torch.Tensor:
        prompt = self.prompt[layer_idx if self.prompt_mode == "deep" else 0]
        prompt = prompt.unsqueeze(0).expand(batch_size, -1, -1)
        return self.prompt_dropout_layer(prompt)

    def encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Runs transformer blocks + final norm with VPT prompt injection.

        Input tokens must contain only:
        [CLS, PATCH_1, ..., PATCH_N]

        Output tokens contain:
        [CLS, PROMPT_1, ..., PROMPT_P, PATCH_1, ..., PATCH_N]
        """
        h = tokens
        batch_size = h.shape[0]

        if self.prompt_length == 0:
            return self.backbone.encode_tokens(h)

        if self.prompt_mode == "shallow":
            prompt = self.get_prompt_tokens(batch_size)
            h = torch.cat([h[:, :1], prompt, h[:, 1:]], dim=1)

            for blk in self.backbone.blocks:
                h, _ = blk(h)

            return self.backbone.norm(h)

        if self.prompt_mode == "deep":
            for layer_idx, blk in enumerate(self.backbone.blocks):
                prompt = self.get_prompt_tokens(batch_size, layer_idx)

                if layer_idx == 0:
                    h = torch.cat([h[:, :1], prompt, h[:, 1:]], dim=1)
                else:
                    h = torch.cat(
                        [h[:, :1], prompt, h[:, 1 + self.prompt_length :]],
                        dim=1,
                    )

                h, _ = blk(h)

            return self.backbone.norm(h)

        raise ValueError(f"Invalid prompt_mode: {self.prompt_mode}")

    def forward_features(self, x: torch.Tensor):
        """
        Standard VPT forward.
        Returns final CLS and patch tokens.
        """
        batch_size = x.shape[0]

        patch_tokens = self.backbone.get_patch_tokens(x)
        cls_token = self.backbone.get_cls_token(batch_size)
        cls_pos, patch_pos = self.backbone.get_positional_embeddings()

        tokens = torch.cat(
            [cls_token + cls_pos, patch_tokens + patch_pos],
            dim=1,
        )
        tokens = self.backbone.input_dropout(tokens)
        encoded = self.encode_tokens(tokens)

        prompt_offset = self.prompt_length if self.prompt_length > 0 else 0

        return {
            "cls": encoded[:, 0],
            "prompt_tokens": encoded[:, 1 : 1 + prompt_offset],
            "patch_tokens": encoded[:, 1 + prompt_offset :],
        }

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
        Save only VPT trainable parameters from model state_dict:
        prompt tokens and classification head.
        """
        trainable_state_dict = {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if key == "prompt" or key.startswith("head.")
        }

        torch.save(trainable_state_dict, path)

    def load_from_path(self, path: str, map_location="cpu"):
        """
        Load only prompt tokens and classification head.
        """
        trainable_state_dict = torch.load(path, map_location=map_location)

        self.load_state_dict(trainable_state_dict, strict=False)

    def _create_functional_artifacts(self):
        """
        Create functional gates, participation accumulators, fixed route masks,
        output-space metrics, and an instrumented ViT forward pass.

        Every gate is initialized to one, so the patched forward is identical
        to the original forward at the functional evaluation point.
        """
        if hasattr(self, "functional_gates"):
            return

        blocks = self.backbone.blocks
        n_layers = len(blocks)
        n_heads = blocks[0].attn.num_attention_heads
        embed_dim = self.backbone.embed_dim
        head_dim = blocks[0].attn.attention_head_size
        mlp_dim = blocks[0].ffn.fc1.out_features
        n_prompts = self.prompt_length
        n_patches = self.backbone.num_patches
        n_tokens = 1 + n_prompts + n_patches

        prompt_slice = slice(1, 1 + n_prompts)
        patch_slice = slice(1 + n_prompts, None)
        reference = next(self.backbone.parameters())

        def gate(shape):
            return nn.Parameter(reference.new_ones(shape), requires_grad=False)

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
                #
                # Scales the normalized pretrained residual-stream channels
                # consumed by each sublayer. Prompt-token channels are excluded
                # and pass unchanged.
                "sublayer_input_channel": gate((n_layers, 2, 2, embed_dim)),
                # [layer, head, route]
                #
                # QK/selection-side routing. Each gate scales the pre-softmax
                # attention logits belonging to one backbone-to-backbone route,
                # measuring the functional importance of where a head selects
                # information from.
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
                # [layer, head, route]
                #
                # OV/transport-side routing. Each gate scales the contribution
                # carried through one backbone-to-backbone route after the
                # attention distribution is formed, measuring the functional
                # importance of the content transported through that route.
                #
                # Routes use the same seven source/destination categories above.
                "attention_head_transport_route": gate((n_layers, n_heads, 7)),
                # [layer, processed token group, hidden MLP neuron]
                #
                # token group:
                #   0 = CLS
                #   1 = patches
                #
                # Scales each post-activation hidden MLP neuron before W_2,
                # measuring the functional importance of each frozen nonlinear
                # MLP computation for CLS and patch tokens separately.
                "mlp_neuron": gate((n_layers, 2, mlp_dim)),
                # [layer, sublayer, token group]
                #
                # sublayer:
                #   0 = attention
                #   1 = MLP
                #
                # token group:
                #   0 = CLS
                #   1 = patches
                #
                # Scales the complete post-dropout residual update produced by
                # each sublayer for CLS or patch tokens. Prompt-token updates
                # pass unchanged.
                "sublayer_token_update": gate((n_layers, 2, 2)),
                # [residual channel]
                #
                # Scales each channel of the final normalized CLS representation
                # immediately before the task classifier.
                "final_cls_readout_channel": gate((embed_dim,)),
                # [layer, head, destination group]
                #
                # VPT-specific QK/selection routes whose source is any prompt
                # token and whose destination is:
                #   0 = CLS
                #   1 = patches
                "prompt_attention_head_selection_route": gate((n_layers, n_heads, 2)),
                # [layer, head, destination group]
                #
                # VPT-specific OV/transport routes whose source is any prompt
                # token and whose destination is:
                #   0 = CLS
                #   1 = patches
                "prompt_attention_head_transport_route": gate((n_layers, n_heads, 2)),
            }
        )

        # [layer, prompt token]
        #
        # Scales each complete prompt state immediately before its Q/K/V
        # projections. These intervention scores define the prompt-side
        # variables used by C and G.
        self.coupling_gates = gate((n_layers, n_prompts))

        # Coarse components are exact tied-gate derivatives constructed from
        # the fine per-sample functional scores. Their participation values are
        # recorded directly at the corresponding computation.
        self._functional_shapes = {
            **{
                name: tuple(value.shape)
                for name, value in self.functional_gates.items()
            },
            # [layer, sublayer]
            #
            # Complete attention/MLP residual update, aggregated over CLS and
            # patch destinations.
            "sublayer_update": (n_layers, 2),
            # [layer, route]
            #
            # Head-aggregated QK/selection importance for each of the seven
            # backbone-to-backbone attention routes.
            "attention_selection_route": (n_layers, 7),
            # [layer, route]
            #
            # Head-aggregated OV/transport importance for each of the seven
            # backbone-to-backbone attention routes.
            "attention_transport_route": (n_layers, 7),
            # [layer, head]
            #
            # Complete output of each attention head delivered to backbone
            # destinations. Its score includes both backbone-source and
            # prompt-source transport contributions.
            "attention_head_output": (n_layers, n_heads),
            # [layer, destination group]
            #
            # Head-aggregated prompt->backbone QK/selection routes:
            #   0 = prompt -> CLS
            #   1 = prompt -> patches
            "prompt_attention_selection_route": (n_layers, 2),
            # [layer, destination group]
            #
            # Head-aggregated prompt->backbone OV/transport routes:
            #   0 = prompt -> CLS
            #   1 = prompt -> patches
            "prompt_attention_transport_route": (n_layers, 2),
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

        # Patch-to-patch spatial bins:
        #   0 = self
        #   1 = local
        #   2 = medium
        #   3 = global
        spatial_bins = torch.bucketize(distances, thresholds) + 1
        spatial_bins.fill_diagonal_(0)

        # Nine mutually exclusive source/destination routes ending at backbone
        # tokens. Routes 0:7 are backbone-to-backbone; routes 7:9 are
        # prompt-to-backbone. Prompt-destination rows are intentionally outside
        # this functional-route decomposition.
        route_masks = reference.new_zeros(
            9,
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

        route_masks[7, 0, prompt_slice] = 1
        route_masks[8, patch_slice, prompt_slice] = 1

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

                    # Columns of W_O corresponding to this head: [D, K].
                    head_weight = output_weight[:, head_slice]

                    # For one head-context vector c:
                    # ||W_O,h c||² = cᵀ(W_O,hᵀ W_O,h)c.
                    layer_metrics.append(head_weight.T @ head_weight)

                head_metrics.append(torch.stack(layer_metrics))

                # Output-space squared norm associated with each hidden MLP
                # neuron: ||W_2[:, j]||².
                mlp_neuron_norms.append(block.ffn.fc2.weight.square().sum(dim=0))

        self._functional_head_metrics = torch.stack(head_metrics)
        self._functional_mlp_neuron_norms = torch.stack(mlp_neuron_norms)

        # ------------------------------------------------------------------
        # Instrumented transformer-block forward
        # ------------------------------------------------------------------

        def make_forward(layer_idx):
            def forward(block, hidden):
                gates = self.functional_gates
                attention = block.attn

                def record(name, index, value):
                    """
                    value has shape [batch, *component_shape_at_index].
                    Participation is accumulated over the current batch.
                    """
                    self.functional_participation[name][index].copy_(
                        value.detach().sum(dim=0)
                    )

                def grouped_channel_energy(value):
                    """
                    [B, T, C] -> [B, 2, C]

                    Group 0 is CLS.
                    Group 1 sums over all patch tokens.
                    Prompt tokens are excluded.
                    """
                    energy = value.detach().square()

                    return torch.stack(
                        (
                            energy[:, 0],
                            energy[:, patch_slice].sum(dim=1),
                        ),
                        dim=1,
                    )

                def apply_channel_groups(value, group_gate):
                    """
                    Apply channel/neuron gates separately to CLS and patches.

                    value:      [B, T, C]
                    group_gate: [2, C]

                    Prompt tokens pass unchanged.
                    """
                    return torch.cat(
                        (
                            value[:, :1] * group_gate[0][None, None],
                            value[:, prompt_slice],
                            value[:, patch_slice] * group_gate[1][None, None],
                        ),
                        dim=1,
                    )

                def apply_token_groups(value, group_gate):
                    """
                    Apply one scalar gate to the complete CLS/patch update.

                    value:      [B, T, C]
                    group_gate: [2]

                    Prompt-token updates pass unchanged.
                    """
                    return torch.cat(
                        (
                            value[:, :1] * group_gate[0],
                            value[:, prompt_slice],
                            value[:, patch_slice] * group_gate[1],
                        ),
                        dim=1,
                    )

                def projected_head_energy(context):
                    """
                    Squared output-space energy of each head after its W_O slice.

                    context: [B, H, T, K]
                    return:  [B, H]
                    """
                    context = context.detach()

                    return torch.einsum(
                        "bhtd,hde,bhte->bh",
                        context,
                        self._functional_head_metrics[layer_idx],
                        context,
                    )

                # ==========================================================
                # Attention input channels
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

                # Prompt coupling is defined on the complete prompt state
                # entering the frozen Q/K/V projections.
                if n_prompts:
                    value = torch.cat(
                        (
                            value[:, :1],
                            value[:, prompt_slice]
                            * self.coupling_gates[layer_idx][None, :, None],
                            value[:, patch_slice],
                        ),
                        dim=1,
                    )

                query = attention.transpose_for_scores(attention.query(value))
                key = attention.transpose_for_scores(attention.key(value))
                attention_value = attention.transpose_for_scores(attention.value(value))

                attention_scores = query @ key.transpose(-1, -2) / math.sqrt(head_dim)

                # ==========================================================
                # Attention selection routes (QK side)
                # ==========================================================

                selection_route_energy = torch.einsum(
                    "bhij,rij->bhr",
                    attention_scores.detach().square(),
                    self._functional_route_masks,
                )

                record(
                    "attention_head_selection_route",
                    layer_idx,
                    selection_route_energy[:, :, :7],
                )

                record(
                    "prompt_attention_head_selection_route",
                    layer_idx,
                    selection_route_energy[:, :, 7:],
                )

                record(
                    "attention_selection_route",
                    layer_idx,
                    selection_route_energy[:, :, :7].sum(dim=1),
                )

                record(
                    "prompt_attention_selection_route",
                    layer_idx,
                    selection_route_energy[:, :, 7:].sum(dim=1),
                )

                selection_route_gates = torch.cat(
                    (
                        gates["attention_head_selection_route"][layer_idx],
                        gates["prompt_attention_head_selection_route"][layer_idx],
                    ),
                    dim=-1,
                )

                attention_scores = (
                    attention_scores
                    * (
                        1
                        + torch.einsum(
                            "hr,rij->hij",
                            selection_route_gates - 1,
                            self._functional_route_masks,
                        )
                    )[None]
                )

                attention_probs = attention.softmax(attention_scores)
                weights = attention_probs if attention.vis else None
                attention_probs = attention.attn_dropout(attention_probs)

                # ==========================================================
                # Attention transport-route participation (OV side)
                # ==========================================================

                with torch.no_grad():
                    # Route-specific context before W_O:
                    # [B, H, route, destination token, K].
                    route_context = torch.einsum(
                        "bhij,rij,bhjk->bhrik",
                        attention_probs.detach(),
                        self._functional_route_masks,
                        attention_value.detach(),
                    )

                    # Route energy after the corresponding W_O head slice.
                    transport_route_energy = torch.einsum(
                        "bhrik,hkl,bhril->bhr",
                        route_context,
                        self._functional_head_metrics[layer_idx],
                        route_context,
                    )

                    record(
                        "attention_head_transport_route",
                        layer_idx,
                        transport_route_energy[:, :, :7],
                    )

                    record(
                        "prompt_attention_head_transport_route",
                        layer_idx,
                        transport_route_energy[:, :, 7:],
                    )

                    record(
                        "attention_transport_route",
                        layer_idx,
                        transport_route_energy[:, :, :7].sum(dim=1),
                    )

                    record(
                        "prompt_attention_transport_route",
                        layer_idx,
                        transport_route_energy[:, :, 7:].sum(dim=1),
                    )

                # ==========================================================
                # Attention transport routes (OV side)
                # ==========================================================

                transport_route_gates = torch.cat(
                    (
                        gates["attention_head_transport_route"][layer_idx],
                        gates["prompt_attention_head_transport_route"][layer_idx],
                    ),
                    dim=-1,
                )

                transport_scale = 1 + torch.einsum(
                    "hr,rij->hij",
                    transport_route_gates - 1,
                    self._functional_route_masks,
                )

                context = (attention_probs * transport_scale[None]) @ attention_value

                # Complete head context delivered to backbone destinations.
                # Because the context already sums over all sources, this
                # participation includes prompt-source contributions.
                backbone_context = torch.cat(
                    (
                        context[:, :, :1],
                        context[:, :, patch_slice],
                    ),
                    dim=2,
                )

                record(
                    "attention_head_output",
                    layer_idx,
                    projected_head_energy(backbone_context),
                )

                context = context.transpose(1, 2).reshape(
                    value.shape[0],
                    n_tokens,
                    n_heads * head_dim,
                )

                attention_output = attention.proj_dropout(attention.out(context))

                # ==========================================================
                # Attention residual update
                # ==========================================================

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

                hidden = residual + apply_token_groups(
                    attention_output,
                    gates["sublayer_token_update"][layer_idx, 0],
                )

                # ==========================================================
                # MLP input channels and hidden neurons
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

                # Each hidden-neuron activation contributes through one column
                # of W_2; this measures its output-space energy.
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
                # MLP residual update
                # ==========================================================

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
                    residual
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

        # Apply the final channel gate after the original encoder LayerNorm,
        # exactly where the task classifier consumes the CLS representation.
        original_norm_forward = self.backbone.norm.forward

        def norm_forward(norm, hidden):
            encoded = original_norm_forward(hidden)

            self.functional_participation["final_cls_readout_channel"].copy_(
                encoded[:, 0].detach().square().sum(dim=0)
            )

            return torch.cat(
                (
                    encoded[:, :1]
                    * self.functional_gates["final_cls_readout_channel"][
                        None,
                        None,
                    ],
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

        P: shared functional participation

        Model distribution y ~ p(y | x):
        F_m: diagonal model Fisher
        S_m: F_m / P
        C_m: prompt-component score correlation
        G_layer_m: RMS C_m across prompts at each prompt layer
        G_global_m: RMS C_m across all prompt layers and prompts

        Empirical distribution y = y*:
        U_e: mean functional score
        F_e: empirical squared-score second moment
        S_e: F_e / P
        B_e: U_e^2 / F_e
        C_e: centered prompt-component score correlation
        G_layer_e: RMS C_e across prompts at each prompt layer
        G_global_e: RMS C_e across all prompt layers and prompts
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
        n_layers, n_prompts = self.coupling_gates.shape
        use_prompts = n_prompts > 0
        device = gates[0].device

        participation_sum = gates[0].new_zeros(total_size)

        fisher_sum_m = gates[0].new_zeros(total_size)
        score_sum_e = gates[0].new_zeros(total_size)
        fisher_sum_e = gates[0].new_zeros(total_size)

        if use_prompts:
            coupling_size = n_layers * n_prompts

            cross_sum_m = gates[0].new_zeros(total_size, coupling_size)
            coupling_square_sum_m = gates[0].new_zeros(coupling_size)

            cross_sum_e = gates[0].new_zeros(total_size, coupling_size)
            coupling_score_sum_e = gates[0].new_zeros(coupling_size)
            coupling_square_sum_e = gates[0].new_zeros(coupling_size)

        params = (*gates, self.coupling_gates) if use_prompts else gates

        def build_scores(raw_scores):
            scores = dict(zip(gate_names, raw_scores[: len(gates)]))

            # Exact tied-gate coarse scores.
            scores["sublayer_update"] = scores["sublayer_token_update"].sum(dim=-1)

            scores["attention_selection_route"] = scores[
                "attention_head_selection_route"
            ].sum(dim=2)

            scores["attention_transport_route"] = scores[
                "attention_head_transport_route"
            ].sum(dim=2)

            scores["attention_head_output"] = scores[
                "attention_head_transport_route"
            ].sum(dim=-1) + scores["prompt_attention_head_transport_route"].sum(dim=-1)

            scores["prompt_attention_selection_route"] = scores[
                "prompt_attention_head_selection_route"
            ].sum(dim=2)

            scores["prompt_attention_transport_route"] = scores[
                "prompt_attention_head_transport_route"
            ].sum(dim=2)

            functional_scores = torch.cat(
                [scores[name].flatten(start_dim=1) for name in names],
                dim=1,
            )

            coupling_scores = (
                raw_scores[-1].flatten(start_dim=1) if use_prompts else None
            )

            return functional_scores, coupling_scores

        bases = {}
        input_count = 0
        sample_count_m = 0
        sample_count_e = 0

        iterator = tqdm(loader, desc="Functional signatures") if verbose else loader

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

            batch_size = x.shape[0]
            input_count += batch_size

            # Model Fisher scores: y ~ p(y | x).
            labels_m = torch.multinomial(
                log_probs.detach().exp(),
                mc_samples,
                replacement=True,
            )

            selected_m = log_probs.gather(1, labels_m).flatten()

            # Empirical scores: y = ground-truth label.
            selected_e = log_probs.gather(1, y.reshape(-1, 1)).flatten()

            n_m = selected_m.numel()
            n_e = selected_e.numel()
            selected = torch.cat((selected_m, selected_e))

            basis = bases.get(selected.numel())

            if basis is None:
                basis = bases[selected.numel()] = torch.eye(
                    selected.numel(),
                    device=device,
                    dtype=selected.dtype,
                )

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

            if use_prompts:
                coupling_m = coupling_scores[:n_m]
                coupling_e = coupling_scores[n_m:]

                cross_sum_m.addmm_(functional_m.T, coupling_m)
                coupling_square_sum_m.add_(coupling_m.square().sum(dim=0))

                cross_sum_e.addmm_(functional_e.T, coupling_e)
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

        if use_prompts:
            coupling_second_m = coupling_square_sum_m.div_(sample_count_m)

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

            C_m_view = C_m.reshape(total_size, n_layers, n_prompts)
            C_e_view = C_e.reshape(total_size, n_layers, n_prompts)

            G_layer_m = torch.linalg.vector_norm(C_m_view, dim=-1).div_(
                math.sqrt(n_prompts)
            )
            G_layer_e = torch.linalg.vector_norm(C_e_view, dim=-1).div_(
                math.sqrt(n_prompts)
            )

            G_global_m = torch.linalg.vector_norm(G_layer_m, dim=-1).div_(
                math.sqrt(n_layers)
            )
            G_global_e = torch.linalg.vector_norm(G_layer_e, dim=-1).div_(
                math.sqrt(n_layers)
            )

            C_m_signature = unpack(C_m, (n_layers, n_prompts))
            C_e_signature = unpack(C_e, (n_layers, n_prompts))
            G_layer_m_signature = unpack(G_layer_m, (n_layers,))
            G_layer_e_signature = unpack(G_layer_e, (n_layers,))
            G_global_m_signature = unpack(G_global_m)
            G_global_e_signature = unpack(G_global_e)

        else:
            C_m_signature = unpack(
                fisher_m.new_zeros(total_size, n_layers, 0),
                (n_layers, 0),
            )
            C_e_signature = unpack(
                fisher_e.new_zeros(total_size, n_layers, 0),
                (n_layers, 0),
            )

            G_layer_m_signature = unpack(
                fisher_m.new_zeros(total_size, n_layers),
                (n_layers,),
            )
            G_layer_e_signature = unpack(
                fisher_e.new_zeros(total_size, n_layers),
                (n_layers,),
            )

            G_global_m_signature = unpack(fisher_m.new_zeros(total_size))
            G_global_e_signature = unpack(fisher_e.new_zeros(total_size))

        self._set_gate_grads(False)

        return {
            "P": P_signature,
            "F_m": F_m_signature,
            "S_m": S_m_signature,
            "C_m": C_m_signature,
            "G_layer_m": G_layer_m_signature,
            "G_global_m": G_global_m_signature,
            "U_e": U_e_signature,
            "F_e": F_e_signature,
            "S_e": S_e_signature,
            "B_e": B_e_signature,
            "C_e": C_e_signature,
            "G_layer_e": G_layer_e_signature,
            "G_global_e": G_global_e_signature,
        }
