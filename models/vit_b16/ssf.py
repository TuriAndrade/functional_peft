import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from types import MethodType
from typing import Optional
from tqdm.auto import tqdm
from .backbone import ViTB16Backbone


class ViTB16SSF(nn.Module):
    """SSF model implemented for a ViT-B/16 backbone."""

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
        super().__init__()

        ssf_targets = tuple(ssf_targets)

        if len(set(ssf_targets)) != len(ssf_targets):
            raise ValueError("ssf_targets must not contain duplicates")

        if any(name not in self._ALL_SSF_TARGETS for name in ssf_targets):
            raise ValueError(
                "ssf_targets may contain only "
                + ", ".join(repr(name) for name in self._ALL_SSF_TARGETS)
            )

        self.img_size = img_size
        self.n_classes = n_classes
        self.is_backbone_frozen = freeze_backbone
        self.ssf_targets = ssf_targets

        # Backbone
        self.backbone = ViTB16Backbone(img_size=img_size)

        if freeze_backbone:
            self.freeze_backbone()

        # Head
        self.head = nn.Linear(self.backbone.embed_dim, n_classes)
        self.init_head()

        # SSF
        self.ssf_scale = nn.ParameterDict()
        self.ssf_shift = nn.ParameterDict()

        if ssf_targets:
            n_layers = len(self.backbone.blocks)
            embed_dim = self.backbone.embed_dim
            mlp_dim = self.backbone.blocks[0].ffn.fc1.out_features
            reference = next(self.backbone.parameters())

            shapes = {
                "patch": (embed_dim,),
                "attention_norm": (n_layers, embed_dim),
                "query": (n_layers, embed_dim),
                "key": (n_layers, embed_dim),
                "value": (n_layers, embed_dim),
                "attention_out": (n_layers, embed_dim),
                "ffn_norm": (n_layers, embed_dim),
                "fc1": (n_layers, mlp_dim),
                "fc2": (n_layers, embed_dim),
                "final_norm": (embed_dim,),
            }

            for name in ssf_targets:
                self.ssf_scale[name] = nn.Parameter(reference.new_empty(shapes[name]))
                self.ssf_shift[name] = nn.Parameter(reference.new_empty(shapes[name]))

            self.init_peft()
            self._patch_ssf_targets()

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
        init_from: Optional[tuple[nn.ParameterDict, nn.ParameterDict]] = None,
    ):
        if not self.ssf_targets:
            return

        if init_from is not None:
            scales, shifts = init_from

            if set(scales) != set(self.ssf_scale) or set(shifts) != set(self.ssf_shift):
                raise ValueError(
                    f"SSF target mismatch: expected {tuple(self.ssf_targets)}, "
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

    def _apply_ssf(
        self,
        x: torch.Tensor,
        name: str,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        scale = self.ssf_scale[name]
        shift = self.ssf_shift[name]

        if layer_idx is not None:
            scale = scale[layer_idx]
            shift = shift[layer_idx]

        if x.shape[-1] == scale.shape[0]:
            return x * scale + shift

        if x.ndim >= 2 and x.shape[1] == scale.shape[0]:
            shape = (1, -1) + (1,) * (x.ndim - 2)
            return x * scale.view(shape) + shift.view(shape)

        raise ValueError(
            f"SSF dimension mismatch for target {name!r}: "
            f"feature shape {tuple(x.shape)}, parameter shape {tuple(scale.shape)}"
        )

    def _patch_ssf_targets(self):
        def patch(module, name, layer_idx=None):
            original_forward = module.forward

            def forward(module, x):
                return self._apply_ssf(
                    original_forward(x),
                    name,
                    layer_idx,
                )

            module.forward = MethodType(forward, module)

        if "patch" in self.ssf_targets:
            patch(self.backbone.embeddings.patch_embeddings, "patch")

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
                    patch(layer_targets[name](block), name, layer_idx)

        if "final_norm" in self.ssf_targets:
            patch(self.backbone.norm, "final_norm")

    def forward_features(self, x: torch.Tensor):
        """Standard SSF forward through the adapted backbone."""
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
        Save SSF parameters and task head.

        If the backbone is unfrozen, also save it so ssf_targets=() can be
        used cleanly for full fine-tuning.
        """
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

        torch.save(state_dict, path)

    def load_from_path(self, path: str, map_location="cpu"):
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict, strict=False)

    def _create_functional_artifacts(self):
        """
        Create the common functional instrumentation and SSF coupling gates.

        Common LayerNorm/readout gates act before SSF, preserving the same
        frozen-backbone coordinates used by the other methods. SSF coupling
        gates contain only real transform sites, ordered as:

            patch (if selected),
            block sites in [layer, target] order,
            final_norm (if selected).

        Each coupling a applies x + a * ((gamma - 1) * x + beta), so a=1 is
        trained SSF and a=0 removes that SSF transform.
        """
        if hasattr(self, "functional_gates"):
            return

        blocks = self.backbone.blocks
        n_layers = len(blocks)
        n_heads = blocks[0].attn.num_attention_heads
        embed_dim = self.backbone.embed_dim
        head_dim = blocks[0].attn.attention_head_size
        mlp_dim = blocks[0].ffn.fc1.out_features
        n_tokens = 1 + self.backbone.num_patches
        patch_slice = slice(1, None)
        reference = next(self.backbone.parameters())

        def gate(shape):
            return nn.Parameter(reference.new_ones(shape), requires_grad=False)

        self.functional_gates = nn.ParameterDict(
            {
                "sublayer_input_channel": gate((n_layers, 2, 2, embed_dim)),
                "attention_head_selection_route": gate((n_layers, n_heads, 7)),
                "attention_head_transport_route": gate((n_layers, n_heads, 7)),
                "mlp_neuron": gate((n_layers, 2, mlp_dim)),
                "residual_token_carry": gate((n_layers, 2, 2)),
                "sublayer_token_update": gate((n_layers, 2, 2)),
                "final_cls_readout_channel": gate((embed_dim,)),
            }
        )

        self._coupling_block_targets = tuple(
            name
            for name in self._ALL_SSF_TARGETS
            if name in self.ssf_targets and name not in {"patch", "final_norm"}
        )
        self._coupling_block_index = {
            name: idx for idx, name in enumerate(self._coupling_block_targets)
        }

        n_block_targets = len(self._coupling_block_targets)
        block_start = int("patch" in self.ssf_targets)
        block_stop = block_start + n_layers * n_block_targets
        self._coupling_block_slice = slice(block_start, block_stop)

        self.coupling_gates = gate(
            (block_stop + int("final_norm" in self.ssf_targets),)
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

        self._functional_shapes = {
            **{name: tuple(g.shape) for name, g in self.functional_gates.items()},
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

        grid = self.backbone.grid_size
        coords = (
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

        distances = torch.cdist(coords, coords)
        thresholds = torch.quantile(
            distances[distances > 0],
            distances.new_tensor([1 / 3, 2 / 3]),
        )
        spatial_bins = torch.bucketize(distances, thresholds) + 1
        spatial_bins.fill_diagonal_(0)

        route_masks = reference.new_zeros(7, n_tokens, n_tokens)
        route_masks[0, 0, 0] = 1
        route_masks[1, 0, patch_slice] = 1
        route_masks[2, patch_slice, 0] = 1

        for band in range(4):
            route_masks[3 + band, patch_slice, patch_slice] = (spatial_bins == band).to(
                reference.dtype
            )

        self._functional_route_masks = route_masks

        head_metrics = []
        mlp_neuron_norms = []

        with torch.no_grad():
            for block in blocks:
                W = block.attn.out.weight

                head_metrics.append(
                    torch.stack(
                        [
                            W[:, h * head_dim : (h + 1) * head_dim].T
                            @ W[:, h * head_dim : (h + 1) * head_dim]
                            for h in range(n_heads)
                        ]
                    )
                )

                mlp_neuron_norms.append(block.ffn.fc2.weight.square().sum(dim=0))

        self._functional_head_metrics = torch.stack(head_metrics)
        self._functional_mlp_neuron_norms = torch.stack(mlp_neuron_norms)

        def raw_norm(norm, x):
            return F.layer_norm(
                x,
                norm.normalized_shape,
                norm.weight,
                norm.bias,
                norm.eps,
            )

        def apply_ssf(name, x, layer_idx=None):
            if name not in self.ssf_targets:
                return x

            scale = self.ssf_scale[name]
            shift = self.ssf_shift[name]

            if name == "patch":
                coupling = self.coupling_gates[0]

            elif name == "final_norm":
                coupling = self.coupling_gates[-1]

            else:
                target_idx = self._coupling_block_index[name]

                coupling = self.coupling_gates[
                    self._coupling_block_slice.start
                    + layer_idx * len(self._coupling_block_targets)
                    + target_idx
                ]

                scale = scale[layer_idx]
                shift = shift[layer_idx]

            scale = 1 + coupling * (scale - 1)
            shift = coupling * shift

            if x.shape[-1] == scale.shape[0]:
                return x * scale + shift

            shape = (1, -1) + (1,) * (x.ndim - 2)
            return x * scale.view(shape) + shift.view(shape)

        patch_embeddings = self.backbone.embeddings.patch_embeddings

        def patch_forward(projection, x):
            x = projection._conv_forward(
                x,
                projection.weight,
                projection.bias,
            )

            return apply_ssf("patch", x)

        patch_embeddings.forward = MethodType(
            patch_forward,
            patch_embeddings,
        )

        def make_forward(layer_idx):
            def forward(block, hidden):
                gates = self.functional_gates
                attention = block.attn

                def record(name, index, value):
                    self.functional_participation[name][index].copy_(
                        value.detach().sum(dim=0)
                    )

                def grouped_channel_energy(x):
                    energy = x.detach().square()

                    return torch.stack(
                        (
                            energy[:, 0],
                            energy[:, patch_slice].sum(dim=1),
                        ),
                        dim=1,
                    )

                def apply_channel_groups(x, group_gate):
                    return torch.cat(
                        (
                            x[:, :1] * group_gate[0][None, None],
                            x[:, patch_slice] * group_gate[1][None, None],
                        ),
                        dim=1,
                    )

                def apply_token_groups(x, group_gate):
                    return torch.cat(
                        (
                            x[:, :1] * group_gate[0],
                            x[:, patch_slice] * group_gate[1],
                        ),
                        dim=1,
                    )

                def adapted_linear(name, module, x):
                    return apply_ssf(
                        name,
                        F.linear(
                            x,
                            module.weight,
                            module.bias,
                        ),
                        layer_idx,
                    )

                def record_residual(sublayer, residual, update):
                    residual_energy = grouped_channel_energy(residual).sum(dim=-1)

                    update_energy = grouped_channel_energy(update)

                    record(
                        "residual_token_carry",
                        (layer_idx, sublayer),
                        residual_energy,
                    )

                    record(
                        "residual_carry",
                        (layer_idx, sublayer),
                        residual_energy.sum(dim=-1),
                    )

                    record(
                        "sublayer_token_update",
                        (layer_idx, sublayer),
                        update_energy.sum(dim=-1),
                    )

                    record(
                        "sublayer_update",
                        (layer_idx, sublayer),
                        update_energy.sum(dim=(1, 2)),
                    )

                # Attention input
                residual = hidden

                value = raw_norm(
                    block.attention_norm,
                    hidden,
                )

                record(
                    "sublayer_input_channel",
                    (layer_idx, 0),
                    grouped_channel_energy(value),
                )

                value = apply_channel_groups(
                    value,
                    gates["sublayer_input_channel"][layer_idx, 0],
                )

                value = apply_ssf(
                    "attention_norm",
                    value,
                    layer_idx,
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

                # Attention selection
                scores = query @ key.transpose(-1, -2) / math.sqrt(head_dim)

                selection_gates = gates["attention_head_selection_route"][layer_idx]

                log_gates = torch.log(
                    selection_gates.clamp_min(torch.finfo(selection_gates.dtype).tiny)
                )

                scores = (
                    scores
                    + torch.einsum(
                        "hr,rij->hij",
                        log_gates,
                        self._functional_route_masks,
                    )[None]
                )

                attention_weights = attention.softmax(scores)

                selection_mass = torch.einsum(
                    "bhij,rij->bhr",
                    attention_weights.detach(),
                    self._functional_route_masks,
                )

                record(
                    "attention_head_selection_route",
                    layer_idx,
                    selection_mass,
                )

                record(
                    "attention_selection_route",
                    layer_idx,
                    selection_mass.sum(dim=1),
                )

                weights = attention_weights if attention.vis else None

                attention_probs = attention.attn_dropout(attention_weights)

                # Attention transport participation
                with torch.no_grad():
                    route_context = torch.einsum(
                        "bhij,rij,bhjk->bhrik",
                        attention_probs.detach(),
                        self._functional_route_masks,
                        attention_value.detach(),
                    )

                    record(
                        "attention_head_transport_route",
                        layer_idx,
                        torch.einsum(
                            "bhrik,hkl,bhril->bhr",
                            route_context,
                            self._functional_head_metrics[layer_idx],
                            route_context,
                        ),
                    )

                    route_context = route_context.permute(
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
                        route_context,
                        attention.out.weight,
                        bias=None,
                    )

                    record(
                        "attention_transport_route",
                        layer_idx,
                        route_output.square().sum(dim=(2, 3)),
                    )

                # Attention transport intervention
                transport_scale = 1 + torch.einsum(
                    "hr,rij->hij",
                    gates["attention_head_transport_route"][layer_idx] - 1,
                    self._functional_route_masks,
                )

                context = (attention_probs * transport_scale[None]) @ attention_value

                context_detached = context.detach()

                record(
                    "attention_head_output",
                    layer_idx,
                    torch.einsum(
                        "bhtd,hde,bhte->bh",
                        context_detached,
                        self._functional_head_metrics[layer_idx],
                        context_detached,
                    ),
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

                # Attention residual
                record_residual(
                    0,
                    residual,
                    attention_output,
                )

                hidden = apply_token_groups(
                    residual,
                    gates["residual_token_carry"][layer_idx, 0],
                ) + apply_token_groups(
                    attention_output,
                    gates["sublayer_token_update"][layer_idx, 0],
                )

                # MLP input
                residual = hidden
                feed_forward = block.ffn

                value = raw_norm(
                    block.ffn_norm,
                    hidden,
                )

                record(
                    "sublayer_input_channel",
                    (layer_idx, 1),
                    grouped_channel_energy(value),
                )

                value = apply_channel_groups(
                    value,
                    gates["sublayer_input_channel"][layer_idx, 1],
                )

                value = apply_ssf(
                    "ffn_norm",
                    value,
                    layer_idx,
                )

                # MLP
                value = adapted_linear(
                    "fc1",
                    feed_forward.fc1,
                    value,
                )

                value = feed_forward.dropout(feed_forward.act_fn(value))

                record(
                    "mlp_neuron",
                    layer_idx,
                    grouped_channel_energy(value)
                    * self._functional_mlp_neuron_norms[layer_idx][
                        None,
                        None,
                    ],
                )

                value = apply_channel_groups(
                    value,
                    gates["mlp_neuron"][layer_idx],
                )

                mlp_output = adapted_linear(
                    "fc2",
                    feed_forward.fc2,
                    value,
                )

                mlp_output = feed_forward.dropout(mlp_output)

                # MLP residual
                record_residual(
                    1,
                    residual,
                    mlp_output,
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

        def norm_forward(norm, hidden):
            encoded = raw_norm(
                norm,
                hidden,
            )

            self.functional_participation["final_cls_readout_channel"].copy_(
                encoded[:, 0].detach().square().sum(dim=0)
            )

            encoded = torch.cat(
                (
                    encoded[:, :1]
                    * self.functional_gates["final_cls_readout_channel"][None, None],
                    encoded[:, 1:],
                ),
                dim=1,
            )

            return apply_ssf(
                "final_norm",
                encoded,
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
        Estimate P, F_m, S_m, U_e, F_e, S_e, B_e and SSF couplings.

        C_unit follows self._coupling_unit_names. C_layer contains only SSF
        transforms inside each Transformer block. C_global contains every
        selected SSF transform, including patch and final_norm.
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

        n_layers = len(self.backbone.blocks)
        n_units = self.coupling_gates.numel()
        n_block_targets = len(self._coupling_block_targets)
        use_couplings = n_units > 0

        coupling_size = n_units + n_layers + 1 if use_couplings else 0

        device = gates[0].device

        participation_sum = gates[0].new_zeros(total_size)
        fisher_sum_m = gates[0].new_zeros(total_size)
        score_sum_e = gates[0].new_zeros(total_size)
        fisher_sum_e = gates[0].new_zeros(total_size)

        if use_couplings:
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

            # Exact tied functional scores.
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

            if not use_couplings:
                return functional_scores, None

            # Exact SSF unit scores: one score per real transform site.
            unit_scores = raw_scores[-1]

            # Exact layer scores: only transforms physically inside block l.
            if n_block_targets:
                layer_scores = (
                    unit_scores[
                        :,
                        self._coupling_block_slice,
                    ]
                    .reshape(
                        unit_scores.shape[0],
                        n_layers,
                        n_block_targets,
                    )
                    .sum(dim=-1)
                )

            else:
                layer_scores = functional_scores.new_zeros(
                    functional_scores.shape[0],
                    n_layers,
                )

            # Exact global score: every selected SSF transform.
            global_scores = unit_scores.sum(
                dim=1,
                keepdim=True,
            )

            return (
                functional_scores,
                torch.cat(
                    (
                        unit_scores,
                        layer_scores,
                        global_scores,
                    ),
                    dim=1,
                ),
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

        P_signature = unpack(participation)

        F_m_signature = unpack(fisher_m)

        U_e_signature = unpack(mean_e)

        F_e_signature = unpack(fisher_e)

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

        S_m_signature = make_S(F_m_signature)

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
            ).clamp_(
                -1,
                1,
            )

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
            ).clamp_(
                -1,
                1,
            )

            unit = slice(
                0,
                n_units,
            )

            layer = slice(
                n_units,
                n_units + n_layers,
            )

            C_unit_m_signature = unpack(
                C_m[:, unit],
                (n_units,),
            )

            C_layer_m_signature = unpack(
                C_m[:, layer],
                (n_layers,),
            )

            C_global_m_signature = unpack(C_m[:, -1])

            C_unit_e_signature = unpack(
                C_e[:, unit],
                (n_units,),
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
                    0,
                ),
                (0,),
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
                    0,
                ),
                (0,),
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
