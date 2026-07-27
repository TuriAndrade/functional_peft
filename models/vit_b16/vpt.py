import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from types import MethodType
from typing import Optional, List
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

    def _create_component_gates(self):
        """
        Create multiplicative gates for transformer components and prompt coupling,
        then instrument every transformer block for Fisher estimation.
        """
        if hasattr(self, "fim_gates") and hasattr(self, "prompt_coupling_gates"):
            return

        blocks = self.backbone.blocks
        n_layers = len(blocks)
        n_heads = blocks[0].attn.num_attention_heads
        mlp_dim = blocks[0].ffn.fc1.out_features
        n_prompts = self.prompt_length
        n_patches = self.backbone.num_patches
        n_tokens = 1 + n_prompts + n_patches

        prompt_slice = slice(1, 1 + n_prompts)
        patch_slice = slice(1 + n_prompts, None)
        reference = next(self.backbone.parameters())

        def gate(shape):
            return nn.Parameter(reference.new_ones(shape), requires_grad=False)

        self.fim_gates = nn.ParameterDict(
            {
                # [layer, branch], branch order: attention, MLP
                "residual_branch": gate((n_layers, 2)),
                # [layer, head]
                "attention_head": gate((n_layers, n_heads)),
                # [layer, head, destination, source]
                # token order: CLS, PATCH
                "backbone_attention_transport": gate((n_layers, n_heads, 2, 2)),
                # [layer, head, destination]
                # PROMPT -> CLS/PATCH; destination order: CLS, PATCH
                "prompt_to_backbone_attention_transport": gate((n_layers, n_heads, 2)),
                # [layer, head, source]
                # CLS/PATCH/PROMPT -> PROMPT; meaningful for shallow VPT
                # source order: CLS, PATCH, PROMPT
                "prompt_destination_attention_transport": gate((n_layers, n_heads, 3)),
                # [layer, head, spatial scale]
                # scale order: SELF, LOCAL, MEDIUM, GLOBAL
                "patch_spatial_transport": gate((n_layers, n_heads, 4)),
                # [layer, hidden MLP unit]
                "mlp_neuron": gate((n_layers, mlp_dim)),
                # [layer, token type], token order: CLS, PATCH
                "mlp_token_output": gate((n_layers, 2)),
                # [layer]; PROMPT MLP output, meaningful for shallow VPT
                "prompt_mlp_token_output": gate((n_layers,)),
            }
        )

        # [layer, prompt]: joint Q/K/V participation of each prompt at each layer.
        self.prompt_coupling_gates = gate((n_layers, n_prompts))

        # Fixed SELF/LOCAL/MEDIUM/GLOBAL bands from patch-grid geometry.
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
        quantiles = torch.quantile(
            distances[distances > 0],
            distances.new_tensor([1 / 3, 2 / 3]),
        )

        spatial_bins = torch.bucketize(distances, quantiles) + 1
        spatial_bins.fill_diagonal_(0)

        self.register_buffer(
            "_fim_patch_spatial_bins",
            spatial_bins,
            persistent=False,
        )

        def make_forward(layer_idx):
            def forward(block, hidden):
                gates = self.fim_gates
                attn = block.attn

                # --------------------------------------------------------------
                # Attention
                # --------------------------------------------------------------
                residual = hidden
                x = block.attention_norm(hidden)

                # Joint participation of each prompt token before Q/K/V projection.
                if n_prompts:
                    token_scale = x.new_ones(n_tokens)
                    token_scale[prompt_slice] = self.prompt_coupling_gates[layer_idx]
                    x = x * token_scale[None, :, None]

                query = attn.transpose_for_scores(attn.query(x))
                key = attn.transpose_for_scores(attn.key(x))
                value = attn.transpose_for_scores(attn.value(x))

                attention_probs = attn.softmax(
                    query @ key.transpose(-1, -2) / math.sqrt(attn.attention_head_size)
                )
                weights = attention_probs if attn.vis else None
                attention_probs = attn.attn_dropout(attention_probs)

                # Attention transport gates scale value flow after softmax.
                transport = attention_probs.new_ones(
                    n_heads,
                    n_tokens,
                    n_tokens,
                )

                backbone_transport = gates["backbone_attention_transport"][layer_idx]
                prompt_to_backbone = gates["prompt_to_backbone_attention_transport"][
                    layer_idx
                ]
                prompt_destination = gates["prompt_destination_attention_transport"][
                    layer_idx
                ]
                spatial_transport = gates["patch_spatial_transport"][layer_idx]

                # Backbone CLS/PATCH routes.
                transport[:, 0, 0] = backbone_transport[:, 0, 0]
                transport[:, 0, patch_slice] = backbone_transport[:, 0, 1, None]
                transport[:, patch_slice, 0] = backbone_transport[:, 1, 0, None]

                # PATCH <- PATCH decomposed by spatial scale.
                spatial_scale = spatial_transport[:, self._fim_patch_spatial_bins]
                transport[:, patch_slice, patch_slice] = (
                    backbone_transport[:, 1, 1, None, None] * spatial_scale
                )

                if n_prompts:
                    # PROMPT -> backbone routes.
                    transport[:, 0, prompt_slice] = prompt_to_backbone[:, 0, None]
                    transport[:, patch_slice, prompt_slice] = prompt_to_backbone[
                        :, 1, None, None
                    ]

                    # Prompt-destination routes; functionally relevant for shallow VPT.
                    transport[:, prompt_slice, 0] = prompt_destination[:, 0, None]
                    transport[:, prompt_slice, patch_slice] = prompt_destination[
                        :, 1, None, None
                    ]
                    transport[:, prompt_slice, prompt_slice] = prompt_destination[
                        :, 2, None, None
                    ]
                else:
                    # Keep prompt-specific families connected with zero derivative.
                    transport = transport + 0.0 * (
                        prompt_to_backbone.sum() + prompt_destination.sum()
                    )

                context = (attention_probs * transport[None]) @ value

                # Attention-head contribution before the output projection.
                context = (
                    context * gates["attention_head"][layer_idx][None, :, None, None]
                )

                batch_size, _, current_tokens, _ = context.shape
                context = context.transpose(1, 2).reshape(
                    batch_size,
                    current_tokens,
                    attn.all_head_size,
                )

                attention_output = attn.proj_dropout(attn.out(context))

                hidden = (
                    residual + gates["residual_branch"][layer_idx, 0] * attention_output
                )

                # --------------------------------------------------------------
                # MLP
                # --------------------------------------------------------------
                residual = hidden
                ffn = block.ffn

                x = ffn.dropout(ffn.act_fn(ffn.fc1(block.ffn_norm(hidden))))

                # MLP hidden-unit contribution before the output projection.
                x = x * gates["mlp_neuron"][layer_idx][None, None, :]

                mlp_output = ffn.dropout(ffn.fc2(x))

                # CLS/PATCH MLP-output dependence.
                token_scale = mlp_output.new_ones(n_tokens)
                token_scale[0] = gates["mlp_token_output"][layer_idx, 0]
                token_scale[patch_slice] = gates["mlp_token_output"][layer_idx, 1]

                if n_prompts:
                    # Persistent prompt MLP output; functionally relevant for shallow VPT.
                    token_scale[prompt_slice] = gates["prompt_mlp_token_output"][
                        layer_idx
                    ]
                else:
                    # Keep the prompt-specific family connected with zero derivative.
                    token_scale = (
                        token_scale + 0.0 * gates["prompt_mlp_token_output"][layer_idx]
                    )

                mlp_output = mlp_output * token_scale[None, :, None]

                hidden = residual + gates["residual_branch"][layer_idx, 1] * mlp_output

                return hidden, weights

            return forward

        for layer_idx, block in enumerate(blocks):
            block.forward = MethodType(
                make_forward(layer_idx),
                block,
            )

    def _reset_gate_values(self):
        """
        Set all diagnostic gates to one so the instrumented model exactly
        reproduces the original model.
        """
        with torch.no_grad():
            for gate in self.fim_gates.values():
                gate.fill_(1)
            self.prompt_coupling_gates.fill_(1)

    def _set_gate_grads(
        self,
        mode: bool = True,
    ):
        """
        Enable or disable gradients for all Fisher and prompt-coupling gates.
        """
        for gate in self.fim_gates.values():
            gate.grad = None
            gate.requires_grad_(mode)

        self.prompt_coupling_gates.grad = None
        self.prompt_coupling_gates.requires_grad_(mode)

    def estimate_fim_profiles(
        self,
        loader,
        mc_samples: int = 20,
        eps: float = 1e-12,
    ):
        """
        Estimate functional Fisher and prompt-coupling profiles.

        Returns:
            F:
                Diagonal model Fisher for every functional gate.

            C:
                Pairwise Fisher-score correlation between each FIM gate and
                every [layer, prompt] coupling gate.

            R_layer:
                Multiple Fisher correlation between each FIM gate and the
                prompt-coupling score subspace of each layer.

            R_global:
                Multiple Fisher correlation between each FIM gate and the
                complete prompt-coupling score subspace.
        """
        self._create_component_gates()
        self._reset_gate_values()
        self._set_gate_grads(True)
        self.eval()

        names, gates = zip(*self.fim_gates.items())
        shapes = tuple(gate.shape for gate in gates)
        sizes = tuple(gate.numel() for gate in gates)

        n_fim = sum(sizes)
        n_layers, n_prompts = self.prompt_coupling_gates.shape
        n_coupling = self.prompt_coupling_gates.numel()
        use_prompts = n_coupling > 0

        device = gates[0].device

        fisher_sum = gates[0].new_zeros(n_fim)

        if use_prompts:
            cross_sum = gates[0].new_zeros(
                n_fim,
                n_coupling,
            )
            prompt_gram_sum = gates[0].new_zeros(
                n_coupling,
                n_coupling,
            )

        params = (*gates, self.prompt_coupling_gates) if use_prompts else gates

        bases = {}
        sample_count = 0

        for batch in loader:
            x = batch[0].to(device, non_blocking=True)

            # Model Fisher: sample labels from the predictive distribution.
            log_probs = self(x).log_softmax(dim=-1)
            labels = torch.multinomial(
                log_probs.detach().exp(),
                mc_samples,
                replacement=True,
            )

            selected = log_probs.gather(1, labels).flatten()
            n_samples = selected.numel()

            # One score vector per sampled prediction.
            basis = bases.get(n_samples)
            if basis is None:
                basis = bases[n_samples] = torch.eye(
                    n_samples,
                    device=device,
                    dtype=selected.dtype,
                )

            scores = torch.autograd.grad(
                selected,
                params,
                grad_outputs=basis,
                is_grads_batched=True,
            )

            fim_scores = torch.cat(
                [score.flatten(1) for score in scores[: len(gates)]],
                dim=1,
            )

            fisher_sum.add_(fim_scores.square().sum(dim=0))

            if use_prompts:
                prompt_scores = scores[-1].flatten(1)

                cross_sum.add_(fim_scores.T @ prompt_scores)
                prompt_gram_sum.add_(prompt_scores.T @ prompt_scores)

            sample_count += n_samples

        fisher = fisher_sum / sample_count

        def unpack(profile, tail=()):
            return {
                name: part.reshape(*shape, *tail)
                for name, shape, part in zip(
                    names,
                    shapes,
                    profile.split(sizes),
                )
            }

        F_profile = unpack(fisher)

        if use_prompts:
            cross = cross_sum / sample_count
            prompt_gram = prompt_gram_sum / sample_count
            prompt_fisher = prompt_gram.diagonal()

            # Pairwise Fisher-score correlation.
            coupling = cross / (
                fisher[:, None] * prompt_fisher[None, :]
            ).sqrt().clamp_min(eps)

            cross_layer = cross.reshape(
                n_fim,
                n_layers,
                n_prompts,
            )

            # Layerwise prompt-coupling Fisher blocks: [L, P, P].
            prompt_layer = (
                prompt_gram.reshape(
                    n_layers,
                    n_prompts,
                    n_layers,
                    n_prompts,
                )
                .diagonal(dim1=0, dim2=2)
                .permute(2, 0, 1)
            )

            # Symmetric PSD Fisher blocks use the Hermitian pseudoinverse path.
            # The default tolerance adapts to matrix dimension and dtype.
            layer_pinv = torch.linalg.pinv(
                prompt_layer,
                hermitian=True,
            )

            global_pinv = torch.linalg.pinv(
                prompt_gram,
                hermitian=True,
            )

            Rl2 = torch.einsum(
                "glp,lpq,glq->gl",
                cross_layer,
                layer_pinv,
                cross_layer,
            ) / fisher[:, None].clamp_min(eps)

            Rg2 = (cross @ global_pinv * cross).sum(dim=-1) / fisher.clamp_min(eps)

            C_profile = unpack(
                coupling,
                (n_layers, n_prompts),
            )
            R_layer_profile = unpack(
                Rl2.clamp(0, 1).sqrt(),
                (n_layers,),
            )
            R_global_profile = unpack(Rg2.clamp(0, 1).sqrt())

        else:
            C_profile = unpack(
                fisher.new_zeros(n_fim, 0),
                (n_layers, 0),
            )
            R_layer_profile = unpack(
                fisher.new_zeros(n_fim, n_layers),
                (n_layers,),
            )
            R_global_profile = unpack(fisher.new_zeros(n_fim))

        self._set_gate_grads(False)

        return {
            "F": F_profile,
            "C": C_profile,
            "R_layer": R_layer_profile,
            "R_global": R_global_profile,
        }
