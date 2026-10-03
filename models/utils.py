from collections.abc import Mapping, Sequence
from pathlib import Path
import math

import torch

_BASIS_VERSION = 1


def _low_rank_product_svd(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Thin SVD of left @ right without materializing the square product."""
    q_left, r_left = torch.linalg.qr(left, mode="reduced")
    q_right, r_right = torch.linalg.qr(right.T, mode="reduced")
    u_small, s, vh_small = torch.linalg.svd(
        r_left @ r_right.T,
        full_matrices=False,
    )
    return q_left @ u_small, s, q_right @ vh_small.T


def _create_spectral_basis(
    attention_input_weights: Sequence[torch.Tensor],
    attention_output_weights: Sequence[torch.Tensor],
    mlp_input_weights: Sequence[torch.Tensor],
    mlp_output_weights: Sequence[torch.Tensor],
    num_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Create the fixed pretrained QK/OV/MLP spectral coordinate system."""
    qk_u, qk_s, qk_v = [], [], []
    ov_u, ov_s, ov_v = [], [], []
    mlp_in_u, mlp_in_s, mlp_in_v = [], [], []
    mlp_out_u = []

    with torch.no_grad():
        for attention_in, attention_out, mlp_in, mlp_out in zip(
            attention_input_weights,
            attention_output_weights,
            mlp_input_weights,
            mlp_output_weights,
            strict=True,
        ):
            attention_in = attention_in.detach().float()
            attention_out = attention_out.detach().float()
            mlp_in = mlp_in.detach().float()
            mlp_out = mlp_out.detach().float()

            query, key, value = attention_in.chunk(3, dim=0)
            layer_qk_u, layer_qk_s, layer_qk_v = [], [], []
            layer_ov_u, layer_ov_s, layer_ov_v = [], [], []

            for head_idx in range(num_heads):
                start = head_idx * head_dim
                stop = start + head_dim

                # W_Q^T W_K: residual-stream bilinear selection operator.
                u, s, v = _low_rank_product_svd(
                    query[start:stop].T,
                    key[start:stop],
                )
                layer_qk_u.append(u)
                layer_qk_s.append(s)
                layer_qk_v.append(v)

                # W_O^h W_V^h: residual-stream transport/write operator.
                u, s, v = _low_rank_product_svd(
                    attention_out[:, start:stop],
                    value[start:stop],
                )
                layer_ov_u.append(u)
                layer_ov_s.append(s)
                layer_ov_v.append(v)

            qk_u.append(torch.stack(layer_qk_u))
            qk_s.append(torch.stack(layer_qk_s))
            qk_v.append(torch.stack(layer_qk_v))
            ov_u.append(torch.stack(layer_ov_u))
            ov_s.append(torch.stack(layer_ov_s))
            ov_v.append(torch.stack(layer_ov_v))

            u, s, vh = torch.linalg.svd(mlp_in, full_matrices=False)
            mlp_in_u.append(u)
            mlp_in_s.append(s)
            mlp_in_v.append(vh.T)

            u, _, _ = torch.linalg.svd(mlp_out, full_matrices=False)
            mlp_out_u.append(u)

    return {
        "attention_qk_u": torch.stack(qk_u).cpu(),
        "attention_qk_s": torch.stack(qk_s).cpu(),
        "attention_qk_v": torch.stack(qk_v).cpu(),
        "attention_ov_u": torch.stack(ov_u).cpu(),
        "attention_ov_s": torch.stack(ov_s).cpu(),
        "attention_ov_v": torch.stack(ov_v).cpu(),
        "mlp_in_u": torch.stack(mlp_in_u).cpu(),
        "mlp_in_s": torch.stack(mlp_in_s).cpu(),
        "mlp_in_v": torch.stack(mlp_in_v).cpu(),
        "mlp_out_u": torch.stack(mlp_out_u).cpu(),
    }


def load_or_create_spectral_basis(
    attention_input_weights: Sequence[torch.Tensor],
    attention_output_weights: Sequence[torch.Tensor],
    mlp_input_weights: Sequence[torch.Tensor],
    mlp_output_weights: Sequence[torch.Tensor],
    *,
    num_heads: int,
    head_dim: int,
    cache_path: str | Path,
    allow_create: bool = True,
) -> dict[str, torch.Tensor]:
    """Load the shared pretrained spectral basis, or create it once."""
    attention_input_weights = tuple(attention_input_weights)
    attention_output_weights = tuple(attention_output_weights)
    mlp_input_weights = tuple(mlp_input_weights)
    mlp_output_weights = tuple(mlp_output_weights)

    n_layers = len(attention_input_weights)
    if not (
        n_layers
        == len(attention_output_weights)
        == len(mlp_input_weights)
        == len(mlp_output_weights)
        > 0
    ):
        raise ValueError(
            "all weight sequences must contain the same non-zero number of layers"
        )

    embed_dim = int(attention_output_weights[0].shape[0])
    mlp_dim = int(mlp_input_weights[0].shape[0])
    metadata = {
        "version": _BASIS_VERSION,
        "num_layers": n_layers,
        "num_heads": int(num_heads),
        "head_dim": int(head_dim),
        "embed_dim": embed_dim,
        "mlp_dim": mlp_dim,
    }

    path = Path(cache_path)
    if path.is_file():
        try:
            cached = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            cached = torch.load(path, map_location="cpu")
        if isinstance(cached, Mapping) and cached.get("metadata") == metadata:
            return cached["basis"]

    if not allow_create:
        raise RuntimeError(
            f"No compatible pretrained spectral basis exists at {path}. "
            "Create it first from a frozen pretrained backbone."
        )

    basis = _create_spectral_basis(
        attention_input_weights,
        attention_output_weights,
        mlp_input_weights,
        mlp_output_weights,
        num_heads,
        head_dim,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "basis": basis}, path)
    return basis


def functional_seed_reliability(signatures: Sequence[Mapping]) -> dict:
    """Mean pairwise cosine similarity and normalized angular distance across model seeds."""
    signatures = tuple(signatures)
    if len(signatures) < 2:
        raise ValueError("at least two model-seed signatures are required")

    statistics = ("F", "P", "S")
    components = tuple(signatures[0]["F"])
    result = {"n_seeds": len(signatures), **{name: {} for name in statistics}}
    pairs = torch.triu_indices(len(signatures), len(signatures), offset=1)

    for statistic in statistics:
        for component in components:
            values = torch.stack(
                [
                    torch.as_tensor(signature[statistic][component])
                    .reshape(-1)
                    .double()
                    for signature in signatures
                ]
            )
            norms = values.norm(dim=1)
            similarity = values @ values.T / (norms[:, None] * norms[None, :])
            pairwise = similarity[pairs[0], pairs[1]].clamp(-1.0, 1.0)
            angular = torch.acos(pairwise) / math.pi
            result[statistic][component] = {
                "mean_cosine_similarity": float(pairwise.mean()),
                "mean_angular_distance": float(angular.mean()),
            }

    return result
