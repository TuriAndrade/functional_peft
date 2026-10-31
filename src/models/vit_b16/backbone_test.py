import pytest
import torch

from ..transformer_backbone import TransformerBackbone
from .backbone import ViTB16Backbone


@pytest.fixture(scope="module")
def model():
    return ViTB16Backbone(img_size=224)


def test_backbone_interface(model):
    assert isinstance(model, TransformerBackbone)


def test_backbone_metadata(model):
    assert model.embed_dim == 768
    assert model.num_layers == 12
    assert model.num_heads == 12
    assert model.head_dim == 64
    assert model.mlp_dim == 3072
    assert model.patch_size == 16
    assert model.grid_size == 14
    assert model.num_patches == 196


def test_get_patch_tokens_shape(model):
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        patch_tokens = model.get_patch_tokens(x)
    assert patch_tokens.shape == (2, 196, 768)


def test_get_cls_token_shape(model):
    assert model.get_cls_token(batch_size=2).shape == (2, 1, 768)


def test_positional_embeddings_shape(model):
    cls_pos, patch_pos = model.get_positional_embeddings()
    assert cls_pos.shape == (1, 1, 768)
    assert patch_pos.shape == (1, 196, 768)


def test_encode_tokens_shape(model):
    tokens = torch.randn(2, 197, 768)
    with torch.no_grad():
        encoded = model.encode_tokens(tokens)
    assert encoded.shape == (2, 197, 768)


def test_forward_features_shape(model):
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model.forward_features(x)
    assert set(out) == {"cls", "patch_tokens"}
    assert out["cls"].shape == (2, 768)
    assert out["patch_tokens"].shape == (2, 196, 768)


def test_freeze_backbone(model):
    model.freeze()
    assert all(not p.requires_grad for p in model.parameters())


def test_unfreeze_backbone(model):
    model.freeze()
    model.unfreeze()
    assert all(p.requires_grad for p in model.parameters())
