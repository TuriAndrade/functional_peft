import pytest
import torch
from .backbone import ViTB16Backbone


@pytest.fixture(scope="module")
def model():
    backbone = ViTB16Backbone(img_size=224)
    return backbone


def test_backbone_metadata(model):
    assert model.embed_dim == 768
    assert model.patch_size == 16
    assert model.grid_size == 14
    assert model.num_patches == 196


def test_get_patch_tokens_shape(model):
    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        patch_tokens = model.get_patch_tokens(x)

    assert patch_tokens.shape == (2, 196, 768)


def test_positional_embeddings_shape(model):
    cls_pos, patch_pos = model.get_positional_embeddings()

    assert cls_pos.shape == (1, 1, 768)
    assert patch_pos.shape == (1, 196, 768)


def test_forward_features_shape(model):
    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        out = model.forward_features(x)

    assert set(out.keys()) == {"cls", "patch_tokens"}
    assert out["cls"].shape == (2, 768)
    assert out["patch_tokens"].shape == (2, 196, 768)


def test_freeze_backbone(model):
    model.freeze()
    assert all(not p.requires_grad for p in model.vit.parameters())
