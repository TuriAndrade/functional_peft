import pytest
import torch
import torch.nn.functional as F

from ..backbone import ViTB16Backbone
from .vpt import ViTB16VPT


IMG_SIZE = 224
N_CLASSES = 10
PROMPT_LENGTH = 4


@pytest.fixture(scope="module")
def backbone():
    model = ViTB16Backbone(img_size=IMG_SIZE)
    model.freeze()
    model.eval()
    return model


@pytest.fixture(params=["shallow", "deep"])
def model(request, backbone):
    return ViTB16VPT(
        backbone=backbone,
        n_classes=N_CLASSES,
        prompt_length=PROMPT_LENGTH,
        prompt_mode=request.param,
    )


def test_forward_features_shapes(model):
    model.eval()
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)

    with torch.no_grad():
        features = model.forward_features(x)

    assert set(features) == {"cls", "prompt_tokens", "patch_tokens"}
    assert features["cls"].shape == (2, model.backbone.embed_dim)
    assert features["prompt_tokens"].shape == (
        2,
        PROMPT_LENGTH,
        model.backbone.embed_dim,
    )
    assert features["patch_tokens"].shape == (
        2,
        model.backbone.num_patches,
        model.backbone.embed_dim,
    )


def test_forward_output_shape(model):
    model.eval()
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, N_CLASSES)


def test_prompt_parameter_shape(model):
    n_layers = model.backbone.num_layers if model.prompt_mode == "deep" else 1
    assert model.prompt.shape == (n_layers, PROMPT_LENGTH, model.backbone.embed_dim)


def test_zero_prompt_length(backbone):
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)

    with torch.no_grad():
        expected = backbone.forward_features(x)

    for prompt_mode in ("shallow", "deep"):
        model = ViTB16VPT(
            backbone=backbone,
            n_classes=N_CLASSES,
            prompt_length=0,
            prompt_mode=prompt_mode,
        )
        model.eval()

        with torch.no_grad():
            features = model.forward_features(x)

        assert features["prompt_tokens"].shape == (2, 0, backbone.embed_dim)
        assert torch.equal(features["cls"], expected["cls"])
        assert torch.equal(features["patch_tokens"], expected["patch_tokens"])


def test_prompt_receives_gradients(model):
    model.train()
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    target = torch.randint(0, N_CLASSES, (2,))

    F.cross_entropy(model(x), target).backward()

    assert model.prompt.grad is not None
    assert model.prompt.grad.shape == model.prompt.shape


def test_frozen_backbone_receives_no_gradients(model):
    model.train()
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    target = torch.randint(0, N_CLASSES, (2,))

    F.cross_entropy(model(x), target).backward()

    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(not p.requires_grad for p in model.backbone.parameters())


def test_train_does_not_change_backbone_mode(model):
    backbone = model.backbone
    assert not backbone.training

    model.train()
    assert model.training
    assert model.head.training
    assert model.prompt_dropout_layer.training
    assert not backbone.training

    model.eval()
    assert not model.training
    assert not model.head.training
    assert not model.prompt_dropout_layer.training
    assert not backbone.training


def test_invalid_prompt_mode_raises(backbone):
    with pytest.raises(ValueError, match="prompt_mode must be 'shallow' or 'deep'"):
        ViTB16VPT(
            backbone=backbone,
            n_classes=N_CLASSES,
            prompt_length=PROMPT_LENGTH,
            prompt_mode="invalid",
        )


def test_compute_loss(model):
    model.train()
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    y = torch.randint(0, N_CLASSES, (2,))

    logits, loss, components = model.compute_loss(x, y)

    assert logits.shape == (2, N_CLASSES)
    assert logits.grad_fn is not None
    assert loss.shape == ()
    assert loss.grad_fn is not None
    assert "cross_entropy_loss" in components

    model.eval()
    with torch.no_grad():
        logits, loss, components = model.compute_loss(x, y)

    assert components["cross_entropy_loss"] == pytest.approx(
        F.cross_entropy(logits, y).item()
    )


def test_save_and_load_trainable_parameters(backbone, tmp_path):
    for prompt_mode in ("shallow", "deep"):
        source = ViTB16VPT(backbone, N_CLASSES, PROMPT_LENGTH, prompt_mode)
        target = ViTB16VPT(backbone, N_CLASSES, PROMPT_LENGTH, prompt_mode)

        with torch.no_grad():
            source.prompt.fill_(1.23)
            source.head.weight.fill_(4.56)
            source.head.bias.fill_(7.89)

        path = tmp_path / f"vpt_{prompt_mode}.pt"
        source.save(path)

        state = torch.load(path, map_location="cpu", weights_only=True)
        assert set(state) == {"prompt", "head.weight", "head.bias"}

        target.load_from_path(path)

        assert torch.equal(target.prompt, source.prompt)
        assert torch.equal(target.head.weight, source.head.weight)
        assert torch.equal(target.head.bias, source.head.bias)
