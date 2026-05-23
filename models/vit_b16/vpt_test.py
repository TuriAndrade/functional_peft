import torch
import pytest
from .vpt import ViTB16VPT
import torch.nn.functional as F

IMG_SIZE = 224
N_CLASSES = 10


@pytest.mark.parametrize("prompt_mode", ["shallow", "deep"])
def test_forward_features_shapes(prompt_mode):
    model = ViTB16VPT(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        prompt_length=4,
        prompt_mode=prompt_mode,
    )
    model.eval()

    batch_size = 2
    x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)

    with torch.no_grad():
        features = model.forward_features(x)

    expected_keys = {"cls", "prompt_tokens", "patch_tokens"}
    assert set(features.keys()) == expected_keys

    assert features["cls"].shape == (batch_size, model.backbone.embed_dim)
    assert features["prompt_tokens"].shape == (
        batch_size,
        model.prompt_length,
        model.backbone.embed_dim,
    )
    assert features["patch_tokens"].shape == (
        batch_size,
        model.backbone.num_patches,
        model.backbone.embed_dim,
    )


@pytest.mark.parametrize("prompt_mode", ["shallow", "deep"])
def test_forward_output_shape(prompt_mode):
    model = ViTB16VPT(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        prompt_length=4,
        prompt_mode=prompt_mode,
    )
    model.eval()

    batch_size = 2
    x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (batch_size, N_CLASSES)


def test_prompt_parameter_shape_shallow():
    model = ViTB16VPT(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        prompt_length=4,
        prompt_mode="shallow",
    )

    assert model.prompt.shape == (1, 4, model.backbone.embed_dim)


def test_prompt_parameter_shape_deep():
    model = ViTB16VPT(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        prompt_length=4,
        prompt_mode="deep",
    )

    assert model.prompt.shape == (
        len(model.backbone.blocks),
        4,
        model.backbone.embed_dim,
    )


@pytest.mark.parametrize("prompt_mode", ["shallow", "deep"])
def test_zero_prompt_length(prompt_mode):
    model = ViTB16VPT(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        prompt_length=0,
        prompt_mode=prompt_mode,
    )
    model.eval()

    batch_size = 2
    x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)

    with torch.no_grad():
        features = model.forward_features(x)

    assert features["cls"].shape == (batch_size, model.backbone.embed_dim)
    assert features["prompt_tokens"].shape == (batch_size, 0, model.backbone.embed_dim)
    assert features["patch_tokens"].shape == (
        batch_size,
        model.backbone.num_patches,
        model.backbone.embed_dim,
    )


@pytest.mark.parametrize("prompt_mode", ["shallow", "deep"])
def test_prompt_receives_gradients(prompt_mode):
    model = ViTB16VPT(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        prompt_length=4,
        prompt_mode=prompt_mode,
    )
    model.train()

    batch_size = 2
    x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)
    target = torch.randint(0, N_CLASSES, (batch_size,))

    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()

    assert model.prompt.grad is not None
    assert model.prompt.grad.shape == model.prompt.shape


def test_invalid_prompt_mode_raises():
    model = ViTB16VPT(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        prompt_length=4,
        prompt_mode="invalid",
    )
    model.eval()

    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)

    with pytest.raises(ValueError, match="Invalid prompt_mode"):
        model.forward_features(x)


@pytest.mark.parametrize("prompt_mode", ["shallow", "deep"])
def test_compute_loss(prompt_mode):
    model = ViTB16VPT(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        prompt_length=4,
        prompt_mode=prompt_mode,
    )
    model.train()

    batch_size = 4
    x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)
    y = torch.randint(0, N_CLASSES, (batch_size,))

    logits, loss, loss_components = model.compute_loss(x, y)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (batch_size, N_CLASSES)
    assert logits.grad_fn is not None

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()
    assert loss.grad_fn is not None

    assert isinstance(loss_components, dict)
    assert "cross_entropy_loss" in loss_components

    model.eval()
    with torch.no_grad():
        logits, loss, loss_components = model.compute_loss(x, y)
        expected_loss = F.cross_entropy(logits, y).item()

    assert pytest.approx(loss_components["cross_entropy_loss"]) == loss.item()
    assert pytest.approx(loss_components["cross_entropy_loss"]) == expected_loss


@pytest.mark.parametrize("prompt_mode", ["shallow", "deep"])
def test_save_and_load_trainable_parameters(tmp_path, prompt_mode):
    source_model = ViTB16VPT(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        prompt_length=4,
        prompt_mode=prompt_mode,
    )

    target_model = ViTB16VPT(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        prompt_length=4,
        prompt_mode=prompt_mode,
    )

    # Force known values so the test is deterministic.
    with torch.no_grad():
        source_model.prompt.fill_(1.23)
        source_model.head.weight.fill_(4.56)
        source_model.head.bias.fill_(7.89)

        target_model.prompt.fill_(0.0)
        target_model.head.weight.fill_(0.0)
        target_model.head.bias.fill_(0.0)

    checkpoint_path = tmp_path / "vpt_trainable.pt"

    source_model.save(checkpoint_path)

    saved_state_dict = torch.load(checkpoint_path, map_location="cpu")

    assert set(saved_state_dict.keys()) == {
        "prompt",
        "head.weight",
        "head.bias",
    }

    target_model.load_from_path(checkpoint_path)

    assert torch.allclose(target_model.prompt, source_model.prompt)
    assert torch.allclose(target_model.head.weight, source_model.head.weight)
    assert torch.allclose(target_model.head.bias, source_model.head.bias)
