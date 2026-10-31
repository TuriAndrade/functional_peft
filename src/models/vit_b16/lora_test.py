import torch
import pytest
import torch.nn.functional as F
from .lora import ViTB16LoRA

IMG_SIZE = 224
N_CLASSES = 10
RANK = 8
LORA_TARGETS = ("query", "value")


def test_forward_features_shapes():
    model = ViTB16LoRA(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        rank=RANK,
        lora_targets=LORA_TARGETS,
    )
    model.eval()
    batch_size = 2
    x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        features = model.forward_features(x)
    expected_keys = {"cls", "patch_tokens"}
    assert set(features.keys()) == expected_keys
    assert features["cls"].shape == (batch_size, model.backbone.embed_dim)
    assert features["patch_tokens"].shape == (
        batch_size,
        model.backbone.num_patches,
        model.backbone.embed_dim,
    )


def test_forward_output_shape():
    model = ViTB16LoRA(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        rank=RANK,
        lora_targets=LORA_TARGETS,
    )
    model.eval()
    batch_size = 2
    x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (batch_size, N_CLASSES)


def test_lora_parameter_shapes():
    model = ViTB16LoRA(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        rank=RANK,
        lora_targets=LORA_TARGETS,
    )
    n_layers = len(model.backbone.blocks)
    n_targets = len(LORA_TARGETS)
    embed_dim = model.backbone.embed_dim
    assert model.lora_A.shape == (
        n_layers,
        n_targets,
        RANK,
        embed_dim,
    )
    assert model.lora_B.shape == (
        n_layers,
        n_targets,
        embed_dim,
        RANK,
    )


def test_lora_initialization():
    model = ViTB16LoRA(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        rank=RANK,
        lora_targets=LORA_TARGETS,
    )
    assert torch.count_nonzero(model.lora_A) > 0
    assert torch.count_nonzero(model.lora_B) == 0


def test_zero_rank():
    model = ViTB16LoRA(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        rank=0,
    )
    model.eval()
    assert model.lora_A is None
    assert model.lora_B is None
    batch_size = 2
    x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        features = model.forward_features(x)
    assert features["cls"].shape == (batch_size, model.backbone.embed_dim)
    assert features["patch_tokens"].shape == (
        batch_size,
        model.backbone.num_patches,
        model.backbone.embed_dim,
    )


def test_lora_receives_gradients():
    model = ViTB16LoRA(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        rank=RANK,
        lora_targets=LORA_TARGETS,
    )
    model.train()
    batch_size = 2
    x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)
    target = torch.randint(0, N_CLASSES, (batch_size,))
    logits = model(x)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    assert model.lora_A.grad is not None
    assert model.lora_A.grad.shape == model.lora_A.shape
    assert model.lora_B.grad is not None
    assert model.lora_B.grad.shape == model.lora_B.shape


def test_frozen_backbone_receives_no_gradients():
    model = ViTB16LoRA(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        rank=RANK,
        lora_targets=LORA_TARGETS,
        freeze_backbone=True,
    )
    model.train()
    batch_size = 2
    x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)
    target = torch.randint(0, N_CLASSES, (batch_size,))
    logits = model(x)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(not p.requires_grad for p in model.backbone.parameters())


def test_train_keeps_frozen_backbone_in_eval_mode():
    model = ViTB16LoRA(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        rank=RANK,
        lora_targets=LORA_TARGETS,
        freeze_backbone=True,
    )
    model.train()
    assert model.training
    assert model.head.training
    assert not model.backbone.training
    assert model.is_backbone_frozen
    assert all(not p.requires_grad for p in model.backbone.parameters())


def test_unfreeze_backbone_restores_train_mode():
    model = ViTB16LoRA(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        rank=RANK,
        lora_targets=LORA_TARGETS,
        freeze_backbone=True,
    )
    model.train()
    model.unfreeze_backbone()
    assert model.training
    assert model.backbone.training
    assert not model.is_backbone_frozen
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_freeze_backbone_sets_eval_mode_and_disables_gradients():
    model = ViTB16LoRA(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        rank=RANK,
        lora_targets=LORA_TARGETS,
        freeze_backbone=False,
    )
    model.train()
    model.freeze_backbone()
    assert model.training
    assert not model.backbone.training
    assert model.is_backbone_frozen
    assert all(not p.requires_grad for p in model.backbone.parameters())


def test_invalid_lora_target_raises():
    with pytest.raises(ValueError, match="lora_targets may contain only"):
        ViTB16LoRA(
            img_size=IMG_SIZE,
            n_classes=N_CLASSES,
            rank=RANK,
            lora_targets=("key",),
        )


def test_compute_loss():
    model = ViTB16LoRA(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        rank=RANK,
        lora_targets=LORA_TARGETS,
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


def test_save_and_load_trainable_parameters(tmp_path):
    source_model = ViTB16LoRA(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        rank=RANK,
        lora_targets=LORA_TARGETS,
    )
    target_model = ViTB16LoRA(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        rank=RANK,
        lora_targets=LORA_TARGETS,
    )
    # Force known values so the test is deterministic.
    with torch.no_grad():
        source_model.lora_A.fill_(1.23)
        source_model.lora_B.fill_(2.34)
        source_model.head.weight.fill_(4.56)
        source_model.head.bias.fill_(7.89)
        target_model.lora_A.fill_(0.0)
        target_model.lora_B.fill_(0.0)
        target_model.head.weight.fill_(0.0)
        target_model.head.bias.fill_(0.0)
    checkpoint_path = tmp_path / "lora_trainable.pt"
    source_model.save(checkpoint_path)
    saved_state_dict = torch.load(checkpoint_path, map_location="cpu")
    assert set(saved_state_dict.keys()) == {
        "lora_A",
        "lora_B",
        "head.weight",
        "head.bias",
    }
    target_model.load_from_path(checkpoint_path)
    assert torch.allclose(target_model.lora_A, source_model.lora_A)
    assert torch.allclose(target_model.lora_B, source_model.lora_B)
    assert torch.allclose(target_model.head.weight, source_model.head.weight)
    assert torch.allclose(target_model.head.bias, source_model.head.bias)
