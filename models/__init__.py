from .vit_b16 import ViTB16VPT, ViTB16AdaptFormer, ViTB16LoRA, ViTB16SSF

models = {
    "vit_b16": {
        "vpt": ViTB16VPT,
        "adaptformer": ViTB16AdaptFormer,
        "lora": ViTB16LoRA,
        "ssf": ViTB16SSF,
    },
}
