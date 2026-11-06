from .vit_b16 import ViTB16VPT, ViTB16AdaptFormer, ViTB16LoRA, ViTB16SSF, ViTB16Backbone
from .peft_model import PEFTModel
from .transformer_backbone import TransformerBackbone

models = {
    "vit_b16": {
        "backbone": ViTB16Backbone,
        "vpt": ViTB16VPT,
        "adaptformer": ViTB16AdaptFormer,
        "lora": ViTB16LoRA,
        "ssf": ViTB16SSF,
    },
}
