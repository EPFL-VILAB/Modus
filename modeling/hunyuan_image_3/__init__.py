"""
HunyuanImage-3.0 integration package for MODUS.

Importing this package registers the "hunyuan_image_3" builder in the
global ModelRegistry.
"""

from .vae_adapter import HunyuanVAEAdapter, HunyuanVAEConfig
from .model_wrapper import HunyuanImageWrapper
from . import builder as _builder  # noqa: F401  triggers register_model("hunyuan_image_3")

__all__ = [
    "HunyuanVAEAdapter",
    "HunyuanVAEConfig",
    "HunyuanImageWrapper",
]
