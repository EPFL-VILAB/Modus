# Copyright 2024 The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)


_import_structure = {
    "configuration_siglip": [
        "SiglipConfig",
        "SiglipTextConfig",
        "SiglipVisionConfig",
    ],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_siglip"] = [
        "SiglipModel",
        "SiglipPreTrainedModel",
        "SiglipTextModel",
        "SiglipVisionModel",
        "SiglipForImageClassification",
    ]


if TYPE_CHECKING:
    from .configuration_siglip import (
        SiglipConfig,
        SiglipTextConfig,
        SiglipVisionConfig,
    )

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_siglip import (
            SiglipForImageClassification,
            SiglipModel,
            SiglipPreTrainedModel,
            SiglipTextModel,
            SiglipVisionModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
