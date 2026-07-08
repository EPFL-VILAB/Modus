# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# NOTE: do NOT eagerly `from . import bagel, qwen2, siglip, autoencoder` here.
# Several bagel submodules do absolute `from modeling.<sibling> import ...`, which
# re-enters this partially-initialized package and raises a circular-import error.
# Submodules are imported on demand (e.g. inferencer does
# `from modeling.bagel.qwen2_navit import NaiveCache`); the builder registration
# below is enough for the model registry.

# Ensure model builders are registered on import.
from .bagel import builder as _bagel_builder  # noqa: F401