from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple


ModelBuilder = Callable[..., Any]


class ModelRegistry:
    """
    Minimal model registry to decouple scripts from concrete model classes.

    Goal:
    - training/inference scripts choose `model.name` in config
    - swapping architectures doesn't require editing scripts, only adding a builder registration
    """

    def __init__(self):
        self._builders: Dict[str, ModelBuilder] = {}

    def register(self, name: str, builder: ModelBuilder) -> None:
        if name in self._builders:
            raise ValueError(f"Model '{name}' is already registered.")
        self._builders[name] = builder

    def build(self, name: str, **kwargs) -> Any:
        if name not in self._builders:
            known = ", ".join(sorted(self._builders.keys()))
            raise KeyError(f"Unknown model '{name}'. Known: {known}")
        return self._builders[name](**kwargs)


GLOBAL_MODEL_REGISTRY = ModelRegistry()


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    def _decorator(fn: ModelBuilder) -> ModelBuilder:
        GLOBAL_MODEL_REGISTRY.register(name, fn)
        return fn
    return _decorator


def build_model(name: str, **kwargs) -> Any:
    return GLOBAL_MODEL_REGISTRY.build(name, **kwargs)


