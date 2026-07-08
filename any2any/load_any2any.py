"""
Public-release friendly loading utilities for any2any demos.

Goal: keep demo scripts thin and avoid duplicating model/tokenizer/modality wiring.

All per-modality settings (pos_embed, loss config, conditions) live in the
modality YAML and are applied automatically via set_modality_registry().
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Optional, Tuple

import yaml
import torch

from core.modality import ModalityRegistry
from core.tokenizer_utils import build_tokenizer_and_special_tokens
from core.model_registry import build_model
import modeling  # register model builders

from modeling.bagel import Bagel
from modeling.qwen2 import Qwen2Tokenizer

from safetensors.torch import load_file
from train.merge_safetensors_stage2 import merge_family


def _infer_target_device(*, init_on_gpu: bool) -> str:
    """Single source of truth for demo loader device placement."""
    if not init_on_gpu:
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
    except Exception:
        pass
    return "cpu"


def _move_and_maybe_bf16(obj, *, device: str):
    """
    Minimal helper to keep demo inference consistent:
    - always move to `device`
    - on CUDA, cast to bfloat16 (matches inferencer autocast and avoids bf16-vs-fp32 conv bias errors)
    """
    try:
        import torch

        has_meta = False
        if hasattr(obj, "parameters"):
            try:
                has_meta = any(getattr(p, "is_meta", False) for p in obj.parameters())
            except Exception:
                has_meta = False

        if has_meta and hasattr(obj, "to_empty"):
            obj = obj.to_empty(device=device)

        if hasattr(obj, "to"):
            if str(device).startswith("cuda"):
                return obj.to(device=device, dtype=torch.bfloat16)
            return obj.to(device=device)
    except Exception:
        pass
    return obj


def _resize_demo_token_embeddings(model, tokenizer_len: int, *, model_name: str):
    mean_resizing = True
    # When loading a trained checkpoint the resized rows are overwritten by the
    # checkpoint weights, so mean_resizing's covariance init is wasted work — and
    # it can deadlock (BLAS/OMP) when gradio/numpy is imported first. Allow turning
    # it off for inference/demo via env.
    if os.environ.get("MODUS_NO_MEAN_RESIZING", "0") == "1":
        mean_resizing = False
    model.language_model.resize_token_embeddings(tokenizer_len, mean_resizing=mean_resizing)
    model.config.llm_config.vocab_size = tokenizer_len
    model.language_model.config.vocab_size = tokenizer_len


def _build_tokenizer_and_registry(
    model_path: str,
    modality_config_path: Optional[str],
    model_name: str = "bagel_from_json",
):
    """Shared helper: tokenizer + modality registry construction."""
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    orig_vocab_size = len(tokenizer)

    if modality_config_path is None:
        modality_config_path = "conf/modalities/instruction_16mod_stage2.yaml"
    with open(modality_config_path, "r") as f:
        modality_cfg = yaml.safe_load(f)

    tok_artifacts = build_tokenizer_and_special_tokens(tokenizer, modalities_cfg=modality_cfg)
    tokenizer = tok_artifacts.tokenizer
    new_token_ids = tok_artifacts.new_token_ids
    num_new_tokens = len(tokenizer) - orig_vocab_size

    modality_registry = ModalityRegistry.from_config(
        modality_cfg,
        token_ranges=tok_artifacts.token_ranges,
        code_token_ids=tok_artifacts.code_token_ids,
    )
    return tokenizer, new_token_ids, num_new_tokens, modality_registry


def _dict_to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_dict_to_namespace(v) for v in value]
    return value


def _maybe_merge_sharded_checkpoint(checkpoint_path: str, ckpt_file: str) -> str:
    merged_path = os.path.join(checkpoint_path, ckpt_file)
    if os.path.exists(merged_path):
        return merged_path

    family = ckpt_file.replace(".safetensors", "")
    shard_prefix = f"{family}."
    has_shards = any(
        name.startswith(shard_prefix) and name.endswith(".safetensors") and "-of-" in name
        for name in os.listdir(checkpoint_path)
    )
    if has_shards:
        merge_family(
            checkpoint_dir=checkpoint_path,
            model_name=family,
            output_name=ckpt_file,
            skip_pos_embed=False,
        )
    return merged_path


def load_any2any_model_hf(
    *,
    model_path: str,
    model_name: str = "bagel_from_json",
    init_on_gpu: bool = True,
    ckpt_file: str = "ema.safetensors",
    modality_config_path: Optional[str] = None,
    **_legacy_kwargs,
) -> Tuple[Bagel, object, object, dict, ModalityRegistry]:
    """
    Load a model from a HF-style folder (configs + tokenizer + weight safetensor).

    Returns: (model, vae_model, tokenizer, new_token_ids, modality_registry)
    """
    model, vae_model, vae_config, vit_config = build_model(
        model_name,
        model_path=model_path,
        init_device=("cpu" if init_on_gpu else "meta"),
        init_on_gpu=init_on_gpu,
    )
    target_device = _infer_target_device(init_on_gpu=init_on_gpu)

    tokenizer, new_token_ids, num_new_tokens, modality_registry = (
        _build_tokenizer_and_registry(model_path, modality_config_path, model_name=model_name)
    )

    # Move off meta before state_dict load (best effort)
    if hasattr(model, "to_empty"):
        try:
            model = model.to_empty(device=target_device)
        except Exception:
            pass

    model = _move_and_maybe_bf16(model, device=target_device)
    vae_model = _move_and_maybe_bf16(vae_model, device=target_device)

    if num_new_tokens > 0:
        _resize_demo_token_embeddings(model, len(tokenizer), model_name=model_name)

    # Attach registry before loading weights so modality-specific pos-embeds
    # exist and can receive checkpoint values when present.
    if hasattr(model, "set_modality_registry"):
        model.set_modality_registry(modality_registry)
    else:
        model.modality_registry = modality_registry

    state = load_file(os.path.join(model_path, ckpt_file), device="cpu")
    model.load_state_dict(state, strict=False, assign=False)
    del state

    try:
        model.eval()
    except Exception:
        pass
    try:
        if hasattr(vae_model, "eval"):
            vae_model.eval()
    except Exception:
        pass

    return model, vae_model, tokenizer, new_token_ids, modality_registry


def load_any2any_model_training_checkpoint(
    *,
    checkpoint_path: str,
    model_path: str,
    model_name: str = "bagel_from_json",
    init_on_gpu: bool = True,
    use_ema: bool = False,
    modality_config_path: Optional[str] = None,
    train_config_path: Optional[str] = None,
    **_legacy_kwargs,
) -> Tuple[Bagel, object, object, dict, ModalityRegistry]:
    """
    Load Bagel from a training checkpoint directory (model.safetensors / ema.safetensors).

    Returns: (model, vae_model, tokenizer, new_token_ids, modality_registry)
    """
    model, vae_model, vae_config, vit_config = build_model(
        model_name,
        model_path=model_path,
        init_device=("cpu" if init_on_gpu else "meta"),
        init_on_gpu=init_on_gpu,
    )
    target_device = _infer_target_device(init_on_gpu=init_on_gpu)
    print(f"[any2any-load] target_device={target_device}")

    tokenizer, new_token_ids, num_new_tokens, modality_registry = (
        _build_tokenizer_and_registry(model_path, modality_config_path, model_name=model_name)
    )
    print(f"[any2any-load] tokenizer ready, num_new_tokens={num_new_tokens}")

    model = _move_and_maybe_bf16(model, device=target_device)
    vae_model = _move_and_maybe_bf16(vae_model, device=target_device)
    print("[any2any-load] model moved to target device")

    if num_new_tokens > 0:
        _resize_demo_token_embeddings(model, len(tokenizer), model_name=model_name)
        print(f"[any2any-load] resized token embeddings to {len(tokenizer)}")

    # Attach registry BEFORE loading weights so pos-embed modules exist.
    if hasattr(model, "set_modality_registry"):
        model.set_modality_registry(modality_registry)
    else:
        model.modality_registry = modality_registry

    ckpt_file = "ema.safetensors" if use_ema else "model.safetensors"
    ckpt_path = _maybe_merge_sharded_checkpoint(checkpoint_path, ckpt_file)
    state = load_file(ckpt_path, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False, assign=False)
    print(f"Missing keys: {missing}\nUnexpected keys: {unexpected}")
    del state

    try:
        model.eval()
    except Exception:
        pass
    try:
        if hasattr(vae_model, "eval"):
            vae_model.eval()
    except Exception:
        pass

    return model, vae_model, tokenizer, new_token_ids, modality_registry
