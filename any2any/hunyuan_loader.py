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
from modeling.hunyuan_image_3.model_wrapper import HunyuanImageWrapper

from safetensors import safe_open
from safetensors.torch import load_file
from .hunyuan_checkpoint import checkpoint_tp_suffixes, combine_tp_checkpoint_parts


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
    mean_resizing = False if model_name == "hunyuan_image_3" else True
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
    if model_name == "hunyuan_image_3":
        from modeling.hunyuan_image_3.src.configuration_hunyuan_image_3 import HunyuanImage3Config
        from modeling.hunyuan_image_3.src.tokenization_hunyuan_image_3 import HunyuanImage3TokenizerFast

        cfg = HunyuanImage3Config.from_pretrained(model_path)
        tokenizer = HunyuanImage3TokenizerFast.from_pretrained(
            model_path,
            model_version=cfg.model_version,
        )
    else:
        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    orig_vocab_size = len(tokenizer)

    if modality_config_path is None:
        modality_config_path = "conf/modalities/instruction.yaml"
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
        # Training-checkpoint support is diagnostic-only.  Keep it out of the
        # normal self-contained release loader's import graph.
        from train.merge_safetensors_stage2 import merge_family

        merge_family(
            checkpoint_dir=checkpoint_path,
            model_name=family,
            output_name=ckpt_file,
            skip_pos_embed=False,
        )
    return merged_path


def _build_janus_model_from_train_config(
    *,
    model_path: str,
    train_config_path: str,
    init_on_gpu: bool,
):
    with open(train_config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_args = _dict_to_namespace(cfg["model"])
    training_args = _dict_to_namespace(cfg["training"])

    if not hasattr(training_args, "freeze_und"):
        training_args.freeze_und = False

    model, vae_model, vae_config, vit_config = build_model(
        "janus_flow",
        model_args=model_args,
        training_args=training_args,
        init_device=("cpu" if init_on_gpu else "meta"),
    )
    return model, vae_model, vae_config, vit_config


def _build_hunyuan_model_from_train_config(
    *,
    model_path: str,
    train_config_path: str,
    init_on_gpu: bool,
):
    with open(train_config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_args = _dict_to_namespace(cfg["model"])
    training_args = _dict_to_namespace(cfg["training"])
    model_args.model_path = model_path

    model, vae_model, vae_config, vit_config = build_model(
        "hunyuan_image_3",
        model_args=model_args,
        training_args=training_args,
        init_device=("cpu" if init_on_gpu else "meta"),
    )
    return model, vae_model, vae_config, vit_config


def _build_hunyuan_model_distributed_for_inference(
    *,
    model_path: str,
    train_config_path: str,
):
    """
    Load HunyuanImage-3 for inference using HF device_map="auto" so the 77B
    model is spread across the available GPUs instead of materializing on one.
    """
    import json

    from modeling.hunyuan_image_3.builder import _load_vae_weights
    from modeling.hunyuan_image_3.src.autoencoder_kl_3d import AutoencoderKLConv3D
    from modeling.hunyuan_image_3.src.modeling_hunyuan_image_3 import HunyuanImage3ForCausalMM
    from modeling.hunyuan_image_3.vae_adapter import HunyuanVAEAdapter, HunyuanVAEConfig

    with open(train_config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_args = _dict_to_namespace(cfg["model"])
    # This reads conf/train/hunyuan_image_3.yaml, whose value is 16 — but the
    # 16-mod runs override it to 64 on the command line
    # (train_16mod_stage1_64g_*.sh:51), so inference caps the ViT grid at
    # 16x16=256 patches while training fed up to 64x64. Measured: 192 ViT tokens
    # per condition image at inference vs 1024 in training.
    vit_max_patch = getattr(model_args, "vit_max_num_patch_per_side", 70)
    _vit_override = os.environ.get("HUNYUAN_INFER_VIT_MAX_PATCH")
    if _vit_override:
        vit_max_patch = int(_vit_override)
        print(f"[hunyuan-load] vit_max_num_patch_per_side overridden to "
              f"{vit_max_patch} (config said {getattr(model_args, 'vit_max_num_patch_per_side', None)})",
              flush=True)

    device_map = os.environ.get("HUNYUAN_DEVICE_MAP", "auto")
    print(f"[hunyuan-load] from_pretrained device_map={device_map}")
    hunyuan_model = HunyuanImage3ForCausalMM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    hunyuan_model.eval()

    cfg_path = os.path.join(model_path, "config.json")
    with open(cfg_path, "r") as f:
        full_cfg = json.load(f)
    vae_cfg_dict = full_cfg.get("vae", {})
    vae_cfg_dict = {k: v for k, v in vae_cfg_dict.items() if not k.startswith("_")}

    vae = AutoencoderKLConv3D(**vae_cfg_dict)
    vae = _load_vae_weights(vae, model_path)
    vae = vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    vae_model = HunyuanVAEAdapter(
        vae,
        scale_factor=vae_cfg_dict.get("scaling_factor", 0.562679178327931),
    )
    vae_model.source_model_path = model_path
    vae_config = HunyuanVAEConfig()

    wrapper = HunyuanImageWrapper(hunyuan_model, vit_max_num_patch_per_side=vit_max_patch)
    wrapper.eval()
    return wrapper, vae_model, vae_config, None


def load_hunyuan_native_model(
    *,
    model_path: str,
    checkpoint_path: Optional[str] = None,
    use_ema: bool = False,
    tp_size: Optional[int] = None,
):
    """
    Load HunyuanImage-3 in its native HF inference form.

    Unlike load_any2any_model_* this does not build MODUS modality tokens,
    does not resize embeddings, and does not wrap generation in the BAGEL-style
    any2any path. It is intended for raw/native-compatible T2I/TI2I inference.
    If checkpoint_path is provided, the Hunyuan train checkpoint is overlaid
    onto the native model by matching the original Hunyuan parameter names.
    """
    from modeling.hunyuan_image_3.src.modeling_hunyuan_image_3 import HunyuanImage3ForCausalMM

    device_map = os.environ.get("HUNYUAN_DEVICE_MAP", "auto")
    moe_impl = os.environ.get("HUNYUAN_NATIVE_MOE_IMPL", "eager")
    # Match the loader used to validate the raw training checkpoint.  Passing
    # an attention implementation here changes Transformers' dispatch and can
    # introduce avoidable numerical drift.  Keep an explicit override for
    # diagnostics, but otherwise let the model config choose its native path.
    attn_impl = os.environ.get("HUNYUAN_NATIVE_ATTN_IMPL")
    print(
        f"[hunyuan-native-load] from_pretrained device_map={device_map} "
        f"attn={attn_impl} moe={moe_impl}"
    )
    model = HunyuanImage3ForCausalMM.from_pretrained(
        model_path,
        attn_implementation=attn_impl,
        torch_dtype="auto",
        device_map=device_map,
        moe_impl=moe_impl,
        moe_drop_tokens=True,
    )
    model.load_tokenizer(model_path)
    model.eval()

    if checkpoint_path:
        _overlay_hunyuan_checkpoint_online(
            model=model,
            checkpoint_path=checkpoint_path,
            use_ema=use_ema,
            tp_size=tp_size or int(os.environ.get("HUNYUAN_TP_SIZE", "4")),
        )
        model.eval()

    return model


def _overlay_hunyuan_checkpoint_online(
    *,
    model,
    checkpoint_path: str,
    use_ema: bool,
    tp_size: int,
):
    # The public release checkpoint never enters this path.  Lazy importing
    # avoids making its training merge utilities a runtime dependency.
    from train.merge_safetensors_stage2 import list_shard_files, merge_tensors_for_key

    family = "ema" if use_ema else "model"
    prefix = "hunyuan_model."
    shard_files = list_shard_files(checkpoint_path, family)
    tp_col_suffixes, tp_row_suffixes, checkpoint_metadata = checkpoint_tp_suffixes(checkpoint_path)
    print(f"[hunyuan-load] online overlay start: family={family} checkpoint={checkpoint_path}")
    print(f"[hunyuan-load] found {len(shard_files)} shards, tp_size={tp_size}")
    if checkpoint_metadata:
        saved_tp = int(checkpoint_metadata.get("tp_size", tp_size))
        if saved_tp != tp_size:
            raise ValueError(f"Checkpoint tp_size={saved_tp}, loader tp_size={tp_size}")
        print(f"[hunyuan-load] checkpoint topology={checkpoint_metadata}")
    replicated_mode = os.environ.get("HUNYUAN_REPLICATED_KEY_TP_MODE", "tp0").strip().lower()
    replicated_rank = int(os.environ.get("HUNYUAN_REPLICATED_KEY_TP_RANK", "0"))
    replicated_mean_contains = tuple(
        item.strip()
        for item in os.environ.get("HUNYUAN_REPLICATED_KEY_TP_MEAN_CONTAINS", "").split(",")
        if item.strip()
    )
    if replicated_mode not in {"tp0", "rank", "mean"}:
        raise ValueError(
            "HUNYUAN_REPLICATED_KEY_TP_MODE must be one of: tp0, rank, mean "
            f"(got {replicated_mode!r})"
        )
    if replicated_rank < 0 or replicated_rank >= tp_size:
        raise ValueError(f"HUNYUAN_REPLICATED_KEY_TP_RANK={replicated_rank} outside [0, {tp_size})")
    print(
        f"[hunyuan-load] replicated-key TP merge mode={replicated_mode} "
        f"rank={replicated_rank}"
    )
    if replicated_mode == "mean" and replicated_mean_contains:
        print(
            "[hunyuan-load] replicated-key TP mean restricted to keys containing: "
            f"{replicated_mean_contains}"
        )
    # How to combine a buffer that every DP rank stores whole (e.g. the
    # loss-free MoE routing bias). "mean" averages the DP copies, "rank0" takes
    # the first, "off" keeps the historical DP-concat behaviour (which makes a
    # (64,) buffer come out (16*64,), fail the shape check and be skipped).
    #
    # Default is "off" = exactly the behaviour every previously-reported number
    # was produced under, so enabling the fix can never silently change a metric.
    # It is NOT simply "the buggy path": on ckpt 0002000 the 64 stored copies of
    # `_lossfree_bias` are bit-identical for layers 0-3/31 (those always loaded)
    # and mutually UNCORRELATED across the 16 DP groups for layers 4-30 (mean
    # pairwise cosine ~0, top-8 expert overlap at chance) — the DP ranks never
    # synchronised there, so no single "true" bias exists for those layers.
    # Measured on dino/8 images: off=+0.0527 vs mean=+0.0014, i.e. averaging 16
    # uncorrelated vectors yields a direction no rank ever used and is worse than
    # zero. Do not change this default without re-running the mode comparison
    # (scripts/hunyuan_image_3/diag_lossfree_modes.py).
    replicated_dp_mode = os.environ.get("HUNYUAN_REPLICATED_DP_MODE", "off").strip().lower()
    if replicated_dp_mode not in {"mean", "rank0", "off"}:
        raise ValueError(
            "HUNYUAN_REPLICATED_DP_MODE must be one of: mean, rank0, off "
            f"(got {replicated_dp_mode!r})"
        )
    print(f"[hunyuan-load] replicated-DP buffer mode={replicated_dp_mode}")
    if len(shard_files) % tp_size != 0:
        raise ValueError(
            f"Found {len(shard_files)} shards for {family}, not divisible by tp_size={tp_size}"
        )

    shard_handles = [safe_open(path, framework="pt", device="cpu") for path in shard_files]
    shard_key_sets = [set(handle.keys()) for handle in shard_handles]
    ordered_keys = list(shard_handles[0].keys())
    for extra_keys in shard_key_sets[1:]:
        for key in extra_keys:
            if key not in ordered_keys:
                ordered_keys.append(key)

    host_model = getattr(model, "language_model", model)
    model_state = host_model.state_dict()
    loaded_keys = []
    shape_mismatches = []
    unexpected = []
    replicated_dp_keys = []

    # heterogeneous_local_orig_v1 is not a conventional list of uniform FSDP
    # shards.  Dense parameters are split over the full DPxEP group, while each
    # packed expert bank is split over the DP ranks of one EP owner.  Treating
    # the 32 files as ordinary DP shards both leaves the packed experts unloaded
    # and produces flat vectors for parameters whose local original-parameter
    # slice crossed an FSDP flat-buffer boundary.  Reconstruct this exact layout
    # explicitly before overlaying it on the native (unpacked) inference model.
    if checkpoint_metadata and checkpoint_metadata.get("checkpoint_layout") == "heterogeneous_local_orig_v1":
        if tp_size != 1:
            raise ValueError(
                "heterogeneous_local_orig_v1 inference overlay currently requires saved TP=1"
            )
        world_size = int(checkpoint_metadata["world_size"])
        ep_size = int(checkpoint_metadata["ep_size"])
        expert_dp_size = int(checkpoint_metadata["expert_fsdp_size"])
        if len(shard_handles) != world_size or world_size != ep_size * expert_dp_size:
            raise ValueError(
                "Invalid heterogeneous checkpoint topology: "
                f"shards={len(shard_handles)} world={world_size} ep={ep_size} "
                f"expert_dp={expert_dp_size}"
            )

        def _parts_for(full_key, ranks):
            parts = []
            for rank in ranks:
                if full_key not in shard_key_sets[rank]:
                    raise RuntimeError(
                        f"Key {full_key!r} missing from heterogeneous shard rank {rank}"
                    )
                part = shard_handles[rank].get_tensor(full_key)
                if part.numel():
                    parts.append(part.reshape(-1))
            return parts

        def _restore_flat(full_key, target):
            parts = _parts_for(full_key, range(world_size))
            # Buffers are not FSDP-sharded.  At a loss-free resync checkpoint
            # all copies are equal; mean remains deterministic and also makes
            # the intended replicated-buffer policy explicit.
            if len(parts) == world_size and all(p.numel() == target.numel() for p in parts):
                if replicated_dp_mode == "mean":
                    source = torch.stack(
                        [p.reshape(target.shape).float() for p in parts], dim=0
                    ).mean(dim=0)
                else:
                    source = parts[0].reshape(target.shape)
                replicated_dp_keys.append(full_key[len(prefix):])
            else:
                total = sum(p.numel() for p in parts)
                if total != target.numel():
                    raise RuntimeError(
                        f"Heterogeneous checkpoint size mismatch for {full_key}: "
                        f"saved_numel={total} target_numel={target.numel()}"
                    )
                source = torch.cat(parts, dim=0).reshape(target.shape)
            target.copy_(source.to(device=target.device, dtype=target.dtype))

        packed_suffixes = {
            ".packed_experts.weight1": "gate_and_up_proj.weight",
            ".packed_experts.weight2": "down_proj.weight",
        }
        packed_keys = {
            key for key in ordered_keys
            if any(key.endswith(suffix) for suffix in packed_suffixes)
        }

        with torch.no_grad():
            for full_key in ordered_keys:
                if not full_key.startswith(prefix) or full_key in packed_keys:
                    continue
                model_key = full_key[len(prefix):]
                target = model_state.get(model_key)
                if target is None:
                    unexpected.append(model_key)
                    continue
                _restore_flat(full_key, target)
                loaded_keys.append(model_key)

            for full_key in sorted(packed_keys):
                packed_suffix = next(
                    suffix for suffix in packed_suffixes if full_key.endswith(suffix)
                )
                expert_leaf = packed_suffixes[packed_suffix]
                model_prefix = full_key[len(prefix):-len(packed_suffix)]
                # The native model has all global experts; the training model
                # stored 1/EP of them on each EP owner, packed as [E,H,O].
                expert_key_prefix = f"{model_prefix}.experts."
                expert_indices = sorted({
                    int(key[len(expert_key_prefix):].split(".", 1)[0])
                    for key in model_state
                    if key.startswith(expert_key_prefix) and key.endswith(expert_leaf)
                })
                if not expert_indices or expert_indices != list(range(len(expert_indices))):
                    raise RuntimeError(
                        f"Native inference model has a non-contiguous expert layout for "
                        f"{full_key}: {expert_indices[:8]}"
                    )
                target_keys = [
                    f"{model_prefix}.experts.{idx}.{expert_leaf}"
                    for idx in expert_indices
                ]
                num_experts = len(target_keys)
                if num_experts % ep_size:
                    raise RuntimeError(
                        f"Expert count {num_experts} is not divisible by EP={ep_size}"
                    )
                local_experts = num_experts // ep_size
                sample_target = model_state[target_keys[0]]
                packed_shape = (
                    local_experts,
                    sample_target.shape[1],
                    sample_target.shape[0],
                )
                expected_numel = int(torch.tensor(packed_shape).prod().item())

                for ep_rank in range(ep_size):
                    ranks = [dp_rank * ep_size + ep_rank for dp_rank in range(expert_dp_size)]
                    parts = _parts_for(full_key, ranks)
                    total = sum(part.numel() for part in parts)
                    if total != expected_numel:
                        raise RuntimeError(
                            f"Packed expert size mismatch for {full_key} EP{ep_rank}: "
                            f"saved_numel={total} expected_numel={expected_numel}"
                        )
                    packed = torch.cat(parts, dim=0).reshape(packed_shape)
                    for local_idx in range(local_experts):
                        global_idx = ep_rank * local_experts + local_idx
                        target_key = target_keys[global_idx]
                        target = model_state[target_key]
                        source = packed[local_idx].transpose(0, 1)
                        if tuple(source.shape) != tuple(target.shape):
                            raise RuntimeError(
                                f"Unpacked expert shape mismatch for {target_key}: "
                                f"saved={tuple(source.shape)} target={tuple(target.shape)}"
                            )
                        target.copy_(source.to(device=target.device, dtype=target.dtype))
                        loaded_keys.append(target_key)
                    del packed

        if unexpected:
            raise RuntimeError(
                "Heterogeneous Hunyuan overlay found checkpoint keys with no native "
                f"inference target (first 20): {unexpected[:20]}"
            )
        missing_model_keys = sorted(set(model_state) - set(loaded_keys))
        externally_managed_missing = [
            key for key in missing_model_keys if key.startswith("vae.")
        ]
        checkpoint_managed_missing = [
            key for key in missing_model_keys if not key.startswith("vae.")
        ]
        if checkpoint_managed_missing:
            raise RuntimeError(
                "Heterogeneous Hunyuan overlay was incomplete; unloaded model keys "
                f"(first 20): {checkpoint_managed_missing[:20]}"
            )
        print(
            "[hunyuan-load] heterogeneous overlay completeness: all "
            f"{len(loaded_keys)} checkpoint-managed model keys loaded"
        )
        if externally_managed_missing:
            print(
                f"[hunyuan-load] overlay completeness: ignored "
                f"{len(externally_managed_missing)} externally managed frozen VAE keys"
            )
        print(
            f"Hunyuan heterogeneous online overlay loaded {len(loaded_keys)} keys "
            f"from {checkpoint_path}"
        )
        return

    def _merge_dp_shards(full_key: str, handles, key_sets):
        parts = []
        for handle, key_set in zip(handles, key_sets):
            if full_key not in key_set:
                raise RuntimeError(f"Key '{full_key}' missing while merging {family} shards")
            parts.append(handle.get_tensor(full_key))
        return merge_tensors_for_key(full_key, parts)

    def _shard_shape(handle, full_key: str):
        """Stored shape in one shard, read from the header (no tensor load)."""
        try:
            return tuple(handle.get_slice(full_key).get_shape())
        except Exception:
            return tuple(handle.get_tensor(full_key).shape)

    def _is_replicated(full_key: str, target_shape) -> bool:
        """True only if EVERY shard holding the key stores it at full size.

        FSDP flat-sharding can give shard 0 the whole tensor and the other ranks
        an empty slice (e.g. [1,1,1152] on rank 0, [0,1,1152] elsewhere); those
        must still be concatenated, not treated as replicas. Checking shard 0
        alone would misclassify them.
        """
        seen = False
        for handle, key_set in zip(shard_handles, shard_key_sets):
            if full_key not in key_set:
                continue
            if _shard_shape(handle, full_key) != target_shape:
                return False
            seen = True
        return seen

    def _replicated_dp_tensor(full_key: str):
        """Combine copies of a buffer that every rank stores WHOLE."""
        parts = [h.get_tensor(full_key)
                 for h, ks in zip(shard_handles, shard_key_sets) if full_key in ks]
        if replicated_dp_mode == "mean":
            return torch.stack([p.float() for p in parts], dim=0).mean(dim=0)
        return parts[0]

    def _combine_tp_parts(model_key: str, tp_parts):
        return combine_tp_checkpoint_parts(
            model_key, tp_parts, tp_col_suffixes, tp_row_suffixes
        )

    def _copy_with_vocab_overlap(model_key: str, target: torch.Tensor, source: torch.Tensor) -> bool:
        if model_key not in {"model.wte.weight", "lm_head.weight"}:
            return False
        if target.ndim != source.ndim:
            return False
        if target.ndim == 2 and target.shape[1] != source.shape[1]:
            return False
        rows_to_copy = min(target.shape[0], source.shape[0])
        target[:rows_to_copy].copy_(source[:rows_to_copy].to(device=target.device, dtype=target.dtype))
        return True

    with torch.no_grad():
        for key_idx, full_key in enumerate(ordered_keys, start=1):
            if not full_key.startswith(prefix):
                continue
            model_key = full_key[len(prefix):]
            needs_tp_merge = model_key.endswith(tp_row_suffixes) or model_key.endswith(tp_col_suffixes)

            target = model_state.get(model_key)
            if target is None:
                unexpected.append(model_key)
                continue

            # Buffers that EVERY rank stores whole (notably the loss-free MoE
            # routing bias `_lossfree_bias`, shape (num_experts,)) must not be
            # DP-merged. merge_tensors_for_key() only recognises them as
            # replicated when every copy is bit-identical; when the ranks
            # disagree even slightly it falls through to a concat whose
            # round-trip check is vacuous (splitting a concatenation always
            # reproduces the parts), yielding (16*64,) for a (64,) buffer. That
            # fails the shape check below and is silently skipped, so inference
            # runs with a ZERO bias and routes tokens differently than training
            # did. A replicated buffer is stored at FULL size in every shard,
            # whereas a DP-sharded tensor is smaller in at least one shard, so
            # the stored shapes identify these unambiguously.
            if (replicated_dp_mode != "off" and not needs_tp_merge
                    and _is_replicated(full_key, tuple(target.shape))):
                merged_tensor = _replicated_dp_tensor(full_key)
                replicated_dp_keys.append(model_key)
                target.copy_(merged_tensor.to(device=target.device, dtype=target.dtype))
                loaded_keys.append(model_key)
                continue

            tp_ranks = range(tp_size) if needs_tp_merge else range(1)
            tp_parts = []
            for tp_rank in tp_ranks:
                tp_handles = shard_handles[tp_rank::tp_size]
                tp_key_sets = shard_key_sets[tp_rank::tp_size]
                merged_tp_tensor, _ = _merge_dp_shards(full_key, tp_handles, tp_key_sets)
                tp_parts.append(merged_tp_tensor)
            merged_tensor = _combine_tp_parts(model_key, tp_parts)

            if not needs_tp_merge:
                if replicated_mode == "rank":
                    tp_rank = replicated_rank
                    tp_handles = shard_handles[tp_rank::tp_size]
                    tp_key_sets = shard_key_sets[tp_rank::tp_size]
                    merged_tensor, _ = _merge_dp_shards(full_key, tp_handles, tp_key_sets)
                elif replicated_mode == "mean" and (
                    not replicated_mean_contains
                    or any(part in model_key for part in replicated_mean_contains)
                ):
                    rank_tensors = []
                    for tp_rank in range(tp_size):
                        tp_handles = shard_handles[tp_rank::tp_size]
                        tp_key_sets = shard_key_sets[tp_rank::tp_size]
                        rank_tensor, _ = _merge_dp_shards(full_key, tp_handles, tp_key_sets)
                        rank_tensors.append(rank_tensor.float())
                    merged_tensor = torch.stack(rank_tensors, dim=0).mean(dim=0)
            if tuple(target.shape) != tuple(merged_tensor.shape):
                if _copy_with_vocab_overlap(model_key, target, merged_tensor):
                    loaded_keys.append(model_key)
                    continue
                shape_mismatches.append((model_key, tuple(merged_tensor.shape), tuple(target.shape)))
                continue

            target.copy_(merged_tensor.to(device=target.device, dtype=target.dtype))
            loaded_keys.append(model_key)
            if key_idx % 200 == 0:
                print(f"[hunyuan-load] processed {key_idx}/{len(ordered_keys)} keys, loaded={len(loaded_keys)}")

    if replicated_dp_keys:
        _lf = [k for k in replicated_dp_keys if k.endswith("_lossfree_bias")]
        print(f"[hunyuan-load] replicated-DP buffers loaded ({replicated_dp_mode}): "
              f"{len(replicated_dp_keys)} keys, of which {len(_lf)} _lossfree_bias")
    if shape_mismatches:
        print(f"Hunyuan online overlay shape mismatches (first 10): {shape_mismatches[:10]}")
        _lf_bad = [m for m in shape_mismatches if m[0].endswith("_lossfree_bias")]
        if _lf_bad:
            print(f"WARNING: {len(_lf_bad)} _lossfree_bias buffers were NOT loaded — MoE routing "
                  f"will differ from training. Set HUNYUAN_REPLICATED_DP_MODE=mean (default).")
    if unexpected:
        print(f"Unexpected Hunyuan checkpoint keys (first 10): {unexpected[:10]}")
    # A copied-key count alone cannot prove that the overlay is complete: the
    # destination model may contain keys that never appeared in the checkpoint.
    # Report that side of the set difference as well so inference diagnostics
    # cannot silently accept base-model/randomly-initialised parameters.
    missing_model_keys = sorted(set(model_state) - set(loaded_keys))
    # The language-model host also owns the separately pretrained, frozen VAE
    # decoder.  It is intentionally absent from Hunyuan training checkpoints.
    externally_managed_missing = [key for key in missing_model_keys if key.startswith("vae.")]
    checkpoint_managed_missing = [key for key in missing_model_keys if not key.startswith("vae.")]
    if checkpoint_managed_missing:
        print(
            f"WARNING: Hunyuan online overlay left {len(checkpoint_managed_missing)} "
            f"checkpoint-managed model keys unloaded (first 20): "
            f"{checkpoint_managed_missing[:20]}"
        )
    else:
        print("[hunyuan-load] overlay completeness: all checkpoint-managed model keys loaded")
    if externally_managed_missing:
        print(
            f"[hunyuan-load] overlay completeness: ignored {len(externally_managed_missing)} "
            "externally managed frozen VAE keys"
        )
    print(f"Hunyuan online overlay loaded {len(loaded_keys)} keys from {checkpoint_path}")


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
    if model_name == "hunyuan_image_3":
        return _load_hunyuan_release_model(
            model_path=model_path,
            modality_config_path=modality_config_path,
        )

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


def _load_hunyuan_release_model(*, model_path: str, modality_config_path: Optional[str]):
    """Load a self-contained ``modus_hunyuan_inference_v1`` export."""
    import json

    from modeling.hunyuan_image_3.src.modeling_hunyuan_image_3 import HunyuanImage3ForCausalMM
    from modeling.hunyuan_image_3.vae_adapter import HunyuanVAEAdapter

    metadata_path = os.path.join(model_path, "modus_config.json")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"Missing {metadata_path}; expected an exported MODUS-Hunyuan checkpoint"
        )
    with open(metadata_path, "r", encoding="utf-8") as handle:
        release_metadata = json.load(handle)
    if release_metadata.get("format") != "modus_hunyuan_inference_v1":
        raise RuntimeError(f"Unsupported Hunyuan checkpoint format: {release_metadata}")
    runtime_profile = release_metadata.get("runtime_profile")
    if not isinstance(runtime_profile, dict) or not runtime_profile:
        raise RuntimeError("Exported Hunyuan checkpoint has no runtime profile")
    runtime_drift = {
        key: (os.environ.get(key), str(value))
        for key, value in runtime_profile.items()
        if os.environ.get(key) != str(value)
    }
    allow_runtime_override = (
        os.environ.get("MODUS_HUNYUAN_ALLOW_RUNTIME_OVERRIDE", "0") == "1"
    )
    if runtime_drift and not allow_runtime_override:
        raise RuntimeError(
            "Hunyuan checkpoint/runtime profile mismatch: "
            f"{runtime_drift}"
        )
    if runtime_drift:
        print(
            "[hunyuan-release-load] WARNING: intentional runtime profile "
            f"override active: {runtime_drift}",
            flush=True,
        )

    embedded_modality_name = release_metadata.get("modality_config", "modality_config.yaml")
    if (
        not isinstance(embedded_modality_name, str)
        or not embedded_modality_name
        or os.path.isabs(embedded_modality_name)
        or ".." in embedded_modality_name.split(os.sep)
    ):
        raise RuntimeError(
            f"Invalid embedded modality config path: {embedded_modality_name!r}"
        )
    embedded_modality_config = os.path.join(model_path, embedded_modality_name)
    if modality_config_path is None:
        modality_config_path = embedded_modality_config
    if not os.path.isfile(modality_config_path):
        raise FileNotFoundError(
            f"Missing Hunyuan modality config: {modality_config_path}"
        )
    device_map = os.environ.get("HUNYUAN_DEVICE_MAP", "auto")
    # Match the loader used for the raw training checkpoint.  Passing an
    # implementation here changes Transformers' dispatch and can introduce
    # avoidable numerical drift.  An explicit environment override remains
    # available for diagnostics.
    attn_impl = os.environ.get("HUNYUAN_NATIVE_ATTN_IMPL")
    print(
        f"[hunyuan-release-load] model={model_path} device_map={device_map} "
        f"attention={attn_impl or 'model-default'}",
        flush=True,
    )
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": device_map,
    }
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl
    host = HunyuanImage3ForCausalMM.from_pretrained(model_path, **load_kwargs)
    host.eval()

    tokenizer, new_token_ids, num_new_tokens, modality_registry = (
        _build_tokenizer_and_registry(
            model_path, modality_config_path, model_name="hunyuan_image_3"
        )
    )
    if num_new_tokens:
        raise RuntimeError(
            "Exported Hunyuan tokenizer is incomplete: modality setup added "
            f"{num_new_tokens} tokens at load time"
        )

    vit_max_patch = int(os.environ.get("HUNYUAN_INFER_VIT_MAX_PATCH", "64"))
    wrapper = HunyuanImageWrapper(host, vit_max_num_patch_per_side=vit_max_patch)
    wrapper.set_modality_registry(modality_registry)
    adapter_path = os.path.join(model_path, "modus_adapter.safetensors")
    if os.path.isfile(adapter_path):
        adapter_state = load_file(adapter_path, device="cpu")
        host_state = host.state_dict()
        absent = sorted(set(adapter_state) - set(host_state))
        if absent:
            raise RuntimeError(f"MODUS adapter keys have no model target: {absent}")
        bad_shapes = sorted(
            (key, tuple(value.shape), tuple(host_state[key].shape))
            for key, value in adapter_state.items()
            if tuple(value.shape) != tuple(host_state[key].shape)
        )
        if bad_shapes:
            raise RuntimeError(f"MODUS adapter shape mismatch: {bad_shapes}")
        missing, unexpected = host.load_state_dict(adapter_state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected MODUS adapter keys: {unexpected}")
        # ``strict=False`` reports the entire base model as missing.  Verify the
        # small adapter explicitly so device-map/offload behavior cannot make a
        # nominally successful call silently retain random initialization.
        host_state = host.state_dict()
        unequal = sorted(
            key for key, value in adapter_state.items()
            if not torch.equal(host_state[key].detach().cpu(), value)
        )
        if unequal:
            raise RuntimeError(f"MODUS adapter tensors were not installed: {unequal}")
        del adapter_state, host_state, missing

    vae_cfg = host.config.vae
    scale_factor = (
        vae_cfg.get("scaling_factor", 0.562679178327931)
        if isinstance(vae_cfg, dict)
        else getattr(vae_cfg, "scaling_factor", 0.562679178327931)
    )
    vae_model = HunyuanVAEAdapter(host.vae, scale_factor=scale_factor)
    vae_model.source_model_path = model_path
    wrapper.eval()
    vae_model.eval()
    return wrapper, vae_model, tokenizer, new_token_ids, modality_registry


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
    if model_name == "janus_flow":
        if train_config_path is None:
            raise ValueError("train_config_path is required for model_name=janus_flow")
        model, vae_model, vae_config, vit_config = _build_janus_model_from_train_config(
            model_path=model_path,
            train_config_path=train_config_path,
            init_on_gpu=init_on_gpu,
        )
    elif model_name == "hunyuan_image_3":
        train_config_path = train_config_path or "conf/train/hunyuan_image_3.yaml"
        if init_on_gpu:
            model, vae_model, vae_config, vit_config = _build_hunyuan_model_distributed_for_inference(
                model_path=model_path,
                train_config_path=train_config_path,
            )
        else:
            model, vae_model, vae_config, vit_config = _build_hunyuan_model_from_train_config(
                model_path=model_path,
                train_config_path=train_config_path,
                init_on_gpu=init_on_gpu,
            )
    else:
        model, vae_model, vae_config, vit_config = build_model(
            model_name,
            model_path=model_path,
            init_device=("cpu" if init_on_gpu else "meta"),
            init_on_gpu=init_on_gpu,
        )
    target_device = _infer_target_device(init_on_gpu=init_on_gpu)
    print(f"[hunyuan-load] target_device={target_device}")

    tokenizer, new_token_ids, num_new_tokens, modality_registry = (
        _build_tokenizer_and_registry(model_path, modality_config_path, model_name=model_name)
    )
    print(f"[hunyuan-load] tokenizer ready, num_new_tokens={num_new_tokens}")

    if not (model_name == "hunyuan_image_3" and init_on_gpu):
        model = _move_and_maybe_bf16(model, device=target_device)
    vae_model = _move_and_maybe_bf16(vae_model, device=target_device)
    print("[hunyuan-load] model moved to target device")

    if num_new_tokens > 0:
        _resize_demo_token_embeddings(model, len(tokenizer), model_name=model_name)
        print(f"[hunyuan-load] resized token embeddings to {len(tokenizer)}")

    # Attach registry BEFORE loading weights so pos-embed modules exist.
    if hasattr(model, "set_modality_registry"):
        model.set_modality_registry(modality_registry)
    else:
        model.modality_registry = modality_registry

    if model_name == "hunyuan_image_3":
        tp_size = int(os.environ.get("HUNYUAN_TP_SIZE", "4"))
        # Input-packing/logit-parity diagnostics compare two paths through the
        # same model, so checkpoint values cancel out.  Allow those diagnostics
        # to skip the very expensive 64-shard online overlay explicitly.
        if os.environ.get("HUNYUAN_SKIP_CHECKPOINT_OVERLAY", "0") == "1":
            print("[hunyuan-load] checkpoint overlay skipped by explicit diagnostic flag")
        else:
            _overlay_hunyuan_checkpoint_online(
                model=model,
                checkpoint_path=checkpoint_path,
                use_ema=use_ema,
                tp_size=tp_size,
            )
    else:
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
