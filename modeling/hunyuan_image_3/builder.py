"""
HunyuanImage-3.0 model builder registered in the global MODUS model registry.

Usage (via MODUS config):
    model:
      model_name: hunyuan_image_3
      model_path: /path/to/HunyuanImage-3-Instruct

The builder:
  1. Loads the AutoencoderKLConv3D VAE (frozen, CPU/meta, not FSDP-wrapped)
  2. Loads HunyuanImage3ForCausalMM WITHOUT the internal VAE (saves memory)
  3. Wraps the model in HunyuanImageWrapper to expose the MODUS training interface
  4. Returns (wrapper, vae_adapter, vae_config, None)
"""

from __future__ import annotations

import os
import sys
import torch

from core.model_registry import register_model
from .vae_adapter import HunyuanVAEAdapter, HunyuanVAEConfig
from .model_wrapper import HunyuanImageWrapper

# Ensure the bundled src package is importable.
_SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


@register_model("hunyuan_image_3")
def build_hunyuan_image_3(
    *,
    model_args,
    training_args,
    init_device: str = "meta",
):
    """
    Build HunyuanImage-3.0 and (optionally) the associated VAE for MODUS training.

    Returns:
        (HunyuanImageWrapper, HunyuanVAEAdapter | None, HunyuanVAEConfig | None, None)
    """
    from modeling.hunyuan_image_3.src.modeling_hunyuan_image_3 import HunyuanImage3ForCausalMM
    from modeling.hunyuan_image_3.src.autoencoder_kl_3d import AutoencoderKLConv3D

    model_path = model_args.model_path

    # ── VAE (optional, loaded only when visual generation is enabled) ─────────
    vae_model = None
    vae_config = None
    if getattr(training_args, "visual_gen", True):
        # Load the VAE from the checkpoint directory.
        # HunyuanImage-3-Instruct/config.json stores vae config under "vae" key.
        import json

        cfg_path = os.path.join(model_path, "config.json")
        with open(cfg_path, "r") as f:
            full_cfg = json.load(f)
        vae_cfg_dict = full_cfg.get("vae", {})

        # Remove _class_name if present (not a constructor arg)
        vae_cfg_dict = {k: v for k, v in vae_cfg_dict.items() if not k.startswith("_")}
        if os.environ.get("HUNYUAN_VAE_ENCODER_ONLY", "0") == "1":
            # Training only calls encode(). Avoid placing the unused decoder on
            # every GPU; this is especially valuable for EP, where dense state
            # is replicated across expert-parallel ranks.
            vae_cfg_dict["only_encoder"] = True
            vae_cfg_dict["only_decoder"] = False
            print("[hunyuan-build] VAE encoder-only training mode enabled", flush=True)

        # Build VAE on CPU first (weights are in the main safetensors shards)
        vae = AutoencoderKLConv3D(**vae_cfg_dict)
        vae = _load_vae_weights(vae, model_path)
        vae = vae.eval()
        for p in vae.parameters():
            p.requires_grad = False

        vae_model = HunyuanVAEAdapter(vae, scale_factor=vae_cfg_dict.get("scaling_factor", 0.562679178327931))
        vae_model.source_model_path = model_path
        vae_config = HunyuanVAEConfig()

    # ── Main model (transformer + ViT + gen heads, no internal VAE) ──────────
    # skip_load_module={'vae'} prevents HunyuanImage3ForCausalMM from loading
    # its own copy of the VAE (we handle it separately above).
    import glob as _glob
    _shards = _glob.glob(os.path.join(model_path, "model-*.safetensors"))
    _has_weights = bool(_shards) or os.path.exists(os.path.join(model_path, "model.safetensors"))

    from .src.configuration_hunyuan_image_3 import HunyuanImage3Config
    _config = HunyuanImage3Config.from_pretrained(model_path)

    # All ranks use meta device — no CPU RAM consumed for parameters.
    # load_hf_weights_before_fsdp will materialise to CPU before FSDP wrap,
    # then copy HF weights onto real tensors. If no HF weights exist, FSDP
    # will materialise them via _fsdp_param_init_fn instead.
    with torch.device("meta"):
        model = HunyuanImage3ForCausalMM(_config, skip_load_module=["vae"])
    # Keep params in default fp32 so FSDP1 MixedPrecision provides an fp32
    # master weight (param_dtype=bf16 only casts for fwd/bwd). Pre-casting
    # the model to bf16 disables that path: AdamW updates apply directly to
    # bf16 params and any update < bf16 LSB at the param's scale rounds to
    # zero, silently freezing wte/lm_head and any sparsely-updated param.

    # model_args comes from conf/train/hunyuan_image_3.yaml, which says 16 — but
    # the 16-mod runs override it to 64 on the command line, so an eval that
    # rebuilds from the yaml caps the ViT grid at 16x16=256 patches while
    # training fed up to 64x64. Measured: 192 ViT tokens per condition image at
    # inference vs 1024 in training.
    vit_max_patch = getattr(model_args, "vit_max_num_patch_per_side", 70)
    _vit_override = os.environ.get("HUNYUAN_INFER_VIT_MAX_PATCH")
    if _vit_override:
        print(f"[hunyuan-build] vit_max_num_patch_per_side {vit_max_patch} -> "
              f"{_vit_override} (HUNYUAN_INFER_VIT_MAX_PATCH)", flush=True)
        vit_max_patch = int(_vit_override)
    wrapper = HunyuanImageWrapper(model, vit_max_num_patch_per_side=vit_max_patch)

    # Signal to train_utils that HF weights should be loaded post-FSDP.
    if _has_weights:
        wrapper._hf_model_path = model_path
        wrapper._hf_weight_prefix = "hunyuan_model."
        wrapper._hf_skip_prefixes = ("vae.",)
    else:
        wrapper._hf_model_path = None

    return wrapper, vae_model, vae_config, None


# ── Internal helper ──────────────────────────────────────────────────────────

def _load_vae_weights(vae: "AutoencoderKLConv3D", model_path: str) -> "AutoencoderKLConv3D":
    """
    Load VAE weights from the model's safetensors shards.

    The VAE parameters in HunyuanImage-3-Instruct are stored inside the main
    model shards with the prefix 'vae.'.  We use model.safetensors.index.json
    (when available) to identify only the shards that contain 'vae.*' keys so
    that we avoid loading all 32 × 5 GB shards just to extract the VAE.
    """
    import glob
    import json
    from safetensors.torch import load_file

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f).get("weight_map", {})
        vae_shard_names = sorted({v for k, v in weight_map.items() if k.startswith("vae.")})
        shard_paths = [os.path.join(model_path, s) for s in vae_shard_names]
    else:
        shard_paths = sorted(glob.glob(os.path.join(model_path, "model-*.safetensors")))
        if not shard_paths:
            single = os.path.join(model_path, "model.safetensors")
            if os.path.exists(single):
                shard_paths = [single]

    if not shard_paths:
        import warnings
        warnings.warn(f"No safetensors shards in {model_path}; VAE is randomly initialized.")
        return vae

    prefix = "vae."
    vae_state: dict = {}
    for shard_path in shard_paths:
        shard = load_file(shard_path, device="cpu")
        for k, v in shard.items():
            if k.startswith(prefix):
                vae_state[k[len(prefix):]] = v
        del shard

    if not vae_state:
        raise RuntimeError(f"No 'vae.*' keys found in safetensors shards under {model_path}")

    missing, unexpected = vae.load_state_dict(vae_state, strict=False)
    if unexpected:
        import warnings
        warnings.warn(f"Unexpected VAE keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
    return vae
