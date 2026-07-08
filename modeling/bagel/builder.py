"""
Bagel model builder registered in the global model registry.

This is the first step towards making the repo model-extendible:
- training/inference scripts should call the registry by name
- adding a new model should only require adding a new builder and config
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

from core.model_registry import register_model

from modeling.autoencoder import load_ae
from modeling.bagel import BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel


@register_model("bagel")
def build_bagel_for_training(
    *,
    model_args,
    training_args,
    init_device: str = "meta",
) -> Tuple[Bagel, Optional[object], Optional[object], Optional[object]]:
    """
    Build Bagel model and (optionally) associated VAE/ViT models for training.

    Returns:
      (model, vae_model, vae_config, vit_config)
    """
    # --- LLM ---
    if training_args.finetune_from_hf:
        llm_config = Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
    else:
        llm_config = Qwen2Config.from_pretrained(model_args.llm_path)
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und

    if training_args.finetune_from_hf:
        language_model = Qwen2ForCausalLM(llm_config).to(init_device)
    else:
        language_model = Qwen2ForCausalLM.from_pretrained(model_args.llm_path, config=llm_config).to(init_device)
    if training_args.copy_init_moe:
        language_model.init_moe()

    # --- ViT (optional) ---
    vit_model = None
    vit_config = None
    if training_args.visual_und:
        if training_args.finetune_from_hf:
            vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_args.model_path, "vit_config.json"))
        else:
            vit_config = SiglipVisionConfig.from_pretrained(model_args.vit_path)
        vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
        vit_config.rope = model_args.vit_rope
        if training_args.finetune_from_hf:
            vit_model = SiglipVisionModel(vit_config).to(init_device)
        else:
            vit_model = SiglipVisionModel.from_pretrained(model_args.vit_path, config=vit_config).to(init_device)

    # --- VAE (optional) ---
    vae_model = None
    vae_config = None
    if training_args.visual_gen:
        vae_model, vae_config = load_ae(
            local_path=os.path.join(model_args.model_path, "ae.safetensors")
            if training_args.finetune_from_hf
            else model_args.vae_path
        )

    # --- Bagel config/model ---
    config = BagelConfig(
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
        llm_config=llm_config,
        vit_config=vit_config if training_args.visual_und else None,
        vae_config=vae_config if training_args.visual_gen else None,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
        timestep_sample=training_args.timestep_sample,
        mode_scale=training_args.mode_scale,
        timestep_sample_mix_prob=training_args.timestep_sample_mix_prob,
    )
    model = Bagel(language_model, vit_model if training_args.visual_und else None, config).to(init_device)

    # Special-case: SigLIP conv2d->linear conversion needs real storage; do a safe CPU hop.
    if training_args.visual_und:
        model = model.to_empty(device="cpu")
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
        model = model.to(init_device)

    return model, vae_model, vae_config, vit_config


@register_model("bagel_from_json")
def build_bagel_from_json_configs(
    *,
    model_path: str,
    init_device: str = "meta",
    init_on_gpu: bool = True,
) -> Tuple[Bagel, object, object, object]:
    """
    Build Bagel from serialized config files in `model_path` (llm_config.json, vit_config.json, ae.safetensors).

    Intended for inference/demo loaders so they can be model-registry driven.
    Returns:
      (model, vae_model, vae_config, vit_config)
    """
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    # Match demo defaults
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    if init_on_gpu:
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
    else:
        from accelerate import init_empty_weights

        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
        model = model.to(init_device)

    return model, vae_model, vae_config, vit_config
