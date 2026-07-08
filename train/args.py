# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Training argument dataclasses.

These define all CLI/config parameters for training.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """Arguments related to model architecture and paths."""

    model_name: str = field(
        default="bagel",
        metadata={"help": "Which model architecture to build (registered name)."},
    )
    model_path: str = field(
        default="hf/BAGEL-7B-MoT",
        metadata={"help": "Path of the pretrained BAGEL model."},
    )
    llm_path: str = field(
        default="hf/Qwen2.5-0.5B-Instruct/",
        metadata={"help": "Path or HuggingFace repo ID of the pretrained Qwen2-style language model."},
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."},
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."},
    )
    layer_module: str = field(
        default="Qwen2MoTDecoderLayer",
        metadata={"help": "Python class name of the decoder layer to instantiate."},
    )
    vae_path: str = field(
        default="flux/vae/ae.safetensors",
        metadata={"help": "Path to the pretrained VAE checkpoint for latent-space image generation."},
    )
    vit_path: str = field(
        default="hf/siglip-so400m-14-980-flash-attn2-navit/",
        metadata={"help": "Path or repo ID of the SigLIP Vision Transformer used for image understanding."},
    )
    max_latent_size: int = field(
        default=32,
        metadata={"help": "Maximum latent grid size (patches per side) for the VAE latent tensor."},
    )
    latent_patch_size: int = field(
        default=2,
        metadata={"help": "Spatial size (in VAE pixels) covered by each latent patch."},
    )
    vit_patch_size: int = field(
        default=14,
        metadata={"help": "Patch size (pixels) for the Vision Transformer encoder."},
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={"help": "Maximum number of ViT patches along one image side after cropping / resize."},
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={"help": "Activation function used in the latent-to-text connector MLP."},
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={"help": "Interpolate positional embeddings when image resolution differs from pre-training."},
    )
    vit_select_layer: int = field(
        default=-2,
        metadata={"help": "Which hidden layer of the ViT to take as the visual feature (negative = from the end)."},
    )
    vit_rope: bool = field(
        default=False,
        metadata={"help": "Replace ViT positional encodings with RoPE."},
    )
    text_cond_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of dropping text embeddings during training."},
    )
    vae_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping VAE latent inputs during training."},
    )
    vit_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping ViT visual features during training."},
    )
    grounding_phrase_dropout_prob: float = field(
        default=0.5,
        metadata={"help": "Probability of dropping phrase text when grounding is used as a condition."},
    )


@dataclass
class DataArguments:
    """Arguments related to data loading and batching."""

    dataset_config_file: str = field(
        default="conf/data/modus_stage1_16mod.yaml",
        metadata={"help": "YAML file specifying dataset groups, weights, and preprocessing rules."},
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "How many batches each DataLoader worker pre-loads in advance."},
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of background workers for the PyTorch DataLoader."},
    )
    max_num_tokens_per_sample: int = field(
        default=16384,
        metadata={"help": "Maximum tokens allowed in one raw sample; longer samples are skipped."},
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={"help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."},
    )
    prefer_buffer_before: int = field(
        default=16384,
        metadata={"help": "While batch length is below this, pop from the overflow buffer before new sampling."},
    )
    max_buffer_size: int = field(
        default=50,
        metadata={"help": "Maximum number of oversized samples kept in the overflow buffer."},
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "Seed used when shuffling / sampling data shards to ensure reproducibility."},
    )


@dataclass
class TrainingArguments:
    """Arguments related to training configuration."""

    # ─────── Modality Switches ─────────────────────────────────────────────────
    visual_gen: bool = field(default=True, metadata={"help": "Train image generation branch."})
    visual_und: bool = field(default=True, metadata={"help": "Train image understanding branch."})

    # ─────── Logging & Checkpointing ───────────────────────────────────────────
    results_dir: str = field(default="results", metadata={"help": "Root directory for logs."})
    checkpoint_dir: str = field(default="results/checkpoints", metadata={"help": "Root directory for model checkpoints."})
    wandb_project: str = field(default="bagel", metadata={"help": "Weights & Biases project name."})
    wandb_name: str = field(default="run", metadata={"help": "Name shown in the Weights & Biases UI for this run."})
    wandb_runid: str = field(default="0", metadata={"help": "Unique identifier to resume a previous W&B run."})
    wandb_resume: str = field(default="allow", metadata={"help": "W&B resume mode: 'allow', 'must', or 'never'."})
    wandb_offline: bool = field(default=False, metadata={"help": "Run W&B in offline mode."})
    log_rgb_condition_loss_group: bool = field(
        default=False,
        metadata={"help": "Log caption2rgb/grounding2rgb/dinolocal2rgb loss+timestep group to W&B."},
    )
    log_every: int = field(default=10, metadata={"help": "Print / log every N training steps."})
    save_every: int = field(default=2000, metadata={"help": "Save a checkpoint every N training steps."})
    total_steps: int = field(default=500_000, metadata={"help": "Total number of optimizer steps to train for."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of micro-steps to accumulate gradients over before an optimizer step. Default 1 = no accumulation."})

    # ─────── Resume & Seed ─────────────────────────────────────────────────────
    global_seed: int = field(default=4396, metadata={"help": "Base random seed; actual seed is offset by rank for DDP."})
    auto_resume: bool = field(default=False, metadata={"help": "Automatically pick up the latest checkpoint."})
    resume_from: Optional[str] = field(default=None, metadata={"help": "Explicit checkpoint path to resume from."})
    resume_model_only: bool = field(default=False, metadata={"help": "Load only model weights, ignoring optimizer/scheduler."})
    finetune_from_ema: bool = field(default=False, metadata={"help": "When resume_model_only=True, load EMA weights."})
    finetune_from_hf: bool = field(default=False, metadata={"help": "Whether finetune from HuggingFace model."})
    checkpoint_name: str = field(default="ema.special_token_patched.safetensors", metadata={"help": "Name of the checkpoint to load."})

    # ─────── Optimization ──────────────────────────────────────────────────────
    warmup_steps: int = field(default=2000, metadata={"help": "Linear warm-up steps."})
    lr_scheduler: str = field(default="constant", metadata={"help": "Type of LR schedule: 'constant' or 'cosine'."})
    lr: float = field(default=1e-4, metadata={"help": "Peak learning rate after warm-up."})
    min_lr: float = field(default=1e-7, metadata={"help": "Minimum learning rate for cosine schedule."})
    beta1: float = field(default=0.9, metadata={"help": "AdamW β₁ coefficient."})
    beta2: float = field(default=0.95, metadata={"help": "AdamW β₂ coefficient."})
    eps: float = field(default=1e-15, metadata={"help": "AdamW ε for numerical stability."})
    ema: float = field(default=0.9999, metadata={"help": "Decay rate for EMA of model weights."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Gradient clipping threshold (L2 norm)."})

    # ─────── Diffusion / Timestep ──────────────────────────────────────────────
    timestep_shift: float = field(default=1.0, metadata={"help": "Shift applied to diffusion timestep indices."})
    timestep_sample: str = field(default="logit_norm", metadata={"help": "How to sample diffusion timesteps."})
    mode_scale: float = field(default=0.0, metadata={"help": "Scale factor for mode-based timestep sampling."})
    timestep_sample_mix_prob: float = field(default=0.5, metadata={"help": "Probability of mixing timestep sample with pure noise."})

    # ─────── Loss Weighting ────────────────────────────────────────────────────
    mse_weight: float = field(default=1.0, metadata={"help": "Scaling factor for the image-reconstruction MSE loss term."})
    ce_weight: float = field(default=1.0, metadata={"help": "Scaling factor for the language cross-entropy loss term."})
    ce_loss_reweighting: bool = field(default=False, metadata={"help": "Reweight CE loss by token importance."})
    ce_loss_average_over_modalities: bool = field(default=False, metadata={"help": "Average CE loss over modalities."})
    expected_num_tokens: int = field(default=32768, metadata={"help": "Soft target token count for batch packing."})

    # ─────── FSDP / Distributed ────────────────────────────────────────────────
    num_replicate: int = field(default=1, metadata={"help": "Number of model replicas per GPU rank."})
    num_shard: int = field(default=8, metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."})
    sharding_strategy: str = field(default="HYBRID_SHARD", metadata={"help": "FSDP sharding strategy."})
    backward_prefetch: str = field(default="BACKWARD_PRE", metadata={"help": "FSDP backward prefetch strategy."})
    cpu_offload: bool = field(default=False, metadata={"help": "Enable FSDP parameter offload to CPU."})

    # ─────── Module Freezing ───────────────────────────────────────────────────
    freeze_llm: bool = field(default=False, metadata={"help": "Keep language-model weights fixed."})
    freeze_vit: bool = field(default=False, metadata={"help": "Keep ViT weights fixed during training."})
    freeze_vae: bool = field(default=True, metadata={"help": "Keep VAE weights fixed."})
    freeze_und: bool = field(default=False, metadata={"help": "Freeze the visual understanding connector layers."})
    copy_init_moe: bool = field(default=True, metadata={"help": "Duplicate initial weights into the MoT gen-expert branch (init_moe)."})

    # ─────── Instruction / Modality Config ─────────────────────────────────────
    use_flex: bool = field(default=False, metadata={"help": "Enable FLEX packing algorithm."})
    use_instruction: bool = field(default=False, metadata={"help": "Use instruction for multimodal generation."})
    use_condition_instruction: bool = field(default=True, metadata={"help": "Use condition instruction."})
    use_target_instruction: bool = field(default=True, metadata={"help": "Use target instruction."})
    num_condition_modalities: int = field(default=0, metadata={"help": "Number of condition modalities."})
    strict_num_condition_modalities: bool = field(
        default=False,
        metadata={"help": "When set, require exactly num_condition_modalities conditions instead of sampling 1..N."},
    )
    do_modality_norm: bool = field(default=False, metadata={"help": "Normalize VAE latents by modality type."})
    use_det_image: bool = field(default=False, metadata={"help": "Use detection image for multimodal generation."})
    modality_config_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to YAML file defining modalities. If None, selected based on use_instruction."},
    )

    # ─────── Online Validation ─────────────────────────────────────────────────
    # Backward-compatible: when validation_pack_path is None / missing, the
    # online validation hook in train/online_validation.py is a no-op and
    # nothing changes for existing runs.
    validation_pack_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a .pt file produced by scripts/prep_online_val_pack.py (None = disable online validation)."},
    )
    validate_every: int = field(
        default=2000,
        metadata={"help": "Run online validation every N optimizer steps (0 = never; ignored if validation_pack_path is None)."},
    )
