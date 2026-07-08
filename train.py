#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Unified training entry-point for MODUS/BAGEL.

Usage:
    torchrun train.py --config debug_any2any
    torchrun train.py --config debug_any2any training.lr=2e-5  # with overrides
"""

import contextlib
import gc
import os
import re
from time import time

import torch
import torch.distributed as dist
import wandb
from transformers import set_seed

from core.model_registry import build_model
from data.dataset_info import MODALITY_STATS, normalize_latents_by_modality
from train.args import ModelArguments, DataArguments, TrainingArguments
from train.train_utils import (
    parse_args,
    init_distributed,
    setup_logger_and_wandb,
    resolve_resume,
    build_tokenizer_and_modality_registry,
    maybe_freeze_components,
    setup_fsdp_and_load_checkpoint,
    build_optimizer_and_scheduler,
    build_train_dataloader,
    compute_loss,
    log_training_step,
    should_skip_vit_batch,
    _stage_debug_log,
)
from train.fsdp_utils import FSDPCheckpoint, FSDPConfig, fsdp_ema_update

import modeling  # register model builders


def main():
    # ─────── Parse args & init distributed ─────────────────────────────────────
    model_args, data_args, training_args = parse_args(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    device = init_distributed()
    logger = setup_logger_and_wandb(training_args, model_args, data_args)

    # Enforce SDPA backend globally so all attention call sites (including HF
    # modules outside our custom wrappers) follow the same backend policy.
    _sdpa_backend = os.environ.get("HUNYUAN_SDPA_BACKEND", "auto").strip().lower()
    if _sdpa_backend in {"math", "efficient", "flash"}:
        if _sdpa_backend == "math":
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        elif _sdpa_backend == "efficient":
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
        elif _sdpa_backend == "flash":
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)  # fallback for ops with head_dim > 256 (e.g. VAE)
        if dist.get_rank() == 0:
            logger.info(f"[SDPA] Global torch SDPA backend forced to: {_sdpa_backend}")
    elif _sdpa_backend in {"no_efficient", "flash_math"} and dist.get_rank() == 0:
        logger.info(
            f"[SDPA] Per-module backend mode requested: {_sdpa_backend} "
            "(global torch SDPA flags left at defaults)"
        )

    logger.info(f"Training arguments: {training_args}")
    logger.info(f"Model arguments: {model_args}")
    logger.info(f"Data arguments: {data_args}")

    if training_args.do_modality_norm:
        logger.info(f"Modality normalization ENABLED - available stats: {list(MODALITY_STATS.keys())}")
    else:
        logger.info("Modality normalization DISABLED")

    # ─────── Resume logic ──────────────────────────────────────────────────────
    resume_from, resume_model_only, finetune_from_ema, checkpoint_name = resolve_resume(training_args)

    # ─────── Seed ──────────────────────────────────────────────────────────────
    dp_rank = dist.get_rank()
    dp_size = dist.get_world_size()
    seed = training_args.global_seed * dp_size + dp_rank
    set_seed(seed)

    # ─────── Build model ───────────────────────────────────────────────────────
    model, vae_model, vae_config, vit_config = build_model(
        model_args.model_name,
        model_args=model_args,
        training_args=training_args,
        init_device="meta",
    )
    tokenizer, new_token_ids, num_new_tokens, modality_registry = (
        build_tokenizer_and_modality_registry(model_args, training_args)
    )
    maybe_freeze_components(model, vae_model, training_args)

    # ─────── FSDP setup & checkpoint ──────────────────────────────────────────
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )
    fsdp_model, ema_model, model_flops_per_token = setup_fsdp_and_load_checkpoint(
        model, training_args, fsdp_config, modality_registry, tokenizer, num_new_tokens,
        resume_from, finetune_from_ema, checkpoint_name, logger,
    )

    # ── Repair BAGEL sin-cos position embeddings after FSDP load ─────────────
    # fsdp_utils.py:load_full_state_dict pops `latent_pos_embed.pos_embed` and
    # `vit_pos_embed.pos_embed` (assuming sin-cos init will refill), but
    # FSDP's param_init_fn calls reset_parameters on parent Bagel which has
    # none — these buffers end up as uninit cuda memory (absmax ≈ 0.007
    # instead of 1.0). This silently breaks image gen & training. Same fix
    # already applied in sanity_check_train_like.py / validate_any2any_matrix.py.
    if model_args.model_name == "bagel":
        from modeling.bagel.modeling_utils import get_2d_sincos_pos_embed as _get_sincos
        import numpy as _np_sincos
        _inner = fsdp_model.module if hasattr(fsdp_model, "module") else fsdp_model
        for _attr, _grid in (
            ("latent_pos_embed", getattr(_inner, "max_latent_size", None)),
            ("vit_pos_embed", getattr(_inner, "vit_max_num_patch_per_side", None)),
        ):
            _mod = getattr(_inner, _attr, None)
            if _mod is None or _grid is None:
                continue
            _buf = getattr(_mod, "pos_embed", None)
            if _buf is None:
                continue
            _np_embed = _get_sincos(_inner.hidden_size, int(_grid))
            _t = torch.from_numpy(_np_sincos.asarray(_np_embed)).to(
                device=_buf.device, dtype=_buf.dtype
            )
            with torch.no_grad():
                _buf.data.copy_(_t)
            if dist.get_rank() == 0:
                logger.info(
                    f"[pos-embed-repair] {_attr}.pos_embed sin-cos refilled: "
                    f"shape={tuple(_buf.shape)} absmax={_buf.detach().float().abs().max().item():.4f}"
                )
            # Also refill EMA model's copy (FSDP load same code path).
            if ema_model is not None:
                _ema_inner = ema_model.module if hasattr(ema_model, "module") else ema_model
                _ema_mod = getattr(_ema_inner, _attr, None)
                if _ema_mod is not None:
                    _ema_buf = getattr(_ema_mod, "pos_embed", None)
                    if _ema_buf is not None:
                        with torch.no_grad():
                            _ema_buf.data.copy_(_t.to(device=_ema_buf.device, dtype=_ema_buf.dtype))
        dist.barrier(device_ids=[torch.cuda.current_device()])

    # ─────── Optimizer, scheduler & data ──────────────────────────────────────
    optimizer, scheduler, train_step, data_status, data_resume_state, training_stats = build_optimizer_and_scheduler(
        fsdp_model, training_args, resume_from, resume_model_only, fsdp_config,
    )
    train_loader = build_train_dataloader(
        data_args, training_args, model_args, tokenizer, new_token_ids,
        modality_registry, vae_config, data_status, data_resume_state,
    )

    # ─────── Prepare for training ──────────────────────────────────────────────
    if training_args.visual_gen:
        vae_model.to(device).eval()
    fsdp_model.train()
    if ema_model is not None:
        ema_model.eval()

    # ─────── Online validation setup (silently disabled if not configured) ────
    # When `training_args.validation_pack_path` is unset, every code path here
    # is a no-op and existing runs are unaffected. When it IS set, we build:
    #   - val_pack: .pt produced by scripts/prep_online_val_pack.py
    #   - dino_tokenizer: VQVAE that decodes dino codebook tokens to features
    #   - inferencer + online_val_generate_dino(pil) callable: actual eval entry
    # Any build failure logs a warning and falls back to no-op.
    _online_val_pack = None
    _online_val_inferencer = None
    _online_val_dino_tokenizer = None
    _val_pack_path = getattr(training_args, "validation_pack_path", None)
    if _val_pack_path:
        try:
            from train.online_validation import load_validation_pack as _load_val_pack
            _online_val_pack = _load_val_pack(_val_pack_path, logger=logger)
        except Exception as _ve:
            if dist.get_rank() == 0:
                logger.warning(f"[online_val] failed to load val pack: {_ve}")
            _online_val_pack = None

        if _online_val_pack is not None:
            # Rank 0 triggers HF download to populate the per-user cache, then
            # a barrier lets all ranks load from disk without racing.
            try:
                from fourm.vq.vqvae import VQVAE as _VQVAE_for_val
                _DINO_TOKENIZER_ID = (
                    "EPFL-VILAB/4M_tokenizers_DINOv2-B14-global_8k_16_224"
                )
                if dist.get_rank() == 0:
                    _ = _VQVAE_for_val.from_pretrained(_DINO_TOKENIZER_ID)
                if dist.is_initialized():
                    dist.barrier()
                _online_val_dino_tokenizer = (
                    _VQVAE_for_val.from_pretrained(_DINO_TOKENIZER_ID).eval().to(device)
                )
            except Exception as _te:
                if dist.get_rank() == 0:
                    logger.warning(f"[online_val] failed to load dino_tokenizer: {_te}")
                _online_val_dino_tokenizer = None

            # ─── KNOWN ISSUE: FSDP × inferencer incompatibility ────────
            # The InterleaveInferencer.unified_inference() path calls into
            # `self.model.forward_cache_update_vae(...)` which then reaches
            # `self.language_model.model.embed_tokens(...)` directly — i.e.
            # it bypasses the FSDP root forward hook. With Bagel wrapped by
            # FSDP HYBRID_SHARD, those nested params are sharded and have
            # no `.data` outside an FSDP forward → conv2d/embedding ops fail
            # with "tensor data not allocated yet".
            #
            # Workarounds tried:
            #   • summon_full_params(writeback=False): OOM (only ~1.3 GiB
            #     free on GPU after the just-finished train step).
            #   • summon_full_params(offload_to_cpu=True, writeback=False):
            #     no OOM, but params land on CPU while VAE / image tensors
            #     stay on GPU → device-mismatch on conv2d weight.
            #   • zero_grad + gc.collect + empty_cache: not enough headroom.
            #
            # Until someone reworks the inferencer to route all submodule
            # access through fsdp_model() forward (or we load a separate
            # non-FSDP eval-only model), build the inferencer is a no-op.
            # The hook below will log "skipped (inferencer or dino_tokenizer
            # missing)" once per validate_every interval. Training itself is
            # unaffected.
            _ONLINE_VAL_INFERENCER_DISABLED = True
            if _online_val_dino_tokenizer is not None and not _ONLINE_VAL_INFERENCER_DISABLED:
                try:
                    from any2any.any2any_tasks import (
                        create_inferencer as _create_val_inferencer,
                        move_tensors_to_device as _move_to_dev,
                    )
                    from data.data_utils import pil_img2rgb as _pil2rgb_for_val
                    _online_val_inferencer = _create_val_inferencer(
                        fsdp_model,
                        vae_model,
                        tokenizer,
                        new_token_ids,
                        _online_val_dino_tokenizer,
                        modality_registry=modality_registry,
                    )

                    # The inferencer's gen path constructs intermediate image
                    # tensors as fp32 (PIL→np→tensor pipeline) but vae_model's
                    # conv layers carry bf16 weights (heterogeneous: not all
                    # params share dtype, so `next(parameters()).dtype` lies).
                    # Hardcode the cast to bf16 to match the training-time
                    # autocast context.
                    _orig_vae_encode = vae_model.encode

                    def _vae_encode_dtype_safe(images, *_a, **_kw):
                        if isinstance(images, torch.Tensor) and images.dtype != torch.bfloat16:
                            images = images.to(torch.bfloat16)
                        return _orig_vae_encode(images, *_a, **_kw)

                    vae_model.encode = _vae_encode_dtype_safe

                    def _online_val_generate_dino(pil_image):
                        """Run a single rgb→dino generation, return (768,) feat.

                        Defaults match scripts/inference_any2any_rgb2dinolocal.sh
                        and any2any/eval/dino_global/eval_cos_sim.py.
                        """
                        img = _pil2rgb_for_val(pil_image).resize((224, 224))
                        inference_hyper = dict(
                            max_think_token_n=17,
                            do_sample=False,
                            text_temperature=0.95,
                            modality_type_dict={
                                "condition": ["rgb"],
                                "target": ["dino"],
                            },
                            use_instruction=False,
                            do_modality_norm=False,
                            use_target_instruction=True,
                            use_condition_instruction=False,
                            dino_pca=None,
                            top_k=0,
                            top_p=1.0,
                            cfg_img_scale=1.0,
                        )
                        model_dev = next(_online_val_inferencer.model.parameters()).device
                        patched = _move_to_dev(inference_hyper, model_dev)
                        # The FSDP-wrapped model carries bf16 weights; mirror the
                        # training-time autocast so inputs match weight dtype.
                        with torch.amp.autocast(
                            "cuda", enabled=True, dtype=torch.bfloat16
                        ):
                            result = _online_val_inferencer(
                                image=img, understanding_output=True, **patched
                            )
                        feat = result.get("dino_feat") if isinstance(result, dict) else None
                        if feat is None:
                            return None
                        return feat.squeeze().float()

                    _online_val_inferencer.online_val_generate_dino = (
                        _online_val_generate_dino
                    )
                    if dist.get_rank() == 0:
                        logger.info(
                            "[online_val] inferencer + dino_tokenizer ready; "
                            "validation will run every "
                            f"{getattr(training_args, 'validate_every', 0)} steps"
                        )
                except Exception as _ie:
                    if dist.get_rank() == 0:
                        logger.warning(
                            f"[online_val] failed to build inferencer wrapper: {_ie}"
                        )
                    _online_val_inferencer = None
            elif _ONLINE_VAL_INFERENCER_DISABLED and dist.get_rank() == 0:
                logger.info(
                    "[online_val] inferencer wire-up DISABLED — known FSDP "
                    "incompatibility (see comment in train.py around the "
                    "_ONLINE_VAL_INFERENCER_DISABLED flag). Val pack + hook "
                    "still load; the hook will log 'skipped' messages but "
                    "training is unaffected. Set the flag to False once "
                    "someone reworks the inferencer to be FSDP-aware."
                )

    # ─────── Training loop ─────────────────────────────────────────────────────
    start_time = time()
    logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")

    if training_stats is not None:
        total_tokens_accumulated = training_stats.get("total_tokens_accumulated", 0)
        total_seq_tokens_accumulated = training_stats.get("total_seq_tokens_accumulated", 0)
        total_samples_accumulated = training_stats.get("total_samples_accumulated", 0)
        total_epoch_samples_accumulated = training_stats.get("total_epoch_samples_accumulated", 0)
        logger.info(
            f"Restored training stats: tokens={total_tokens_accumulated:,}, "
            f"seq_tokens={total_seq_tokens_accumulated:,}, samples={total_samples_accumulated:,}, "
            f"epoch_samples={total_epoch_samples_accumulated:,}"
        )
    else:
        total_tokens_accumulated = 0
        total_seq_tokens_accumulated = 0
        total_samples_accumulated = 0
        total_epoch_samples_accumulated = 0
    total_dataset_samples = max(train_loader.dataset.total_dataset_samples, 1)
    logger.info(f"Total unique dataset samples (for epoch tracking): {total_dataset_samples:,}")

    opt_step = train_step
    num_accum = getattr(training_args, 'gradient_accumulation_steps', 1)
    _tokens_at_last_log = total_tokens_accumulated
    _samples_at_last_log = total_samples_accumulated
    if not isinstance(data_resume_state, dict):
        data_resume_state = {}

    # ─────── PyTorch profiler (optional, rank 0 only) ────────────────────────
    # Enable with HUNYUAN_PROFILE=1. Tune wait/warmup/active via env vars.
    # Output: {results_dir}/profile/*.json (open in chrome://tracing or tensorboard).
    _prof = None
    if os.environ.get("HUNYUAN_PROFILE", "0") == "1" and dist.get_rank() == 0:
        _prof_dir = os.path.join(training_args.results_dir, "profile")
        os.makedirs(_prof_dir, exist_ok=True)
        _prof_wait = int(os.environ.get("HUNYUAN_PROFILE_WAIT", "5"))
        _prof_warmup = int(os.environ.get("HUNYUAN_PROFILE_WARMUP", "2"))
        _prof_active = int(os.environ.get("HUNYUAN_PROFILE_ACTIVE", "3"))
        logger.info(
            f"[PROFILE] enabled on rank 0: wait={_prof_wait}, warmup={_prof_warmup}, "
            f"active={_prof_active}, output={_prof_dir}"
        )
        _prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=_prof_wait, warmup=_prof_warmup, active=_prof_active, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(_prof_dir),
            record_shapes=False,
            with_stack=False,
            profile_memory=False,
        )
        _prof.start()

    if os.environ.get("HUNYUAN_STAGE_TIMING_DEBUG", "0") == "1":
        os.environ["HUNYUAN_STAGE_TIMING_DEBUG_START_STEP"] = str(train_step)

    for curr_step, data in enumerate(train_loader, start=train_step):
        if opt_step >= training_args.total_steps:
            break
        _stage_debug_steps = int(os.environ.get("HUNYUAN_STAGE_TIMING_DEBUG_STEPS", "5"))
        _stage_debug = (
            os.environ.get("HUNYUAN_STAGE_TIMING_DEBUG", "0") == "1"
            and curr_step < (train_step + _stage_debug_steps)
        )
        _batch_shape_debug = os.environ.get("HUNYUAN_BATCH_SHAPE_DEBUG", "0") == "1"
        _batch_shape_debug_steps = int(os.environ.get("HUNYUAN_BATCH_SHAPE_DEBUG_STEPS", "200"))

        def _stage_mark(name: str):
            if not _stage_debug:
                return
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            _msg = f"[STAGE][rank={dist.get_rank()}][step={curr_step}] {name} t={time():.3f}"
            logger.info(_msg)
            print(_msg, flush=True)
            _stage_debug_log(f"[step={curr_step}] {name} t={time():.3f}")

        _stage_mark("loop_enter")
        accum_idx = (curr_step - train_step) % num_accum
        is_last_accum = (accum_idx == num_accum - 1)

        _stage_mark("before_data_cuda")
        data = data.cuda(device).to_dict()
        _stage_mark("after_data_cuda")
        data_indexes = data.pop("batch_data_indexes", None)
        batch_data_resume_state = data.pop("data_resume_state", None)
        ce_loss_weights = data.pop("ce_loss_weights", None)
        mse_loss_route_ids = data.pop("mse_loss_route_ids", None)
        mse_loss_image_ids = data.pop("mse_loss_image_ids", None)
        mse_loss_timesteps = data.pop("mse_loss_timesteps", None)
        batch_num_samples = data.pop("num_samples")
        batch_num_epoch_samples = data.pop("num_epoch_samples", 0)

        if _batch_shape_debug and curr_step < train_step + _batch_shape_debug_steps:
            sample_lens = data.get("sample_lens", []) or []
            nested_attention_masks = data.get("nested_attention_masks", None)
            split_lens = data.get("split_lens", None)
            local_shape = torch.tensor(
                [
                    int(data.get("sequence_length", 0)),
                    int(batch_num_samples),
                    int(max(sample_lens) if len(sample_lens) > 0 else 0),
                    int(len(data["ce_loss_indexes"]) if data.get("ce_loss_indexes", None) is not None else 0),
                    int(len(data["mse_loss_indexes"]) if data.get("mse_loss_indexes", None) is not None else 0),
                    int(len(nested_attention_masks) if isinstance(nested_attention_masks, list) else 0),
                    int(max((int(m.shape[-1]) for m in nested_attention_masks), default=0) if isinstance(nested_attention_masks, list) else 0),
                    int(max(split_lens) if split_lens is not None and len(split_lens) > 0 else 0),
                ],
                device=device,
                dtype=torch.long,
            )
            gathered_shapes = [torch.empty_like(local_shape) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_shapes, local_shape)
            if dist.get_rank() == 0:
                shape_rows = torch.stack(gathered_shapes).cpu()
                seq_col = shape_rows[:, 0]
                max_sample_col = shape_rows[:, 2]
                ce_col = shape_rows[:, 3]
                mse_col = shape_rows[:, 4]
                max_mask_col = shape_rows[:, 6]
                logger.info(
                    f"[BATCH_SHAPE] step={curr_step} "
                    f"seq_min={int(seq_col.min())} seq_max={int(seq_col.max())} "
                    f"max_sample={int(max_sample_col.max())} "
                    f"ce_min={int(ce_col.min())} ce_max={int(ce_col.max())} "
                    f"mse_min={int(mse_col.min())} mse_max={int(mse_col.max())} "
                    f"max_nested_mask={int(max_mask_col.max())} "
                    f"rank_rows={shape_rows.tolist()[:8]}"
                )

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            # Encode images to latents
            if training_args.visual_gen:
                # Mixed batches can be CE-only on some ranks. In that case we keep
                # training and simply skip VAE encode on those ranks.
                _has_padded_images_local = ("padded_images" in data and data["padded_images"] is not None)
                if not _has_padded_images_local:
                    if dist.get_rank() == 0 and curr_step == train_step:
                        logger.warning(
                            "[DBG] Step %s: local rank has no padded_images; proceeding CE-only on this rank",
                            curr_step,
                        )
                else:
                    with torch.no_grad():
                        _stage_mark("before_vae_encode")
                        data["padded_latent"] = vae_model.encode(data.pop("padded_images"))
                        _stage_mark("after_vae_encode")
                        if training_args.do_modality_norm and "vae_image_modality_types" in data:
                            data["padded_latent"] = normalize_latents_by_modality(
                                data["padded_latent"], data["vae_image_modality_types"], device
                            )
            else:
                # Generation-disabled runs still carry padded_images from the
                # dataloader; remove it because HunyuanImageWrapper.forward
                # does not accept this kwarg.
                data.pop("padded_images", None)

            # Skip batches with invalid ViT seq lens (all ranks must agree)
            _stage_mark("before_should_skip_vit_batch")
            if training_args.visual_und and should_skip_vit_batch(data, device):
                continue
            _stage_mark("after_should_skip_vit_batch")

            # Enable anomaly detection only when explicitly requested.
            # Step-0 anomaly mode is very memory-heavy for this model.
            _anomaly_env = os.environ.get("HUNYUAN_ENABLE_ANOMALY", "0") == "1"
            _anomaly = _anomaly_env and (curr_step == train_step)
            torch.autograd.set_detect_anomaly(_anomaly)
            _stage_mark("before_model_forward")
            loss_dict = fsdp_model(**data)
            _stage_mark("after_model_forward")
            if _stage_debug:
                _ce_raw = loss_dict.get("ce", None)
                _mse_raw = loss_dict.get("mse", None)
                _stage_debug_log(
                    f"[step={curr_step}] loss_dict "
                    f"ce_is_none={int(_ce_raw is None)} "
                    f"ce_numel={-1 if not isinstance(_ce_raw, torch.Tensor) else int(_ce_raw.numel())} "
                    f"mse_is_none={int(_mse_raw is None)} "
                    f"mse_numel={-1 if not isinstance(_mse_raw, torch.Tensor) else int(_mse_raw.numel())}"
                )
            if dist.get_rank() == 0 and curr_step == train_step:
                _fmsgs = []
                _ce_raw = loss_dict.get("ce", None)
                _mse_raw = loss_dict.get("mse", None)
                if isinstance(_ce_raw, torch.Tensor):
                    _fmsgs.append(f"ce_finite={bool(torch.isfinite(_ce_raw).all())}")
                if isinstance(_mse_raw, torch.Tensor):
                    _fmsgs.append(f"mse_finite={bool(torch.isfinite(_mse_raw).all())}")
                _cpm = loss_dict.get("ce_per_modality", None)
                if isinstance(_cpm, dict):
                    _bad = [k for k, v in _cpm.items() if isinstance(v, torch.Tensor) and (not torch.isfinite(v).all())]
                    _fmsgs.append(f"ce_per_modality_bad={_bad[:3]}")
                _mpm = loss_dict.get("mse_per_modality", None)
                if isinstance(_mpm, dict):
                    _bad = [k for k, v in _mpm.items() if isinstance(v, torch.Tensor) and (not torch.isfinite(v).all())]
                    _fmsgs.append(f"mse_per_modality_bad={_bad[:3]}")
                logger.info(f"[DBG] Step {curr_step}: forward finite check: " + ", ".join(_fmsgs))

        # ─────── Loss, backward & step ────────────────────────────────────────
        loss, total_ce_tokens, total_mse_tokens = compute_loss(
            loss_dict, data, training_args, ce_loss_weights, device,
            debug_step=curr_step,
            mse_loss_route_ids=mse_loss_route_ids,
            mse_loss_image_ids=mse_loss_image_ids,
            mse_loss_timesteps=mse_loss_timesteps,
        )
        _stage_mark("after_compute_loss")

        # Distinguish "loss is non-finite" from "backward produced non-finite grads".
        _loss_is_finite_local = torch.tensor(
            1 if torch.isfinite(loss).all().item() else 0,
            device=device,
            dtype=torch.int32,
        )
        nonfinite_loss_local = 1 - _loss_is_finite_local
        dist.all_reduce(nonfinite_loss_local, op=dist.ReduceOp.MAX)
        if nonfinite_loss_local.item() > 0:
            if dist.get_rank() == 0:
                _ce = loss_dict.get("ce")
                _mse = loss_dict.get("mse")
                _rank_finite = [torch.zeros_like(_loss_is_finite_local) for _ in range(dist.get_world_size())]
                dist.all_gather(_rank_finite, _loss_is_finite_local)
                _bad_ranks = [i for i, _v in enumerate(_rank_finite) if int(_v.item()) == 0]
                logger.warning(
                    f"[DBG] Step {curr_step}: non-finite loss detected before backward "
                    f"(loss={loss}, ce={_ce}, mse={_mse}, bad_ranks={_bad_ranks[:16]}); skipping step"
                )
            else:
                # Keep collectives symmetric with rank0 branch above.
                _rank_finite = [torch.zeros_like(_loss_is_finite_local) for _ in range(dist.get_world_size())]
                dist.all_gather(_rank_finite, _loss_is_finite_local)
            optimizer.zero_grad(set_to_none=True)
            continue

        # Zero grads only at the start of each accumulation cycle.
        if accum_idx == 0:
            _stage_mark("before_zero_grad")
            optimizer.zero_grad()

        # Scale loss so accumulated gradient magnitude matches a single-step update.
        if num_accum > 1:
            loss = loss / num_accum

        _stage_mark("before_backward")
        # Skip FSDP reduce-scatter on all but the last micro-step. no_sync keeps the
        # full UNSHARDED grads in memory across the accumulation window — with a fully
        # unfrozen 77B model + large grad_accum this OOMs. HUNYUAN_GRAD_ACCUM_NO_SYNC=0
        # reduce-scatters every micro-step instead (only sharded grads kept; correct,
        # just more comm) so big-batch full-FT ablations fit.
        _grad_accum_no_sync = os.environ.get("HUNYUAN_GRAD_ACCUM_NO_SYNC", "1") == "1"
        _sync_ctx = (
            contextlib.nullcontext()
            if (is_last_accum or not _grad_accum_no_sync)
            else fsdp_model.no_sync()
        )
        with _sync_ctx:
            loss.backward()
        _stage_mark("after_backward")
        torch.autograd.set_detect_anomaly(False)

        # On non-last micro-steps: accumulate stats and move on without optimizer step.
        if not is_last_accum:
            batch_tokens = torch.tensor(data["sequence_length"], device=device)
            dist.all_reduce(batch_tokens, op=dist.ReduceOp.SUM)
            total_tokens_accumulated += batch_tokens.item()
            total_seq_tokens_accumulated += data_args.max_num_tokens * dist.get_world_size()
            batch_samples = torch.tensor(batch_num_samples, device=device)
            dist.all_reduce(batch_samples, op=dist.ReduceOp.SUM)
            total_samples_accumulated += batch_samples.item()
            batch_epoch_samples = torch.tensor(batch_num_epoch_samples, device=device)
            dist.all_reduce(batch_epoch_samples, op=dist.ReduceOp.SUM)
            total_epoch_samples_accumulated += batch_epoch_samples.item()
            if data_status is None:
                data_status = {}
            for item in data_indexes:
                if item["dataset_name"] not in data_status:
                    data_status[item["dataset_name"]] = {}
                data_status[item["dataset_name"]][item["worker_id"]] = item["data_indexes"]
            if isinstance(batch_data_resume_state, dict):
                _wid = batch_data_resume_state.get("worker_id")
                if _wid is not None:
                    data_resume_state[_wid] = batch_data_resume_state
            continue

        disable_grad_guard = os.environ.get("HUNYUAN_DISABLE_GRAD_GUARD", "0") == "1"
        nan_grad_params = []
        if not disable_grad_guard:
            # IMPORTANT: detect/sanitize non-finite grads BEFORE clip_grad_norm_.
            # If clip_grad_norm_ sees NaN/Inf in any grad, it can propagate NaN into
            # additional gradients, which hides the true source and stalls training.
            nan_grad_groups = {
                "vision_model": 0,
                "vision_aligner": 0,
                "timestep_emb": 0,
                "other": 0,
            }
            nan_grad_types = {
                "self_attn.qkv_proj": 0,
                "self_attn.o_proj": 0,
                "mlp": 0,
                "gate": 0,
                "other": 0,
            }
            nan_grad_layers: dict[int, int] = {}
            has_nonfinite_grad_local = torch.tensor(0, device=device, dtype=torch.int32)
            local_nonfinite_names = []
            for _dn, _dp in fsdp_model.named_parameters():
                if _dp.grad is None:
                    continue
                if _dp.grad.isnan().any() or _dp.grad.isinf().any():
                    has_nonfinite_grad_local.fill_(1)
                    if curr_step < train_step + 5:
                        local_nonfinite_names.append(_dn)
                    if dist.get_rank() == 0 and curr_step < train_step + 5:
                        nan_grad_params.append(_dn)
                        if ".vision_model." in _dn:
                            nan_grad_groups["vision_model"] += 1
                        elif ".vision_aligner." in _dn:
                            nan_grad_groups["vision_aligner"] += 1
                        elif ".timestep_emb." in _dn:
                            nan_grad_groups["timestep_emb"] += 1
                        else:
                            nan_grad_groups["other"] += 1
                        if ".self_attn.qkv_proj." in _dn:
                            nan_grad_types["self_attn.qkv_proj"] += 1
                        elif ".self_attn.o_proj." in _dn:
                            nan_grad_types["self_attn.o_proj"] += 1
                        elif ".mlp." in _dn:
                            nan_grad_types["mlp"] += 1
                        elif ".gate.wg." in _dn:
                            nan_grad_types["gate"] += 1
                        else:
                            nan_grad_types["other"] += 1
                        _m_layer = re.search(r"\.model\.layers\.(\d+)\.", _dn)
                        if _m_layer is not None:
                            _li = int(_m_layer.group(1))
                            nan_grad_layers[_li] = nan_grad_layers.get(_li, 0) + 1

            has_nonfinite_grad = has_nonfinite_grad_local.clone()
            dist.all_reduce(has_nonfinite_grad, op=dist.ReduceOp.MAX)

            gathered_nonfinite = None
            if curr_step < train_step + 5 and has_nonfinite_grad.item() > 0:
                # Collective must be called on all ranks.
                gathered_nonfinite = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered_nonfinite, local_nonfinite_names[:64])

            if dist.get_rank() == 0 and curr_step < train_step + 5:
                if curr_step == train_step:
                    logger.info(
                        f"[DBG] Step {curr_step}: batch shape summary: "
                        f"sequence_length={data.get('sequence_length')}, "
                        f"num_samples={len(data.get('sample_lens', []))}, "
                        f"num_vit_tokens={int(data['packed_vit_tokens'].shape[0]) if 'packed_vit_tokens' in data else 0}, "
                        f"num_vae_tokens={int(data['packed_vae_token_indexes'].numel()) if 'packed_vae_token_indexes' in data else 0}"
                    )
                if nan_grad_params:
                    logger.warning(f"[DBG] Step {curr_step}: NaN/Inf grad in {len(nan_grad_params)} params: {nan_grad_params[:3]}")
                    logger.warning(f"[DBG] Step {curr_step}: NaN/Inf grad groups: {nan_grad_groups}")
                    logger.warning(f"[DBG] Step {curr_step}: NaN/Inf grad types: {nan_grad_types}")
                    if nan_grad_layers:
                        logger.warning(
                            f"[DBG] Step {curr_step}: NaN/Inf grad layer histogram: "
                            f"{dict(sorted(nan_grad_layers.items()))}"
                        )
                # Cross-rank diagnostic: rank 0 may not hold the offending shards.
                if gathered_nonfinite is not None:
                    cross_rank_hits = {
                        r: names for r, names in enumerate(gathered_nonfinite) if names
                    }
                    if cross_rank_hits:
                        # Log compactly: first 3 names per rank.
                        compact = {r: v[:3] for r, v in cross_rank_hits.items()}
                        logger.warning(
                            f"[DBG] Step {curr_step}: cross-rank non-finite grad shards "
                            f"on ranks={sorted(cross_rank_hits.keys())}, sample_names={compact}"
                        )

            if has_nonfinite_grad.item() > 0:
                # Pragmatic guardrail: sanitize non-finite grads and retry this step.
                # This avoids getting permanently stuck at step-0 when a subset of
                # gradients are NaN/Inf with otherwise-finite forward losses.
                sanitize_nonfinite = os.environ.get("HUNYUAN_SANITIZE_NONFINITE_GRAD", "1") == "1"
                if sanitize_nonfinite:
                    repaired_local = torch.tensor(0, device=device, dtype=torch.int32)
                    for _p in fsdp_model.parameters():
                        if _p.grad is None:
                            continue
                        if not torch.isfinite(_p.grad).all():
                            torch.nan_to_num(_p.grad, nan=0.0, posinf=0.0, neginf=0.0, out=_p.grad)
                            repaired_local += 1

                    repaired_total = repaired_local.clone()
                    dist.all_reduce(repaired_total, op=dist.ReduceOp.SUM)

                    # Re-check non-finite grads after repair.
                    nonfinite_after_local = torch.tensor(0, device=device, dtype=torch.int32)
                    for _p in fsdp_model.parameters():
                        if _p.grad is not None and not torch.isfinite(_p.grad).all():
                            nonfinite_after_local.fill_(1)
                            break
                    dist.all_reduce(nonfinite_after_local, op=dist.ReduceOp.MAX)

                    if dist.get_rank() == 0:
                        logger.warning(
                            f"[DBG] Step {curr_step}: non-finite grads repaired "
                            f"(repaired_param_shards={int(repaired_total.item())})"
                        )

                    if nonfinite_after_local.item() > 0:
                        if dist.get_rank() == 0:
                            logger.warning(
                                f"[DBG] Step {curr_step}: grads still non-finite after repair; "
                                "skipping optimizer/scheduler/EMA step"
                            )
                        optimizer.zero_grad(set_to_none=True)
                        continue
                else:
                    if dist.get_rank() == 0:
                        logger.warning(
                            f"[DBG] Step {curr_step}: non-finite grad_norm detected; "
                            "skipping optimizer/scheduler/EMA step"
                        )
                    optimizer.zero_grad(set_to_none=True)
                    continue

            # Clip only after non-finite gradients were repaired/cleared.
            total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
            # clip_grad_norm_ can return inf with all-finite grads when norm
            # accumulation overflows (very large but finite gradients). In that case
            # clipping would effectively zero all grads (coef=0). Clamp first, then
            # re-clip to keep updates meaningful.
            if not torch.isfinite(total_norm):
                has_nonfinite_value_local = torch.tensor(0, device=device, dtype=torch.int32)
                for _p in fsdp_model.parameters():
                    if _p.grad is not None and (not torch.isfinite(_p.grad).all()):
                        has_nonfinite_value_local.fill_(1)
                        break
                has_nonfinite_value = has_nonfinite_value_local.clone()
                dist.all_reduce(has_nonfinite_value, op=dist.ReduceOp.MAX)

                if has_nonfinite_value.item() == 0:
                    _grad_abs_clamp = float(os.environ.get("HUNYUAN_GRAD_ABS_CLAMP", "1000.0"))
                    for _p in fsdp_model.parameters():
                        if _p.grad is None:
                            continue
                        _g = _p.grad
                        if _g.dtype in (torch.bfloat16, torch.float16):
                            _tmp = _g.float().clamp_(min=-_grad_abs_clamp, max=_grad_abs_clamp)
                            _g.copy_(_tmp.to(_g.dtype))
                        else:
                            _g.clamp_(min=-_grad_abs_clamp, max=_grad_abs_clamp)
                    total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
                    if dist.get_rank() == 0:
                        logger.warning(
                            f"[DBG] Step {curr_step}: grad_norm overflow repaired by grad clamp "
                            f"(abs_clamp={_grad_abs_clamp}, post_repair_grad_norm={total_norm})"
                        )
        else:
            if dist.get_rank() == 0 and curr_step == train_step:
                logger.warning("[DBG] HUNYUAN_DISABLE_GRAD_GUARD=1: non-finite grad guard is disabled")
            total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)

        if dist.get_rank() == 0 and curr_step < train_step + 5:
            logger.info(f"[DBG] Step {curr_step}: grad_norm={total_norm:.4f}, nan_grads={len(nan_grad_params)}")

        _stage_mark("before_optimizer_step")
        optimizer.step()
        _stage_mark("after_optimizer_step")
        scheduler.step()
        if ema_model is not None:
            fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)
        opt_step += 1

        # ─────── Online Validation (no-op if validation_pack_path is unset) ───
        _val_every = int(getattr(training_args, "validate_every", 0) or 0)
        if (
            _val_every > 0
            and _online_val_pack is not None
            and opt_step > 0
            and opt_step % _val_every == 0
        ):
            try:
                from train.online_validation import run_online_validation as _run_online_val
                _val_metrics = _run_online_val(
                    fsdp_model=fsdp_model,
                    vae_model=vae_model,
                    val_pack=_online_val_pack,
                    inferencer=_online_val_inferencer,
                    dino_tokenizer=_online_val_dino_tokenizer,
                    step=opt_step,
                    device=device,
                    logger=logger,
                )
                if _val_metrics is not None and dist.get_rank() == 0:
                    try:
                        wandb.log(
                            {f"val/{k}": v for k, v in _val_metrics.items()},
                            step=opt_step,
                        )
                    except Exception as _wbe:
                        logger.warning(f"[online_val] wandb log failed: {_wbe}")
            except Exception as _vae:
                # Validation must never break training. Log + continue.
                if dist.get_rank() == 0:
                    logger.warning(f"[online_val] step {opt_step}: hook raised, skipping: {_vae}")

        # Accumulate stats
        batch_tokens = torch.tensor(data["sequence_length"], device=device)
        dist.all_reduce(batch_tokens, op=dist.ReduceOp.SUM)
        total_tokens_accumulated += batch_tokens.item()
        total_seq_tokens_accumulated += data_args.max_num_tokens * dist.get_world_size()

        batch_samples = torch.tensor(batch_num_samples, device=device)
        dist.all_reduce(batch_samples, op=dist.ReduceOp.SUM)
        total_samples_accumulated += batch_samples.item()

        batch_epoch_samples = torch.tensor(batch_num_epoch_samples, device=device)
        dist.all_reduce(batch_epoch_samples, op=dist.ReduceOp.SUM)
        total_epoch_samples_accumulated += batch_epoch_samples.item()

        # ─────── Bench: memory + GPU utilisation at step 1 ───────────────────
        if opt_step == 1 and dist.get_rank() == 0:
            logger.info(
                f"[BENCH] mem_allocated={torch.cuda.max_memory_allocated()/1e9:.1f}GB "
                f"mem_reserved={torch.cuda.max_memory_reserved()/1e9:.1f}GB "
                f"gpu_util={torch.cuda.utilization()}%"
            )

        # ─────── Logging ──────────────────────────────────────────────────────
        if opt_step % training_args.log_every == 0:
            start_time = log_training_step(
                opt_step, loss_dict, training_args, optimizer,
                total_tokens_accumulated, total_seq_tokens_accumulated,
                total_samples_accumulated, total_epoch_samples_accumulated,
                total_dataset_samples,
                total_ce_tokens, total_mse_tokens, total_norm,
                start_time, logger, device,
                tokens_per_interval=total_tokens_accumulated - _tokens_at_last_log,
                samples_per_interval=total_samples_accumulated - _samples_at_last_log,
                model_flops_per_token=model_flops_per_token,
            )
            _tokens_at_last_log = total_tokens_accumulated
            _samples_at_last_log = total_samples_accumulated

        # Track data status for resumption
        if data_status is None:
            data_status = {}
        for item in data_indexes:
            if item["dataset_name"] not in data_status:
                data_status[item["dataset_name"]] = {}
            data_status[item["dataset_name"]][item["worker_id"]] = item["data_indexes"]
        if isinstance(batch_data_resume_state, dict):
            _wid = batch_data_resume_state.get("worker_id")
            if _wid is not None:
                data_resume_state[_wid] = batch_data_resume_state
                # Advance data_status past buffer samples so that on resume the
                # sub-dataset iterator starts AFTER the last buffered position,
                # preventing buffer samples from being re-read and duplicated.
                # data_indexes format varies by dataset: list of 3 (t2i), list
                # of 2 (interleave_t2i), or a plain int (vlm) — normalise to
                # tuple for a uniform "later position" comparison.
                def _pos_key(v):
                    return tuple(v) if isinstance(v, (list, tuple)) else (v,)
                for _buf_sample in batch_data_resume_state.get("buffer", []):
                    _buf_idx = _buf_sample.get("data_indexes", {})
                    _dname = _buf_idx.get("dataset_name")
                    _bw    = _buf_idx.get("worker_id")
                    _pos   = _buf_idx.get("data_indexes")
                    if _dname is None or _bw is None or _pos is None:
                        continue
                    if _dname not in data_status:
                        data_status[_dname] = {}
                    _existing = data_status[_dname].get(_bw)
                    if _existing is None or _pos_key(_pos) > _pos_key(_existing):
                        data_status[_dname][_bw] = _pos

        # ─────── Checkpoint ───────────────────────────────────────────────────
        if opt_step > 0 and opt_step % training_args.save_every == 0:
            dist.barrier()
            FSDPCheckpoint.fsdp_save_ckpt(
                ckpt_dir=training_args.checkpoint_dir,
                train_steps=opt_step,
                model=fsdp_model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                data_status=data_status,
                data_resume_state=data_resume_state,
                logger=logger,
                fsdp_config=fsdp_config,
                training_stats={
                    "total_tokens_accumulated": total_tokens_accumulated,
                    "total_seq_tokens_accumulated": total_seq_tokens_accumulated,
                    "total_samples_accumulated": total_samples_accumulated,
                    "total_epoch_samples_accumulated": total_epoch_samples_accumulated,
                },
            )
            gc.collect()
            torch.cuda.empty_cache()

        # Periodic GC
        if opt_step > 0 and opt_step % 1000 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        if _prof is not None:
            _prof.step()

    if _prof is not None:
        _prof.stop()
        logger.info(f"[PROFILE] trace written to {training_args.results_dir}/profile/")

    # ─────── Done ──────────────────────────────────────────────────────────────
    logger.info("Training complete!")
    if dist.get_rank() == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
