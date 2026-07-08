# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import os

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from safetensors.torch import load_file, save_file


def _fsdp_param_init_fn(module: torch.nn.Module) -> None:
    """
    param_init_fn for FSDP meta-device materialization.

    FSDP calls reset_parameters() on each module when materialising from meta
    device. torch.nn.MultiheadAttention (and some other PyTorch built-ins) only
    define _reset_parameters (private) instead of the public reset_parameters,
    which causes AttributeError. This shim handles both cases.
    """
    device_id = dist.get_rank() % torch.cuda.device_count()
    module.to_empty(device=f"cuda:{device_id}")
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
    elif hasattr(module, "_reset_parameters"):
        module._reset_parameters()


class FSDPConfig:
    def __init__(
        self,
        sharding_strategy,
        backward_prefetch,
        cpu_offload,
        num_replicate,
        num_shard=8,
    ):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard


def fsdp_wrapper(
    original_model,
    fsdp_config,
    transformer_layer_cls,
    ignored_modules=[],
    sync_module_states=False,
    dp_mesh=None,
):
    if fsdp_config.sharding_strategy == 'HYBRID_SHARD':
        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard")
        )
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        device_mesh = None
        sharding_strategy = ShardingStrategy[fsdp_config.sharding_strategy]

    # Only use param_init_fn when the model has meta-device parameters (random
    # init / debug mode).  For normal pretrained models the parameters are
    # already on a real device and param_init_fn is not needed.
    has_meta = any(p.device.type == "meta" for p in original_model.parameters())
    param_init_fn = _fsdp_param_init_fn if has_meta else None

    return FSDP(
        original_model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_layer_cls,
        ),
        ignored_modules=ignored_modules,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16 if os.environ.get("HUNYUAN_REDUCE_DTYPE", "fp32") == "bf16" else torch.float32,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=sharding_strategy,
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        forward_prefetch=True,  # overlap next all-gather with current forward compute
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
        param_init_fn=param_init_fn,
        sync_module_states=sync_module_states,
        use_orig_params=False,
    )


class FSDPCheckpoint:
    @staticmethod
    def fsdp_save_ckpt(
        ckpt_dir, 
        train_steps, 
        model, 
        ema_model, 
        optimizer, 
        scheduler, 
        data_status,
        data_resume_state,
        logger, 
        fsdp_config,
        training_stats=None,
    ):
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving checkpoint to {save_path}.")

        def _get_fsdp_state_dict_with_fallback(fsdp_module, prefer_local: bool, module_label: str):
            preferred = StateDictType.LOCAL_STATE_DICT if prefer_local else StateDictType.SHARDED_STATE_DICT
            fallback = StateDictType.SHARDED_STATE_DICT if prefer_local else StateDictType.LOCAL_STATE_DICT
            try:
                with FSDP.state_dict_type(fsdp_module, preferred):
                    return fsdp_module.state_dict(), preferred
            except RuntimeError as e:
                # TP+FSDP DeviceMesh can reject LOCAL_STATE_DICT even when sharding_strategy is FULL_SHARD.
                if preferred == StateDictType.LOCAL_STATE_DICT and "DeviceMesh is not compatible with LOCAL_STATE_DICT" in str(e):
                    logger.warning(
                        f"{module_label}: LOCAL_STATE_DICT is incompatible with DeviceMesh; "
                        "falling back to SHARDED_STATE_DICT for checkpoint save."
                    )
                    with FSDP.state_dict_type(fsdp_module, fallback):
                        return fsdp_module.state_dict(), fallback
                raise

        # Save model and EMA using sharded approach to reduce memory pressure
        if ema_model is not None:
            if dist.get_rank() == 0:
                logger.info(f"Rank 0 memory before EMA save: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            
            # Determine shard configuration for EMA
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                ema_shard_index = dist.get_rank()
                ema_total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                ema_shard_index = dist.get_rank() % fsdp_config.num_shard
                ema_total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError
            
            # Save EMA model shard
            ema_raw_dict, _ = _get_fsdp_state_dict_with_fallback(
                ema_model,
                prefer_local=(fsdp_config.sharding_strategy == "FULL_SHARD"),
                module_label="ema_model",
            )
            # Convert DTensors to regular tensors for safetensors compatibility
            ema_regular_dict = {}
            for key, value in ema_raw_dict.items():
                if hasattr(value, 'to_local'):
                    # Convert DTensor to local tensor
                    ema_regular_dict[key] = value.to_local()
                else:
                    ema_regular_dict[key] = value
            ema_save_path = os.path.join(save_path, f"ema.{ema_shard_index:05d}-of-{ema_total_shards:05d}.safetensors")
            # Use safetensors for more reliable serialization
            from safetensors.torch import save_file
            save_file(ema_regular_dict, ema_save_path)
            
            if dist.get_rank() == 0:
                logger.info(f"Rank 0 memory after EMA save: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            del ema_raw_dict
            del ema_regular_dict
        torch.cuda.empty_cache()
        if dist.get_rank() == 0:
            logger.info(f"Rank 0 memory after EMA cleanup: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

        # Save main model shard
        # Determine shard configuration for model
        if fsdp_config.sharding_strategy == "FULL_SHARD":
            model_shard_index = dist.get_rank()
            model_total_shards = dist.get_world_size()
        elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
            model_shard_index = dist.get_rank() % fsdp_config.num_shard
            model_total_shards = fsdp_config.num_shard
        else:
            raise NotImplementedError

        model_raw_dict, _ = _get_fsdp_state_dict_with_fallback(
            model,
            prefer_local=(fsdp_config.sharding_strategy == "FULL_SHARD"),
            module_label="model",
        )
        # Convert DTensors to regular tensors for safetensors compatibility
        model_regular_dict = {}
        for key, value in model_raw_dict.items():
            if hasattr(value, 'to_local'):
                # Convert DTensor to local tensor
                model_regular_dict[key] = value.to_local()
            else:
                model_regular_dict[key] = value
        model_save_path = os.path.join(save_path, f"model.{model_shard_index:05d}-of-{model_total_shards:05d}.safetensors")
        # Use safetensors for more reliable serialization
        from safetensors.torch import save_file
        save_file(model_regular_dict, model_save_path)
        
        if dist.get_rank() == 0:
            logger.info(f"Rank 0 memory after main model save: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        del model_raw_dict
        del model_regular_dict
        torch.cuda.empty_cache()
        if dist.get_rank() == 0:
            logger.info(f"Rank 0 memory after main model cleanup: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

        # Determine shard configuration for optimizer
        if fsdp_config.sharding_strategy == "FULL_SHARD":
            optimizer_shard_index = dist.get_rank()
            optimizer_total_shards = dist.get_world_size()
            _opt_should_save = True
        elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
            optimizer_shard_index = dist.get_rank() % fsdp_config.num_shard
            optimizer_total_shards = fsdp_config.num_shard
            _opt_should_save = dist.get_rank() < fsdp_config.num_shard
        else:
            raise NotImplementedError

        optimizer_save_path = os.path.join(
            save_path, f"optimizer.{optimizer_shard_index:05d}-of-{optimizer_total_shards:05d}.pt"
        )
        optimizer_state_dict_type = StateDictType.LOCAL_STATE_DICT
        if fsdp_config.sharding_strategy == "HYBRID_SHARD":
            optimizer_state_dict_type = StateDictType.SHARDED_STATE_DICT
        try:
            with FSDP.state_dict_type(model, optimizer_state_dict_type):
                if fsdp_config.sharding_strategy == "FULL_SHARD":
                    torch.save(optimizer.state_dict(), optimizer_save_path)
                elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                    if _opt_should_save:
                        torch.save(optimizer.state_dict(), optimizer_save_path)
                else:
                    raise NotImplementedError
        except RuntimeError as e:
            if (
                optimizer_state_dict_type == StateDictType.LOCAL_STATE_DICT
                and "DeviceMesh is not compatible with LOCAL_STATE_DICT" in str(e)
            ):
                logger.warning(
                    "optimizer: LOCAL_STATE_DICT is incompatible with DeviceMesh; "
                    "falling back to SHARDED_STATE_DICT for checkpoint save."
                )
                with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                    if fsdp_config.sharding_strategy == "FULL_SHARD":
                        torch.save(optimizer.state_dict(), optimizer_save_path)
                    elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                        if _opt_should_save:
                            torch.save(optimizer.state_dict(), optimizer_save_path)
                    else:
                        raise NotImplementedError
            else:
                raise

        if dist.get_rank() == 0 and scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

        if data_status is not None:
            torch.save(data_status, os.path.join(save_path, f"data_status.rank{dist.get_rank()}.pt"))
        if data_resume_state is not None:
            torch.save(
                data_resume_state,
                os.path.join(save_path, f"data_resume_state.rank{dist.get_rank()}.pt"),
            )

        if training_stats is not None and dist.get_rank() == 0:
            torch.save(training_stats, os.path.join(save_path, "training_stats.pt"))

        dist.barrier()
        logger.info("Checkpoint saved.")
        return

    @staticmethod
    def try_load_ckpt(resume_from, logger, model, ema_model=None, resume_from_ema=False):
        if resume_from is not None and os.path.exists(resume_from):
            logger.info(f"Loading checkpoint from {resume_from}.")
            if resume_from_ema:
                model_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
            else:
                model_state_dict_path = os.path.join(resume_from, f"model.safetensors")
            model_state_dict = load_file(model_state_dict_path, device="cpu")
            # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
            # which makes it easier to adapt to different resolutions.
            model_state_dict.pop('latent_pos_embed.pos_embed', None)
            model_state_dict.pop('vit_pos_embed.pos_embed', None)
            msg = model.load_state_dict(model_state_dict, strict=False)
            logger.info(msg)
            del model_state_dict

            if ema_model is not None:
                ema_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
                if not os.path.exists(ema_state_dict_path):
                    logger.info(f"replicaing ema model from {model_state_dict_path}.")
                    ema_state_dict_path = model_state_dict_path
                ema_state_dict = load_file(ema_state_dict_path, device="cpu")
                # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
                # which makes it easier to adapt to different resolutions.
                ema_state_dict.pop('latent_pos_embed.pos_embed', None)
                ema_state_dict.pop('vit_pos_embed.pos_embed', None)
                msg = ema_model.load_state_dict(ema_state_dict, strict=False)
                logger.info(msg)
                del ema_state_dict
        else:
            logger.info(f"Training from scratch.")
        return model, ema_model

    @staticmethod
    def get_system_memory_usage_gb():
        meminfo = {}
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                parts = line.split()
                meminfo[parts[0].rstrip(':')] = int(parts[1])
        mem_total = meminfo['MemTotal']  # in kB
        mem_available = meminfo['MemAvailable']  # in kB
        mem_used = mem_total - mem_available
        return mem_used / 1024 / 1024  # Convert kB to GB

    @staticmethod
    def try_load_ckpt_after_fsdp(resume_from, logger, fsdp_model, ema_model=None, resume_from_ema=False, model_name='ema.special_token_patched.safetensors"'):
        """Load checkpoint after FSDP wrapping when model is on actual device, handling resized embeddings."""

        def load_full_state_dict(model, state_dict_path, model_label="model"):
            # Model stays on CUDA!
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                logger.info(f"Loading {model_label} state dict from {state_dict_path}")
                state_dict = load_file(state_dict_path, device="cpu")
                # Fixed sin-cos position embeddings should be regenerated from the
                # current config rather than restored from a checkpoint with a
                # different grid size.
                state_dict.pop("latent_pos_embed.pos_embed", None)
                state_dict.pop("vit_pos_embed.pos_embed", None)
                msg = model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded checkpoint into FSDP {model_label}: {msg}")
            return model

        def load_sharded_state_dict(model, checkpoint_dir, model_label="model"):
            """Load sharded state dict for FSDP model."""
            def _wrap_plain_tensors_for_sharded_load(state_dict, template_state_dict):
                wrapped = {}
                wrapped_count = 0
                for key, value in state_dict.items():
                    template_value = template_state_dict.get(key)
                    if isinstance(value, DTensor) or not isinstance(template_value, DTensor):
                        wrapped[key] = value
                        continue
                    if not isinstance(value, torch.Tensor):
                        wrapped[key] = value
                        continue

                    # Match the exact DTensor metadata expected by the current
                    # wrapped model instead of guessing mesh / placements.
                    wrapped[key] = DTensor.from_local(
                        value,
                        template_value.device_mesh,
                        template_value.placements,
                        shape=tuple(template_value.shape),
                        stride=tuple(template_value.stride()),
                        run_check=False,
                    )
                    wrapped_count += 1

                logger.info(
                    f"Wrapped {wrapped_count} plain tensors as DTensors for {model_label} "
                    f"SHARDED_STATE_DICT load."
                )
                return wrapped

            # Opt-in escape hatch: when the saved sharded checkpoint was written
            # with a different num_shard than the current FSDP setup (e.g.
            # loading an 8-shard training ckpt onto a 4-GPU validation node),
            # skip sharded loading entirely and let the caller fall back to the
            # consolidated `model.safetensors`. Set MODUS_FORCE_FULL_STATE_LOAD=1.
            if os.environ.get("MODUS_FORCE_FULL_STATE_LOAD", "0") == "1":
                logger.info(
                    f"MODUS_FORCE_FULL_STATE_LOAD=1: skipping sharded {model_label} "
                    f"load in {checkpoint_dir}, falling back to full state dict."
                )
                return None

            # Support both legacy torch-saved shards and safetensors shards.
            shard_files = [
                f for f in os.listdir(checkpoint_dir)
                if f.startswith(f"{model_label}.") and (f.endswith(".pt") or f.endswith(".safetensors"))
            ]
            if not shard_files:
                return None

            shard_files.sort()

            # Parse shard info from filename (e.g., "model.00000-of-00008.pt")
            sample_file = shard_files[0]
            shard_info = sample_file.split(".")[1]  # "00000-of-00008"
            if "-of-" not in shard_info:
                # Monolithic safetensors (e.g. "ema.special_token_patched....safetensors"),
                # not a sharded checkpoint — let caller fall back to load_full_state_dict.
                return None
            shard_index, total_shards = map(int, shard_info.split("-of-"))

            shard_ext = ".safetensors" if sample_file.endswith(".safetensors") else ".pt"
            local_rank_for_shard = dist.get_rank() % total_shards
            local_shard_path = os.path.join(
                checkpoint_dir,
                f"{model_label}.{local_rank_for_shard:05d}-of-{total_shards:05d}{shard_ext}",
            )
            if not os.path.exists(local_shard_path):
                logger.warning(
                    f"Local shard {local_shard_path} not found, falling back to full state dict"
                )
                return None

            if shard_ext == ".safetensors":
                local_dict = load_file(local_shard_path, device="cpu")
            else:
                local_dict = torch.load(local_shard_path, map_location="cpu")

            # Fixed sin-cos position embeddings should follow the current model
            # geometry, not the checkpoint geometry.
            local_dict.pop("latent_pos_embed.pos_embed", None)
            local_dict.pop("vit_pos_embed.pos_embed", None)

            # These per-rank shard files contain plain tensors, so LOCAL_STATE_DICT
            # is the correct resume path for FULL_SHARD checkpoints. Some torch
            # builds also reject SHARDED_STATE_DICT here unless tensors carry
            # DTensor/device_mesh metadata.
            local_exc = None
            try:
                with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
                    logger.info(f"Loading {model_label} shard from {local_shard_path} with LOCAL_STATE_DICT")
                    msg = model.load_state_dict(local_dict, strict=False)
                    logger.info(f"Loaded sharded checkpoint into FSDP {model_label}: {msg}")
                    return model
            except (RuntimeError, AttributeError, AssertionError) as e:
                local_exc = e
                logger.warning(
                    f"LOCAL_STATE_DICT load for {model_label} shard failed: {e}. "
                    "Retrying with SHARDED_STATE_DICT."
                )

            try:
                with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                    logger.info(f"Loading {model_label} shard from {local_shard_path} with SHARDED_STATE_DICT")
                    template_state_dict = model.state_dict()
                    sharded_dict = _wrap_plain_tensors_for_sharded_load(local_dict, template_state_dict)
                    del template_state_dict
                    msg = model.load_state_dict(sharded_dict, strict=False)
                    logger.info(f"Loaded sharded checkpoint into FSDP {model_label}: {msg}")
                    return model
            except (RuntimeError, AttributeError, AssertionError) as e:
                if local_exc is not None:
                    logger.error(
                        f"Both LOCAL_STATE_DICT and SHARDED_STATE_DICT failed for {model_label} shard "
                        f"{local_shard_path}. local_error={local_exc}; sharded_error={e}"
                    )
                raise

        if resume_from is not None and os.path.exists(resume_from):
            logger.info(f"Loading checkpoint from {resume_from} after FSDP wrapping.")
            
            # Try sharded loading first, fall back to full state dict
            model_loaded = load_sharded_state_dict(fsdp_model, resume_from, "model")
            if model_loaded is None:
                # Fall back to full state dict loading
                if resume_from_ema:
                    model_state_dict_path = os.path.join(resume_from, f"{model_name}")
                    logger.info(f"Loading {model_name} from {model_state_dict_path}")
                else:
                    model_state_dict_path = os.path.join(resume_from, f"model.safetensors")
                logger.info(f"System memory usage before loading checkpoint: {FSDPCheckpoint.get_system_memory_usage_gb()} GB")
                fsdp_model = load_full_state_dict(fsdp_model, model_state_dict_path, model_label="model")
                logger.info(f"System memory usage after loading checkpoint: {FSDPCheckpoint.get_system_memory_usage_gb()} GB")
            
            if ema_model is not None:
                ema_loaded = load_sharded_state_dict(ema_model, resume_from, "ema")
                if ema_loaded is None:
                    # Fall back to full state dict loading
                    if resume_from_ema:
                        ema_state_dict_path = os.path.join(resume_from, f"{model_name}")
                        if not os.path.exists(ema_state_dict_path):
                            ema_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
                    else:
                        ema_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
                    if not os.path.exists(ema_state_dict_path):
                        logger.info(f"Replicating ema model from {model_state_dict_path}.")
                        ema_state_dict_path = model_state_dict_path
                    logger.info(f"System memory usage before loading ema checkpoint: {FSDPCheckpoint.get_system_memory_usage_gb()} GB")
                    ema_model = load_full_state_dict(ema_model, ema_state_dict_path, model_label="ema")
                    logger.info(f"System memory usage after loading ema checkpoint: {FSDPCheckpoint.get_system_memory_usage_gb()} GB")
        else:
            logger.info(f"Training from scratch.")
        return fsdp_model, ema_model

    @staticmethod
    def load_hf_weights_before_fsdp(
        model,
        model_path: str,
        prefix: str = "hunyuan_model.",
        skip_prefixes: tuple = ("vae.",),
        load_all_ranks: bool = False,
        logger=None,
    ):
        """Load HF safetensors into a plain (pre-FSDP) model.

        All ranks load from disk in parallel, one shard at a time.
        Peak extra RAM = one shard (~500 MB) per rank — negligible compared to
        the model already held in CPU RAM before FSDP wrapping.
        FSDP wrap afterwards naturally shards the initialized weights correctly.
        """
        import gc
        import glob
        import json
        from safetensors.torch import load_file

        def _log(msg):
            if logger is not None and dist.get_rank() == 0:
                logger.info(msg)

        is_rank0 = dist.get_rank() == 0
        # Optional all-ranks loading path: materialize/load on every rank so FSDP
        # wrapping can skip sync_module_states broadcast (reduces GPU init peak).
        # Default remains rank0-only for lower host-memory use.
        is_meta = any(p.device.type == "meta" for p in model.parameters())
        if is_meta:
            if load_all_ranks or is_rank0:
                _log(
                    "Model is on meta device; "
                    + ("all ranks" if load_all_ranks else "rank 0")
                    + " materializing to CPU for pre-FSDP HF loading"
                )
                model.to_empty(device="cpu")
                hf_submodel = getattr(model, "hunyuan_model", model)
                if hasattr(hf_submodel, "init_weights"):
                    hf_submodel.init_weights()
                else:
                    for m in model.modules():
                        if hasattr(m, "reset_parameters"):
                            m.reset_parameters()
            else:
                # Non-rank-0 stays on meta until rank 0 finishes HF load.
                dist.barrier()
                return

        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                shard_names = sorted(set(json.load(f).get("weight_map", {}).values()))
            shard_paths = [os.path.join(model_path, s) for s in shard_names]
        else:
            shard_paths = sorted(glob.glob(os.path.join(model_path, "model-*.safetensors")))
            single = os.path.join(model_path, "model.safetensors")
            if not shard_paths and os.path.exists(single):
                shard_paths = [single]

        _log(f"Loading HF weights (pre-FSDP) from {len(shard_paths)} shard(s) in {model_path}")

        param_dict = dict(model.named_parameters())
        loaded, skipped = 0, 0
        for shard_path in shard_paths:
            shard = load_file(shard_path, device="cpu")
            for hf_key, tensor in shard.items():
                if any(hf_key.startswith(p) for p in skip_prefixes):
                    continue
                fsdp_key = prefix + hf_key
                if fsdp_key not in param_dict:
                    skipped += 1
                    continue
                param = param_dict[fsdp_key]
                t = tensor.to(dtype=param.dtype)
                if tuple(t.shape) == tuple(param.shape):
                    param.data.copy_(t)
                    loaded += 1
                elif t.ndim == param.ndim and t.shape[1:] == param.shape[1:]:
                    # Dim-0 mismatch (e.g. vocab size): truncate or zero-pad
                    src, tgt = t.shape[0], param.shape[0]
                    if src >= tgt:
                        param.data.copy_(t[:tgt])
                    else:
                        param.data[:src].copy_(t)
                        param.data[src:].zero_()
                    loaded += 1
                else:
                    skipped += 1
            del shard
            gc.collect()

        _log(f"Pre-FSDP HF load: {loaded} parameters loaded, {skipped} skipped")
        if is_meta and (not load_all_ranks):
            # Unblock non-rank-0 processes waiting above.
            dist.barrier()

    @staticmethod
    def load_hf_weights_after_fsdp(
        fsdp_model,
        model_path: str,
        prefix: str = "hunyuan_model.",
        skip_prefixes: tuple = ("vae.",),
        model_shapes: dict | None = None,
        logger=None,
    ):
        """
        Load HuggingFace safetensors checkpoint weights into an already-FSDP-wrapped
        model.

        Only rank 0 reads the checkpoint from disk; FSDP scatters the parameters to
        all ranks via FULL_STATE_DICT + rank0_only.  This keeps peak CPU RAM at
        ~(one copy of the non-VAE model weights) rather than N-processes copies.

        Shape handling (requires model_shapes to be supplied):
          - Exact shape match  → load directly.
          - Dim-0 mismatch only (same rank, same trailing dims) → truncate or
            zero-pad the HF tensor along dim 0 to match model shape.  This
            handles vocab-size differences between the HF checkpoint and the
            MODUS tokenizer in either direction.
          - Any other mismatch → skip the key entirely (the FSDP-wrapped model
            keeps its randomly-initialised value for that parameter).

        Args:
            fsdp_model:    FSDP-wrapped model (e.g. HunyuanImageWrapper).
            model_path:    Path to the HF checkpoint directory.
            prefix:        Module path prefix prepended to HF parameter names so
                           they match the FSDP state dict keys
                           (e.g. "hunyuan_model." for HunyuanImageWrapper).
            skip_prefixes: HF keys starting with these prefixes are excluded
                           (e.g. ("vae.",) — VAE is loaded separately).
            model_shapes:  Dict mapping FSDP state-dict key → tuple(shape),
                           collected from the model BEFORE FSDP wrapping.  When
                           provided, incompatible shapes are skipped instead of
                           raising RuntimeError.
            logger:        Optional logger for progress messages.
        """
        import gc
        import glob
        import json
        from safetensors.torch import load_file

        def _log(msg):
            if logger is not None and dist.get_rank() == 0:
                logger.info(msg)

        def _reconcile(fsdp_key, tensor):
            """Return tensor reshaped to model_shapes[fsdp_key], or None to skip."""
            if model_shapes is None:
                return tensor
            expected = model_shapes.get(fsdp_key)
            if expected is None:
                return None  # key not in model → skip
            if tuple(tensor.shape) == tuple(expected):
                return tensor  # exact match
            # Allow dim-0 adjustment only (vocab / sequence length axis).
            # All trailing dims must agree.
            if tensor.ndim == len(expected) and tensor.shape[1:] == expected[1:]:
                tgt = expected[0]
                src = tensor.shape[0]
                if src > tgt:
                    return tensor[:tgt]  # truncate
                else:
                    pad_shape = (tgt - src,) + tensor.shape[1:]
                    pad = torch.zeros(pad_shape, dtype=tensor.dtype)
                    return torch.cat([tensor, pad], dim=0)
            # Incompatible shape → skip (log only on rank 0 to avoid 16× duplicates).
            if dist.get_rank() == 0:
                _log(
                    f"  Skipping {fsdp_key}: HF shape {tuple(tensor.shape)} "
                    f"incompatible with model shape {tuple(expected)}"
                )
            return None

        def _get_shard_paths():
            index_path = os.path.join(model_path, "model.safetensors.index.json")
            if os.path.exists(index_path):
                with open(index_path) as f:
                    shard_names = sorted(set(json.load(f).get("weight_map", {}).values()))
                shard_paths = [os.path.join(model_path, s) for s in shard_names]
            else:
                shard_paths = sorted(glob.glob(os.path.join(model_path, "model-*.safetensors")))
                single = os.path.join(model_path, "model.safetensors")
                if not shard_paths and os.path.exists(single):
                    shard_paths = [single]
            return shard_paths

        def _build_full_sd():
            shard_paths = _get_shard_paths()

            _log(f"Loading HF weights from {len(shard_paths)} shard(s) in {model_path}")

            _full_sd: dict = {}
            skipped = 0
            for shard_path in shard_paths:
                shard = load_file(shard_path, device="cpu")
                for hf_key, tensor in shard.items():
                    if any(hf_key.startswith(p) for p in skip_prefixes):
                        continue
                    fsdp_key = prefix + hf_key
                    reconciled = _reconcile(fsdp_key, tensor)
                    if reconciled is None:
                        skipped += 1
                    else:
                        _full_sd[fsdp_key] = reconciled
                del shard

            gc.collect()
            return _full_sd, skipped

        # The preferred path keeps peak CPU RAM minimal by reading only on rank 0.
        if dist.get_rank() == 0:
            full_sd, skipped = _build_full_sd()
            _log(
                f"Built HF state dict with {len(full_sd)} entries "
                f"({skipped} skipped due to shape mismatch)"
            )
            # Report model params that are NOT covered by the HF checkpoint
            # (they stay randomly / zero initialized after FSDP materialises them).
            if model_shapes is not None:
                _not_in_hf = sorted(
                    k for k in model_shapes if k not in full_sd
                )
                _log(
                    f"[WeightCheck] Model params NOT in HF checkpoint "
                    f"(randomly init): {len(_not_in_hf)} / {len(model_shapes)}"
                )
                if _not_in_hf:
                    _log("[WeightCheck] First 30 randomly-init param names:")
                    for _k in _not_in_hf[:30]:
                        _log(f"  {_k}  shape={model_shapes[_k]}")
        else:
            full_sd = {}

        # rank0_only=True: only rank 0 sees/writes the unsharded params on CPU.
        # Non-zero ranks participate in the all-gather (required collective) but
        # discard the result (empty tensor) — zero extra CPU/GPU on those ranks.
        # On exit, writeback=True scatters rank-0's updated params to all ranks
        # via FSDP's internal NCCL scatter.  offload_to_cpu=True keeps GPU peak
        # to one FSDP unit at a time (hundreds of MB, not the full model).
        loaded = 0
        try:
            with FSDP.summon_full_params(
                fsdp_model, recurse=True, rank0_only=True, writeback=True, offload_to_cpu=True
            ):
                if dist.get_rank() == 0:
                    for name, param in fsdp_model.named_parameters():
                        if name in full_sd:
                            param.data.copy_(full_sd[name].to(dtype=param.dtype))
                            loaded += 1
            _log(f"HF weights loaded: {loaded} parameters via rank0_only summon_full_params")
        except NotImplementedError as e:
            if "writeback=True and rank0_only=True is not supported yet" not in str(e):
                raise
            _log(
                "FSDP summon_full_params(rank0_only=True, writeback=True) is not "
                "supported in this torch build; falling back to per-unit all-ranks load."
            )
            # DO NOT free full_sd here — rank 0 still needs it.
            # Other ranks already have full_sd = {} so they hold nothing extra.
            #
            # Previous fallback (recurse=True, rank0_only=False, offload_to_cpu=True)
            # placed the FULL 157 GB on every rank's CPU: 4 ranks/node × 157 GB = 628 GB,
            # which exceeds the ~460 GB node limit.
            #
            # New approach: iterate one FSDP unit at a time with offload_to_cpu=False.
            # Each unit's gathered params stay on GPU (~4.65 GiB per rank).
            # Rank 0 copies HF values from full_sd, then broadcasts to all ranks via
            # NCCL.  writeback=True scatters the updated params back to each rank's shard.
            #
            # Peak CPU per node: rank 0 holds full_sd (~157 GB); ranks 1-3 hold nothing.
            # Peak GPU per rank: own shards (~5 GiB) + one unit (~4.65 GiB) ≈ 9.65 GiB.
            loaded = 0
            _fsdp_units: list = [
                # Strip every "_fsdp_wrapped_module" segment from the path.
                # named_modules() returns paths like:
                #   "_fsdp_wrapped_module.hunyuan_model._fsdp_wrapped_module.model.layers.0"
                # The old replace("._fsdp_wrapped_module", "") only removed interior
                # occurrences (with a leading dot), leaving the leading segment intact
                # → all non-root units had a stale "_fsdp_wrapped_module." prefix that
                # never matched any full_sd key.
                (".".join(p for p in _mname.split(".") if p != "_fsdp_wrapped_module"), _fsdp_mod)
                for _mname, _fsdp_mod in fsdp_model.named_modules()
                if isinstance(_fsdp_mod, FSDP)
            ]
            _log(f"Per-unit HF load: {len(_fsdp_units)} FSDP units")
            for _unit_prefix, _fsdp_unit in _fsdp_units:
                # offload_to_cpu=False: gathered params remain on GPU so non-rank-0
                # processes sharing the same node memory hold nothing on CPU.
                with FSDP.summon_full_params(
                    _fsdp_unit, recurse=False, rank0_only=False,
                    writeback=True, offload_to_cpu=False,
                ):
                    _unit_tensors: list = []
                    for _local_name, _param in _fsdp_unit.named_parameters():
                        # Skip child FSDP units' params (handled in their own iteration).
                        # With use_orig_params=False they appear as FlatParameter.
                        # With use_orig_params=True they are original params but may
                        # still be sharded (empty view) since summon recurse=False only
                        # unshards the current unit's params.
                        if type(_param).__name__ == "FlatParameter":
                            continue
                        _clean_local = _local_name.replace("_fsdp_wrapped_module.", "")
                        _full_key = (
                            f"{_unit_prefix}.{_clean_local}" if _unit_prefix else _clean_local
                        )
                        # Only rank 0 has full_sd; broadcast after writing.
                        if dist.get_rank() == 0 and _full_key in full_sd:
                            _param.data.copy_(
                                full_sd[_full_key].to(device=_param.device, dtype=_param.dtype)
                            )
                            loaded += 1
                        _unit_tensors.append(_param.data)
                    # Broadcast rank-0's values so all shard-groups
                    # scatter the same correct data via writeback.
                    for _t in _unit_tensors:
                        dist.broadcast(_t, src=0)

            # Release rank-0's state dict; set to None rather than del so the
            # cleanup block below can safely call del without NameError.
            _full_sd_size = len(full_sd) if dist.get_rank() == 0 else 0
            if dist.get_rank() == 0:
                full_sd = None
            gc.collect()
            _log(f"HF weights loaded: {loaded} / {_full_sd_size} full_sd entries via per-unit fallback")

        if dist.get_rank() == 0:
            del full_sd
        gc.collect()
        return fsdp_model

    @staticmethod
    def try_load_train_state(resume_from, optimizer, scheduler, fsdp_config):
        if resume_from is not None and os.path.exists(resume_from):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_state_dict_path = os.path.join(
                resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            optimizer_state_dict = torch.load(optimizer_state_dict_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(optimizer_state_dict)
            del optimizer_state_dict

            scheduler_state_dict_path = os.path.join(resume_from, "scheduler.pt")
            scheduler_state_dict = torch.load(scheduler_state_dict_path, weights_only=True, map_location="cpu")
            scheduler.load_state_dict(scheduler_state_dict)
            del scheduler_state_dict

            train_steps = int(os.path.basename(os.path.normpath(resume_from))) + 1
            """
            data_status = [
                {
                    dataset_name: {
                        worker_id: [parquet_idx, row_group_id, row_idx],
                    },
                },
            ]
            """
            data_status_path = os.path.join(resume_from, f"data_status.rank{dist.get_rank()}.pt")
            if os.path.exists(data_status_path):
                data_status = torch.load(data_status_path, weights_only=True, map_location="cpu")
            else:
                data_status = None

            data_resume_state_path = os.path.join(
                resume_from, f"data_resume_state.rank{dist.get_rank()}.pt"
            )
            if os.path.exists(data_resume_state_path):
                data_resume_state = torch.load(
                    data_resume_state_path, weights_only=False, map_location="cpu"
                )
            else:
                data_resume_state = None

            training_stats_path = os.path.join(resume_from, "training_stats.pt")
            if os.path.exists(training_stats_path):
                training_stats = torch.load(training_stats_path, weights_only=True, map_location="cpu")
            else:
                training_stats = None
        else:
            train_steps = 0
            data_status = None
            data_resume_state = None
            training_stats = None
        return optimizer, scheduler, train_steps, data_status, data_resume_state, training_stats


def make_grad_checkpoint_check_fn(checkpoint_modules):
    """Return a check_fn closure for ``apply_activation_checkpointing``."""
    def check_fn(module):
        return isinstance(module, checkpoint_modules)
    return check_fn


def fsdp_ema_setup(
    ema_model,
    fsdp_config,
    transformer_layer_cls,
    ignored_modules=[],
    sync_module_states=False,
    dp_mesh=None,
):
    for param in ema_model.parameters():
        param.requires_grad = False

    ema_model = fsdp_wrapper(
        ema_model,
        fsdp_config,
        transformer_layer_cls,
        ignored_modules=ignored_modules,
        sync_module_states=sync_module_states,
        dp_mesh=dp_mesh,
    )
    return ema_model


@torch.no_grad()
def fsdp_ema_update(ema_model, model, decay=0.9999):
    ema_handles = traversal_utils._get_fsdp_handles(ema_model)
    new_handles = traversal_utils._get_fsdp_handles(model)
    assert len(ema_handles) == len(new_handles)
    ema_params = []
    new_params = []

    for ema_handle, new_handle in zip(ema_handles, new_handles):
        if ema_handle.flat_param is not None and new_handle.flat_param.requires_grad:
            ema_params.append(ema_handle.flat_param.data)
            new_params.append(new_handle.flat_param.data.to(dtype=ema_handle.flat_param.dtype))

    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, new_params, alpha=1 - decay)

    # The loss-free MoE routing bias lives in a buffer (not flat_param), so the
    # param-only EMA above never touches it -> EMA-eval would route with a stale
    # zero bias. Mirror (copy, not EMA-smooth) the live bias into the EMA model so
    # EMA-eval routing matches the routing the experts are trained under.
    ema_bufs = dict(ema_model.named_buffers())
    for name, buf in model.named_buffers():
        if name.endswith("_lossfree_bias") and name in ema_bufs:
            ema_bufs[name].data.copy_(buf.data.to(ema_bufs[name].dtype))
