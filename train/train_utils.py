# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Training utilities: distributed init, logging, tokenizer setup, model freezing,
FSDP setup, optimizer/scheduler, dataloader, loss computation, and logging.
"""

import datetime
import functools
import gc
import logging
import os
import re
import sys
import yaml
from copy import deepcopy
from time import time

import torch
import torch.distributed as dist
import wandb
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from core.modality import ModalityRegistry
from core.tokenizer_utils import build_tokenizer_and_special_tokens, load_base_tokenizer
from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from train.fsdp_utils import (
    FSDPCheckpoint,
    fsdp_wrapper,
    fsdp_ema_setup,
    fsdp_ema_update,
    make_grad_checkpoint_check_fn,
)


# ─────── Logging ───────────────────────────────────────────────────────────────

def create_logger(logging_dir, rank, filename="log"):
    """Create a logger that writes to a log file and stdout."""
    if rank == 0 and logging_dir is not None:
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/{filename}.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


_MODALITY_ID_RE = re.compile(r"^modality_(\d+)$")
_MODALITY_NAME_CACHE = {}


def _stage_debug_log(msg: str):
    if os.environ.get("HUNYUAN_STAGE_TIMING_DEBUG", "0") != "1":
        return
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    line = f"[STAGE][rank={rank}] {msg}"
    print(line, flush=True)
    debug_dir = os.environ.get("HUNYUAN_STAGE_DEBUG_DIR", "logs/stage_debug")
    try:
        os.makedirs(debug_dir, exist_ok=True)
        jobid = os.environ.get("SLURM_JOB_ID", "nojid")
        with open(os.path.join(debug_dir, f"stage_{jobid}_rank{rank}.log"), "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _load_modality_id_to_name(training_args):
    cfg_path = getattr(training_args, "modality_config_file", None)
    if cfg_path is None:
        cfg_path = (
            "conf/modalities/instruction.yaml"
            if training_args.use_instruction
            else "conf/modalities/legacy.yaml"
        )
    if cfg_path in _MODALITY_NAME_CACHE:
        return _MODALITY_NAME_CACHE[cfg_path]

    try:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        mapping = {
            int(spec["id"]): spec["name"]
            for spec in cfg.get("modalities", [])
            if "id" in spec and "name" in spec
        }
    except Exception:
        mapping = {}

    _MODALITY_NAME_CACHE[cfg_path] = mapping
    return mapping


def _pretty_modality_key(mod_key, id_to_name):
    match = _MODALITY_ID_RE.match(mod_key)
    if not match:
        return mod_key
    mod_id = int(match.group(1))
    mod_name = id_to_name.get(mod_id, mod_key)
    return f"{mod_name}[{mod_id}]"


def get_latest_ckpt(checkpoint_dir):
    """Find the latest checkpoint directory by step number."""
    # checkpoint_dir may not exist yet on a fresh run (on Bolt the launcher does
    # not pre-mkdir it the way the CSCS train.sh does). Treat "missing dir" as
    # "no checkpoint" so resolve_resume() falls back to the init ckpt
    # (resume_from), instead of os.listdir() raising FileNotFoundError.
    if not os.path.isdir(checkpoint_dir):
        return None
    step_dirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    if len(step_dirs) == 0:
        return None
    step_dirs = sorted(step_dirs, key=lambda x: int(x))
    return os.path.join(checkpoint_dir, step_dirs[-1])


# ─────── Argument Parsing ──────────────────────────────────────────────────────

def parse_args(arg_classes):
    """
    Parse CLI arguments.

    Usage:
        torchrun train.py --config debug_any2any              # Hydra config
        torchrun train.py --config debug_any2any training.lr=2e-5  # with overrides

    When ``--config <name>`` is present the config is loaded from
    ``conf/train/<name>.yaml`` via Hydra compose. Otherwise falls back to
    HfArgumentParser for legacy CLI flags.
    """
    from transformers import HfArgumentParser
    from train.args import ModelArguments, DataArguments, TrainingArguments

    # torchrun may inject a local rank argument; we rely on env vars instead.
    for opt in ("--local_rank", "--local-rank"):
        if opt in sys.argv:
            i = sys.argv.index(opt)
            sys.argv.pop(i)
            if i < len(sys.argv) and not sys.argv[i].startswith("--"):
                sys.argv.pop(i)

    # Also accept the old --hydra flag for backward compat (just strip it).
    if "--hydra" in sys.argv:
        sys.argv.remove("--hydra")

    # ── Hydra path: triggered by --config <name> ──────────────────────────
    if "--config" in sys.argv:
        try:
            from hydra import initialize_config_dir, compose
            from omegaconf import OmegaConf
        except Exception as e:
            raise RuntimeError("--config requires hydra-core to be installed.") from e

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        conf_dir = os.path.join(repo_root, "conf", "train")

        i = sys.argv.index("--config")
        if i + 1 >= len(sys.argv):
            raise ValueError("--config requires a value (e.g. --config debug_any2any)")
        config_name = sys.argv[i + 1]
        sys.argv.pop(i)
        sys.argv.pop(i)

        # key=value tokens are Hydra overrides.
        overrides = [a for a in sys.argv[1:] if (not a.startswith("--") and "=" in a)]
        with initialize_config_dir(config_dir=conf_dir, version_base="1.3"):
            cfg = compose(config_name=config_name, overrides=overrides)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(cfg_dict, dict)
        model_args = ModelArguments(**cfg_dict.get("model", {}))
        data_args = DataArguments(**cfg_dict.get("data", {}))
        training_args = TrainingArguments(**cfg_dict.get("training", {}))
        return model_args, data_args, training_args

    # ── Legacy path: HfArgumentParser ─────────────────────────────────────
    parser = HfArgumentParser(arg_classes)
    return parser.parse_args_into_dataclasses()


# ─────── Distributed Training ──────────────────────────────────────────────────

def init_distributed() -> int:
    """Initialize distributed training and return local device ID."""
    assert torch.cuda.is_available(), "CUDA required for training"
    # Helps reduce CUDA allocator fragmentation during large FSDP init/broadcasts.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = local_rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        timeout=datetime.timedelta(minutes=30),
        # device_id intentionally omitted: PyTorch's eager_connect_single_device
        # immediately registers GPU memory via CXI DMA-BUF at init time, which
        # triggers CUDA error 801 on GH200/Slingshot. Without device_id, NCCL
        # connections are established lazily at the first collective.
    )
    return device


def setup_logger_and_wandb(training_args, model_args, data_args):
    """Setup logging and Weights & Biases tracking."""
    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())
        wandb.init(
            project=training_args.wandb_project,
            id=f"{training_args.wandb_name}-run{training_args.wandb_runid}",
            name=training_args.wandb_name,
            resume=training_args.wandb_resume,
            mode="offline" if training_args.wandb_offline else "online",
        )
        wandb.config.update(training_args, allow_val_change=True)
        wandb.config.update(model_args, allow_val_change=True)
        wandb.config.update(data_args, allow_val_change=True)
    else:
        logger = create_logger(None, dist.get_rank())
    dist.barrier(device_ids=[torch.cuda.current_device()])
    return logger


# ─────── Checkpoint & Resume ───────────────────────────────────────────────────

def resolve_resume(training_args):
    """
    Compute resume behavior.

    Returns: (resume_from, resume_model_only, finetune_from_ema, checkpoint_name)
    """
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            finetune_from_ema = training_args.finetune_from_ema if resume_model_only else False
        else:
            resume_model_only = False
            finetune_from_ema = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        finetune_from_ema = training_args.finetune_from_ema if resume_model_only else False
    return resume_from, resume_model_only, finetune_from_ema, training_args.checkpoint_name


# ─────── Tokenizer & Modality Registry ─────────────────────────────────────────

def build_tokenizer_and_modality_registry(model_args, training_args):
    """
    Build tokenizer, add special tokens, and create a ModalityRegistry.

    Returns: (tokenizer, new_token_ids, num_new_tokens, modality_registry)
    """
    modality_cfg_path = training_args.modality_config_file
    if modality_cfg_path is None:
        modality_cfg_path = (
            "conf/modalities/instruction.yaml"
            if training_args.use_instruction
            else "conf/modalities/legacy.yaml"
        )
    with open(modality_cfg_path, "r") as f:
        modality_cfg = yaml.safe_load(f)

    tokenizer = load_base_tokenizer(model_args=model_args, training_args=training_args)
    orig_vocab_size = len(tokenizer)

    tok_artifacts = build_tokenizer_and_special_tokens(
        tokenizer,
        modalities_cfg=modality_cfg,
    )
    tokenizer = tok_artifacts.tokenizer
    num_new_tokens = len(tokenizer) - orig_vocab_size
    new_token_ids = tok_artifacts.new_token_ids

    modality_registry = ModalityRegistry.from_config(
        modality_cfg,
        token_ranges=tok_artifacts.token_ranges,
        code_token_ids=tok_artifacts.code_token_ids,
    )
    return tokenizer, new_token_ids, num_new_tokens, modality_registry


# ─────── Model Component Freezing ──────────────────────────────────────────────


def maybe_freeze_components(model, vae_model, training_args):
    """Optionally freeze model components based on training args."""
    if training_args.freeze_vae and training_args.visual_gen:
        for param in vae_model.parameters():
            param.requires_grad = False
    if training_args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
    if training_args.freeze_vit and training_args.visual_und:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False


# ─────── FSDP Setup & Checkpoint Loading ───────────────────────────────────────

def _compute_model_flops_per_token(model, logger=None):
    """Estimate FLOPs per token for one forward+backward pass.

    Uses the standard 6N approximation (2 multiply-adds × 3 for fwd+bwd).
    For MoE layers only the active experts contribute; controlled by env vars:
      HUNYUAN_MFU_TOP_K            (default 8)
      HUNYUAN_MFU_NUM_EXPERTS      (default 64)
      HUNYUAN_MFU_NUM_SHARED       (default 1)
    Embedding parameters are excluded (they don't contribute matmul FLOPs).
    """
    top_k       = int(os.environ.get("HUNYUAN_MFU_TOP_K", "8"))
    num_experts = int(os.environ.get("HUNYUAN_MFU_NUM_EXPERTS", "64"))
    num_shared  = int(os.environ.get("HUNYUAN_MFU_NUM_SHARED", "1"))

    expert_total = 0
    dense_total  = 0
    for name, param in model.named_parameters():
        n = param.numel()
        # Skip pure embedding tables (no matmul FLOPs per token)
        if "wte" in name or "embed_tokens" in name or "lm_head" in name:
            continue
        if "experts." in name:
            expert_total += n
        else:
            dense_total += n

    active_expert = expert_total * (top_k + num_shared) / num_experts
    active_params = dense_total + active_expert
    flops_per_token = 6 * active_params
    if logger is not None and dist.get_rank() == 0:
        logger.info(
            f"[MFU] param breakdown: dense={dense_total/1e9:.2f}B, "
            f"expert_total={expert_total/1e9:.2f}B, "
            f"active_expert(top{top_k}+{num_shared}shared)={active_expert/1e9:.2f}B, "
            f"active_total={active_params/1e9:.2f}B → "
            f"flops/token={flops_per_token/1e12:.3f}T"
        )
    return flops_per_token


def _stabilize_new_token_rows(fsdp_model, tokenizer_len, num_new_tokens, logger=None, model_label="model"):
    """Initialize newly added embedding/head rows from old-row mean after HF load.

    Hunyuan models are resized while still on meta tensors, so HF's
    mean_resizing path is disabled during resize_token_embeddings. That leaves
    newly added rows randomly initialized. For large modality token expansions
    this is a bad starting point and was correlated with early non-finite grads.

    This helper runs *after* HF weights are loaded into the resized model and
    overwrites the newly added rows with the mean of the pre-existing rows.
    """
    if num_new_tokens <= 0:
        return
    old_vocab = int(tokenizer_len) - int(num_new_tokens)
    if old_vocab <= 0:
        return

    # Iterate FSDP units one at a time with recurse=False instead of a single
    # recursive summon. The recursive form gathers every FSDP unit's full flat
    # param onto each rank simultaneously — fine when the model is sharded in
    # bf16 (~150 GB total), but with fp32 master weights (~308 GB total) it
    # OOMs the 95 GiB H100s. Per-unit summon caps GPU memory at one unit's
    # full size (~10 GB).
    touched = []
    with torch.no_grad():
        for _mname, _fsdp_mod in fsdp_model.named_modules():
            if not isinstance(_fsdp_mod, FSDP):
                continue
            with FSDP.summon_full_params(
                _fsdp_mod, recurse=False, rank0_only=False, writeback=True, offload_to_cpu=False
            ):
                for name, param in _fsdp_mod.named_parameters():
                    if param is None or param.ndim != 2:
                        continue
                    if type(param).__name__ == "FlatParameter":
                        continue  # nested FSDP unit's param appears here too; skip
                    if param.shape[0] < tokenizer_len or param.shape[0] <= old_vocab:
                        continue
                    if not (
                        name.endswith(".model.wte.weight")
                        or name.endswith(".lm_head.weight")
                        or ".model.wte.weight" in name
                        or ".lm_head.weight" in name
                    ):
                        continue
                    base_rows = param.data[:old_vocab].float()
                    if base_rows.numel() == 0:
                        continue
                    row_mean = base_rows.mean(dim=0, keepdim=True)
                    new_rows_count = tokenizer_len - old_vocab
                    # Diverse init: mean + per-row noise scaled by per-dim std of
                    # existing rows. Default OFF (mean-only) preserves prior
                    # numerical behaviour. Setting HUNYUAN_EMBED_INIT_NOISE_SCALE
                    # > 0 makes each new row distinct so the MoE router at layer 0
                    # can differentiate codebook tokens (otherwise 8192 identical
                    # rows -> all routed to same experts -> permanent overload).
                    try:
                        _noise_scale = float(os.environ.get("HUNYUAN_EMBED_INIT_NOISE_SCALE", "0.0"))
                    except ValueError:
                        _noise_scale = 0.0
                    if _noise_scale > 0.0:
                        row_std = base_rows.std(dim=0, keepdim=True)  # [1, hidden]
                        # Use a fixed seed for reproducibility across ranks (FSDP
                        # summon broadcasts the same param shard, so the noise
                        # tensor must be identical on every rank).
                        gen = torch.Generator(device=param.device)
                        gen.manual_seed(0xCAFEBABE)
                        noise = torch.randn(
                            new_rows_count,
                            base_rows.shape[1],
                            device=param.device,
                            dtype=torch.float32,
                            generator=gen,
                        ) * row_std * _noise_scale
                        new_rows = (row_mean + noise).to(dtype=param.dtype, device=param.device)
                    else:
                        new_rows = row_mean.to(dtype=param.dtype, device=param.device).expand(new_rows_count, -1)
                    param.data[old_vocab:tokenizer_len].copy_(new_rows)
                    touched.append(name)

    if logger is not None and dist.get_rank() == 0 and touched:
        logger.info(
            f"[WeightCheck] stabilized new token rows for {model_label}: "
            f"old_vocab={old_vocab}, new_tokens={num_new_tokens}, tensors={touched}"
        )


def setup_fsdp_and_load_checkpoint(
    model, training_args, fsdp_config, modality_registry, tokenizer, num_new_tokens,
    resume_from, finetune_from_ema, checkpoint_name, logger,
):
    """
    Deep-copy model for EMA, attach the modality registry, resize embeddings,
    wrap both models with FSDP, load checkpoint, and apply activation
    checkpointing.

    Returns: (fsdp_model, ema_model)
    """
    # Attach modality registry BEFORE deepcopy so that pos-embed modules
    # (created lazily in set_modality_registry) are initialised once and then
    # copied identically into ema_model.  Previously, calling it after
    # deepcopy caused model & ema to get independent nn.init.normal_() calls,
    # giving different starting weights and shifting the torch RNG state.
    if hasattr(model, "set_modality_registry"):
        model.set_modality_registry(modality_registry)
    else:
        model.modality_registry = modality_registry

    # Sanity-check registry-requested learnable pos-embed modules.
    if dist.get_rank() == 0 and modality_registry is not None:
        try:
            requested = modality_registry.modalities_with_forward_pos_embed()
        except Exception:
            requested = []
        if requested:
            host_model = getattr(model, "hunyuan_model", model)
            missing = []
            for spec in requested:
                attr_name = f"{spec.pos_embed_name or spec.name}_pos_embed"
                if not hasattr(host_model, attr_name):
                    missing.append(attr_name)
            if missing:
                logger.info(
                    "[WeightCheck] registry requested pos-embed attrs missing on model: "
                    f"{missing}"
                )
            else:
                logger.info("[WeightCheck] all registry-requested pos-embed attrs exist on model")

    # Compute FLOPs/token BEFORE TP/EP slicing so we see the full parameter
    # count.  Model is on meta device here — numel() still works.
    model_flops_per_token = _compute_model_flops_per_token(model, logger)

    use_ema = getattr(training_args, "ema", 0.0) > 0.0
    ema_model = deepcopy(model) if use_ema else None
    logger.info(f"System memory before FSDP wrap: {system_memory_usage_gb():.1f} GB")

    # Resize embeddings if new tokens were added.
    # NOTE: with init_device="meta", HF mean_resizing path tries to read tensor
    # values (e.g. .item()) and crashes on meta tensors. In that case we
    # automatically fall back to mean_resizing=False.
    if num_new_tokens > 0:
        _mean_resize_req = os.environ.get("HUNYUAN_EMBED_MEAN_RESIZE", "1") == "1"
        _mean_resize_used = _mean_resize_req
        models_to_resize = [model]
        if ema_model is not None:
            models_to_resize.append(ema_model)
        for m in models_to_resize:
            _mean_resize_local = _mean_resize_req
            try:
                _emb = m.language_model.get_input_embeddings()
                if _emb is not None and getattr(_emb.weight, "device", None) is not None:
                    if _emb.weight.device.type == "meta":
                        _mean_resize_local = False
            except Exception:
                _mean_resize_local = False if _mean_resize_req else False
            m.language_model.resize_token_embeddings(
                len(tokenizer), mean_resizing=_mean_resize_local
            )
            m.config.llm_config.vocab_size = len(tokenizer)
            m.language_model.config.vocab_size = len(tokenizer)
            _mean_resize_used = _mean_resize_used and _mean_resize_local
        if dist.get_rank() == 0:
            logger.info(
                f"[WeightCheck] resize_token_embeddings: num_new_tokens={num_new_tokens}, "
                f"mean_resizing_requested={int(_mean_resize_req)}, "
                f"mean_resizing_used={int(_mean_resize_used)}"
            )

    # Query model for FSDP layer classes (model-driven declarations)
    wrap_cls = model.fsdp_wrap_modules() if hasattr(model, "fsdp_wrap_modules") else set()
    ckpt_cls = model.fsdp_checkpoint_modules() if hasattr(model, "fsdp_checkpoint_modules") else ()

    hf_model_path = getattr(model, '_hf_model_path', None)
    hf_prefix = getattr(model, '_hf_weight_prefix', 'hunyuan_model.')
    hf_skip = getattr(model, '_hf_skip_prefixes', ('vae.',))
    # Allow run-time skipping of selected HF checkpoint submodules when
    # pretrained weights are numerically unstable for a given recipe.
    _hf_skip_extra = os.environ.get("HUNYUAN_HF_SKIP_PREFIXES", "").strip()
    if _hf_skip_extra:
        _extra = tuple(p.strip() for p in _hf_skip_extra.split(",") if p.strip())
        if _extra:
            hf_skip = tuple(hf_skip) + _extra
            logger.info(f"[HFLoad] Extra HF skip prefixes from env: {_extra}")

    # Minimize allocator fragmentation before FSDP init.
    gc.collect()
    torch.cuda.empty_cache()

    # VAE is not used during training forward — remove from both models before
    # FSDP wrapping to eliminate its FP32 params (~4.7 GiB) from the root FSDP
    # unit.  Without this, the root unit's combined all-gather + fp32-reduce
    # backward buffer is ~13.57 GiB, which exceeds the ~9.25 GiB free after
    # optimizer states are created at step 0 (step 1+ OOM).
    # With VAE removed: ~6.5 GiB combined → fits within available memory.
    models_with_vae = [model]
    if ema_model is not None:
        models_with_vae.append(ema_model)
    for _m in models_with_vae:
        _hm = getattr(_m, 'hunyuan_model', _m)
        if hasattr(_hm, 'vae') and _hm.vae is not None:
            del _hm.vae
            _hm.vae = None
    gc.collect()
    torch.cuda.empty_cache()

    # Capture parameter shapes so that _reconcile can pad/truncate vocab-size
    # dims (wte/lm_head) when loading the HF checkpoint.
    hf_model_shapes: dict | None = None
    if hf_model_path:
        hf_model_shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}

    # Wrap with FSDP using sync_module_states=False so that each inner FSDP
    # unit's flat_param is sharded immediately after creation.  With
    # sync_module_states=True the flat_params of all inner units accumulate
    # unsharded on GPU until the root sync completes; for a 157 GB MoE model
    # (~18 decoder layers × 4.65 GiB each already fills a 94 GB GPU before
    # the 19th layer can be wrapped).  HF weights are loaded post-FSDP below.
    fsdp_model = fsdp_wrapper(model, fsdp_config, wrap_cls, sync_module_states=False)
    if ema_model is not None:
        ema_model = fsdp_ema_setup(ema_model, fsdp_config, wrap_cls, sync_module_states=False)
    logger.info(f"System memory after FSDP wrap: {system_memory_usage_gb():.1f} GB")

    # NOTE: router-gate gradient TP all-reduce is handled explicitly in the training
    # loop (train.py) after loss.backward(), NOT via register_hook here.
    # register_hook on FSDP-managed params with use_orig_params=True fires BEFORE
    # FSDP's own reduce-scatter, which causes double-counting when the training loop
    # also all-reduces. Using an explicit post-backward all-reduce is more reliable.

    # Verify no parameters remain on meta after FSDP wrapping.
    _local_meta = sum(1 for _, p in fsdp_model.named_parameters() if p.device.type == "meta")
    _meta_tensor = torch.tensor(_local_meta, device=torch.cuda.current_device(), dtype=torch.int32)
    dist.all_reduce(_meta_tensor, op=dist.ReduceOp.SUM)
    if dist.get_rank() == 0:
        logger.info(f"[WeightCheck] post-FSDP total meta-parameter-count across ranks: {_meta_tensor.item()}")

    gc.collect()
    torch.cuda.empty_cache()

    # Load HF pretrained weights post-FSDP, one FSDP unit at a time via
    # summon_full_params(offload_to_cpu=True).  Peak GPU overhead is one
    # unit's worth of params (~4.65 GiB for a decoder layer), not the full
    # model.  Immediately after, copy weights to EMA via a decay=0 update
    # (direct copy of sharded flat_params, no extra GPU allocation).
    if hf_model_path and resume_from is None:
        logger.info(f"Loading HF pretrained weights (post-FSDP) from {hf_model_path}")
        FSDPCheckpoint.load_hf_weights_after_fsdp(
            fsdp_model, hf_model_path, hf_prefix, hf_skip,
            model_shapes=hf_model_shapes, logger=logger,
        )
        gc.collect()
        torch.cuda.empty_cache()
        if ema_model is not None:
            logger.info("Initializing EMA model from loaded weights (fsdp_ema_update, decay=0.0)...")
            fsdp_ema_update(ema_model, fsdp_model, decay=0.0)
            logger.info(f"[WeightCheck] model and EMA initialized from HF checkpoint: {hf_model_path}")

        _stabilize_new_token_rows(
            fsdp_model,
            tokenizer_len=len(tokenizer),
            num_new_tokens=num_new_tokens,
            logger=logger,
            model_label="fsdp_model",
        )
        if ema_model is not None:
            _stabilize_new_token_rows(
                ema_model,
                tokenizer_len=len(tokenizer),
                num_new_tokens=num_new_tokens,
                logger=logger,
                model_label="ema_model",
            )

        # Post-load sanity: confirm loaded params are finite and log coarse scale stats.
        with torch.no_grad():
            _d = torch.cuda.current_device()
            _nf_local = torch.tensor(0, device=_d, dtype=torch.int64)
            _max_all_local = torch.tensor(0.0, device=_d, dtype=torch.float32)
            _max_vit_local = torch.tensor(0.0, device=_d, dtype=torch.float32)
            _max_va_local = torch.tensor(0.0, device=_d, dtype=torch.float32)
            _max_t_local = torch.tensor(0.0, device=_d, dtype=torch.float32)
            for _n, _p in fsdp_model.named_parameters():
                if _p is None or _p.data.numel() == 0:
                    continue
                _pd = _p.data
                if not torch.isfinite(_pd).all():
                    _nf_local += 1
                _absmax = _pd.detach().abs().max().float()
                _max_all_local = torch.maximum(_max_all_local, _absmax)
                if ".vision_model." in _n:
                    _max_vit_local = torch.maximum(_max_vit_local, _absmax)
                if ".vision_aligner." in _n:
                    _max_va_local = torch.maximum(_max_va_local, _absmax)
                if ".timestep_emb." in _n:
                    _max_t_local = torch.maximum(_max_t_local, _absmax)
            dist.all_reduce(_nf_local, op=dist.ReduceOp.SUM)
            dist.all_reduce(_max_all_local, op=dist.ReduceOp.MAX)
            dist.all_reduce(_max_vit_local, op=dist.ReduceOp.MAX)
            dist.all_reduce(_max_va_local, op=dist.ReduceOp.MAX)
            dist.all_reduce(_max_t_local, op=dist.ReduceOp.MAX)
            if dist.get_rank() == 0:
                logger.info(
                    "[WeightCheck] post-load finite check: "
                    f"nonfinite_param_shards={int(_nf_local.item())}, "
                    f"max_abs(all)={float(_max_all_local.item()):.4e}, "
                    f"max_abs(vision_model)={float(_max_vit_local.item()):.4e}, "
                    f"max_abs(vision_aligner)={float(_max_va_local.item()):.4e}, "
                    f"max_abs(timestep_emb)={float(_max_t_local.item()):.4e}"
                )

    # Load checkpoint
    if resume_from is not None and os.path.exists(resume_from):
        # Resume from an existing MODUS checkpoint (preemption recovery or
        # fine-tuning from a previously saved MODUS run).
        logger.info(f"Loading checkpoint from {resume_from}")
        fsdp_model, ema_model = FSDPCheckpoint.try_load_ckpt_after_fsdp(
            resume_from, logger, fsdp_model, ema_model,
            resume_from_ema=finetune_from_ema, model_name=checkpoint_name,
        )
        gc.collect()
        torch.cuda.empty_cache()
    logger.info(f"System memory after checkpoint load: {system_memory_usage_gb():.1f} GB")

    # Activation checkpointing (set NO_ACTIVATION_CHECKPOINT=1 to disable)
    no_activation_checkpoint = os.environ.get("NO_ACTIVATION_CHECKPOINT", "").strip().lower()
    disable_activation_checkpoint = no_activation_checkpoint in {"1", "true", "yes", "on"}
    if not disable_activation_checkpoint:
        check_fn = make_grad_checkpoint_check_fn(ckpt_cls)
        apply_activation_checkpointing(
            fsdp_model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            ),
            check_fn=check_fn,
        )
    else:
        logger.info("[AC] Activation checkpointing DISABLED (NO_ACTIVATION_CHECKPOINT=1)")

    return fsdp_model, ema_model, model_flops_per_token


# ─────── Optimizer & Scheduler ─────────────────────────────────────────────────

def build_optimizer_and_scheduler(
    fsdp_model, training_args, resume_from, resume_model_only, fsdp_config,
):
    """
    Create AdamW optimizer and LR scheduler, optionally restoring their state
    from a checkpoint.

    Returns: (optimizer, scheduler, train_step, data_status, data_resume_state, training_stats)
    """
    embed_head_lr_mult = float(os.environ.get("HUNYUAN_EMBED_HEAD_LR_MULT", "1.0"))
    embed_head_keys = (
        ".hunyuan_model.model.wte.",
        ".hunyuan_model.lm_head.",
        ".model.wte.",
        ".lm_head.",
    )

    if embed_head_lr_mult != 1.0:
        base_params = []
        embed_head_params = []
        base_tensor_count = 0
        embed_head_tensor_count = 0
        base_local_numel = 0
        embed_head_local_numel = 0
        base_nonempty_local_tensors = 0
        embed_head_nonempty_local_tensors = 0
        # Include currently-frozen params too so that later unfreeze hooks
        # (e.g. HUNYUAN_UNFREEZE_WARMUP_STEP) actually take effect. AdamW.step()
        # skips params whose grad is None, so frozen params cost nothing until
        # they start accumulating grads.
        base_trainable = 0
        embed_head_trainable = 0
        for name, param in fsdp_model.named_parameters():
            if any(key in name for key in embed_head_keys):
                embed_head_params.append(param)
                embed_head_tensor_count += 1
                _local_numel = param.numel()
                embed_head_local_numel += _local_numel
                if _local_numel > 0:
                    embed_head_nonempty_local_tensors += 1
                if param.requires_grad:
                    embed_head_trainable += 1
            else:
                base_params.append(param)
                base_tensor_count += 1
                _local_numel = param.numel()
                base_local_numel += _local_numel
                if _local_numel > 0:
                    base_nonempty_local_tensors += 1
                if param.requires_grad:
                    base_trainable += 1

        param_groups = []
        if base_params:
            param_groups.append(
                {
                    "params": base_params,
                    "lr": training_args.lr,
                    "group_name": "base",
                }
            )
        if embed_head_params:
            param_groups.append(
                {
                    "params": embed_head_params,
                    "lr": training_args.lr * embed_head_lr_mult,
                    "group_name": "embed_head",
                }
            )

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(training_args.beta1, training_args.beta2),
            eps=training_args.eps,
            weight_decay=0,
        )
        if dist.is_initialized() and dist.get_rank() == 0:
            print(
                "[Optimizer] split param groups enabled: "
                f"base_lr={training_args.lr:.3e}, "
                f"embed_head_lr={training_args.lr * embed_head_lr_mult:.3e}, "
                f"embed_head_lr_mult={embed_head_lr_mult:.2f}, "
                f"base_tensors={base_tensor_count}, "
                f"embed_head_tensors={embed_head_tensor_count}, "
                f"base_local_numel={base_local_numel}, "
                f"embed_head_local_numel={embed_head_local_numel}, "
                f"base_nonempty_local_tensors={base_nonempty_local_tensors}, "
                f"embed_head_nonempty_local_tensors={embed_head_nonempty_local_tensors}, "
                f"base_trainable_at_init={base_trainable}, "
                f"embed_head_trainable_at_init={embed_head_trainable}"
            )
    else:
        optimizer = torch.optim.AdamW(
            fsdp_model.parameters(),
            lr=training_args.lr,
            betas=(training_args.beta1, training_args.beta2),
            eps=training_args.eps,
            weight_decay=0,
        )

    if training_args.lr_scheduler == "cosine":
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError(f"Unknown lr_scheduler: {training_args.lr_scheduler}")

    if resume_model_only:
        train_step = 0
        data_status = None
        data_resume_state = None
        training_stats = None
    else:
        optimizer, scheduler, train_step, data_status, data_resume_state, training_stats = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_config
        )

    return optimizer, scheduler, train_step, data_status, data_resume_state, training_stats


# ─────── Dataset & DataLoader ──────────────────────────────────────────────────

def build_train_dataloader(
    data_args, training_args, model_args,
    tokenizer, new_token_ids, modality_registry,
    vae_config, data_status, data_resume_state,
):
    """
    Build the dataset configuration, create a PackedDataset, and wrap it in a
    DataLoader ready for the training loop.

    Returns: train_loader
    """
    with open(data_args.dataset_config_file, "r") as f:
        dataset_meta = yaml.safe_load(f)

    dataset_config = DataConfig(grouped_datasets=dataset_meta)
    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    if training_args.visual_gen:
        vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
        dataset_config.vae_image_downsample = vae_image_downsample
        dataset_config.max_latent_size = model_args.max_latent_size
        dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob
        dataset_config.grounding_phrase_dropout_prob = model_args.grounding_phrase_dropout_prob

    data_rank = dist.get_rank()
    data_world = dist.get_world_size()

    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        modality_registry=modality_registry,
        local_rank=data_rank,
        world_size=data_world,
        num_workers=data_args.num_workers,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex,
        data_status=data_status,
        data_resume_state=data_resume_state,
        use_instruction=training_args.use_instruction,
        use_condition_instruction=training_args.use_condition_instruction,
        use_target_instruction=training_args.use_target_instruction,
        num_condition_modalities=training_args.num_condition_modalities,
        strict_num_condition_modalities=training_args.strict_num_condition_modalities,
        timestep_sample=training_args.timestep_sample,
        mode_scale=training_args.mode_scale,
        timestep_sample_mix_prob=training_args.timestep_sample_mix_prob,
        use_det_image=training_args.use_det_image,
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
    )
    train_dataset.set_epoch(data_args.data_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # batch size is 1 packed sequence
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        prefetch_factor=None if data_args.num_workers == 0 else data_args.prefetch_factor,
    )
    return train_loader


# ─────── ViT Batch Validation ──────────────────────────────────────────────────

def should_skip_vit_batch(data, device):
    """
    Check whether the current batch has valid ViT sequence lengths.

    All ranks must agree (via all-reduce MIN), so this must be called
    collectively.  Returns ``True`` when the batch should be skipped.
    """
    vit_ok = (
        ("vit_token_seqlens" in data)
        and (data["vit_token_seqlens"] is not None)
        and (torch.numel(data["vit_token_seqlens"]) > 0)
    )
    if os.environ.get("HUNYUAN_STAGE_TIMING_DEBUG", "0") == "1":
        _numel = (
            int(torch.numel(data["vit_token_seqlens"]))
            if ("vit_token_seqlens" in data and data["vit_token_seqlens"] is not None)
            else -1
        )
        _stage_debug_log(
            f"should_skip_vit_batch local vit_ok={int(vit_ok)} vit_numel={_numel}"
        )
    vit_ok_tensor = torch.tensor(1 if vit_ok else 0, device=device, dtype=torch.int32)
    dist.all_reduce(vit_ok_tensor, op=dist.ReduceOp.MIN)
    if os.environ.get("HUNYUAN_STAGE_TIMING_DEBUG", "0") == "1":
        _stage_debug_log(
            f"should_skip_vit_batch reduced vit_ok={int(vit_ok_tensor.item())}"
        )
    return vit_ok_tensor.item() == 0


# ─────── Loss Computation ──────────────────────────────────────────────────────

def compute_loss(
    loss_dict,
    data,
    training_args,
    ce_loss_weights,
    device,
    debug_step=None,
    mse_loss_route_ids=None,
    mse_loss_image_ids=None,
    mse_loss_timesteps=None,
):
    """
    Aggregate CE and MSE losses from the model forward output.

    Modifies *loss_dict* in-place (replaces raw tensors with detached scalars).

    Returns: (total_loss, total_ce_tokens, total_mse_tokens)
    """
    loss = 0
    _stage_debug_start = int(os.environ.get("HUNYUAN_STAGE_TIMING_DEBUG_START_STEP", "0"))
    _stage_debug_steps = int(os.environ.get("HUNYUAN_STAGE_TIMING_DEBUG_STEPS", "5"))
    _stage_debug = (
        os.environ.get("HUNYUAN_STAGE_TIMING_DEBUG", "0") == "1"
        and debug_step is not None
        and _stage_debug_start <= debug_step < (_stage_debug_start + _stage_debug_steps)
    )

    def _dbg(msg: str):
        if not _stage_debug:
            return
        _stage_debug_log(f"[compute_loss][step={debug_step}] {msg}")

    # ── CE loss ───────────────────────────────────────────────────────────
    ce = loss_dict["ce"]
    local_ce_tokens = (
        int(len(data["ce_loss_indexes"]))
        if ("ce_loss_indexes" in data and data["ce_loss_indexes"] is not None)
        else 0
    )
    total_ce_tokens = torch.tensor(local_ce_tokens, device=device)
    _dbg(
        "entry "
        f"ce_is_none={int(ce is None)} "
        f"ce_indexes_is_none={int(data.get('ce_loss_indexes', None) is None)} "
        f"ce_indexes_len={-1 if data.get('ce_loss_indexes', None) is None else int(len(data['ce_loss_indexes']))}"
    )
    _dbg(f"before_ce_allreduce local_ce_tokens={local_ce_tokens} ce_numel={-1 if ce is None else int(ce.numel())}")
    dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
    _dbg(f"after_ce_allreduce total_ce_tokens={int(total_ce_tokens.item())}")
    if ce is not None:
        if total_ce_tokens.item() == 0 or ce.numel() == 0:
            ce = ce.sum() * 0.0
        elif training_args.ce_loss_reweighting:
            ce = ce * ce_loss_weights
            total_ce_loss_weights = ce_loss_weights.sum()
            _dbg("before_ce_weight_allreduce")
            dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
            _dbg("after_ce_weight_allreduce")
            if total_ce_loss_weights.item() == 0:
                ce = ce.sum() * 0.0
            else:
                ce = ce.sum() * dist.get_world_size() / total_ce_loss_weights
        elif training_args.ce_loss_average_over_modalities:
            ce_per_modality = loss_dict.get("ce_per_modality", None)
            if isinstance(ce_per_modality, dict) and len(ce_per_modality) > 0:
                modality_means = [
                    t.mean() for t in ce_per_modality.values() if t.numel() > 0
                ]
                # v37: HUNYUAN_CE_LOSS_MODALITY_SUM=1 sums per-modality means
                # instead of averaging — gives each modality full single-task
                # per-token gradient, vs being divided by N_modalities (~9x
                # dilution under multi-task). See codex_response_to_rereview.
                if os.environ.get("HUNYUAN_CE_LOSS_MODALITY_SUM", "0") == "1":
                    ce = torch.stack(modality_means).sum() if modality_means else (ce.sum() * 0.0)
                else:
                    ce = torch.stack(modality_means).mean() if modality_means else (ce.sum() * 0.0)
            else:
                ce = ce.mean() if ce.numel() > 0 else (ce.sum() * 0.0)
        else:
            ce = ce.sum() * dist.get_world_size() / total_ce_tokens

        loss_dict["ce"] = ce.detach()
        loss = loss + ce * training_args.ce_weight
    else:
        # Pure visual-generation batches can legitimately carry ViT condition
        # tokens while having no CE-supervised targets. Treat this as zero CE
        # loss instead of requiring visual_und=False.
        loss_dict["ce"] = torch.tensor(0, device=device)

    # ── MSE loss ──────────────────────────────────────────────────────────
    if training_args.visual_gen:
        mse = loss_dict.get("mse", None)
        has_mse_local = isinstance(mse, torch.Tensor)
        if (
            has_mse_local
            and training_args.log_rgb_condition_loss_group
            and mse_loss_route_ids is not None
            and mse_loss_image_ids is not None
            and mse_loss_timesteps is not None
        ):
            token_mse = mse.mean(dim=-1)
            route_defs = (
                (1, "caption2rgb"),
                (2, "grounding2rgb"),
                (3, "dinolocal2rgb"),
            )
            rgb_route_group = {
                route_name: {
                    "loss_sum": torch.tensor(0.0, device=device),
                    "count": torch.tensor(0.0, device=device),
                }
                for _, route_name in route_defs
            }
            bin_defs = (
                (0, "0.0_0.2"),
                (1, "0.2_0.4"),
                (2, "0.4_0.6"),
                (3, "0.6_0.8"),
                (4, "0.8_1.0"),
            )
            route_id_to_name = {route_id: route_name for route_id, route_name in route_defs}
            rgb_timestep_bins_by_route = {
                route_name: {
                    bin_name: {
                        "loss_sum": torch.tensor(0.0, device=device),
                        "count": torch.tensor(0.0, device=device),
                    }
                    for _, bin_name in bin_defs
                }
                for _, route_name in route_defs
            }
            if (
                token_mse.numel() == mse_loss_route_ids.numel()
                and token_mse.numel() == mse_loss_image_ids.numel()
                and token_mse.numel() == mse_loss_timesteps.numel()
            ):
                for image_id in torch.unique(mse_loss_image_ids):
                    image_mask = mse_loss_image_ids == image_id
                    if not image_mask.any():
                        continue
                    sample_loss = token_mse[image_mask].mean()
                    sample_timestep = float(mse_loss_timesteps[image_mask][0].item())
                    sample_route_id = int(mse_loss_route_ids[image_mask][0].item())

                    for route_id, route_name in route_defs:
                        if sample_route_id == route_id:
                            rgb_route_group[route_name] = {
                                "loss_sum": rgb_route_group[route_name]["loss_sum"] + sample_loss.detach(),
                                "count": rgb_route_group[route_name]["count"] + torch.tensor(1.0, device=device),
                            }
                            break

                    if sample_route_id in route_id_to_name:
                        bin_idx = int(sample_timestep * 5.0)
                        if bin_idx < 0:
                            bin_idx = 0
                        if bin_idx > 4:
                            bin_idx = 4
                        _, bin_name = bin_defs[bin_idx]
                        route_name = route_id_to_name[sample_route_id]
                        rgb_timestep_bins_by_route[route_name][bin_name] = {
                            "loss_sum": rgb_timestep_bins_by_route[route_name][bin_name]["loss_sum"] + sample_loss.detach(),
                            "count": rgb_timestep_bins_by_route[route_name][bin_name]["count"] + torch.tensor(1.0, device=device),
                        }
            loss_dict["rgb_route_group"] = rgb_route_group
            loss_dict["rgb_timestep_bins_by_route"] = rgb_timestep_bins_by_route
        local_mse_tokens = (
            int(len(data["mse_loss_indexes"]))
            if ("mse_loss_indexes" in data and data["mse_loss_indexes"] is not None)
            else 0
        )
        total_mse_tokens = torch.tensor(local_mse_tokens, device=device)
        _dbg(f"before_mse_allreduce local_mse_tokens={int(total_mse_tokens.item())} has_mse_local={int(has_mse_local)}")
        dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
        _dbg(f"after_mse_allreduce total_mse_tokens={int(total_mse_tokens.item())}")

        if not has_mse_local:
            # Keep collectives symmetric across ranks when some ranks are CE-only.
            mse = torch.tensor(0.0, device=device)
        if total_mse_tokens.item() == 0 or mse.numel() == 0:
            mse = mse.sum() * 0.0
        else:
            mse = mse.mean(dim=-1).sum() * dist.get_world_size() / total_mse_tokens
        loss_dict["mse"] = mse.detach()
        loss = loss + mse * training_args.mse_weight
    else:
        loss_dict["mse"] = torch.tensor(0, device=device)
        total_mse_tokens = torch.tensor(0, device=device)
        if training_args.log_rgb_condition_loss_group:
            loss_dict["rgb_route_group"] = {
                "caption2rgb": {
                    "loss_sum": torch.tensor(0.0, device=device),
                    "count": torch.tensor(0.0, device=device),
                },
                "grounding2rgb": {
                    "loss_sum": torch.tensor(0.0, device=device),
                    "count": torch.tensor(0.0, device=device),
                },
                "dinolocal2rgb": {
                    "loss_sum": torch.tensor(0.0, device=device),
                    "count": torch.tensor(0.0, device=device),
                },
            }
            loss_dict["rgb_timestep_bins_by_route"] = {
                "caption2rgb": {
                    "0.0_0.2": {"loss_sum": torch.tensor(0.0, device=device), "count": torch.tensor(0.0, device=device)},
                    "0.2_0.4": {"loss_sum": torch.tensor(0.0, device=device), "count": torch.tensor(0.0, device=device)},
                    "0.4_0.6": {"loss_sum": torch.tensor(0.0, device=device), "count": torch.tensor(0.0, device=device)},
                    "0.6_0.8": {"loss_sum": torch.tensor(0.0, device=device), "count": torch.tensor(0.0, device=device)},
                    "0.8_1.0": {"loss_sum": torch.tensor(0.0, device=device), "count": torch.tensor(0.0, device=device)},
                },
                "grounding2rgb": {
                    "0.0_0.2": {"loss_sum": torch.tensor(0.0, device=device), "count": torch.tensor(0.0, device=device)},
                    "0.2_0.4": {"loss_sum": torch.tensor(0.0, device=device), "count": torch.tensor(0.0, device=device)},
                    "0.4_0.6": {"loss_sum": torch.tensor(0.0, device=device), "count": torch.tensor(0.0, device=device)},
                    "0.6_0.8": {"loss_sum": torch.tensor(0.0, device=device), "count": torch.tensor(0.0, device=device)},
                    "0.8_1.0": {"loss_sum": torch.tensor(0.0, device=device), "count": torch.tensor(0.0, device=device)},
                },
                "dinolocal2rgb": {
                    "0.0_0.2": {"loss_sum": torch.tensor(0.0, device=device), "count": torch.tensor(0.0, device=device)},
                    "0.2_0.4": {"loss_sum": torch.tensor(0.0, device=device), "count": torch.tensor(0.0, device=device)},
                    "0.4_0.6": {"loss_sum": torch.tensor(0.0, device=device), "count": torch.tensor(0.0, device=device)},
                    "0.6_0.8": {"loss_sum": torch.tensor(0.0, device=device), "count": torch.tensor(0.0, device=device)},
                    "0.8_1.0": {"loss_sum": torch.tensor(0.0, device=device), "count": torch.tensor(0.0, device=device)},
                },
            }

    # ── MoE router aux loss ───────────────────────────────────────────────
    router_aux = loss_dict.get("router_aux", None)
    router_aux_weight = float(os.environ.get("HUNYUAN_MOE_AUX_LOSS_WEIGHT", "0.0"))
    if isinstance(router_aux, torch.Tensor):
        loss_dict["router_aux"] = router_aux.detach()
        if router_aux_weight != 0.0:
            loss = loss + router_aux * router_aux_weight
    else:
        loss_dict["router_aux"] = torch.tensor(0.0, device=device)

    _dbg("return")
    return loss, total_ce_tokens, total_mse_tokens


# ─────── Training Step Logging ─────────────────────────────────────────────────

def log_training_step(
    curr_step, loss_dict, training_args, optimizer,
    total_tokens_accumulated, total_seq_tokens_accumulated,
    total_samples_accumulated, total_epoch_samples_accumulated,
    total_dataset_samples,
    total_ce_tokens, total_mse_tokens, total_norm,
    start_time, logger, device,
    tokens_per_interval=0,
    samples_per_interval=0,
    model_flops_per_token=0,
):
    """
    Log losses, per-modality breakdowns, throughput, and memory to console and
    W&B.

    Returns: new *start_time* for the next logging interval.
    """
    torch.cuda.synchronize()
    end_time = time()
    steps_per_sec = training_args.log_every / (end_time - start_time)
    modality_id_to_name = _load_modality_id_to_name(training_args)

    message = f"(step={curr_step:07d}) "
    wandb_log = {}

    for key, value in loss_dict.items():
        if key in ["mse", "ce", "router_aux"]:
            avg_loss = torch.tensor(value.item(), device=device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / dist.get_world_size()
            if key == "router_aux":
                message += f"Loss {key}: {avg_loss:.4f}, "
            else:
                message += f"Loss {key}: {avg_loss:.4f}, "
            wandb_log[key] = avg_loss

    # Per-modality losses
    for prefix, per_mod in [("mse", loss_dict.get("mse_per_modality", {})),
                             ("ce", loss_dict.get("ce_per_modality", {}))]:
        if per_mod:
            for mod_name, loss_tensor in per_mod.items():
                if loss_tensor.numel() > 0 and not mod_name.endswith("_timesteps"):
                    avg = loss_tensor.mean().item()
                    pretty_name = _pretty_modality_key(mod_name, modality_id_to_name)
                    message += f"{prefix.upper()} {pretty_name}: {avg:.4f} ({loss_tensor.numel()} tok), "
                    wandb_log[f"loss/{prefix}_{pretty_name}"] = avg
                    wandb_log[f"loss/{prefix}_tokens_{pretty_name}"] = loss_tensor.numel()

    if training_args.log_rgb_condition_loss_group:
        rgb_route_group = loss_dict.get("rgb_route_group", None)
        if rgb_route_group:
            for route_name, stats in rgb_route_group.items():
                loss_sum = stats["loss_sum"].to(device)
                count = stats["count"].to(device=device, dtype=torch.float32)
                dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
                if count.item() > 0:
                    wandb_log[f"route_rgb/loss_{route_name}"] = (loss_sum / count).item()

        rgb_timestep_bins_by_route = loss_dict.get("rgb_timestep_bins_by_route", None)
        if rgb_timestep_bins_by_route:
            for route_name, route_bins in rgb_timestep_bins_by_route.items():
                for bin_name, stats in route_bins.items():
                    loss_sum = stats["loss_sum"].to(device)
                    count = stats["count"].to(device=device, dtype=torch.float32)
                    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(count, op=dist.ReduceOp.SUM)
                    wandb_log[f"route_rgb/bin_count_{route_name}_{bin_name}"] = count.item()
                    if count.item() > 0:
                        wandb_log[f"route_rgb/bin_loss_{route_name}_{bin_name}"] = (loss_sum / count).item()

    elapsed = end_time - start_time
    wps = tokens_per_interval / elapsed if elapsed > 0 else 0.0

    # MFU: peak BF16 TFLOPs/GPU configurable via env (default = GH200 989 TFLOPS)
    peak_tflops_per_gpu = float(os.environ.get("HUNYUAN_PEAK_TFLOPS_PER_GPU", "989"))
    world_size = dist.get_world_size()
    mfu = (wps * model_flops_per_token) / (world_size * peak_tflops_per_gpu * 1e12) if model_flops_per_token > 0 else 0.0
    tflops_per_gpu = (wps * model_flops_per_token) / (world_size * 1e12) if model_flops_per_token > 0 else 0.0

    # Log LR(s): if multiple param groups (e.g. base + embed_head), show all.
    lr_strs = []
    for pg in optimizer.param_groups:
        gn = pg.get("group_name", f"g{len(lr_strs)}")
        lr_strs.append(f"{gn}={pg['lr']:.2e}")
    message += f"GradNorm: {total_norm.item():.4f}, LR: {'/'.join(lr_strs)}, "
    message += f"Steps/Sec: {steps_per_sec:.2f}, WPS: {wps/1e3:.1f}K, TFLOPS/GPU: {tflops_per_gpu:.1f}, MFU: {mfu*100:.2f}%"
    logger.info(message)

    iter_time = elapsed / training_args.log_every  # seconds per step
    tokens_per_gpu_per_step = tokens_per_interval / training_args.log_every / world_size
    samples_per_sec = samples_per_interval / elapsed if elapsed > 0 else 0.0
    wandb_log["throughput/wps"] = wps
    wandb_log["throughput/tokens_per_sec_per_gpu"] = wps / world_size
    wandb_log["throughput/tflops_per_gpu"] = tflops_per_gpu
    wandb_log["throughput/mfu"] = mfu
    wandb_log["throughput/iter_time_sec"] = iter_time
    wandb_log["throughput/batch_tokens_per_gpu"] = tokens_per_gpu_per_step
    wandb_log["throughput/samples_per_sec"] = samples_per_sec
    wandb_log["lr"] = optimizer.param_groups[0]["lr"]
    # Per-group LR (base / embed_head) so the warmup schedule is visible per group.
    for pg in optimizer.param_groups:
        gn = pg.get("group_name", None)
        if gn:
            wandb_log[f"lr/{gn}"] = pg["lr"]
    wandb_log["totals/tokens"] = total_tokens_accumulated
    wandb_log["totals/seq_tokens"] = total_seq_tokens_accumulated
    wandb_log["totals/samples"] = total_samples_accumulated
    wandb_log["totals/epoch_samples"] = total_epoch_samples_accumulated
    wandb_log["totals/epoch"] = total_epoch_samples_accumulated / total_dataset_samples
    wandb_log["total_mse_tokens"] = total_mse_tokens.item()
    wandb_log["total_ce_tokens"] = total_ce_tokens.item()
    wandb_log["total_norm"] = total_norm.item()

    mem_allocated = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
    dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
    wandb_log["mem_allocated"] = mem_allocated.item()

    if dist.get_rank() == 0:
        wandb.log(wandb_log, step=curr_step)

    return time()


# ─────── Memory Monitoring ─────────────────────────────────────────────────────

def system_memory_usage_gb() -> float:
    """Return system memory usage in GB (Linux only)."""
    meminfo = {}
    with open("/proc/meminfo", "r") as f:
        for line in f:
            parts = line.split()
            meminfo[parts[0].rstrip(":")] = int(parts[1])
    mem_total = meminfo["MemTotal"]
    mem_available = meminfo["MemAvailable"]
    mem_used = mem_total - mem_available
    return mem_used / 1024 / 1024
