"""Online validation hooks for the 13-modality BAGEL training loop.

Used from `train.py`. Backward-compatible: every entry point returns None /
no-op when the validation pack is missing or the config flag is off, so any
existing run that didn't ask for online validation is unaffected.

Current v1 metric (cheap, single-rank-friendly):
- **dino_global cos sim**: For each val rgb image, do a teacher-forced forward
  pass on the FSDP-wrapped model to obtain top-1 dino codebook tokens given
  the rgb condition, decode them via the dino VQVAE tokenizer to a 768-dim
  feature, and compute cosine similarity against the pre-extracted reference
  DINOv2-ViT-B/14 global feature from the val pack.

Future extensions (todo):
- Add other codebook modalities (dinolocal/clip/imagebind/imagebindlocal) once
  their reference encoders are dumped into the val pack.
- Add a real 50-step diffusion-generation eval (`run_tier2_generation`) for
  rgb → depth → AbsRel (tracks the active NYU AbsRel goal directly).

All validation is sharded across world ranks for parallelism. Results are
reduced to rank 0 for wandb logging.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F


def load_validation_pack(path: Optional[str], logger=None) -> Optional[dict]:
    """Load the val pack from disk. Returns None if path is empty / missing."""
    if path is None or path == "" or not os.path.exists(path):
        if logger is not None and dist.is_initialized() and dist.get_rank() == 0:
            logger.info(
                f"[online_val] no validation pack at '{path}', online validation disabled"
            )
        return None
    pack = torch.load(path, map_location="cpu", weights_only=False)
    if logger is not None and dist.is_initialized() and dist.get_rank() == 0:
        n = pack["dino_global_feats"].shape[0]
        logger.info(
            f"[online_val] loaded val pack from {path}: N={n}, "
            f"ref={pack.get('ref_encoder', '?')}"
        )
    return pack


@torch.no_grad()
def run_online_validation(
    *,
    fsdp_model,
    vae_model,
    val_pack: Optional[dict],
    inferencer=None,
    dino_tokenizer=None,
    step: int,
    device,
    logger=None,
) -> Optional[dict]:
    """Run the cheap online validation hook. Returns dict of metric → value.

    No-op (returns None) if val_pack is missing or inferencer not available.
    Caller is responsible for filtering by `step % validate_every == 0`.

    Sharding:
    - val_pack has N images. Each rank handles ceil(N / world) of them.
    - Per-rank features are averaged via all_reduce, then rank 0 returns the
      global mean. Non-rank-0 receive the same dict so caller can broadcast.
    """
    if val_pack is None:
        return None
    if inferencer is None or dino_tokenizer is None:
        # Hard requirement: caller must pass a usable inferencer + dino tokenizer.
        # Stay quiet by default but log once for transparency.
        if logger is not None and dist.is_initialized() and dist.get_rank() == 0:
            logger.info(
                f"[online_val] step {step}: skipped (inferencer or dino_tokenizer missing)"
            )
        return None

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    rgb_paths = val_pack["rgb_paths"]
    ref_feats = val_pack["dino_global_feats"].to(device).float()  # (N, 768)
    n = len(rgb_paths)
    # Per-rank slice — deterministic so resume gives the same split.
    per_rank = (n + world_size - 1) // world_size
    start = rank * per_rank
    end = min(start + per_rank, n)
    local_idx = list(range(start, end))

    t0 = time.time()
    local_cos_sum = torch.zeros(1, device=device, dtype=torch.float32)
    local_n = torch.zeros(1, device=device, dtype=torch.float32)

    fsdp_model_was_training = fsdp_model.training
    fsdp_model.eval()
    # Free any cached gradient/activation memory from the just-finished
    # train step so the inference forward pass has room to all-gather.
    fsdp_model.zero_grad(set_to_none=True)
    import gc as _gc
    _gc.collect()
    torch.cuda.empty_cache()

    # The inferencer's gen path bypasses FSDP root forward (calls e.g.
    # self.model.forward_cache_update_vae(...) which then touches
    # self.language_model.model.embed_tokens.weight directly), so FSDP
    # never all-gathers the sharded weights → "tensor data not allocated".
    # summon_full_params with offload_to_cpu=True materialises everything
    # but parks it on CPU instead of GPU to avoid OOM. Inference is much
    # slower but it actually runs.
    from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP
    _summon_cm = None
    try:
        _summon_cm = _FSDP.summon_full_params(
            fsdp_model, writeback=False, offload_to_cpu=True
        )
        _summon_cm.__enter__()
    except Exception as _e:
        if rank == 0 and logger is not None:
            logger.warning(
                f"[online_val] step {step}: summon_full_params failed: {_e}; "
                "proceeding without all-gather (val will be no-op)"
            )
        _summon_cm = None
    _printed_traceback = False
    try:
        for img_i in local_idx:
            try:
                # Lazy import to avoid bloating top-of-train import time when
                # validation is disabled.
                from PIL import Image
                img = Image.open(rgb_paths[img_i]).convert("RGB")
                # The inferencer's dino-generation path takes a PIL image and
                # returns either dino token ids or already-decoded features
                # depending on its configuration. We rely on the same call
                # path used by any2any/eval/dino_global/eval_cos_sim.py.
                #
                # NOTE: To keep this hook minimal and avoid wiring through all
                # the inferencer args, we delegate to a small helper attribute
                # set up by the caller (`inferencer.online_val_generate_dino`).
                # If that attribute isn't present, skip this image.
                gen_fn = getattr(inferencer, "online_val_generate_dino", None)
                if gen_fn is None:
                    raise RuntimeError(
                        "inferencer.online_val_generate_dino not set up; "
                        "wire it from train.py before enabling online validation"
                    )
                pred_feat = gen_fn(img)  # expected (768,) torch.float on device
                if pred_feat is None:
                    continue
                cos = F.cosine_similarity(
                    pred_feat.float().view(1, -1),
                    ref_feats[img_i].view(1, -1),
                ).item()
                local_cos_sum += float(cos)
                local_n += 1.0
            except Exception as e:
                if rank == 0 and logger is not None:
                    logger.warning(
                        f"[online_val] step {step}: image {img_i} failed: {e}"
                    )
                    if not _printed_traceback:
                        import traceback as _tb
                        logger.warning(
                            f"[online_val] first-image traceback:\n"
                            + _tb.format_exc()
                        )
                        _printed_traceback = True
                continue
    finally:
        if _summon_cm is not None:
            try:
                _summon_cm.__exit__(None, None, None)
            except Exception:
                pass
        torch.cuda.empty_cache()
        if fsdp_model_was_training:
            fsdp_model.train()

    # Reduce across ranks.
    if dist.is_initialized():
        dist.all_reduce(local_cos_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_n, op=dist.ReduceOp.SUM)
    mean_cos = float(local_cos_sum.item() / max(local_n.item(), 1.0))
    elapsed = time.time() - t0

    metrics = {
        "dino_global_cos_sim": mean_cos,
        "n_samples_used": int(local_n.item()),
        "elapsed_sec": elapsed,
    }
    if rank == 0 and logger is not None:
        logger.info(
            f"[online_val] step {step}: dino_global_cos_sim={mean_cos:.4f} "
            f"({int(local_n.item())} samples, {elapsed:.1f}s)"
        )
    return metrics
