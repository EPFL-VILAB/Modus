"""
HunyuanImageWrapper: bridges MODUS training contract to HunyuanImage3ForCausalMM.

MODUS calls fsdp_model(**data) where data contains packed-sequence keys.
This wrapper:
  1. Translates MODUS packed-sequence format → HunyuanImage-3.0 internal format
  2. Runs Siglip2 ViT on packed understanding tokens (Bagel format)
  3. Applies flow-matching noise to clean VAE latents
  4. Calls model components directly (embed → inject ViT/VAE → transformer → heads)
  5. Computes CE + MSE losses and returns {"ce", "mse", "ce_per_modality", "mse_per_modality"}

The forward is decomposed (not a black-box call to HunyuanImage3ForCausalMM.forward)
because MODUS uses variable-length packed sequences for ViT tokens, which is
incompatible with HunyuanImage3's native instantiate_vit_image_tokens (which assumes
fixed-padded-length images).

Attributes required by train_utils.py:
  self.language_model            - used for freeze_llm / resize_token_embeddings
  self.vit_model                 - Siglip2VisionTransformer, used for freeze_vit
  self.config.llm_config.vocab_size
  self.language_model.config.vocab_size
  self.set_modality_registry()
"""

from __future__ import annotations

import contextvars
import os
import types
from types import SimpleNamespace
from typing import List, Optional, Tuple, Dict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from modeling.bagel.modeling_utils import LearnableEmbedding
from data.data_utils import patchify


_NATIVE_SIZE_TIERS = [256, 512, 768, 1024]


def _native_size_ratio(width, height, align, return_resolution=False):
    """Nearest native (size-tier, aspect-ratio) bucket for an image, mirroring
    the dataset's _pick_native_size_ratio so a condition block emitted at
    inference carries the same <img_size_*>/<img_ratio_*> pair training used."""
    import numpy as np
    from modeling.hunyuan_image_3.src.tokenization_hunyuan_image_3 import ResolutionGroup

    eff = (width * height) ** 0.5
    base_size = min(_NATIVE_SIZE_TIERS, key=lambda b: abs(b - eff))
    group = ResolutionGroup(base_size=base_size, align=align)
    ratio_idx = int(np.argmin(np.abs(group.ratio - (height / width))))
    if return_resolution:
        reso = group.data[ratio_idx]
        return base_size, ratio_idx, int(reso.w), int(reso.h)
    return base_size, ratio_idx


def _require_timestep_scale(var_hint: str) -> float:
    """Resolve the flow-matching timestep scale, or fail loudly.

    Training uses sigma * 1000 (the native convention, validated in v55: only
    ckpt + scale=1000 produced clean native T2I; every other combination was
    noise). The old default of 1.0 meant an eval that forgot to export the var
    silently fed t in [0, 1] to a model trained on [0, 1000] -- which looks
    exactly like "the model drifted", and cost us several bogus drift reports.

    Silence is the bug, so there is no default. Set the scale explicitly:
    HUNYUAN_TRAIN_TIMESTEP_SCALE=1000 for any MODUS-trained checkpoint, or
    HUNYUAN_INFER_TIMESTEP_SCALE=1 to deliberately run raw native weights.
    """
    raw = os.environ.get("HUNYUAN_INFER_TIMESTEP_SCALE")
    if raw is None:
        raw = os.environ.get("HUNYUAN_TRAIN_TIMESTEP_SCALE")
    if raw is None:
        raise RuntimeError(
            "HunyuanImage-3 timestep scale is unset. Export "
            "HUNYUAN_TRAIN_TIMESTEP_SCALE=1000 (every MODUS hunyuan run trains "
            "at 1000), or HUNYUAN_INFER_TIMESTEP_SCALE=1 to run raw native "
            f"weights on purpose. Refusing to guess -- the old 1.0 default "
            f"silently produced noise that reads as model drift. [{var_hint}]"
        )
    return float(raw)



def _image_block_rope_matches_training() -> bool:
    """Should an image block occupy ONE rope position, as training packs it?

    `data/dataset_base.py:786` (vit) and `:893` (vae) give every token of an
    image block the SAME `curr_rope_id` and then advance by 1. The inference
    builders instead hand out `arange(start, start+seq_len)` and advance by
    `seq_len`, so after a single condition image the rope counter is ~1574
    instead of ~2 and the codebook target block is decoded at positions the
    model never saw in training. Off by default so the flag can be A/B'd
    against the current behaviour before anything ships.
    """
    return os.environ.get("HUNYUAN_INFER_IMAGE_ROPE_BLOCK", "0") == "1"

def _debug_dump_codebook_logits(
    *,
    prefix: str,
    step: int,
    logits: Tensor,
    chosen_tokens: Optional[Tensor],
    code_base: int,
    code_length: int,
    topk: int,
) -> None:
    if os.environ.get("MODUS_DUMP_CODEBOOK_LOGITS", "0") != "1":
        return

    steps_env = os.environ.get(
        "MODUS_DUMP_CODEBOOK_LOGIT_STEPS",
        "0,1,2,3,4,5,10,31,32,63,64,127,255,511,1023",
    )
    try:
        wanted_steps = {int(item) for item in steps_env.split(",") if item.strip()}
    except ValueError:
        wanted_steps = {0}
    if step not in wanted_steps:
        return

    with torch.no_grad():
        local_logits = logits[:, code_base:code_base + code_length].detach().float()
        local_probs = torch.softmax(local_logits, dim=-1)
        k = min(int(topk), local_logits.size(-1))
        top_vals, top_ids = torch.topk(local_logits, k, dim=-1)
        top_probs = local_probs.gather(1, top_ids)
        entropy = -(local_probs * torch.log(local_probs.clamp_min(1e-30))).sum(dim=-1)
        norm_entropy = entropy / torch.log(torch.tensor(float(code_length), device=entropy.device))
        margin = top_vals[:, 0] - top_vals[:, 1] if k > 1 else torch.full_like(top_vals[:, 0], float("nan"))

        probe_local_id = 3000
        if 0 <= probe_local_id < code_length:
            probe_logits = local_logits[:, probe_local_id]
            probe_probs = local_probs[:, probe_local_id]
            probe_rank = (local_logits > probe_logits.unsqueeze(1)).sum(dim=-1) + 1
            probe_text = (
                f" token3000_rank={probe_rank.cpu().tolist()}"
                f" token3000_prob={[round(x, 6) for x in probe_probs.cpu().tolist()]}"
                f" token3000_logit={[round(x, 4) for x in probe_logits.cpu().tolist()]}"
            )
        else:
            probe_text = ""

        chosen_local = None
        if chosen_tokens is not None:
            chosen_local = (chosen_tokens.detach().cpu() - int(code_base)).tolist()

        print(
            f"[DBG_LOGITS] backend={prefix} step={step} code_base={code_base} "
            f"entropy={[round(x, 4) for x in entropy.cpu().tolist()]} "
            f"norm_entropy={[round(x, 4) for x in norm_entropy.cpu().tolist()]} "
            f"top_margin={[round(x, 4) for x in margin.cpu().tolist()]} "
            f"top_local_ids={top_ids.cpu().tolist()} "
            f"top_probs={[[round(float(v), 6) for v in row] for row in top_probs.cpu().tolist()]} "
            f"top_logits={[[round(float(v), 4) for v in row] for row in top_vals.cpu().tolist()]} "
            f"chosen_local={chosen_local}{probe_text}"
        )


def _debug_dump_ce_logits(
    *,
    prefix: str,
    ce_preds: Tensor,
    labels: Tensor,
    modality_ids: Optional[Tensor],
    modality_registry,
    topk: int,
) -> None:
    if os.environ.get("MODUS_DUMP_CE_LOGITS", "0") != "1":
        return

    positions_env = os.environ.get(
        "MODUS_DUMP_CE_LOGIT_POSITIONS",
        "0,1,2,3,4,5,31,63,127,255,511,1023",
    )
    try:
        positions = [int(item) for item in positions_env.split(",") if item.strip()]
    except ValueError:
        positions = [0]

    with torch.no_grad():
        ce_preds_f = ce_preds.detach().float()
        labels_i = labels.detach().long()
        modality_ids_i = modality_ids.detach().long() if modality_ids is not None else None
        for pos in positions:
            if pos < 0 or pos >= ce_preds_f.size(0):
                continue

            label = int(labels_i[pos].item())
            mid = int(modality_ids_i[pos].item()) if modality_ids_i is not None else None
            spec = None
            if modality_registry is not None and mid is not None:
                try:
                    spec = modality_registry.get_by_id(mid)
                except Exception:
                    spec = None

            if spec is not None and spec.code_token_range is not None:
                base, length = spec.code_token_range
                modality_name = getattr(spec, "name", str(mid))
                if base <= label < base + length:
                    logits = ce_preds_f[pos, base:base + length]
                    local_label = label - base
                    token_offset = base
                else:
                    logits = ce_preds_f[pos]
                    local_label = label
                    token_offset = 0
            else:
                modality_name = str(mid) if mid is not None else "full"
                logits = ce_preds_f[pos]
                local_label = label
                token_offset = 0

            probs = torch.softmax(logits, dim=-1)
            k = min(int(topk), logits.numel())
            top_vals, top_ids = torch.topk(logits, k, dim=-1)
            top_probs = probs.gather(0, top_ids)
            label_logit = logits[local_label]
            label_prob = probs[local_label]
            rank = int((logits > label_logit).sum().item()) + 1
            entropy = -(probs * torch.log(probs.clamp_min(1e-30))).sum()
            norm_entropy = entropy / torch.log(torch.tensor(float(logits.numel()), device=entropy.device))
            top_token_ids = (top_ids + int(token_offset)).cpu().tolist()
            print(
                f"[DBG_CE_LOGITS] backend={prefix} pos={pos} modality={modality_name} "
                f"vocab_base={int(token_offset)} vocab_size={int(logits.numel())} "
                f"label_token={label} label_local={int(local_label)} "
                f"label_rank={rank} label_prob={float(label_prob.item()):.8f} "
                f"label_logit={float(label_logit.item()):.6f} "
                f"entropy={float(entropy.item()):.6f} norm_entropy={float(norm_entropy.item()):.6f} "
                f"top_tokens={top_token_ids} top_local={top_ids.cpu().tolist()} "
                f"top_probs={[round(float(v), 8) for v in top_probs.cpu().tolist()]} "
                f"top_logits={[round(float(v), 6) for v in top_vals.cpu().tolist()]}"
            )


def _debug_dump_ce_argmax_stats(
    *,
    prefix: str,
    ce_preds: Tensor,
    labels: Tensor,
    modality_ids: Optional[Tensor],
    modality_registry,
) -> None:
    """Per-modality teacher-forcing argmax statistics over the CE positions.

    For each modality with a code_token_range, restrict the logits to that
    range, take argmax across each CE position, and report:
      - n_positions          number of CE positions for this modality
      - unique_argmax        number of distinct argmax tokens
      - top1_acc             fraction of positions where argmax == GT label
      - first10_global       first 10 argmax tokens (in global vocab IDs)
      - most_freq_global     most frequent argmax token IDs (top 10)
      - most_freq_counts     their frequencies

    This is the cheapest signal that distinguishes "FSDP forward greedy
    is collapsed (e.g. unique=27, dominated by token 3000)" from "FSDP
    forward greedy is healthy (e.g. unique~300+)". Compare against BAGEL
    AR `unique=383/1024` and Hunyuan standalone-AR `unique=27/1024`.
    """
    if os.environ.get("MODUS_DUMP_CE_ARGMAX", "0") != "1":
        return
    # Default: rank-0 only (one print per call). Override with
    # MODUS_DUMP_CE_ARGMAX_ALL_RANKS=1 to see how argmax differs across TP
    # ranks (ranks have different non-TP weights so logits differ).
    if os.environ.get("MODUS_DUMP_CE_ARGMAX_ALL_RANKS", "0") != "1":
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
    if modality_ids is None or modality_registry is None:
        return

    with torch.no_grad():
        ce_preds_f = ce_preds.detach().float()
        labels_i = labels.detach().long()
        modality_ids_i = modality_ids.detach().long()
        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        for mid in torch.unique(modality_ids_i).cpu().tolist():
            try:
                spec = modality_registry.get_by_id(int(mid))
            except Exception:
                spec = None
            if spec is None or spec.code_token_range is None:
                continue
            modality_name = getattr(spec, "name", str(mid))
            mask = modality_ids_i == int(mid)
            if not mask.any():
                continue
            base, length = spec.code_token_range
            mod_logits = ce_preds_f[mask, base:base + length]
            mod_labels = labels_i[mask] - base
            argmax_local = mod_logits.argmax(dim=-1)
            argmax_global = argmax_local + base
            n_pos = int(mask.sum().item())
            unique_argmax = torch.unique(argmax_local)
            n_unique = int(unique_argmax.numel())
            counts = torch.bincount(argmax_local, minlength=length)
            kfreq = min(10, length)
            top_freq_vals, top_freq_ids = torch.topk(counts, k=kfreq)
            top_freq_ids_global = (top_freq_ids + base).cpu().tolist()
            top_freq_vals_l = top_freq_vals.cpu().tolist()
            first10_global = argmax_global[:min(10, n_pos)].cpu().tolist()
            top1_correct = float((argmax_local == mod_labels).float().mean().item())
            print(
                f"[DBG_CE_ARGMAX] backend={prefix} rank={rank} modality={modality_name} "
                f"vocab_base={int(base)} vocab_size={int(length)} "
                f"n_positions={n_pos} unique_argmax={n_unique} top1_acc={top1_correct:.6f} "
                f"first10_global={first10_global} "
                f"most_freq_global={top_freq_ids_global} "
                f"most_freq_counts={top_freq_vals_l}"
            )


def _debug_dump_activation_fingerprint(
    *,
    prefix: str,
    stage: str,
    hidden_states: Tensor,
    sequence_positions: List[int],
) -> None:
    if os.environ.get("MODUS_DUMP_ACTIVATIONS", "0") != "1":
        return
    if os.environ.get("MODUS_DUMP_ACT_RANK0_ONLY", "1") == "1":
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

    with torch.no_grad():
        h = hidden_states.detach().float()
        if h.dim() == 3:
            h = h[0]
        flat = h.reshape(-1, h.shape[-1])
        global_mean = float(flat.mean().item())
        global_std = float(flat.std(unbiased=False).item())
        global_norm = float(flat.norm().item())
        print(
            f"[DBG_ACT] backend={prefix} stage={stage} "
            f"shape={tuple(hidden_states.shape)} mean={global_mean:.8f} "
            f"std={global_std:.8f} norm={global_norm:.8f}"
        )
        for pos in sequence_positions:
            if pos < 0 or pos >= h.size(0):
                continue
            row = h[pos]
            head = row[:8].cpu().tolist()
            print(
                f"[DBG_ACT_POS] backend={prefix} stage={stage} pos={pos} "
                f"mean={float(row.mean().item()):.8f} "
                f"std={float(row.std(unbiased=False).item()):.8f} "
                f"norm={float(row.norm().item()):.8f} "
                f"head={[round(float(v), 6) for v in head]}"
            )


_MODE_CAUSAL, _MODE_FULL, _MODE_NOISE = 0, 1, 2


def _build_flex_block_mask_v2(sample_lens, split_lens, attn_modes, seqlen, device):
    """BlockMask reproducing prepare_attention_mask_per_sample EXACTLY.

    The old builder returned None whenever per-sample masks existed and, when it
    did run, only expressed "same packed sample" -- it ignored the causal / full
    / noise structure inside a sample, so turning it on silently made text
    bidirectional. This one encodes the real rule, which is (data_utils.py:80-101):

      allowed(q, k)  =  same_sample
                     &  not (mode[k] == noise and split[q] != split[k])
                     &  split[k] <= split[q]
                     &  (split[k] != split[q] or mode[q] != causal or k <= q)

    That predicate was checked element-wise against the dense construction over
    all 120 combinations of 1-4 splits x {causal, full, noise}: exact match.
    Set HUNYUAN_FLEX_VERIFY=1 to re-check it against the dense mask at runtime.
    """
    if not split_lens or not attn_modes:
        return None
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError:
        return None

    split_of = torch.zeros(seqlen, dtype=torch.int32, device=device)
    mode_of = torch.zeros(seqlen, dtype=torch.int32, device=device)
    sample_of = torch.zeros(seqlen, dtype=torch.int32, device=device)
    _codes = {"causal": _MODE_CAUSAL, "full": _MODE_FULL, "noise": _MODE_NOISE}
    off = 0
    for gi, (sl, md) in enumerate(zip(split_lens, attn_modes)):
        end = min(off + int(sl), seqlen)
        if end > off:
            split_of[off:end] = gi
            mode_of[off:end] = _codes.get(md, _MODE_FULL)
        off = end
        if off >= seqlen:
            break
    off = 0
    for si, sl in enumerate(sample_lens):
        end = min(off + int(sl), seqlen)
        if end > off:
            sample_of[off:end] = si
        off = end
        if off >= seqlen:
            break

    def mask_mod(b, h, q_idx, kv_idx):
        same_sample = sample_of[q_idx] == sample_of[kv_idx]
        iq, ik = split_of[q_idx], split_of[kv_idx]
        visible = ~((mode_of[kv_idx] == _MODE_NOISE) & (iq != ik))
        ordered = ik <= iq
        intra = (ik != iq) | (mode_of[q_idx] != _MODE_CAUSAL) | (kv_idx <= q_idx)
        return same_sample & visible & ordered & intra

    return create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=seqlen, KV_LEN=seqlen, device=device,
    )


def _verify_flex_matches_dense(sample_lens, nested_attention_masks, split_lens,
                               attn_modes, seqlen, device, logger=None):
    """Element-wise check that the flex predicate equals the dense mask.

    Only runs under HUNYUAN_FLEX_VERIFY=1 -- it materialises the dense mask,
    which is the very thing flex exists to avoid.
    """
    if nested_attention_masks is None or not split_lens:
        return
    _codes = {"causal": _MODE_CAUSAL, "full": _MODE_FULL, "noise": _MODE_NOISE}
    split_of = torch.zeros(seqlen, dtype=torch.long, device=device)
    mode_of = torch.zeros(seqlen, dtype=torch.long, device=device)
    sample_of = torch.zeros(seqlen, dtype=torch.long, device=device)
    off = 0
    for gi, (sl, md) in enumerate(zip(split_lens, attn_modes)):
        end = min(off + int(sl), seqlen)
        split_of[off:end] = gi; mode_of[off:end] = _codes.get(md, _MODE_FULL); off = end
    off = 0
    for si, sl in enumerate(sample_lens):
        end = min(off + int(sl), seqlen); sample_of[off:end] = si; off = end

    q = torch.arange(seqlen, device=device).unsqueeze(1)
    k = torch.arange(seqlen, device=device).unsqueeze(0)
    iq, ik = split_of[q], split_of[k]
    pred = ((sample_of[q] == sample_of[k])
            & ~((mode_of[k] == _MODE_NOISE) & (iq != ik))
            & (ik <= iq)
            & ((ik != iq) | (mode_of[q] != _MODE_CAUSAL) | (k <= q)))

    dense = torch.zeros((seqlen, seqlen), dtype=torch.bool, device=device)
    off = 0
    for i, sl in enumerate(sample_lens):
        sl = int(sl)
        dense[off:off + sl, off:off + sl] = nested_attention_masks[i].to(device) > -1e30
        off += sl
    n_bad = int((pred != dense).sum().item())
    msg = (f"[FLEX_VERIFY] seqlen={seqlen} mismatching entries={n_bad} "
           f"({100.0 * n_bad / (seqlen * seqlen):.4f}%)")
    if logger is not None:
        logger.info(msg)
    else:
        print(msg, flush=True)
    if n_bad:
        raise RuntimeError(
            f"flex mask_mod does not match the dense mask ({n_bad} entries). "
            f"Refusing to train on a different attention pattern.")


def _build_flex_block_mask(sample_lens, nested_attention_masks, seqlen, device):
    """Build a FlexAttention BlockMask for packed samples.

    Returns None when FlexAttention cannot or should not be used (e.g. when
    per-sample custom attention masks are provided, which encode intra-sample
    structure beyond a simple block-diagonal pattern).

    When *nested_attention_masks* is None every sample gets full self-attention
    within its block and no cross-sample attention — i.e. a block-diagonal mask.
    This is the common training case.

    The returned BlockMask is compiled and cached by PyTorch's flex_attention
    machinery; subsequent calls with the same seqlen reuse the compiled kernel.
    """
    if nested_attention_masks is not None:
        # Per-sample custom masks: fall back to dense attn_mask in caller.
        return None

    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError:
        return None

    # Build a 1-D tensor mapping each token position → its sample index.
    sample_tok = torch.zeros(seqlen, dtype=torch.int32, device=device)
    offset = 0
    for samp_i, slen in enumerate(sample_lens):
        sample_tok[offset:offset + slen] = samp_i
        offset += slen

    def mask_mod(b, h, q_idx, kv_idx):
        # Allow attention only within the same packed sample.
        return sample_tok[q_idx] == sample_tok[kv_idx]

    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=seqlen,
        KV_LEN=seqlen,
        device=device,
    )


class HunyuanImageWrapper(nn.Module):
    """
    Wraps HunyuanImage3ForCausalMM with the MODUS training interface.

    Args:
        hunyuan_model: an instantiated HunyuanImage3ForCausalMM
        vit_max_num_patch_per_side: grid size used by the MODUS data pipeline to
            encode ViT position IDs as ``y * max_per_side + x``.  Defaults to 70
            (matching Bagel's default).
    """

    def __init__(self, hunyuan_model: nn.Module, vit_max_num_patch_per_side: int = 70):
        super().__init__()
        self.hunyuan_model = hunyuan_model

        # ── Attributes required by train_utils.py ────────────────────────────
        # Use object.__setattr__ to store these as plain Python attributes
        # (not registered nn.Module submodules).  hunyuan_model is already
        # registered above as self.hunyuan_model; re-registering the same
        # object under a second name (language_model, vit_model) creates
        # module aliases that confuse FSDP's _recursive_wrap — it wraps
        # hunyuan_model's children, then finds them already-wrapped when it
        # recurses into the alias, triggering an AssertionError.
        # NOTE: language_model points to the full HunyuanImage3ForCausalMM (including ViT,
        # patch_embed, final_layer, etc.), not just the LLM backbone. freeze_llm=True would
        # freeze ALL parameters. Keep freeze_llm: false for HunyuanImage-3.
        object.__setattr__(self, "language_model", hunyuan_model)

        # vit_model: used for freeze_vit
        object.__setattr__(self, "vit_model", getattr(hunyuan_model, "vision_model", None))

        # config: train_utils accesses model.config.llm_config.vocab_size
        vocab_size = hunyuan_model.config.vocab_size
        num_hidden_layers = getattr(hunyuan_model.config, "num_hidden_layers", None)
        llm_cfg = SimpleNamespace(
            vocab_size=vocab_size,
            num_hidden_layers=num_hidden_layers,
        )
        self.config = SimpleNamespace(llm_config=llm_cfg)

        # Image token placeholder ID (positions are overwritten by patch_embed)
        self.image_token_id: int = hunyuan_model.config.image_token_id  # typically 128006 for HunyuanImage-3
        self.hidden_size: int = int(hunyuan_model.config.hidden_size)

        # VAE latent channels (from config)
        vae_cfg = hunyuan_model.config.vae
        self.latent_channels: int = (
            vae_cfg["latent_channels"] if isinstance(vae_cfg, dict) else int(vae_cfg.latent_channels)
        )
        self.latent_channel: int = self.latent_channels
        self.latent_patch_size: int = 1
        self.latent_downsample: int = 16
        vae_sample_size = vae_cfg.get("sample_size", 1024) if isinstance(vae_cfg, dict) else getattr(vae_cfg, "sample_size", 1024)
        self.max_latent_size: int = max(1, int(vae_sample_size) // self.latent_downsample)

        # ViT position-ID decoding parameter (from MODUS data pipeline config)
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.vit_patch_size = int(
            getattr(getattr(hunyuan_model, "vision_model", None), "config", SimpleNamespace()).patch_size
            if getattr(getattr(hunyuan_model, "vision_model", None), "config", None) is not None
            else 16
        )

        # Modality registry (set externally via set_modality_registry)
        self.modality_registry = None
        self._codebook_rope_warn_count = 0

        # Patch embedding accessors onto hunyuan_model so that
        # transformers' resize_token_embeddings (called via
        # m.language_model.resize_token_embeddings) can find them.
        _patch_embedding_accessors(hunyuan_model)

    # ── Modality registry ─────────────────────────────────────────────────────

    def set_modality_registry(self, registry) -> None:
        """Called by train_utils.setup_fsdp_and_load_checkpoint before FSDP wrap."""
        self.modality_registry = registry
        if self.modality_registry is None:
            return

        try:
            ref_param = next(self.hunyuan_model.parameters())
            target_device = ref_param.device
            target_dtype = ref_param.dtype
        except StopIteration:
            target_device = None
            target_dtype = None

        for spec in self.modality_registry.modalities_with_forward_pos_embed():
            if spec.pos_embed_size is None:
                continue
            attr_name = f"{spec.pos_embed_name or spec.name}_pos_embed"
            if hasattr(self.hunyuan_model, attr_name):
                continue
            embed = LearnableEmbedding(int(spec.pos_embed_size), self.hidden_size)
            init_mode = os.environ.get("HUNYUAN_FORWARD_POS_EMBED_INIT", "").strip().lower()
            init_std = os.environ.get("HUNYUAN_FORWARD_POS_EMBED_INIT_STD", "").strip()
            if init_mode or init_std:
                with torch.no_grad():
                    if init_mode in {"zero", "zeros"}:
                        embed.embedding.zero_()
                    elif init_std:
                        embed.embedding.normal_(mean=0.0, std=float(init_std))
                    elif init_mode not in {"", "default"}:
                        raise ValueError(
                            "HUNYUAN_FORWARD_POS_EMBED_INIT must be 'default' or 'zero' "
                            f"when set, got {init_mode!r}"
                        )
            if target_device is not None:
                embed = embed.to(device=target_device, dtype=target_dtype)
            setattr(self.hunyuan_model, attr_name, embed)

    @property
    def device(self):
        return next(self.hunyuan_model.parameters()).device

    def _require_single_sample(self, values, name: str):
        if len(values) != 1:
            raise NotImplementedError(f"Hunyuan inference currently supports one sample at a time, got {name}={values}")

    def _build_query_position_ids(self, curr_position_id: int, seq_len: int, device: torch.device) -> Tensor:
        return torch.arange(curr_position_id, curr_position_id + seq_len, device=device, dtype=torch.long).unsqueeze(0)

    def _resolve_forward_pos_embed_module(self, modality_name: Optional[str]):
        """Resolve a learnable forward-pos-embed module for a modality.

        Kept for checkpoint compatibility, but Hunyuan training currently does
        not consume these embeddings in the main forward path. Inference should
        therefore not use them by default; otherwise it diverges from training.
        """
        if modality_name is None or self.modality_registry is None:
            return None
        try:
            spec = self.modality_registry.get(modality_name)
        except Exception:
            return None
        attr_name = f"{getattr(spec, 'pos_embed_name', None) or spec.name}_pos_embed"
        return getattr(self.hunyuan_model, attr_name, None)

    def _use_forward_pos_embed_in_inference(self) -> bool:
        """Whether to apply codebook forward-pos-embed in Hunyuan inference.

        The base Hunyuan path defaults to disabled, while MODUS-Hunyuan
        checkpoints trained with this feature must set the same environment
        flag during inference.  The release runtime profile pins it to 1.
        """
        return os.environ.get("HUNYUAN_USE_FORWARD_CODEBOOK_POS_EMBED", "0") == "1"

    def _build_prefill_attention_mask(
        self,
        *,
        position_ids: Tensor,
        full_attention_slices: Optional[List[slice]] = None,
    ) -> Tensor:
        """Build a native-Hunyuan-style prefill attention mask for a chunk.

        Text tokens remain causal with respect to absolute positions.
        Tokens inside any slice in ``full_attention_slices`` get full attention
        within that slice, matching Hunyuan's image-block behavior during
        gen_text prefilling.
        """
        if position_ids.dim() != 2 or position_ids.size(0) != 1:
            raise NotImplementedError(f"Expected position_ids shape (1, q_len), got {tuple(position_ids.shape)}")

        pos = position_ids[0]
        q_len = pos.numel()
        kv_len = int(pos[-1].item()) + 1
        kv_positions = torch.arange(kv_len, device=pos.device, dtype=pos.dtype)

        allowed = kv_positions.unsqueeze(0) <= pos.unsqueeze(1)
        if full_attention_slices:
            for full_slice in full_attention_slices:
                start = int(full_slice.start)
                stop = int(full_slice.stop)
                if stop <= start:
                    continue
                q_in_block = (pos >= start) & (pos < stop)
                kv_in_block = (kv_positions >= start) & (kv_positions < stop)
                if q_in_block.any() and kv_in_block.any():
                    allowed[q_in_block] |= kv_in_block.unsqueeze(0)

        return allowed.unsqueeze(0).unsqueeze(0)

    def _build_codebook_rope_info(
        self,
        *,
        start_pos: int,
        total_tokens: int,
        target_name: Optional[str],
    ):
        # Diagnostic gate: force 1D arange RoPE for the codebook output region.
        # Used to test whether the 2D codebook RoPE during AR generation is the
        # source of token-3000 mode collapse (job 1997032: step 0 token3000
        # rank=4150, step 2 rank=1; step 32 logits ≈ step 1 logits, suggesting
        # 2D RoPE is broken or mismatched between training and inference).
        if os.environ.get("HUNYUAN_DISABLE_AR_2D_ROPE", "0") == "1":
            return None
        if target_name is None or self.modality_registry is None:
            return None
        try:
            spec = self.modality_registry.get(target_name)
        except Exception:
            return None
        if spec.codebook_spatial_shape is None:
            return None
        h, w = spec.codebook_spatial_shape
        if h * w != total_tokens:
            return None
        return [[(slice(start_pos, start_pos + total_tokens), (h, w))]]

    def _run_decoder_with_cache(
        self,
        *,
        inputs_embeds: Tensor,
        position_ids: Tensor,
        past_key_values,
        attention_mask: Optional[Tensor] = None,
        rope_image_info=None,
    ):
        cos, sin = self.hunyuan_model.cached_rope(
            self.hunyuan_model.config.max_position_embeddings,
            device=inputs_embeds.device,
            rope_image_info=rope_image_info,
            position_ids=position_ids,
        )
        outputs = self.hunyuan_model.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            return_dict=True,
            custom_pos_emb=(cos, sin),
            mode="gen_text",
            first_step=bool(position_ids[0, 0].item() == 0),
        )
        return outputs

    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids, modality_type=None, raw=False):
        self._require_single_sample(curr_kvlens, "curr_kvlens")
        self._require_single_sample(curr_rope, "curr_rope")
        self._require_single_sample(prompts, "prompts")

        start_key = "bos_token_id"
        end_key = "eos_token_id"
        if modality_type is not None and self.modality_registry is not None:
            try:
                start_key = self.modality_registry.start_token_key(modality_type)
                end_key = self.modality_registry.end_token_key(modality_type)
            except Exception:
                pass

        text_ids = list(tokenizer.encode(prompts[0]))
        if not raw:
            text_ids = [int(new_token_ids[start_key])] + text_ids + [int(new_token_ids[end_key])]
        curr_kvlen = int(curr_kvlens[0])
        curr_position_id = int(curr_rope[0])
        generation_input = {
            "packed_text_ids": torch.tensor(text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.arange(
                curr_position_id, curr_position_id + len(text_ids), dtype=torch.long
            ),
            "text_token_lens": torch.tensor([len(text_ids)], dtype=torch.int),
            "packed_text_indexes": torch.arange(curr_kvlen, curr_kvlen + len(text_ids), dtype=torch.long),
            "packed_key_value_indexes": torch.arange(curr_kvlen, dtype=torch.long),
            "key_values_lens": torch.tensor([curr_kvlen], dtype=torch.int),
        }
        return generation_input, [curr_kvlen + len(text_ids)], [curr_position_id + len(text_ids)]

    @torch.no_grad()
    def forward_cache_update_text(
        self,
        past_key_values,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        pos_embed_key: Optional[str] = None,
    ):
        del packed_text_indexes, packed_key_value_indexes, key_values_lens
        if int(text_token_lens.numel()) != 1:
            raise NotImplementedError("Hunyuan text cache update currently supports one sample.")

        device = self.device
        packed_text_ids = packed_text_ids.to(device)
        packed_text_position_ids = packed_text_position_ids.to(device)
        text_embeds = self.hunyuan_model.model.wte(packed_text_ids)

        if pos_embed_key is not None and self._use_forward_pos_embed_in_inference():
            pos_embed_mod = self._resolve_forward_pos_embed_module(pos_embed_key)
            if pos_embed_mod is not None and self.modality_registry is not None:
                try:
                    spec = self.modality_registry.get(pos_embed_key)
                    code_range = getattr(spec, "code_token_range", None)
                except Exception:
                    code_range = None
                if code_range is not None:
                    base, length = code_range
                    token_mask = (packed_text_ids >= base) & (packed_text_ids < base + length)
                    if token_mask.any():
                        code_indices = torch.nonzero(token_mask, as_tuple=False).squeeze(1)
                        code_positions = torch.arange(
                            code_indices.numel(), device=device, dtype=torch.long
                        )
                        text_embeds[code_indices] = (
                            text_embeds[code_indices] + pos_embed_mod(code_positions)
                        )

        outputs = self._run_decoder_with_cache(
            inputs_embeds=text_embeds.unsqueeze(0),
            position_ids=packed_text_position_ids.unsqueeze(0),
            past_key_values=past_key_values,
            attention_mask=self._build_prefill_attention_mask(
                position_ids=packed_text_position_ids.unsqueeze(0),
            ),
        )
        return outputs.past_key_values

    @staticmethod
    def _fit_image_to_max_side(image_tensor, max_side, stride):
        """Mirror data/dataset_base.py:_fit_vit_image_tensor / _fit_vae_image_tensor.

        Bilinear-downsample H,W to <= max_side, aligning to stride. This is the
        clamp step that the training data pipeline applies AFTER the PIL
        ImageTransform. Inference's prepare_vit_images / prepare_vae_images
        previously skipped this step, producing pixel tensors that are not
        byte-identical to training's even when both target the same final size.
        Codex's logit parity test (job 2014122/2014383) confirmed this caused
        ~33% argmax disagreement at codebook positions under TF prefix.
        """
        import math
        _, img_h, img_w = image_tensor.shape

        def _align_down(size, s):
            return max(s, (size // s) * s)

        if img_h > max_side or img_w > max_side:
            scale = min(max_side / img_h, max_side / img_w)
            h2 = _align_down(int(math.floor(img_h * scale)), stride)
            w2 = _align_down(int(math.floor(img_w * scale)), stride)
            h2 = min(h2, max_side)
            w2 = min(w2, max_side)
        else:
            h2 = _align_down(img_h, stride)
            w2 = _align_down(img_w, stride)

        if h2 == img_h and w2 == img_w:
            return image_tensor

        return torch.nn.functional.interpolate(
            image_tensor.unsqueeze(0),
            size=(h2, w2),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def prepare_vae_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids, timestep=0, modality_type=None):
        del modality_type
        self._require_single_sample(curr_kvlens, "curr_kvlens")
        self._require_single_sample(curr_rope, "curr_rope")
        self._require_single_sample(images, "images")

        image_tensor = transforms(images[0]).contiguous()
        # Apply training-pipeline parity clamp: after PIL transform, training
        # pipeline (data/dataset_base.py:_fit_vae_image_tensor) does an extra
        # bilinear interpolate to <= latent_downsample * train_max_latent_size.
        # Without this, inference VAE input is up to 1024x1024 while training
        # was clamped to 256x256 — same model sees very different conditioning.
        # Env-gated default-on; set HUNYUAN_INFER_PIPE_PARITY=0 to restore
        # legacy behavior.
        if os.environ.get("HUNYUAN_INFER_PIPE_PARITY", "1") == "1":
            try:
                _train_max_latent_size = int(os.environ.get(
                    "HUNYUAN_INFER_MAX_LATENT_SIZE", str(self.max_latent_size)
                ))
            except ValueError:
                _train_max_latent_size = self.max_latent_size
            _max_vae_side = self.latent_downsample * _train_max_latent_size
            image_tensor = self._fit_image_to_max_side(
                image_tensor, _max_vae_side, self.latent_downsample
            ).contiguous()
        h = image_tensor.shape[1] // self.latent_downsample
        w = image_tensor.shape[2] // self.latent_downsample
        num_img_tokens = h * w
        seq_len = num_img_tokens + 2
        curr_kvlen = int(curr_kvlens[0])
        curr_position_id = int(curr_rope[0])

        input_ids = torch.full((seq_len,), self.image_token_id, dtype=torch.long)
        _boi, _eoi = self._image_delims(new_token_ids)
        input_ids[0] = _boi
        input_ids[-1] = _eoi
        image_mask = torch.zeros((1, seq_len), dtype=torch.bool)
        image_mask[:, 1:-1] = True

        generation_input = {
            "packed_text_ids": input_ids,
            "packed_query_position_ids": (
                torch.full((seq_len,), curr_position_id, dtype=torch.long)
                if _image_block_rope_matches_training()
                else torch.arange(
                    curr_position_id, curr_position_id + seq_len, dtype=torch.long
                )
            ),
            "packed_vae_images": image_tensor.unsqueeze(0),
            "packed_vae_image_mask": image_mask,
            "packed_timesteps": torch.tensor([float(timestep)], dtype=torch.float32),
            "key_values_lens": torch.tensor([curr_kvlen], dtype=torch.int),
            "packed_key_value_indexes": torch.arange(curr_kvlen, dtype=torch.long),
        }
        # Also stash the 2D-RoPE segment this image occupies in the cache.
        # `_diffusion_step_forward` needs to replay it when computing cos/sin
        # for cross-image attention; otherwise target queries get 1D rotation
        # at input image positions while the cached K's were rotated with 2D.
        # (See cached_rope() — it rebuilds cos/sin each call from the passed
        #  rope_image_info, so leaving prior cond images out silently breaks
        #  spatial alignment between cond and target.)
        rope_segment = (slice(curr_position_id + 1, curr_position_id + 1 + h * w), (h, w))
        generation_input["__cond_rope_segment__"] = rope_segment
        _rope_adv = 1 if _image_block_rope_matches_training() else seq_len
        return generation_input, [curr_kvlen + seq_len], [curr_position_id + _rope_adv]

    # ── VAE-target image generation (flow-matching denoise) ───────────────────
    #
    # Inferencer.gen_image calls three methods on the wrapper that the bagel
    # path provides natively (bagel.py:1079-1358) but hunyuan did not:
    #   1) prepare_vae_latent     — build the new-query layout (start_of_image,
    #      image_token_id*N, end_of_image) + init noise for a target image.
    #   2) prepare_vae_latent_cfg — index/position bookkeeping for a CFG-uncond
    #      branch (same target shape, attending against a different prefix
    #      KV cache).
    #   3) generate_image         — flow-matching denoise loop with parallel
    #      text/img CFG, using hunyuan-native instantiate_vae_image_tokens +
    #      ragged_final_layer rather than bagel's vae2llm/llm2vae heads.
    #
    # Per-step cache reuse: HunyuanStaticCache.update() writes K/V via
    # index_copy_ at the positions given by position_ids (modeling_hunyuan_*:1943),
    # so calling the decoder with the *same* position_ids each step overwrites
    # the patch slots [curr_kvlen : curr_kvlen + seq_len) without disturbing the
    # prefix [0 : curr_kvlen). No deepcopy needed.

    def native_generation_image_size(self, image_size):
        """Snap an inference target to the exact native training bucket.

        The dataset resizes every native generation target to ResolutionGroup's
        selected (w, h), not merely to a stride-aligned approximation.  Both
        token count and <img_ratio_*> must therefore use that bucket here too.
        """
        return self.native_generation_layout(image_size)[0]

    def native_generation_layout(self, image_size):
        """Return ``((H, W), (base_size, ratio_idx))`` exactly as training.

        Native-chat first snaps the transformed target to ResolutionGroup, then
        the shared packer applies ``_fit_vae_image_tensor``. Some extreme native
        buckets are slightly wider than max_latent_size (for example 1056), so
        that second stage clamps them to 1024 while retaining the metadata chosen
        before the clamp. Reproduce that seemingly redundant two-stage contract;
        current checkpoints were trained on it.
        """
        import math

        H, W = image_size
        base, ratio, bucket_w, bucket_h = _native_size_ratio(
            W, H, self.latent_downsample, return_resolution=True
        )
        # The released base config advertises a 384px VAE sample size, while
        # MODUS training explicitly uses max_latent_size=64 (1024px).  The
        # condition-image path already honors HUNYUAN_INFER_MAX_LATENT_SIZE;
        # target generation must use the same audited override.  Otherwise a
        # square target declares <img_size_512> but is silently clamped to
        # 384px (24x24 latents), a direct train/infer format mismatch.
        infer_max_latent_size = int(os.environ.get(
            "HUNYUAN_INFER_MAX_LATENT_SIZE", str(self.max_latent_size)
        ))
        if infer_max_latent_size <= 0:
            raise ValueError(
                "HUNYUAN_INFER_MAX_LATENT_SIZE must be positive, got "
                f"{infer_max_latent_size}"
            )
        max_side = infer_max_latent_size * self.latent_downsample
        if bucket_h > max_side or bucket_w > max_side:
            scale = min(max_side / bucket_h, max_side / bucket_w)

            def _aligned(value):
                return max(
                    self.latent_downsample,
                    (int(math.floor(value)) // self.latent_downsample)
                    * self.latent_downsample,
                )

            bucket_h = min(_aligned(bucket_h * scale), max_side)
            bucket_w = min(_aligned(bucket_w * scale), max_side)
        return (int(bucket_h), int(bucket_w)), (int(base), int(ratio))

    def prepare_vae_latent(
        self, curr_kvlens, curr_rope, image_sizes, new_token_ids,
        modality_type=None, tokenizer=None, native_meta=None,
    ):
        # modality_type accepted for API parity with bagel; for hunyuan all image
        # modalities share start_of_image / end_of_image (see instruction_stage2.yaml),
        # so target modality is signaled upstream via the conditioning instruction
        # text — not by changing the image start token here.
        del modality_type
        self._require_single_sample(curr_kvlens, "curr_kvlens")
        self._require_single_sample(curr_rope, "curr_rope")
        self._require_single_sample(image_sizes, "image_sizes")

        H, W = image_sizes[0]
        h = H // self.latent_downsample
        w = W // self.latent_downsample
        num_image_tokens = h * w
        curr_kvlen = int(curr_kvlens[0])
        curr_position_id = int(curr_rope[0])

        # Training's native_gen_image serializer is:
        #   <boi> <img_size_N> <img_ratio_K> <timestep> PATCHES <eoi>
        # (data/dataset_base.py).  The legacy inference path omitted all three
        # metadata tokens, shifting every patch and leaving the model without
        # the global diffusion-timestep signal it saw during training.
        meta_ids = []
        if self._wire_format_v2():
            if tokenizer is None:
                raise RuntimeError(
                    "HUNYUAN_WIRE_FORMAT_V2 target generation requires the tokenizer "
                    "to emit <img_size_*>, <img_ratio_*>, and <timestep> exactly as training."
                )

            def _tok(name):
                tid = tokenizer.convert_tokens_to_ids(name)
                if tid is None or tid < 0:
                    raise ValueError(f"Native Hunyuan token missing from tokenizer: {name}")
                return int(tid)

            if native_meta is None:
                base_size, ratio_idx = _native_size_ratio(
                    W, H, self.latent_downsample
                )
            else:
                base_size, ratio_idx = map(int, native_meta)
            meta_ids = [
                _tok(f"<img_size_{base_size}>"),
                _tok(f"<img_ratio_{ratio_idx}>"),
                self._timestep_token_id(),
            ]

        patch_start = 1 + len(meta_ids)
        seq_len = patch_start + num_image_tokens + 1
        input_ids = torch.full((seq_len,), self.image_token_id, dtype=torch.long)
        _boi, _eoi = self._image_delims(new_token_ids)
        input_ids[0] = _boi
        if meta_ids:
            input_ids[1:patch_start] = torch.tensor(meta_ids, dtype=torch.long)
        input_ids[-1] = _eoi

        # ragged_final_layer's masked_select path expects True at patch positions
        # only (delimiters and native metadata are False).
        image_mask = torch.zeros((1, seq_len), dtype=torch.bool)
        image_mask[:, patch_start:patch_start + num_image_tokens] = True

        packed_init_noises = torch.randn(
            num_image_tokens, self.latent_channel * self.latent_patch_size ** 2
        )

        generation_input = {
            "packed_text_ids": input_ids,
            "packed_init_noises": packed_init_noises,
            "packed_query_position_ids": torch.arange(
                curr_position_id, curr_position_id + seq_len, dtype=torch.long
            ),
            "packed_image_mask": image_mask,
            "packed_image_h": int(h),
            "packed_image_w": int(w),
            "key_values_lens": torch.tensor([curr_kvlen], dtype=torch.int),
            "packed_key_value_indexes": torch.arange(curr_kvlen, dtype=torch.long),
        }
        return generation_input

    def prepare_vae_latent_cfg(self, curr_kvlens, curr_rope, image_sizes):
        self._require_single_sample(curr_kvlens, "curr_kvlens")
        self._require_single_sample(curr_rope, "curr_rope")
        self._require_single_sample(image_sizes, "image_sizes")

        H, W = image_sizes[0]
        h = H // self.latent_downsample
        w = W // self.latent_downsample
        # Must have exactly the same target-query layout as the conditional
        # branch because all three branches reuse packed_text_ids/image_mask.
        seq_len = h * w + (5 if self._wire_format_v2() else 2)
        curr_kvlen = int(curr_kvlens[0])
        curr_position_id = int(curr_rope[0])

        return {
            "cfg_packed_position_ids": torch.arange(
                curr_position_id, curr_position_id + seq_len, dtype=torch.long
            ),
            "cfg_packed_query_indexes": torch.arange(
                curr_kvlen, curr_kvlen + seq_len, dtype=torch.long
            ),
            "cfg_key_values_lens": torch.tensor([curr_kvlen], dtype=torch.int),
            "cfg_packed_key_value_indexes": torch.arange(curr_kvlen, dtype=torch.long),
        }

    @torch.no_grad()
    def _diffusion_step_forward(
        self,
        *,
        x_t_4d: torch.Tensor,
        packed_text_ids: torch.LongTensor,
        position_ids_1d: torch.LongTensor,
        image_mask: torch.Tensor,
        h: int,
        w: int,
        timesteps: torch.Tensor,
        past_key_values,
        cond_rope_segments: Optional[List[Tuple[slice, Tuple[int, int]]]] = None,
    ) -> torch.Tensor:
        """Run a single diffusion-step decoder forward and return velocity (N, C).

        The decoder writes K/V at the absolute positions given by position_ids_1d
        (the static cache's index_copy_ uses position_ids as cache_position), so
        repeated calls with the same position_ids overwrite the patch slots in
        place — leaving the prefix slots [0 : curr_kvlen) untouched.

        `cond_rope_segments` lists the 2D-RoPE (slice, (h, w)) entries for every
        cond image already in the cache. Replaying them in rope_image_info is
        what keeps target-Q ↔ cond-image-K attention aligned (cached_rope
        rebuilds the cos/sin table on every call from the passed image_infos).
        """
        device = self.device
        seq_len = position_ids_1d.numel()
        position_ids = position_ids_1d.to(device).unsqueeze(0)  # (1, seq_len)

        # 1. Embed start_of_image / image_token_id placeholders / end_of_image
        hidden_states = self.hunyuan_model.model.wte(packed_text_ids.to(device).unsqueeze(0))

        # 2. Scatter patch embeddings into the placeholder positions.
        hidden_states = self.hunyuan_model.instantiate_vae_image_tokens(
            hidden_states,
            timesteps.to(device),
            x_t_4d.to(device=device, dtype=hidden_states.dtype),
            image_mask.to(device),
        )

        # Training replaces the target block's <timestep> vocabulary embedding
        # with timestep_emb(t) on every forward.  Do the same at every denoise
        # step; otherwise inference sees a constant wte row in this slot.
        if self._wire_format_v2():
            ts_pos = (
                packed_text_ids.to(device) == self._timestep_token_id()
            ).nonzero(as_tuple=False).squeeze(-1)
            if ts_pos.numel() != timesteps.numel():
                raise RuntimeError(
                    f"target <timestep> slot count ({ts_pos.numel()}) != timestep "
                    f"count ({timesteps.numel()})"
                )
            hidden_states[0, ts_pos] = self.hunyuan_model.timestep_emb(
                timesteps.to(device).reshape(-1)
            ).to(hidden_states.dtype)

        # 3. 2D RoPE for patch positions; include ALL cond image segments so the
        #    cos/sin table matches what forward_cache_update_vae used when it
        #    rotated and stored cond K's. Without this, cross-image attention is
        #    silently broken: target Q at the same conceptual position rotates
        #    differently than cond K, breaking the spatial RoPE correspondence
        #    and producing "depth-like but pixel-misaligned" output.
        patch_pos = image_mask[0].nonzero(as_tuple=False).squeeze(-1)
        if patch_pos.numel() != h * w:
            raise RuntimeError(
                f"target image mask has {patch_pos.numel()} patches, expected {h * w}"
            )
        vae_start = int(position_ids_1d[int(patch_pos[0])].item())
        target_segment = (slice(vae_start, vae_start + h * w), (h, w))
        all_segments = list(cond_rope_segments or []) + [target_segment]
        rope_image_info = [all_segments]

        # 4. Causal-to-prefix + full-inside-image attention (matches training,
        #    cf. forward_cache_update_vae:796-810).
        attention_mask = self._build_prefill_attention_mask(
            position_ids=position_ids,
            full_attention_slices=[
                slice(int(position_ids_1d[0].item()), int(position_ids_1d[-1].item()) + 1)
            ],
        )

        outputs = self._run_decoder_with_cache(
            inputs_embeds=hidden_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_image_info=rope_image_info,
        )

        # 5. Velocity head: ragged_final_layer with first_step=None uses
        #    image_mask.masked_select to extract patch positions.
        #    Under HF device_map=auto, last_hidden_state may land on a different
        #    GPU than self.device (typically the last TP rank). Co-locate all
        #    inputs to final_layer's device to avoid cross-device masked_select.
        final_device = next(self.hunyuan_model.final_layer.parameters()).device
        last_hidden = outputs.last_hidden_state.to(final_device)
        diff_pred = self.hunyuan_model.ragged_final_layer(
            last_hidden,
            image_mask.to(final_device),
            timesteps.to(final_device),
            h,
            w,
        )
        # diff_pred shape: (1, latent_channels, h*ps, w*ps); reshape to (N, C).
        C = self.latent_channels
        v_flat = diff_pred.squeeze(0).reshape(C, h * w).permute(1, 0).contiguous()
        return v_flat

    @torch.no_grad()
    def generate_image(
        self,
        # geometry / new-query layout (from prepare_vae_latent)
        packed_text_ids: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_query_position_ids: torch.LongTensor,
        packed_image_mask: torch.Tensor,
        packed_image_h: int,
        packed_image_w: int,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        past_key_values,
        # diffusion / CFG knobs
        num_timesteps: int = 50,
        timestep_shift: float = 3.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        # cfg_text branch (text-uncond)
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_past_key_values=None,
        cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img branch (image-uncond)
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_past_key_values=None,
        cfg_img_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # accepted for API parity with bagel.generate_image
        target_modality_name: Optional[str] = None,
        # 2D-RoPE segments for every cond image in each branch's prefix cache.
        # Must replay them in cached_rope so target Q rotates against the same
        # cos/sin table that was used when cond K was stored (see
        # _diffusion_step_forward for the long-form why).
        cond_rope_segments: Optional[List[Tuple[slice, Tuple[int, int]]]] = None,
        cfg_text_cond_rope_segments: Optional[List[Tuple[slice, Tuple[int, int]]]] = None,
        cfg_img_cond_rope_segments: Optional[List[Tuple[slice, Tuple[int, int]]]] = None,
        **_unused,
    ):
        del key_values_lens, packed_key_value_indexes
        del cfg_text_packed_query_indexes, cfg_text_key_values_lens, cfg_text_packed_key_value_indexes
        del cfg_img_packed_query_indexes, cfg_img_key_values_lens, cfg_img_packed_key_value_indexes
        del target_modality_name, _unused

        device = self.device
        h, w = int(packed_image_h), int(packed_image_w)
        C = self.latent_channels
        ps = self.latent_patch_size

        x_t = packed_init_noises.to(device)  # (N, C * ps²)

        # Flow-matching timestep schedule (matches bagel.generate_image:1209-1212).
        # Tested codex's N+1 fix (matches native) but it gave WORSE edge ρ —
        # reverted. The bagel-style N-1 integration steps is what the model
        # was trained against in its sister bagel impl.
        # Default to the TRAINING scale, not 1.0. These were two independent env
        # vars with no cross-check: training uses sigma*1000 (native convention,
        # validated in v55 -- only ckpt+scale=1000 produced clean native T2I;
        # every other combination was noise). An eval that forgot to set the
        # infer var silently fed t in [0,1] to a model trained on [0,1000],
        # which looks exactly like "the model drifted".
        timestep_embed_scale = _require_timestep_scale("inference sampling loop")
        timesteps = torch.linspace(1, 0, num_timesteps, device=device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts = timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]

        for i, t in enumerate(timesteps):
            t_scalar = float(t.item())
            if cfg_interval[0] < t_scalar <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0

            # Reshape (N, C*ps²) → (1, C, h*ps, w*ps) for instantiate_vae_image_tokens.
            x_t_4d = (
                x_t.reshape(h, w, C, ps, ps)
                .permute(2, 0, 3, 1, 4)
                .reshape(1, C, h * ps, w * ps)
            )
            timestep_tensor = torch.tensor(
                [t_scalar * timestep_embed_scale],
                dtype=torch.float32,
                device=device,
            )

            v_cond = self._diffusion_step_forward(
                x_t_4d=x_t_4d,
                packed_text_ids=packed_text_ids,
                position_ids_1d=packed_query_position_ids,
                image_mask=packed_image_mask,
                h=h,
                w=w,
                timesteps=timestep_tensor,
                past_key_values=past_key_values,
                cond_rope_segments=cond_rope_segments,
            )

            # CFG combination — three renorm modes, matching bagel._forward_flow
            # (bagel.py:1360-1389):
            #   - "text_channel": apply text-CFG first, channel-wise renorm
            #     to |v_cond|, THEN apply img-CFG with no further renorm.
            #   - "global"      : apply both CFGs, single scalar renorm at end.
            #   - "channel"     : apply both CFGs, channel-wise renorm at end.
            if cfg_text_scale_ > 1.0 and cfg_text_past_key_values is not None:
                v_text_uncond = self._diffusion_step_forward(
                    x_t_4d=x_t_4d,
                    packed_text_ids=packed_text_ids,
                    position_ids_1d=cfg_text_packed_position_ids,
                    image_mask=packed_image_mask,
                    h=h,
                    w=w,
                    timesteps=timestep_tensor,
                    past_key_values=cfg_text_past_key_values,
                    cond_rope_segments=cfg_text_cond_rope_segments,
                )
            else:
                v_text_uncond = None

            if cfg_img_scale_ > 1.0 and cfg_img_past_key_values is not None:
                v_img_uncond = self._diffusion_step_forward(
                    x_t_4d=x_t_4d,
                    packed_text_ids=packed_text_ids,
                    position_ids_1d=cfg_img_packed_position_ids,
                    image_mask=packed_image_mask,
                    h=h,
                    w=w,
                    timesteps=timestep_tensor,
                    past_key_values=cfg_img_past_key_values,
                    cond_rope_segments=cfg_img_cond_rope_segments,
                )
            else:
                v_img_uncond = None

            if v_text_uncond is None and v_img_uncond is None:
                v_t = v_cond
            elif cfg_renorm_type == "text_channel":
                v_t_text_ = v_cond
                if v_text_uncond is not None:
                    v_t_text_ = v_text_uncond + cfg_text_scale_ * (v_cond - v_text_uncond)
                    norm_cond = torch.norm(v_cond, dim=-1, keepdim=True)
                    norm_extrap = torch.norm(v_t_text_, dim=-1, keepdim=True)
                    scale = (norm_cond / (norm_extrap + 1e-8)).clamp(
                        min=cfg_renorm_min, max=1.0
                    )
                    v_t_text_ = v_t_text_ * scale
                if v_img_uncond is not None:
                    v_t = v_img_uncond + cfg_img_scale_ * (v_t_text_ - v_img_uncond)
                else:
                    v_t = v_t_text_
            else:
                v_t_ = v_cond
                if v_text_uncond is not None:
                    v_t_ = v_text_uncond + cfg_text_scale_ * (v_t_ - v_text_uncond)
                if v_img_uncond is not None:
                    v_t_ = v_img_uncond + cfg_img_scale_ * (v_t_ - v_img_uncond)
                if cfg_renorm_type == "channel":
                    norm_cond = torch.norm(v_cond, dim=-1, keepdim=True)
                    norm_extrap = torch.norm(v_t_, dim=-1, keepdim=True)
                else:  # "global"
                    norm_cond = torch.norm(v_cond)
                    norm_extrap = torch.norm(v_t_)
                scale = (norm_cond / (norm_extrap + 1e-8)).clamp(
                    min=cfg_renorm_min, max=1.0
                )
                v_t = v_t_ * scale

            x_t = x_t - v_t.to(x_t.device, dtype=x_t.dtype) * dts[i]

        # Inferencer.decode_image takes unpacked_latent[0] of shape (N, C*ps²).
        return [x_t]

    @torch.no_grad()
    def forward_cache_update_vae(
        self,
        vae_model,
        past_key_values,
        packed_text_ids: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        packed_vae_images: torch.Tensor,
        packed_vae_image_mask: torch.Tensor,
        packed_timesteps: torch.Tensor,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.Tensor,
        modality_type=None,
        do_modality_norm=False,
    ):
        del key_values_lens, packed_key_value_indexes, modality_type, do_modality_norm
        device = self.device
        input_ids = packed_text_ids.to(device).unsqueeze(0)
        hidden_states = self.hunyuan_model.model.wte(input_ids)
        # Training passes normalized pixels to the VAE in fp32 and lets
        # autocast choose internal compute dtypes.  Casting the pixels to bf16
        # here quantized them before the first convolution and caused a real
        # train/infer logit drift.
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            image_latents = vae_model.encode(
                packed_vae_images.to(device=device, dtype=torch.float32)
            )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            hidden_states = self.hunyuan_model.instantiate_vae_image_tokens(
                hidden_states,
                packed_timesteps.to(device),
                image_latents,
                packed_vae_image_mask.to(device),
            )
        # Build 2D spatial RoPE for VAE patch tokens, matching training rope_batch0 assembly.
        # In training, every VAE image block is appended to rope_batch0 as:
        #   (slice(start_pos, start_pos + h*w), (h, w))
        # Inference must do the same for the condition VAE prefill chunk; otherwise the model
        # receives 1D sequential RoPE for patches that were trained with 2D spatial RoPE.
        _vae_h = packed_vae_images.shape[2] // self.latent_downsample
        _vae_w = packed_vae_images.shape[3] // self.latent_downsample
        _vae_n = _vae_h * _vae_w
        # packed_query_position_ids[0] = start_of_image token; [1] begins VAE patch tokens
        _vae_start = int(packed_query_position_ids[1].item())
        _vae_rope_info = [[(slice(_vae_start, _vae_start + _vae_n), (_vae_h, _vae_w))]]

        outputs = self._run_decoder_with_cache(
            inputs_embeds=hidden_states,
            position_ids=packed_query_position_ids.to(device).unsqueeze(0),
            past_key_values=past_key_values,
            attention_mask=self._build_prefill_attention_mask(
                position_ids=packed_query_position_ids.to(device).unsqueeze(0),
                # Include start_of_image (p0) and end_of_image (p_last) inside the
                # full-attention block to match (a) MODUS training (data/dataset_base.py:719
                # marks the entire 2+N block as attn_mode="full") and (b) the original
                # Hunyuan inference convention (tokenization_hunyuan_image_3.py:1104-1106
                # uses slice(start, end+1) — closed on both ends). Previously this slice
                # excluded both special tokens, leaving start_of_image causal-only at
                # inference but bidirectional at training — a train/infer mismatch on
                # an attention-sink token that contaminated K/V for all downstream AR steps.
                # Set HUNYUAN_INFER_PREFILL_FULL_INCLUSIVE=0 to revert to the old buggy
                # behaviour for A/B comparison.
                full_attention_slices=[(
                    slice(int(packed_query_position_ids[0].item()), int(packed_query_position_ids[-1].item()) + 1)
                    if os.environ.get("HUNYUAN_INFER_PREFILL_FULL_INCLUSIVE", "1") == "1"
                    else slice(int(packed_query_position_ids[0].item()) + 1, int(packed_query_position_ids[-1].item()))
                )],
            ),
            rope_image_info=_vae_rope_info,
        )
        return outputs.past_key_values

    def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids, modality_type=None):
        del modality_type
        self._require_single_sample(curr_kvlens, "curr_kvlens")
        self._require_single_sample(curr_rope, "curr_rope")
        self._require_single_sample(images, "images")

        # Training-pipeline parity: training's vit_image_transform_args has
        # max_image_size=512 (yaml config), but inference any2any_tasks.py uses
        # max=256. The PIL Lanczos resize to 256 directly does NOT match training's
        # PIL-Lanczos-to-512-then-bilinear-to-256 cascade. Override the transform
        # under parity mode to mirror training exactly.
        if os.environ.get("HUNYUAN_INFER_PIPE_PARITY", "1") == "1":
            from data.transforms import ImageTransform
            try:
                _train_max = int(os.environ.get("HUNYUAN_INFER_VIT_TRAIN_MAX", "512"))
            except ValueError:
                _train_max = 512
            try:
                _train_min = int(os.environ.get("HUNYUAN_INFER_VIT_TRAIN_MIN", "224"))
            except ValueError:
                _train_min = 224
            _train_pipe = ImageTransform(_train_max, _train_min, self.vit_patch_size)
            image_tensor = _train_pipe(images[0]).contiguous()
            _max_vit_side = self.vit_patch_size * self.vit_max_num_patch_per_side
            image_tensor = self._fit_image_to_max_side(
                image_tensor, _max_vit_side, self.vit_patch_size
            ).contiguous()
        else:
            image_tensor = transforms(images[0]).contiguous()
        patch_size = self.vit_patch_size
        token_h = image_tensor.shape[1] // patch_size
        token_w = image_tensor.shape[2] // patch_size
        vit_tokens = patchify(image_tensor, patch_size)
        num_img_tokens = vit_tokens.shape[0]
        seq_len = num_img_tokens + 2
        curr_kvlen = int(curr_kvlens[0])
        curr_position_id = int(curr_rope[0])

        input_ids = torch.full((seq_len,), self.image_token_id, dtype=torch.long)
        _boi, _eoi = self._image_delims(new_token_ids)
        input_ids[0] = _boi
        input_ids[-1] = _eoi
        image_mask = torch.zeros((1, seq_len), dtype=torch.bool)
        image_mask[:, 1:-1] = True
        spatial_shapes = torch.tensor([[[token_h, token_w]]], dtype=torch.long)
        attention_mask = torch.ones((1, 1, num_img_tokens), dtype=torch.long)

        generation_input = {
            "packed_text_ids": input_ids,
            "packed_query_position_ids": (
                torch.full((seq_len,), curr_position_id, dtype=torch.long)
                if _image_block_rope_matches_training()
                else torch.arange(
                    curr_position_id, curr_position_id + seq_len, dtype=torch.long
                )
            ),
            "packed_vit_images": [vit_tokens.unsqueeze(0)],
            "packed_vit_image_mask": image_mask,
            "packed_vit_image_kwargs": {
                "spatial_shapes": spatial_shapes,
                "attention_mask": attention_mask,
            },
            "key_values_lens": torch.tensor([curr_kvlen], dtype=torch.int),
            "packed_key_value_indexes": torch.arange(curr_kvlen, dtype=torch.long),
        }
        # Same as prepare_vae_images: stash the 2D-RoPE segment this cond ViT
        # image occupies so `_diffusion_step_forward` can replay it. Without
        # this, target Q's cos/sin at ViT positions defaults to 1D while the
        # cached ViT K's were rotated with 2D → cross-image attention breaks
        # for the ViT path (which is the dominant semantic conditioning path).
        # Gated by the same env that controls whether forward_cache_update_vit
        # actually applies 2D RoPE (default on).
        if os.environ.get("HUNYUAN_INFER_VIT_2D_ROPE", "1") == "1":
            rope_segment = (
                slice(curr_position_id + 1, curr_position_id + 1 + token_h * token_w),
                (token_h, token_w),
            )
            generation_input["__cond_rope_segment__"] = rope_segment
        _rope_adv = 1 if _image_block_rope_matches_training() else seq_len
        return generation_input, [curr_kvlen + seq_len], [curr_position_id + _rope_adv]

    # ── Joint condition image block (wire-format v2) ─────────────────────────
    #
    # Training emits ONE block per condition image:
    #   <boi> [<img_size_N><img_ratio_K><timestep>] VAE... <joint_img_sep> ViT... <eoi>
    # (data/dataset_base.py, `native_cond_joint_image`). Inference historically
    # emitted TWO independent blocks instead -- prepare_vae_images then
    # prepare_vit_images, each with its own <boi>/<eoi> -- so a chained round-2
    # condition (the image round 1 just produced) was fed back in a shape the
    # model was never trained on. These two methods close that gap.

    def _image_delims(self, new_token_ids):
        """(start, end) token ids for an image block, matching what TRAINING emits.

        Training has two serializers and they use different delimiters:
          - strict native (HUNYUAN_STRICT_NATIVE_IMAGE_BLOCK=1, what every live
            16-mod run uses) emits the NATIVE <boi>/<eoi> tokens
            (data/dataset_base.py:906/911 and :976/978);
          - the legacy path emits start_token_mapping[modality], i.e.
            <|vision_start|>/<|vision_end|> -- MODUS-added tokens that do not
            exist in the pretrained Hunyuan vocabulary (data/data_utils.py:171).

        Inference used <|vision_start|>/<|vision_end|> unconditionally, so under
        strict native EVERY image block -- condition and target -- opened and
        closed on a token the model was never trained to see there. Adding
        <img_size>/<img_ratio>/<timestep> closes a 3-token gap; this closes the
        2-token gap at the block's own boundaries. No retraining needed.
        """
        if os.environ.get("HUNYUAN_STRICT_NATIVE_IMAGE_BLOCK", "0") != "1":
            return int(new_token_ids["start_of_image"]), int(new_token_ids["end_of_image"])
        tok = getattr(self, "_native_delims_cache", None)
        if tok is None:
            ids = getattr(self, "_native_boi_eoi", None)
            if ids is None:
                raise RuntimeError(
                    "Strict native image blocks are on, so inference must emit "
                    "<boi>/<eoi> to match training -- but their ids were never "
                    "registered. Call model.set_native_image_delims(...) at setup."
                )
            tok = ids
            object.__setattr__(self, "_native_delims_cache", tok)
        return tok

    def set_native_image_delims(self, boi_id: int, eoi_id: int):
        object.__setattr__(self, "_native_boi_eoi", (int(boi_id), int(eoi_id)))

    def set_timestep_token_id(self, token_id: int):
        """Register the <timestep> vocabulary id.

        The wrapper holds no tokenizer, so whoever builds it (train.py, the
        inferencer) has to hand this over. Called once at setup.
        """
        object.__setattr__(self, "_ts_token_id", int(token_id))

    def _timestep_token_id(self):
        tid = getattr(self, "_ts_token_id", None)
        if tid is None:
            raise RuntimeError(
                "HUNYUAN_WIRE_FORMAT_V2 is on but the <timestep> token id was "
                "never registered. Call model.set_timestep_token_id("
                "tokenizer.convert_tokens_to_ids('<timestep>')) at setup. "
                "Refusing to skip the timestep embedding silently -- an unfilled "
                "slot holds a constant wte row and every t looks identical."
            )
        return tid

    def _wire_format_v2(self):
        return os.environ.get("HUNYUAN_WIRE_FORMAT_V2", "0") == "1"

    def prepare_cond_joint_image(
        self, curr_kvlens, curr_rope, images, vae_transforms, vit_transforms,
        new_token_ids, tokenizer, modality_type=None,
    ):
        del modality_type

        def _tok(name):
            tid = tokenizer.convert_tokens_to_ids(name)
            if tid is None or tid < 0:
                raise ValueError(f"Native Hunyuan token missing from tokenizer: {name}")
            return int(tid)
        self._require_single_sample(curr_kvlens, "curr_kvlens")
        self._require_single_sample(curr_rope, "curr_rope")
        self._require_single_sample(images, "images")

        vae_input, _, _ = self.prepare_vae_images(
            curr_kvlens, curr_rope, images, vae_transforms, new_token_ids,
        )
        vit_input, _, _ = self.prepare_vit_images(
            curr_kvlens, curr_rope, images, vit_transforms, new_token_ids,
        )
        vae_tensor = vae_input["packed_vae_images"]
        vae_h = vae_tensor.shape[2] // self.latent_downsample
        vae_w = vae_tensor.shape[3] // self.latent_downsample
        n_vae = vae_h * vae_w
        vit_shapes = vit_input["packed_vit_image_kwargs"]["spatial_shapes"]
        vit_h, vit_w = int(vit_shapes[0, 0, 0]), int(vit_shapes[0, 0, 1])
        n_vit = vit_h * vit_w

        # <boi> [meta] VAE*n_vae <joint_img_sep> ViT*n_vit <eoi>
        meta_ids = []
        if self._wire_format_v2():
            base_size, ratio_idx = _native_size_ratio(
                vae_tensor.shape[3], vae_tensor.shape[2], self.latent_downsample,
            )
            meta_ids = [
                _tok(f"<img_size_{base_size}>"),
                _tok(f"<img_ratio_{ratio_idx}>"),
                _tok("<timestep>"),
            ]
        n_meta = len(meta_ids)
        seq_len = 1 + n_meta + n_vae + 1 + n_vit + 1

        input_ids = torch.full((seq_len,), self.image_token_id, dtype=torch.long)
        _boi, _eoi = self._image_delims(new_token_ids)
        input_ids[0] = _boi
        for i, tid in enumerate(meta_ids):
            input_ids[1 + i] = int(tid)
        vae_lo = 1 + n_meta
        sep_pos = vae_lo + n_vae
        vit_lo = sep_pos + 1
        input_ids[sep_pos] = _tok("<joint_img_sep>")
        input_ids[-1] = _eoi

        vae_mask = torch.zeros((1, seq_len), dtype=torch.bool)
        vae_mask[:, vae_lo:vae_lo + n_vae] = True
        vit_mask = torch.zeros((1, seq_len), dtype=torch.bool)
        vit_mask[:, vit_lo:vit_lo + n_vit] = True

        curr_kvlen = int(curr_kvlens[0])
        curr_position_id = int(curr_rope[0])
        position_ids = (
            torch.full((seq_len,), curr_position_id, dtype=torch.long)
            if _image_block_rope_matches_training()
            else torch.arange(
                curr_position_id, curr_position_id + seq_len, dtype=torch.long
            )
        )

        generation_input = {
            "packed_text_ids": input_ids,
            "packed_query_position_ids": position_ids,
            "packed_vae_images": vae_tensor,
            "packed_vae_image_mask": vae_mask,
            "packed_timesteps": torch.tensor([0.0], dtype=torch.float32),
            "packed_vit_images": vit_input["packed_vit_images"],
            "packed_vit_image_mask": vit_mask,
            "packed_vit_image_kwargs": vit_input["packed_vit_image_kwargs"],
            "key_values_lens": torch.tensor([curr_kvlen], dtype=torch.int),
            "packed_key_value_indexes": torch.arange(curr_kvlen, dtype=torch.long),
        }
        # Both halves need their 2D-RoPE segment replayed at target-gen time.
        vae_start = curr_position_id + vae_lo
        vit_start = curr_position_id + vit_lo
        generation_input["__cond_rope_segments__"] = [
            (slice(vae_start, vae_start + n_vae), (vae_h, vae_w)),
            (slice(vit_start, vit_start + n_vit), (vit_h, vit_w)),
        ]
        _rope_adv = 1 if _image_block_rope_matches_training() else seq_len
        return generation_input, [curr_kvlen + seq_len], [curr_position_id + _rope_adv]

    @torch.no_grad()
    def forward_cache_update_cond_joint(
        self, vae_model, past_key_values,
        packed_text_ids, packed_query_position_ids,
        packed_vae_images, packed_vae_image_mask, packed_timesteps,
        packed_vit_images, packed_vit_image_mask, packed_vit_image_kwargs=None,
        key_values_lens=None, packed_key_value_indexes=None,
        modality_type=None, do_modality_norm=False,
    ):
        del key_values_lens, packed_key_value_indexes, modality_type, do_modality_norm
        device = self.device
        input_ids = packed_text_ids.to(device).unsqueeze(0)
        hidden_states = self.hunyuan_model.model.wte(input_ids)

        # Match training: keep normalized input pixels fp32 at the VAE boundary.
        vae_images = packed_vae_images.to(device=device, dtype=torch.float32)
        # Diagnostic for a train/infer mismatch: training zero-pads every VAE
        # image to the micro-batch's largest H/W *before* VAE encode, then crops
        # the latent back to its recorded shape.  The VAE contains GroupNorm,
        # so the surrounding zeros change even the retained latent values.
        # Existing checkpoints therefore saw padded condition latents.  Keep
        # the default inference contract unchanged; this gate measures whether
        # reproducing a canonical training canvas closes the TF-CE gap.
        _diag_pad_to = int(os.environ.get("HUNYUAN_DIAG_VAE_PAD_TO", "0"))
        _orig_latent_h = vae_images.shape[-2] // self.latent_downsample
        _orig_latent_w = vae_images.shape[-1] // self.latent_downsample
        if _diag_pad_to > 0:
            pad_h = max(0, _diag_pad_to - int(vae_images.shape[-2]))
            pad_w = max(0, _diag_pad_to - int(vae_images.shape[-1]))
            if pad_h or pad_w:
                vae_images = F.pad(vae_images, (0, pad_w, 0, pad_h))
                print(
                    f"[diag-vae-pad] input={tuple(packed_vae_images.shape[-2:])} "
                    f"canvas={tuple(vae_images.shape[-2:])}", flush=True,
                )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            image_latents = vae_model.encode(vae_images)
        if _diag_pad_to > 0:
            image_latents = image_latents[..., :_orig_latent_h, :_orig_latent_w]
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            hidden_states = self.hunyuan_model.instantiate_vae_image_tokens(
                hidden_states, packed_timesteps.to(device), image_latents,
                packed_vae_image_mask.to(device),
            )

        try:
            vit_dtype = next(self.hunyuan_model.vision_model.parameters()).dtype
        except StopIteration:
            vit_dtype = hidden_states.dtype
        vit_images = [item.to(device=device, dtype=vit_dtype) for item in packed_vit_images]
        if packed_vit_image_kwargs is not None:
            packed_vit_image_kwargs = {
                k: v.to(device=device) for k, v in packed_vit_image_kwargs.items()
            }
        hidden_states = self.hunyuan_model.instantiate_vit_image_tokens(
            hidden_states, vit_images, packed_vit_image_mask.to(device),
            packed_vit_image_kwargs,
        )

        # Fill this block's <timestep> slot, same as training does (see 5c-bis).
        # A condition is clean, so t = 0 -- but it must be the timestep EMBEDDING
        # of 0, not the constant wte row, or the slot says nothing at all.
        if self._wire_format_v2():
            ts_pos = (
                input_ids[0] == self._timestep_token_id()
            ).nonzero(as_tuple=False).squeeze(-1)
            if ts_pos.numel() > 0:
                ts_emb = self.hunyuan_model.timestep_emb(
                    packed_timesteps.to(device).reshape(-1)
                )
                hidden_states[0, ts_pos] = ts_emb.to(hidden_states.dtype)

        if os.environ.get("HUNYUAN_DIAG_COND_SHAPE", "0") == "1":
            # Training packs this condition at max_latent_size=64 (4096 VAE
            # tokens) + a 32x32 ViT grid (1024). Print what inference actually
            # builds so the two can be compared as numbers, not inferred from
            # kv_lens.
            print(f"[cond-shape] vae_tokens={int(packed_vae_image_mask.sum())} "
                  f"vit_tokens={int(packed_vit_image_mask.sum())} "
                  f"vae_img={tuple(packed_vae_images.shape)} "
                  f"seq_len={int(packed_vae_image_mask.shape[1])}", flush=True)
        vae_pos = packed_vae_image_mask[0].nonzero().flatten()
        vit_pos = packed_vit_image_mask[0].nonzero().flatten()
        vae_h = packed_vae_images.shape[2] // self.latent_downsample
        vae_w = packed_vae_images.shape[3] // self.latent_downsample
        shapes = packed_vit_image_kwargs["spatial_shapes"]
        vit_h, vit_w = int(shapes[0, 0, 0]), int(shapes[0, 0, 1])
        vae_start = int(packed_query_position_ids[int(vae_pos[0])].item())
        vit_start = int(packed_query_position_ids[int(vit_pos[0])].item())
        rope_info = [[
            (slice(vae_start, vae_start + vae_h * vae_w), (vae_h, vae_w)),
            (slice(vit_start, vit_start + vit_h * vit_w), (vit_h, vit_w)),
        ]]

        outputs = self._run_decoder_with_cache(
            inputs_embeds=hidden_states,
            position_ids=packed_query_position_ids.to(device).unsqueeze(0),
            past_key_values=past_key_values,
            attention_mask=self._build_prefill_attention_mask(
                position_ids=packed_query_position_ids.to(device).unsqueeze(0),
                full_attention_slices=[(
                    slice(int(packed_query_position_ids[0].item()),
                          int(packed_query_position_ids[-1].item()) + 1)
                )],
            ),
            rope_image_info=rope_info,
        )
        return outputs.past_key_values

    @torch.no_grad()
    def forward_cache_update_vit(
        self,
        past_key_values,
        packed_text_ids: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        packed_vit_images,
        packed_vit_image_mask: torch.Tensor,
        packed_vit_image_kwargs=None,
        key_values_lens: torch.IntTensor = None,
        packed_key_value_indexes: torch.LongTensor = None,
    ):
        del key_values_lens, packed_key_value_indexes
        device = self.device
        input_ids = packed_text_ids.to(device).unsqueeze(0)
        hidden_states = self.hunyuan_model.model.wte(input_ids)
        try:
            vit_dtype = next(self.hunyuan_model.vision_model.parameters()).dtype
        except StopIteration:
            vit_dtype = hidden_states.dtype
        vit_images = [item.to(device=device, dtype=vit_dtype) for item in packed_vit_images]
        if packed_vit_image_kwargs is not None:
            packed_vit_image_kwargs = {
                k: v.to(device=device) for k, v in packed_vit_image_kwargs.items()
            }
        hidden_states = self.hunyuan_model.instantiate_vit_image_tokens(
            hidden_states,
            vit_images,
            packed_vit_image_mask.to(device),
            packed_vit_image_kwargs,
        )
        # Build 2D spatial RoPE for ViT input patch tokens, matching v22d's
        # training rope_batch0 assembly (model_wrapper.py:1328-1342 consumes
        # vit_image_token_shapes and appends 2D entries). Inference previously
        # did NOT do this — passed rope_image_info=None → 1D arange RoPE for
        # ViT input. v22c was trained 1D so 1D inference matched. v22d was
        # retrained with 2D ViT RoPE so 1D inference creates a new train/
        # infer mismatch. This mirrors forward_cache_update_vae:710-720.
        # Env-gated default-on for safety; set HUNYUAN_INFER_VIT_2D_ROPE=0
        # to revert to 1D (e.g. when evaluating v22c).
        _vit_rope_info = None
        if (
            os.environ.get("HUNYUAN_INFER_VIT_2D_ROPE", "1") == "1"
            and packed_vit_image_kwargs is not None
            and "spatial_shapes" in packed_vit_image_kwargs
        ):
            try:
                _shape = packed_vit_image_kwargs["spatial_shapes"]
                _vit_h, _vit_w = int(_shape[0, 0, 0].item()), int(_shape[0, 0, 1].item())
                _vit_n = _vit_h * _vit_w
                # packed_query_position_ids[0] = start_of_image; [1] = first ViT token;
                # [-1] = end_of_image. So the 2D ViT block is [pos[1], pos[1]+_vit_n).
                _vit_start = int(packed_query_position_ids[1].item())
                _vit_rope_info = [[(slice(_vit_start, _vit_start + _vit_n), (_vit_h, _vit_w))]]
            except Exception:
                _vit_rope_info = None
        outputs = self._run_decoder_with_cache(
            inputs_embeds=hidden_states,
            position_ids=packed_query_position_ids.to(device).unsqueeze(0),
            past_key_values=past_key_values,
            attention_mask=self._build_prefill_attention_mask(
                position_ids=packed_query_position_ids.to(device).unsqueeze(0),
                # See note at the matching VAE prefill site above. Same fix:
                # include start_of_image and end_of_image inside the full-attention
                # block to match MODUS training and original Hunyuan inference.
                # HUNYUAN_INFER_PREFILL_FULL_INCLUSIVE=0 reverts to the old buggy form.
                full_attention_slices=[(
                    slice(int(packed_query_position_ids[0].item()), int(packed_query_position_ids[-1].item()) + 1)
                    if os.environ.get("HUNYUAN_INFER_PREFILL_FULL_INCLUSIVE", "1") == "1"
                    else slice(int(packed_query_position_ids[0].item()) + 1, int(packed_query_position_ids[-1].item()))
                )],
            ),
            rope_image_info=_vit_rope_info,
        )
        return outputs.past_key_values

    def prepare_start_tokens(self, curr_kvlens, curr_rope, new_token_ids, modality_type=None):
        self._require_single_sample(curr_kvlens, "curr_kvlens")
        self._require_single_sample(curr_rope, "curr_rope")

        start_key = "start_of_image"
        if modality_type is not None and self.modality_registry is not None:
            try:
                start_key = self.modality_registry.start_token_key(modality_type)
            except Exception:
                pass

        curr_kvlen = int(curr_kvlens[0])
        curr_position_id = int(curr_rope[0])
        return {
            "packed_start_tokens": torch.tensor([int(new_token_ids[start_key])], dtype=torch.long),
            "packed_query_position_ids": torch.tensor([curr_position_id], dtype=torch.long),
            "key_values_lens": torch.tensor([curr_kvlen], dtype=torch.int),
            "packed_key_value_indexes": torch.arange(curr_kvlen, dtype=torch.long),
        }

    @torch.no_grad()
    def generate_text(
        self,
        past_key_values,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
        text_vocab_size: int = None,
    ):
        """Autoregressive text decode using Hunyuan's native cache contract.

        ``packed_start_tokens`` is the registry start token for the requested
        text modality (caption or plain text).  Codebook/coordinate vocabulary
        is masked at ``text_vocab_size`` while the explicit modality end token
        remains allowed.
        """
        del packed_key_value_indexes, key_values_lens
        device = self.device
        curr_tokens = packed_start_tokens.to(device)
        curr_position = packed_query_position_ids.to(device)
        generated = []

        for _ in range(max_length):
            generated.append(curr_tokens.clone())
            text_embeds = self.hunyuan_model.model.wte(curr_tokens).unsqueeze(0)
            outputs = self._run_decoder_with_cache(
                inputs_embeds=text_embeds,
                position_ids=curr_position.unsqueeze(0),
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            logits = self.hunyuan_model.lm_head(
                self.hunyuan_model.model.ln_f(outputs.last_hidden_state)
            )[:, -1, :]

            if text_vocab_size is not None and 0 < int(text_vocab_size) < logits.shape[-1]:
                masked = torch.full_like(logits, float("-inf"))
                masked[:, :int(text_vocab_size)] = logits[:, :int(text_vocab_size)]
                if end_token_id is not None:
                    masked[:, int(end_token_id)] = logits[:, int(end_token_id)]
                logits = masked

            if do_sample:
                probs = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(logits, dim=-1)
            curr_position = curr_position + 1
            if end_token_id is not None and int(curr_tokens[0].item()) == int(end_token_id):
                break

        output_device = generated[0].device if generated else curr_tokens.device
        return torch.stack([tok.to(output_device) for tok in generated], dim=0)

    @torch.no_grad()
    def generate_dino(
        self,
        past_key_values,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 8000,
        top_p: float = 1.0,
        end_token_id: int = None,
        cfg_scale: float = 1.0,
        cfg_past_key_values=None,
        cfg_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_packed_query_position_ids: Optional[torch.LongTensor] = None,
        dino_base: int = 155773,
        dino_length: int = 8192,
        dinolocal: bool = False,
        pos_embed_key: str = None,
    ):
        del packed_key_value_indexes, key_values_lens, cfg_key_values_lens, cfg_packed_key_value_indexes
        device = self.device
        curr_tokens = packed_start_tokens.to(device)
        curr_position = packed_query_position_ids.to(device)
        cfg_position = cfg_packed_query_position_ids.to(device) if cfg_packed_query_position_ids is not None else None

        pos_embed_mod = None
        if pos_embed_key is not None and self._use_forward_pos_embed_in_inference():
            pos_embed_mod = self._resolve_forward_pos_embed_module(pos_embed_key)

        target_name = pos_embed_key if pos_embed_key is not None else ("dinolocal" if dinolocal else "dino")
        rope_image_info = self._build_codebook_rope_info(
            start_pos=int(curr_position[0].item()) + 1,
            total_tokens=max_length - 1,
            target_name=target_name,
        )
        cfg_rope_image_info = self._build_codebook_rope_info(
            start_pos=int(cfg_position[0].item()) + 1 if cfg_position is not None else 0,
            total_tokens=max_length - 1,
            target_name=target_name,
        ) if cfg_position is not None else None

        generated = []
        step = 0
        while step < max_length:
            generated.append(curr_tokens.clone())
            text_embeds = self.hunyuan_model.model.wte(curr_tokens).unsqueeze(0)
            if step > 0 and pos_embed_mod is not None:
                pos_index = torch.tensor([step - 1], device=device, dtype=torch.long)
                text_embeds = text_embeds + pos_embed_mod(pos_index).unsqueeze(0)

            outputs = self._run_decoder_with_cache(
                inputs_embeds=text_embeds,
                position_ids=curr_position.unsqueeze(0),
                past_key_values=past_key_values,
                rope_image_info=rope_image_info,
            )
            past_key_values = outputs.past_key_values
            logits = self.hunyuan_model.lm_head(self.hunyuan_model.model.ln_f(outputs.last_hidden_state))[:, -1, :]

            if cfg_scale > 1.0 and cfg_past_key_values is not None and cfg_position is not None:
                cfg_outputs = self._run_decoder_with_cache(
                    inputs_embeds=text_embeds,
                    position_ids=cfg_position.unsqueeze(0),
                    past_key_values=cfg_past_key_values,
                    rope_image_info=cfg_rope_image_info,
                )
                cfg_past_key_values = cfg_outputs.past_key_values
                cfg_logits = self.hunyuan_model.lm_head(
                    self.hunyuan_model.model.ln_f(cfg_outputs.last_hidden_state)
                )[:, -1, :]
                logits = logits + cfg_scale * (logits - cfg_logits)

            mask = torch.full_like(logits, float("-inf"))
            mask[:, dino_base:dino_base + dino_length] = 0
            logits = logits + mask

            # Anti-collapse decoding: penalize codebook tokens already emitted
            # in the current AR sequence to escape attractors (e.g. token 3000
            # in v22c/v22d). Two penalties, additive on logits within the
            # codebook range only:
            #
            #   HUNYUAN_AR_FREQ_PENALTY  : float  (default 0.0)
            #     Subtracts count(tok) * freq_penalty from logits[tok].
            #     Linear in occurrence count, hits attractors hard.
            #
            #   HUNYUAN_AR_REP_PENALTY   : float  (default 1.0)
            #     If the token already appears in prefix at all, divide its
            #     logit by rep_penalty (or multiply if logit < 0). HF-style.
            #
            # Both apply BEFORE temperature/top-k/top-p so they shape the
            # eventual sampling distribution.
            try:
                freq_penalty = float(os.environ.get("HUNYUAN_AR_FREQ_PENALTY", "0.0"))
            except ValueError:
                freq_penalty = 0.0
            try:
                rep_penalty = float(os.environ.get("HUNYUAN_AR_REP_PENALTY", "1.0"))
            except ValueError:
                rep_penalty = 1.0
            if (freq_penalty != 0.0 or rep_penalty != 1.0) and len(generated) > 0:
                # device_map=auto can place generated tensors across multiple
                # GPUs; collapse to logits.device before cat.
                prefix_tokens = torch.cat(
                    [t.to(logits.device) for t in generated], dim=0
                )  # [num_emitted]
                cb_mask = (prefix_tokens >= dino_base) & (prefix_tokens < dino_base + dino_length)
                cb_in_prefix = prefix_tokens[cb_mask]
                if cb_in_prefix.numel() > 0:
                    counts = torch.bincount(
                        cb_in_prefix - dino_base, minlength=dino_length
                    ).to(dtype=logits.dtype, device=logits.device)
                    if freq_penalty != 0.0:
                        logits[:, dino_base:dino_base + dino_length] = (
                            logits[:, dino_base:dino_base + dino_length] - counts * freq_penalty
                        )
                    if rep_penalty != 1.0:
                        seen_mask = (counts > 0).unsqueeze(0)
                        cb_logits = logits[:, dino_base:dino_base + dino_length]
                        scaled = torch.where(
                            cb_logits > 0,
                            cb_logits / rep_penalty,
                            cb_logits * rep_penalty,
                        )
                        logits[:, dino_base:dino_base + dino_length] = torch.where(
                            seen_mask, scaled, cb_logits
                        )
            debug_logits = logits

            if do_sample:
                logits = logits / temperature
                if top_k > 0:
                    top_k_clamped = min(int(top_k), logits.size(-1))
                    kth = torch.topk(logits, top_k_clamped, dim=-1).values[..., -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits = logits.masked_fill(indices_to_remove, float("-inf"))
                probs = torch.softmax(logits, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(logits, dim=-1)

            _debug_dump_codebook_logits(
                prefix="hunyuan",
                step=step,
                logits=debug_logits,
                chosen_tokens=curr_tokens,
                code_base=dino_base,
                code_length=dino_length,
                topk=int(os.environ.get("MODUS_DUMP_CODEBOOK_LOGIT_TOPK", "10")),
            )

            curr_position = curr_position + 1
            if cfg_position is not None:
                cfg_position = cfg_position + 1
            step += 1
            if end_token_id is not None and int(curr_tokens[0].item()) == int(end_token_id):
                break

        output_device = generated[0].device if generated else curr_tokens.device
        return torch.stack([tok.to(output_device) for tok in generated], dim=0)

    @torch.no_grad()
    def generate_detection_coordonly(
        self,
        past_key_values,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
        cfg_scale: float = 1.0,
        cfg_past_key_values=None,
        cfg_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_packed_query_position_ids: Optional[torch.LongTensor] = None,
        x1_base: int = None,
        y1_base: int = None,
        x2_base: int = None,
        y2_base: int = None,
    ):
        """Generate one grounding box with Hunyuan's native KV-cache path.

        The training format is ``det_start, x1, y1, x2, y2, det_end``.  It is
        a flat coordinate sequence (1D RoPE, no learnable position embedding),
        matching the v16 modality configuration and training forward.
        """
        del packed_key_value_indexes, key_values_lens
        del cfg_key_values_lens, cfg_packed_key_value_indexes
        if None in (x1_base, y1_base, x2_base, y2_base):
            raise ValueError("Hunyuan grounding decode requires explicit coordinate token bases")

        device = self.device
        curr_tokens = packed_start_tokens.to(device)
        curr_position = packed_query_position_ids.to(device)
        cfg_position = (
            cfg_packed_query_position_ids.to(device)
            if cfg_packed_query_position_ids is not None else None
        )
        coord_bases = [int(x1_base), int(y1_base), int(x2_base), int(y2_base)]
        decode_state = 0  # x1 -> y1 -> x2 -> y2 -> end
        x1_token = y1_token = None
        generated = []
        chosen_step_probs = []

        for _step in range(max_length):
            generated.append(curr_tokens.clone())
            text_embeds = self.hunyuan_model.model.wte(curr_tokens).unsqueeze(0)
            outputs = self._run_decoder_with_cache(
                inputs_embeds=text_embeds,
                position_ids=curr_position.unsqueeze(0),
                past_key_values=past_key_values,
                rope_image_info=None,
            )
            past_key_values = outputs.past_key_values
            logits = self.hunyuan_model.lm_head(
                self.hunyuan_model.model.ln_f(outputs.last_hidden_state)
            )[:, -1, :].float()

            if cfg_scale > 1.0 and cfg_past_key_values is not None and cfg_position is not None:
                cfg_outputs = self._run_decoder_with_cache(
                    inputs_embeds=text_embeds,
                    position_ids=cfg_position.unsqueeze(0),
                    past_key_values=cfg_past_key_values,
                    rope_image_info=None,
                )
                cfg_past_key_values = cfg_outputs.past_key_values
                cfg_logits = self.hunyuan_model.lm_head(
                    self.hunyuan_model.model.ln_f(cfg_outputs.last_hidden_state)
                )[:, -1, :].float()
                logits = logits + cfg_scale * (logits - cfg_logits)

            mask = torch.full_like(logits, float("-inf"))
            if decode_state in (0, 1):
                base = coord_bases[decode_state]
                mask[:, base:base + 1000] = 0
            elif decode_state in (2, 3):
                base = coord_bases[decode_state]
                lower_offset = int(x1_token) - coord_bases[0] if decode_state == 2 else int(y1_token) - coord_bases[1]
                mask[:, base + lower_offset:base + 1000] = 0
            else:
                mask[:, int(end_token_id)] = 0
            logits = logits + mask

            probs = torch.softmax(logits / temperature if do_sample else logits, dim=-1)
            if do_sample:
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(logits, dim=-1)
            chosen_step_probs.append(probs[0, curr_tokens][0])

            if decode_state == 0:
                x1_token = curr_tokens[0]
            elif decode_state == 1:
                y1_token = curr_tokens[0]
            decode_state += 1
            curr_position = curr_position + 1
            if cfg_position is not None:
                cfg_position = cfg_position + 1
            if decode_state > 4 or (
                end_token_id is not None and int(curr_tokens[0]) == int(end_token_id)
            ):
                break

        output_device = generated[0].device if generated else curr_tokens.device
        tokens_stack = torch.stack([t.to(output_device) for t in generated], dim=0)
        probs_stack = (
            torch.stack(chosen_step_probs, dim=0).to(output_device)
            if chosen_step_probs else None
        )
        return tokens_stack, probs_stack

    def generate_cocodet(
        self,
        past_key_values,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        # cocodet vocab layout (passed by the inferencer from the tokenizer):
        x1_base: int,
        y1_base: int,
        x2_base: int,
        y2_base: int,
        cls_base: int,
        n_cls: int,
        cocodet_end_token: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        cfg_scale: float = 1.0,
        cfg_past_key_values=None,
        cfg_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_packed_query_position_ids: Optional[torch.LongTensor] = None,
    ):
        """AR cocodet (Pix2seq) detection with a coords+class state machine.

        Mirrors BAGEL.generate_cocodet (bagel.py) but on Hunyuan's KV-cache AR
        loop (same scaffolding as generate_dino). cocodet tokens are a flat box
        sequence — NOT a 2D grid — so RoPE is 1D (rope_image_info=None) and there
        is no learnable pos_embed, matching how cocodet is packed as 'text' in
        training.

        Format per box: <|x1|><|y1|><|x2|><|y2|><|coco_cls_K|> ... <|cocodet_end|>
        State machine: x1 -> y1 -> x2 -> y2 -> cls -> end/next-x1.
        """
        del packed_key_value_indexes, key_values_lens, cfg_key_values_lens, cfg_packed_key_value_indexes
        device = self.device
        curr_tokens = packed_start_tokens.to(device)
        curr_position = packed_query_position_ids.to(device)
        cfg_position = cfg_packed_query_position_ids.to(device) if cfg_packed_query_position_ids is not None else None

        # 0=x1, 1=y1, 2=x2, 3=y2, 4=cls, 9=next/end, 10=done
        decode_state = 0
        x1_token = y1_token = None
        coord_bases = [x1_base, y1_base, x2_base, y2_base]

        generated = []
        chosen_step_probs = []
        step = 0
        while step < max_length:
            generated.append(curr_tokens.clone())
            text_embeds = self.hunyuan_model.model.wte(curr_tokens).unsqueeze(0)

            outputs = self._run_decoder_with_cache(
                inputs_embeds=text_embeds,
                position_ids=curr_position.unsqueeze(0),
                past_key_values=past_key_values,
                rope_image_info=None,   # flat box sequence -> 1D RoPE (no 2D grid)
            )
            past_key_values = outputs.past_key_values
            logits = self.hunyuan_model.lm_head(self.hunyuan_model.model.ln_f(outputs.last_hidden_state))[:, -1, :]

            if cfg_scale > 1.0 and cfg_past_key_values is not None and cfg_position is not None:
                cfg_outputs = self._run_decoder_with_cache(
                    inputs_embeds=text_embeds,
                    position_ids=cfg_position.unsqueeze(0),
                    past_key_values=cfg_past_key_values,
                    rope_image_info=None,
                )
                cfg_past_key_values = cfg_outputs.past_key_values
                cfg_logits = self.hunyuan_model.lm_head(
                    self.hunyuan_model.model.ln_f(cfg_outputs.last_hidden_state)
                )[:, -1, :]
                logits = logits + cfg_scale * (logits - cfg_logits)

            # ── Constrained decoding: mask logits to the slot's valid tokens ──
            mask = torch.full_like(logits, float("-inf"))
            if decode_state in (0, 1):
                base = coord_bases[decode_state]
                mask[:, base:base + 1000] = 0
            elif decode_state in (2, 3):                  # x2>=x1, y2>=y1
                base = coord_bases[decode_state]
                lower = base + (int(x1_token) - x1_base if decode_state == 2 else int(y1_token) - y1_base)
                mask[:, lower:base + 1000] = 0
            elif decode_state == 4:                       # class token
                mask[:, cls_base:cls_base + n_cls] = 0
            elif decode_state == 9:                       # next box (x1) OR stop
                mask[:, x1_base:x1_base + 1000] = 0
                mask[:, cocodet_end_token] = 0
            logits = logits + mask

            probs = torch.softmax(logits / temperature if do_sample else logits, dim=-1)
            if do_sample:
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(logits, dim=-1)
            chosen_step_probs.append(probs[0, curr_tokens][0])

            if decode_state == 0:
                x1_token = curr_tokens[0]
            elif decode_state == 1:
                y1_token = curr_tokens[0]

            # ── State transitions ─────────────────────────────────────────────
            if decode_state in (0, 1, 2, 3):
                decode_state += 1                          # 0→1→2→3→4
            elif decode_state == 4:
                decode_state = 9                           # cls → next/end
            elif decode_state == 9:
                if int(curr_tokens[0]) == cocodet_end_token:
                    decode_state = 10
                elif x1_base <= int(curr_tokens[0]) <= x1_base + 999:
                    x1_token = curr_tokens[0]               # this slot IS next box's x1
                    decode_state = 1                       # → predict y1

            curr_position = curr_position + 1
            if cfg_position is not None:
                cfg_position = cfg_position + 1
            step += 1
            if decode_state == 10:
                break

        output_device = generated[0].device if generated else curr_tokens.device
        tokens_stack = torch.stack([t.to(output_device) for t in generated], dim=0)
        probs_stack = torch.stack(chosen_step_probs, dim=0).to(output_device) if chosen_step_probs else None
        return tokens_stack, probs_stack

    # ── FSDP layer declarations ───────────────────────────────────────────────

    @staticmethod
    def fsdp_wrap_modules():
        """Layer classes for FSDP auto-wrap policy.

        Only leaf transformer-layer types should be listed here.  Do NOT
        include a container class (e.g. Siglip2VisionTransformer) when one of
        its child layer types (e.g. Siglip2EncoderLayer) is also in this set:
        FSDP would wrap the children first and then try to re-wrap the parent
        container, causing an AssertionError in _recursive_wrap.
        """
        from modeling.hunyuan_image_3.src.modeling_hunyuan_image_3 import HunyuanImage3DecoderLayer
        if os.environ.get("HUNYUAN_FSDP_COALESCE_VISION", "0") == "1":
            # The ViT is frozen in stage 1. Wrapping every encoder layer as a
            # separate FSDP unit creates ~28 small forward AllGathers per step
            # without any matching parameter gradients. Leave those parameters
            # in the root FSDP unit instead: still sharded, same maths, fewer
            # collective launches. Opt-in until the memory/speed smoke passes.
            return {HunyuanImage3DecoderLayer}
        from modeling.hunyuan_image_3.src.siglip2 import Siglip2EncoderLayer, LightProjector
        return {HunyuanImage3DecoderLayer, Siglip2EncoderLayer, LightProjector}

    @staticmethod
    def fsdp_checkpoint_modules():
        """Layer classes for activation checkpointing."""
        from modeling.hunyuan_image_3.src.modeling_hunyuan_image_3 import HunyuanImage3DecoderLayer
        from modeling.hunyuan_image_3.src.siglip2 import Siglip2EncoderLayer, LightProjector
        return (HunyuanImage3DecoderLayer, Siglip2EncoderLayer, LightProjector)

    # ── ViT processing ────────────────────────────────────────────────────────

    def _compute_vit_embeddings(
        self,
        packed_vit_tokens: Tensor,
        packed_vit_position_ids: Tensor,
        vit_token_seqlens: Tensor,
        device: torch.device,
    ) -> Tensor:
        """
        Convert MODUS packed ViT tokens to Siglip2's padded-batch format,
        run through HunyuanImage3's vision_model + vision_aligner,
        and return unpacked embeddings.

        Args:
            packed_vit_tokens: (total_tokens, C * patch_size^2) pre-patchified pixels
            packed_vit_position_ids: (total_tokens,) flattened 2D grid positions
            vit_token_seqlens: (n_images,) number of patches per image
            device: target device

        Returns:
            (total_tokens, model_hidden_size) ViT embeddings projected to model dim
        """
        seqlens = vit_token_seqlens.tolist()
        per_image_tokens = packed_vit_tokens.split(seqlens)
        per_image_positions = packed_vit_position_ids.split(seqlens)
        n_images = len(per_image_tokens)
        max_patches = max(seqlens)
        feat_dim = packed_vit_tokens.size(-1)

        pixel_values = torch.zeros(
            n_images, max_patches, feat_dim, device=device, dtype=packed_vit_tokens.dtype
        )
        attn_mask = torch.zeros(n_images, max_patches, dtype=torch.long, device=device)
        spatial_shapes = torch.zeros(n_images, 2, dtype=torch.long, device=device)

        max_per_side = self.vit_max_num_patch_per_side
        for i, (tokens, positions) in enumerate(zip(per_image_tokens, per_image_positions)):
            slen = tokens.size(0)
            pixel_values[i, :slen] = tokens
            attn_mask[i, :slen] = 1
            x = positions % max_per_side
            y = positions // max_per_side
            spatial_shapes[i, 0] = y.max().item() + 1
            spatial_shapes[i, 1] = x.max().item() + 1

        vit_embeds = self.hunyuan_model._forward_vision_encoder(
            pixel_values,
            spatial_shapes=spatial_shapes,
            attention_mask=attn_mask,
        )  # (n_images, max_patches, model_hidden_size)

        parts = [vit_embeds[i, :seqlens[i]] for i in range(n_images)]
        return torch.cat(parts, dim=0)  # (total_tokens, model_hidden_size)

    # ── Loss helpers ──────────────────────────────────────────────────────────

    def _per_modality_losses(self, loss: Tensor, modality_ids: Optional[Tensor], detach: bool = False) -> Optional[Dict]:
        if modality_ids is None or self.modality_registry is None:
            return None
        result: Dict[str, Tensor] = {}
        for mid in torch.unique(modality_ids):
            mask = modality_ids == mid
            if mask.any():
                try:
                    name = self.modality_registry.modality_name(mid.item())
                except Exception:
                    name = str(mid.item())
                result[name] = loss[mask].detach() if detach else loss[mask]
        return result

    def _code_token_gather_lut(self, spec, vocab_size, device):
        """Cache & return (ids LongTensor, lut LongTensor) for a dispersed
        ``code_token_ids`` modality (e.g. cocodet).

        ``ids`` is the (possibly non-contiguous) set of CE-loss token ids.
        ``lut`` maps global token id -> local index into ``ids`` (-1 if absent),
        so target remapping and membership tests are O(1) gathers.
        """
        cache = getattr(self, "_code_token_gather_cache", None)
        if cache is None:
            cache = {}
            self._code_token_gather_cache = cache
        entry = cache.get(spec.name)
        if entry is None or entry[0].device != device:
            ids = torch.as_tensor(list(spec.code_token_ids), dtype=torch.long, device=device)
            lut = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
            lut[ids] = torch.arange(ids.numel(), dtype=torch.long, device=device)
            entry = (ids, lut)
            cache[spec.name] = entry
        return entry

    def _modality_ce(self, spec, ce_preds: Tensor, label_ids: Tensor, modality_mask: Tensor):
        """Per-modality CE with vocab slicing (mirrors BAGEL's _modality_ce).

        For codebook modalities (dino, clip, imagebind, …) slices logits to the
        modality's own token range [base, base+length) before computing softmax.
        This reduces the effective vocab from ~151K → ~8K, giving gradients that
        are properly focused and loss values comparable to BAGEL's.

        Returns (loss, active_mask) or (None, None) when nothing to compute.
        """
        # Dispersed-alphabet modalities (e.g. cocodet) cannot use a single
        # contiguous range: CE is gathered over the explicit id set instead.
        if spec.code_token_ids is not None:
            if not modality_mask.any():
                return None, None
            ids, lut = self._code_token_gather_lut(spec, ce_preds.size(-1), ce_preds.device)
            local = lut[label_ids]
            active_mask = modality_mask & (local >= 0)
            if not active_mask.any():
                return None, None
            logits = ce_preds[active_mask][:, ids]   # gather dispersed columns -> [N, len(ids)]
            targets = local[active_mask]
            loss = F.cross_entropy(logits.float(), targets, reduction="none")
            return loss, active_mask

        rng = spec.code_token_range
        if rng is None or not modality_mask.any():
            return None, None

        base, length = rng
        # Exclude delimiter / boundary tokens whose label falls outside the range.
        label_in_range = (label_ids >= base) & (label_ids < base + length)
        active_mask = modality_mask & label_in_range
        if not active_mask.any():
            return None, None

        # Slice logits and remap labels into local codebook space.
        logits = ce_preds[active_mask][:, base : base + length]
        targets = label_ids[active_mask] - base
        loss = F.cross_entropy(logits.float(), targets, reduction="none")
        return loss, active_mask

    def _compute_ce_loss(self, ce_preds: Tensor, ce_loss_modality_ids: Optional[Tensor], packed_label_ids: Tensor):
        """CE loss with per-modality vocab slicing when registry is available.

        Falls back to full-vocab CE for:
          - tokens whose modality has no code_token_range (text / rgb)
          - delimiter tokens whose label falls outside the modality range
          - when no registry is set
        """
        has_registry = (
            self.modality_registry is not None
            and ce_loss_modality_ids is not None
        )

        if not has_registry:
            return F.cross_entropy(ce_preds.float(), packed_label_ids, reduction="none")

        ce = torch.zeros(packed_label_ids.shape[0], dtype=torch.float32, device=packed_label_ids.device)
        handled = torch.zeros_like(ce_loss_modality_ids, dtype=torch.bool)

        for mid in torch.unique(ce_loss_modality_ids):
            try:
                spec = self.modality_registry.get_by_id(int(mid.item()))
            except Exception:
                continue
            m_mask = ce_loss_modality_ids == mid
            loss, active_mask = self._modality_ce(spec, ce_preds, packed_label_ids, m_mask)
            if loss is not None:
                handled[active_mask] = True
                ce[active_mask] = loss

        # Safety fallback: unhandled tokens (text, delimiters, unknown) → full-vocab CE.
        if not handled.all():
            remaining = ~handled
            ce[remaining] = F.cross_entropy(
                ce_preds[remaining].float(), packed_label_ids[remaining], reduction="none"
            )

        return ce

    def _compute_ce_loss_from_hidden(
        self,
        ce_hidden: Tensor,
        ce_loss_modality_ids: Tensor,
        packed_label_ids: Tensor,
    ) -> Tensor:
        """Compute the identical modality-sliced CE without a full-vocab projection.

        The existing path projects every CE row to the full ~175K vocabulary and
        then immediately slices codebook modalities to ~8K columns.  Here those
        rows are multiplied by the matching lm_head weight slice directly. Text,
        delimiters and unknown labels retain the full-vocabulary fallback.
        """
        weight = self.hunyuan_model.lm_head.weight
        bias = self.hunyuan_model.lm_head.bias
        ce = torch.zeros(
            packed_label_ids.shape[0], dtype=torch.float32, device=packed_label_ids.device
        )
        handled = torch.zeros_like(ce_loss_modality_ids, dtype=torch.bool)

        for mid in torch.unique(ce_loss_modality_ids):
            try:
                spec = self.modality_registry.get_by_id(int(mid.item()))
            except Exception:
                continue
            modality_mask = ce_loss_modality_ids == mid

            if spec.code_token_ids is not None:
                ids, lut = self._code_token_gather_lut(
                    spec, weight.shape[0], packed_label_ids.device
                )
                local = lut[packed_label_ids]
                active = modality_mask & (local >= 0)
                if active.any():
                    local_weight = weight.index_select(0, ids)
                    local_bias = bias.index_select(0, ids) if bias is not None else None
                    preds = F.linear(ce_hidden[active], local_weight, local_bias)
                    ce[active] = F.cross_entropy(
                        preds.float(), local[active], reduction="none"
                    )
                    handled[active] = True
                continue

            rng = spec.code_token_range
            if rng is None:
                continue
            base, length = rng
            in_range = (packed_label_ids >= base) & (packed_label_ids < base + length)
            active = modality_mask & in_range
            if active.any():
                local_bias = bias[base:base + length] if bias is not None else None
                preds = F.linear(
                    ce_hidden[active], weight[base:base + length], local_bias
                )
                ce[active] = F.cross_entropy(
                    preds.float(), packed_label_ids[active] - base, reduction="none"
                )
                handled[active] = True

        remaining = ~handled
        if remaining.any():
            preds = F.linear(ce_hidden[remaining], weight, bias)
            ce[remaining] = F.cross_entropy(
                preds.float(), packed_label_ids[remaining], reduction="none"
            )
        return ce

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        # optional BAGEL fields (passed through but not all needed for Hunyuan)
        packed_position_ids: Optional[torch.LongTensor] = None,
        nested_attention_masks: Optional[List[Tensor]] = None,
        split_lens: Optional[List[int]] = None,
        attn_modes: Optional[List[str]] = None,
        target_modality_types: Optional[List[str]] = None,
        vae_image_modality_types: Optional[List[str]] = None,
        # visual understanding
        ce_loss_indexes: Optional[torch.LongTensor] = None,
        ce_loss_modality_ids: Optional[torch.LongTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        vit_image_modality_types: Optional[List[str]] = None,
        # Per-image (token_h, token_w) for ViT-input images, used to build 2D
        # rope_image_info matching the pretrained Hunyuan convention.
        # Optional — old data without this field falls back to 1D RoPE only.
        vit_image_token_shapes: Optional[List[Tuple[int, int]]] = None,
        # visual generation
        padded_latent: Optional[Tensor] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        packed_latent_position_ids: Optional[torch.LongTensor] = None,
        packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[Tensor] = None,
        mse_loss_indexes: Optional[torch.LongTensor] = None,
        mse_loss_modality_ids: Optional[torch.LongTensor] = None,
    ) -> Dict:
        """
        Translate MODUS packed-sequence batch into HunyuanImage-3.0 format,
        run the model via decomposed component calls, compute CE + MSE losses.

        Returns:
            dict with keys: "ce", "mse", "ce_per_modality", "mse_per_modality"
        """
        device = packed_text_ids.device
        seqlen = sequence_length

        # ── 1. Build input_ids (1, seqlen) ───────────────────────────────────
        input_ids = torch.zeros(1, seqlen, dtype=torch.long, device=device)
        input_ids[0, packed_text_indexes] = packed_text_ids

        has_images = (
            padded_latent is not None
            and patchified_vae_latent_shapes is not None
            and packed_vae_token_indexes is not None
            and packed_timesteps is not None
        )
        if has_images:
            input_ids[0, packed_vae_token_indexes] = self.image_token_id

        # ── 2. Build image_mask (1, seqlen) bool ─────────────────────────────
        image_mask: Optional[Tensor] = None
        if has_images:
            image_mask = torch.zeros(1, seqlen, dtype=torch.bool, device=device)
            image_mask[0, packed_vae_token_indexes] = True

        # ── 3. Build the dense mask only when SDPA will consume it ────────────
        # A Flex-ready batch previously allocated/fill-copied the full O(S²)
        # float mask (~0.8 GiB at S=14K) and then discarded it. Besides wasting
        # time, that transient made the supposedly memory-saving path harder to
        # run. FLEX_VERIFY reconstructs its reference directly from the nested
        # per-sample masks, so it does not need this global dense tensor either.
        _use_flex = os.environ.get("HUNYUAN_USE_FLEX", "0") == "1"
        _flex_ready = _use_flex and bool(split_lens) and bool(attn_modes)
        attn_mask = None
        if not _flex_ready:
            attn_mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=device)
            offset = 0
            for i, slen in enumerate(sample_lens):
                if nested_attention_masks is not None:
                    sample_mask = nested_attention_masks[i].to(device=device, dtype=torch.float32)
                    attn_mask[0, 0, offset:offset + slen, offset:offset + slen] = sample_mask
                else:
                    attn_mask[0, 0, offset:offset + slen, offset:offset + slen] = 0.0
                offset += slen
            # Ensure no query row is entirely masked.
            _seq = torch.arange(seqlen, device=device)
            attn_mask[0, 0, _seq, _seq] = attn_mask[0, 0, _seq, _seq].clamp(min=0.0)

        # ── 3b. Optionally replace dense mask with FlexAttention BlockMask ────
        # HUNYUAN_USE_FLEX=1 enables torch.nn.attention.flex_attention which
        # avoids materialising the O(S²) float mask and uses a compiled sparse
        # kernel — near-flash speed even with packed-sequence masking.
        _flex_block_mask = None
        if _use_flex and split_lens and attn_modes:
            # v2 builder: reproduces the dense causal/full/noise structure. The
            # old builder below returns None whenever nested_attention_masks is
            # present, which is ALWAYS unless training.use_flex=true -- so
            # HUNYUAN_USE_FLEX=1 alone silently fell back to SDPA and measured
            # nothing. Requires training.use_flex=true so the dataset emits
            # split_lens/attn_modes instead of dense per-sample masks.
            if os.environ.get("HUNYUAN_FLEX_VERIFY", "0") == "1":
                _verify_flex_matches_dense(
                    sample_lens, nested_attention_masks, split_lens, attn_modes,
                    seqlen, device,
                )
            _flex_block_mask = _build_flex_block_mask_v2(
                sample_lens, split_lens, attn_modes, seqlen, device
            )
            if _flex_block_mask is not None and dist.get_rank() == 0 and not getattr(
                    self, "_flex_active_logged", False):
                print("[FLEX] active: v2 block mask built from split_lens/attn_modes",
                      flush=True)
                object.__setattr__(self, "_flex_active_logged", True)
        elif _use_flex:
            if dist.get_rank() == 0 and not getattr(self, "_flex_warned", False):
                print("[FLEX] HUNYUAN_USE_FLEX=1 but split_lens/attn_modes are absent "
                      "-> falling back to dense SDPA. Set training.use_flex=true.",
                      flush=True)
                object.__setattr__(self, "_flex_warned", True)
            _flex_block_mask = _build_flex_block_mask(
                sample_lens, nested_attention_masks, seqlen, device
            )

        # A missing/unsupported FlexAttention implementation must never turn
        # into unmasked bidirectional attention. Reconstruct the dense fallback
        # if mask creation returned None after we optimistically skipped it.
        if _flex_block_mask is None and attn_mask is None:
            attn_mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=device
            )
            offset = 0
            for i, slen in enumerate(sample_lens):
                if nested_attention_masks is not None:
                    sample_mask = nested_attention_masks[i].to(
                        device=device, dtype=torch.float32
                    )
                    attn_mask[0, 0, offset:offset + slen, offset:offset + slen] = sample_mask
                else:
                    attn_mask[0, 0, offset:offset + slen, offset:offset + slen] = 0.0
                offset += slen
            _seq = torch.arange(seqlen, device=device)
            attn_mask[0, 0, _seq, _seq] = attn_mask[
                0, 0, _seq, _seq
            ].clamp(min=0.0)

        # ── 4. Build rope_batch0: VAE latents + 2D codebook token blocks ──────
        # rope_batch0 is always built (not gated on has_images) because 2D
        # codebook modalities (dinolocal, clip, imagebindlocal) also need 2D RoPE
        # even in pure-understanding samples that have no VAE generation.
        rope_batch0: List = []

        images_hunyuan: Optional[List] = None
        timesteps_hunyuan: Optional[List[Tensor]] = None
        noise_tensors: List[Tensor] = []
        per_image_timesteps: List[float] = []
        clean_latents: List[Tensor] = []
        train_timestep_scale = _require_timestep_scale("training forward")

        if has_images:
            images_batch0: List[Tensor] = []
            vae_token_offset = 0
            ts_offset = 0

            for h, w in patchified_vae_latent_shapes:
                n_tokens = h * w
                t = float(packed_timesteps[ts_offset].item())
                per_image_timesteps.append(t)

                img_idx = len(images_batch0)
                clean = padded_latent[img_idx][:, :h, :w].clone()
                clean_latents.append(clean)

                # Deterministic train/infer parity diagnostics may provide the
                # exact inference noise on this one wrapper instance.  Normal
                # training never defines this private attribute and therefore
                # retains the original randn_like path unchanged.
                noise_override = getattr(self, "_diag_noise_override", None)
                if noise_override is None:
                    noise = torch.randn_like(clean)
                else:
                    noise = noise_override[img_idx].to(
                        device=clean.device, dtype=clean.dtype
                    )
                    if noise.shape != clean.shape:
                        raise ValueError(
                            "_diag_noise_override shape mismatch: "
                            f"override={tuple(noise.shape)} clean={tuple(clean.shape)}"
                        )
                noise_tensors.append(noise)
                noisy = (1.0 - t) * clean + t * noise
                images_batch0.append(noisy)

                start_pos = int(packed_vae_token_indexes[vae_token_offset].item())
                rope_batch0.append((slice(start_pos, start_pos + n_tokens), (h, w)))

                vae_token_offset += n_tokens
                ts_offset += n_tokens

            images_hunyuan = [images_batch0]
            ts_tensor = torch.tensor(per_image_timesteps, dtype=torch.float32, device=device)
            if train_timestep_scale != 1.0:
                ts_tensor = ts_tensor * train_timestep_scale
            timesteps_hunyuan = [ts_tensor]

        # 4b. Add 2D codebook token blocks to rope_batch0
        if self.modality_registry is not None:
            validate_codebook_rope = os.environ.get("HUNYUAN_DIAG_CODEBOOK_ROPE", "0") == "1"
            safe_codebook_rope = os.environ.get("HUNYUAN_SAFE_CODEBOOK_ROPE", "0") == "1"
            strict_codebook_rope = os.environ.get("HUNYUAN_STRICT_CODEBOOK_ROPE", "0") == "1"
            max_rope_warnings = int(os.environ.get("HUNYUAN_CODEBOOK_ROPE_MAX_WARN", "20"))

            def _sample_id_for_pos(pos: int) -> int:
                offset = 0
                for sid, sample_len in enumerate(sample_lens):
                    end = offset + int(sample_len)
                    if offset <= pos < end:
                        return sid
                    offset = end
                return -1

            for spec in self.modality_registry.specs:
                if spec.codebook_spatial_shape is None or spec.code_token_range is None:
                    continue
                h_cb, w_cb = spec.codebook_spatial_shape
                expected_len = int(h_cb) * int(w_cb)
                base_id, length = spec.code_token_range
                cb_mask = (packed_text_ids >= base_id) & (packed_text_ids < base_id + length)
                if not cb_mask.any():
                    continue
                cb_pos = packed_text_indexes[cb_mask]  # absolute positions in [0, seqlen), ascending
                # Find block boundaries: gap > 1 between consecutive positions marks a new block
                if cb_pos.numel() > 1:
                    breaks = torch.where(torch.diff(cb_pos) > 1)[0] + 1
                    block_start_idxs = torch.cat([
                        torch.zeros(1, dtype=torch.long, device=device), breaks
                    ])
                else:
                    block_start_idxs = torch.zeros(1, dtype=torch.long, device=device)
                block_starts = block_start_idxs.tolist()
                block_ends = block_starts[1:] + [int(cb_pos.numel())]
                for idx, next_idx in zip(block_starts, block_ends):
                    s = int(cb_pos[int(idx)].item())
                    actual_len = int(next_idx) - int(idx)
                    legacy_end = s + expected_len
                    start_sid = _sample_id_for_pos(s)
                    end_sid = _sample_id_for_pos(legacy_end - 1)
                    block_ok = (
                        actual_len == expected_len
                        and start_sid >= 0
                        and start_sid == end_sid
                        and legacy_end <= seqlen
                    )
                    if not block_ok and (validate_codebook_rope or strict_codebook_rope):
                        rank_str = "na"
                        if dist.is_available() and dist.is_initialized():
                            rank_str = str(dist.get_rank())
                        msg = (
                            "[HUNYUAN_CODEBOOK_ROPE_DIAG] "
                            f"rank={rank_str} "
                            f"modality={getattr(spec, 'name', '<unknown>')} "
                            f"start={s} legacy_end={legacy_end} seqlen={seqlen} "
                            f"actual_contiguous_code_tokens={actual_len} expected={expected_len} "
                            f"start_sample={start_sid} end_sample={end_sid} "
                            f"safe_mode={int(safe_codebook_rope)}"
                        )
                        if strict_codebook_rope:
                            raise ValueError(msg)
                        if self._codebook_rope_warn_count < max_rope_warnings:
                            print(msg, flush=True)
                        self._codebook_rope_warn_count += 1
                    if safe_codebook_rope and not block_ok:
                        continue
                    rope_batch0.append((slice(s, legacy_end), (h_cb, w_cb)))

        # 4c. Add 2D rope_image_info entries for ViT-input image segments. This
        # matches the pretrained Hunyuan convention (build_batch_rope_image_info
        # registers cond_vit_image as 2D), which earlier MODUS training had silently
        # dropped (inherited from BAGEL data pipeline). Gated on availability of
        # vit_image_token_shapes — old datasets without this field stay 1D only.
        if (
            vit_image_token_shapes is not None
            and packed_vit_token_indexes is not None
            and vit_token_seqlens is not None
            and len(vit_image_token_shapes) > 0
        ):
            seqlens_list = vit_token_seqlens.tolist() if hasattr(vit_token_seqlens, "tolist") else list(vit_token_seqlens)
            indexes_list = packed_vit_token_indexes.tolist() if hasattr(packed_vit_token_indexes, "tolist") else list(packed_vit_token_indexes)
            cursor = 0
            for n_tokens, (h, w) in zip(seqlens_list, vit_image_token_shapes):
                if cursor < len(indexes_list):
                    start = int(indexes_list[cursor])
                    rope_batch0.append((slice(start, start + int(n_tokens)), (int(h), int(w))))
                cursor += int(n_tokens)

        # Sort all entries by start position (VAE and codebook entries may interleave)
        rope_batch0.sort(key=lambda x: x[0].start)
        rope_image_info = [rope_batch0] if rope_batch0 else None

        # ── 5. Decomposed model forward ──────────────────────────────────────

        # 5a. Embed tokens
        hidden_states = self.hunyuan_model.model.wte(input_ids)

        # 5a.5. (Optional) BAGEL-style additive codebook positional embedding.
        # Gated by HUNYUAN_USE_FORWARD_CODEBOOK_POS_EMBED=1 so the default
        # behaviour (no additive pos_embed, RoPE-only) is preserved. When
        # enabled this mirrors bagel.py:466-491: for every modality with
        # pos_embed_size, look up `<name>_pos_embed[k]` for each codebook
        # token within its sample and add to its hidden state. Per-sample
        # re-enumeration uses sample_lens. Train and infer must agree —
        # the matching inference paths (forward_cache_update_text:621,
        # generate_dino:991) already gate on the same env var via
        # _use_forward_pos_embed_in_inference().
        if (
            os.environ.get("HUNYUAN_USE_FORWARD_CODEBOOK_POS_EMBED", "0") == "1"
            and self.modality_registry is not None
        ):
            for spec in self.modality_registry.modalities_with_forward_pos_embed():
                rng = getattr(spec, "code_token_range", None)
                if rng is None or getattr(spec, "pos_embed_size", None) is None:
                    continue
                base, length = rng
                attr_name = f"{spec.pos_embed_name or spec.name}_pos_embed"
                pos_embed_mod = getattr(self.hunyuan_model, attr_name, None)
                if pos_embed_mod is None:
                    continue
                token_mask = (input_ids[0] >= base) & (input_ids[0] < base + length)
                if not token_mask.any():
                    continue
                positions_in_sequence = torch.nonzero(token_mask, as_tuple=False).squeeze(-1)
                # Per-sample re-enumeration so codebook[i] in sample s gets
                # position i regardless of where sample s sits in the pack.
                positions = torch.zeros_like(positions_in_sequence, dtype=torch.long)
                start_pos = 0
                for sample_len in sample_lens:
                    end_pos = start_pos + sample_len
                    sample_mask = (positions_in_sequence >= start_pos) & (positions_in_sequence < end_pos)
                    if sample_mask.any():
                        sample_positions = torch.arange(
                            int(sample_mask.sum().item()), device=device, dtype=torch.long
                        )
                        positions[sample_mask] = sample_positions
                    start_pos = end_pos
                pos_emb = pos_embed_mod(positions).to(hidden_states.dtype)
                hidden_states[0, positions_in_sequence] = (
                    hidden_states[0, positions_in_sequence] + pos_emb
                )

        if os.environ.get("HUNYUAN_DIAG_WTE_GRAD", "0") == "1":
            dl_start = int(os.environ.get("HUNYUAN_DIAG_DL_START", "142416"))
            dl_end = int(os.environ.get("HUNYUAN_DIAG_DL_END", "150608"))
            input_dl_mask = (input_ids >= dl_start) & (input_ids < dl_end)
            label_count = 0
            if packed_label_ids is not None:
                label_mask = (packed_label_ids >= dl_start) & (packed_label_ids < dl_end)
                label_count = int(label_mask.sum().item())
            self._last_wte_grad_diag = {
                "input_count": int(input_dl_mask.sum().item()),
                "label_count": label_count,
                "grad_count": 0,
                "grad_l2_sum": 0.0,
                "grad_l2_max": 0.0,
                "grad_abs_sum": 0.0,
            }

            def _record_wte_grad(grad):
                if not input_dl_mask.any():
                    return
                g = grad[input_dl_mask]
                if g.numel() == 0:
                    return
                row_l2 = g.float().pow(2).sum(dim=-1).sqrt()
                self._last_wte_grad_diag.update({
                    "grad_count": int(row_l2.numel()),
                    "grad_l2_sum": float(row_l2.sum().item()),
                    "grad_l2_max": float(row_l2.max().item()),
                    "grad_abs_sum": float(g.float().abs().sum().item()),
                })

            hidden_states.register_hook(_record_wte_grad)

        # 5b. Inject ViT embeddings (visual understanding)
        has_vit = (
            packed_vit_tokens is not None
            and packed_vit_token_indexes is not None
            and packed_vit_position_ids is not None
            and vit_token_seqlens is not None
        )
        if has_vit:
            vit_embeds = self._compute_vit_embeddings(
                packed_vit_tokens, packed_vit_position_ids, vit_token_seqlens, device
            )
            hidden_states[0, packed_vit_token_indexes] = vit_embeds.to(hidden_states.dtype)

        # 5c. Inject VAE image tokens (visual generation)
        if has_images:
            hidden_states = self.hunyuan_model.instantiate_vae_image_tokens(
                hidden_states, timesteps_hunyuan, images_hunyuan, image_mask
            )

        # 5c-bis. Fill the <timestep> slots with the real timestep embedding.
        #
        # There are two timestep pathways in native, and they are not redundant:
        #   A) time_embed(t) modulates the latent patches inside patch_embed's
        #      ResBlocks (UNetDown.forward:815) -- LOCAL: tells those patches
        #      what noise level they are at.
        #   B) the <timestep> token slot carries timestep_emb(t) -- GLOBAL: lets
        #      text and other image blocks read this image's t through attention,
        #      which is how a clean condition (t=0) is told apart from a noisy
        #      target in the same sequence.
        # Native fills B via instantiate_continuous_tokens (modeling:3106/3118,
        # gen_timestep + cond_timestep). We bypass the outer forward and call
        # hunyuan_model.model(...) directly, so B's slot was left holding its
        # plain wte row -- a constant, identical for every t. This restores it.
        # HUNYUAN_TIMESTEP_SLOT_FILL=0 keeps the v2 token layout but leaves the
        # slot holding its wte row, which isolates the fill's contribution to
        # GradNorm: timestep_emb receives gradient only when the fill is on, and
        # it received none at all before v2.
        _fill = os.environ.get("HUNYUAN_TIMESTEP_SLOT_FILL", "1") == "1"
        if has_images and self._wire_format_v2() and _fill:
            ts_token_id = self._timestep_token_id()
            ts_pos = (input_ids[0] == ts_token_id).nonzero(as_tuple=False).squeeze(-1)
            ts_all = timesteps_hunyuan[0]
            if ts_pos.numel() != ts_all.numel():
                raise RuntimeError(
                    f"<timestep> slot count ({ts_pos.numel()}) != image count "
                    f"({ts_all.numel()}). Under HUNYUAN_WIRE_FORMAT_V2 every image "
                    f"block -- condition and target alike -- emits exactly one "
                    f"<timestep> token, in the same order as the VAE images."
                )
            if ts_pos.numel() > 0:
                ts_emb = self.hunyuan_model.timestep_emb(ts_all)
                hidden_states[0, ts_pos] = ts_emb.to(hidden_states.dtype)

        # 5d. Pre-compute 2D RoPE (multimodal positional embeddings)
        cos, sin = self.hunyuan_model.cached_rope(
            seqlen, device, rope_image_info=rope_image_info
        )

        dump_acts = os.environ.get("MODUS_DUMP_ACTIVATIONS", "0") == "1"
        act_positions: List[int] = []
        if dump_acts:
            positions_env = os.environ.get("MODUS_DUMP_ACT_POSITIONS")
            if positions_env:
                try:
                    act_positions = [int(item) for item in positions_env.split(",") if item.strip()]
                except ValueError:
                    act_positions = []
            if not act_positions:
                act_positions = [0, 1]
                if packed_vit_token_indexes is not None and packed_vit_token_indexes.numel() > 0:
                    act_positions.extend([int(packed_vit_token_indexes[0].item()), int(packed_vit_token_indexes[-1].item())])
                if ce_loss_indexes is not None and ce_loss_indexes.numel() > 0:
                    ce_probe_positions = [0, 1, 2, 3, 4, 5, 31, 63, 127, 255, 511, 1023]
                    for ce_pos in ce_probe_positions:
                        if ce_pos < ce_loss_indexes.numel():
                            act_positions.append(int(ce_loss_indexes[ce_pos].item()))
                act_positions = sorted(set(act_positions))
            _debug_dump_activation_fingerprint(
                prefix=os.environ.get("MODUS_ACT_BACKEND", os.environ.get("MODUS_CE_LOGIT_BACKEND", "forward")),
                stage="input_injected",
                hidden_states=hidden_states,
                sequence_positions=act_positions,
            )

        # 5e. Run backbone transformer
        # Store the mask on each attention module as well as the short-lived
        # global. Activation-checkpoint recomputation happens later in backward,
        # after this wrapper forward has reset the global; the old code therefore
        # ran Flex in forward but silently ran dense SDPA in recompute (82 vs 73
        # saved tensors). The module attribute remains valid until backward ends
        # and is overwritten at the start of every subsequent batch.
        from modeling.hunyuan_image_3.src.modeling_hunyuan_image_3 import HunyuanImage3SDPAAttention
        for _module in self.hunyuan_model.modules():
            if isinstance(_module, HunyuanImage3SDPAAttention):
                object.__setattr__(_module, "_modus_flex_block_mask", _flex_block_mask)

        # When FlexAttention is active, pass attention_mask=None so layers do
        # not materialise or consume the O(S²) dense mask.
        if _flex_block_mask is not None:
            from modeling.hunyuan_image_3.src.modeling_hunyuan_image_3 import _flex_block_mask_var
            _flex_token = _flex_block_mask_var.set(_flex_block_mask)
            try:
                outputs = self.hunyuan_model.model(
                    attention_mask=None,
                    inputs_embeds=hidden_states,
                    custom_pos_emb=(cos, sin),
                    output_hidden_states=dump_acts,
                    return_dict=True,
                )
            finally:
                _flex_block_mask_var.reset(_flex_token)
        else:
            outputs = self.hunyuan_model.model(
                attention_mask=attn_mask,
                inputs_embeds=hidden_states,
                custom_pos_emb=(cos, sin),
                output_hidden_states=dump_acts,
                return_dict=True,
            )
        hidden_states = outputs[0]
        if dump_acts:
            all_hidden = getattr(outputs, "hidden_states", None)
            if all_hidden is not None:
                n_hidden = len(all_hidden)
                layer_indexes_env = os.environ.get("MODUS_DUMP_ACT_LAYER_INDEXES")
                wanted = {}
                if layer_indexes_env:
                    try:
                        for layer_idx in [int(item) for item in layer_indexes_env.split(",") if item.strip()]:
                            hidden_idx = layer_idx + 1
                            wanted[f"layer{layer_idx}_out"] = hidden_idx
                    except ValueError:
                        wanted = {}
                if not wanted:
                    wanted = {
                        "layer0_out": 1,
                        "layer_mid_out": max(1, n_hidden // 2),
                        "layer_final_out": n_hidden - 1,
                    }
                for stage, idx in wanted.items():
                    if 0 <= idx < n_hidden:
                        _debug_dump_activation_fingerprint(
                            prefix=os.environ.get("MODUS_ACT_BACKEND", os.environ.get("MODUS_CE_LOGIT_BACKEND", "forward")),
                            stage=stage,
                            hidden_states=all_hidden[idx],
                            sequence_positions=act_positions,
                        )

        # 5f. Diffusion prediction (uses hidden_states before ln_f)
        diff_pred = None
        if has_images and os.environ.get("HUNYUAN_SKIP_DIFF_PRED", "0") != "1":
            token_h, token_w = self.hunyuan_model.get_image_tokens_hw(images_hunyuan)
            diff_pred = self.hunyuan_model.ragged_final_layer(
                hidden_states, image_mask, timesteps_hunyuan, token_h, token_w
            )

        # 5g. Text logits (uses hidden_states after ln_f)
        logits_hidden = self.hunyuan_model.model.ln_f(hidden_states)
        if dump_acts:
            _debug_dump_activation_fingerprint(
                prefix=os.environ.get("MODUS_ACT_BACKEND", os.environ.get("MODUS_CE_LOGIT_BACKEND", "forward")),
                stage="ln_f",
                hidden_states=logits_hidden,
                sequence_positions=act_positions,
            )
        # The loss only consumes logits at ce_loss_indexes. Projecting every
        # physical slot to the ~175K vocabulary wastes several GiB and a large
        # GEMM on VAE/ViT/metadata slots whose logits are immediately discarded.
        # Keep this opt-in until paired CE/gradient parity is measured: changing
        # GEMM shape can introduce harmless bf16 rounding even though the maths
        # and selected rows are identical.
        ce_only_head = os.environ.get("HUNYUAN_CE_ONLY_LM_HEAD", "0") == "1"
        modality_sliced_head = (
            ce_only_head
            and os.environ.get("HUNYUAN_MODALITY_SLICED_LM_HEAD", "0") == "1"
            and self.modality_registry is not None
            and ce_loss_modality_ids is not None
            # Diagnostics explicitly need the full argmax/logit tensor.
            and os.environ.get("HUNYUAN_DIAG_CE_ARGMAX", "0") != "1"
            and os.environ.get("MODUS_DUMP_CE_LOGITS", "0") != "1"
        )
        ce_preds: Optional[Tensor] = None
        ce_hidden: Optional[Tensor] = None
        logits = None
        if ce_only_head:
            if ce_loss_indexes is not None and packed_label_ids is not None:
                ce_loss_indexes = ce_loss_indexes.to(logits_hidden.device)
                ce_hidden = logits_hidden[0, ce_loss_indexes, :]
                if not modality_sliced_head:
                    ce_preds = self.hunyuan_model.lm_head(ce_hidden)
        else:
            logits = self.hunyuan_model.lm_head(logits_hidden)

        # ── 6. CE loss (text / token prediction) ─────────────────────────────
        ce: Optional[Tensor] = None
        ce_per_modality: Optional[Dict] = None
        if ce_loss_indexes is not None and packed_label_ids is not None and (logits is not None or ce_preds is not None or ce_hidden is not None):
            target_device = ce_hidden.device if ce_hidden is not None else (ce_preds.device if ce_preds is not None else logits.device)
            ce_loss_indexes = ce_loss_indexes.to(target_device)
            packed_label_ids = packed_label_ids.to(target_device)
            if ce_loss_modality_ids is not None:
                ce_loss_modality_ids = ce_loss_modality_ids.to(target_device)
            if ce_preds is None and not modality_sliced_head:
                ce_preds = logits[0, ce_loss_indexes, :]
            if modality_sliced_head:
                ce = self._compute_ce_loss_from_hidden(
                    ce_hidden, ce_loss_modality_ids, packed_label_ids
                )
                ce_per_modality = self._per_modality_losses(
                    ce, ce_loss_modality_ids, detach=False
                )
            elif os.environ.get("HUNYUAN_DIAG_CE_ARGMAX", "0") == "1":
                # Stash the teacher-forced argmax at every CE position so a
                # caller can decode it and get a cos-sim measured on the
                # TRAINING sequence — same packing, resolution, positions and
                # attention as the loss. The inference context builder produces
                # a different condition block (192 ViT tokens vs training's
                # 1024), so a cos-sim taken there is not the model's number.
                object.__setattr__(self, "_last_ce_argmax", {
                    "argmax": ce_preds.detach().argmax(dim=-1).cpu(),
                    "labels": packed_label_ids.detach().cpu(),
                    "modality_ids": (ce_loss_modality_ids.detach().cpu()
                                     if ce_loss_modality_ids is not None else None),
                    "indexes": ce_loss_indexes.detach().cpu(),
                })
            if not modality_sliced_head:
                _debug_dump_ce_logits(
                prefix=os.environ.get("MODUS_CE_LOGIT_BACKEND", "forward"),
                ce_preds=ce_preds,
                labels=packed_label_ids,
                modality_ids=ce_loss_modality_ids,
                modality_registry=self.modality_registry,
                topk=int(os.environ.get("MODUS_DUMP_CE_LOGIT_TOPK", "10")),
                )
                _debug_dump_ce_argmax_stats(
                prefix=os.environ.get("MODUS_CE_LOGIT_BACKEND", "forward"),
                ce_preds=ce_preds,
                labels=packed_label_ids,
                modality_ids=ce_loss_modality_ids,
                modality_registry=self.modality_registry,
                )
                # Per-modality vocab slicing: codebook modalities (dino, clip, imagebind, …)
                # compute CE over their own ~8K token range instead of the full vocabulary.
                ce = self._compute_ce_loss(ce_preds, ce_loss_modality_ids, packed_label_ids)
                ce_per_modality = self._per_modality_losses(ce, ce_loss_modality_ids, detach=False)

        # ── 7. MSE loss (diffusion / image generation) ───────────────────────
        mse: Optional[Tensor] = None
        mse_per_modality: Optional[Dict] = None
        if has_images and diff_pred is not None:
            C = self.latent_channels
            mse_items: List[Tensor] = []

            for img_idx, (h, w) in enumerate(patchified_vae_latent_shapes):
                t = per_image_timesteps[img_idx]
                if t <= 0.0:
                    continue

                pred_i = diff_pred[0][img_idx]
                pred_flat = pred_i.squeeze(0).reshape(C, h * w).permute(1, 0)

                target_flat = (noise_tensors[img_idx] - clean_latents[img_idx])
                target_flat = target_flat.reshape(C, h * w).permute(1, 0)

                # Stability: compute diffusion residual/loss in fp32; bf16 can
                # overflow in backward for large residuals from pretrained init.
                mse_i = (pred_flat.float() - target_flat.float()) ** 2
                mse_items.append(mse_i)

            if mse_items:
                mse = torch.cat(mse_items, dim=0)
                if mse_loss_modality_ids is not None:
                    active_mse_modality_ids = _select_active_mse_modality_ids(
                        mse_loss_modality_ids, packed_timesteps
                    )
                    mse_per_modality = self._per_modality_losses(
                        mse, active_mse_modality_ids, detach=True
                    )

        router_aux = None
        if os.environ.get("HUNYUAN_ENABLE_MOE_AUX_LOSS", "0") == "1":
            aux_terms = []
            for module in self.hunyuan_model.model.modules():
                aux = getattr(module, "_last_router_aux_loss", None)
                if isinstance(aux, torch.Tensor):
                    aux_terms.append(aux)
            if aux_terms:
                router_aux = torch.stack(aux_terms).mean()

        return dict(
            ce=ce,
            mse=mse,
            ce_per_modality=ce_per_modality,
            mse_per_modality=mse_per_modality,
            router_aux=router_aux,
        )


def _get_input_embeddings(self):
    return self.model.wte

def _set_input_embeddings(self, value):
    self.model.wte = value

def _get_output_embeddings(self):
    return self.lm_head

def _set_output_embeddings(self, value):
    self.lm_head = value

def _patch_embedding_accessors(model: nn.Module) -> None:
    """Bind get/set_input/output_embeddings onto *model* instance.

    HunyuanImage3ForCausalMM doesn't override these from PreTrainedModel,
    so transformers' resize_token_embeddings raises NotImplementedError.
    We patch the instance here so the change stays in the wrapper file.
    """
    model.get_input_embeddings = types.MethodType(_get_input_embeddings, model)
    model.set_input_embeddings = types.MethodType(_set_input_embeddings, model)
    model.get_output_embeddings = types.MethodType(_get_output_embeddings, model)
    model.set_output_embeddings = types.MethodType(_set_output_embeddings, model)


def _select_active_mse_modality_ids(
    mse_loss_modality_ids: Tensor,
    packed_timesteps: Tensor,
) -> Optional[Tensor]:
    """
    Filter mse_loss_modality_ids to those entries where packed_timesteps > 0
    (i.e., non-conditioning / actually-noised tokens).

    Returns the full tensor as fallback if lengths don't align.
    """
    if mse_loss_modality_ids is None:
        return None
    has_mse = packed_timesteps > 0.0
    if has_mse.shape[0] == mse_loss_modality_ids.shape[0]:
        return mse_loss_modality_ids[has_mse]
    return mse_loss_modality_ids
