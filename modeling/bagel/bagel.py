# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import copy
import os
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
import torch.distributed as dist

from data.data_utils import (
    create_sparse_mask, 
    get_flattened_position_ids_extrapolate, 
    get_flattened_position_ids_interpolate,
    patchify, 
)
from .qwen2_navit import (
    NaiveCache,
    Qwen2DecoderLayer,
    Qwen2MoEDecoderLayer,
    Qwen2MoTDecoderLayer,
)
from .modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding, LearnableEmbedding
from .siglip_navit import SiglipEncoderLayer, SiglipVisionTransformer

from tqdm import tqdm

from data.dataset_info import normalize_latents_by_modality


def _debug_dump_codebook_logits(
    *,
    prefix: str,
    step: int,
    logits: torch.Tensor,
    chosen_tokens: Optional[torch.Tensor],
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


class BagelConfig(PretrainedConfig):
    def __init__(
        self,
        visual_gen=True,
        visual_und=True,
        llm_config=None,
        vit_config=None,
        vae_config=None,
        latent_patch_size=2,
        max_latent_size=32,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
        timestep_sample='logit_norm',
        mode_scale=0.0,
        timestep_sample_mix_prob=0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.llm_config = llm_config
        self.vit_config = vit_config
        self.vae_config = vae_config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift
        self.timestep_sample = timestep_sample
        self.mode_scale = mode_scale
        self.timestep_sample_mix_prob = timestep_sample_mix_prob

class Bagel(PreTrainedModel):
    config_class = BagelConfig
    base_model_prefix = 'bagel'

    def __init__(self, language_model, vit_model, config: BagelConfig, modality_registry=None):
        super().__init__(config)    
        self.language_model = language_model
        self.hidden_size = config.llm_config.hidden_size
        self.use_moe = "Mo" in config.llm_config.layer_module
        self.num_heads = config.llm_config.num_attention_heads
        # Optional registry to avoid hard-coded modality ids / token ranges.
        self.modality_registry = modality_registry

        if config.visual_gen:
            self.latent_patch_size = config.latent_patch_size
            self.timestep_shift = config.timestep_shift
            self.latent_downsample = config.vae_config.downsample * config.latent_patch_size
            self.max_latent_size = config.max_latent_size
            self.latent_channel = config.vae_config.z_channels
            self.patch_latent_dim = self.latent_patch_size ** 2 * self.latent_channel
            self.time_embedder = TimestepEmbedder(self.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
            self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.hidden_size)
            self.timestep_sample = config.timestep_sample
            self.mode_scale = config.mode_scale
            self.timestep_sample_mix_prob = config.timestep_sample_mix_prob

        if config.visual_und:
            self.vit_model = vit_model
            self.vit_patch_size = config.vit_config.patch_size
            self.vit_max_num_patch_per_side = config.vit_max_num_patch_per_side
            self.vit_hidden_size = config.vit_config.hidden_size
            self.connector = MLPconnector(self.vit_hidden_size, self.hidden_size, config.connector_act)
            self.vit_pos_embed = PositionEmbedding(self.vit_max_num_patch_per_side, self.hidden_size)

        # Code-modality positional embeddings are created lazily in
        # set_modality_registry() from the YAML config (pos_embed_size field).

        if config.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        self.config = config
        self._init_weights()
        if modality_registry is not None:
            self.set_modality_registry(modality_registry)

    def set_modality_registry(self, modality_registry):
        """
        Attach a ModalityRegistry at runtime and create positional embeddings
        for any codebook modalities that request them (``pos_embed_size`` set).

        Embeddings are stored as named attributes ``self.{attr}_pos_embed``
        so that checkpoint keys are consistent with legacy weights.
        The attribute name defaults to the modality name but can be overridden
        via ``pos_embed_name`` in the YAML (e.g. det → "grounding").
        """
        self.modality_registry = modality_registry
        if self.modality_registry is None:
            return

        # Infer device/dtype from existing model parameters.
        try:
            ref_param = next(self.parameters())
            target_device = ref_param.device
            target_dtype = ref_param.dtype
        except StopIteration:
            target_device = None
            target_dtype = None

        for spec in self.modality_registry.modalities_with_forward_pos_embed():
            if spec.pos_embed_size is None:
                continue
            attr_base = spec.pos_embed_name or spec.name
            attr_name = f"{attr_base}_pos_embed"
            # Don't overwrite if already present (e.g. from a loaded checkpoint).
            if not hasattr(self, attr_name):
                embed = LearnableEmbedding(int(spec.pos_embed_size), self.hidden_size)
                # Move to same device/dtype as the model.
                if target_device is not None:
                    embed = embed.to(device=target_device, dtype=target_dtype)
                setattr(self, attr_name, embed)

    # ── FSDP layer declarations ────────────────────────────────────────────

    @staticmethod
    def fsdp_wrap_modules():
        """Layer classes for FSDP auto-wrap policy."""
        return {
            Qwen2DecoderLayer,
            Qwen2MoEDecoderLayer,
            Qwen2MoTDecoderLayer,
            SiglipEncoderLayer,
            SiglipVisionTransformer,
            MLPconnector,
            TimestepEmbedder,
            PositionEmbedding,
        }

    @staticmethod
    def fsdp_checkpoint_modules():
        """Layer classes for activation checkpointing."""
        return (
            Qwen2DecoderLayer,
            SiglipEncoderLayer,
            MLPconnector,
            Qwen2MoEDecoderLayer,
            Qwen2MoTDecoderLayer,
        )

    def _init_weights(self):
        if self.config.visual_gen:
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)

    def get_modality_name(self, modality_id: int) -> str:
        """Convert modality ID to modality name via the registry."""
        return self.modality_registry.modality_name(modality_id)

    # ── Loss helpers (extracted for readability) ─────────────────────────────

    def _per_modality_losses(self, loss, modality_ids, detach=False):
        """Split a flat loss tensor into a {name: tensor} dict for logging."""
        result = {}
        for mid in torch.unique(modality_ids):
            mask = modality_ids == mid
            if mask.any():
                name = self.get_modality_name(mid.item())
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

    def _modality_ce(self, spec, ce_preds, label_ids, modality_mask):
        """Compute CE for a single modality with vocab slicing.

        Uses ``spec.code_token_range`` (set for every modality by
        ``ModalityRegistry.from_config``).

        Returns ``(loss, active_mask)`` or ``(None, None)`` when nothing applies.
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
            loss = F.cross_entropy(logits, targets, reduction="none")
            return loss, active_mask

        rng = spec.code_token_range
        if rng is None or not modality_mask.any():
            return None, None

        base, length = rng
        # Only tokens whose labels fall within [base, base+length).
        # Delimiter tokens (e.g. <|det_end|>) are tagged with the codebook
        # modality ID but have label IDs outside the range; they are left
        # for the safety fallback (full-vocab CE).
        label_in_range = (label_ids >= base) & (label_ids < base + length)
        active_mask = modality_mask & label_in_range
        if not active_mask.any():
            return None, None

        # Slice logits/labels into the local vocab for this modality.
        logits = ce_preds[active_mask][:, base : base + length]
        targets = label_ids[active_mask] - base

        # Per-modality loss config (reweight / smoothing) — read from registry.
        loss_cfg = spec.loss
        w_loss = None
        if loss_cfg.reweight:
            det_vocab_dim = min(loss_cfg.reweight_det_vocab_dim, int(logits.size(1)))
            w_loss = build_ce_det_loss_weights(
                det_vocab_dim=det_vocab_dim,
                total_vocab_dim=logits.size(1),
                min_w=loss_cfg.reweight_min_w,
            ).to(logits.device)
        loss = F.cross_entropy(logits, targets, reduction="none", weight=w_loss)

        return loss, active_mask

    def _compute_mse_loss(self, last_hidden_state, mse_loss_indexes,
                          noise, packed_latent_clean, packed_timesteps,
                          mse_loss_modality_ids):
        """Flow-matching MSE loss for visual generation tokens."""
        packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
        target = noise - packed_latent_clean  # v_t = dx_t/dt = x_1 - x_0
        has_mse = packed_timesteps > 0
        if packed_mse_preds.numel() == 0 or not has_mse.any():
            dummy_mse = self.llm2vae(last_hidden_state[:1]).sum() * 0.0
            return dummy_mse.reshape(1, 1), {}
        mse = (packed_mse_preds - target[has_mse]) ** 2

        # ── DIAG (gated by env, rank 0, first call only) ────────────────
        if os.environ.get("MODUS_DIAG_FORWARD", "0") == "1":
            try:
                _is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
            except Exception:
                _is_rank0 = True
            if _is_rank0 and not getattr(Bagel, "_diag_printed", False):
                Bagel._diag_printed = True
                _hs = last_hidden_state[mse_loss_indexes]
                _t_used = target[has_mse]
                _diff = packed_mse_preds - _t_used
                print(f"[DIAG_MSE] last_hidden_state[mse_idx]: shape={tuple(_hs.shape)} "
                      f"norm={_hs.detach().float().norm().item():.4f} "
                      f"std={_hs.detach().float().std().item():.4f} "
                      f"absmax={_hs.detach().float().abs().max().item():.4f}", flush=True)
                print(f"[DIAG_MSE] packed_mse_preds (llm2vae out): shape={tuple(packed_mse_preds.shape)} "
                      f"norm={packed_mse_preds.detach().float().norm().item():.4f} "
                      f"std={packed_mse_preds.detach().float().std().item():.4f} "
                      f"absmax={packed_mse_preds.detach().float().abs().max().item():.4f}", flush=True)
                print(f"[DIAG_MSE] target[has_mse] (noise-clean): shape={tuple(_t_used.shape)} "
                      f"norm={_t_used.detach().float().norm().item():.4f} "
                      f"std={_t_used.detach().float().std().item():.4f} "
                      f"absmax={_t_used.detach().float().abs().max().item():.4f}", flush=True)
                print(f"[DIAG_MSE] noise (raw): shape={tuple(noise.shape)} "
                      f"norm={noise.detach().float().norm().item():.4f} "
                      f"std={noise.detach().float().std().item():.4f}", flush=True)
                print(f"[DIAG_MSE] packed_latent_clean (VAE out): shape={tuple(packed_latent_clean.shape)} "
                      f"norm={packed_latent_clean.detach().float().norm().item():.4f} "
                      f"std={packed_latent_clean.detach().float().std().item():.4f}", flush=True)
                print(f"[DIAG_MSE] packed_timesteps[has_mse]: shape={tuple(packed_timesteps[has_mse].shape)} "
                      f"min={packed_timesteps[has_mse].detach().float().min().item():.4f} "
                      f"max={packed_timesteps[has_mse].detach().float().max().item():.4f} "
                      f"mean={packed_timesteps[has_mse].detach().float().mean().item():.4f}", flush=True)
                print(f"[DIAG_MSE] mse element stats: shape={tuple(mse.shape)} "
                      f"mean={mse.detach().float().mean().item():.4f} "
                      f"max={mse.detach().float().max().item():.4f}", flush=True)
                # Cosine similarity of prediction vs target (good proxy for alignment)
                _p = packed_mse_preds.detach().float().reshape(-1)
                _t = _t_used.detach().float().reshape(-1)
                _cos = (_p * _t).sum() / (_p.norm() * _t.norm() + 1e-12)
                print(f"[DIAG_MSE] cos(pred,target)={_cos.item():.4f} "
                      f"(should be >0 if model predicts roughly correct direction)", flush=True)

                # Per-modality breakdown — answers "is bug uniform or modality-specific?"
                if mse_loss_modality_ids is not None:
                    _name_fn = getattr(self, "get_modality_name", None)
                    for _mid in torch.unique(mse_loss_modality_ids).tolist():
                        _mask = mse_loss_modality_ids == _mid
                        if _mask.sum().item() == 0:
                            continue
                        _p_m = packed_mse_preds[_mask].detach().float()
                        _t_m = _t_used[_mask].detach().float()
                        _pflat = _p_m.reshape(-1)
                        _tflat = _t_m.reshape(-1)
                        _cos_m = (_pflat * _tflat).sum() / (_pflat.norm() * _tflat.norm() + 1e-12)
                        _name = _name_fn(int(_mid)) if _name_fn is not None else str(_mid)
                        print(f"[DIAG_MSE_PER_MOD] {_name:>10s} (id={_mid}): "
                              f"n_tok={_mask.sum().item():>5d} "
                              f"pred_std={_p_m.std().item():.4f} "
                              f"target_std={_t_m.std().item():.4f} "
                              f"cos={_cos_m.item():+.4f} "
                              f"mse_mean={((_p_m - _t_m) ** 2).mean().item():.4f}", flush=True)

        mse_per_modality = None
        if mse_loss_modality_ids is not None:
            mse_per_modality = self._per_modality_losses(mse, mse_loss_modality_ids, detach=True)
        return mse, mse_per_modality

    def _compute_ce_loss(self, last_hidden_state, ce_loss_indexes,
                         ce_loss_modality_ids, packed_label_ids):
        """Cross-entropy loss with unified per-modality vocab slicing.

        Every modality's ``spec.code_token_range`` is set at registry
        construction time (codebook modalities get their own range,
        text/image modalities get ``(0, text_vocab_end)``), so the same
        slicing logic applies uniformly to all modalities.
        """
        packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])
        if packed_ce_preds.numel() == 0:
            dummy_ce = self.language_model.lm_head(last_hidden_state[:1]).sum() * 0.0
            return dummy_ce.reshape(1), {}

        has_registry = (
            self.modality_registry is not None
            and ce_loss_modality_ids is not None
            and packed_label_ids is not None
        )

        if not has_registry:
            # No per-modality info → plain full-vocab CE.
            ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")
        else:
            assert packed_label_ids is not None
            ce = torch.zeros_like(packed_label_ids, dtype=torch.float32,
                                  device=packed_label_ids.device)
            handled = torch.zeros_like(ce_loss_modality_ids, dtype=torch.bool)

            for mid in torch.unique(ce_loss_modality_ids):
                try:
                    spec = self.modality_registry.get_by_id(int(mid.item()))
                except Exception:
                    continue
                m_mask = ce_loss_modality_ids == mid
                loss, active_mask = self._modality_ce(
                    spec, packed_ce_preds, packed_label_ids, m_mask,
                )
                if loss is not None:
                    handled[active_mask] = True
                    ce[active_mask] = loss

            # Safety fallback: tokens whose labels fall outside their
            # modality's range (e.g. delimiter tokens) → full-vocab CE.
            if not handled.all():
                remaining = ~handled
                ce[remaining] = F.cross_entropy(
                    packed_ce_preds[remaining], packed_label_ids[remaining],
                    reduction="none",
                )

        # Not detached: ce_loss_average_over_modalities backprops through these.
        ce_per_modality = None
        if ce_loss_modality_ids is not None:
            ce_per_modality = self._per_modality_losses(ce, ce_loss_modality_ids, detach=False)
        return ce, ce_per_modality

    # ── End loss helpers ─────────────────────────────────────────────────────

    def forward(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        target_modality_types: List[str] = None,
        vae_image_modality_types: List[str] = None,
        # for visual understanding
        ce_loss_indexes: Optional[torch.BoolTensor] = None,
        ce_loss_modality_ids: Optional[torch.LongTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        vit_image_modality_types: Optional[List[str]] = None,
        vit_image_token_shapes: Optional[List[Tuple[int, int]]] = None,
        # for visual generation
        padded_latent: Optional[torch.Tensor] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        packed_latent_position_ids: Optional[torch.LongTensor] = None,
        packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[torch.LongTensor] = None,
        mse_loss_indexes: Optional[torch.BoolTensor] = None,
        mse_loss_modality_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            sequence_length: length of sequence.
            packed_text_ids: 1-D int tensor, packed text token ids.
            packed_text_indexes: 1-D int tensor, packed text token indexes in sequence.
            sample_lens: A list of N ints, length of each sample in packed_sequence.
            nested_attention_masks: A list of N 2-D float tensor,  where 0.0 means attention and 
                -inf means ignore.
            packed_position_ids: packed 1-D positions, an image has only one global position shared
                by all latent tokens.

            packed_vit_tokens: packed patchified image tokens for vit model.
            packed_vit_position_ids: 1-D int tensor, the position of each token for vit model.
            packed_vit_token_indexes: 1-D int tensor, packed vit token indexes in sequence.
            vit_token_seqlens: 1-D int tensor, the length of each image tokens for vit model.
            packed_label_ids: 1-D int tensor, packed label token ids.
            ce_loss_indexes: 1-D bool tensor, where to compute ce loss.
            ce_loss_modality_ids: 1-D int tensor, modality ID for each CE loss token.

            padded_latent: padded latent from VAE encoder.
            patchified_vae_latent_shapes: A list of (h, w) tuples, patchfied latent shapes of each image.
            packed_latent_position_ids: 1-D int tensor, the position of each token for latent.
            packed_vae_token_indexes: 1-D int tensor, padded image token indexes in sequence.
            packed_timesteps: 1-D float tensor, final flow timesteps in [0,1] sampled
                by the dataset. 0 = clean / conditioning image, >0 = noisy target.
            mse_loss_indexes: 1-D bool tensor, where to compute mse loss.
            mse_loss_modality_ids: 1-D int tensor, modality ID for each MSE loss token.
        """

        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        if nested_attention_masks is None:
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            seqlen = sum(sample_lens)
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen, 
                device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask
        else:
            attention_mask = nested_attention_masks

        if (
            self.config.visual_und
            and vit_token_seqlens is not None
            and vit_token_seqlens.numel() > 0
        ):
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
            cu_seqlens = cu_seqlens.to(torch.int32)
            max_seqlen = torch.max(vit_token_seqlens).item()
            packed_vit_token_embed = self.vit_model(
                packed_pixel_values=packed_vit_tokens, 
                packed_flattened_position_ids=packed_vit_position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            packed_vit_token_embed = self.connector(packed_vit_token_embed)
            vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
            packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        # --- Codebook positional embeddings (config-driven, one loop for all) ---
        if self.modality_registry is not None:
            for spec in self.modality_registry.modalities_with_forward_pos_embed():
                rng = spec.code_token_range
                if rng is None:
                    continue
                base, length = rng
                attr_name = f"{spec.pos_embed_name or spec.name}_pos_embed"
                pos_embed_mod = getattr(self, attr_name, None)
                if pos_embed_mod is None:
                    continue
                token_mask = (packed_text_ids >= base) & (packed_text_ids < base + length)
                if not token_mask.any():
                    continue
                positions_in_sequence = packed_text_indexes[token_mask]
                positions = torch.zeros_like(positions_in_sequence, dtype=torch.long)
                start_pos = 0
                for sample_len in sample_lens:
                    end_pos = start_pos + sample_len
                    sample_mask = (positions_in_sequence >= start_pos) & (positions_in_sequence < end_pos)
                    if sample_mask.any():
                        sample_positions = torch.arange(sample_mask.sum(), device=packed_text_ids.device)
                        positions[sample_mask] = sample_positions
                    start_pos = end_pos
                pos_emb = pos_embed_mod(positions)
                packed_sequence[positions_in_sequence] = packed_sequence[positions_in_sequence] + pos_emb

        if self.config.visual_gen and patchified_vae_latent_shapes is not None and len(patchified_vae_latent_shapes) > 0:
            p = self.latent_patch_size
            packed_latent = []
            for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
                packed_latent.append(latent)
            packed_latent_clean = torch.cat(packed_latent, dim=0)

            noise = torch.randn_like(packed_latent_clean)
            # packed_timesteps already contains final [0,1] values sampled by
            # the dataset (see PackedDataset._sample_timestep).  0 → clean /
            # conditioning image, >0 → noisy target.  Apply timestep shift:
            packed_timesteps = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)
            packed_latent = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:, None] * noise
            packed_timestep_embeds = self.time_embedder(packed_timesteps)
            latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
            packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + latent_token_pos_emb
            packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes=torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_vae_token_indexes,
            )

        last_hidden_state = self.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )

        mse, mse_per_modality = None, None
        if self.config.visual_gen:
            if patchified_vae_latent_shapes is not None and len(patchified_vae_latent_shapes) > 0:
                mse, mse_per_modality = self._compute_mse_loss(
                    last_hidden_state, mse_loss_indexes,
                    noise, packed_latent_clean, packed_timesteps,
                    mse_loss_modality_ids,
                )
            else:
                # Some non-mandatory packed batches are CE-only on a local rank
                # while other ranks have visual-generation tokens. Keep the
                # visual-generation modules in the local autograd graph so FSDP
                # backward collectives stay symmetric across ranks.
                dummy_latent = last_hidden_state.new_zeros((1, self.patch_latent_dim))
                dummy_timestep = last_hidden_state.new_zeros((1,))
                dummy_pos = torch.zeros(
                    (1,), dtype=torch.long, device=last_hidden_state.device
                )
                dummy_mse = (
                    self.vae2llm(dummy_latent).sum()
                    + self.time_embedder(dummy_timestep).sum()
                    + self.latent_pos_embed(dummy_pos).sum()
                    + self.llm2vae(last_hidden_state[:1]).sum()
                ) * 0.0
                mse = dummy_mse.reshape(1, 1)
                mse_per_modality = {}

        ce, ce_per_modality = None, None
        if ce_loss_indexes is not None:
            ce, ce_per_modality = self._compute_ce_loss(
                last_hidden_state, ce_loss_indexes,
                ce_loss_modality_ids, packed_label_ids,
            )
        elif self.config.visual_und:
            ce = (self.language_model.lm_head(last_hidden_state[:1]).sum() * 0.0).reshape(1)
            ce_per_modality = {}

        return dict(
            mse=mse, 
            ce=ce,
            mse_per_modality=mse_per_modality,
            ce_per_modality=ce_per_modality
        )


    def _resolve_token_keys(self, modality_type):
        """Return ``(start_key, end_key)`` for wrapping text in the given modality."""
        if modality_type is None:
            return 'bos_token_id', 'eos_token_id'
        # Legacy alias: many inference/eval paths (VQA, dino eval, ...) pass the
        # literal "image" instead of the registered name "rgb".  Map it to the image
        # start/end tokens so the image block is wrapped with <|vision_start|> — not
        # bos.  Without this the registry lookup misses and the silent fallback below
        # used bos, silently breaking VQA/grounding (BLINK 88%->30%).
        if modality_type == "image":
            return 'start_of_image', 'end_of_image'
        if self.modality_registry is not None:
            try:
                spec = self.modality_registry.get(modality_type)
                return spec.start_token_key, spec.end_token_key
            except KeyError:
                pass
        # Unknown modality: warn loudly instead of silently using plain-text
        # delimiters (the silent fallback once masked an image-token regression).
        import warnings
        warnings.warn(
            f"_resolve_token_keys: unknown modality_type {modality_type!r}; falling "
            f"back to bos/eos delimiters — likely a train/inference token mismatch.",
            RuntimeWarning, stacklevel=2,
        )
        return 'bos_token_id', 'eos_token_id'

    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids, modality_type=None):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        start_key, end_key = self._resolve_token_keys(modality_type)

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids[start_key]] + text_ids + [new_token_ids[end_key]]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        pos_embed_key: Optional[str] = None,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        if self.modality_registry is not None:
            # Match training behavior: for codebook tokens present in this text
            # update, add each modality's learnable positional embedding.
            if pos_embed_key is not None:
                specs = []
                try:
                    spec = self.modality_registry.get(pos_embed_key)
                except Exception:
                    spec = None
                if spec is not None:
                    specs = [spec]
            else:
                try:
                    specs = self.modality_registry.modalities_with_forward_pos_embed()
                except Exception:
                    specs = []

            text_lens = [int(x) for x in text_token_lens.detach().cpu().tolist()]
            for spec in specs:
                code_range = getattr(spec, "code_token_range", None)
                if code_range is None:
                    continue
                base, length = code_range
                pos_attr = f"{spec.pos_embed_name or spec.name}_pos_embed"
                pos_embed_mod = getattr(self, pos_attr, None)
                if pos_embed_mod is None:
                    continue

                token_mask = (packed_text_ids >= base) & (packed_text_ids < base + length)
                if not token_mask.any():
                    continue

                code_indices = torch.nonzero(token_mask, as_tuple=False).squeeze(1)
                code_positions = torch.zeros_like(code_indices, dtype=torch.long)
                seq_start = 0
                for seq_len in text_lens:
                    seq_end = seq_start + seq_len
                    in_seq = (code_indices >= seq_start) & (code_indices < seq_end)
                    if in_seq.any():
                        code_positions[in_seq] = torch.arange(
                            int(in_seq.sum().item()),
                            device=packed_text_ids.device,
                            dtype=torch.long,
                        )
                    seq_start = seq_end
                packed_text_embedding[code_indices] = (
                    packed_text_embedding[code_indices] + pos_embed_mod(code_positions)
                )

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids, modality_type=None):
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids[self._resolve_token_keys(modality_type)[0]])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vit_position_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2), 
                self.vit_patch_size, 
                max_num_patches_per_side=self.vit_max_num_patch_per_side
            )
            vit_tokens = patchify(image_tensor, self.vit_patch_size)
            packed_vit_tokens.append(vit_tokens)
            num_img_tokens = vit_tokens.shape[0]
            packed_vit_position_ids.append(vit_position_ids)
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            modality_key = 'end_of_image'
            packed_text_ids.append(new_token_ids[modality_key])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vit(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item()
        packed_vit_token_embed = self.vit_model(
            packed_pixel_values=packed_vit_tokens, 
            packed_flattened_position_ids=packed_vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        packed_vit_token_embed = self.connector(packed_vit_token_embed)
        pos_emb = self.vit_pos_embed(packed_vit_position_ids)
        packed_vit_token_embed = packed_vit_token_embed + pos_emb
        if packed_vit_token_embed.dtype != packed_sequence.dtype:
            packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
        packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids, timestep=0, modality_type=None):
        patchified_vae_latent_shapes, packed_vae_position_ids = list(), list()
        packed_vae_token_indexes = list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        vae_image_tensors = list()
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids[self._resolve_token_keys(modality_type)[0]])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vae_image_tensors.append(image_tensor)
            vae_posiiton_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2),
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)
            H, W = image_tensor.shape[1:]
            h = H // self.latent_downsample
            w = W // self.latent_downsample
            patchified_vae_latent_shapes.append((h, w))

            num_img_tokens = w * h
            packed_vae_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            modality_key = 'end_of_image'
            packed_text_ids.append(new_token_ids[modality_key])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        image_sizes = [item.shape for item in vae_image_tensors]
        max_image_size = [max(item) for item in list(zip(*image_sizes))]
        padded_images = torch.zeros(size=(len(vae_image_tensors), *max_image_size))
        for i, image_tensor in enumerate(vae_image_tensors):
            padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

        generation_input = {
            "padded_images": padded_images,
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_timesteps": torch.tensor([timestep]),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vae(
        self,
        vae_model,
        past_key_values: NaiveCache,
        padded_images: torch.Tensor,
        patchified_vae_latent_shapes: List,
        packed_vae_position_ids: torch.LongTensor,
        packed_timesteps: torch.Tensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.Tensor,
        modality_type: str,
        do_modality_norm: bool,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        padded_latent = vae_model.encode(padded_images)
        if do_modality_norm and modality_type is not None:
            device = next(self.parameters()).device
            padded_latent = normalize_latents_by_modality(padded_latent, modality_type, device)

        p = self.latent_patch_size
        packed_latent = list()
        for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
            latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
            packed_latent.append(latent)
        packed_latent = torch.cat(packed_latent, dim=0)
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        if packed_latent.dtype != packed_sequence.dtype:
            packed_latent = packed_latent.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    # ── AR-loop helpers (shared by generate_text / generate_dino / generate_detection_coordonly) ──

    @staticmethod
    def _reindex_kv(packed_key_value_indexes, key_values_lens):
        """Shift packed KV indexes before a forward step to interleave query tokens."""
        uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
        for i in range(len(uppacked)):
            uppacked[i] += i
        return torch.cat(uppacked, dim=0)

    @staticmethod
    def _advance_kv(packed_key_value_indexes, key_values_lens):
        """Extend each sample's KV index span by one token after a decode step."""
        uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
        for i in range(len(uppacked)):
            # Empty prefix (e.g. the unconditional CFG stream for an image-only
            # condition, which CFG drops): the first decoded token's KV lands at
            # index 0. Otherwise it lands one past this sample's last index.
            next_idx = uppacked[i][-1] + 1 if uppacked[i].numel() > 0 else 0
            uppacked[i] = torch.cat(
                [uppacked[i], torch.tensor([next_idx], device=uppacked[i].device, dtype=uppacked[i].dtype)], dim=0
            )
        return torch.cat(uppacked, dim=0), key_values_lens + 1

    def _ar_forward_step(self, packed_text_embedding, packed_query_position_ids,
                         key_values_lens, packed_key_value_indexes, past_key_values):
        """Run one autoregressive forward step and return ``(past_key_values, pred_logits)``."""
        query_lens = torch.ones(packed_text_embedding.shape[0], device=packed_text_embedding.device,
                                dtype=torch.long)
        packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
            0, len(key_values_lens), device=key_values_lens.device, dtype=key_values_lens.dtype,
        )
        extra_inputs = {"mode": "und"} if self.use_moe else {}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        pred_logits = self.language_model.lm_head(output.packed_query_sequence)
        return output.past_key_values, pred_logits

    def _cfg_forward_and_merge(
        self, packed_text_embedding, pred_logits,
        cfg_scale, cfg_past_key_values, cfg_key_values_lens,
        cfg_packed_key_value_indexes, cfg_packed_query_position_ids,
    ):
        """Run the CFG (unconditional) forward and merge with conditional logits.

        Returns ``(cfg_past_key_values, cfg_packed_key_value_indexes, merged_logits)``.
        """
        if not (cfg_scale > 1.0 and cfg_past_key_values is not None
                and cfg_key_values_lens is not None
                and cfg_packed_query_position_ids is not None
                and cfg_packed_key_value_indexes is not None
                and cfg_key_values_lens.shape == cfg_key_values_lens.shape):
            return cfg_past_key_values, cfg_packed_key_value_indexes, pred_logits

        cfg_packed_key_value_indexes = self._reindex_kv(cfg_packed_key_value_indexes, cfg_key_values_lens)
        cfg_past_key_values, cfg_pred_logits = self._ar_forward_step(
            packed_text_embedding, cfg_packed_query_position_ids,
            cfg_key_values_lens, cfg_packed_key_value_indexes, cfg_past_key_values,
        )
        merged = pred_logits + cfg_scale * (pred_logits - cfg_pred_logits)
        return cfg_past_key_values, cfg_packed_key_value_indexes, merged

    def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids, modality_type=None):
        packed_text_ids, packed_text_indexes = list(), list()
        packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = list(), list(), list()
        packed_position_ids, packed_seqlens, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids[self._resolve_token_keys(modality_type)[0]])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            vae_posiiton_ids = self.get_flattened_position_ids(
                H, W,
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_init_noises.append(
                torch.randn(num_image_tokens, self.latent_channel * self.latent_patch_size ** 2)
            )
            packed_vae_token_indexes.extend(range(query_curr, query_curr + num_image_tokens))
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            modality_key = 'end_of_image'
            packed_text_ids.append(new_token_ids[modality_key])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))
            packed_seqlens.append(num_image_tokens + 2)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    def prepare_vae_latent_cfg(self, curr_kvlens, curr_rope, image_sizes):
        packed_position_ids, packed_indexes, packed_key_value_indexes = list(), list(), list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))

        generation_input = {
            "cfg_packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "cfg_key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "cfg_packed_query_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "cfg_packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    @torch.no_grad
    def generate_image(
        self,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        num_timesteps: int = 24,
        timestep_shift: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Optional[Tuple[float, float]] = [0, 1],
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        target_modality_name: Optional[str] = None,  # accepted for API compat; unused in base Bagel
        cond_rope_segments=None,  # accepted for API compat; used by Hunyuan, ignored by Bagel
        cfg_text_cond_rope_segments=None,
        cfg_img_cond_rope_segments=None,
    ):
        x_t = packed_init_noises

        timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts =  timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):

            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0
            v_t = self._forward_flow(
                x_t=x_t,
                timestep=timestep, 
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                # cfg_text
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                # cfg_img
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                cfg_img_key_values_lens=cfg_img_key_values_lens,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                cfg_type=cfg_type,
            )

            x_t = x_t - v_t.to(x_t.device) * dts[i] # velocity pointing from data to noise

        unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
        return unpacked_latent

    @torch.no_grad
    def _forward_flow(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_key_values_lens: Optional[torch.Tensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_key_values_lens: Optional[torch.Tensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(timestep)
        x_t = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = x_t

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            **extra_inputs,
        )
        v_t = self.llm2vae(output.packed_query_sequence)
        v_t = v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            cfg_text_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids,
                packed_query_indexes=cfg_text_packed_query_indexes,
                past_key_values=cfg_text_past_key_values,
                key_values_lens=cfg_text_key_values_lens,
                packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_text_v_t = self.llm2vae(cfg_text_output.packed_query_sequence)
            cfg_text_v_t = cfg_text_v_t[packed_vae_token_indexes]

        if cfg_img_scale > 1.0:
            cfg_img_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_img_packed_position_ids,
                packed_query_indexes=cfg_img_packed_query_indexes,
                past_key_values=cfg_img_past_key_values,
                key_values_lens=cfg_img_key_values_lens,
                packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_img_v_t = self.llm2vae(cfg_img_output.packed_query_sequence)
            cfg_img_v_t = cfg_img_v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if cfg_renorm_type == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t_text = v_t_text_ * scale
                if cfg_img_scale > 1.0:
                    v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                else:
                    v_t = v_t_text
            else:
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                
                if cfg_img_scale > 1.0:
                    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                else:
                    v_t_ = v_t_text_

                # NOTE norm is computed over all dimensions, thus currently only supports batch_size = 1 with navit
                if cfg_renorm_type == "global":
                    norm_v_t = torch.norm(v_t)
                    norm_v_t_ = torch.norm(v_t_)
                elif cfg_renorm_type == "channel":
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                else:
                    raise NotImplementedError(f"{cfg_renorm_type} is not suppoprted")
                scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t = v_t_ * scale
        else:
            # No CFG
            pass

        return v_t

    def prepare_start_tokens(self, curr_kvlens, curr_rope, new_token_ids, modality_type=None):
        packed_start_tokens, packed_key_value_indexes = list(), list()
        packed_query_position_ids = list()

        start_key, _ = self._resolve_token_keys(modality_type)

        curr = 0
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            packed_start_tokens.append(new_token_ids[start_key])
            packed_query_position_ids.append(curr_position_id)
            curr += curr_kvlen

        generation_input = {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long),
            "packed_query_position_ids": torch.tensor(packed_query_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    @torch.no_grad
    def generate_text(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
    ):
        # Determine text-only vocab upper bound (mask out codebook tokens).
        x1_base = 151671

        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)
            packed_key_value_indexes = self._reindex_kv(packed_key_value_indexes, key_values_lens)
            past_key_values, pred_logits = self._ar_forward_step(
                packed_text_embedding, packed_query_position_ids,
                key_values_lens, packed_key_value_indexes, past_key_values,
            )

            # Mask: allow only text vocab tokens.
            mask = torch.ones_like(pred_logits) * float('-inf')
            mask[:, :x1_base] = 0
            pred_logits = pred_logits + mask

            if do_sample:
                probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            packed_key_value_indexes, key_values_lens = self._advance_kv(packed_key_value_indexes, key_values_lens)
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if end_token_id is not None and curr_tokens[0] == end_token_id:
                break

        output_device = generated_sequence[0].device
        return torch.stack([i.to(output_device) for i in generated_sequence], dim=0)

    @torch.no_grad
    def generate_dino(
        self,
        past_key_values: NaiveCache,
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
        # CFG parameters
        cfg_scale: float = 1.0,
        cfg_past_key_values: Optional[NaiveCache] = None,
        cfg_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_packed_query_position_ids: Optional[torch.LongTensor] = None,

        dino_base: int = 155773,
        dino_length: int = 8192,
        dinolocal: bool = False,
        pos_embed_key: str = None,
    ):
        """AR generation for DINO / DINO-local codebook tokens with optional CFG."""

        # Resolve positional embedding module once.
        # Use pos_embed_key if provided; fall back to legacy dinolocal boolean.
        if pos_embed_key is not None:
            pos_embed_attr = f'{pos_embed_key}_pos_embed'
        else:
            pos_embed_attr = 'dinolocal_pos_embed' if dinolocal else 'dino_pos_embed'
        pos_embed_mod = getattr(self, pos_embed_attr, None)

        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens

        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)

            # Add positional embedding for code tokens (step > 0).
            dino_position = step - 1
            if dino_position >= 0 and pos_embed_mod is not None:
                packed_text_embedding = packed_text_embedding + pos_embed_mod(
                    torch.tensor([dino_position], device=curr_tokens.device)
                )

            packed_key_value_indexes = self._reindex_kv(packed_key_value_indexes, key_values_lens)
            past_key_values, pred_logits = self._ar_forward_step(
                packed_text_embedding, packed_query_position_ids,
                key_values_lens, packed_key_value_indexes, past_key_values,
            )

            # CFG: merge with unconditional logits.
            cfg_past_key_values, cfg_packed_key_value_indexes, pred_logits = self._cfg_forward_and_merge(
                packed_text_embedding, pred_logits,
                cfg_scale, cfg_past_key_values, cfg_key_values_lens,
                cfg_packed_key_value_indexes, cfg_packed_query_position_ids,
            )

            # Upcast to fp32 before the constrained codebook argmax/sample (same
            # bf16 tie-fragility as generate_cocodet): a near-flat distribution over
            # the codebook slice rounds to bf16 ties and argmax picks the lowest
            # index = degenerate constant token, arch/run-dependent. Serves all 5
            # codebook feature modalities (dino/dinolocal/clip/imagebind/imagebindlocal).
            pred_logits = pred_logits.float()

            # Mask: allow only DINO codebook tokens.
            mask = torch.ones_like(pred_logits) * float('-inf')
            mask[:, dino_base:dino_base + dino_length] = 0
            pred_logits = pred_logits + mask
            debug_logits = pred_logits

            if do_sample:
                pred_logits = pred_logits / temperature

                # Top-k filtering.
                if top_k > 0:
                    top_k_clamped = min(int(top_k), pred_logits.size(-1))
                    kth_vals = torch.topk(pred_logits, top_k_clamped, dim=-1).values[..., -1, None]
                    pred_logits = pred_logits.masked_fill(pred_logits < kth_vals, float("-inf"))

                # Top-p (nucleus) filtering.
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(pred_logits, descending=True, dim=-1)
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                    cumprobs = torch.cumsum(sorted_probs, dim=-1)
                    to_remove = cumprobs > top_p
                    to_remove[..., 0] = False
                    to_remove[..., 1:] = to_remove[..., :-1].clone()
                    sorted_logits = sorted_logits.masked_fill(to_remove, float("-inf"))
                    pred_logits = torch.full_like(pred_logits, float("-inf"))
                    pred_logits.scatter_(-1, sorted_idx, sorted_logits)

                probs = nn.functional.softmax(pred_logits, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            _debug_dump_codebook_logits(
                prefix="bagel",
                step=step,
                logits=debug_logits,
                chosen_tokens=curr_tokens,
                code_base=dino_base,
                code_length=dino_length,
                topk=int(os.environ.get("MODUS_DUMP_CODEBOOK_LOGIT_TOPK", "10")),
            )

            # Advance KV caches.
            packed_key_value_indexes, key_values_lens = self._advance_kv(packed_key_value_indexes, key_values_lens)
            packed_query_position_ids = packed_query_position_ids + 1
            if cfg_scale > 1.0 and cfg_key_values_lens is not None:
                cfg_packed_key_value_indexes, cfg_key_values_lens = self._advance_kv(
                    cfg_packed_key_value_indexes, cfg_key_values_lens,
                )
                cfg_packed_query_position_ids = cfg_packed_query_position_ids + 1
            step += 1

            if end_token_id is not None and curr_tokens[0] == end_token_id:
                break

        output_device = generated_sequence[0].device
        return torch.stack([i.to(output_device) for i in generated_sequence], dim=0)

    # NOTE: generate_detection, generate_detection_new were removed — they
    # were unused experimental variants.  Only generate_detection_coordonly
    # (below) is called from the inference pipeline.

    @torch.no_grad
    def generate_detection_coordonly(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
        # CFG parameters
        cfg_scale: float = 1.0,
        cfg_past_key_values: Optional[NaiveCache] = None,
        cfg_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_packed_query_position_ids: Optional[torch.LongTensor] = None,
    ):
        """AR detection with coord-only state machine and optional CFG.

        Format: ``<|x1_XXX|><|y1_XXX|><|x2_XXX|><|y2_XXX|><|det_end|>``
        State machine: x1 → y1 → x2 → y2 → end/next
        """
        # Token ranges (x1/y1/x2/y2 each occupy 1000 ids).
        x1_base, y1_base, x2_base, y2_base = 151671, 152671, 153671, 154671
        det_end_token = 155772

        pos_embed_mod = getattr(self, 'grounding_pos_embed', None)

        step = 0
        generated_sequence = []
        chosen_step_probs = []
        curr_tokens = packed_start_tokens

        decode_state = 0  # 0=x1, 1=y1, 2=x2, 3=y2, 9=next/end, 10=done
        x1_token = y1_token = None  # Track for x2/y2 lower-bound constraint.

        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)

            # Grounding positional embedding.
            grounding_position = step - 1
            if grounding_position >= 0 and pos_embed_mod is not None:
                packed_text_embedding = packed_text_embedding + pos_embed_mod(
                    torch.tensor([grounding_position], device=curr_tokens.device)
                )

            packed_key_value_indexes = self._reindex_kv(packed_key_value_indexes, key_values_lens)
            past_key_values, pred_logits = self._ar_forward_step(
                packed_text_embedding, packed_query_position_ids,
                key_values_lens, packed_key_value_indexes, past_key_values,
            )

            # CFG: merge with unconditional logits.
            cfg_past_key_values, cfg_packed_key_value_indexes, pred_logits = self._cfg_forward_and_merge(
                packed_text_embedding, pred_logits,
                cfg_scale, cfg_past_key_values, cfg_key_values_lens,
                cfg_packed_key_value_indexes, cfg_packed_query_position_ids,
            )

            # Upcast to fp32 before the constrained coordinate argmax/sample (same
            # bf16 tie-fragility as generate_cocodet — det/grounding shares the
            # x1/y1/x2/y2 state machine): a near-flat coordinate distribution rounds
            # to bf16 ties and argmax picks the lowest valid coordinate index =
            # degenerate left-edge boxes, arch/run-dependent.
            pred_logits = pred_logits.float()

            # ── Constrained decoding state machine ────────────────────────────
            coord_bases = [x1_base, y1_base, x2_base, y2_base]
            if decode_state in [0, 1]:
                base = coord_bases[decode_state]
                mask = torch.ones_like(pred_logits) * float('-inf')
                mask[:, base:base + 1000] = 0
                pred_logits = pred_logits + mask
            elif decode_state in [2, 3]:
                base = coord_bases[decode_state]
                lower = base + (x1_token - x1_base if decode_state == 2 else y1_token - y1_base)
                mask = torch.ones_like(pred_logits) * float('-inf')
                mask[:, lower:base + 1000] = 0
                pred_logits = pred_logits + mask
            elif decode_state == 9:
                mask = torch.ones_like(pred_logits) * float('-inf')
                mask[:, det_end_token] = 0
                pred_logits = pred_logits + mask

            # ── Sample / argmax ───────────────────────────────────────────────
            probs = nn.functional.softmax(pred_logits / temperature if do_sample else pred_logits, dim=-1)
            if do_sample:
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)
            chosen_step_probs.append(probs[0, curr_tokens][0])

            # Track coordinate values for x2/y2 lower bound constraint.
            if decode_state == 0:
                x1_token = curr_tokens[0]
            elif decode_state == 1:
                y1_token = curr_tokens[0]

            # ── State transitions ─────────────────────────────────────────────
            if decode_state in [0, 1, 2]:
                decode_state += 1
            elif decode_state == 3:
                decode_state = 9
            elif decode_state == 9:
                if curr_tokens[0] == det_end_token:
                    decode_state = 10
                elif x1_base <= curr_tokens[0] <= x1_base + 999:
                    x1_token = curr_tokens[0]
                    decode_state = 1

            # ── Advance KV caches ─────────────────────────────────────────────
            packed_key_value_indexes, key_values_lens = self._advance_kv(packed_key_value_indexes, key_values_lens)
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1
            if cfg_scale > 1.0 and cfg_key_values_lens is not None:
                cfg_packed_key_value_indexes, cfg_key_values_lens = self._advance_kv(
                    cfg_packed_key_value_indexes, cfg_key_values_lens,
                )
                cfg_packed_query_position_ids = cfg_packed_query_position_ids + 1

            if end_token_id is not None and curr_tokens[0] == end_token_id:
                break
            if decode_state == 10:
                break

        output_device = generated_sequence[0].device
        tokens_stack = torch.stack([i.to(output_device) for i in generated_sequence], dim=0)
        probs_stack = torch.stack(chosen_step_probs, dim=0).to(output_device) if chosen_step_probs else None
        return tokens_stack, probs_stack


    # NOTE: generate_detection_with_space was removed — it was an unused
    # experimental variant.

    @torch.no_grad()
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
        # CFG parameters
        cfg_scale: float = 1.0,
        cfg_past_key_values: Optional[NaiveCache] = None,
        cfg_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_packed_query_position_ids: Optional[torch.LongTensor] = None,
    ):
        """AR cocodet (Pix2seq) detection with a coords+class state machine.

        Format (per box): ``<|x1|><|y1|><|x2|><|y2|><|coco_cls_K|>`` ... ``<|cocodet_end|>``
        State machine: x1 → y1 → x2 → y2 → cls → end/next-x1.

        This is the cocodet-specific decode. It does NOT reuse
        ``generate_detection_coordonly`` (which is coords-only and serves the
        grounding modality). cocodet has NO learnable pos_embed (RoPE-only), so
        unlike the grounding path no positional embedding is added here.
        """
        step = 0
        generated_sequence = []
        chosen_step_probs = []
        curr_tokens = packed_start_tokens

        # 0=x1, 1=y1, 2=x2, 3=y2, 4=cls, 9=next/end, 10=done
        decode_state = 0
        x1_token = y1_token = None
        coord_bases = [x1_base, y1_base, x2_base, y2_base]

        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)
            # NB: no pos_embed for cocodet (RoPE + per-corner/class token ids
            # already disambiguate slot/role).

            packed_key_value_indexes = self._reindex_kv(packed_key_value_indexes, key_values_lens)
            past_key_values, pred_logits = self._ar_forward_step(
                packed_text_embedding, packed_query_position_ids,
                key_values_lens, packed_key_value_indexes, past_key_values,
            )
            cfg_past_key_values, cfg_packed_key_value_indexes, pred_logits = self._cfg_forward_and_merge(
                packed_text_embedding, pred_logits,
                cfg_scale, cfg_past_key_values, cfg_key_values_lens,
                cfg_packed_key_value_indexes, cfg_packed_query_position_ids,
            )

            # Upcast logits to fp32 before the constrained argmax. In bf16, a
            # near-flat coordinate distribution (a weak box) rounds many tokens to
            # the SAME value, and torch.argmax tie-breaks to index 0 = <|x1_000|>
            # (degenerate x1=0 left-edge boxes). The tie outcome depends on
            # sub-bf16 flash-attn/lm_head reduction order, which differs by GPU
            # arch (GH200 vs H100) and run-to-run. fp32 preserves the true logit
            # ordering so the decode is deterministic + arch-independent. (The
            # training CE path already upcasts to fp32 — bagel.py:432.)
            pred_logits = pred_logits.float()

            # ── Constrained decoding: mask logits to the slot's valid tokens ──
            mask = torch.ones_like(pred_logits) * float('-inf')
            if decode_state in (0, 1):
                base = coord_bases[decode_state]
                mask[:, base:base + 1000] = 0
            elif decode_state in (2, 3):
                base = coord_bases[decode_state]
                lower = base + (int(x1_token) - x1_base if decode_state == 2 else int(y1_token) - y1_base)
                mask[:, lower:base + 1000] = 0
            elif decode_state == 4:                       # class token
                mask[:, cls_base:cls_base + n_cls] = 0
            elif decode_state == 9:                       # next box (x1) OR stop
                mask[:, x1_base:x1_base + 1000] = 0
                mask[:, cocodet_end_token] = 0
            pred_logits = pred_logits + mask

            probs = nn.functional.softmax(pred_logits / temperature if do_sample else pred_logits, dim=-1)
            if do_sample:
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)
            chosen_step_probs.append(probs[0, curr_tokens][0])

            if decode_state == 0:
                x1_token = curr_tokens[0]
            elif decode_state == 1:
                y1_token = curr_tokens[0]

            # ── State transitions ───────────────────────────────────────────
            if decode_state in (0, 1, 2, 3):
                decode_state += 1                          # 0→1→2→3→4
            elif decode_state == 4:
                decode_state = 9                           # cls → next/end
            elif decode_state == 9:
                if int(curr_tokens[0]) == cocodet_end_token:
                    decode_state = 10
                elif x1_base <= int(curr_tokens[0]) <= x1_base + 999:
                    x1_token = curr_tokens[0]
                    decode_state = 1                       # x1 consumed → predict y1

            packed_key_value_indexes, key_values_lens = self._advance_kv(packed_key_value_indexes, key_values_lens)
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1
            if cfg_scale > 1.0 and cfg_key_values_lens is not None:
                cfg_packed_key_value_indexes, cfg_key_values_lens = self._advance_kv(
                    cfg_packed_key_value_indexes, cfg_key_values_lens,
                )
                cfg_packed_query_position_ids = cfg_packed_query_position_ids + 1

            if decode_state == 10:
                break

        output_device = generated_sequence[0].device
        tokens_stack = torch.stack([i.to(output_device) for i in generated_sequence], dim=0)
        probs_stack = torch.stack(chosen_step_probs, dim=0).to(output_device) if chosen_step_probs else None
        return tokens_stack, probs_stack

    # for evaluation
    @torch.no_grad()
    def chat(
        self,
        tokenizer,
        new_token_ids,
        image_transform,
        images,
        prompt,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        device = next(self.parameters()).device

        if isinstance(new_token_ids, dict):
            for k, v in new_token_ids.items():
                if torch.is_tensor(v):
                    new_token_ids[k] = v.to(device)
        elif torch.is_tensor(new_token_ids):
            new_token_ids = new_token_ids.to(device)

        # prefill
        past_key_values = NaiveCache(self.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        # add images
        for image in images:
            generation_input, newlens, new_rope = self.prepare_vit_images(
                curr_kvlens=newlens,
                curr_rope=new_rope, 
                images=[image], 
                transforms=image_transform,
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.forward_cache_update_vit(past_key_values, **generation_input)

        # add text
        generation_input, newlens, new_rope = self.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[prompt],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)

        # decode
        generation_input = self.prepare_start_tokens(newlens, new_rope, new_token_ids)
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = self.generate_text(
                past_key_values=past_key_values,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=new_token_ids['eos_token_id'],
                **generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]

        return output



def build_ce_det_loss_weights(det_vocab_dim, total_vocab_dim, min_w=0.10):
    """
    Create a weight tensor for CE where:
    - the first `det_vocab_dim` tokens (coords) use the custom weighting
    - the remaining tokens use 1.0
    """
    bins_per_coord = det_vocab_dim // 4
    left_curve  = [min_w, 0.18, 0.25, 0.32, 0.40, 0.50, 0.60, 0.75, 0.90, 0.98]
    right_curve = left_curve[::-1]

    w = torch.ones(total_vocab_dim, dtype=torch.float32)

    # x1
    for j, val in enumerate(left_curve):
        w[j] = val
    # y1
    offset = bins_per_coord
    for j, val in enumerate(left_curve):
        w[offset + j] = val
    # x2
    offset = 2 * bins_per_coord
    for j, val in enumerate(right_curve):
        w[offset + bins_per_coord - 10 + j] = val
    # y2
    offset = 3 * bins_per_coord
    for j, val in enumerate(right_curve):
        w[offset + bins_per_coord - 10 + j] = val

    # normalize only within det slice
    det_slice = w[:det_vocab_dim]
    w[:det_vocab_dim] *= 1.0 / det_slice.mean()

    return w
