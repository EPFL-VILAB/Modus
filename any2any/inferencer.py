# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
from copy import deepcopy
from typing import List, Dict, Optional, Union, Any

from PIL import Image
import torch
import numpy as np

from data.data_utils import pil_img2rgb
from modeling.bagel.qwen2_navit import NaiveCache

from data.dataset_info import denormalize_latents_by_modality
from data.interleave_datasets.any2any_dataset import get_det_image


VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''


# Number of times to repeat the modality name in instruction prompts.
_INSTRUCTION_REPEAT = 10


class InterleaveInferencer:
    def __init__(
        self,
        model,
        vae_model,
        tokenizer,
        vae_transform,
        vit_transform,
        new_token_ids,
        dino_tokenizer=None,
        dinolocal_tokenizer=None,
        clip_tokenizer=None,
        imagebind_tokenizer=None,
        imagebindlocal_tokenizer=None,
        modality_registry=None,
    ):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        self.dino_tokenizer = dino_tokenizer
        self.dinolocal_tokenizer = dinolocal_tokenizer
        self.clip_tokenizer = clip_tokenizer
        self.imagebind_tokenizer = imagebind_tokenizer
        self.imagebindlocal_tokenizer = imagebindlocal_tokenizer
        # Optional registry to avoid hard-coded token base ids.
        self.modality_registry = modality_registry
        self._dinov2_model = None
        self._clip_model = None
        self._clip_transform = None
        self._imagebind_model = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_instruction(modality_name: str, repeat: int = _INSTRUCTION_REPEAT) -> str:
        """Build the target-modality instruction string.

        Most modalities (depth/normal/canny/rgb/caption/dino*/clip/...) use the
        generic ``[start <name> <name> ...]`` format (matching
        ``any2any_dataset.py:279``).

        ``seg`` is the exception: training uses
        ``f"start segment the mask of {category}"`` and a single-category
        binary mask as target (``any2any_dataset.py:_add_seg``). We replicate
        that here; the category is read from ``HUNYUAN_SEG_CATEGORY``
        (default ``"wall"``).
        """
        if modality_name == "seg":
            category = os.environ.get("HUNYUAN_SEG_CATEGORY", "wall")
            return f"start segment the mask of {category}"
        return "[start " + " ".join([modality_name] * repeat) + "]"

    def _resolve_decode_method(self, target_name: str) -> str:
        """Determine the decode method for *target_name* using the registry (or heuristics)."""
        if self.modality_registry is not None:
            try:
                return self.modality_registry.resolve_decode_method(target_name)
            except KeyError:
                pass
        # Fallback heuristics (backward compat when no registry).
        if target_name == "cocodet":
            return "cocodet"
        if "det" in target_name:
            return "detection"
        if target_name == "dinolocal":
            return "dinolocal"
        if "dino" in target_name:
            return "dino"
        # Treat everything else as image or text based on whether understanding_output is set.
        return "auto"

    def _resolve_cfg_uncond(self, target_name: str) -> str:
        """Determine CFG-uncond context type for *target_name*."""
        if self.modality_registry is not None:
            try:
                return self.modality_registry.resolve_cfg_uncond(target_name)
            except KeyError:
                pass
        dm = self._resolve_decode_method(target_name)
        if dm == "image":
            return "both"
        if dm == "detection":
            return "text"
        if dm in ("dino", "dinolocal", "clip", "imagebind", "imagebindlocal"):
            return "img"
        return "none"

    def _should_add_instruction(self, target_name: str) -> bool:
        """Whether to add ``[start <name> ...]`` instruction before generating *target_name*.

        Default heuristic:
          - Detection uses ``<start_of_det>`` token instead of a text instruction.
          - Plain VQA text answer (``target='text'``) uses no instruction.
          - Every other modality (caption, dino, depth, normal, image, ...)
            gets ``[start <name> × 10]`` injected before generation.
        """
        if self.modality_registry is not None:
            try:
                spec = self.modality_registry.get(target_name)
                return spec.inference_add_instruction
            except (KeyError, AttributeError):
                pass
        if target_name == "text":
            return False
        if "det" in target_name:
            return False
        return True

    def _get_max_tokens(self, target_name: str, fallback: int) -> int:
        """Return max AR tokens for *target_name*, using registry or *fallback*."""
        if self.modality_registry is not None:
            try:
                spec = self.modality_registry.get(target_name)
                if spec.inference_max_tokens is not None:
                    return spec.inference_max_tokens
            except (KeyError, AttributeError):
                pass
        return fallback

    def _is_understanding_target(self, target_name: str) -> bool:
        """Return True if *target_name* produces understanding (AR) output, not image."""
        dm = self._resolve_decode_method(target_name)
        return dm in ("detection", "cocodet", "dino", "dinolocal", "clip", "imagebind", "imagebindlocal", "text")

    def _normalize_target_modalities(self, modality_type_dict):
        if modality_type_dict is None:
            return []
        t = modality_type_dict.get("target")
        if t is None:
            return []
        if isinstance(t, list):
            return [str(x) for x in t]
        return [str(t)]

    def _inference_conditioning_flags(self, *, understanding_output: bool, modality_type_dict):
        """
        Decide whether to encode input images with VAE and/or ViT for conditioning.

        Rule: only for VQA (target='text') do we use ViT-only.  For every other
        target — caption / det / dino / image / depth / normal / seg / canny /
        rgb — use VAE+ViT.  This matches the per-task training distribution:
        text-answer training saw ViT-only image inputs; everything else saw
        VAE+ViT.

        The ``understanding_output`` parameter is kept for backward compatibility
        but is no longer used in the rule itself — the decision is target-only.
        """
        del understanding_output  # unused, target-only decision
        force_no_vit = os.environ.get("BAGEL_INFER_FORCE_NO_VIT_COND", "0") == "1"

        targets = self._normalize_target_modalities(modality_type_dict) or []
        is_vqa = "text" in targets
        need_vae = not is_vqa
        need_vit = (False if force_no_vit else True)

        # Compatibility fallback is opt-in. By default, fail loudly if the
        # wrapper cannot satisfy the conditioning path requested by config.
        allow_conditioning_fallback = os.environ.get(
            "INTERLEAVE_ALLOW_CONDITIONING_FALLBACK", "0"
        ) == "1"
        if need_vae and not (
            hasattr(self.model, "prepare_vae_images")
            and hasattr(self.model, "forward_cache_update_vae")
        ):
            message = (
                "[inferencer] VAE image conditioning requested by modality config, "
                "but the model wrapper does not implement prepare_vae_images / "
                "forward_cache_update_vae."
            )
            if allow_conditioning_fallback:
                print(message + " Falling back to non-VAE image conditioning.")
                need_vae = False
            else:
                raise NotImplementedError(
                    message + " Set INTERLEAVE_ALLOW_CONDITIONING_FALLBACK=1 to force fallback."
                )

        if need_vit and not (
            hasattr(self.model, "prepare_vit_images")
            and hasattr(self.model, "forward_cache_update_vit")
        ):
            message = (
                "[inferencer] ViT image conditioning requested by modality config, "
                "but the model wrapper does not implement prepare_vit_images / "
                "forward_cache_update_vit."
            )
            if allow_conditioning_fallback:
                print(message + " Falling back to non-ViT image conditioning.")
                need_vit = False
            else:
                raise NotImplementedError(
                    message + " Set INTERLEAVE_ALLOW_CONDITIONING_FALLBACK=1 to force fallback."
                )
        return need_vae, need_vit

    def _get_code_base(self, modality_name: str, fallback: int) -> int:
        """Resolve code-token base from registry for a modality."""
        code_base = fallback
        if self.modality_registry is not None:
            try:
                rng = self.modality_registry.code_token_range(modality_name)
                if rng is not None:
                    code_base = int(rng[0])
            except Exception:
                pass
        return code_base

    @staticmethod
    def _serialize_cocodet_boxes(boxes) -> str:
        """Serialize gen_cocodet boxes [{bbox:[x1,y1,x2,y2] norm, label:int}] into the
        cocodet token string (matches any2any_dataset._add_cocodet) for re-injection."""
        inst = sorted(boxes, key=lambda b: b["bbox"][0] ** 2 + b["bbox"][1] ** 2)

        def q(v):
            return min(max(int(round(float(v) * 1000)), 0), 999)

        seq = ""
        for b in inst:
            bb = b.get("bbox")
            lab = b.get("label")
            if bb is None or len(bb) != 4 or lab is None:
                continue
            lab = int(lab)
            if lab < 0 or lab > 90:
                continue
            seq += (f"<|x1_{q(bb[0]):03d}|><|y1_{q(bb[1]):03d}|>"
                    f"<|x2_{q(bb[2]):03d}|><|y2_{q(bb[3]):03d}|><|coco_cls_{lab:02d}|>")
        return seq

    def _ensure_dinov2_model(self, device):
        if self._dinov2_model is None:
            self._dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval().to(device)
        return self._dinov2_model

    @torch.no_grad()
    def _extract_dino_global_feature(self, image: Image.Image, device):
        image_dino = image.resize((224, 224), Image.Resampling.LANCZOS)
        image_tensor = torch.from_numpy(np.array(image_dino)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=image_tensor.dtype).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        image_tensor = image_tensor.unsqueeze(0).to(device)
        dinov2 = self._ensure_dinov2_model(device)
        outputs = dinov2(image_tensor, is_training=True)
        if "x_norm_clstoken" not in outputs:
            raise RuntimeError("DINOv2 output missing 'x_norm_clstoken' for GT dino conditioning.")
        return outputs["x_norm_clstoken"]

    @torch.no_grad()
    def _extract_dino_local_feature_map(self, image: Image.Image, device):
        image_dino = image.resize((448, 448), Image.Resampling.LANCZOS)
        image_tensor = torch.from_numpy(np.array(image_dino)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=image_tensor.dtype).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        image_tensor = image_tensor.unsqueeze(0).to(device)
        dinov2 = self._ensure_dinov2_model(device)
        outputs = dinov2(image_tensor, is_training=True)
        if "x_norm_patchtokens" not in outputs:
            raise RuntimeError("DINOv2 output missing 'x_norm_patchtokens' for GT dinolocal conditioning.")
        gt_feat = outputs["x_norm_patchtokens"]  # [B, 1024, 768]
        return gt_feat.transpose(1, 2).reshape(1, 768, 32, 32)

    @torch.no_grad()
    def update_context_dino_tokens_from_image(self, image, gen_context):
        """Condition with GT DINO global tokenizer codes instead of image VAE/ViT."""
        if self.dino_tokenizer is None:
            raise RuntimeError("dino_tokenizer is required for use_gt_dino_condition=true.")
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        device = next(self.model.parameters()).device

        image_rgb = pil_img2rgb(image)
        gt_global_feat = self._extract_dino_global_feature(image_rgb, device=device)  # [1, 768]
        encoded = self.dino_tokenizer.encode(gt_global_feat.float().unsqueeze(2).unsqueeze(3))
        token_ids = encoded[-1] if isinstance(encoded, (tuple, list)) else encoded
        token_ids = token_ids.reshape(-1).detach().long().to(device)

        code_base = self._get_code_base("dino", fallback=155773)
        code_token_ids = token_ids + int(code_base)

        start_key = "start_of_dino"
        end_key = "end_of_dino"
        if start_key not in self.new_token_ids or end_key not in self.new_token_ids:
            raise RuntimeError("Missing start/end dino tokens in new_token_ids.")
        packed_text_ids = torch.cat([
            torch.tensor([self.new_token_ids[start_key]], dtype=torch.long, device=device),
            code_token_ids,
            torch.tensor([self.new_token_ids[end_key]], dtype=torch.long, device=device),
        ], dim=0)

        curr_kvlen = int(kv_lens[0])
        curr_position_id = int(ropes[0])
        seq_len = int(packed_text_ids.numel())
        generation_input = {
            "text_token_lens": torch.tensor([seq_len], dtype=torch.int, device=device),
            "packed_text_ids": packed_text_ids,
            "packed_text_position_ids": torch.arange(curr_position_id, curr_position_id + seq_len, dtype=torch.long, device=device),
            "packed_text_indexes": torch.arange(curr_kvlen, curr_kvlen + seq_len, dtype=torch.long, device=device),
            "packed_key_value_indexes": torch.arange(0, curr_kvlen, dtype=torch.long, device=device),
            "key_values_lens": torch.tensor([curr_kvlen], dtype=torch.int, device=device),
        }

        past_key_values = self.model.forward_cache_update_text(
            past_key_values,
            pos_embed_key="dino",
            **generation_input,
        )
        gen_context['kv_lens'] = [curr_kvlen + seq_len]
        gen_context['ropes'] = [curr_position_id + seq_len]
        gen_context['past_key_values'] = past_key_values
        return gen_context

    @torch.no_grad()
    def update_context_dinolocal_tokens_from_image(self, image, gen_context):
        """Condition with GT DINO-local tokenizer codes instead of image VAE/ViT."""
        if self.dinolocal_tokenizer is None:
            raise RuntimeError("dinolocal_tokenizer is required for use_gt_dinolocal_condition=true.")
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        device = next(self.model.parameters()).device

        image_rgb = pil_img2rgb(image)
        gt_feat_map = self._extract_dino_local_feature_map(image_rgb, device=device)  # [1, 768, 32, 32]
        encoded = self.dinolocal_tokenizer.encode(gt_feat_map.float())
        token_ids = encoded[-1] if isinstance(encoded, (tuple, list)) else encoded
        token_ids = token_ids.reshape(-1).detach().long().to(device)

        code_base = self._get_code_base("dinolocal", fallback=163967)
        code_token_ids = token_ids + int(code_base)

        start_key = "start_of_dinolocal"
        end_key = "end_of_dinolocal"
        if start_key not in self.new_token_ids or end_key not in self.new_token_ids:
            raise RuntimeError("Missing start/end dinolocal tokens in new_token_ids.")
        packed_text_ids = torch.cat([
            torch.tensor([self.new_token_ids[start_key]], dtype=torch.long, device=device),
            code_token_ids,
            torch.tensor([self.new_token_ids[end_key]], dtype=torch.long, device=device),
        ], dim=0)

        curr_kvlen = int(kv_lens[0])
        curr_position_id = int(ropes[0])
        seq_len = int(packed_text_ids.numel())
        generation_input = {
            "text_token_lens": torch.tensor([seq_len], dtype=torch.int, device=device),
            "packed_text_ids": packed_text_ids,
            "packed_text_position_ids": torch.arange(curr_position_id, curr_position_id + seq_len, dtype=torch.long, device=device),
            "packed_text_indexes": torch.arange(curr_kvlen, curr_kvlen + seq_len, dtype=torch.long, device=device),
            "packed_key_value_indexes": torch.arange(0, curr_kvlen, dtype=torch.long, device=device),
            "key_values_lens": torch.tensor([curr_kvlen], dtype=torch.int, device=device),
        }

        past_key_values = self.model.forward_cache_update_text(
            past_key_values,
            pos_embed_key="dinolocal",
            **generation_input,
        )
        gen_context['kv_lens'] = [curr_kvlen + seq_len]
        gen_context['ropes'] = [curr_position_id + seq_len]
        gen_context['past_key_values'] = past_key_values
        return gen_context

    # ── CLIP / ImageBind online feature extractors (mirror DINO path) ──────────
    # Feature recipes (verified against 4M tokenizer configs + ml-4m source):
    #   clip           : OpenAI CLIP ViT-B/16 @448, ln_post(x)[:,1:] @ proj -> [1,512,28,28]
    #   imagebind      : ImageBind ViT-H/14 @224, vision-trunk CLS (pre-head)  -> [1,1280,1,1]
    #   imagebindlocal : ImageBind ViT-H/14 @448, vision-trunk patches 32x32   -> [1,1280,32,32]
    _FEAT_MEAN = (0.48145466, 0.4578275, 0.40821073)   # CLIP/ImageBind image norm
    _FEAT_STD = (0.26862954, 0.26130258, 0.27577711)

    def _ensure_clip_model(self, device):
        if self._clip_model is None:
            from fourm.utils.clip import clip as _clipmod
            self._clip_model, _ = _clipmod.load("ViT-B/16", device=device)
            self._clip_model = self._clip_model.eval()
            self._clip_transform = _clipmod._transform(448)
        return self._clip_model

    def _ensure_imagebind_model(self, device):
        if self._imagebind_model is None:
            import sys, types
            # ImageBind pulls video-only deps at import; stub them (image path unaffected).
            for m in ['pytorchvideo', 'pytorchvideo.transforms', 'pytorchvideo.data',
                      'pytorchvideo.data.clip_sampling', 'pytorchvideo.data.encoded_video']:
                sys.modules.setdefault(m, types.ModuleType(m))
            sys.modules['pytorchvideo.data.clip_sampling'].ConstantClipsPerVideoSampler = object
            sys.modules['pytorchvideo.transforms'].ShortSideScale = object
            sys.modules['pytorchvideo.data.encoded_video'].EncodedVideo = object
            from imagebind.models import imagebind_model
            ckpt_path = os.environ.get("MODUS_IMAGEBIND_CKPT")
            if ckpt_path:
                ib = imagebind_model.imagebind_huge(pretrained=False)
                ib.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            else:
                ib = imagebind_model.imagebind_huge(pretrained=True)
            self._imagebind_model = ib.eval().to(device)
        return self._imagebind_model

    @torch.no_grad()
    def _extract_clip_local_feature_map(self, image: Image.Image, device):
        m = self._ensure_clip_model(device)
        x = self._clip_transform(image).unsqueeze(0).to(device)
        feat = m.visual(x.type(m.dtype), return_final_tokens_no_cls=True)  # [1, 784, 512]
        return feat.float().transpose(1, 2).reshape(1, 512, 28, 28)

    @torch.no_grad()
    def _imagebind_trunk_tokens(self, image: Image.Image, device, res):
        from torchvision import transforms
        ib = self._ensure_imagebind_model(device)
        tf = transforms.Compose([
            transforms.Resize(res, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(res),
            transforms.ToTensor(),
            transforms.Normalize(self._FEAT_MEAN, self._FEAT_STD),
        ])
        x = tf(image).unsqueeze(0).to(device)
        pre = ib.modality_preprocessors['vision']
        trunk = ib.modality_trunks['vision']
        out = pre(vision=x)
        return trunk(**out['trunk'])  # [1, N, 1280] (token 0 = CLS)

    @torch.no_grad()
    def _extract_imagebind_global_feature(self, image: Image.Image, device):
        tok_out = self._imagebind_trunk_tokens(image, device, 224)
        return tok_out[:, 0, :].reshape(1, 1280, 1, 1)  # CLS

    @torch.no_grad()
    def _extract_imagebindlocal_feature_map(self, image: Image.Image, device):
        tok_out = self._imagebind_trunk_tokens(image, device, 448)
        patches = tok_out[:, 1:, :]  # drop CLS -> [1, 1024, 1280]
        return patches[0].transpose(0, 1).reshape(1, 1280, 32, 32)

    @torch.no_grad()
    def _append_codebook_condition(self, token_ids, modality, gen_context, device):
        """Shared tail: shift codes by code_base, wrap start/end, push into KV cache.

        Mirrors the dino/dinolocal helpers; used for clip/imagebind/imagebindlocal.
        """
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        code_base = self._get_code_base(modality, fallback=0)
        code_token_ids = token_ids.reshape(-1).detach().long().to(device) + int(code_base)

        start_key, end_key = f"start_of_{modality}", f"end_of_{modality}"
        if start_key not in self.new_token_ids or end_key not in self.new_token_ids:
            raise RuntimeError(f"Missing start/end {modality} tokens in new_token_ids.")
        packed_text_ids = torch.cat([
            torch.tensor([self.new_token_ids[start_key]], dtype=torch.long, device=device),
            code_token_ids,
            torch.tensor([self.new_token_ids[end_key]], dtype=torch.long, device=device),
        ], dim=0)

        curr_kvlen = int(kv_lens[0])
        curr_position_id = int(ropes[0])
        seq_len = int(packed_text_ids.numel())
        generation_input = {
            "text_token_lens": torch.tensor([seq_len], dtype=torch.int, device=device),
            "packed_text_ids": packed_text_ids,
            "packed_text_position_ids": torch.arange(curr_position_id, curr_position_id + seq_len, dtype=torch.long, device=device),
            "packed_text_indexes": torch.arange(curr_kvlen, curr_kvlen + seq_len, dtype=torch.long, device=device),
            "packed_key_value_indexes": torch.arange(0, curr_kvlen, dtype=torch.long, device=device),
            "key_values_lens": torch.tensor([curr_kvlen], dtype=torch.int, device=device),
        }
        past_key_values = self.model.forward_cache_update_text(
            past_key_values, pos_embed_key=modality, **generation_input,
        )
        gen_context['kv_lens'] = [curr_kvlen + seq_len]
        gen_context['ropes'] = [curr_position_id + seq_len]
        gen_context['past_key_values'] = past_key_values
        return gen_context

    @torch.no_grad()
    def update_context_clip_tokens_from_image(self, image, gen_context):
        if self.clip_tokenizer is None:
            raise RuntimeError("clip_tokenizer is required for use_gt_clip_condition=true.")
        device = next(self.model.parameters()).device
        feat = self._extract_clip_local_feature_map(pil_img2rgb(image), device=device)
        encoded = self.clip_tokenizer.encode(feat.float())
        token_ids = encoded[-1] if isinstance(encoded, (tuple, list)) else encoded
        return self._append_codebook_condition(token_ids, "clip", gen_context, device)

    @torch.no_grad()
    def update_context_imagebind_tokens_from_image(self, image, gen_context):
        if self.imagebind_tokenizer is None:
            raise RuntimeError("imagebind_tokenizer is required for use_gt_imagebind_condition=true.")
        device = next(self.model.parameters()).device
        feat = self._extract_imagebind_global_feature(pil_img2rgb(image), device=device)
        encoded = self.imagebind_tokenizer.encode(feat.float())
        token_ids = encoded[-1] if isinstance(encoded, (tuple, list)) else encoded
        return self._append_codebook_condition(token_ids, "imagebind", gen_context, device)

    @torch.no_grad()
    def update_context_imagebindlocal_tokens_from_image(self, image, gen_context):
        if self.imagebindlocal_tokenizer is None:
            raise RuntimeError("imagebindlocal_tokenizer is required for use_gt_imagebindlocal_condition=true.")
        device = next(self.model.parameters()).device
        feat = self._extract_imagebindlocal_feature_map(pil_img2rgb(image), device=device)
        encoded = self.imagebindlocal_tokenizer.encode(feat.float())
        token_ids = encoded[-1] if isinstance(encoded, (tuple, list)) else encoded
        return self._append_codebook_condition(token_ids, "imagebindlocal", gen_context, device)

    # ── Context management ────────────────────────────────────────────────────

    def init_gen_context(self):
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
            'cond_rope_segments': [],
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context, modality_type=None):
        # used for interleave data, currently only support 1 data inference, 

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
            modality_type=modality_type,
        )

        device = next(self.model.parameters()).device
        generation_input = move_tensors_to_device(generation_input, device)
        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True, modality_type=None, do_modality_norm=False):
        # used for interleave data, currently only support 1 data inference, 

        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']
        device = next(self.model.parameters()).device

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vae_transform,
                new_token_ids=self.new_token_ids,
                modality_type=modality_type,
            )
            # Extract the 2D-RoPE segment (if the wrapper provided one — hunyuan
            # path only). Bagel/janus paths leave it absent. We track these to
            # rebuild the full cos/sin table at target-image gen time so
            # cross-image attention has consistent RoPE rotation.
            cond_rope_segment = generation_input.pop("__cond_rope_segment__", None)
            generation_input = move_tensors_to_device(generation_input, device)
            generation_input['modality_type'] = [modality_type]
            generation_input['do_modality_norm'] = do_modality_norm
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
            if cond_rope_segment is not None:
                gen_context.setdefault('cond_rope_segments', []).append(cond_rope_segment)
        
        if vit:
            ## update vit
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
                modality_type=modality_type,
            )
            # ViT path: same 2D-RoPE replay requirement as VAE — without this,
            # diffusion target Q's see ViT cond K with mismatched rotation.
            vit_cond_rope_segment = generation_input.pop("__cond_rope_segment__", None)
            generation_input = move_tensors_to_device(generation_input, device)
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)
            if vit_cond_rope_segment is not None:
                gen_context.setdefault('cond_rope_segments', []).append(vit_cond_rope_segment)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    # ── Low-level generators ──────────────────────────────────────────────────

    @torch.no_grad()
    def gen_image(
        self, 
        image_shape, 
        gen_context, 
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,

        cfg_text_precontext=None, 
        cfg_img_precontext=None, 
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        
        num_timesteps=50, 
        timestep_shift=3.0,
        modality_type_dict=None,
        do_modality_norm=False,
    ):
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=[image_shape], 
            new_token_ids=self.new_token_ids,
            modality_type=modality_type_dict['target'],
        ) 
        # Ensure all tensors in generation_input are on the model's device
        device = next(self.model.parameters()).device
        generation_input = move_tensors_to_device(generation_input, device)

        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )
        generation_input_cfg_text = move_tensors_to_device(generation_input_cfg_text, device)

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )
        generation_input_cfg_img = move_tensors_to_device(generation_input_cfg_img, device)

        _tgt = modality_type_dict.get('target') if modality_type_dict else None
        _target_modality = (_tgt if isinstance(_tgt, str) else _tgt[0]) if _tgt else None
        # Forward the per-context cond-image RoPE segments so the wrapper can
        # rebuild a cos/sin table consistent with how K's were rotated at cache
        # update time. (Bagel ignores these via **_unused.)
        cond_rope_segments = gen_context.get('cond_rope_segments', [])
        cfg_text_cond_rope_segments = cfg_text_precontext.get('cond_rope_segments', [])
        cfg_img_cond_rope_segments = cfg_img_precontext.get('cond_rope_segments', [])
        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            target_modality_name=_target_modality,
            cond_rope_segments=cond_rope_segments,
            cfg_text_cond_rope_segments=cfg_text_cond_rope_segments,
            cfg_img_cond_rope_segments=cfg_img_cond_rope_segments,
        )

        # Get modality type for normalization
        modality_type = modality_type_dict.get('target') if modality_type_dict else None
        
        image = self.decode_image(
            unpacked_latent[0], 
            image_shape, 
            do_modality_norm=do_modality_norm,
            modality_type=modality_type
        )
        return image

        
    def decode_image(self, latent, image_shape, do_modality_norm=False, modality_type=None):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        
        # Apply modality normalization if enabled
        if do_modality_norm and modality_type is not None:
            device = next(self.model.parameters()).device
            latent = denormalize_latents_by_modality(latent, modality_type, device)
        
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        generation_input = move_tensors_to_device(generation_input, self.model.device)
        
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        return output
        
    @torch.no_grad()
    def gen_detection(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0, modality_type_dict=None, cfg_scale: float = 1.0, cfg_text_context=None):
        
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids, modality_type_dict['target'][0])
        generation_input = move_tensors_to_device(generation_input, self.model.device)

        # Prepare CFG context (unconditional - text-only context)
        cfg_past_key_values = None
        cfg_generation_input = None
        if cfg_scale > 1.0 and cfg_text_context is not None:
            cfg_gen_context = deepcopy(cfg_text_context)
            cfg_past_key_values = cfg_gen_context['past_key_values']
            cfg_kv_lens = cfg_gen_context['kv_lens']
            cfg_ropes = cfg_gen_context['ropes']
            cfg_generation_input = self.model.prepare_start_tokens(cfg_kv_lens, cfg_ropes, self.new_token_ids, modality_type_dict['target'][0])
            cfg_generation_input = move_tensors_to_device(cfg_generation_input, self.model.device)

        tokens_stack, step_probs = self.model.generate_detection_coordonly(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['end_of_det'],
            cfg_scale=cfg_scale,
            cfg_past_key_values=cfg_past_key_values,
            cfg_key_values_lens=cfg_generation_input['key_values_lens'] if cfg_generation_input else None,
            cfg_packed_key_value_indexes=cfg_generation_input['packed_key_value_indexes'] if cfg_generation_input else None,
            cfg_packed_query_position_ids=cfg_generation_input['packed_query_position_ids'] if cfg_generation_input else None,
            **generation_input,
        )
        output = self.tokenizer.decode(tokens_stack[:,0])
        output = output.split('<|det_start|>')[1]

        # Capture per-coordinate probabilities and an aggregate confidence
        try:
            if step_probs is not None and step_probs.numel() > 0:
                # Use first 4 decode steps for x1,y1,x2,y2
                coord_probs_tensor = step_probs[:4].detach().float().cpu()
                coord_probs = coord_probs_tensor.tolist()
                self.last_det_coord_probs = coord_probs
                # Conservative aggregate: min of the four
                self.last_det_confidence = float(min(coord_probs)) if len(coord_probs) > 0 else None
            else:
                self.last_det_coord_probs = None
                self.last_det_confidence = None
        except Exception:
            self.last_det_coord_probs = None
            self.last_det_confidence = None

        return output

    @torch.no_grad()
    def gen_cocodet(self, gen_context, max_length: int = 1000, do_sample: bool = False, temperature: float = 1.0, modality_type_dict=None, cfg_scale: float = 1.0, cfg_text_context=None):
        """AR cocodet (Pix2seq) detection: coords+class boxes ending on cocodet_end.

        Mirrors ``gen_detection`` but calls ``model.generate_cocodet`` and parses
        the returned token-id sequence directly (groups of 5 ids per box).
        Returns a list of dicts ``[{"bbox":[x1,y1,x2,y2] (norm 0..1), "label":int, "score":float}]``.
        """
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        target_key = modality_type_dict['target'][0] if modality_type_dict else "cocodet"
        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids, target_key)
        generation_input = move_tensors_to_device(generation_input, self.model.device)

        # Prepare CFG context (unconditional - text-only context).
        cfg_past_key_values = None
        cfg_generation_input = None
        if cfg_scale > 1.0 and cfg_text_context is not None:
            cfg_gen_context = deepcopy(cfg_text_context)
            cfg_past_key_values = cfg_gen_context['past_key_values']
            cfg_kv_lens = cfg_gen_context['kv_lens']
            cfg_ropes = cfg_gen_context['ropes']
            cfg_generation_input = self.model.prepare_start_tokens(cfg_kv_lens, cfg_ropes, self.new_token_ids, target_key)
            cfg_generation_input = move_tensors_to_device(cfg_generation_input, self.model.device)

        # cocodet vocab layout from the tokenizer.
        x1_base = self.tokenizer.convert_tokens_to_ids("<|x1_000|>")
        y1_base = self.tokenizer.convert_tokens_to_ids("<|y1_000|>")
        x2_base = self.tokenizer.convert_tokens_to_ids("<|x2_000|>")
        y2_base = self.tokenizer.convert_tokens_to_ids("<|y2_000|>")
        cls_base = self.tokenizer.convert_tokens_to_ids("<|coco_cls_00|>")
        cocodet_end = self.new_token_ids["end_of_cocodet"]

        tokens_stack, step_probs = self.model.generate_cocodet(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            x1_base=x1_base,
            y1_base=y1_base,
            x2_base=x2_base,
            y2_base=y2_base,
            cls_base=cls_base,
            n_cls=91,
            cocodet_end_token=cocodet_end,
            cfg_scale=cfg_scale,
            cfg_past_key_values=cfg_past_key_values,
            cfg_key_values_lens=cfg_generation_input['key_values_lens'] if cfg_generation_input else None,
            cfg_packed_key_value_indexes=cfg_generation_input['packed_key_value_indexes'] if cfg_generation_input else None,
            cfg_packed_query_position_ids=cfg_generation_input['packed_query_position_ids'] if cfg_generation_input else None,
            **generation_input,
        )

        # Parse token ids directly into boxes (groups of 5: x1,y1,x2,y2,cls).
        ids = tokens_stack[:, 0].tolist()
        probs = step_probs.detach().float().cpu().tolist() if step_probs is not None else None

        boxes = []
        # generate_cocodet records the start token as tokens_stack[0]; the first
        # predicted coord (x1) is tokens_stack[1].  step_probs has one entry per
        # *predicted* token, so step_probs[k] is the prob of ids[k+1].  Drop the
        # start token so ids[k] aligns with probs[k], then walk 5-id groups.
        ids = ids[1:]
        i = 0
        n = len(ids)
        while i < n:
            tid = ids[i]
            if tid == cocodet_end:
                break
            # A box needs 5 ids: x1,y1,x2,y2,cls.
            if i + 4 >= n:
                break
            x1_id, y1_id, x2_id, y2_id, cls_id = ids[i:i + 5]
            # Stop if any slot ran into the end token (malformed tail).
            if cocodet_end in (x1_id, y1_id, x2_id, y2_id, cls_id):
                break
            bbox = [
                (x1_id - x1_base) / 1000.0,
                (y1_id - y1_base) / 1000.0,
                (x2_id - x2_base) / 1000.0,
                (y2_id - y2_base) / 1000.0,
            ]
            label = int(cls_id - cls_base)
            if probs is not None and i + 4 < len(probs):
                box_probs = probs[i:i + 5]
                score = float(min(box_probs)) if box_probs else 1.0
            else:
                score = 1.0
            boxes.append({"bbox": bbox, "label": label, "score": score})
            i += 5

        return boxes

    @torch.no_grad()
    def gen_dino_family(
        self,
        target_name: str,
        gen_context,
        max_length: int = 500,
        do_sample: bool = True,
        temperature: float = 1.0,
        modality_type_dict=None,
        dino_pca=None,
        top_k=0,
        top_p=1.0,
        cfg_scale: float = 1.0,
        cfg_img_context=None,
        return_string=False,
    ):
        """Unified codebook generation for all codebook modalities.

        Handles dino, dinolocal, clip, imagebind, imagebindlocal.
        The *target_name* drives all behavioural differences (base token,
        end token, reshape, external tokenizer).
        """
        # Codebook decode info: is_local, spatial_shape, tokenizer attribute name.
        _CODEBOOK_DECODE_INFO = {
            "dino":           {"is_local": False, "shape": None,     "tokenizer_attr": "dino_tokenizer"},
            "dinolocal":      {"is_local": True,  "shape": (32, 32), "tokenizer_attr": "dinolocal_tokenizer"},
            "clip":           {"is_local": True,  "shape": (28, 28), "tokenizer_attr": "clip_tokenizer"},
            "imagebind":      {"is_local": False, "shape": None,     "tokenizer_attr": "imagebind_tokenizer"},
            "imagebindlocal": {"is_local": True,  "shape": (32, 32), "tokenizer_attr": "imagebindlocal_tokenizer"},
        }

        decode_info = _CODEBOOK_DECODE_INFO.get(target_name, _CODEBOOK_DECODE_INFO.get("dino"))
        is_local = decode_info["is_local"]

        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        target_key = modality_type_dict['target'][0] if modality_type_dict else target_name
        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids, target_key)
        generation_input = move_tensors_to_device(generation_input, self.model.device)

        # Prepare CFG context (unconditional - without image)
        cfg_past_key_values = None
        cfg_generation_input = None
        if cfg_scale > 1.0 and cfg_img_context is not None:
            cfg_gen_context = deepcopy(cfg_img_context)
            cfg_past_key_values = cfg_gen_context['past_key_values']
            cfg_kv_lens = cfg_gen_context['kv_lens']
            cfg_ropes = cfg_gen_context['ropes']
            cfg_generation_input = self.model.prepare_start_tokens(cfg_kv_lens, cfg_ropes, self.new_token_ids, target_key)
            cfg_generation_input = move_tensors_to_device(cfg_generation_input, self.model.device)

        # Resolve base token id and end token from registry (with fallbacks).
        fallback_base = 163967 if is_local else 155773
        code_base = fallback_base
        if self.modality_registry is not None:
            try:
                rng = self.modality_registry.code_token_range(target_name)
                if rng is not None:
                    code_base = rng[0]
            except Exception:
                pass

        end_key = f'end_of_{target_name}'
        end_token_id = self.new_token_ids[end_key]

        # Pass codebook range and pos_embed key for ALL modalities.
        extra_dino_kwargs: Dict[str, Any] = {}
        extra_dino_kwargs.update(
            dino_base=code_base,
            dino_length=8192,
            pos_embed_key=target_name,
        )

        unpacked_latent = self.model.generate_dino(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=end_token_id,
            top_k=top_k,
            top_p=top_p,
            cfg_scale=cfg_scale,
            cfg_past_key_values=cfg_past_key_values,
            cfg_key_values_lens=cfg_generation_input['key_values_lens'] if cfg_generation_input else None,
            cfg_packed_key_value_indexes=cfg_generation_input['packed_key_value_indexes'] if cfg_generation_input else None,
            cfg_packed_query_position_ids=cfg_generation_input['packed_query_position_ids'] if cfg_generation_input else None,
            **extra_dino_kwargs,
            **generation_input,
        )

        # Decode tokens → features via the external VQVAE tokenizer.
        ext_tokenizer = getattr(self, decode_info["tokenizer_attr"], None)
        decode_on_cpu = ext_tokenizer is not None and hasattr(self.model, "hunyuan_model")
        if is_local:
            spatial_shape = list(decode_info["shape"])
            raw_tokens = unpacked_latent[1:, 0].clone()
            code_tokens = raw_tokens.reshape(spatial_shape) - code_base
            if ext_tokenizer is not None:
                if decode_on_cpu:
                    ext_tokenizer = ext_tokenizer.to("cpu")
                    code_tokens = code_tokens.to("cpu")
                output = ext_tokenizer.decode_tokens(code_tokens.unsqueeze(0))
            else:
                output = code_tokens.unsqueeze(0)
        else:
            raw_tokens = unpacked_latent[1:, 0].clone()
            code_tokens = raw_tokens - code_base
            if ext_tokenizer is not None:
                if decode_on_cpu:
                    ext_tokenizer = ext_tokenizer.to("cpu")
                    code_tokens = code_tokens.to("cpu")
                output = ext_tokenizer.decode_tokens(code_tokens.unsqueeze(0).unsqueeze(2).unsqueeze(3))
                output = output.squeeze(3).squeeze(2)
            else:
                output = code_tokens.unsqueeze(0)

        if return_string:
            output = self.tokenizer.decode(unpacked_latent[1:, 0])

        return output

    # ── Condition feeding (shared by all inference paths) ─────────────────────

    def _feed_conditions(
        self,
        input_lists,
        gen_context,
        cfg_text_context,
        cfg_img_context,
        *,
        modality_type_dict,
        understanding_output: bool,
        use_instruction: bool,
        use_condition_instruction: bool,
        use_det_image: bool,
        do_modality_norm: bool,
        image_shapes,
        use_gt_dino_condition: bool = False,
        use_gt_dinolocal_condition: bool = False,
        use_gt_clip_condition: bool = False,
        use_gt_imagebind_condition: bool = False,
        use_gt_imagebindlocal_condition: bool = False,
        cond_feature_mode: str = "both",
    ):
        """Feed all condition inputs (text/image) into KV caches.

        Returns ``(gen_context, cfg_text_context, cfg_img_context, image_shapes)``.
        """
        condition_list = modality_type_dict['condition'] if modality_type_dict is not None else []
        for i, input_term in enumerate(input_lists):
            if i >= len(condition_list):
                # More inputs than declared conditions — skip extras (e.g. empty
                # prompt text for tasks that don't condition on text).
                break

            if use_instruction and use_condition_instruction:
                gen_context = self.update_context_text(
                    f"[start {condition_list[i]}]", gen_context,
                )
                print('condition:', condition_list[i])
                print(f"[start {condition_list[i]}]")

            modality_type = condition_list[i]

            if isinstance(input_term, str):
                cfg_text_context = deepcopy(gen_context)

                if modality_type == 'det':
                    input_term = input_term.split('<sep>')

                    if use_det_image:
                        det_image = get_det_image(input_term[1], image_shapes)
                        gen_context = self.update_context_image(det_image, gen_context, modality_type='rgb')
                        cfg_text_context = deepcopy(gen_context)

                    gen_context = self.update_context_text(input_term[0], gen_context, modality_type='text')
                    cfg_img_context = self.update_context_text(input_term[0], cfg_img_context, modality_type='text')
                    input_term = input_term[1]

                gen_context = self.update_context_text(input_term, gen_context, modality_type=modality_type)
                cfg_img_context = self.update_context_text(input_term, cfg_img_context, modality_type=modality_type)

            elif isinstance(input_term, Image.Image):
                if modality_type == "dino" and use_gt_dino_condition:
                    gen_context = self.update_context_dino_tokens_from_image(input_term, gen_context)
                    cfg_text_context = deepcopy(gen_context)
                elif modality_type == "dinolocal" and use_gt_dinolocal_condition:
                    gen_context = self.update_context_dinolocal_tokens_from_image(input_term, gen_context)
                    cfg_text_context = deepcopy(gen_context)
                elif modality_type == "clip" and use_gt_clip_condition:
                    gen_context = self.update_context_clip_tokens_from_image(input_term, gen_context)
                    cfg_text_context = deepcopy(gen_context)
                elif modality_type == "imagebind" and use_gt_imagebind_condition:
                    gen_context = self.update_context_imagebind_tokens_from_image(input_term, gen_context)
                    cfg_text_context = deepcopy(gen_context)
                elif modality_type == "imagebindlocal" and use_gt_imagebindlocal_condition:
                    gen_context = self.update_context_imagebindlocal_tokens_from_image(input_term, gen_context)
                    cfg_text_context = deepcopy(gen_context)
                elif modality_type == "seg":
                    # Seg-as-condition is trained with a "mask category: {cat}" text
                    # prefix before the mask image (any2any_dataset.py:550). Mirror it
                    # so seg-conditioned generation matches the training distribution;
                    # otherwise the model sees an unlabeled mask. Category from the same
                    # env var as the seg target instruction.
                    cat = os.environ.get("HUNYUAN_SEG_CATEGORY", "person")
                    prefix = f"mask category: {cat}"
                    seg_img = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    need_vae, need_vit = self._inference_conditioning_flags(
                        understanding_output=understanding_output,
                        modality_type_dict=modality_type_dict,
                    )
                    gen_context = self.update_context_text(prefix, gen_context)
                    cfg_img_context = self.update_context_text(prefix, cfg_img_context)
                    gen_context = self.update_context_image(
                        seg_img, gen_context,
                        vae=need_vae, vit=need_vit,
                        modality_type="seg", do_modality_norm=do_modality_norm,
                    )
                    image_shapes = seg_img.size[::-1]
                    cfg_text_context = deepcopy(gen_context)
                else:
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    need_vae, need_vit = self._inference_conditioning_flags(
                        understanding_output=understanding_output,
                        modality_type_dict=modality_type_dict,
                    )
                    # Tab-3 representation-analysis hook: optionally restrict the
                    # image CONDITION feed to ViT-only / VAE-only.  Default "both"
                    # leaves the per-task conditioning flags untouched.
                    if cond_feature_mode == "vit":
                        need_vae = False
                    elif cond_feature_mode == "vae":
                        need_vit = False
                    gen_context = self.update_context_image(
                        input_term, gen_context,
                        vae=need_vae, vit=need_vit,
                        modality_type=modality_type,
                        do_modality_norm=do_modality_norm,
                    )
                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

            else:
                raise ValueError(f"Unsupported input type: {type(input_term)}")

        return gen_context, cfg_text_context, cfg_img_context, image_shapes

    # ── Target generation (shared dispatch) ───────────────────────────────────

    def _generate_for_modality(
        self,
        target_name: str,
        gen_context,
        cfg_text_context,
        cfg_img_context,
        image_shapes,
        *,
        modality_type_dict,
        understanding_output: bool,
        use_instruction: bool,
        use_target_instruction: bool,
        cfg_text_scale: float,
        cfg_img_scale: float,
        cfg_interval,
        timestep_shift: float,
        num_timesteps: int,
        cfg_renorm_min: float,
        cfg_renorm_type: str,
        do_modality_norm: bool,
        do_sample: bool,
        text_temperature: float,
        max_think_token_n: int,
        dino_pca=None,
        top_k: int = 0,
        top_p: float = 1.0,
        return_string: bool = False,
        seg_category: Optional[str] = None,
    ):
        """Generate output for a single target modality.

        Dispatches to the correct low-level generator based on modality kind.
        Handles instruction injection and CFG context selection automatically.

        Returns the generated output (Image, str, or Tensor).
        """
        dm = self._resolve_decode_method(target_name)
        # Determine understanding_output automatically if "auto".
        if dm == "auto":
            dm = "text" if understanding_output else "image"

        add_instr = use_instruction and use_target_instruction and self._should_add_instruction(target_name)

        if dm == "detection":
            # Detection: no instruction text; CFG on text context.
            target_list = modality_type_dict.get('target', [target_name])
            if not isinstance(target_list, list):
                target_list = [target_list]
            max_tok = self._get_max_tokens(target_name, max_think_token_n)

            # gen_detection's CFG path requires a non-empty cfg_text_context
            # (it advances position-ids from the last existing token).  When the
            # only condition is text — e.g. chained_t2i `text -> det -> rgb` —
            # dropping text gives an empty CFG context and `_advance_kv` crashes
            # on `tensor[-1]` of a size-0 tensor.  Disable CFG in that case.
            effective_cfg_scale = cfg_text_scale
            cfg_lens = cfg_text_context.get('kv_lens') if cfg_text_context is not None else None
            if cfg_lens is not None:
                try:
                    total_cfg_tokens = int(sum(
                        cfg_lens.tolist() if hasattr(cfg_lens, "tolist") else cfg_lens
                    ))
                except Exception:
                    total_cfg_tokens = None
                if total_cfg_tokens == 0:
                    effective_cfg_scale = 1.0

            return self.gen_detection(
                gen_context,
                do_sample=do_sample,
                temperature=text_temperature,
                max_length=max_tok,
                modality_type_dict={'target': target_list},
                cfg_scale=effective_cfg_scale,
                cfg_text_context=cfg_text_context,
            )

        elif dm == "cocodet":
            # cocodet: no instruction text; CFG on text context (like detection).
            # stage3 trained cocodet WITH the "[start cocodet x10]" target instruction
            # (use_target_instruction=true, and cocodet is NOT in the det/seg/grounding
            # exception in any2any_dataset). Inference omits it (inference_add_instruction
            # =false, a stage2-era setting) -> degenerate boxes on stage3. Opt-in to add
            # the matching instruction (default off so stage2 is unaffected).
            if os.environ.get("MODUS_COCODET_TARGET_INSTR", "0") == "1":
                instr = self._make_instruction("cocodet")
                gen_context = self.update_context_text(instr, gen_context)
                # gen_cocodet's CFG branch is the TEXT context (like detection),
                # NOT the img context. The instruction must be added there so the
                # unconditional branch matches the conditional one. (Was a copy-
                # paste bug that added it to cfg_img_context, which gen_cocodet
                # never reads → the earlier "instruction didn't help" retest was
                # invalid under cfg_scale>1.)
                if cfg_text_context is not None:
                    cfg_text_context = self.update_context_text(instr, cfg_text_context)
            target_list = modality_type_dict.get('target', [target_name])
            if not isinstance(target_list, list):
                target_list = [target_list]
            max_tok = self._get_max_tokens(target_name, max_think_token_n)

            # Same empty-CFG-context guard as detection.
            effective_cfg_scale = cfg_text_scale
            cfg_lens = cfg_text_context.get('kv_lens') if cfg_text_context is not None else None
            if cfg_lens is not None:
                try:
                    total_cfg_tokens = int(sum(
                        cfg_lens.tolist() if hasattr(cfg_lens, "tolist") else cfg_lens
                    ))
                except Exception:
                    total_cfg_tokens = None
                if total_cfg_tokens == 0:
                    effective_cfg_scale = 1.0

            return self.gen_cocodet(
                gen_context,
                do_sample=do_sample,
                temperature=text_temperature,
                max_length=max_tok,
                modality_type_dict={'target': target_list},
                cfg_scale=effective_cfg_scale,
                cfg_text_context=cfg_text_context,
            )

        elif dm in ("dino", "dinolocal", "clip", "imagebind", "imagebindlocal"):
            # Codebook family: add instruction to gen_context + cfg_img_context.
            if add_instr:
                instr = self._make_instruction(target_name)
                gen_context = self.update_context_text(instr, gen_context)
                cfg_img_context = self.update_context_text(instr, cfg_img_context)

            target_list = modality_type_dict.get('target', [target_name])
            if not isinstance(target_list, list):
                target_list = [target_list]
            max_tok = self._get_max_tokens(target_name, max_think_token_n)
            return self.gen_dino_family(
                target_name, gen_context,
                do_sample=do_sample,
                temperature=text_temperature,
                max_length=max_tok,
                modality_type_dict={'target': target_list},
                dino_pca=dino_pca,
                top_k=top_k,
                top_p=top_p,
                cfg_scale=cfg_img_scale,
                cfg_img_context=cfg_img_context,
                return_string=return_string,
            )

        elif dm == "text":
            # Text generation (no CFG).  Adds [start <name> × 10] only when the
            # target is not the plain VQA-answer 'text' modality (e.g. 'caption'
            # gets the instruction; 'text' itself does not).
            if add_instr:
                instr = self._make_instruction(target_name)
                gen_context = self.update_context_text(instr, gen_context)
            return self.gen_text(
                gen_context,
                do_sample=do_sample,
                temperature=text_temperature,
                max_length=max_think_token_n,
            )

        else:
            # Image generation (flow matching).
            if add_instr:
                # seg is category-conditioned: training uses
                # "start segment the mask of {category}" (any2any_dataset._add_seg).
                # Use the caller-provided seg_category (demo UI) rather than the
                # generic _make_instruction, which reads a fixed env var default.
                if target_name == "seg":
                    cat = seg_category or os.environ.get("HUNYUAN_SEG_CATEGORY", "wall")
                    instr = f"start segment the mask of {cat}"
                else:
                    instr = self._make_instruction(target_name)
                print('use_instruction:', use_instruction)
                print(instr)
                gen_context = self.update_context_text(instr, gen_context)
                cfg_img_context = self.update_context_text(instr, cfg_img_context)

            return self.gen_image(
                image_shapes,
                gen_context,
                cfg_text_precontext=cfg_text_context,
                cfg_img_precontext=cfg_img_context,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                modality_type_dict=modality_type_dict,
                do_modality_norm=do_modality_norm,
            )

    # ── Intermediate generation (chained only) ────────────────────────────────

    def _generate_intermediate(
        self,
        intermediate_modality: str,
        gen_context,
        cfg_text_context,
        cfg_img_context,
        base_context_no_intermediate,
        base_cfg_text_context,
        base_cfg_img_context,
        image_shapes,
        target_modality,
        *,
        understanding_output: bool,
        use_instruction: bool,
        use_intermediate_instruction: bool,
        cfg_text_scale: float,
        cfg_img_scale: float,
        cfg_interval,
        timestep_shift: float,
        num_timesteps: int,
        cfg_renorm_min: float,
        cfg_renorm_type: str,
        do_modality_norm: bool,
        do_sample: bool,
        text_temperature: float,
        max_think_token_n: int,
        dino_pca=None,
        top_k: int = 0,
        top_p: float = 1.0,
        force_inject_vae: Optional[bool] = None,
        seg_category: Optional[str] = None,
        det_categories: Optional[List[str]] = None,
    ):
        """Generate the intermediate modality and rebuild contexts.

        ``force_inject_vae``: when not None, override the VAE/ViT decision for
        injecting an *image-like* intermediate back into context.  Used by the
        chained-VQA dual-stack: the inter-stack rebuild forces VAE+ViT to match
        Pass-1 image-gen training distribution, while the pass2-stack mirror
        uses ViT-only (no VAE) to match Pass-2 text-answer training.

        Returns ``(intermediate_output, gen_context, cfg_text_context, cfg_img_context, image_shapes)``.
        """
        inter_dm = self._resolve_decode_method(intermediate_modality)

        # Understanding-like intermediate (text/det/dino)
        if inter_dm in ("detection", "cocodet", "dino", "dinolocal",
                        "clip", "imagebind", "imagebindlocal", "text"):
            # det is special: the validated detection inference recipe is the
            # *grounding* format `"[start grounding the phrase] <phrase>"` (see
            # data/interleave_datasets/any2any_dataset.py:497 grounding training
            # + old BAGEL eval/grounding/evaluate_grounding.py inference recipe).
            # The alternate `"start detect the box of <cats>"` format (det
            # training in any2any_dataset.py:467) is also valid but the
            # grounding-phrase form is the one Phase-1 verified to produce sane
            # boxes (cfg_text_scale=1.0 + temp=0.03).  For multi-category input
            # we use the first as the single phrase (grounding is single-phrase
            # at training).
            det_phrase = None
            if intermediate_modality == "det":
                cats = det_categories or ["person"]
                det_phrase = cats[0] if isinstance(cats, (list, tuple)) and cats else str(cats)
                if use_instruction and use_intermediate_instruction:
                    det_gen_instr = f"[start grounding the phrase] {det_phrase}"
                    gen_context = self.update_context_text(det_gen_instr, gen_context)
                    cfg_img_context = self.update_context_text(det_gen_instr, cfg_img_context)

            # For detection generation we must override cfg_text_scale=1.0
            # (CFG off) and text_temperature=0.03 — the canonical detection
            # recipe.  Other modalities (caption/dino/text) keep the passed
            # hypers so chained VQA / chained image-gen behaviour is unchanged.
            # det & cocodet are CFG-sensitive codebook detection: force canonical
            # hypers (CFG off, near-greedy) or boxes collapse to x1=x2=999.
            _det_like = intermediate_modality == "det" or inter_dm in ("detection", "cocodet")
            inter_cfg_text_scale = 1.0 if _det_like else cfg_text_scale
            inter_text_temp = 0.03 if _det_like else text_temperature
            inter_do_sample = False if _det_like else do_sample

            intermediate_output = self._generate_for_modality(
                intermediate_modality,
                gen_context, cfg_text_context, cfg_img_context, image_shapes,
                modality_type_dict={'condition': [], 'target': [intermediate_modality]},
                understanding_output=True,
                use_instruction=use_instruction,
                use_target_instruction=use_intermediate_instruction,
                cfg_text_scale=inter_cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                do_modality_norm=do_modality_norm,
                do_sample=inter_do_sample,
                text_temperature=inter_text_temp,
                max_think_token_n=max_think_token_n,
                dino_pca=dino_pca,
                top_k=top_k,
                top_p=top_p,
                seg_category=seg_category,
                return_string=True,  # For dino intermediate, return string for context injection
            )

            # cocodet returns parsed boxes; serialize to its token string so it can
            # be injected back as a condition (mirrors the dataset cocodet format).
            if inter_dm == "cocodet" and isinstance(intermediate_output, list):
                intermediate_output = self._serialize_cocodet_boxes(intermediate_output)

            # Rebuild contexts from clean baselines
            if isinstance(intermediate_output, str):
                # det-as-condition (grounding training, any2any_dataset.py:500)
                # prepends the phrase as plain text before the bbox tokens; we
                # mirror that here so the injected intermediate matches the
                # training distribution for the grounding format.
                det_cond_prefix = (
                    det_phrase if intermediate_modality == "det" and det_phrase else None
                )

                gen_context = deepcopy(base_context_no_intermediate)
                if det_cond_prefix is not None:
                    gen_context = self.update_context_text(det_cond_prefix, gen_context)
                gen_context = self.update_context_text(intermediate_output, gen_context, modality_type=intermediate_modality)
                cfg_text_context = deepcopy(base_cfg_text_context)
                cfg_img_context = deepcopy(base_cfg_img_context)
                if det_cond_prefix is not None:
                    cfg_img_context = self.update_context_text(det_cond_prefix, cfg_img_context)
                cfg_img_context = self.update_context_text(intermediate_output, cfg_img_context, modality_type=intermediate_modality)
            elif isinstance(intermediate_output, torch.Tensor):
                gen_context = deepcopy(base_context_no_intermediate)
                cfg_text_context = deepcopy(base_cfg_text_context)
                cfg_img_context = deepcopy(base_cfg_img_context)

        else:
            # Image-like intermediate (e.g., depth/rgb/seg/etc).
            #
            # Seg is special: training uses a category-conditioned plain-text
            # instruction (see data/interleave_datasets/any2any_dataset.py:578),
            # NOT the generic "[start seg × 10]" template.  Override here to
            # match training distribution.
            if intermediate_modality == 'seg':
                cat = seg_category or 'person'
                if use_instruction and use_intermediate_instruction:
                    instr = f"start segment the mask of {cat}"
                    gen_context = self.update_context_text(instr, gen_context)
                    cfg_img_context = self.update_context_text(instr, cfg_img_context)
            elif (use_instruction and use_intermediate_instruction
                  and self._should_add_instruction(intermediate_modality)):
                instr = self._make_instruction(intermediate_modality)
                gen_context = self.update_context_text(instr, gen_context)
                cfg_img_context = self.update_context_text(instr, cfg_img_context)

            intermediate_output = self.gen_image(
                image_shapes,
                gen_context,
                cfg_text_precontext=cfg_text_context,
                cfg_img_precontext=cfg_img_context,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                modality_type_dict={'target': intermediate_modality},
                do_modality_norm=do_modality_norm,
            )

            # Compress intermediate image into KV cache
            inter_img = self.vae_transform.resize_transform(pil_img2rgb(intermediate_output))
            image_shapes = inter_img.size[::-1]
            if force_inject_vae is not None:
                need_vae = force_inject_vae
            else:
                need_vae, _need_vit = self._inference_conditioning_flags(
                    understanding_output=understanding_output,
                    modality_type_dict={"target": target_modality},
                )
            # Seg-as-condition is trained with a "mask category: {cat}" text
            # prefix before the mask image (any2any_dataset.py:550).  Mirror
            # that here so the injected intermediate matches the training
            # conditioning distribution.
            seg_cond_prefix = (
                f"mask category: {seg_category or 'person'}"
                if intermediate_modality == 'seg' else None
            )

            gen_context = deepcopy(base_context_no_intermediate)
            if seg_cond_prefix is not None:
                gen_context = self.update_context_text(seg_cond_prefix, gen_context)
            gen_context = self.update_context_image(
                inter_img, gen_context,
                vae=need_vae,
                modality_type=intermediate_modality,
                do_modality_norm=do_modality_norm,
            )
            cfg_text_context = deepcopy(base_cfg_text_context)
            if seg_cond_prefix is not None:
                cfg_text_context = self.update_context_text(seg_cond_prefix, cfg_text_context)
            cfg_text_context = self.update_context_image(
                inter_img, cfg_text_context,
                vae=need_vae,
                modality_type=intermediate_modality,
                do_modality_norm=do_modality_norm,
            )
            cfg_img_context = deepcopy(base_cfg_img_context)

        return intermediate_output, gen_context, cfg_text_context, cfg_img_context, image_shapes

    # ══════════════════════════════════════════════════════════════════════════
    # Unified inference — replaces both interleave_inference and chained_inference.
    # ══════════════════════════════════════════════════════════════════════════

    def unified_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,

        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        modality_type_dict=None,
        use_instruction=False,
        use_target_instruction=True,
        use_condition_instruction=False,
        use_intermediate_instruction=True,
        do_modality_norm=False,
        use_det_image=False,
        use_gt_dino_condition=False,
        use_gt_dinolocal_condition=False,
        use_gt_clip_condition=False,
        use_gt_imagebind_condition=False,
        use_gt_imagebindlocal_condition=False,
        dino_pca=None,
        top_k=0,
        top_p=1.0,
        seg_category: Optional[str] = None,
        det_categories: Optional[List[str]] = None,
        cond_feature_mode: str = "both",
    ) -> List[Union[str, Image.Image, torch.Tensor]]:
        """Unified any-to-any inference pipeline.

        Handles direct (condition→target) and chained (condition→intermediate→target)
        inference in a single method.  The presence of an ``'intermediate'`` key in
        *modality_type_dict* triggers the chained path.

        This replaces the former ``interleave_inference`` and ``chained_inference``
        methods which shared ~80% identical code.
        """
        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        # Determine target modality name (str).
        target_modality = modality_type_dict.get('target', 'rgb') if modality_type_dict else 'rgb'
        target_name = target_modality if isinstance(target_modality, str) else target_modality[0]

        # Determine if there are intermediates (supports single str or list of str).
        intermediate_modality = modality_type_dict.get('intermediate') if modality_type_dict else None
        if intermediate_modality is not None:
            if isinstance(intermediate_modality, str):
                intermediate_modality = None if intermediate_modality.lower() in ['none', ''] else [intermediate_modality]
            elif isinstance(intermediate_modality, list):
                intermediate_modality = [
                    m for m in intermediate_modality
                    if isinstance(m, str) and m.lower() not in ['none', '']
                ]
                intermediate_modality = intermediate_modality or None
        is_chained = intermediate_modality is not None

        # Auto-detect understanding_output from target modality.
        if not understanding_output:
            understanding_output = self._is_understanding_target(target_name)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            # ── 0) System prompt (think mode) ─────────────────────────────────
            if think:
                system_prompt = VLM_THINK_SYSTEM_PROMPT if understanding_output else GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            # ── 1) Feed all conditions ────────────────────────────────────────
            gen_context, cfg_text_context, cfg_img_context, image_shapes = self._feed_conditions(
                input_lists, gen_context, cfg_text_context, cfg_img_context,
                modality_type_dict=modality_type_dict,
                understanding_output=understanding_output,
                use_instruction=use_instruction,
                use_condition_instruction=use_condition_instruction,
                use_det_image=use_det_image,
                do_modality_norm=do_modality_norm,
                image_shapes=image_shapes,
                use_gt_dino_condition=use_gt_dino_condition,
                use_gt_dinolocal_condition=use_gt_dinolocal_condition,
                use_gt_clip_condition=use_gt_clip_condition,
                use_gt_imagebind_condition=use_gt_imagebind_condition,
                use_gt_imagebindlocal_condition=use_gt_imagebindlocal_condition,
                cond_feature_mode=cond_feature_mode,
            )

            # ── 2) Optional intermediate(s) (chained only) ───────────────────
            if is_chained:
                # Two independent decisions for chained VQA (target=='text'):
                #   - defer_question: ALWAYS true for VQA chains.  The question
                #     must not be in the intermediate's context (training:
                #     `image + phrase → det/dino/...`, NEVER `image + question +
                #     phrase → intermediate`).  Append it only before the final
                #     gen_text.
                #   - needs_dual_stack: only when the intermediate is *image-like*
                #     (depth/normal/seg/canny/rgb).  Then we maintain a parallel
                #     "inter" stack with rgb VAE+ViT (matches image-target
                #     training) for Pass-1 image gen, and a "pass2" stack with
                #     rgb ViT-only (matches text-target training) for the final
                #     gen_text.  For codebook intermediates (det/dino/...) a
                #     single ViT-only stack suffices — the intermediate is a
                #     text-token string, not an image to re-encode.
                #
                # Earlier versions tied question-deferral to needs_dual_stack,
                # which silently broke chained VQA + det/dino: the question
                # leaked into the intermediate context and the final answer
                # context had the wrong token order (`image+question+phrase+
                # det_tokens+answer` instead of `image+phrase+det_tokens+
                # question+answer`).
                is_vqa = (target_name == "text")
                # det-target chains (chained grounding: rgb -> inter -> det)
                # defer the grounding phrase the same way VQA defers the
                # question: training is `image -> inter` (no text) and
                # `image + phrase -> det`, never `image + phrase -> inter`.
                # No dual stack needed — det targets keep VAE+ViT throughout
                # (encoding is target-driven in _inference_conditioning_flags).
                is_det_target = self._resolve_decode_method(target_name) == "detection"
                _feature_dms = (
                    "text", "detection", "dino", "dinolocal",
                    "clip", "imagebind", "imagebindlocal",
                )
                any_image_like_inter = any(
                    self._resolve_decode_method(m) not in _feature_dms
                    for m in intermediate_modality
                )
                defer_question = is_vqa or is_det_target
                needs_dual_stack = is_vqa and any_image_like_inter

                # Default contexts for non-VQA single-stack: keep what
                # _feed_conditions returned (question/text already in context).
                inter_gen = gen_context
                inter_cfg_text = cfg_text_context
                inter_cfg_img = cfg_img_context
                pass2_gen = pass2_cfg_text = pass2_cfg_img = None
                post_loop_text: List[Any] = []

                if defer_question:
                    # Split inputs: text inputs (question) get deferred to after
                    # the chained loop, for ALL VQA chains (image-like or
                    # codebook intermediate).
                    cond_list = (
                        modality_type_dict.get("condition", [])
                        if modality_type_dict else []
                    )
                    pre_loop_inputs: List[Any] = []
                    pre_loop_mods: List[Any] = []
                    paired_mods = list(cond_list) + [None] * max(
                        0, len(input_lists) - len(cond_list)
                    )
                    for inp, mod in zip(input_lists, paired_mods):
                        if isinstance(inp, str):
                            post_loop_text.append((inp, mod or "text"))
                        else:
                            pre_loop_inputs.append(inp)
                            pre_loop_mods.append(mod)
                    pre_loop_md = (
                        {**modality_type_dict, "condition": pre_loop_mods}
                        if modality_type_dict is not None
                        else {"condition": pre_loop_mods}
                    )

                if needs_dual_stack:
                    # Inter stack — fake target as image-like so the encoding-flag
                    # decision returns VAE+ViT for the input rgb.
                    inter_md = {**pre_loop_md, "target": "image"}
                    fresh1 = self.init_gen_context()
                    inter_gen, inter_cfg_text, inter_cfg_img, image_shapes = \
                        self._feed_conditions(
                            pre_loop_inputs,
                            fresh1, deepcopy(fresh1), deepcopy(fresh1),
                            modality_type_dict=inter_md,
                            understanding_output=False,
                            use_instruction=use_instruction,
                            use_condition_instruction=use_condition_instruction,
                            use_det_image=use_det_image,
                            do_modality_norm=do_modality_norm,
                            image_shapes=image_shapes,
                            use_gt_dino_condition=use_gt_dino_condition,
                            use_gt_dinolocal_condition=use_gt_dinolocal_condition,
                            use_gt_clip_condition=use_gt_clip_condition,
                            use_gt_imagebind_condition=use_gt_imagebind_condition,
                            use_gt_imagebindlocal_condition=use_gt_imagebindlocal_condition,
                        )

                    # Pass-2 stack — original target (text), rgb encoded ViT-only.
                    fresh2 = self.init_gen_context()
                    pass2_gen, pass2_cfg_text, pass2_cfg_img, _ = \
                        self._feed_conditions(
                            pre_loop_inputs,
                            fresh2, deepcopy(fresh2), deepcopy(fresh2),
                            modality_type_dict=pre_loop_md,
                            understanding_output=True,
                            use_instruction=use_instruction,
                            use_condition_instruction=use_condition_instruction,
                            use_det_image=use_det_image,
                            do_modality_norm=do_modality_norm,
                            image_shapes=image_shapes,
                            use_gt_dino_condition=use_gt_dino_condition,
                            use_gt_dinolocal_condition=use_gt_dinolocal_condition,
                            use_gt_clip_condition=use_gt_clip_condition,
                            use_gt_imagebind_condition=use_gt_imagebind_condition,
                            use_gt_imagebindlocal_condition=use_gt_imagebindlocal_condition,
                        )
                elif defer_question:
                    # Single ViT-only stack for VQA + codebook-only chain.
                    # Re-feed conditions with image-only inputs so the question
                    # is NOT in the intermediate's context.  rgb is encoded
                    # ViT-only (matches text-target training distribution).
                    fresh = self.init_gen_context()
                    inter_gen, inter_cfg_text, inter_cfg_img, image_shapes = \
                        self._feed_conditions(
                            pre_loop_inputs,
                            fresh, deepcopy(fresh), deepcopy(fresh),
                            modality_type_dict=pre_loop_md,
                            understanding_output=True,
                            use_instruction=use_instruction,
                            use_condition_instruction=use_condition_instruction,
                            use_det_image=use_det_image,
                            do_modality_norm=do_modality_norm,
                            image_shapes=image_shapes,
                            use_gt_dino_condition=use_gt_dino_condition,
                            use_gt_dinolocal_condition=use_gt_dinolocal_condition,
                            use_gt_clip_condition=use_gt_clip_condition,
                            use_gt_imagebind_condition=use_gt_imagebind_condition,
                            use_gt_imagebindlocal_condition=use_gt_imagebindlocal_condition,
                        )

                # Loop: each intermediate runs on the inter stack; for dual-stack
                # we also mirror the result into the pass2 stack with ViT-only
                # encoding (image-like) or as text tokens (codebook).
                for inter_mod in intermediate_modality:
                    base_context = deepcopy(inter_gen)
                    base_cfg_text = deepcopy(inter_cfg_text)
                    base_cfg_img = deepcopy(inter_cfg_img)

                    inter_out, inter_gen, inter_cfg_text, inter_cfg_img, image_shapes = \
                        self._generate_intermediate(
                            inter_mod,
                            inter_gen, inter_cfg_text, inter_cfg_img,
                            base_context, base_cfg_text, base_cfg_img,
                            image_shapes, target_modality,
                            understanding_output=understanding_output,
                            use_instruction=use_instruction,
                            use_intermediate_instruction=use_intermediate_instruction,
                            cfg_text_scale=cfg_text_scale,
                            cfg_img_scale=cfg_img_scale,
                            cfg_interval=cfg_interval,
                            timestep_shift=timestep_shift,
                            num_timesteps=num_timesteps,
                            cfg_renorm_min=cfg_renorm_min,
                            cfg_renorm_type=cfg_renorm_type,
                            do_modality_norm=do_modality_norm,
                            do_sample=do_sample,
                            text_temperature=text_temperature,
                            max_think_token_n=max_think_token_n,
                            dino_pca=dino_pca,
                            top_k=top_k,
                            top_p=top_p,
                            force_inject_vae=True if needs_dual_stack else None,
                            seg_category=seg_category,
                            det_categories=det_categories,
                        )
                    output_list.append(inter_out)

                    if needs_dual_stack:
                        if isinstance(inter_out, Image.Image):
                            mirror_img = self.vae_transform.resize_transform(
                                pil_img2rgb(inter_out)
                            )
                            # Seg-as-condition is trained with a "mask category: {cat}"
                            # text prefix (any2any_dataset.py:550) — mirror that here too.
                            if inter_mod == "seg":
                                seg_prefix = f"mask category: {seg_category or 'person'}"
                                pass2_gen = self.update_context_text(seg_prefix, pass2_gen)
                                pass2_cfg_text = self.update_context_text(seg_prefix, pass2_cfg_text)
                            pass2_gen = self.update_context_image(
                                mirror_img, pass2_gen,
                                vae=False, modality_type=inter_mod,
                                do_modality_norm=do_modality_norm,
                            )
                            pass2_cfg_text = self.update_context_image(
                                mirror_img, pass2_cfg_text,
                                vae=False, modality_type=inter_mod,
                                do_modality_norm=do_modality_norm,
                            )
                            # pass2_cfg_img stays at base — image-CFG drops the intermediate.
                        elif isinstance(inter_out, str):
                            # det-as-condition (grounding training format,
                            # any2any_dataset.py:500) prepends the phrase as
                            # plain text before the bbox tokens.  Mirror here
                            # for the Pass-2 ViT-only stack.
                            if inter_mod == "det":
                                cats = det_categories or ["person"]
                                det_phrase = cats[0] if isinstance(cats, (list, tuple)) and cats else str(cats)
                                pass2_gen = self.update_context_text(det_phrase, pass2_gen)
                                pass2_cfg_text = self.update_context_text(det_phrase, pass2_cfg_text)
                                pass2_cfg_img = self.update_context_text(det_phrase, pass2_cfg_img)
                            pass2_gen = self.update_context_text(
                                inter_out, pass2_gen, modality_type=inter_mod,
                            )
                            pass2_cfg_text = self.update_context_text(
                                inter_out, pass2_cfg_text, modality_type=inter_mod,
                            )
                            pass2_cfg_img = self.update_context_text(
                                inter_out, pass2_cfg_img, modality_type=inter_mod,
                            )

                # After loop: append deferred question text in the natural
                # order (rgb + intermediates + question).  For dual-stack the
                # question goes onto the ViT-only pass2 stack; for non-dual VQA
                # (codebook intermediate) it goes onto the single inter stack.
                if needs_dual_stack:
                    for txt, mod in post_loop_text:
                        pass2_cfg_text = deepcopy(pass2_gen)
                        pass2_gen = self.update_context_text(
                            txt, pass2_gen, modality_type=mod,
                        )
                        pass2_cfg_img = self.update_context_text(
                            txt, pass2_cfg_img, modality_type=mod,
                        )
                    gen_context = pass2_gen
                    cfg_text_context = pass2_cfg_text
                    cfg_img_context = pass2_cfg_img
                elif defer_question:
                    for txt, mod in post_loop_text:
                        inter_cfg_text = deepcopy(inter_gen)
                        inter_gen = self.update_context_text(
                            txt, inter_gen, modality_type=mod,
                        )
                        inter_cfg_img = self.update_context_text(
                            txt, inter_cfg_img, modality_type=mod,
                        )
                    gen_context = inter_gen
                    cfg_text_context = inter_cfg_text
                    cfg_img_context = inter_cfg_img
                else:
                    gen_context = inter_gen
                    cfg_text_context = inter_cfg_text
                    cfg_img_context = inter_cfg_img

            # ── 3) Think step (non-chained generation only) ──────────────────
            if not is_chained and not understanding_output and think:
                gen_text = self.gen_text(
                    gen_context, do_sample=do_sample,
                    temperature=text_temperature, max_length=max_think_token_n,
                )
                gen_context = self.update_context_text(gen_text, gen_context)
                output_list.append(gen_text)

            # ── 4) Generate final target ─────────────────────────────────────
            target_output = self._generate_for_modality(
                target_name,
                gen_context, cfg_text_context, cfg_img_context, image_shapes,
                modality_type_dict=modality_type_dict,
                understanding_output=understanding_output,
                use_instruction=use_instruction,
                use_target_instruction=use_target_instruction,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                do_modality_norm=do_modality_norm,
                do_sample=do_sample,
                text_temperature=text_temperature,
                max_think_token_n=max_think_token_n,
                dino_pca=dino_pca,
                top_k=top_k,
                top_p=top_p,
                seg_category=seg_category,
            )
            output_list.append(target_output)

        return output_list

    # ── __call__ ──────────────────────────────────────────────────────────────

    def __call__(
        self, 
        image: Optional[Union[Image.Image, List[Image.Image]]] = None, 
        text: Optional[str] = None, 
        **kargs
    ) -> Dict[str, Any]:
        output_dict = {'image': None, 'text': None}

        if image is None and text is None:
            print('Please provide at least one input: either an image or text.')
            return output_dict

        input_list = []
        if image is not None:
            if isinstance(image, list):
                input_list.extend(image)
                assert kargs['modality_type_dict']['target'][0] == "text"
            else:
                input_list.append(image)
        if text is not None and text:
            input_list.append(text)

        if kargs['modality_type_dict']['target'] == 'seg':
            if kargs['modality_type_dict']['condition'][0] == 'caption':
                _seg_cat = kargs.get('seg_category') or 'person'
                input_list.append(f'start segment the mask of {_seg_cat}')

        is_chained = kargs.pop('chained_inference', False)
        output_list = self.unified_inference(input_list, **kargs)

        if is_chained:
            output_dict['image'] = []
            output_dict['text'] = []
            output_dict['dino_feat'] = []
            for i in output_list:
                if isinstance(i, Image.Image):
                    output_dict['image'].append(i)
                elif isinstance(i, str):
                    output_dict['text'].append(i)
                elif isinstance(i, torch.Tensor):
                    output_dict['dino_feat'].append(i)
            # Collapse single/empty lists for backward compat.
            if not output_dict['text']:
                output_dict['text'] = None
            elif len(output_dict['text']) == 1:
                output_dict['text'] = output_dict['text'][0]
            if not output_dict['dino_feat']:
                output_dict['dino_feat'] = None
            elif len(output_dict['dino_feat']) == 1:
                output_dict['dino_feat'] = output_dict['dino_feat'][0]
        else:
            for i in output_list:
                if isinstance(i, Image.Image):
                    output_dict['image'] = i
                elif isinstance(i, str):
                    output_dict['text'] = i
                elif isinstance(i, torch.Tensor):
                    output_dict['dino_feat'] = i
                elif isinstance(i, list):
                    # cocodet → list of {bbox, label, score} dicts.
                    output_dict['cocodet_boxes'] = i

        # Attach detection confidence if available
        if hasattr(self, 'last_det_confidence') and self.last_det_confidence is not None:
            output_dict['det_confidence'] = self.last_det_confidence
            output_dict['det_coord_probs'] = getattr(self, 'last_det_coord_probs', None)
        return output_dict


def move_tensors_to_device(obj, device):
    if isinstance(obj, dict):
        return {k: move_tensors_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_tensors_to_device(v, device) for v in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    else:
        return obj


def pca_image_sklearn(feat_map: torch.Tensor) -> Image.Image:
    """
    Create a PCA visualization image from a feature map using sklearn PCA.

    Args:
        feat_map: torch.Tensor of shape [1, C, H, W]

    Returns:
        (image, pca):
            - image: PIL.Image.Image of size (W, H) in RGB representing the first 3 PCs
            - pca: fitted sklearn.decomposition.PCA instance with 3 components
    """
    import numpy as np
    from sklearn.decomposition import PCA

    assert feat_map.ndim == 4 and feat_map.shape[0] == 1, "Expect [1,C,H,W]"
    _, C, H, W = feat_map.shape

    x = (
        feat_map[0]
        .permute(1, 2, 0)          # [H, W, C]
        .contiguous()
        .view(-1, C)               # [N, C]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    # sklearn PCA centers internally; no manual mean subtraction needed
    pca = PCA(n_components=3)
    y = pca.fit_transform(x)       # [N, 3]
    y = y.reshape(H, W, 3)

    # Per-channel min-max normalization to [0,1]
    y_min = y.min(axis=(0, 1), keepdims=True)
    y_max = y.max(axis=(0, 1), keepdims=True)
    y = (y - y_min) / (y_max - y_min + 1e-8)

    img = (y * 255.0).astype(np.uint8)
    return Image.fromarray(img), pca


def pca_apply_sklearn(feat_map: torch.Tensor, pca) -> Image.Image:
    """
    Apply a pre-fitted sklearn PCA (with 3 components) to a new feature map
    and return a 3-channel visualization image.

    Args:
        feat_map: torch.Tensor of shape [1, C, H, W]
        pca: a fitted sklearn.decomposition.PCA with n_components=3

    Returns:
        PIL.Image.Image visualization using the provided PCA bases.
    """
    import numpy as np

    assert feat_map.ndim == 4 and feat_map.shape[0] == 1, "Expect [1,C,H,W]"
    _, C, H, W = feat_map.shape
    # Validate PCA compatibility
    assert getattr(pca, 'n_components_', 3) == 3, "PCA must have 3 components"
    assert pca.components_.shape[1] == C, f"PCA expects {pca.components_.shape[1]} channels, got {C}"

    x = (
        feat_map[0]
        .permute(1, 2, 0)          # [H, W, C]
        .contiguous()
        .view(-1, C)               # [N, C]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    y = pca.transform(x)          # [N, 3]
    y = y.reshape(H, W, 3)

    # Per-channel min-max normalization to [0,1]
    y_min = y.min(axis=(0, 1), keepdims=True)
    y_max = y.max(axis=(0, 1), keepdims=True)
    y = (y - y_min) / (y_max - y_min + 1e-8)

    img = (y * 255.0).astype(np.uint8)
    return Image.fromarray(img)
