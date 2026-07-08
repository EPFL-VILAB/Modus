# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import os
import json
import random
from PIL import Image, ImageFile, PngImagePlugin, ImageDraw, ImageFont
import numpy as np
import pycocotools.mask as mask_util
import re

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class PredIntermediateMissing(Exception):
    """Raised in pred-intermediate mode when the saved PNG for this row is
    missing. Caller (interleave_t2i_dataset.__iter__) catches it silently
    to skip the row without falling back to GT."""
    pass


class UnifiedAny2AnyIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    IMAGE_LIKE_TARGETS = {'rgb', 'depth', 'normal', 'canny', 'seg', 'samseg', 'samedge'}
    FEATURE_TARGETS = {'dino', 'dinolocal', 'clip', 'imagebind', 'imagebindlocal'}
    # New direct-PNG image modalities → parquet column name mapping.
    SAM_IMAGE_COLUMN = {'samseg': 'sam_seg', 'samedge': 'sam_edge'}
    PRED_INTERMEDIATE_IMAGE_MODS = {'depth', 'normal', 'canny', 'seg'}
    PRED_INTERMEDIATE_TEXT_MODS = {'caption'}
    PRED_INTERMEDIATE_DINO_MODS = {'dino'}

    def _load_pred_text(self, modality):
        """Load Phase-1 generated text (caption) keyed by row provenance.
        Opt-in: only when pred_intermediate_dir is set. Returns str or None."""
        if getattr(self, 'pred_intermediate_dir', None) is None:
            return None
        if modality not in self.PRED_INTERMEDIATE_TEXT_MODS:
            return None
        parquet_path = getattr(self, '_current_parquet_path', None)
        rg_id = getattr(self, '_current_rg_id', None)
        row_idx = getattr(self, '_current_row_idx', None)
        if parquet_path is None or rg_id is None or row_idx is None:
            return None
        basename = os.path.splitext(os.path.basename(parquet_path))[0]
        path = os.path.join(
            self.pred_intermediate_dir, modality,
            f"{basename}_rg{rg_id}_row{int(row_idx)}_pred.txt",
        )
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            return f.read().strip()

    def _load_pred_dino(self, modality):
        """Load Phase-1 generated dino feature (.npy) keyed by row provenance.
        Returns np.ndarray or None."""
        if getattr(self, 'pred_intermediate_dir', None) is None:
            return None
        if modality not in self.PRED_INTERMEDIATE_DINO_MODS:
            return None
        parquet_path = getattr(self, '_current_parquet_path', None)
        rg_id = getattr(self, '_current_rg_id', None)
        row_idx = getattr(self, '_current_row_idx', None)
        if parquet_path is None or rg_id is None or row_idx is None:
            return None
        basename = os.path.splitext(os.path.basename(parquet_path))[0]
        path = os.path.join(
            self.pred_intermediate_dir, modality,
            f"{basename}_rg{rg_id}_row{int(row_idx)}_pred.npy",
        )
        if not os.path.exists(path):
            return None
        arr = np.load(path)
        # Dino tokens are codebook ints; if Phase 1 saved as float (older
        # runs) round to nearest int to avoid float16 precision loss.
        if arr.dtype.kind == 'f':
            arr = np.rint(arr).astype(np.int64)
        # Phase 1 saves with leading batch dim (1, 16); _add_dino expects
        # a flat (16,) array. Squeeze any singleton leading dims.
        while arr.ndim > 1 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        return arr

    def _load_pred_pil(self, modality):
        """Load a model-generated intermediate image keyed by
        (parquet_basename, rg_id, row_idx). Returns PIL or None.
        Opt-in: only active when pred_intermediate_dir is set.
        Filename convention written by
        scripts/paper/modus_stage3/generate_chained_intermediates.py:
            <pred_dir>/<modality>/<basename>_rg<rg>_row<row>_pred.png
        """
        if getattr(self, 'pred_intermediate_dir', None) is None:
            return None
        if modality not in self.PRED_INTERMEDIATE_IMAGE_MODS:
            return None
        parquet_path = getattr(self, '_current_parquet_path', None)
        rg_id = getattr(self, '_current_rg_id', None)
        row_idx = getattr(self, '_current_row_idx', None)
        if parquet_path is None or rg_id is None or row_idx is None:
            return None
        basename = os.path.splitext(os.path.basename(parquet_path))[0]
        path = os.path.join(
            self.pred_intermediate_dir, modality,
            f"{basename}_rg{rg_id}_row{int(row_idx)}_pred.png",
        )
        if not os.path.exists(path):
            return None
        return pil_img2rgb(Image.open(path))
    NATIVE_CHAT_SYSTEM_PROMPT = (
        "You are an advanced multimodal model. Given the user's text, visual, "
        "or structured inputs, follow the user instruction and produce the "
        "requested output modality accurately."
    )

    def _native_aligned_serialization(self):
        return os.environ.get("HUNYUAN_NATIVE_ALIGNED_SERIALIZATION", "0") == "1"

    def _native_chat_serialization(self):
        return os.environ.get("HUNYUAN_NATIVE_CHAT_SERIALIZATION", "0") == "1"

    def _add_native_chat_preamble(self, data):
        data = self._add_text(
            data,
            self.NATIVE_CHAT_SYSTEM_PROMPT,
            need_loss=False,
            enable_cfg=False,
            modality_type='text',
        )
        return self._add_text(
            data,
            "User:",
            need_loss=False,
            enable_cfg=False,
            modality_type='text',
        )

    def _add_native_chat_assistant_prefix(self, data):
        return self._add_text(
            data,
            "Assistant: <answer>",
            need_loss=False,
            enable_cfg=False,
            modality_type='text',
        )

    def _target_instruction_text_native_chat(self, target_modality):
        if target_modality == 'rgb':
            return 'Generate an image.'
        if target_modality == 'caption':
            return 'Describe the image.'
        if target_modality in self.IMAGE_LIKE_TARGETS:
            return f'Generate the {target_modality} image.'
        if target_modality == 'det':
            return 'Detect the requested bounding boxes.'
        if target_modality == 'grounding':
            return 'Ground the requested phrase.'
        if target_modality in self.FEATURE_TARGETS:
            return f'Extract {target_modality} features.'
        return f'Generate the {target_modality}.'

    def _target_instruction_text(self, target_modality):
        if not self._native_aligned_serialization():
            return (
                f'[start {target_modality} {target_modality} {target_modality} '
                f'{target_modality} {target_modality} {target_modality} '
                f'{target_modality} {target_modality} {target_modality} {target_modality}]'
            )
        if target_modality == 'rgb':
            return 'Generate an image.'
        if target_modality in self.IMAGE_LIKE_TARGETS:
            return f'Generate the {target_modality} image.'
        if target_modality == 'caption':
            return 'Describe the image.'
        return f'Generate the {target_modality}.'

    def parse_row(self, row):
        # Parse dataset name to get target modality
        # Format: "any2rgb", "any2caption", etc.
        target_modality = self.dataset_name.split("2")[-1]
        if target_modality == 'grounding' and ('grounding' not in row.index or row['grounding'] is None):
            return None

        enable_cfg = False if target_modality in ('caption', 'det', 'cocodet') else True

        # cocodet target: skip rows with no detections (would be an empty
        # target sequence). is_mandatory: false lets DataPack pick another row.
        if target_modality == 'cocodet':
            cd_dets, _ = self._parse_coco_det(row)
            if not cd_dets:
                return None
        
        # Get all available modalities except uid
        available_modalities = [col for col in row.index if col != 'uid']
        if 'grounding' in available_modalities:
            if row['grounding'] is None:
                available_modalities.remove('grounding')


        if 'det_seg' in available_modalities:
            available_modalities.remove('det_seg')
            # available_modalities.append('det')
            available_modalities.append('seg')

        if 'dino' in available_modalities:
            available_modalities.remove('dino')
            available_modalities.append('dinolocal')

        if 'dino_global' in available_modalities:
            available_modalities.remove('dino_global')
            available_modalities.append('dino')

        # Remap parquet column 'clip448' to modality name 'clip'
        if 'clip448' in available_modalities:
            available_modalities.remove('clip448')
            available_modalities.append('clip')

        if 'imagebind' in available_modalities:
            available_modalities.remove('imagebind')
            available_modalities.append('imagebindlocal')

        if 'imagebind_global' in available_modalities:
            available_modalities.remove('imagebind_global')
            available_modalities.append('imagebind')

        # New modalities: remap parquet column names → modality names.
        if 'sam_seg' in available_modalities:
            available_modalities.remove('sam_seg')
            available_modalities.append('samseg')
        if 'sam_edge' in available_modalities:
            available_modalities.remove('sam_edge')
            available_modalities.append('samedge')
        if 'coco_det' in available_modalities:
            available_modalities.remove('coco_det')
            # Only offer cocodet as a condition when there are detections.
            cd_dets, _ = self._parse_coco_det(row)
            if cd_dets:
                available_modalities.append('cocodet')

        # Remove target modality from available modalities for conditioning
        condition_modalities = [mod for mod in available_modalities if mod != target_modality]
        condition_sampling_probs = None

        # Drop modalities the active registry doesn't recognise. This codepath
        # matters when the parquet predates a remap added later (e.g. the
        # `dino` column was renamed to `dinolocal` in cbba109/2026-01-20, but
        # a stage3 ckpt trained at 0a337b1/2025-11-06 has no `dinolocal`
        # modality). Without this guard the dataset feeds an unrecognised
        # modality_type into pack_sequence and triggers KeyError on
        # self.modality_to_id[...].
        if self.modality_registry is not None:
            known = set(self.modality_registry.name_to_id().keys())
            # `grounding` is a dataset-internal conditioning alias (phrase+bbox).
            # Its emitted tokens are tagged `modality_type='det'`/`'text'` (see
            # `_add_grounding`), so the registry KeyError that motivated this
            # filter doesn't apply. Keep it as a usable selection name.
            known.add('grounding')
            condition_modalities = [mod for mod in condition_modalities if mod in known]

        # Filter conditions via config-driven modality registry.
        if self.modality_registry is not None:
            allowed = self.modality_registry.conditions_for(target_modality)
            if allowed is not None:
                condition_modalities = [mod for mod in condition_modalities if mod in allowed]
                configured_probs = self.modality_registry.condition_probs_for(target_modality)
                if configured_probs is not None:
                    prob_by_modality = {
                        cond: prob for cond, prob in zip(allowed, configured_probs)
                    }
                    condition_sampling_probs = [
                        prob_by_modality[mod]
                        for mod in condition_modalities
                        if mod in prob_by_modality
                    ]

        if len(condition_modalities) == 0:
            return None

        # Validation path: force a single fixed condition modality so the loss
        # can be bucketed into a (cond, tgt) matrix. Skip rows where the
        # requested modality is not available for this target.
        if getattr(self, 'force_condition_modalities', None) is not None:
            required = list(self.force_condition_modalities)
            if not all(m in condition_modalities for m in required):
                return None
            selected_conditions = required
        elif getattr(self, 'force_condition_modality', None) is not None:
            if self.force_condition_modality not in condition_modalities:
                return None
            selected_conditions = [self.force_condition_modality]
        elif self.num_condition_modalities == 0:
            num_conditions = random.randint(1, len(condition_modalities))
            selected_conditions = None  # set below
        else:
            if self.strict_num_condition_modalities:
                if len(condition_modalities) < self.num_condition_modalities:
                    return None
                num_conditions = self.num_condition_modalities
            else:
                max_conditions = min(self.num_condition_modalities, len(condition_modalities))
                if max_conditions == 1:
                    num_conditions = max_conditions
                else:
                    num_conditions = random.randint(1, max_conditions)
            selected_conditions = None  # set below

        if selected_conditions is None:
            if condition_sampling_probs is None:
                selected_conditions = random.sample(condition_modalities, num_conditions)
            else:
                probs = np.array(condition_sampling_probs, dtype=np.float64)
                probs_sum = float(probs.sum())
                if probs_sum <= 0.0:
                    probs = np.full(len(condition_modalities), 1.0 / len(condition_modalities))
                else:
                    probs = probs / probs_sum
                selected_indices = np.random.choice(
                    len(condition_modalities),
                    size=num_conditions,
                    replace=False,
                    p=probs,
                )
                selected_conditions = [condition_modalities[i] for i in selected_indices]

        # Optional monitoring tag for route-specific rgb generation logging.
        # Only tag unambiguous single-condition rgb targets.
        if target_modality == 'rgb' and len(selected_conditions) == 1:
            cond = selected_conditions[0]
            if cond in ('caption', 'grounding', 'dinolocal'):
                data_route = f"{cond}2rgb"
            else:
                data_route = None
        else:
            data_route = None
        

        # if num_conditions > 1:
        #     print("selected conditions:", selected_conditions)
        #     print("num conditions:", num_conditions)
        #     print('target modality:', target_modality)
        #     if 'grounding' in selected_conditions:
        #         print("grounding instances:", row['grounding'])
        data = self._init_data()
        if data_route is not None:
            data['rgb_loss_route'] = data_route

        native_chat = self._native_chat_serialization()
        if native_chat:
            data = self._add_native_chat_preamble(data)
        
        # Add condition modalities in random order
        random.shuffle(selected_conditions)

        for modality in selected_conditions:
            if self.use_instruction and self.use_condition_instruction:
                data = self._add_text(data, f'[start {modality}]', need_loss=False, enable_cfg=enable_cfg, modality_type='text')
            if modality == 'caption':
                # Pred caption: load from disk if Phase 1 wrote it; else
                # use parquet GT.
                pred_caption = self._load_pred_text('caption')
                if pred_caption is not None:
                    caption_text = pred_caption
                elif (getattr(self, 'pred_intermediate_dir', None) is not None
                        and 'caption' in self.PRED_INTERMEDIATE_TEXT_MODS):
                    raise PredIntermediateMissing(
                        'no pred TXT for caption at this row')
                else:
                    caption_text = row[modality]
                data = self._add_text(data, caption_text, need_loss=False, enable_cfg=enable_cfg, modality_type=modality)
            elif modality in ['rgb', 'depth', 'normal', 'canny']:
                # Handle image modalities. In pred-intermediate mode, depth/
                # normal/canny conditions come from disk (Phase 1 outputs);
                # rgb is always real. If pred mode is on and the file for
                # this row is missing, skip the row entirely — do NOT
                # silently fall back to GT.
                pred_pil = self._load_pred_pil(modality)
                if pred_pil is not None:
                    image = pred_pil
                elif (getattr(self, 'pred_intermediate_dir', None) is not None
                        and modality in self.PRED_INTERMEDIATE_IMAGE_MODS):
                    raise PredIntermediateMissing(
                        f'no pred PNG for {modality} at this row')
                else:
                    image = pil_img2rgb(Image.open(io.BytesIO(row[modality])))
                data = self._add_image(
                    data,
                    image,
                    need_loss=False,
                    enable_cfg=enable_cfg,
                    need_vae=True,
                    need_vit=True,
                    modality_type=modality,
                )
            elif modality in ['samseg', 'samedge']:
                # Direct-PNG SAM image modalities (samseg is RGBA → pil_img2rgb).
                image = pil_img2rgb(Image.open(io.BytesIO(row[self.SAM_IMAGE_COLUMN[modality]])))
                data = self._add_image(
                    data,
                    image,
                    need_loss=False,
                    enable_cfg=enable_cfg,
                    need_vae=True,
                    need_vit=True,
                    modality_type=modality,
                )
            elif modality in ['cocodet']:
                cd_dets, cd_size = self._parse_coco_det(row)
                data = self._add_cocodet(
                    data,
                    detections=cd_dets,
                    image_size=cd_size,
                    need_loss=False,
                    enable_cfg=enable_cfg,
                    modality_type=modality,
                )
            elif modality in ['det']:
                det_seg_instances = row['det_seg']['instances']
                det_seg_raw_size = row['det_seg']['size']
                det_seg_prompts = row['det_seg']['prompt'].split(' .')
                data = self._add_det(
                    data, 
                    det_seg_instances=det_seg_instances, 
                    det_seg_raw_size=det_seg_raw_size, 
                    det_seg_prompts=det_seg_prompts, 
                    need_loss=False, 
                    enable_cfg=enable_cfg, 
                    modality_type=modality
                )

            elif modality in ['seg']:
                # Pred-mode seg: if Phase 1 saved a pred seg PNG, use it
                # via _add_image directly (skip _add_seg's RLE rendering).
                pred_seg = self._load_pred_pil('seg')
                if pred_seg is not None:
                    data = self._add_image(
                        data,
                        pred_seg,
                        need_loss=False,
                        enable_cfg=enable_cfg,
                        need_vae=True,
                        need_vit=True,
                        modality_type='seg',
                    )
                    continue
                elif (getattr(self, 'pred_intermediate_dir', None) is not None
                        and 'seg' in self.PRED_INTERMEDIATE_IMAGE_MODS):
                    raise PredIntermediateMissing(
                        'no pred PNG for seg at this row')
                det_seg_instances = row['det_seg']['instances']
                det_seg_raw_size = row['det_seg']['size']
                det_seg_prompts = row['det_seg']['prompt'].split(' .')
                data = self._add_seg(
                    data,
                    det_seg_instances=det_seg_instances,
                    det_seg_raw_size=det_seg_raw_size,
                    det_seg_prompts=det_seg_prompts,
                    need_loss=False,
                    enable_cfg=enable_cfg,
                    need_vae=True,
                    need_vit=True,
                    modality_type=modality
                )

            elif modality in ['grounding']:
                grounding_instances = row['grounding']
                data = self._add_grounding(
                    data, 
                    grounding_instances=grounding_instances, 
                    need_loss=False, 
                    enable_cfg=enable_cfg,
                    modality_type=modality
                )
            elif modality in ['dino']:
                # Pred dino: load .npy from disk if Phase 1 wrote it.
                pred_dino = self._load_pred_dino('dino')
                if pred_dino is not None:
                    dino_tokens = pred_dino
                elif (getattr(self, 'pred_intermediate_dir', None) is not None
                        and 'dino' in self.PRED_INTERMEDIATE_DINO_MODS):
                    raise PredIntermediateMissing(
                        'no pred NPY for dino at this row')
                else:
                    dino_tokens = row['dino_global']
                data = self._add_dino(
                    data,
                    dino_tokens=dino_tokens,
                    need_loss=False,
                    enable_cfg=enable_cfg,
                    modality_type=modality
                )
            elif modality in ['dinolocal']:
                dino_tokens = row['dino']
                data = self._add_dinolocal(
                    data, 
                    dino_tokens=dino_tokens, 
                    need_loss=False, 
                    enable_cfg=enable_cfg,
                    modality_type=modality
                )
            elif modality in ['clip']:
                clip_tokens = row['clip448']
                data = self._add_clip(
                    data,
                    clip_tokens=clip_tokens,
                    need_loss=False,
                    enable_cfg=enable_cfg,
                    modality_type=modality
                )
            elif modality in ['imagebind']:
                imagebind_tokens = row['imagebind_global']
                data = self._add_imagebind(
                    data,
                    imagebind_tokens=imagebind_tokens,
                    need_loss=False,
                    enable_cfg=enable_cfg,
                    modality_type=modality
                )
            elif modality in ['imagebindlocal']:
                imagebind_tokens = row['imagebind']
                data = self._add_imagebindlocal(
                    data,
                    imagebind_tokens=imagebind_tokens,
                    need_loss=False,
                    enable_cfg=enable_cfg,
                    modality_type=modality
                )
        
        if native_chat:
            if target_modality not in ('det', 'seg', 'grounding'):
                if self.use_instruction and self.use_target_instruction:
                    data = self._add_text(
                        data,
                        self._target_instruction_text_native_chat(target_modality),
                        need_loss=False,
                        enable_cfg=False,
                        modality_type='text',
                    )
                data = self._add_native_chat_assistant_prefix(data)
        elif self.use_instruction and self.use_target_instruction:
            if target_modality == 'det' or target_modality == 'seg' or target_modality == 'grounding':
                pass # separate instruction in add_det() function
            else:
                data = self._add_text(
                    data,
                    self._target_instruction_text(target_modality),
                    need_loss=False,
                    enable_cfg=enable_cfg,
                    modality_type='text',
                )
        # Add target modality at the end with loss=True
        if target_modality == 'caption':
            # Handle text target
            # data = self._add_text(data, row[target_modality], enable_cfg=enable_cfg, need_loss=True, modality_type=target_modality)
            caption_text = row[target_modality]
            if native_chat:
                caption_text = f"{caption_text}</answer>"
            data = self._add_text(data, caption_text, enable_cfg=enable_cfg, need_loss=True, modality_type=target_modality)
        elif target_modality in ['rgb', 'depth', 'normal', 'canny']:
            # Handle image target
            image = pil_img2rgb(Image.open(io.BytesIO(row[target_modality])))
            data = self._add_image(
                data,
                image,
                need_loss=True,
                need_vae=False,
                need_vit=False,
                modality_type=target_modality,
            )
        elif target_modality in ['samseg', 'samedge']:
            # Direct-PNG SAM image target (samseg is RGBA → pil_img2rgb).
            image = pil_img2rgb(Image.open(io.BytesIO(row[self.SAM_IMAGE_COLUMN[target_modality]])))
            data = self._add_image(
                data,
                image,
                need_loss=True,
                need_vae=False,
                need_vit=False,
                modality_type=target_modality,
            )
        elif target_modality == 'cocodet':
            cd_dets, cd_size = self._parse_coco_det(row)
            data = self._add_cocodet(
                data,
                detections=cd_dets,
                image_size=cd_size,
                need_loss=True,
                enable_cfg=enable_cfg,
                modality_type=target_modality,
            )
        elif target_modality == 'det':
            det_seg_instances = row['det_seg']['instances']
            det_seg_raw_size = row['det_seg']['size']
            det_seg_prompts = row['det_seg']['prompt'].split(' .')
            data = self._add_det(
                data, 
                det_seg_instances=det_seg_instances, 
                det_seg_raw_size=det_seg_raw_size, 
                det_seg_prompts=det_seg_prompts,
                need_loss=True, 
                enable_cfg=enable_cfg, 
                modality_type=target_modality
            )
        elif target_modality == 'seg':
            det_seg_instances = row['det_seg']['instances']
            det_seg_raw_size = row['det_seg']['size']
            det_seg_prompts = row['det_seg']['prompt'].split(' .')
            data = self._add_seg(
                data, 
                det_seg_instances=det_seg_instances, 
                det_seg_raw_size=det_seg_raw_size, 
                det_seg_prompts=det_seg_prompts,
                need_loss=True, 
                enable_cfg=enable_cfg, 
                need_vae=False, 
                need_vit=False, 
                modality_type=target_modality
            )
        elif target_modality == 'grounding':
            grounding_instances = row['grounding']
            data = self._add_grounding(
                data, 
                grounding_instances=grounding_instances, 
                need_loss=True, 
                enable_cfg=False,
                modality_type=target_modality
            )
        elif target_modality == 'dino':
            dino_tokens = row['dino_global']
            data = self._add_dino(
                data, 
                dino_tokens=dino_tokens, 
                need_loss=True, 
                enable_cfg=False,
                modality_type=target_modality
            )
        elif target_modality == 'dinolocal':
            dino_tokens = row['dino']
            data = self._add_dinolocal(
                data, 
                dino_tokens=dino_tokens, 
                need_loss=True, 
                enable_cfg=False,
                modality_type=target_modality
            )
        elif target_modality == 'clip':
            clip_tokens = row['clip448']
            data = self._add_clip(
                data,
                clip_tokens=clip_tokens,
                need_loss=True,
                enable_cfg=False,
                modality_type=target_modality
            )
        elif target_modality == 'imagebind':
            imagebind_tokens = row['imagebind_global']
            data = self._add_imagebind(
                data,
                imagebind_tokens=imagebind_tokens,
                need_loss=True,
                enable_cfg=False,
                modality_type=target_modality
            )
        elif target_modality == 'imagebindlocal':
            imagebind_tokens = row['imagebind']
            data = self._add_imagebindlocal(
                data,
                imagebind_tokens=imagebind_tokens,
                need_loss=True,
                enable_cfg=False,
                modality_type=target_modality
            )

        return data


    def _add_det(self, data, det_seg_instances, det_seg_raw_size, det_seg_prompts, need_loss, enable_cfg, modality_type):

        if not need_loss:
            is_all_sampled = False
            num_instances = len(det_seg_instances)
            num_selected = random.randint(1, num_instances)
            # Sample by indices to support containers that aren't proper sequences
            selected_indices = random.sample(range(num_instances), num_selected)
            selected_instances = [det_seg_instances[i] for i in selected_indices]
            is_all_sampled = (num_selected == num_instances)
        else:
            unique_categories = sorted(set(inst.get('category') for inst in det_seg_instances))
            num_categories = len(unique_categories)

            # num_selected_categories = random.randint(1, num_categories)
            num_selected_categories = 1
            selected_categories = random.sample(unique_categories, num_selected_categories)
            
            selected_instances = []
            for inst in det_seg_instances:
                if inst.get('category') in selected_categories:
                    selected_instances.append(inst)

        # Size assumed as [width, height]
        width, height = det_seg_raw_size[0], det_seg_raw_size[1]
        width = max(float(width), 1.0)
        height = max(float(height), 1.0)

        normalized_dets = ''
        # selected_instances = sort_instances_by_distance_to_origin(selected_instances)
        selected_instances = sort_instances_by_raster(selected_instances)

        for inst in selected_instances:
            bbox = inst.get('bbox')
            category = inst.get('category')
            score = inst.get('score', None)

            if bbox is None or len(bbox) != 4:
                continue

            # bbox assumed [x1, y1, x2, y2] in raw pixel coords
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

            # Normalize to [0,1]
            x1_n = x1 / width
            y1_n = y1 / height
            x2_n = x2 / width
            y2_n = y2 / height

            # Clamp
            x1_n = min(max(x1_n, 0.0), 0.999)
            y1_n = min(max(y1_n, 0.0), 0.999)
            x2_n = min(max(x2_n, 0.0), 0.999)
            y2_n = min(max(y2_n, 0.0), 0.999)

            # Quantize: keep 3 digits by *1000 and round
            def q(v, max_digits=3):
                return int(round(v * 10 ** max_digits))

            bbox_q = [q(x1_n), q(y1_n), q(x2_n), q(y2_n)]
            score_q = q(score, max_digits=2)

            coord_types = ['x1', 'y1', 'x2', 'y2']
            coord_tokens = [f'<|{coord_types[i]}_{bbox_q[i]:03d}|>' for i in range(4)]
            score_token = f'<|score_{score_q:02d}|>'
            # bbox_str = f'{"".join(coord_tokens)}{category}{score_token}'
            # bbox_str = f'{"".join(coord_tokens)} || {category} ||{score_token}'
            # bbox_str = f'{"".join(coord_tokens)}{category}'


            bbox_str = f'{"".join(coord_tokens)}'

            # bbox_str = f'{" ".join(coord_tokens)} {category}'
            # bbox_str = f'<|box_start|>{"".join(coord_tokens)}<|box_end|>{category},score={score_q}'

            # bbox_str = f'x1={bbox_q[0]},y1={bbox_q[1]},x2={bbox_q[2]},y2={bbox_q[3]},{category},score={score_q}'
            
            # normalized_dets = normalized_dets + bbox_str + ','
            normalized_dets = normalized_dets + bbox_str

        if need_loss:
            # if len(selected_categories) == len(unique_categories):
            #     instruction_text = "start detect the box of everything"
            # else:
            #     instruction_text = f"start detect the box of {', '.join(selected_categories)}"
            if self._native_chat_serialization():
                instruction_text = f"Detect the box of {', '.join(selected_categories)}."
            elif self._native_aligned_serialization():
                instruction_text = f"Detect the box of {', '.join(selected_categories)}."
            else:
                instruction_text = f"start detect the box of {', '.join(selected_categories)}"
            data = self._add_text(data, instruction_text, need_loss=False, enable_cfg=enable_cfg, modality_type='text')
            if self._native_chat_serialization():
                data = self._add_native_chat_assistant_prefix(data)
            # print("det instruction text:", instruction_text)
        else:
            prefix = "bounding boxes (all): " if is_all_sampled else "bounding boxes (partial): "
            # normalized_dets = prefix + normalized_dets

            if self.use_det_image:
                # det_image = get_det_image(normalized_dets, det_seg_raw_size)
                det_image = get_det_image_new(normalized_dets, det_seg_raw_size)
                data = self._add_image(data, det_image, need_loss=False, enable_cfg=enable_cfg, need_vae=True, need_vit=True, modality_type='rgb')

                
            data = self._add_text(data, prefix, need_loss=False, enable_cfg=enable_cfg, modality_type='text')

        
        data = self._add_text(data, normalized_dets, need_loss=need_loss, enable_cfg=enable_cfg, modality_type=modality_type)
        # print("det normalized_dets:", normalized_dets)

        return data


    def _parse_coco_det(self, row):
        """Parse the `coco_det` JSON string column.

        Returns (detections_list, image_size). On missing/blank/malformed
        input returns ([], None) so callers can treat it as 'no detections'.
        """
        raw = row['coco_det'] if 'coco_det' in row.index else None
        if raw is None:
            return [], None
        try:
            obj = json.loads(raw)
        except (TypeError, ValueError):
            return [], None
        dets = obj.get('detections') or []
        size = obj.get('image_size') or None
        return dets, size

    def _add_cocodet(self, data, detections, image_size, need_loss, enable_cfg, modality_type):
        """Pix2seq-style detection serialization (cocodet).

        Per box: <|x1_q|><|y1_q|><|x2_q|><|y2_q|><|coco_cls_label|>, boxes sorted
        by distance-to-origin. Coords reuse det's quantized coordinate tokens;
        class is a dedicated token aligned to the COCO id. No score, no separators.
        The start/end delimiters are added by the packing layer (modality_type).
        """
        width = max(float(image_size[0]), 1.0) if image_size else 1.0
        height = max(float(image_size[1]), 1.0) if image_size else 1.0

        instances = sort_instances_by_distance_to_origin(detections or [])

        def q(v):
            return min(max(int(round(v * 1000)), 0), 999)

        seq = ''
        for inst in instances:
            bbox = inst.get('bbox')
            if bbox is None or len(bbox) != 4:
                continue
            label = inst.get('label')
            if label is None:
                continue
            label = int(label)
            if label < 0 or label > 90:   # COCO id range; out-of-range → skip
                continue
            x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]),
                              float(bbox[2]), float(bbox[3]))
            x1q, y1q = q(x1 / width), q(y1 / height)
            x2q, y2q = q(x2 / width), q(y2 / height)
            seq += (f'<|x1_{x1q:03d}|><|y1_{y1q:03d}|>'
                    f'<|x2_{x2q:03d}|><|y2_{y2q:03d}|><|coco_cls_{label:02d}|>')

        # Empty sequence is valid (e.g. as a "no objects" condition); the
        # packing layer still wraps it with the cocodet start/end delimiters.
        data = self._add_text(data, seq, need_loss=need_loss,
                              enable_cfg=enable_cfg, modality_type=modality_type)
        return data

    def _add_grounding(self, data, grounding_instances, need_loss, enable_cfg, modality_type):

        selected_grounding = random.choice(grounding_instances)
        phrase = selected_grounding['phrase']
        bbox = selected_grounding['bbox']


        if need_loss:
            if self._native_chat_serialization():
                instruction_text = f"Ground the phrase: {phrase}"
            elif self._native_aligned_serialization():
                instruction_text = f"Ground the phrase: {phrase}"
            else:
                instruction_text = f"[start grounding the phrase] {phrase}"
            data = self._add_text(data, instruction_text, need_loss=False, enable_cfg=enable_cfg, modality_type='text')
            if self._native_chat_serialization():
                data = self._add_native_chat_assistant_prefix(data)
        elif random.random() >= self.grounding_phrase_dropout_prob:
            data = self._add_text(data, phrase, need_loss=False, enable_cfg=enable_cfg, modality_type='text')

        coord_types = ['x1', 'y1', 'x2', 'y2']
        coord_tokens = [f'<|{coord_types[i]}_{bbox[i]:03d}|>' for i in range(4)]
        bbox_str = f'{"".join(coord_tokens)}'
        data = self._add_text(data, bbox_str, need_loss=need_loss, enable_cfg=enable_cfg, modality_type='det')

        return data

    def _add_seg(self, data, det_seg_instances, det_seg_raw_size, det_seg_prompts, need_loss, enable_cfg, need_vae, need_vit, modality_type):
        # Helper: decode COCO RLE to binary mask; fallback to empty mask if not available
        def decode_rle_to_mask(rle_counts, size):
            try:
                rle = {'counts': rle_counts, 'size': [int(size[0]), int(size[1])]}
                mask = mask_util.decode(rle)
                if mask is None:
                    return None
                # pycocotools returns (H,W,1) sometimes; squeeze to (H,W)
                if hasattr(mask, 'shape') and len(mask.shape) == 3:
                    mask = mask[:, :, 0]
                return mask.astype('uint8')
            except Exception:
                try:
                    height, width = int(size[1]), int(size[0])
                    return np.zeros((height, width), dtype='uint8')
                except Exception:
                    return None

        width, height = float(det_seg_raw_size[0]), float(det_seg_raw_size[1])
        width = max(width, 1.0)
        height = max(height, 1.0)

        mask_images = get_mask_image(det_seg_instances, det_seg_raw_size)
        # Build mask(s)
        if not need_loss:
            # Conditioning: select at most 2 categories from mask_images, add category name and mask image
            if len(mask_images) > 0:
                k = random.randint(1, min(2, len(mask_images)))
                selected_categories = random.sample(list(mask_images.keys()), k)
            else:
                selected_categories = []

            if not selected_categories:
                return data

            for category in selected_categories:
                # Use pre-computed merged mask for this category
                pil_mask = mask_images[category]

                # Add category text
                data = self._add_text(
                    data,
                    f"mask category: {category}",
                    need_loss=False,
                    enable_cfg=enable_cfg,
                    modality_type='text',
                )

                data = self._add_image(
                    data,
                    pil_mask,
                    need_loss=False,
                    enable_cfg=enable_cfg,
                    need_vae=need_vae,
                    need_vit=need_vit,
                    modality_type=modality_type,
                )

            return data
        else:
            # Supervision: select one category, add instruction, then add mask image with loss
            if len(mask_images) == 0:
                return data

            # Select one category
            category = random.choice(list(mask_images.keys()))

            # Create instruction text
            if self._native_chat_serialization():
                instruction_text = f"Generate the segmentation mask of {category}."
            elif self._native_aligned_serialization():
                instruction_text = f"Generate the segmentation mask of {category}."
            else:
                instruction_text = f"start segment the mask of {category}"
            
            data = self._add_text(data, instruction_text, need_loss=False, enable_cfg=enable_cfg, modality_type='text')
            if self._native_chat_serialization():
                data = self._add_native_chat_assistant_prefix(data)

            # Use pre-computed mask image for this category
            pil_mask = mask_images[category]

            data = self._add_image(
                data,
                pil_mask,
                need_loss=True,
                enable_cfg=enable_cfg,
                need_vae=False,
                need_vit=False,
                modality_type=modality_type,
            )

            return data


    def _add_dino(self, data, dino_tokens, need_loss, enable_cfg, modality_type):
        text_ids = dino_tokens + self.tokenizer.encode('<|dino_0000|>')[0]
        text_ids = text_ids.tolist()
        data['num_tokens'] += len(text_ids)
        data['text_ids_list'].append(text_ids)
        data['sequence_plan'].append(
            {
                'type': 'text',
                'enable_cfg': int(enable_cfg),
                'loss': int(need_loss),
                'special_token_loss': 0,
                'special_token_label': None,
                'modality_type': modality_type,
            }
        )
        return data

    def _add_dinolocal(self, data, dino_tokens, need_loss, enable_cfg, modality_type):
        text_ids = dino_tokens + self.tokenizer.encode('<|dinolocal_0000|>')[0]
        text_ids = text_ids.tolist()
        data['num_tokens'] += len(text_ids)
        data['text_ids_list'].append(text_ids)
        data['sequence_plan'].append(
            {
                'type': 'text',
                'enable_cfg': int(enable_cfg),
                'loss': int(need_loss),
                'special_token_loss': 0,
                'special_token_label': None,
                'modality_type': modality_type,
            }
        )
        return data

    def _add_clip(self, data, clip_tokens, need_loss, enable_cfg, modality_type):
        text_ids = clip_tokens + self.tokenizer.encode('<|clip_0000|>')[0]
        text_ids = text_ids.tolist()
        data['num_tokens'] += len(text_ids)
        data['text_ids_list'].append(text_ids)
        data['sequence_plan'].append(
            {
                'type': 'text',
                'enable_cfg': int(enable_cfg),
                'loss': int(need_loss),
                'special_token_loss': 0,
                'special_token_label': None,
                'modality_type': modality_type,
            }
        )
        return data

    def _add_imagebind(self, data, imagebind_tokens, need_loss, enable_cfg, modality_type):
        text_ids = imagebind_tokens + self.tokenizer.encode('<|imagebind_0000|>')[0]
        text_ids = text_ids.tolist()
        data['num_tokens'] += len(text_ids)
        data['text_ids_list'].append(text_ids)
        data['sequence_plan'].append(
            {
                'type': 'text',
                'enable_cfg': int(enable_cfg),
                'loss': int(need_loss),
                'special_token_loss': 0,
                'special_token_label': None,
                'modality_type': modality_type,
            }
        )
        return data

    def _add_imagebindlocal(self, data, imagebind_tokens, need_loss, enable_cfg, modality_type):
        text_ids = imagebind_tokens + self.tokenizer.encode('<|imagebindlocal_0000|>')[0]
        text_ids = text_ids.tolist()
        data['num_tokens'] += len(text_ids)
        data['text_ids_list'].append(text_ids)
        data['sequence_plan'].append(
            {
                'type': 'text',
                'enable_cfg': int(enable_cfg),
                'loss': int(need_loss),
                'special_token_loss': 0,
                'special_token_label': None,
                'modality_type': modality_type,
            }
        )
        return data

def get_det_image(normalized_dets, det_seg_raw_size):
    """
    Create an image with bounding boxes drawn on it based on normalized detections.
    
    Args:
        normalized_dets (str): String containing normalized bounding box information
        det_seg_raw_size (list): [width, height] of the image in pixels
    
    Returns:
        PIL.Image: Image with bounding boxes drawn in random colors
    """

    # Extract image dimensions
    width, height = int(det_seg_raw_size[0]), int(det_seg_raw_size[1])
    
    # Create a blank white image
    image = Image.new('RGB', (width, height), color='black')
    draw = ImageDraw.Draw(image)
    
    # Define a palette of random colors for bounding boxes
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (255, 192, 203), # Pink
        (0, 128, 0),    # Dark Green
        (128, 128, 0),  # Olive
        (0, 128, 128),  # Teal
    ]
    
    # Dictionary to map categories to colors
    category_colors = {}
    
    # Parse the normalized_dets string
    if not normalized_dets or normalized_dets.strip() == '':
        return image
    
    # Split by comma to get individual bounding box strings
    bbox_strings = [s.strip() for s in normalized_dets.split(',') if s.strip()]
    
    for i, bbox_str in enumerate(bbox_strings):
        try:
            # Parse the bounding box string format: "<|x1_xxx|><|y1_xxx|><|x2_xxx|><|y2_xxx|> || category ||<|score_xx|>"
            # We need to be more careful with the splitting since category names might contain "||"
            
            # First, find the score part at the end
            score_match = re.search(r'<\|score_\d+\|>$', bbox_str)
            if not score_match:
                continue
            score_part = score_match.group(0)
            
            # Remove the score part to get the rest
            remaining = bbox_str[:-len(score_part)]
            
            # Split by " || " to separate coordinates from category
            if ' || ' not in remaining:
                continue
            coord_part, category = remaining.split(' || ', 1)
            
            # Clean up the category name (remove any trailing "||" that might be left)
            category = category.strip().rstrip('||').strip()
            
            # Extract coordinates from tokens like <|x1_123|><|y1_456|><|x2_789|><|y2_012|>
            coord_pattern = r'<\|([xy][12])_(\d+)\|>'
            coord_matches = re.findall(coord_pattern, coord_part)
            
            if len(coord_matches) != 4:
                continue
                
            # Convert to dictionary for easier access
            coords = {}
            for coord_type, value in coord_matches:
                coords[coord_type] = int(value)
            
            # Convert quantized coordinates back to pixel coordinates
            # The coordinates are quantized by multiplying by 1000, so divide by 1000
            x1 = int((coords['x1'] / 1000.0) * width)
            y1 = int((coords['y1'] / 1000.0) * height)
            x2 = int((coords['x2'] / 1000.0) * width)
            y2 = int((coords['y2'] / 1000.0) * height)
            
            # Clamp coordinates to image bounds
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            # Skip invalid bounding boxes
            if x1 >= x2 or y1 >= y2:
                continue
            
            # Assign color based on category - same category gets same color
            if category not in category_colors:
                # Assign next available color to this category
                color_index = len(category_colors) % len(colors)
                category_colors[category] = colors[color_index]
            color = category_colors[category]
            
            # Draw the bounding box rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            # Draw the bounding box rectangle (filled)
            # draw.rectangle([x1, y1, x2, y2], fill=color)
            
            # Extract score if available
            score_match = re.search(r'<\|score_(\d+)\|>', score_part)
            score = float(score_match.group(1)) / 100.0 if score_match else 0.0
            
            # Draw label with category and score
            label = f"{category}: {score:.2f}"
            
            # Calculate text position (above the bounding box with better spacing)
            text_y = max(5, y1 - 15)
            
            # Draw text background rectangle aligned to left edge of bounding box
            text_bbox = draw.textbbox((x1, text_y), label)
            text_x1, text_y1, text_x2, text_y2 = text_bbox
            # Keep left edge aligned with bounding box, add padding only on right/bottom
            draw.rectangle([x1, text_y1-2, text_x2+4, text_y2+2], fill=color, outline=color)
            
            font = ImageFont.load_default()
            draw.text((x1, text_y), label, fill=(255, 255, 255), font=font)
            
        except Exception as e:
            # Skip malformed bounding box strings
            print(f"Warning: Could not parse bounding box string '{bbox_str}': {e}")
            continue
    
    return image


def get_det_image_new(normalized_dets, det_seg_raw_size):
    """
    Create an image with bounding boxes drawn based on the NEW detections format:
    "<|x1_100|><|y1_100|><|x2_499|><|y2_499|>person,<|x1_700|>...>apple,"

    Categories can include commas, so we must NOT split by comma. We instead
    locate each coordinate token group and take the category text that follows
    it up to the start of the next coordinate group (or end of string).

    Args:
        normalized_dets (str): Detection string in the new format
        det_seg_raw_size (list): [width, height]

    Returns:
        PIL.Image: Image with bounding boxes drawn
    """

    width, height = int(det_seg_raw_size[0]), int(det_seg_raw_size[1])
    image = Image.new('RGB', (width, height), color='black')
    draw = ImageDraw.Draw(image)

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (255, 128, 0), (128, 0, 255), (255, 192, 203),
        (0, 128, 0), (128, 128, 0), (0, 128, 128),
    ]
    category_colors = {}

    if not normalized_dets or normalized_dets.strip() == '':
        return image

    # Find all coordinate token groups and their spans
    group_pattern = re.compile(r"<\|x1_(\d+)\|><\|y1_(\d+)\|><\|x2_(\d+)\|><\|y2_(\d+)\|>")
    matches = list(group_pattern.finditer(normalized_dets))

    for idx, m in enumerate(matches):
        try:
            x1_q = int(m.group(1))
            y1_q = int(m.group(2))
            x2_q = int(m.group(3))
            y2_q = int(m.group(4))

            # Determine category slice: after this match to before next match (or end)
            start_cat = m.end()
            end_cat = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized_dets)
            category = normalized_dets[start_cat:end_cat]
            # Strip a single trailing comma and whitespace
            category = category.rstrip()
            if category.endswith(','):
                category = category[:-1]
            category = category.strip()
            if category == '':
                category = 'object'

            # De-quantize from [0,1000) back to pixel coords
            x1 = int((x1_q / 1000.0) * width)
            y1 = int((y1_q / 1000.0) * height)
            x2 = int((x2_q / 1000.0) * width)
            y2 = int((y2_q / 1000.0) * height)

            # Clamp
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            if x1 >= x2 or y1 >= y2:
                continue

            if category not in category_colors:
                color_index = len(category_colors) % len(colors)
                category_colors[category] = colors[color_index]
            color = category_colors[category]

            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Label (category only; no score in new format)
            label = f"{category}"
            text_y = max(5, y1 - 15)
            text_bbox = draw.textbbox((x1, text_y), label)
            tb_x1, tb_y1, tb_x2, tb_y2 = text_bbox
            draw.rectangle([x1, tb_y1 - 2, tb_x2 + 4, tb_y2 + 2], fill=color, outline=color)
            font = ImageFont.load_default()
            draw.text((x1, text_y), label, fill=(255, 255, 255), font=font)

        except Exception as e:
            print(f"Warning: Could not parse detection entry around index {m.start()}: {e}")
            continue

    return image

def get_mask_image(det_seg_instances, det_seg_raw_size):
    """
    Create mask images for each category with different colors for different instances.
    
    Args:
        det_seg_instances: List of detection/segmentation instances
        det_seg_raw_size: Raw size of the image [width, height]
    
    Returns:
        dict: Dictionary mapping category_name to PIL Image (RGB mode)
    """
    # Helper: decode COCO RLE to binary mask; fallback to empty mask if not available
    def decode_rle_to_mask(rle_counts, size):
        try:
            rle = {'counts': rle_counts, 'size': [int(size[0]), int(size[1])]}
            mask = mask_util.decode(rle)
            if mask is None:
                return None
            # pycocotools returns (H,W,1) sometimes; squeeze to (H,W)
            if hasattr(mask, 'shape') and len(mask.shape) == 3:
                mask = mask[:, :, 0]
            return mask.astype('uint8')
        except Exception:
            try:
                height, width = int(size[1]), int(size[0])
                return np.zeros((height, width), dtype='uint8')
            except Exception:
                return None

    width, height = float(det_seg_raw_size[0]), float(det_seg_raw_size[1])
    width = max(width, 1.0)
    height = max(height, 1.0)

    # Group instances by category
    category_groups = {}
    for inst in det_seg_instances:
        category = inst.get('category', 'object')
        if category not in category_groups:
            category_groups[category] = []
        category_groups[category].append(inst)
    
    # Define colors for different instances (20 colors) - matplotlib colorblind-friendly palette
    colors = [
        (31, 119, 180),   # C1 - Blue
        (174, 199, 232),  # C2 - Light Blue
        (255, 127, 14),   # C3 - Orange
        (255, 187, 120),  # C4 - Light Orange
        (44, 160, 44),    # C5 - Green
        (152, 223, 138),  # C6 - Light Green
        (214, 39, 40),    # C7 - Red
        (255, 152, 150),  # C8 - Light Red
        (148, 103, 189),  # C9 - Purple
        (197, 176, 213),  # C10 - Light Purple
        (140, 86, 75),    # C11 - Brown
        (196, 156, 148),  # C12 - Light Brown
        (227, 119, 194),  # C13 - Pink
        (247, 182, 210),  # C14 - Light Pink
        (127, 127, 127),  # C15 - Gray
        (199, 199, 199),  # C16 - Light Gray
        (188, 189, 34),   # C17 - Olive
        (219, 219, 141),  # C18 - Light Olive
        (23, 190, 207),   # C19 - Cyan
        (158, 218, 229),  # C20 - Light Cyan
    ]
    
    # Create mask images for each category
    category_masks = {}
    for category, instances in category_groups.items():
        # Initialize merged mask for this category
        merged_mask = np.zeros((int(height), int(width), 3), dtype='uint8')
        
        # Randomly shuffle colors for this category to avoid bias
        available_colors = colors.copy()
        random.shuffle(available_colors)
        
        for i, inst in enumerate(instances):
            seg = inst.get('segmentation', {}) or {}
            rle_counts = seg.get('counts', None)
            size = seg.get('size', [int(width), int(height)])

            mask = None
            if rle_counts is not None and size is not None:
                mask = decode_rle_to_mask(rle_counts, size)
            
            if mask is not None:
                # Get color for this instance (randomly selected, no overlap within category)
                color = available_colors[i % len(available_colors)]
                
                # Apply color to mask regions
                for c in range(3):  # RGB channels
                    merged_mask[:, :, c] = np.where(mask > 0, color[c], merged_mask[:, :, c])
        
        # Convert to PIL Image
        category_masks[category] = Image.fromarray(merged_mask, mode='RGB')
    
    return category_masks

def sort_instances_by_distance_to_origin(instances):
    return sorted(instances, key=lambda x: x.get('bbox')[0]**2 + x.get('bbox')[1]**2)

def sort_instances_by_raster(instances):
    return sorted(instances, key=lambda obj: (obj['bbox'][1], obj['bbox'][0]))
