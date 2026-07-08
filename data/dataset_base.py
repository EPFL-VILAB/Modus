# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import math
import random
import json
import os
import copy

import numpy as np
import torch
import torch.nn.functional as F

from .data_utils import (
    get_flattened_position_ids_interpolate,
    get_flattened_position_ids_extrapolate, 
    len2weight,
    patchify, 
    prepare_attention_mask_per_sample, 
)
from .dataset_info import DATASET_INFO, DATASET_REGISTRY
from .interleave_datasets.any2any_dataset import UnifiedAny2AnyIterableDataset
from .transforms import ImageTransform
from .video_utils import FrameSampler


class DataConfig:
    def __init__(
        self, 
        grouped_datasets, 
        text_cond_dropout_prob=0.1,
        vit_cond_dropout_prob=0.4,
        vae_cond_dropout_prob=0.1,
        vae_image_downsample=16,
        max_latent_size=32,
        vit_patch_size=14,
        max_num_patch_per_side=70,
        grounding_phrase_dropout_prob=0.5,
    ):
        self.grouped_datasets = grouped_datasets
        self.text_cond_dropout_prob = text_cond_dropout_prob
        self.vit_cond_dropout_prob = vit_cond_dropout_prob
        self.vit_patch_size = vit_patch_size
        self.max_num_patch_per_side = max_num_patch_per_side
        self.vae_cond_dropout_prob = vae_cond_dropout_prob
        self.vae_image_downsample = vae_image_downsample
        self.max_latent_size = max_latent_size
        self.grounding_phrase_dropout_prob = grounding_phrase_dropout_prob


class PackedDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        data_config, 
        tokenizer, 
        special_tokens,
        modality_registry,
        local_rank, 
        world_size, 
        num_workers,
        expected_num_tokens=32768, 
        max_num_tokens_per_sample=16384,
        max_num_tokens=36864,
        prefer_buffer_before=16384,
        max_buffer_size=50,
        interpolate_pos=False,
        use_flex=False,
        data_status=None,
        data_resume_state=None,
        use_instruction=False,
        use_condition_instruction=True,
        use_target_instruction=True,
        num_condition_modalities=1,
        strict_num_condition_modalities=False,
        timestep_sample=None,
        mode_scale=None,
        timestep_sample_mix_prob=None,
        use_det_image=False,
        visual_gen=True,
        visual_und=True,
        tail_n_rows=None,
        force_condition_modality=None,
        force_condition_modalities=None,
        pred_intermediate_dir=None,
        gt_restricted_dir=None,
        fixed_row_list_path=None,
    ):
        super().__init__()
        self.expected_num_tokens = expected_num_tokens
        self.max_num_tokens_per_sample = max_num_tokens_per_sample
        self.prefer_buffer_before = prefer_buffer_before
        self.max_num_tokens = max_num_tokens
        self.max_buffer_size = max_buffer_size
        self.tokenizer = tokenizer
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.use_flex = use_flex
        self.data_resume_state = data_resume_state
        self.use_instruction = use_instruction
        self.use_condition_instruction = use_condition_instruction
        self.use_target_instruction = use_target_instruction
        self.num_condition_modalities = num_condition_modalities
        self.strict_num_condition_modalities = strict_num_condition_modalities
        self.timestep_sample = timestep_sample
        self.mode_scale = mode_scale
        self.timestep_sample_mix_prob = timestep_sample_mix_prob
        self.use_det_image = use_det_image
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.tail_n_rows = tail_n_rows
        self.pred_intermediate_dir = pred_intermediate_dir
        self.gt_restricted_dir = gt_restricted_dir
        self.fixed_row_list_path = fixed_row_list_path
        self.force_condition_modality = force_condition_modality
        self.force_condition_modalities = force_condition_modalities
        for k, v in special_tokens.items():
            setattr(self, k, v)
        self.modality_registry = modality_registry

        # Derive token mappings and modality IDs from the registry.
        self.start_token_mapping = {
            s.name: s.start_token_key for s in self.modality_registry.specs
        }
        self.end_token_mapping = {
            s.name: s.end_token_key for s in self.modality_registry.specs
        }
        self.modality_to_id = self.modality_registry.name_to_id()
        self.id_to_modality = self.modality_registry.id_to_name()

        self.data_config = data_config
        self.total_dataset_samples = 0
        self._seen_parquet_files = set()
        self._epoch_tracked_groups = set()
        grouped_datasets, is_mandatory, grouped_weights = self.build_datasets(
            data_config.grouped_datasets, data_status
        )
        self.grouped_datasets = grouped_datasets
        self.dataset_iters = [iter(dataset) for dataset in grouped_datasets]
        self.is_mandatory = is_mandatory
        self.grouped_weights = grouped_weights
        self.interpolate_pos = interpolate_pos
        if self.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate
        self._vit_size_warning_emitted = False
        self._latent_size_warning_emitted = False

    def _fit_vit_image_tensor(self, image_tensor, modality_type):
        patch_size = self.data_config.vit_patch_size
        max_num_patch_per_side = self.data_config.max_num_patch_per_side
        max_vit_side = patch_size * max_num_patch_per_side
        _, img_h, img_w = image_tensor.shape

        def _align_down_to_patch_multiple(size):
            return max(patch_size, (size // patch_size) * patch_size)

        needs_resize = img_h > max_vit_side or img_w > max_vit_side
        if needs_resize:
            scale = min(max_vit_side / img_h, max_vit_side / img_w)
            resized_h = _align_down_to_patch_multiple(int(math.floor(img_h * scale)))
            resized_w = _align_down_to_patch_multiple(int(math.floor(img_w * scale)))
            resized_h = min(resized_h, max_vit_side)
            resized_w = min(resized_w, max_vit_side)
        else:
            resized_h = _align_down_to_patch_multiple(img_h)
            resized_w = _align_down_to_patch_multiple(img_w)

        if resized_h == img_h and resized_w == img_w:
            return image_tensor

        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(resized_h, resized_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        if not self._vit_size_warning_emitted:
            print(
                "[JanusFlow][ViTClamp] resized visual-understanding tensor "
                f"for modality={modality_type} from {img_h}x{img_w} to {resized_h}x{resized_w} "
                f"(patch_size={patch_size}, max_num_patch_per_side={max_num_patch_per_side})",
                flush=True,
            )
            self._vit_size_warning_emitted = True

        return image_tensor

    def _fit_vae_image_tensor(self, image_tensor, modality_type):
        downsample = self.data_config.vae_image_downsample
        max_latent_size = self.data_config.max_latent_size
        max_vae_side = downsample * max_latent_size
        _, img_h, img_w = image_tensor.shape

        def _align_down_to_latent_multiple(size):
            return max(downsample, (size // downsample) * downsample)

        needs_resize = img_h > max_vae_side or img_w > max_vae_side
        if needs_resize:
            scale = min(max_vae_side / img_h, max_vae_side / img_w)
            resized_h = _align_down_to_latent_multiple(int(math.floor(img_h * scale)))
            resized_w = _align_down_to_latent_multiple(int(math.floor(img_w * scale)))
            resized_h = min(resized_h, max_vae_side)
            resized_w = min(resized_w, max_vae_side)
        else:
            resized_h = _align_down_to_latent_multiple(img_h)
            resized_w = _align_down_to_latent_multiple(img_w)

        if resized_h == img_h and resized_w == img_w:
            return image_tensor

        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(resized_h, resized_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        if not self._latent_size_warning_emitted:
            print(
                "[JanusFlow][LatentClamp] resized visual-generation tensor "
                f"for modality={modality_type} from {img_h}x{img_w} to {resized_h}x{resized_w} "
                f"(vae_downsample={downsample}, max_latent_size={max_latent_size})",
                flush=True,
            )
            self._latent_size_warning_emitted = True

        return image_tensor


    def build_datasets(self, datasets_metainfo, data_status):
        datasets = []
        is_mandatory = []
        grouped_weights = []
        for grouped_dataset_name, dataset_args in datasets_metainfo.items():
            is_mandatory.append(dataset_args.pop('is_mandatory', False))
            grouped_weights.append(dataset_args.pop('weight', 0.0))

            if 'frame_sampler_args' in dataset_args.keys():
                frame_sampler = FrameSampler(**dataset_args.pop('frame_sampler_args'))
                dataset_args['frame_sampler'] = frame_sampler
            if 'image_transform_args' in dataset_args.keys():
                transform = ImageTransform(**dataset_args.pop('image_transform_args'))
                dataset_args['transform'] = transform
            if 'vit_image_transform_args' in dataset_args.keys():
                vit_transform = ImageTransform(**dataset_args.pop('vit_image_transform_args'))
                dataset_args['vit_transform'] = vit_transform

            assert 'dataset_names' in dataset_args.keys()
            dataset_names = dataset_args.pop('dataset_names')
            dataset_args['data_dir_list'] = []
            for item in dataset_names:
                if self.local_rank == 0:
                    print(f'Preparing Dataset {grouped_dataset_name}/{item}')
                meta_info = DATASET_INFO[grouped_dataset_name][item]
                dataset_args['data_dir_list'].append(meta_info['data_dir'])

                if "parquet_info_path" in meta_info.keys():
                    if 'parquet_info' not in dataset_args.keys():
                        dataset_args['parquet_info'] = {}
                    info_path = meta_info['parquet_info_path']
                    if not os.path.exists(info_path):
                        # Fall back to the copy bundled in the repo. The configured
                        # path lives under datasets/ which may be a symlink that
                        # isn't present on a fresh clone / a different cluster.
                        bundled = os.path.join(
                            os.path.dirname(__file__), 'bundled_parquet_info',
                            os.path.basename(info_path))
                        if os.path.exists(bundled):
                            info_path = bundled
                    with open(info_path, 'r') as f:
                        parquet_info = json.load(f)
                    dataset_args['parquet_info'].update(parquet_info)
                    self._epoch_tracked_groups.add(grouped_dataset_name)
                    for path, info in parquet_info.items():
                        if path not in self._seen_parquet_files:
                            self._seen_parquet_files.add(path)
                            self.total_dataset_samples += info.get('num_rows', 0)

                if 'json_dir' in meta_info.keys():
                    # parquet/tar with json
                    if 'json_dir_list' not in dataset_args.keys():
                        dataset_args['json_dir_list'] = [meta_info['json_dir']]
                    else:
                        dataset_args['json_dir_list'].append(meta_info['json_dir'])

                if 'jsonl_path' in meta_info.keys():
                    # jsonl with jpeg
                    if 'jsonl_path_list' not in dataset_args.keys():
                        dataset_args['jsonl_path_list'] = [meta_info['jsonl_path']]
                    else:
                        dataset_args['jsonl_path_list'].append(meta_info['jsonl_path'])

            resume_data_status = dataset_args.pop('resume_data_status', True)
            if data_status is not None and grouped_dataset_name in data_status.keys() and resume_data_status:
                data_status_per_group = data_status[grouped_dataset_name]
            else:
                data_status_per_group = None
            # Validation hooks (force_condition_modality, tail_n_rows) only make
            # sense for any2any parquet datasets; passing them to e.g. vlm_sft
            # (SftJSONLIterableDataset) would raise TypeError on unknown kwargs.
            extra_args = {}
            if issubclass(DATASET_REGISTRY[grouped_dataset_name], UnifiedAny2AnyIterableDataset):
                if self.tail_n_rows is not None:
                    extra_args['tail_n_rows'] = self.tail_n_rows
                if self.force_condition_modality is not None:
                    extra_args['force_condition_modality'] = self.force_condition_modality
                if self.force_condition_modalities is not None:
                    extra_args['force_condition_modalities'] = self.force_condition_modalities
                if self.pred_intermediate_dir is not None:
                    extra_args['pred_intermediate_dir'] = self.pred_intermediate_dir
                if self.gt_restricted_dir is not None:
                    extra_args['gt_restricted_dir'] = self.gt_restricted_dir
                if self.fixed_row_list_path is not None:
                    extra_args['fixed_row_list_path'] = self.fixed_row_list_path
            dataset = DATASET_REGISTRY[grouped_dataset_name](
                dataset_name=grouped_dataset_name,
                tokenizer=self.tokenizer,
                local_rank=self.local_rank,
                world_size=self.world_size,
                num_workers=self.num_workers,
                data_status=data_status_per_group,
                use_instruction=self.use_instruction,
                use_condition_instruction=self.use_condition_instruction,
                use_target_instruction=self.use_target_instruction,
                num_condition_modalities=self.num_condition_modalities,
                strict_num_condition_modalities=self.strict_num_condition_modalities,
                use_det_image=self.use_det_image,
                modality_registry=self.modality_registry,
                **extra_args,
                **dataset_args,
            )
            dataset.grounding_phrase_dropout_prob = getattr(
                self.data_config, 'grounding_phrase_dropout_prob', 0.0
            )
            datasets.append(dataset)

        return datasets, is_mandatory, grouped_weights

    def set_epoch(self, seed):
        for dataset in self.grouped_datasets:
            dataset.set_epoch(seed)

    def set_sequence_status(self):
        sequence_status = dict(
            curr                        = 0,
            sample_lens                 = list(),
            packed_position_ids         = list(),
            nested_attention_masks      = list(),
            split_lens                  = list(),
            attn_modes                  = list(),
            packed_text_ids             = list(), 
            packed_text_indexes         = list(),
            packed_label_ids            = list(),
            ce_loss_indexes             = list(),
            ce_loss_weights             = list(),
            ce_loss_modality_ids        = list(),  # Track modality ID for each CE loss token
            vae_image_tensors           = list(), 
            vae_image_modality_types    = list(),
            packed_latent_position_ids  = list(),
            vae_latent_shapes           = list(), 
            packed_vae_token_indexes    = list(), 
            packed_timesteps            = list(), 
            mse_loss_indexes            = list(),
            mse_loss_modality_ids       = list(),  # Track modality ID for each MSE loss token
            mse_loss_route_ids          = list(),  # Route tag for rgb target monitoring
            mse_loss_image_ids          = list(),  # Per-token image id for per-sample MSE logging
            mse_loss_timesteps          = list(),  # Per-token sampled timestep for MSE tokens
            packed_vit_tokens           = list(),
            vit_token_seqlens           = list(),
            vit_image_modality_types    = list(),
            packed_vit_position_ids     = list(),
            packed_vit_token_indexes    = list(),
            # (token_h, token_w) per ViT-input image; consumed by model_wrapper.forward
            # to build 2D rope_image_info for the ViT-input image segment, matching the
            # pretrained Hunyuan convention (build_batch_rope_image_info, cond_vit_image).
            vit_image_token_shapes      = list(),
            target_modality_types       = list(),
            next_mse_image_id           = 0,
        )
        return sequence_status

    def to_tensor(self, sequence_status):
        data = dict(
            sequence_length=sum(sequence_status['sample_lens']),
            num_samples=len(sequence_status['sample_lens']),
            sample_lens=sequence_status['sample_lens'],
            target_modality_types=sequence_status['target_modality_types'],
            vae_image_modality_types=sequence_status['vae_image_modality_types'],
            packed_text_ids=torch.tensor(sequence_status['packed_text_ids']),
            packed_text_indexes=torch.tensor(sequence_status['packed_text_indexes']),
            packed_position_ids=torch.tensor(sequence_status['packed_position_ids']),
        )
        if not self.use_flex:
            data['nested_attention_masks'] = sequence_status['nested_attention_masks']
        else:
            sequence_len = data['sequence_length']
            pad_len = self.max_num_tokens - sequence_len
            data['split_lens'] = sequence_status['split_lens'] + [pad_len]
            data['attn_modes'] = sequence_status['attn_modes'] + ['causal']
            data['sample_lens'] += [pad_len]

        # if the model has a convnet vae (e.g., as visual tokenizer)
        if len(sequence_status['vae_image_tensors']) > 0:
            image_tensors = sequence_status.pop('vae_image_tensors')
            image_sizes = [item.shape for item in image_tensors]
            max_image_size = [max(item) for item in list(zip(*image_sizes))]
            padded_images = torch.zeros(size=(len(image_tensors), *max_image_size))
            for i, image_tensor in enumerate(image_tensors):
                padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

            data['padded_images'] = padded_images
            data['patchified_vae_latent_shapes'] = sequence_status['vae_latent_shapes']
            data['packed_latent_position_ids'] = torch.cat(sequence_status['packed_latent_position_ids'], dim=0)
            data['packed_vae_token_indexes'] = torch.tensor(sequence_status['packed_vae_token_indexes'])

        # if the model has a vit (e.g., as visual tokenizer)
        if len(sequence_status['packed_vit_tokens']) > 0:
            data['packed_vit_tokens'] = torch.cat(sequence_status['packed_vit_tokens'], dim=0)
            data['packed_vit_position_ids'] = torch.cat(sequence_status['packed_vit_position_ids'], dim=0)
            data['packed_vit_token_indexes'] = torch.tensor(sequence_status['packed_vit_token_indexes'])
            data['vit_image_token_shapes'] = list(sequence_status['vit_image_token_shapes'])
            data['vit_token_seqlens'] = torch.tensor(sequence_status['vit_token_seqlens'])
            data['vit_image_modality_types'] = sequence_status['vit_image_modality_types']

        # if the model is required to perform visual generation
        if len(sequence_status['packed_timesteps']) > 0:
            data['packed_timesteps'] = torch.tensor(sequence_status['packed_timesteps'])
            data['mse_loss_indexes'] = torch.tensor(sequence_status['mse_loss_indexes'], dtype=torch.long)
            data['mse_loss_modality_ids'] = torch.tensor(sequence_status['mse_loss_modality_ids'], dtype=torch.long)
            data['mse_loss_route_ids'] = torch.tensor(sequence_status['mse_loss_route_ids'], dtype=torch.long)
            data['mse_loss_image_ids'] = torch.tensor(sequence_status['mse_loss_image_ids'], dtype=torch.long)
            data['mse_loss_timesteps'] = torch.tensor(sequence_status['mse_loss_timesteps'])

        # if the model is required to perform text generation
        if len(sequence_status['packed_label_ids']) > 0:
            data['packed_label_ids'] = torch.tensor(sequence_status['packed_label_ids'], dtype=torch.long)
            data['ce_loss_indexes'] = torch.tensor(sequence_status['ce_loss_indexes'], dtype=torch.long)
            data['ce_loss_weights'] = torch.tensor(sequence_status['ce_loss_weights'])
            data['ce_loss_modality_ids'] = torch.tensor(sequence_status['ce_loss_modality_ids'], dtype=torch.long)

        return data

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        total_weights = sum(self.grouped_weights)
        assert total_weights > 0.0
        group_cumprobs = [sum(self.grouped_weights[:i + 1]) / total_weights 
                          for i in range(len(self.grouped_weights))]
        sequence_status = self.set_sequence_status()
        batch_data_indexes = []

        buffer = []
        if self.data_resume_state is not None:
            worker_resume_state = self.data_resume_state.get(worker_id, self.data_resume_state)
            buffer = copy.deepcopy(worker_resume_state.get("buffer", []))
            py_state = worker_resume_state.get("python_random_state")
            np_state = worker_resume_state.get("numpy_random_state")
            torch_state = worker_resume_state.get("torch_random_state")
            if py_state is not None:
                random.setstate(py_state)
            if np_state is not None:
                np.random.set_state(np_state)
            if torch_state is not None:
                torch.random.set_rng_state(torch_state)
            # Consume once. Future epochs in the same process should proceed normally.
            self.data_resume_state = None
        while True:
            # Ensure at least one sample from each group unless explicitly disabled.
            # Disabling this can drastically reduce packed sequence length and
            # avoid pathological long-sequence attention OOM/NaN behavior.
            _disable_mandatory = os.environ.get("HUNYUAN_DISABLE_MANDATORY_WARMSTART", "0") == "1"
            if sequence_status['curr'] == 0 and (not _disable_mandatory):
                _max_mandatory_trials = int(os.environ.get("HUNYUAN_MANDATORY_MAX_TRIALS", "2000"))
                for group_index, group_iter in enumerate(self.dataset_iters):
                    if self.is_mandatory[group_index]:
                        _trials = 0
                        while True:
                            sample = next(group_iter)
                            # Tentative pack on a cloned status: this gives the
                            # true token expansion for visual branches.
                            trial_status = self.pack_sequence(
                                sample, self._clone_sequence_status(sequence_status)
                            )
                            num_tokens = trial_status['curr'] - sequence_status['curr']
                            if (
                                num_tokens < self.max_num_tokens_per_sample
                                and trial_status['curr'] <= self.max_num_tokens
                            ):
                                sequence_status = trial_status
                                batch_data_indexes.append(sample['data_indexes'])
                                break
                            else:
                                _trials += 1
                                if _max_mandatory_trials > 0 and _trials >= _max_mandatory_trials:
                                    print(
                                        f"[DataPack] mandatory group {group_index} exceeded "
                                        f"{_max_mandatory_trials} trials under token constraints "
                                        f"(max_per_sample={self.max_num_tokens_per_sample}, max_total={self.max_num_tokens}); "
                                        "skipping mandatory warmstart for this group in this batch"
                                    )
                                    break
                                continue

            if sequence_status['curr'] < self.prefer_buffer_before and len(buffer) > 0:
                sample = buffer.pop(0)
                sample_from_buffer = True
            else:
                # sample normally across all groups
                n = random.random()
                group_index = 0
                for i, cumprob in enumerate(group_cumprobs):
                    if n < cumprob:
                        group_index = i
                        break
                sample = next(self.dataset_iters[group_index])
                sample_from_buffer = False

            trial_status = self.pack_sequence(
                sample, self._clone_sequence_status(sequence_status)
            )
            num_tokens = trial_status['curr'] - sequence_status['curr']
            if num_tokens > self.max_num_tokens_per_sample:
                # print(f"skip a sample with length {num_tokens}")
                continue

            if trial_status['curr'] > self.max_num_tokens:
                if len(buffer) < self.max_buffer_size and not sample_from_buffer:
                    buffer.append(sample)
                else:
                    if sequence_status['curr'] > 0:
                        data = self.to_tensor(sequence_status)
                        data['batch_data_indexes'] = batch_data_indexes
                        data['num_epoch_samples'] = sum(
                            1 for idx in batch_data_indexes
                            if idx["dataset_name"] in self._epoch_tracked_groups
                        )
                        data['data_resume_state'] = self._build_resume_state(buffer, worker_id)
                        yield data
                    sequence_status = self.set_sequence_status()
                    batch_data_indexes = []
                continue

            sequence_status = trial_status
            batch_data_indexes.append(sample['data_indexes'])

            if sequence_status['curr'] >= self.expected_num_tokens:
                data = self.to_tensor(sequence_status)
                data['batch_data_indexes'] = batch_data_indexes
                data['num_epoch_samples'] = sum(
                    1 for idx in batch_data_indexes
                    if idx["dataset_name"] in self._epoch_tracked_groups
                )
                data['data_resume_state'] = self._build_resume_state(buffer, worker_id)
                yield data
                sequence_status = self.set_sequence_status()
                batch_data_indexes = []

    def _build_resume_state(self, buffer, worker_id):
        """Capture the iterable packer's next-step resume state.

        This stores only CPU-side iterator state. Sub-dataset row cursors are
        still tracked separately via ``data_status`` in the training loop.
        """
        return {
            "worker_id": worker_id,
            "version": 1,
            "buffer": copy.deepcopy(buffer),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state().cpu(),
        }

    def _clone_sequence_status(self, sequence_status):
        """Clone mutable sequence status containers for tentative packing."""
        cloned = {}
        for k, v in sequence_status.items():
            if isinstance(v, list):
                cloned[k] = v.copy()
            else:
                cloned[k] = v
        return cloned

    def _sample_timestep(self):
        """Sample a final timestep value in (0, 1] for one image chunk.

        The sampling strategy is controlled by ``self.timestep_sample``:

        * ``logit_norm`` – logistic-distribution sample (sigmoid of a normal).
        * ``mode``       – uniform sample optionally warped by a cosine mode
                           schedule controlled by ``self.mode_scale``.
        * ``linear``     – inverse-CDF sample giving P(t)=2t (higher density
                           near t=1).
        * ``pure_noise`` – constant ``1.0`` (pure-noise target).
        * ``mix``        – with probability ``timestep_sample_mix_prob`` draw a
                           uniform sample, otherwise ``1.0``.

        Returns a ``float`` ready to be used as the flow-matching timestep.
        """
        eps = 1e-6
        strategy = self.timestep_sample

        if strategy == 'logit_norm':
            # Sigmoid of a standard-normal sample → logistic distribution on (0,1).
            t = 1.0 / (1.0 + math.exp(-np.random.randn()))
        elif strategy == 'mode':
            t = float(np.random.rand())
            if self.mode_scale != 0.0:
                t = (
                    1.0 - t
                    - self.mode_scale
                    * (math.cos(math.pi * t / 2) ** 2 - 1.0 + t)
                )
            t = max(eps, min(1.0 - eps, t))
        elif strategy == 'linear':
            t = math.sqrt(float(np.random.rand()))
            t = max(eps, min(1.0 - eps, t))
        elif strategy == 'pure_noise':
            t = 1.0
        elif strategy == 'mix':
            if np.random.rand() < self.timestep_sample_mix_prob:
                t = float(np.random.rand())
            else:
                t = 1.0
        else:
            raise ValueError(f"Unknown timestep_sample strategy: {strategy}")

        return t

    def _native_token_id(self, token):
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0:
            raise ValueError(f"Native Hunyuan token is not in tokenizer: {token}")
        return int(token_id)

    def pack_sequence(self, sample, sequence_status):
        image_tensor_list = list(sample['image_tensor_list'])
        text_ids_list = list(sample['text_ids_list'])
        sequence_plan = sample['sequence_plan']

        split_lens, attn_modes = list(), list()
        curr = sequence_status['curr']
        curr_rope_id = 0
        sample_lens = 0
        last_loss_modality_type = None

        for item in sequence_plan:
            split_start = item.get('split_start', True)
            if split_start:
                curr_split_len = 0

            if item['type'] == 'raw_text':
                text_ids = text_ids_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.text_cond_dropout_prob:
                    continue
                if item['loss'] == 1:
                    raise NotImplementedError("raw_text sequence items are for no-loss chat scaffolding only")

                sequence_status['packed_text_ids'].extend(text_ids)
                sequence_status['packed_text_indexes'].extend(range(curr, curr + len(text_ids)))
                curr += len(text_ids)
                curr_split_len += len(text_ids)

                attn_modes.append("causal")
                sequence_status['packed_position_ids'].extend(range(curr_rope_id, curr_rope_id + curr_split_len))
                curr_rope_id += curr_split_len

            elif item['type'] == 'text':
                start_token_attr = self.start_token_mapping[item['modality_type']]
                end_token_attr = self.end_token_mapping[item['modality_type']]
                text_ids = text_ids_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.text_cond_dropout_prob:
                    continue


                shifted_text_ids = [getattr(self, start_token_attr)] + text_ids
                sequence_status['packed_text_ids'].extend(shifted_text_ids)
                sequence_status['packed_text_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                if item['loss'] == 1:
                    last_loss_modality_type = item['modality_type']
                    sequence_status['ce_loss_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                    sequence_status['ce_loss_weights'].extend(
                        [len2weight(len(shifted_text_ids))] * len(shifted_text_ids)
                    )
                    sequence_status['ce_loss_modality_ids'].extend(
                        [self.modality_to_id[item['modality_type']]] * len(shifted_text_ids)
                    )
                    sequence_status['packed_label_ids'].extend(text_ids + [getattr(self, end_token_attr)])
                curr += len(shifted_text_ids)
                curr_split_len += len(shifted_text_ids)

                # add a <|im_end|> token
                sequence_status['packed_text_ids'].append(getattr(self, end_token_attr))
                sequence_status['packed_text_indexes'].append(curr)
                if item['special_token_loss'] == 1: # <|im_end|> may have loss
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                # update sequence status
                attn_modes.append("causal")
                sequence_status['packed_position_ids'].extend(range(curr_rope_id, curr_rope_id + curr_split_len))
                curr_rope_id += curr_split_len

            elif item['type'] == 'vit_image':
                start_token_attr = self.start_token_mapping[item['modality_type']]
                end_token_attr = self.end_token_mapping[item['modality_type']]
                if not self.visual_und:
                    curr_rope_id += 1
                    continue
                image_tensor = image_tensor_list.pop(0)
                # Config-driven conditioning: allow modalities to opt out of ViT conditioning in BOTH train/infer.
                # Only applies to conditioning branches (loss==0). Target branches are kept intact.
                if self.modality_registry is not None and int(item.get('loss', 0)) == 0:
                    try:
                        spec = self.modality_registry.get(item['modality_type'])
                        if not spec.represent_vit:
                            curr_rope_id += 1
                            continue
                    except Exception:
                        # Unknown modality -> keep legacy behavior
                        pass
                if item['enable_cfg'] == 1 and random.random() < self.data_config.vit_cond_dropout_prob:
                    curr_rope_id += 1
                    continue

                # add a <|startofimage|> token

                sequence_status['packed_text_ids'].append(getattr(self, start_token_attr))
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                # preprocess image
                image_tensor = self._fit_vit_image_tensor(image_tensor, item['modality_type'])
                vit_tokens = patchify(image_tensor, self.data_config.vit_patch_size)
                num_img_tokens = vit_tokens.shape[0]
                sequence_status['packed_vit_token_indexes'].extend(range(curr, curr + num_img_tokens))
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                sequence_status['packed_vit_tokens'].append(vit_tokens)
                sequence_status['vit_token_seqlens'].append(num_img_tokens)
                sequence_status['vit_image_modality_types'].append(item['modality_type'])
                sequence_status['vit_image_token_shapes'].append((
                    image_tensor.size(1) // self.data_config.vit_patch_size,
                    image_tensor.size(2) // self.data_config.vit_patch_size,
                ))
                vit_position_ids = self.get_flattened_position_ids(
                    image_tensor.size(1), image_tensor.size(2),
                    self.data_config.vit_patch_size,
                    max_num_patches_per_side=self.data_config.max_num_patch_per_side
                )
                max_vit_position_id = self.data_config.max_num_patch_per_side ** 2 - 1
                if vit_position_ids.numel() > 0 and int(vit_position_ids.max().item()) > max_vit_position_id:
                    raise ValueError(
                        "ViT position id out of range after transform: "
                        f"modality={item['modality_type']}, "
                        f"image_hw={tuple(image_tensor.shape[-2:])}, "
                        f"patch_size={self.data_config.vit_patch_size}, "
                        f"max_num_patch_per_side={self.data_config.max_num_patch_per_side}, "
                        f"max_position_id={int(vit_position_ids.max().item())}, "
                        f"allowed_max={max_vit_position_id}"
                    )
                sequence_status['packed_vit_position_ids'].append(vit_position_ids)
                # import pdb; pdb.set_trace()

                # add a <|endofimage|> token
                sequence_status['packed_text_ids'].append(getattr(self, end_token_attr))
                sequence_status['packed_text_indexes'].append(curr)
                if item['special_token_loss'] == 1: # <|endofimage|> may have loss
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                # update sequence status
                attn_modes.append("full")
                sequence_status['packed_position_ids'].extend([curr_rope_id] * curr_split_len)
                curr_rope_id += 1

            elif item['type'] == 'vae_image':
                start_token_attr = self.start_token_mapping[item['modality_type']]
                end_token_attr = self.end_token_mapping[item['modality_type']]
                if not self.visual_gen:
                    curr_rope_id += 1
                    continue
                image_tensor = image_tensor_list.pop(0)
                # Config-driven conditioning: allow modalities to opt out of VAE conditioning in BOTH train/infer.
                # Only applies to conditioning branches (loss==0). Target branches are kept intact.
                if self.modality_registry is not None and int(item.get('loss', 0)) == 0:
                    try:
                        spec = self.modality_registry.get(item['modality_type'])
                        if not spec.represent_vae:
                            curr_rope_id += 1
                            continue
                    except Exception:
                        # Unknown modality -> keep legacy behavior
                        pass
                if item['enable_cfg'] == 1 and random.random() < self.data_config.vae_cond_dropout_prob:
                    # FIXME fix vae dropout in video2video setting.
                    curr_rope_id += 1
                    continue

                # add a <|startofimage|> token

                sequence_status['packed_text_ids'].append(getattr(self, start_token_attr))
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                # preprocess image
                image_tensor = self._fit_vae_image_tensor(image_tensor, item['modality_type'])
                sequence_status['vae_image_tensors'].append(image_tensor)
                sequence_status['vae_image_modality_types'].append(item['modality_type'])
                latent_position_ids = self.get_flattened_position_ids(
                    image_tensor.size(1), image_tensor.size(2),
                    self.data_config.vae_image_downsample,
                    max_num_patches_per_side=self.data_config.max_latent_size
                )
                max_latent_position_id = self.data_config.max_latent_size ** 2 - 1
                if (
                    latent_position_ids.numel() > 0
                    and int(latent_position_ids.max().item()) > max_latent_position_id
                ):
                    raise ValueError(
                        "Latent position id out of range after transform: "
                        f"modality={item['modality_type']}, "
                        f"image_hw={tuple(image_tensor.shape[-2:])}, "
                        f"vae_downsample={self.data_config.vae_image_downsample}, "
                        f"max_latent_size={self.data_config.max_latent_size}, "
                        f"max_position_id={int(latent_position_ids.max().item())}, "
                        f"allowed_max={max_latent_position_id}"
                    )
                sequence_status['packed_latent_position_ids'].append(latent_position_ids)
                H, W = image_tensor.shape[1:]
                h = H // self.data_config.vae_image_downsample
                w = W // self.data_config.vae_image_downsample
                sequence_status['vae_latent_shapes'].append((h, w))

                num_img_tokens = w * h
                sequence_status['packed_vae_token_indexes'].extend(range(curr, curr + num_img_tokens))
                if item['loss'] == 1:
                    last_loss_modality_type = item['modality_type']
                    mse_image_id = sequence_status['next_mse_image_id']
                    sequence_status['next_mse_image_id'] += 1
                    sequence_status['mse_loss_indexes'].extend(range(curr, curr + num_img_tokens))
                    sequence_status['mse_loss_modality_ids'].extend(
                        [self.modality_to_id[item['modality_type']]] * num_img_tokens
                    )
                    if split_start:
                        timestep = self._sample_timestep()
                    route_name = sample.get('rgb_loss_route', None)
                    if item['modality_type'] == 'rgb' and route_name is not None:
                        route_to_id = {'caption2rgb': 1, 'grounding2rgb': 2, 'dinolocal2rgb': 3}
                        route_id = route_to_id.get(route_name, 0)
                    else:
                        route_id = 0
                    sequence_status['mse_loss_route_ids'].extend([route_id] * num_img_tokens)
                    sequence_status['mse_loss_image_ids'].extend([mse_image_id] * num_img_tokens)
                    sequence_status['mse_loss_timesteps'].extend([timestep] * num_img_tokens)
                else:
                    timestep = 0.0

                sequence_status['packed_timesteps'].extend([timestep] * num_img_tokens)
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                # add a <|endofimage|> token
                sequence_status['packed_text_ids'].append(getattr(self, end_token_attr))
                sequence_status['packed_text_indexes'].append(curr)
                # <|endofimage|> may have loss
                if item['special_token_loss'] == 1:
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                # update sequence status
                if split_start:
                    if item['loss'] == 1 and 'frame_delta' not in item.keys():
                        attn_modes.append("noise")
                    else:
                        attn_modes.append("full")
                sequence_status['packed_position_ids'].extend([curr_rope_id] * (num_img_tokens + 2))
                if 'frame_delta' in item.keys():
                    curr_rope_id += item['frame_delta']
                elif item['loss'] == 0:
                    curr_rope_id += 1

            elif item['type'] == 'native_gen_image':
                if not self.visual_gen:
                    curr_rope_id += 1
                    continue
                image_tensor = image_tensor_list.pop(0)

                prefix_ids = [
                    self._native_token_id('<boi>'),
                    self._native_token_id(f"<img_size_{int(item.get('base_size', 1024))}>"),
                    self._native_token_id(f"<img_ratio_{int(item.get('ratio_idx', 16))}>"),
                    self._native_token_id('<timestep>'),
                ]
                eoi_id = self._native_token_id('<eoi>')

                sequence_status['packed_text_ids'].extend(prefix_ids)
                sequence_status['packed_text_indexes'].extend(range(curr, curr + len(prefix_ids)))
                curr += len(prefix_ids)
                curr_split_len += len(prefix_ids)

                image_tensor = self._fit_vae_image_tensor(image_tensor, item['modality_type'])
                sequence_status['vae_image_tensors'].append(image_tensor)
                sequence_status['vae_image_modality_types'].append(item['modality_type'])
                latent_position_ids = self.get_flattened_position_ids(
                    image_tensor.size(1), image_tensor.size(2),
                    self.data_config.vae_image_downsample,
                    max_num_patches_per_side=self.data_config.max_latent_size
                )
                sequence_status['packed_latent_position_ids'].append(latent_position_ids)
                H, W = image_tensor.shape[1:]
                h = H // self.data_config.vae_image_downsample
                w = W // self.data_config.vae_image_downsample
                sequence_status['vae_latent_shapes'].append((h, w))

                num_img_tokens = w * h
                sequence_status['packed_vae_token_indexes'].extend(range(curr, curr + num_img_tokens))
                last_loss_modality_type = item['modality_type']
                mse_image_id = sequence_status['next_mse_image_id']
                sequence_status['next_mse_image_id'] += 1
                sequence_status['mse_loss_indexes'].extend(range(curr, curr + num_img_tokens))
                sequence_status['mse_loss_modality_ids'].extend(
                    [self.modality_to_id[item['modality_type']]] * num_img_tokens
                )
                timestep = self._sample_timestep()
                route_name = sample.get('rgb_loss_route', None)
                if item['modality_type'] == 'rgb' and route_name is not None:
                    route_to_id = {'caption2rgb': 1, 'grounding2rgb': 2, 'dinolocal2rgb': 3}
                    route_id = route_to_id.get(route_name, 0)
                else:
                    route_id = 0
                sequence_status['mse_loss_route_ids'].extend([route_id] * num_img_tokens)
                sequence_status['mse_loss_image_ids'].extend([mse_image_id] * num_img_tokens)
                sequence_status['mse_loss_timesteps'].extend([timestep] * num_img_tokens)
                sequence_status['packed_timesteps'].extend([timestep] * num_img_tokens)
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                sequence_status['packed_text_ids'].append(eoi_id)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                if split_start:
                    attn_modes.append("noise")
                sequence_status['packed_position_ids'].extend(
                    [curr_rope_id] * (len(prefix_ids) + num_img_tokens + 1)
                )

            elif item['type'] == 'native_cond_joint_image':
                if not (self.visual_gen or self.visual_und):
                    curr_rope_id += 1
                    continue
                vae_image_tensor = image_tensor_list.pop(0)
                vit_image_tensor = image_tensor_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.vae_cond_dropout_prob:
                    curr_rope_id += 1
                    continue

                boi_id = self._native_token_id('<boi>')
                sep_id = self._native_token_id('<joint_img_sep>')
                eoi_id = self._native_token_id('<eoi>')

                sequence_status['packed_text_ids'].append(boi_id)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                vae_image_tensor = self._fit_vae_image_tensor(vae_image_tensor, item['modality_type'])
                sequence_status['vae_image_tensors'].append(vae_image_tensor)
                sequence_status['vae_image_modality_types'].append(item['modality_type'])
                latent_position_ids = self.get_flattened_position_ids(
                    vae_image_tensor.size(1), vae_image_tensor.size(2),
                    self.data_config.vae_image_downsample,
                    max_num_patches_per_side=self.data_config.max_latent_size
                )
                sequence_status['packed_latent_position_ids'].append(latent_position_ids)
                H, W = vae_image_tensor.shape[1:]
                h = H // self.data_config.vae_image_downsample
                w = W // self.data_config.vae_image_downsample
                num_vae_tokens = w * h
                sequence_status['vae_latent_shapes'].append((h, w))
                sequence_status['packed_vae_token_indexes'].extend(range(curr, curr + num_vae_tokens))
                sequence_status['packed_timesteps'].extend([0.0] * num_vae_tokens)
                curr += num_vae_tokens
                curr_split_len += num_vae_tokens

                sequence_status['packed_text_ids'].append(sep_id)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                vit_image_tensor = self._fit_vit_image_tensor(vit_image_tensor, item['modality_type'])
                vit_tokens = patchify(vit_image_tensor, self.data_config.vit_patch_size)
                num_vit_tokens = vit_tokens.shape[0]
                sequence_status['packed_vit_token_indexes'].extend(range(curr, curr + num_vit_tokens))
                curr += num_vit_tokens
                curr_split_len += num_vit_tokens
                sequence_status['packed_vit_tokens'].append(vit_tokens)
                sequence_status['vit_token_seqlens'].append(num_vit_tokens)
                sequence_status['vit_image_modality_types'].append(item['modality_type'])
                sequence_status['vit_image_token_shapes'].append((
                    vit_image_tensor.size(1) // self.data_config.vit_patch_size,
                    vit_image_tensor.size(2) // self.data_config.vit_patch_size,
                ))
                vit_position_ids = self.get_flattened_position_ids(
                    vit_image_tensor.size(1), vit_image_tensor.size(2),
                    self.data_config.vit_patch_size,
                    max_num_patches_per_side=self.data_config.max_num_patch_per_side
                )
                sequence_status['packed_vit_position_ids'].append(vit_position_ids)

                sequence_status['packed_text_ids'].append(eoi_id)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                if split_start:
                    attn_modes.append("full")
                sequence_status['packed_position_ids'].extend([curr_rope_id] * curr_split_len)
                curr_rope_id += 1

            if item.get('split_end', True):
                split_lens.append(curr_split_len)
                sample_lens += curr_split_len

        sequence_status['curr'] = curr
        sequence_status['sample_lens'].append(sample_lens)
        sequence_status['target_modality_types'].append(last_loss_modality_type or item['modality_type'])
        # prepare attention mask
        if not self.use_flex:
            sequence_status['nested_attention_masks'].append(
                prepare_attention_mask_per_sample(split_lens, attn_modes)
            )
        else:
            sequence_status['split_lens'].extend(split_lens)
            sequence_status['attn_modes'].extend(attn_modes)

        return sequence_status


class SimpleCustomBatch:
    def __init__(self, batch):
        data = batch[0]
        self.batch_data_indexes = data['batch_data_indexes']
        self.data_resume_state = data.get("data_resume_state")
        self.sequence_length = data["sequence_length"]
        self.num_samples = data["num_samples"]
        self.num_epoch_samples = data.get("num_epoch_samples", data["num_samples"])
        self.sample_lens = data["sample_lens"]
        self.target_modality_types = data["target_modality_types"]
        self.vae_image_modality_types = data["vae_image_modality_types"]
        self.packed_text_ids = data["packed_text_ids"]
        self.packed_text_indexes = data["packed_text_indexes"]
        self.packed_position_ids = data["packed_position_ids"]

        self.use_flex = "nested_attention_masks" not in data.keys()

        if self.use_flex:
            self.split_lens = data["split_lens"]
            self.attn_modes = data["attn_modes"]
        else:
            self.nested_attention_masks = data["nested_attention_masks"]

        if "padded_images" in data.keys():
            self.padded_images = data["padded_images"]
            self.patchified_vae_latent_shapes = data["patchified_vae_latent_shapes"]
            self.packed_latent_position_ids = data["packed_latent_position_ids"]
            self.packed_vae_token_indexes = data["packed_vae_token_indexes"]

        if "packed_vit_tokens" in data.keys():
            self.packed_vit_tokens = data["packed_vit_tokens"]
            self.packed_vit_position_ids = data["packed_vit_position_ids"]
            self.packed_vit_token_indexes = data["packed_vit_token_indexes"]
            self.vit_token_seqlens = data["vit_token_seqlens"]
            self.vit_image_modality_types = data["vit_image_modality_types"]
            # Per-image (token_h, token_w) for 2D rope_image_info on ViT inputs (B fix)
            self.vit_image_token_shapes = data.get("vit_image_token_shapes", [])

        if "packed_timesteps" in data.keys():
            self.packed_timesteps = data["packed_timesteps"]
            self.mse_loss_indexes = data["mse_loss_indexes"]
            self.mse_loss_modality_ids = data["mse_loss_modality_ids"]
            self.mse_loss_route_ids = data["mse_loss_route_ids"]
            self.mse_loss_image_ids = data["mse_loss_image_ids"]
            self.mse_loss_timesteps = data["mse_loss_timesteps"]

        if "packed_label_ids" in data.keys():
            self.packed_label_ids = data["packed_label_ids"]
            self.ce_loss_indexes = data["ce_loss_indexes"]
            self.ce_loss_weights = data["ce_loss_weights"]
            self.ce_loss_modality_ids = data["ce_loss_modality_ids"]

    def pin_memory(self):
        self.packed_text_ids = self.packed_text_ids.pin_memory()
        self.packed_text_indexes = self.packed_text_indexes.pin_memory()
        self.packed_position_ids = self.packed_position_ids.pin_memory()

        if not self.use_flex:
            self.nested_attention_masks = [item.pin_memory() for item in self.nested_attention_masks]

        if hasattr(self, 'padded_images'):
            self.padded_images = self.padded_images.pin_memory()
            self.packed_vae_token_indexes = self.packed_vae_token_indexes.pin_memory()
            self.packed_latent_position_ids = self.packed_latent_position_ids.pin_memory()

        if hasattr(self, 'packed_timesteps'):
            self.packed_timesteps = self.packed_timesteps.pin_memory()
            self.mse_loss_indexes = self.mse_loss_indexes.pin_memory()
            self.mse_loss_modality_ids = self.mse_loss_modality_ids.pin_memory()
            self.mse_loss_route_ids = self.mse_loss_route_ids.pin_memory()
            self.mse_loss_image_ids = self.mse_loss_image_ids.pin_memory()
            self.mse_loss_timesteps = self.mse_loss_timesteps.pin_memory()

        if hasattr(self, 'packed_vit_tokens'):
            self.packed_vit_tokens = self.packed_vit_tokens.pin_memory()
            self.packed_vit_position_ids = self.packed_vit_position_ids.pin_memory()
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.pin_memory()
            self.vit_token_seqlens = self.vit_token_seqlens.pin_memory()

        if hasattr(self, 'packed_label_ids'):
            self.packed_label_ids = self.packed_label_ids.pin_memory()
            self.ce_loss_indexes = self.ce_loss_indexes.pin_memory()
            self.ce_loss_weights = self.ce_loss_weights.pin_memory()
            self.ce_loss_modality_ids = self.ce_loss_modality_ids.pin_memory()

        return self

    def cuda(self, device):
        self.packed_text_ids = self.packed_text_ids.to(device)
        self.packed_text_indexes = self.packed_text_indexes.to(device)
        self.packed_position_ids = self.packed_position_ids.to(device)

        if not self.use_flex:
            self.nested_attention_masks = [item.to(device) for item in self.nested_attention_masks]

        if hasattr(self, 'padded_images'):
            self.padded_images = self.padded_images.to(device)
            self.packed_vae_token_indexes = self.packed_vae_token_indexes.to(device)
            self.packed_latent_position_ids = self.packed_latent_position_ids.to(device)

        if hasattr(self, 'packed_timesteps'):
            self.packed_timesteps = self.packed_timesteps.to(device)
            self.mse_loss_indexes = self.mse_loss_indexes.to(device)
            self.mse_loss_modality_ids = self.mse_loss_modality_ids.to(device)
            self.mse_loss_route_ids = self.mse_loss_route_ids.to(device)
            self.mse_loss_image_ids = self.mse_loss_image_ids.to(device)
            self.mse_loss_timesteps = self.mse_loss_timesteps.to(device)

        if hasattr(self, 'packed_vit_tokens'):
            self.packed_vit_tokens = self.packed_vit_tokens.to(device)
            self.packed_vit_position_ids = self.packed_vit_position_ids.to(device)
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.to(device)
            self.vit_token_seqlens = self.vit_token_seqlens.to(device)

        if hasattr(self, 'packed_label_ids'):
            self.packed_label_ids = self.packed_label_ids.to(device)
            self.ce_loss_indexes = self.ce_loss_indexes.to(device)
            self.ce_loss_weights = self.ce_loss_weights.to(device)
            self.ce_loss_modality_ids = self.ce_loss_modality_ids.to(device)

        return self

    def to_dict(self):
        data = dict(
            sequence_length = self.sequence_length,
            num_samples = self.num_samples,
            num_epoch_samples = self.num_epoch_samples,
            sample_lens = self.sample_lens,
            target_modality_types = self.target_modality_types,
            vae_image_modality_types = self.vae_image_modality_types,
            packed_text_ids = self.packed_text_ids,
            packed_text_indexes = self.packed_text_indexes,
            packed_position_ids = self.packed_position_ids,
            batch_data_indexes = self.batch_data_indexes,
            data_resume_state = self.data_resume_state,
        )

        if not self.use_flex:
            data['nested_attention_masks'] = self.nested_attention_masks
        else:
            data['split_lens'] = self.split_lens
            data['attn_modes'] = self.attn_modes

        if hasattr(self, 'padded_images'):
            data['padded_images'] = self.padded_images
            data['patchified_vae_latent_shapes'] = self.patchified_vae_latent_shapes
            data['packed_latent_position_ids'] = self.packed_latent_position_ids
            data['packed_vae_token_indexes'] = self.packed_vae_token_indexes

        if hasattr(self, 'packed_vit_tokens'):
            data['packed_vit_tokens'] = self.packed_vit_tokens
            data['packed_vit_position_ids'] = self.packed_vit_position_ids
            data['packed_vit_token_indexes'] = self.packed_vit_token_indexes
            data['vit_token_seqlens'] = self.vit_token_seqlens
            data['vit_image_modality_types'] = self.vit_image_modality_types
            data['vit_image_token_shapes'] = getattr(self, 'vit_image_token_shapes', [])

        if hasattr(self, 'packed_timesteps'):
            data['packed_timesteps'] = self.packed_timesteps
            data['mse_loss_indexes'] = self.mse_loss_indexes
            data['mse_loss_modality_ids'] = self.mse_loss_modality_ids
            data['mse_loss_route_ids'] = self.mse_loss_route_ids
            data['mse_loss_image_ids'] = self.mse_loss_image_ids
            data['mse_loss_timesteps'] = self.mse_loss_timesteps

        if hasattr(self, 'packed_label_ids'):
            data['packed_label_ids'] = self.packed_label_ids
            data['ce_loss_indexes'] = self.ce_loss_indexes
            data['ce_loss_weights'] = self.ce_loss_weights
            data['ce_loss_modality_ids'] = self.ce_loss_modality_ids

        return data


def collate_wrapper():
    def collate_fn(batch):
        return SimpleCustomBatch(batch)
    return collate_fn
