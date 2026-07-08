# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import os
import traceback
from PIL import Image, ImageFile, PngImagePlugin

import pyarrow.parquet as pq

from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset
from .parquet_utils import (
    apply_data_root_override,
    get_parquet_data_paths,
    init_arrow_pf_fs,
    read_file_bytes,
    strip_s3_scheme,
)


def _open_image(path):
    # Local Image.open, or read bytes from S3 first (vlm_sft S3 streaming).
    if path.startswith("s3://"):
        return Image.open(io.BytesIO(read_file_bytes(path)))
    return Image.open(path)


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class SftJSONLIterableDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, frame_sampler=None,
        jsonl_path_list=None, data_dir_list=None, num_used_data=None,
        local_rank=0, world_size=1, num_workers=8, data_status=None, 
        shuffle_lines=False, shuffle_seed=0, use_instruction=False,
        use_condition_instruction=False, use_target_instruction=False, num_condition_modalities=0,
        strict_num_condition_modalities=False,
        use_det_image=False, modality_registry=None,
    ):
        """
        jsonl_path_list: list of jsonl file paths
        data_dir_list: list of image directories containing the images of each jsonl file
        num_used_data: list of number of sampled data points for each jsonl
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.frame_sampler = frame_sampler
        self.data_status = data_status
        self.data_paths = self.get_data_paths(
            jsonl_path_list, 
            data_dir_list, 
            num_used_data, 
            shuffle_lines, 
            shuffle_seed,
        )
        self.set_epoch()

    def get_data_paths(
        self, 
        jsonl_path_list, 
        data_dir_list, 
        num_used_data, 
        shuffle_lines, 
        shuffle_seed,
    ):
        data_paths = []
        for jsonl_path, image_dir, num_data_point in zip(
            jsonl_path_list, data_dir_list, num_used_data
        ):
            # MODUS_DATA_ROOT redirects these to S3 (mirrors datasets/); unset → local.
            jsonl_path = apply_data_root_override(jsonl_path)
            image_dir = apply_data_root_override(image_dir)
            if jsonl_path.startswith("s3://"):
                raw_data = read_file_bytes(jsonl_path).decode("utf-8").splitlines(keepends=True)
            else:
                with open(jsonl_path, 'r') as f:
                    raw_data = f.readlines()
            if shuffle_lines:
                self.rng.seed(shuffle_seed)
                self.rng.shuffle(raw_data)
            raw_data = raw_data[:num_data_point]
            data_paths.extend([(json_data, image_dir) for json_data in raw_data])
        return data_paths

    def change_format(self, data, num_images):
        elements = []
        for conversation in data['conversations']:
            if conversation['from'] == 'human':
                if '<image>' not in conversation['value']:
                    elements.append({
                        'type': 'text',
                        'has_loss': 0,
                        'text': conversation['value'],
                    })
                else:
                    text_list = conversation['value'].split('<image>')
                    for idx, text in enumerate(text_list):
                        if text.strip() != '':
                            elements.append({
                                'type': 'text',
                                'has_loss': 0,
                                'text': text.strip(),
                            })
                        if (idx != len(text_list) - 1) and (idx < num_images):
                            elements.append({'type': 'image',})
            elif conversation['from'] == 'gpt':
                elements.append({
                    'type': 'text',
                    'has_loss': 1,
                    'text': conversation['value'],
                })
        return elements

    def _build_sample(self, data_item, raw_images):
        # Shared record→sample logic for the jsonl and parquet backends.
        # Returns the sample dict (without data_indexes) or None when the
        # record produces no loss tokens.
        num_tokens = 0
        image_tensor_list = []
        text_ids_list = []
        sequence_plan = []

        if raw_images:
            for raw_image in raw_images:
                image_tensor = self.transform(raw_image, img_num=len(raw_images))
                image_tensor_list.append(image_tensor)
                height, width = image_tensor.shape[1:]
                num_tokens += width * height // self.transform.stride ** 2

        elements = self.change_format(data_item, len(image_tensor_list))

        for item in elements:
            if item['type'] == 'text':
                text_data = item['text']
                text_ids = self.tokenizer.encode(text_data)
                if len(text_ids) > 0:
                    text_ids_list.append(text_ids)
                    num_tokens += len(text_ids)
                    current_plan = {
                        'type': 'text',
                        'enable_cfg': 0,
                        'loss': item['has_loss'],
                        'special_token_loss': 0,
                        'special_token_label': None,
                        'modality_type': 'text',
                    }
                    sequence_plan.append(current_plan)
            elif item['type'] == 'image':
                current_plan = {
                    'type': 'vit_image',
                    'enable_cfg': 0,
                    'loss': 0,
                    'special_token_loss': 0,
                    'special_token_label': None,
                    'modality_type': 'rgb',
                }
                sequence_plan.append(current_plan)
                # NOTE: add VAE or not?
                # firstly just ViT and check result, then maybe VAE but not very important for now. train two versions maybe.

        has_loss = [item['loss'] for item in sequence_plan]
        if sum(has_loss) == 0:
            print(f'No loss defined, skipped.')
            return None

        return dict(
            image_tensor_list=image_tensor_list,
            text_ids_list=text_ids_list,
            sequence_plan=sequence_plan,
            num_tokens=num_tokens,
        )

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, (data, image_dir) in enumerate(data_paths_per_worker_, start=row_start_id):
                try:
                    data_item = json.loads(data)
                    raw_images = None
                    if 'image' in data_item:
                        images = data_item['image'] if isinstance(data_item['image'], list) else [data_item['image']]
                        valid_paths = []
                        for img_name in images:
                            if isinstance(img_name, str) and img_name.strip() != '':
                                img_path = os.path.join(image_dir, img_name)
                                # S3 paths can't be cheaply stat'd; include them and
                                # let the read below (in the try/except) drop a record
                                # whose image is missing.
                                if img_path.startswith("s3://") or os.path.isfile(img_path):
                                    valid_paths.append(img_path)
                        if len(valid_paths) == 0:
                            raise FileNotFoundError("No valid image files found for this record.")
                        raw_images = [pil_img2rgb(_open_image(p)) for p in valid_paths]
                    elif 'video' in data_item:
                        vid_name = data_item['video']
                        if not (isinstance(vid_name, str) and vid_name.strip() != ''):
                            raise ValueError("Invalid video name in record.")
                        vid_path = os.path.join(image_dir, vid_name)
                        if not os.path.isfile(vid_path):
                            raise FileNotFoundError("Video file not found for this record.")
                        raw_images = self.frame_sampler(vid_path)
                        special_tokens = '<image>' * len(raw_images)
                        for item in data_item['conversations']:
                            if '<video>' in item['value']:
                                item['value'] = item['value'].replace('<video>', special_tokens)
                                break
                        else:
                            raise ValueError("Cannot find <video> in the conversation!")
                except:
                    traceback.print_exc()
                    continue

                sample = self._build_sample(data_item, raw_images)
                if sample is None:
                    continue
                sample['data_indexes'] = {
                    "data_indexes": row_idx,
                    "worker_id": worker_id,
                    "dataset_name": self.dataset_name,
                }
                yield sample

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


class SftParquetIterableDataset(SftJSONLIterableDataset):
    """vlm_sft from bundled parquet shards instead of jsonl + loose images.

    Motivation: llava-onevision is ~4M small files, which is impractical to
    upload to / stream from S3. The shards are written by
    data/any2any_preprocess/generate_parquet_vlm_sft.py, which bakes in the
    jsonl-time sample selection (shuffle_lines + shuffle_seed + num_used_data),
    so each parquet row is exactly one record the jsonl loader would have used.

    Row schema:
        record: str                original jsonl line (conversations etc.)
        image_bytes: list<binary>  raw image files, in the order the jsonl
                                   loader's valid_paths would produce

    Iteration is sharded by (parquet_file, row_group) like the blip3o
    parquet datasets (S3 streaming via init_arrow_pf_fs); per-record
    processing reuses _build_sample, so emitted samples are identical to
    the jsonl path. Resume status is [global_row_group_idx, row_idx].
    """

    def __init__(
        self, dataset_name, transform, tokenizer, parquet_info,
        data_dir_list=None, num_used_data=None, frame_sampler=None,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
        shuffle_lines=False, shuffle_seed=0, use_instruction=False,
        use_condition_instruction=False, use_target_instruction=False,
        num_condition_modalities=0, strict_num_condition_modalities=False,
        use_det_image=False, modality_registry=None,
    ):
        """
        data_dir_list: list of directories containing the parquet shards
        num_used_data: list of number of parquet files used per directory
            (None → all files; sample selection is baked in at conversion)
        """
        DistributedIterableDataset.__init__(
            self, dataset_name, local_rank, world_size, num_workers
        )
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data, parquet_info)
        self.set_epoch()

    def get_data_paths(self, data_dir_list, num_used_data, parquet_info):
        # Same (file, row_group) expansion as ParquetStandardIterableDataset,
        # including the basename fallback so the parquet_info JSON keys are
        # independent of the data_dir prefix (local path vs s3://...).
        if num_used_data is None:
            num_used_data = [None] * len(data_dir_list)
        info_by_base = {os.path.basename(k): v for k, v in parquet_info.items()}
        row_groups = []
        for data_dir, num_data_path in zip(data_dir_list, num_used_data):
            data_paths = get_parquet_data_paths([data_dir], [num_data_path])
            for data_path in data_paths:
                info = parquet_info.get(data_path) or info_by_base.get(os.path.basename(data_path))
                if info is not None:
                    for rg_idx in range(info['num_row_groups']):
                        row_groups.append((data_path, rg_idx))
        return row_groups

    def __iter__(self):
        file_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            global_row_group_start_id = self.data_status[worker_id][0]
            row_start_id = self.data_status[worker_id][1] + 1
        else:
            global_row_group_start_id = 0
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at global_rg#{global_row_group_start_id}, row#{row_start_id}"
        )

        while True:
            file_paths_per_worker_ = file_paths_per_worker[global_row_group_start_id:]
            for global_row_group_idx, (parquet_file_path, row_group_id) in enumerate(
                file_paths_per_worker_, start=global_row_group_start_id
            ):
                fs = init_arrow_pf_fs(parquet_file_path)
                with fs.open_input_file(strip_s3_scheme(parquet_file_path)) as f:
                    try:
                        fr = pq.ParquetFile(f)
                        df = fr.read_row_group(row_group_id).to_pandas()
                        df = df.iloc[row_start_id:]
                    except Exception as e:
                        print(f'Error {e} in rg#{row_group_id}, {parquet_file_path}')
                        continue

                    for row_idx, row in df.iterrows():
                        try:
                            data_item = json.loads(row['record'])
                            image_bytes = row['image_bytes']
                            raw_images = None
                            if image_bytes is not None and len(image_bytes) > 0:
                                raw_images = [
                                    pil_img2rgb(Image.open(io.BytesIO(b)))
                                    for b in image_bytes
                                ]
                        except Exception as e:
                            print(f'Error {e} in rg#{row_group_id}, {parquet_file_path}')
                            continue

                        sample = self._build_sample(data_item, raw_images)
                        if sample is None:
                            continue
                        sample['data_indexes'] = {
                            "data_indexes": [global_row_group_idx, row_idx],
                            "worker_id": worker_id,
                            "dataset_name": self.dataset_name,
                        }
                        yield sample

                    row_start_id = 0
            global_row_group_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
