# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pyarrow.parquet as pq

from ..distributed_iterable_dataset import DistributedIterableDataset
from ..parquet_utils import get_parquet_data_paths, init_arrow_pf_fs, strip_s3_scheme


class InterleavedBaseIterableDataset(DistributedIterableDataset):

    def _init_data(self):
        data = {
            'sequence_plan': [],
            'text_ids_list': [],
            'image_tensor_list': [],
            'num_tokens': 0,
        }
        return data

    def _add_text(self, data, text, need_loss, enable_cfg=True, modality_type=None):
        text_ids = self.tokenizer.encode(text)
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

    def _add_image(self, data, image, need_loss, need_vae, need_vit, enable_cfg=True, modality_type=None):
        assert need_loss or need_vae or need_vit

        if need_loss:
            data['sequence_plan'].append(
                {
                    'type': 'vae_image', 
                    'enable_cfg': 0, 
                    'loss': 1, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                    'modality_type': modality_type,
                }
            )

            image_tensor = self.transform(image)
            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.transform.stride ** 2
            data['image_tensor_list'].append(image_tensor)

        if need_vae:
            data['sequence_plan'].append(
                {
                    'type': 'vae_image', 
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                    'modality_type': modality_type,
                }
            )

            image_tensor = self.transform(image)
            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.transform.stride ** 2
            data['image_tensor_list'].append(image_tensor.clone())

        if need_vit:
            data['sequence_plan'].append(
                {
                    'type': 'vit_image',
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0,
                    'special_token_loss': 0,
                    'special_token_label': None,
                    'modality_type': modality_type,
                },
            )
            vit_image_tensor = self.vit_transform(image)
            height, width = vit_image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.vit_transform.stride ** 2
            data['image_tensor_list'].append(vit_image_tensor)

        return data

    def _add_video(self, data, frames, frame_indexes, need_loss, need_vae, enable_cfg=True):
        assert int(need_loss) + int(need_vae) == 1

        if need_loss:
            for idx, (image, frame_idx) in enumerate(zip(frames, frame_indexes)):
                current_sequence_plan = {
                    'type': 'vae_image', 
                    'enable_cfg': 0, 
                    'loss': 1, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                    'split_start': idx == 0,
                    'split_end': idx == len(frames) - 1,
                }
                if idx < len(frame_indexes) - 1:
                    current_sequence_plan['frame_delta'] = frame_indexes[idx + 1] - frame_idx
                data['sequence_plan'].append(current_sequence_plan)
                image_tensor = self.transform(image)
                height, width = image_tensor.shape[1:]
                data['image_tensor_list'].append(image_tensor)
                data['num_tokens'] += width * height // self.transform.stride ** 2

        elif need_vae:
            for idx, (image, frame_idx) in enumerate(zip(frames, frame_indexes)):
                current_sequence_plan = {
                    'type': 'vae_image', 
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                    'split_start': idx == 0,
                    'split_end': idx == len(frames) - 1,
                }
                if idx < len(frame_indexes) - 1:
                    current_sequence_plan['frame_delta'] = frame_indexes[idx + 1] - frame_idx
                data['sequence_plan'].append(current_sequence_plan)
                image_tensor = self.transform(image)
                height, width = image_tensor.shape[1:]
                data['image_tensor_list'].append(image_tensor)
                data['num_tokens'] += width * height // self.transform.stride ** 2

        return data


class ParquetStandardIterableDataset(DistributedIterableDataset):

    def __init__(
        self, dataset_name, transform, tokenizer, vit_transform,
        data_dir_list, num_used_data, parquet_info,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
        use_instruction=False, use_condition_instruction=True, use_target_instruction=True,
        num_condition_modalities=1,
        strict_num_condition_modalities=False,
        use_det_image=False,
        modality_registry=None,
        tail_n_rows=None,
        force_condition_modality=None,
        force_condition_modalities=None,
        pred_intermediate_dir=None,
        gt_restricted_dir=None,
        fixed_row_list_path=None,
    ):
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        vit_transform: input transform for vit model.
        tail_n_rows: if set, only the last N rows of each row-group are yielded
            (held-out validation slice; default None preserves training behavior).
        force_condition_modality: if set, every sample uses exactly this single
            condition modality (used by the validation matrix script). Default
            None preserves training behavior (random sampling).
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data, parquet_info)
        # Pred-intermediate mode: keep only (file, rg) pairs that have at
        # least one Phase-1-saved PNG. Otherwise distributed ranks whose
        # data slice contains no pred PNGs would iterate forever skipping
        # every row and deadlock at NCCL all-reduce. Done BEFORE set_epoch
        # so the shuffle+chunk operates on the filtered set.
        #
        # gt_restricted_dir is the apples-to-apples sibling: applies the
        # SAME pair filter (so the GT matrix runs on the exact same 8 pairs
        # PRED uses) but does NOT substitute pred PNGs — the intermediate
        # condition is still loaded from parquet GT. Used to compare GT vs
        # PRED on identical sample slices.
        self.pred_intermediate_dir = pred_intermediate_dir
        self.gt_restricted_dir = gt_restricted_dir
        _filter_dir = pred_intermediate_dir or gt_restricted_dir
        if _filter_dir is not None:
            self.data_paths = self._filter_data_paths_by_pred(
                self.data_paths, _filter_dir)

        # Fixed val set mode: V_fixed.json lists exact (basename, rg, row)
        # tuples to use. Each row is keyed (basename, rg, row_idx) →
        # global_idx for deterministic rank striping. Pairs are filtered
        # to only those touched by V_fixed.
        self.fixed_row_list_path = fixed_row_list_path
        self._fixed_rows_by_pair = None  # (basename, rg) -> set(row_idx)
        self._fixed_global_idx = None     # (basename, rg, row_idx) -> global_idx
        if fixed_row_list_path is not None:
            with open(fixed_row_list_path, "r") as _f:
                _payload = json.load(_f)
            _rows = _payload.get("rows", _payload)  # accept bare list too
            _by_pair = {}
            _gidx = {}
            for r in _rows:
                key_pair = (r["parquet_basename"], int(r["rg_id"]))
                _by_pair.setdefault(key_pair, set()).add(int(r["row_idx"]))
                _gidx[(r["parquet_basename"], int(r["rg_id"]), int(r["row_idx"]))] = (
                    int(r.get("global_idx", len(_gidx))))
            self._fixed_rows_by_pair = _by_pair
            self._fixed_global_idx = _gidx
            # Filter data_paths to only the (file, rg) pairs touched by V_fixed.
            _keep = set(_by_pair.keys())
            _filtered = []
            for path, rg in self.data_paths:
                base = os.path.splitext(os.path.basename(path))[0]
                if (base, int(rg)) in _keep:
                    _filtered.append((path, rg))
            print(f"[fixed-row] V_fixed has {len(_rows)} rows across "
                  f"{len(_by_pair)} pairs; kept {len(_filtered)} matching "
                  f"data_paths (out of {len(self.data_paths)})")
            self.data_paths = _filtered

        self.set_epoch()
        self.use_instruction = use_instruction
        self.use_condition_instruction = use_condition_instruction
        self.use_target_instruction = use_target_instruction
        self.num_condition_modalities = num_condition_modalities
        self.strict_num_condition_modalities = strict_num_condition_modalities
        self.use_det_image = use_det_image
        self.modality_registry = modality_registry
        self.tail_n_rows = tail_n_rows
        self.force_condition_modality = force_condition_modality
        self.force_condition_modalities = force_condition_modalities
        print(f"use_instruction: {self.use_instruction}")

    def _filter_data_paths_by_pred(self, data_paths, pred_intermediate_dir):
        """Keep only (parquet_path, rg_id) pairs covered by Phase-1 outputs.
        Phase 1 writes files as
            <pred_dir>/<modality>/<parquet_basename>_rg<rg>_row*_pred.png
        We consider a pair 'covered' if ANY modality subdir has at least
        one PNG matching `<basename>_rg<rg>_row*_pred.png`.
        Backward compat: only invoked when pred_intermediate_dir is set.
        """
        import glob as _glob
        covered = set()
        if not os.path.isdir(pred_intermediate_dir):
            print(f'[pred-filter] WARN: pred_intermediate_dir not found: {pred_intermediate_dir}')
            return []
        # Scan each modality subdir, harvest (basename, rg_id) keys.
        for sub in os.listdir(pred_intermediate_dir):
            sub_dir = os.path.join(pred_intermediate_dir, sub)
            if not os.path.isdir(sub_dir):
                continue
            for fn in os.listdir(sub_dir):
                if not fn.endswith('_pred.png'):
                    continue
                # filename: <basename>_rg<rg>_row<row>_pred.png
                # split from the right: ['_pred.png'] then row, rg, basename
                stem = fn[:-len('_pred.png')]
                # find last '_row' before number
                idx_row = stem.rfind('_row')
                if idx_row < 0:
                    continue
                pre_row = stem[:idx_row]   # <basename>_rg<rg>
                idx_rg = pre_row.rfind('_rg')
                if idx_rg < 0:
                    continue
                basename = pre_row[:idx_rg]
                try:
                    rg_id = int(pre_row[idx_rg + len('_rg'):])
                except ValueError:
                    continue
                covered.add((basename, rg_id))
        filtered = []
        for path, rg in data_paths:
            base = os.path.splitext(os.path.basename(path))[0]
            if (base, rg) in covered:
                filtered.append((path, rg))
        print(f'[pred-filter] kept {len(filtered)}/{len(data_paths)} (file, rg) pairs '
              f'covered by pred_intermediate_dir={pred_intermediate_dir}')
        return filtered

    def get_data_paths(self, data_dir_list, num_used_data, parquet_info):
        # Index parquet_info by basename too, so the lookup is independent of the
        # data_dir prefix (local path vs s3://...). Basenames are unique within a
        # dataset's parquet_info, so this is equivalent to exact-path matching for
        # existing local runs (forward-compatible) while letting an s3:// data_dir
        # reuse the same JSON unchanged.
        info_by_base = {os.path.basename(k): v for k, v in parquet_info.items()}
        row_groups = []
        for data_dir, num_data_path in zip(data_dir_list, num_used_data):
            data_paths = get_parquet_data_paths([data_dir], [num_data_path])
            for data_path in data_paths:
                info = parquet_info.get(data_path) or info_by_base.get(os.path.basename(data_path))
                if info is not None:
                    num_row_groups = info['num_row_groups']
                    for rg_idx in range(num_row_groups):
                        row_groups.append((data_path, rg_idx))
        return row_groups

    def parse_row(self, row):
        raise NotImplementedError

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
                        if self.tail_n_rows is not None and self.tail_n_rows > 0:
                            df = df.iloc[-int(self.tail_n_rows):]
                        df = df.iloc[row_start_id:]
                    except Exception as e:
                        print(f'Error {e} in rg#{row_group_id}, {parquet_file_path}')
                        continue

                    # Fixed val set: pre-compute which row_idxs of THIS
                    # (file, rg) pair are in V_fixed. Skip rows not in the
                    # set. No-op when fixed_row_list_path is None.
                    _fixed_set_for_pair = None
                    if self._fixed_rows_by_pair is not None:
                        _basename = os.path.splitext(
                            os.path.basename(parquet_file_path))[0]
                        _fixed_set_for_pair = self._fixed_rows_by_pair.get(
                            (_basename, int(row_group_id)), set())

                    for row_idx, row in df.iterrows():
                        try:
                            if _fixed_set_for_pair is not None and int(row_idx) not in _fixed_set_for_pair:
                                continue
                            # Expose per-row parquet provenance to subclasses
                            # via transient attrs (subclasses ignore if unused).
                            # Backward compatible — old code paths don't read these.
                            self._current_parquet_path = parquet_file_path
                            self._current_rg_id = row_group_id
                            self._current_row_idx = row_idx
                            data = self.parse_row(row)
                            if data is None or len(data) == 0:
                                continue
                            data['data_indexes'] = {
                                "data_indexes": [global_row_group_idx, row_idx],
                                "worker_id": worker_id,
                                "dataset_name": self.dataset_name,
                            }
                        except Exception as e:
                            # Silently skip rows whose pred-intermediate
                            # PNG is missing — that's the expected case
                            # outside Phase 1's generated subset, not an
                            # error.
                            if type(e).__name__ == 'PredIntermediateMissing':
                                continue
                            print(f'Error {e} in rg#{row_group_id}, {parquet_file_path}')
                            continue
                        yield data

                    row_start_id = 0
            global_row_group_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
