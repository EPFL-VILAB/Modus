# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import os
import subprocess
import logging

import pyarrow.fs as pf
import torch.distributed as dist

logger = logging.getLogger(__name__)


def apply_data_root_override(path):
    # If MODUS_DATA_ROOT is set, replace the local 'datasets/' prefix with it,
    # preserving the sub-path. This mirrors the datasets/ tree under the root and
    # works for BOTH parquet dirs (datasets/blip3o/parquet_...) and vlm_sft
    # image/jsonl paths (datasets/llava_onevision_vqa/<name>/...). Switching
    # local<->S3 — and 13mod<->16mod<->Hunyuan — becomes a pure env/config change
    # with NO edit to dataset_info.py. Paths not under datasets/ and the unset
    # case are returned unchanged (forward-compatible).
    root = os.environ.get("MODUS_DATA_ROOT")
    if not root:
        return path
    for prefix in ("./datasets/", "datasets/"):
        if path.startswith(prefix):
            return root.rstrip("/") + "/" + path[len(prefix):]
    return path


def read_file_bytes(path):
    # Read a whole file as bytes, scheme-selected (local / hdfs / s3) via the same
    # filesystem logic as parquet. Used for S3 streaming of vlm_sft jsonl + images.
    fs = init_arrow_pf_fs(path)
    with fs.open_input_file(strip_s3_scheme(path)) as f:
        return f.read()


def get_parquet_data_paths(data_dir_list, num_sampled_data_paths, rank=0, world_size=1):
    data_dir_list = [apply_data_root_override(d) for d in data_dir_list]
    num_data_dirs = len(data_dir_list)
    if world_size > 1:
        chunk_size = (num_data_dirs + world_size - 1) // world_size
        start_idx = rank * chunk_size
        end_idx = min(start_idx + chunk_size, num_data_dirs)
        local_data_dir_list = data_dir_list[start_idx:end_idx]
        local_num_sampled_data_paths = num_sampled_data_paths[start_idx:end_idx]
    else:
        local_data_dir_list = data_dir_list
        local_num_sampled_data_paths = num_sampled_data_paths

    local_data_paths = []
    for data_dir, num_data_path in zip(local_data_dir_list, local_num_sampled_data_paths):
        if data_dir.startswith("hdfs://"):
            files = hdfs_ls_cmd(data_dir)
            data_paths_per_dir = [
                file for file in files if file.endswith(".parquet")
            ]
        elif data_dir.startswith("s3://"):
            fs = init_arrow_pf_fs(data_dir)
            selector = pf.FileSelector(strip_s3_scheme(data_dir), recursive=False)
            data_paths_per_dir = [
                "s3://" + info.path
                for info in fs.get_file_info(selector)
                if info.path.endswith(".parquet")
            ]
        else:
            files = os.listdir(data_dir)
            data_paths_per_dir = [
                os.path.join(data_dir, name)
                for name in files
                if name.endswith(".parquet")
            ]
        if num_data_path is None:
            # None → use every parquet file in the dir exactly once
            # (no repeat/truncate). Used by vlm_sft_parquet where the
            # sample selection is already baked in at conversion time.
            local_data_paths.extend(data_paths_per_dir)
            continue
        repeat = num_data_path // len(data_paths_per_dir)
        data_paths_per_dir = data_paths_per_dir * (repeat + 1)
        local_data_paths.extend(data_paths_per_dir[:num_data_path])

    if world_size > 1:
        gather_list = [None] * world_size
        dist.all_gather_object(gather_list, local_data_paths)

        combined_chunks = []
        for chunk_list in gather_list:
            if chunk_list is not None:
                combined_chunks.extend(chunk_list)
    else:
        combined_chunks = local_data_paths

    return combined_chunks


# NOTE: cumtomize this function for your cluster
def get_hdfs_host():
    return "hdfs://xxx"


# NOTE: cumtomize this function for your cluster
def get_hdfs_block_size():
    return 134217728


# NOTE: cumtomize this function for your cluster
def get_hdfs_extra_conf():
    return None


def strip_s3_scheme(path):
    # pyarrow's S3FileSystem operates on 'bucket/key' (no scheme). Local/hdfs
    # paths are returned unchanged, so their behavior is untouched.
    if path.startswith("s3://"):
        return path[len("s3://"):]
    return path


def init_arrow_pf_fs(parquet_file_path):
    if parquet_file_path.startswith("hdfs://"):
        fs = pf.HadoopFileSystem(
            host=get_hdfs_host(),
            port=0,
            buffer_size=get_hdfs_block_size(),
            extra_conf=get_hdfs_extra_conf(),
        )
    elif parquet_file_path.startswith("s3://"):
        # Credentials + endpoint come from the environment so no secrets live in
        # code/config. Standard AWS_* names work; MODUS_S3_ENDPOINT/REGION/SCHEME
        # cover custom (non-AWS) S3 endpoints (e.g. Apple conductor, SCITAS).
        s3_kwargs = dict(
            access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
            secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            scheme=os.environ.get("MODUS_S3_SCHEME", "https"),
        )
        if os.environ.get("MODUS_S3_ENDPOINT"):
            s3_kwargs["endpoint_override"] = os.environ["MODUS_S3_ENDPOINT"]
        if os.environ.get("MODUS_S3_REGION"):
            s3_kwargs["region"] = os.environ["MODUS_S3_REGION"]
        if os.environ.get("AWS_SESSION_TOKEN"):
            s3_kwargs["session_token"] = os.environ["AWS_SESSION_TOKEN"]
        fs = pf.S3FileSystem(**s3_kwargs)
    else:
        fs = pf.LocalFileSystem()
    return fs


def hdfs_ls_cmd(dir):
    result = subprocess.run(["hdfs", "dfs", "ls", dir], capture_output=True, text=True).stdout
    return ['hdfs://' + i.split('hdfs://')[-1].strip() for i in result.split('\n') if 'hdfs://' in i]
