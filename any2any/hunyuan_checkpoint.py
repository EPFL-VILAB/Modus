"""Hunyuan-only helpers for reconstructing tensor-parallel train checkpoints.

These are intentionally isolated from ``train.fsdp_utils``: importing an
inference backend must not initialize FSDP or require distributed training
support, and the existing BAGEL code path must remain untouched.
"""

from __future__ import annotations

import json
import os

import torch


_TP_COL_SUFFIXES = (".gate_and_up_proj.weight", ".gate_and_up_proj.bias")
_TP_ROW_SUFFIXES = (".down_proj.weight",)
_TP_ATTN_COL_SUFFIXES = (".qkv_proj.weight", ".qkv_proj.bias")
_TP_ATTN_ROW_SUFFIXES = (".o_proj.weight",)
_CHECKPOINT_METADATA_FILE = "checkpoint_metadata.json"


def read_checkpoint_metadata(checkpoint_dir: str) -> dict:
    path = os.path.join(checkpoint_dir, _CHECKPOINT_METADATA_FILE)
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Invalid checkpoint metadata in {path}: expected an object")
    return metadata


def checkpoint_tp_suffixes(checkpoint_dir: str):
    """Return tensor roles advertised by a Hunyuan training checkpoint."""
    metadata = read_checkpoint_metadata(checkpoint_dir)
    col = _TP_COL_SUFFIXES
    row = _TP_ROW_SUFFIXES
    if metadata.get("tp_attention", False):
        col += _TP_ATTN_COL_SUFFIXES
        row += _TP_ATTN_ROW_SUFFIXES
    return col, row, metadata


def combine_tp_checkpoint_parts(model_key, tp_parts, col_suffixes, row_suffixes):
    """Reconstruct a dense tensor from ordered TP-local tensors."""
    if len(tp_parts) == 1:
        return tp_parts[0]
    if model_key.endswith(row_suffixes):
        return torch.cat(tp_parts, dim=1)
    if model_key.endswith(col_suffixes):
        if "gate_and_up_proj" in model_key:
            # Local SwiGLU layout is [local gate | local up].  Restore the
            # dense [all gate | all up] layout instead of rank-major ordering.
            gates = []
            ups = []
            for part in tp_parts:
                half = part.shape[0] // 2
                gates.append(part[:half])
                ups.append(part[half:])
            return torch.cat(gates + ups, dim=0)
        return torch.cat(tp_parts, dim=0)
    return tp_parts[0]
