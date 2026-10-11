#!/usr/bin/env python3
"""Bitwise-verify an exported MODUS-Hunyuan checkpoint against its sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from scripts.hunyuan.export_hunyuan_checkpoint import (
    PACKED_SUFFIXES,
    PREFIX,
    _list_shards,
    _load_metadata,
    _target_shapes,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--modality-config", required=True)
    parser.add_argument("--export", required=True)
    parser.add_argument("--family", choices=("model", "ema"), default="ema")
    return parser.parse_args()


def _indexed_handles(directory: Path, index_name: str):
    index = json.loads((directory / index_name).read_text())
    handles = {
        filename: safe_open(str(directory / filename), framework="pt", device="cpu")
        for filename in sorted(set(index["weight_map"].values()))
    }
    return index["weight_map"], handles


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    base_model = Path(args.base_model).resolve()
    modality_config = Path(args.modality_config).resolve()
    exported = Path(args.export).resolve()

    metadata = _load_metadata(checkpoint)
    source_paths = _list_shards(checkpoint, args.family)
    if len(source_paths) != int(metadata["world_size"]):
        raise RuntimeError("Checkpoint shard count does not match metadata")
    source_handles = [safe_open(str(p), framework="pt", device="cpu") for p in source_paths]
    source_keys = [set(handle.keys()) for handle in source_handles]
    ordered_keys = sorted(set().union(*source_keys))

    _, _, target_shapes = _target_shapes(base_model, modality_config)
    export_map, export_handles = _indexed_handles(
        exported, "model.safetensors.index.json"
    )
    adapter_path = exported / "modus_adapter.safetensors"
    adapter = (
        safe_open(str(adapter_path), framework="pt", device="cpu")
        if adapter_path.is_file()
        else None
    )
    adapter_keys = set(adapter.keys()) if adapter is not None else set()

    def actual(key: str):
        if key in adapter_keys:
            return adapter.get_tensor(key)
        filename = export_map.get(key)
        if filename is None:
            raise RuntimeError(f"Export is missing tensor {key!r}")
        return export_handles[filename].get_tensor(key)

    def parts_for(full_key: str, ranks):
        parts = []
        for rank in ranks:
            if full_key not in source_keys[rank]:
                raise RuntimeError(f"Source rank {rank} is missing {full_key!r}")
            part = source_handles[rank].get_tensor(full_key)
            if part.numel():
                parts.append(part.reshape(-1))
        return parts

    checked = set()
    packed_keys = {
        key for key in ordered_keys if any(key.endswith(s) for s in PACKED_SUFFIXES)
    }
    world_size = int(metadata["world_size"])
    for full_key in ordered_keys:
        if not full_key.startswith(PREFIX) or full_key in packed_keys:
            continue
        key = full_key[len(PREFIX):]
        shape = target_shapes[key]
        parts = parts_for(full_key, range(world_size))
        target_numel = 1
        for dim in shape:
            target_numel *= dim
        if len(parts) == world_size and all(p.numel() == target_numel for p in parts):
            expected = torch.stack([p.reshape(shape).float() for p in parts]).mean(0)
        else:
            if sum(p.numel() for p in parts) != target_numel:
                raise RuntimeError(f"Source size mismatch for {key!r}")
            expected = torch.cat(parts).reshape(shape)
        if expected.is_floating_point():
            expected = expected.to(torch.bfloat16)
        observed = actual(key)
        if not torch.equal(expected, observed):
            raise RuntimeError(f"Bitwise mismatch for exported tensor {key!r}")
        checked.add(key)

    ep_size = int(metadata["ep_size"])
    expert_dp_size = int(metadata["expert_fsdp_size"])
    for full_key in sorted(packed_keys):
        packed_suffix = next(s for s in PACKED_SUFFIXES if full_key.endswith(s))
        expert_leaf = PACKED_SUFFIXES[packed_suffix]
        model_prefix = full_key[len(PREFIX):-len(packed_suffix)]
        expert_prefix = f"{model_prefix}.experts."
        target_keys = sorted(
            (
                key for key in target_shapes
                if key.startswith(expert_prefix) and key.endswith(expert_leaf)
            ),
            key=lambda key: int(key[len(expert_prefix):].split(".", 1)[0]),
        )
        local_experts = len(target_keys) // ep_size
        sample_shape = target_shapes[target_keys[0]]
        packed_shape = (local_experts, sample_shape[1], sample_shape[0])
        for ep_rank in range(ep_size):
            ranks = [dp * ep_size + ep_rank for dp in range(expert_dp_size)]
            packed = torch.cat(parts_for(full_key, ranks)).reshape(packed_shape)
            for local_idx in range(local_experts):
                key = target_keys[ep_rank * local_experts + local_idx]
                expected = packed[local_idx].transpose(0, 1).to(torch.bfloat16).contiguous()
                if not torch.equal(expected, actual(key)):
                    raise RuntimeError(f"Bitwise mismatch for exported expert {key!r}")
                checked.add(key)

    base_map, base_handles = _indexed_handles(base_model, "model.safetensors.index.json")
    vae_keys = {key for key in base_map if key.startswith("vae.")}
    for key in sorted(vae_keys):
        expected = base_handles[base_map[key]].get_tensor(key)
        if not torch.equal(expected, actual(key)):
            raise RuntimeError(f"Bitwise mismatch for exported VAE tensor {key!r}")
        checked.add(key)

    exported_keys = set(export_map) | adapter_keys
    if exported_keys != checked:
        raise RuntimeError(
            "Export tensor set mismatch: "
            f"missing={sorted(checked - exported_keys)[:20]} "
            f"unexpected={sorted(exported_keys - checked)[:20]}"
        )
    report = {
        "result": "PASS",
        "comparison": "bitwise",
        "checkpoint": checkpoint.name,
        "family": args.family,
        "verified_tensors": len(checked),
        "export_shards": len(export_handles),
        "adapter_tensors": len(adapter_keys),
        "vae_tensors": len(vae_keys),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
