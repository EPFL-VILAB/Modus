#!/usr/bin/env python3
"""Export a MODUS-Hunyuan FSDP/EP checkpoint as inference-native BF16 shards.

The training checkpoint is FP32, topology-dependent, and contains replicated
EP/DP state.  This exporter reconstructs the dense native Hunyuan parameter
layout one tensor at a time, unpacks experts, includes the frozen VAE from the
base checkpoint, and writes a self-contained Hugging Face-style directory.
Peak host memory is bounded by one source tensor plus one output shard.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from any2any.hunyuan_loader import _build_tokenizer_and_registry
from any2any.hunyuan_runtime import configure_hunyuan_runtime
from modeling.hunyuan_image_3.model_wrapper import HunyuanImageWrapper
from modeling.hunyuan_image_3.src.configuration_hunyuan_image_3 import HunyuanImage3Config
from modeling.hunyuan_image_3.src.modeling_hunyuan_image_3 import HunyuanImage3ForCausalMM


PREFIX = "hunyuan_model."
PACKED_SUFFIXES = {
    ".packed_experts.weight1": "gate_and_up_proj.weight",
    ".packed_experts.weight2": "down_proj.weight",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--modality-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--family", choices=("model", "ema"), default="ema")
    parser.add_argument("--max-shard-gb", type=float, default=4.0)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate topology and meta-model shapes without writing weights.",
    )
    return parser.parse_args()


def _list_shards(directory: Path, family: str) -> list[Path]:
    shards = []
    prefix = f"{family}."
    for path in directory.glob(f"{family}.*-of-*.safetensors"):
        middle = path.name[len(prefix):-len(".safetensors")]
        left, right = middle.split("-of-", 1)
        shards.append((int(left), int(right), path))
    if not shards:
        raise FileNotFoundError(f"No {family} shards found in {directory}")
    totals = {total for _, total, _ in shards}
    if len(totals) != 1 or len(shards) != next(iter(totals)):
        raise RuntimeError(f"Incomplete or ambiguous {family} shard set: {shards[:3]}")
    return [path for _, _, path in sorted(shards)]


def _load_metadata(checkpoint: Path) -> dict:
    path = checkpoint / "checkpoint_metadata.json"
    metadata = json.loads(path.read_text())
    if not checkpoint.name.isdigit():
        raise RuntimeError(
            f"Checkpoint directory must be a numeric completed-step count: {checkpoint}"
        )
    directory_steps = int(checkpoint.name)
    metadata_steps = metadata.get("completed_optimizer_steps")
    if metadata_steps != directory_steps:
        raise RuntimeError(
            "Checkpoint step invariant failed: directory records "
            f"{directory_steps} completed updates but metadata records "
            f"{metadata_steps!r}"
        )
    required = {
        "checkpoint_layout": "heterogeneous_local_orig_v1",
        "tp_size": 1,
        "packed_experts": True,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"Unsupported checkpoint metadata {key}={metadata.get(key)!r}; "
                f"expected {expected!r}"
            )
    return metadata


def _target_shapes(base_model: Path, modality_config: Path):
    """Build only meta tensors to obtain authoritative native parameter shapes."""
    config = HunyuanImage3Config.from_pretrained(str(base_model))
    with torch.device("meta"):
        host = HunyuanImage3ForCausalMM(config, skip_load_module=["vae"])
    wrapper = HunyuanImageWrapper(host, vit_max_num_patch_per_side=64)
    tokenizer, _, _, registry = _build_tokenizer_and_registry(
        str(base_model), str(modality_config), model_name="hunyuan_image_3"
    )
    host.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    host.config.vocab_size = len(tokenizer)
    wrapper.config.llm_config.vocab_size = len(tokenizer)
    wrapper.set_modality_registry(registry)
    shapes = {key: tuple(value.shape) for key, value in host.state_dict().items()}
    return host.config, tokenizer, shapes


class ShardWriter:
    def __init__(self, output: Path, max_bytes: int):
        self.output = output
        self.max_bytes = max_bytes
        self.pending = {}
        self.pending_bytes = 0
        self.parts = []
        self.weight_map = {}
        self.total_size = 0

    def add(self, key: str, tensor: torch.Tensor):
        tensor = tensor.detach().to(device="cpu").contiguous()
        size = tensor.numel() * tensor.element_size()
        if self.pending and self.pending_bytes + size > self.max_bytes:
            self.flush()
        if key in self.weight_map or key in self.pending:
            raise RuntimeError(f"Duplicate exported tensor: {key}")
        self.pending[key] = tensor
        self.pending_bytes += size
        self.total_size += size

    def flush(self):
        if not self.pending:
            return
        index = len(self.parts) + 1
        path = self.output / f"model-part-{index:05d}.safetensors"
        save_file(self.pending, str(path), metadata={"format": "pt"})
        for key in self.pending:
            self.weight_map[key] = index
        self.parts.append(path)
        self.pending = {}
        self.pending_bytes = 0

    def finish(self):
        self.flush()
        total = len(self.parts)
        names = {}
        for index, old_path in enumerate(self.parts, 1):
            name = f"model-{index:05d}-of-{total:05d}.safetensors"
            new_path = old_path.with_name(name)
            os.replace(old_path, new_path)
            names[index] = name
        mapped = {key: names[index] for key, index in self.weight_map.items()}
        index = {"metadata": {"total_size": self.total_size}, "weight_map": mapped}
        (self.output / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n"
        )


def _copy_support_files(base_model: Path, output: Path):
    for name in (
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "chat_template.jinja",
    ):
        source = base_model / name
        if source.is_file():
            shutil.copy2(source, output / name)


def main():
    args = parse_args()
    configure_hunyuan_runtime()
    checkpoint = Path(args.checkpoint).resolve()
    base_model = Path(args.base_model).resolve()
    modality_config = Path(args.modality_config).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    metadata = _load_metadata(checkpoint)
    shard_paths = _list_shards(checkpoint, args.family)
    if len(shard_paths) != int(metadata["world_size"]):
        raise RuntimeError("Checkpoint shard count does not match world_size metadata")
    handles = [safe_open(str(path), framework="pt", device="cpu") for path in shard_paths]
    key_sets = [set(handle.keys()) for handle in handles]
    ordered_keys = sorted(set().union(*key_sets))

    config, tokenizer, target_shapes = _target_shapes(base_model, modality_config)
    if args.validate_only:
        summary = {
            "result": "PASS",
            "checkpoint_layout": metadata["checkpoint_layout"],
            "completed_optimizer_steps": metadata.get("completed_optimizer_steps"),
            "source_shards": len(shard_paths),
            "target_tensors": len(target_shapes),
            "tokenizer_size": len(tokenizer),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    config.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    _copy_support_files(base_model, output)
    shutil.copy2(modality_config, output / "modality_config.yaml")

    writer = ShardWriter(output, int(args.max_shard_gb * 1024**3))
    adapter_tensors = {}
    emitted = set()
    world_size = int(metadata["world_size"])
    ep_size = int(metadata["ep_size"])
    expert_dp_size = int(metadata["expert_fsdp_size"])

    def parts_for(full_key, ranks):
        parts = []
        for rank in ranks:
            if full_key not in key_sets[rank]:
                raise RuntimeError(f"Missing {full_key!r} from rank {rank}")
            part = handles[rank].get_tensor(full_key)
            if part.numel():
                parts.append(part.reshape(-1))
        return parts

    packed_keys = {
        key for key in ordered_keys if any(key.endswith(s) for s in PACKED_SUFFIXES)
    }
    for full_key in ordered_keys:
        if not full_key.startswith(PREFIX) or full_key in packed_keys:
            continue
        key = full_key[len(PREFIX):]
        if key not in target_shapes:
            raise RuntimeError(f"Checkpoint key has no native target: {key}")
        shape = target_shapes[key]
        parts = parts_for(full_key, range(world_size))
        target_numel = 1
        for dim in shape:
            target_numel *= dim
        if len(parts) == world_size and all(p.numel() == target_numel for p in parts):
            tensor = torch.stack([p.reshape(shape).float() for p in parts]).mean(0)
        else:
            if sum(p.numel() for p in parts) != target_numel:
                raise RuntimeError(f"Size mismatch for {key}: target={shape}")
            tensor = torch.cat(parts).reshape(shape)
        tensor = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
        if key.endswith("_pos_embed.embedding"):
            adapter_tensors[key] = tensor.contiguous()
        else:
            writer.add(key, tensor)
        emitted.add(key)

    for full_key in sorted(packed_keys):
        packed_suffix = next(s for s in PACKED_SUFFIXES if full_key.endswith(s))
        expert_leaf = PACKED_SUFFIXES[packed_suffix]
        model_prefix = full_key[len(PREFIX):-len(packed_suffix)]
        expert_prefix = f"{model_prefix}.experts."
        target_keys = sorted(
            (key for key in target_shapes
             if key.startswith(expert_prefix) and key.endswith(expert_leaf)),
            key=lambda key: int(key[len(expert_prefix):].split(".", 1)[0]),
        )
        if len(target_keys) % ep_size:
            raise RuntimeError(f"Expert count is not divisible by EP for {full_key}")
        local_experts = len(target_keys) // ep_size
        sample_shape = target_shapes[target_keys[0]]
        packed_shape = (local_experts, sample_shape[1], sample_shape[0])
        expected = local_experts * sample_shape[0] * sample_shape[1]
        for ep_rank in range(ep_size):
            ranks = [dp_rank * ep_size + ep_rank for dp_rank in range(expert_dp_size)]
            parts = parts_for(full_key, ranks)
            if sum(part.numel() for part in parts) != expected:
                raise RuntimeError(f"Packed expert size mismatch for {full_key} EP{ep_rank}")
            packed = torch.cat(parts).reshape(packed_shape)
            for local_idx in range(local_experts):
                key = target_keys[ep_rank * local_experts + local_idx]
                tensor = packed[local_idx].transpose(0, 1).to(torch.bfloat16).contiguous()
                writer.add(key, tensor)
                emitted.add(key)

    # The frozen VAE is intentionally absent from training checkpoints.  Copy
    # it from the licensed base checkpoint so the exported folder is runnable.
    base_index = json.loads((base_model / "model.safetensors.index.json").read_text())
    base_handles = {}
    for key, filename in base_index["weight_map"].items():
        if not key.startswith("vae."):
            continue
        handle = base_handles.setdefault(
            filename,
            safe_open(str(base_model / filename), framework="pt", device="cpu"),
        )
        tensor = handle.get_tensor(key)
        writer.add(key, tensor)
        emitted.add(key)

    expected_checkpoint = set(target_shapes)
    # target_shapes excludes VAE because the meta host deliberately skips it.
    missing = sorted(expected_checkpoint - emitted)
    unexpected = sorted(
        emitted - expected_checkpoint - {key for key in emitted if key.startswith("vae.")}
    )
    if missing or unexpected:
        raise RuntimeError(f"Export set mismatch: missing={missing[:20]} unexpected={unexpected[:20]}")
    writer.finish()
    if adapter_tensors:
        save_file(adapter_tensors, str(output / "modus_adapter.safetensors"), metadata={"format": "pt"})

    release_metadata = {
        "format": "modus_hunyuan_inference_v1",
        # Keep provenance without leaking a cluster-local filesystem path into
        # the public artifact.
        "source_checkpoint": checkpoint.name,
        "source_family": args.family,
        "completed_optimizer_steps": metadata.get("completed_optimizer_steps"),
        "dtype": "bfloat16",
        "modality_config": "modality_config.yaml",
        "runtime_profile": configure_hunyuan_runtime(),
        "weight_bytes": writer.total_size,
        "weight_shards": len(writer.parts),
    }
    (output / "modus_config.json").write_text(
        json.dumps(release_metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(release_metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
