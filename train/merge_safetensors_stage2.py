import argparse
import os
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def list_shard_files(checkpoint_dir, model_name, total_shards=None):
    files = []
    for name in sorted(os.listdir(checkpoint_dir)):
        if not name.startswith(f"{model_name}.") or not name.endswith(".safetensors"):
            continue
        middle = name[len(model_name) + 1 : -len(".safetensors")]
        if "-of-" not in middle:
            continue
        shard_idx_str, total_str = middle.split("-of-", 1)
        if not (shard_idx_str.isdigit() and total_str.isdigit()):
            continue
        if total_shards is not None and int(total_str) != total_shards:
            continue
        files.append((int(shard_idx_str), int(total_str), os.path.join(checkpoint_dir, name)))

    if not files:
        raise FileNotFoundError(f"No sharded safetensors found for '{model_name}' in {checkpoint_dir}")

    totals = {total for _, total, _ in files}
    if len(totals) != 1:
        raise RuntimeError(f"Ambiguous shard counts for {model_name}: {sorted(totals)}")

    inferred_total = totals.pop()
    if len(files) != inferred_total:
        found = sorted(idx for idx, _, _ in files)
        raise RuntimeError(
            f"Expected {inferred_total} shards for {model_name}, found {len(files)}: {found}"
        )

    return [path for _, _, path in sorted(files)]


def infer_concat_dim_from_shapes(shapes):
    ref = shapes[0]
    if all(s == ref for s in shapes[1:]):
        return None

    candidates = []
    for dim in range(len(ref)):
        ok = True
        for shape in shapes:
            if len(shape) != len(ref):
                ok = False
                break
            for other_dim in range(len(ref)):
                if other_dim == dim:
                    continue
                if shape[other_dim] != ref[other_dim]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            candidates.append(dim)

    if len(candidates) == 1:
        return candidates[0]
    if 0 in candidates:
        return 0
    if 1 in candidates:
        return 1
    raise RuntimeError(f"Ambiguous concat dim for shapes {shapes} (candidates={candidates})")


def merge_tensors_for_key(key, parts):
    shapes = [tuple(part.shape) for part in parts]

    if all(shape == shapes[0] for shape in shapes[1:]):
        ref = parts[0]
        if all(torch.equal(ref, part) for part in parts[1:]):
            return ref.clone(), "replicated"

        for try_dim in (0, 1):
            if try_dim >= ref.ndim:
                continue
            split_sizes = [part.shape[try_dim] for part in parts]
            candidate = torch.cat(parts, dim=try_dim)
            roundtrip = torch.split(candidate, split_sizes, dim=try_dim)
            if all(torch.equal(chunk, part) for chunk, part in zip(roundtrip, parts)):
                return candidate, f"sharded_equal_shape_dim{try_dim}"

        raise RuntimeError(
            f"Key '{key}' has equal shapes across shards but tensors are not identical "
            "and no safe concat dimension could be inferred."
        )

    concat_dim = infer_concat_dim_from_shapes(shapes)
    merged = torch.cat(parts, dim=concat_dim)
    return merged, f"sharded_dim{concat_dim}"


def should_skip_key(key, skip_pos_embed):
    if skip_pos_embed and key.endswith("pos_embed"):
        return True
    return False


def merge_family(checkpoint_dir, model_name, output_name=None, skip_pos_embed=True, total_shards=None):
    shard_files = list_shard_files(checkpoint_dir, model_name, total_shards=total_shards)

    shard_handles = [safe_open(path, framework="pt", device="cpu") for path in shard_files]
    shard_key_sets = [set(handle.keys()) for handle in shard_handles]

    ordered_keys = list(shard_handles[0].keys())
    for extra_keys in shard_key_sets[1:]:
        for key in extra_keys:
            if key not in ordered_keys:
                ordered_keys.append(key)

    output_path = os.path.join(checkpoint_dir, output_name or f"{model_name}.safetensors")
    full_state_dict = {}
    mode_counts = defaultdict(int)
    for key_idx, key in enumerate(ordered_keys, start=1):
        if should_skip_key(key, skip_pos_embed):
            continue
        parts = []
        for shard_idx, (handle, key_set) in enumerate(zip(shard_handles, shard_key_sets), start=1):
            if key not in key_set:
                raise RuntimeError(
                    f"Key '{key}' is missing from shard {shard_idx} / {len(shard_handles)}"
                )
            parts.append(handle.get_tensor(key))
        merged_tensor, mode = merge_tensors_for_key(key, parts)
        full_state_dict[key] = merged_tensor
        mode_counts[mode] += 1
        if key_idx % 200 == 0:
            print(f"[merge] collected {key_idx}/{len(ordered_keys)} keys")

    save_file(full_state_dict, output_path)

    print(f"Saved {output_path}")
    for mode, count in sorted(mode_counts.items()):
        print(f"  {mode}: {count} keys")


def main():
    parser = argparse.ArgumentParser(
        description="Merge FSDP sharded safetensors checkpoints without blindly concatenating replicated tensors."
    )
    parser.add_argument("ckpt_dir", help="Checkpoint directory containing model.*-of-*.safetensors shards")
    parser.add_argument("--model-name", default="model", help="Checkpoint family to merge, e.g. model or ema")
    parser.add_argument("--output-name", default=None, help="Output filename; defaults to <model-name>.safetensors")
    parser.add_argument(
        "--keep-pos-embed",
        action="store_true",
        help="Keep positional embedding tensors instead of dropping keys ending with 'pos_embed'",
    )
    parser.add_argument("--total-shards", type=int, default=None, help="Optional expected shard count")
    args = parser.parse_args()

    merge_family(
        checkpoint_dir=args.ckpt_dir,
        model_name=args.model_name,
        output_name=args.output_name,
        skip_pos_embed=not args.keep_pos_embed,
        total_shards=args.total_shards,
    )


if __name__ == "__main__":
    main()
