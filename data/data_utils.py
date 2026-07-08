# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import math
import random
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

import torch
from torch.nn.attention.flex_attention import or_masks, and_masks


def create_sparse_mask(document_lens, split_lens, attn_modes, device):
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def full_and_noise_mask(b, h, q_idx, kv_idx):
        return (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx]) & (full_and_noise_seq_id[q_idx] >= 0)

    def remove_noise_mask(b, h, q_idx, kv_idx):
        return (~((noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx])))

    def sample_mask(b, h, q_idx, kv_idx):
        return document_id[q_idx] == document_id[kv_idx]

    full_and_noise_tmp = []
    noise_tmp = []

    for i, (length, model) in enumerate(zip(split_lens, attn_modes)):
        value = i if model in ['full', 'noise'] else -1
        full_and_noise_tmp.extend([value] * length)
        value_noise = i if model == 'noise' else -1
        noise_tmp.extend([value_noise] * length)

    full_and_noise_seq_id = torch.Tensor(full_and_noise_tmp).to(device)
    noise_seq_id = torch.Tensor(noise_tmp).to(device)

    document_id = torch.cat([torch.full((l,), i) for i, l in enumerate(document_lens, start=1)]).to(device)

    return and_masks(or_masks(causal_mask, full_and_noise_mask), remove_noise_mask, sample_mask)


def patchify(image, patch_size):
    p = patch_size
    c, h, w = image.shape
    assert h % p == 0 and w % p == 0
    image = image.reshape(c, h // p, p, w // p, p)
    image = torch.einsum("chpwq->hwpqc", image)
    image = image.reshape(-1, p**2 * c)
    return image


def get_flattened_position_ids_extrapolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    coords_h = torch.arange(0, num_patches_h)
    coords_w = torch.arange(0, num_patches_w)
    pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()
    return pos_ids


def get_flattened_position_ids_interpolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    boundaries = torch.arange(1 / max_num_patches_per_side, 1.0, 1 / max_num_patches_per_side)
    fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / num_patches_h)
    fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / num_patches_w)
    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
    pos_ids = (bucket_coords_h[:, None] * max_num_patches_per_side + bucket_coords_w).flatten()
    return pos_ids


def prepare_attention_mask_per_sample(split_lens, attn_modes, device="cpu"):
    """
    nested_split_lens: A list of N lists of ints. Each int indicates the length of a split within 
        a sample, where each sample contains multiple splits with different attn modes.
    nested_attn_modes: whether to use full attn in each split.
    """
    sample_len = sum(split_lens)
    attention_mask = torch.zeros((sample_len, sample_len), dtype=torch.bool, device=device)

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        assert attn_mode in ['causal', 'full', 'noise']
        if attn_mode == "causal":
            attention_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s), device=device).tril()
            attention_mask[csum:csum + s, :csum] = 1
        else:
            attention_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s))
            attention_mask[csum:csum + s, :csum] = 1
        csum += s

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        if attn_mode == "noise":
            attention_mask[:, csum : csum + s] = torch.zeros((sample_len, s))
            attention_mask[csum : csum + s, csum : csum + s] = torch.ones((s, s))
        csum += s

    attention_mask = torch.zeros_like(attention_mask, dtype=torch.float).masked_fill_(
        ~attention_mask, float("-inf")
    )

    return attention_mask


def split_integer_exp_decay(S, ng_sample_decay=1.0):
    if ng_sample_decay == 1.0:
        N = random.randint(1, S)
    else:
        base = (1 - ng_sample_decay) / (1 - math.pow(ng_sample_decay, S))
        p = [base * math.pow(ng_sample_decay, i) for i in range(S)]
        N = random.choices(list(range(1, S + 1)), p, k=1)[0]
    cumsum = [0] + sorted(random.sample(range(1, S), N - 1)) + [S]
    result = [cumsum[i+1] - cumsum[i] for i in range(len(cumsum) - 1)]
    return result, cumsum


def pil_img2rgb(image):
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")

    return image


def add_special_tokens(tokenizer):
    all_special_tokens = []
    for k, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v

    new_tokens = []

    if '<|im_start|>' not in all_special_tokens:
        new_tokens.append('<|im_start|>')

    if '<|im_end|>' not in all_special_tokens:
        new_tokens.append('<|im_end|>')

    if '<|vision_start|>' not in all_special_tokens:
        new_tokens.append('<|vision_start|>')

    if '<|vision_end|>' not in all_special_tokens:
        new_tokens.append('<|vision_end|>')

    if '<|caption_start|>' not in all_special_tokens:
        new_tokens.append('<|caption_start|>')
    if '<|caption_end|>' not in all_special_tokens:
        new_tokens.append('<|caption_end|>')

    if '<|depth_start|>' not in all_special_tokens:
        new_tokens.append('<|depth_start|>')
    if '<|depth_end|>' not in all_special_tokens:
        new_tokens.append('<|depth_end|>')
    
    if '<|normal_start|>' not in all_special_tokens:
        new_tokens.append('<|normal_start|>')
    if '<|normal_end|>' not in all_special_tokens:
        new_tokens.append('<|normal_end|>')

    num_new_tokens = tokenizer.add_tokens(new_tokens)
    bos_token_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
    eos_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    start_of_image = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    end_of_image = tokenizer.convert_tokens_to_ids('<|vision_end|>')

    start_of_caption = tokenizer.convert_tokens_to_ids('<|caption_start|>')
    end_of_caption = tokenizer.convert_tokens_to_ids('<|caption_end|>')
    start_of_depth = tokenizer.convert_tokens_to_ids('<|depth_start|>')
    end_of_depth = tokenizer.convert_tokens_to_ids('<|depth_end|>')
    start_of_normal = tokenizer.convert_tokens_to_ids('<|normal_start|>')
    end_of_normal = tokenizer.convert_tokens_to_ids('<|normal_end|>')

    new_token_ids = dict(
        bos_token_id=bos_token_id, 
        eos_token_id=eos_token_id, 
        start_of_image=start_of_image, 
        end_of_image=end_of_image, 
        start_of_caption=start_of_caption,
        end_of_caption=end_of_caption,
        start_of_depth=start_of_depth,
        end_of_depth=end_of_depth,
        start_of_normal=start_of_normal,
        end_of_normal=end_of_normal,
    )

    return tokenizer, new_token_ids, num_new_tokens


def _unique_preserve_order(tokens: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _format_tokens(token_format: str, start: int, end: int) -> List[str]:
    # Inclusive.
    return [token_format.format(i=i) for i in range(int(start), int(end) + 1)]


def add_codebook_tokens(
    tokenizer,
    *,
    token_groups: List[Dict[str, Any]],
    delimiter_tokens: List[str],
    extra_tokens: Optional[List[str]] = None,
) -> Tuple[Any, Dict[str, List[int]], Dict[str, int], int]:
    """
    Generic helper to add codebook-style tokens (DET/DINO/DINOLOCAL/custom) in one format.

    Returns:
      tokenizer,
      group_token_ids: dict[group_name] -> list[token_id] aligned to (start..end),
      delimiter_token_ids: dict[token_str] -> token_id,
      num_new_tokens
    """
    extra_tokens = extra_tokens or []

    group_tokens: Dict[str, List[str]] = {}
    all_code_tokens: List[str] = []
    for g in token_groups:
        name = str(g["name"])
        fmt = str(g["token_format"])
        start = int(g.get("start", 0))
        end = g.get("end")
        if end is None:
            raise ValueError(f"token_groups entry '{name}' missing 'end'")
        end = int(end)
        toks = _format_tokens(fmt, start, end)
        group_tokens[name] = toks
        all_code_tokens.extend(toks)

    tokens_to_add = _unique_preserve_order(all_code_tokens + list(delimiter_tokens) + list(extra_tokens))
    num_new_tokens = tokenizer.add_tokens(tokens_to_add)

    group_token_ids: Dict[str, List[int]] = {
        name: [int(tokenizer.convert_tokens_to_ids(t)) for t in toks]
        for name, toks in group_tokens.items()
    }
    delimiter_token_ids: Dict[str, int] = {
        str(t): int(tokenizer.convert_tokens_to_ids(str(t)))
        for t in _unique_preserve_order(list(delimiter_tokens) + list(extra_tokens))
    }
    return tokenizer, group_token_ids, delimiter_token_ids, int(num_new_tokens)


def add_det_special_tokens(tokenizer):
    """
    Add special tokens for detection box coordinates.
    Creates tokens for quantized coordinate values (0-999) to make them learnable.
    """
    token_groups = [
        {"name": "x1", "token_format": "<|x1_{i:03d}|>", "start": 0, "end": 999},
        {"name": "y1", "token_format": "<|y1_{i:03d}|>", "start": 0, "end": 999},
        {"name": "x2", "token_format": "<|x2_{i:03d}|>", "start": 0, "end": 999},
        {"name": "y2", "token_format": "<|y2_{i:03d}|>", "start": 0, "end": 999},
        {"name": "score", "token_format": "<|score_{i:02d}|>", "start": 0, "end": 99},
    ]
    delimiter_tokens = ["<|det_start|>", "<|det_end|>"]
    extra_tokens = ["<|box_start|>", "<|box_end|>"]

    tokenizer, group_ids, delim_ids, num_new_tokens = add_codebook_tokens(
        tokenizer,
        token_groups=token_groups,
        delimiter_tokens=delimiter_tokens,
        extra_tokens=extra_tokens,
    )

    coord_token_map: Dict[str, Dict[int, int]] = {}
    for coord_type in ["x1", "y1", "x2", "y2"]:
        coord_token_map[coord_type] = {i: int(tok_id) for i, tok_id in enumerate(group_ids[coord_type])}

    score_token_map: Dict[int, int] = {i: int(tok_id) for i, tok_id in enumerate(group_ids["score"])}

    det_delimiter_map = {
        "det_start": int(delim_ids["<|det_start|>"]),
        "det_end": int(delim_ids["<|det_end|>"]),
        "box_start": int(delim_ids["<|box_start|>"]),
        "box_end": int(delim_ids["<|box_end|>"]),
    }
    
    new_token_ids = dict(
        coord_token_map=coord_token_map,
        score_token_map=score_token_map,
        det_delimiter_map=det_delimiter_map,
    )

    return tokenizer, new_token_ids, num_new_tokens


def add_dino_feature_tokens(tokenizer, vocab_size=8192):
    """
    Add special tokens for DINO features.
    - 8192 code tokens: <|dino_0000|> ... <|dino_8191|>
    - 2 delimiter tokens: <|dino_start|>, <|dino_end|>
    Returns mapping dicts similar to add_det_special_tokens.
    """
    token_groups = [
        {"name": "dino", "token_format": "<|dino_{i:04d}|>", "start": 0, "end": int(vocab_size) - 1},
    ]
    delimiter_tokens = ["<|dino_start|>", "<|dino_end|>"]
    tokenizer, group_ids, delim_ids, num_new_tokens = add_codebook_tokens(
        tokenizer,
        token_groups=token_groups,
        delimiter_tokens=delimiter_tokens,
        extra_tokens=None,
    )

    dino_token_map = {i: int(tok_id) for i, tok_id in enumerate(group_ids["dino"])}
    dino_delimiter_map = {
        "dino_start": int(delim_ids["<|dino_start|>"]),
        "dino_end": int(delim_ids["<|dino_end|>"]),
    }

    new_token_ids = dict(
        dino_token_map=dino_token_map,
        dino_delimiter_map=dino_delimiter_map,
    )

    return tokenizer, new_token_ids, num_new_tokens


def add_dinolocal_feature_tokens(tokenizer, vocab_size=8192):
    """
    Add special tokens for DINO-Local features.
    - vocab_size code tokens: <|dinolocal_0000|> ... <|dinolocal_{vocab_size-1:04d}|>
    - 2 delimiter tokens: <|dinolocal_start|>, <|dinolocal_end|>
    Returns mapping dicts similar to add_det_special_tokens.
    """
    token_groups = [
        {"name": "dinolocal", "token_format": "<|dinolocal_{i:04d}|>", "start": 0, "end": int(vocab_size) - 1},
    ]
    delimiter_tokens = ["<|dinolocal_start|>", "<|dinolocal_end|>"]
    tokenizer, group_ids, delim_ids, num_new_tokens = add_codebook_tokens(
        tokenizer,
        token_groups=token_groups,
        delimiter_tokens=delimiter_tokens,
        extra_tokens=None,
    )

    dinolocal_token_map = {i: int(tok_id) for i, tok_id in enumerate(group_ids["dinolocal"])}
    dinolocal_delimiter_map = {
        "dinolocal_start": int(delim_ids["<|dinolocal_start|>"]),
        "dinolocal_end": int(delim_ids["<|dinolocal_end|>"]),
    }

    new_token_ids = dict(
        dinolocal_token_map=dinolocal_token_map,
        dinolocal_delimiter_map=dinolocal_delimiter_map,
    )

    return tokenizer, new_token_ids, num_new_tokens


def add_clip_feature_tokens(tokenizer, vocab_size=8192):
    """
    Add special tokens for CLIP features.
    - vocab_size code tokens: <|clip_0000|> ... <|clip_{vocab_size-1:04d}|>
    - 2 delimiter tokens: <|clip_start|>, <|clip_end|>
    Returns mapping dicts similar to add_det_special_tokens.
    """
    token_groups = [
        {"name": "clip", "token_format": "<|clip_{i:04d}|>", "start": 0, "end": int(vocab_size) - 1},
    ]
    delimiter_tokens = ["<|clip_start|>", "<|clip_end|>"]
    tokenizer, group_ids, delim_ids, num_new_tokens = add_codebook_tokens(
        tokenizer,
        token_groups=token_groups,
        delimiter_tokens=delimiter_tokens,
        extra_tokens=None,
    )

    clip_token_map = {i: int(tok_id) for i, tok_id in enumerate(group_ids["clip"])}
    clip_delimiter_map = {
        "clip_start": int(delim_ids["<|clip_start|>"]),
        "clip_end": int(delim_ids["<|clip_end|>"]),
    }

    new_token_ids = dict(
        clip_token_map=clip_token_map,
        clip_delimiter_map=clip_delimiter_map,
    )

    return tokenizer, new_token_ids, num_new_tokens


def add_imagebind_feature_tokens(tokenizer, vocab_size=8192):
    """
    Add special tokens for ImageBind (global) features.
    - vocab_size code tokens: <|imagebind_0000|> ... <|imagebind_{vocab_size-1:04d}|>
    - 2 delimiter tokens: <|imagebind_start|>, <|imagebind_end|>
    Returns mapping dicts similar to add_det_special_tokens.
    """
    token_groups = [
        {"name": "imagebind", "token_format": "<|imagebind_{i:04d}|>", "start": 0, "end": int(vocab_size) - 1},
    ]
    delimiter_tokens = ["<|imagebind_start|>", "<|imagebind_end|>"]
    tokenizer, group_ids, delim_ids, num_new_tokens = add_codebook_tokens(
        tokenizer,
        token_groups=token_groups,
        delimiter_tokens=delimiter_tokens,
        extra_tokens=None,
    )

    imagebind_token_map = {i: int(tok_id) for i, tok_id in enumerate(group_ids["imagebind"])}
    imagebind_delimiter_map = {
        "imagebind_start": int(delim_ids["<|imagebind_start|>"]),
        "imagebind_end": int(delim_ids["<|imagebind_end|>"]),
    }

    new_token_ids = dict(
        imagebind_token_map=imagebind_token_map,
        imagebind_delimiter_map=imagebind_delimiter_map,
    )

    return tokenizer, new_token_ids, num_new_tokens


def add_imagebindlocal_feature_tokens(tokenizer, vocab_size=8192):
    """
    Add special tokens for ImageBind-Local features.
    - vocab_size code tokens: <|imagebindlocal_0000|> ... <|imagebindlocal_{vocab_size-1:04d}|>
    - 2 delimiter tokens: <|imagebindlocal_start|>, <|imagebindlocal_end|>
    Returns mapping dicts similar to add_det_special_tokens.
    """
    token_groups = [
        {"name": "imagebindlocal", "token_format": "<|imagebindlocal_{i:04d}|>", "start": 0, "end": int(vocab_size) - 1},
    ]
    delimiter_tokens = ["<|imagebindlocal_start|>", "<|imagebindlocal_end|>"]
    tokenizer, group_ids, delim_ids, num_new_tokens = add_codebook_tokens(
        tokenizer,
        token_groups=token_groups,
        delimiter_tokens=delimiter_tokens,
        extra_tokens=None,
    )

    imagebindlocal_token_map = {i: int(tok_id) for i, tok_id in enumerate(group_ids["imagebindlocal"])}
    imagebindlocal_delimiter_map = {
        "imagebindlocal_start": int(delim_ids["<|imagebindlocal_start|>"]),
        "imagebindlocal_end": int(delim_ids["<|imagebindlocal_end|>"]),
    }

    new_token_ids = dict(
        imagebindlocal_token_map=imagebindlocal_token_map,
        imagebindlocal_delimiter_map=imagebindlocal_delimiter_map,
    )

    return tokenizer, new_token_ids, num_new_tokens


def add_special_tokens_text_image(tokenizer):
    all_special_tokens = []
    for k, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v

    new_tokens = []

    if '<|im_start|>' not in all_special_tokens:
        new_tokens.append('<|im_start|>')

    if '<|im_end|>' not in all_special_tokens:
        new_tokens.append('<|im_end|>')

    if '<|vision_start|>' not in all_special_tokens:
        new_tokens.append('<|vision_start|>')

    if '<|vision_end|>' not in all_special_tokens:
        new_tokens.append('<|vision_end|>')

    num_new_tokens = tokenizer.add_tokens(new_tokens)
    bos_token_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
    eos_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    start_of_image = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    end_of_image = tokenizer.convert_tokens_to_ids('<|vision_end|>')

    new_token_ids = dict(
        bos_token_id=bos_token_id, 
        eos_token_id=eos_token_id, 
        start_of_image=start_of_image, 
        end_of_image=end_of_image, 
    )

    return tokenizer, new_token_ids, num_new_tokens

def len2weight(x, loss_reduction='square'):
    if x == 0:
        return x
    if loss_reduction == 'token':
        return 1
    if loss_reduction == 'sample':
        return 1 / x
    if loss_reduction == 'square':
        return 1 / (x ** 0.5)
    raise NotImplementedError(loss_reduction)
