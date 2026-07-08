"""
Config-driven tokenizer setup.

All modality tokens (delimiters, codebook entries) are specified in the modality
YAML and added here.  No hard-coded ``use_det`` / ``use_dino`` flags.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.modality import TokenRange, infer_contiguous_token_range
from data.data_utils import add_special_tokens


@dataclass
class TokenizerArtifacts:
    """Return value of :func:`build_tokenizer_and_special_tokens`."""
    tokenizer: Any
    new_token_ids: Dict[str, Any]
    token_ranges: Dict[str, TokenRange]
    # Explicit dispersed CE-token id sets, for modalities that set
    # ``dispersed_code_tokens: true`` (e.g. cocodet). modality name -> id list.
    code_token_ids: Dict[str, List[int]]


def load_base_tokenizer(*, model_args: Any, training_args: Any):
    """Load the base tokenizer (before any token additions).

    Uses the standard Qwen2Tokenizer.
    """
    from modeling.qwen2.tokenization_qwen2 import Qwen2Tokenizer

    pretrained_path = (
        model_args.model_path
        if getattr(training_args, "finetune_from_hf", False)
        else model_args.llm_path
    )
    return Qwen2Tokenizer.from_pretrained(pretrained_path)


# ── internal helpers ─────────────────────────────────────────────────────────

def _add_tokens(tokenizer, tokens: List[str]) -> None:
    """Add tokens in order, deduplicating while preserving first occurrence."""
    seen: set = set()
    ordered: List[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    tokenizer.add_tokens(ordered)


def _ensure_token_key(
    tokenizer,
    new_token_ids: Dict[str, Any],
    *,
    token_key: str,
    token_str: Optional[str],
) -> None:
    """
    Make sure ``new_token_ids[token_key]`` exists, creating the token if needed.
    """
    if token_key in new_token_ids:
        return
    if hasattr(tokenizer, token_key):
        new_token_ids[token_key] = int(getattr(tokenizer, token_key))
        return
    if token_str is None:
        raise KeyError(
            f"Token key '{token_key}' not in tokenizer or new_token_ids and no token_str provided."
        )
    try:
        tokenizer.add_special_tokens({"additional_special_tokens": [token_str]})
    except Exception:
        tokenizer.add_tokens([token_str])
    new_token_ids[token_key] = int(tokenizer.convert_tokens_to_ids(token_str))


def _code_tokens_from_cfg(m: Dict[str, Any]) -> List[str]:
    """Build the list of code-token strings from a modality config dict."""

    # Preferred: explicit groups (e.g. DET coordinate tokens).
    groups = m.get("code_token_groups")
    if groups is not None:
        out: List[str] = []
        for g in groups:
            if not isinstance(g, dict):
                g = dict(g)
            token_format = str(g["token_format"])
            start = int(g.get("start", 0))
            end = int(g["end"])
            for i in range(start, end + 1):
                out.append(token_format.format(
                    i=i,
                    prefix=m.get("code_token_prefix", m.get("name")),
                ))
        return out

    # Fallback: contiguous vocab with format / prefix.
    vocab = m.get("code_vocab_size")
    if vocab is None:
        raise ValueError(
            f"Codebook modality '{m.get('name')}' needs 'code_vocab_size' or 'code_token_groups'."
        )
    vocab = int(vocab)
    token_format = str(m.get("code_token_format", "<|{prefix}_{i:04d}|>"))
    prefix = str(m.get("code_token_prefix", m.get("name")))
    return [token_format.format(prefix=prefix, i=i) for i in range(vocab)]


def _add_modality_tokens(
    tokenizer,
    *,
    m: Dict[str, Any],
    new_token_ids: Dict[str, Any],
    token_ranges: Dict[str, TokenRange],
    code_token_ids_out: Dict[str, List[int]],
    deferred_tokens: List[Any],
) -> None:
    """
    Add all tokens for one modality entry (text/image/codebook) from its YAML dict.

    For **codebook** modalities the addition order is *code tokens first, then
    delimiter / extra tokens*.  This matches the ordering used when the
    original checkpoint was trained so that token IDs remain identical.
    """
    kind = str(m.get("kind", "text"))
    start_key = str(m["start_token_key"])
    end_key = str(m["end_token_key"])

    if kind != "codebook":
        # Image/text modalities: resolve already-existing start/end tokens now
        # (shared start_of_image, or start_of_depth/normal added by
        # add_special_tokens). Any genuinely NEW token (e.g. a per-modality
        # start_of_seg/canny/samseg/samedge under REPLACE) is DEFERRED and added
        # at the very tail in build_tokenizer_and_special_tokens — so introducing
        # an image modality with its own start token does NOT shift the ids of
        # codebook tokens added later in the loop, preserving cross-stage ckpt
        # alignment.
        vocab = tokenizer.get_vocab()
        for token_key, token_str in ((start_key, m.get("start_token")), (end_key, m.get("end_token"))):
            if token_key in new_token_ids:
                continue
            if token_str is not None and str(token_str) in vocab:
                new_token_ids[token_key] = int(vocab[str(token_str)])
            elif token_str is not None:
                deferred_tokens.append((token_key, str(token_str)))
            else:
                _ensure_token_key(tokenizer, new_token_ids, token_key=token_key, token_str=None)
        return

    # ── codebook modality ────────────────────────────────────────────────
    code_tokens = _code_tokens_from_cfg(m)

    start_tok = m.get("start_token")
    end_tok = m.get("end_token")
    delim_tokens: List[str] = []
    if start_tok is not None:
        delim_tokens.append(str(start_tok))
    if end_tok is not None:
        delim_tokens.append(str(end_tok))
    delim_tokens.extend(str(t) for t in (m.get("extra_tokens") or []))

    # One batch: code tokens first, delimiters after → matches checkpoint ordering.
    _add_tokens(tokenizer, code_tokens + delim_tokens)

    # Populate delimiter keys.
    if start_tok is not None and start_key not in new_token_ids:
        new_token_ids[start_key] = int(tokenizer.convert_tokens_to_ids(str(start_tok)))
    elif start_key not in new_token_ids:
        _ensure_token_key(tokenizer, new_token_ids, token_key=start_key, token_str=None)
    if end_tok is not None and end_key not in new_token_ids:
        new_token_ids[end_key] = int(tokenizer.convert_tokens_to_ids(str(end_tok)))
    elif end_key not in new_token_ids:
        _ensure_token_key(tokenizer, new_token_ids, token_key=end_key, token_str=None)

    name = str(m["name"])
    token_ids = [int(tokenizer.convert_tokens_to_ids(t)) for t in code_tokens]
    if bool(m.get("dispersed_code_tokens", False)):
        # Dispersed alphabet (e.g. cocodet: reused det coords at low vocab ids +
        # new class tokens at the tail) — cannot be a single contiguous range.
        # Record the explicit CE-token id set = all code tokens + the end
        # delimiter (a CE target / stop token). The start delimiter is
        # input-only, so it is excluded.
        ids = list(token_ids)
        end_id = new_token_ids.get(end_key)
        if end_id is not None:
            ids.append(int(end_id))
        code_token_ids_out[name] = ids
    else:
        # Record contiguous code-token range.
        token_ranges[name] = infer_contiguous_token_range(token_ids)


# ── public API ───────────────────────────────────────────────────────────────

def build_tokenizer_and_special_tokens(
    tokenizer,
    *,
    modalities_cfg: Dict[str, Any],
) -> TokenizerArtifacts:
    """
    Fully config-driven tokenizer setup.

    1. Adds universal special tokens (``<|im_start|>``, ``<|vision_start|>``, …)
       via the legacy ``add_special_tokens`` helper (for checkpoint compat).
    2. Iterates over ``modalities_cfg["modalities"]`` and adds delimiter / code
       tokens for each entry.

    Returns a :class:`TokenizerArtifacts` with the final tokenizer, a dict of
    all special-token IDs, and a dict of code-token ranges for codebook modalities.
    """
    token_ranges: Dict[str, TokenRange] = {}
    code_token_ids: Dict[str, List[int]] = {}
    deferred_tokens: List[Any] = []

    # Step 1: universal special tokens (checkpoint-compatible ordering).
    tokenizer, new_token_ids, _num = add_special_tokens(tokenizer)

    # Step 2: per-modality tokens from YAML.
    modalities = modalities_cfg.get("modalities", modalities_cfg)
    for m in modalities:
        if not isinstance(m, dict):
            m = dict(m)
        _add_modality_tokens(
            tokenizer,
            m=m,
            new_token_ids=new_token_ids,
            token_ranges=token_ranges,
            code_token_ids_out=code_token_ids,
            deferred_tokens=deferred_tokens,
        )

    # Step 3: append deferred NEW image-modality start/end tokens at the tail
    # (after all codebook tokens) so they never shift existing ids across stages.
    for token_key, token_str in deferred_tokens:
        if token_key not in new_token_ids:
            _ensure_token_key(tokenizer, new_token_ids, token_key=token_key, token_str=token_str)

    return TokenizerArtifacts(
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
        token_ranges=token_ranges,
        code_token_ids=code_token_ids,
    )
