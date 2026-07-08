from __future__ import annotations

from dataclasses import dataclass, field as dc_field, replace as dc_replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


TokenRange = Tuple[int, int]  # (base_id, length)


@dataclass(frozen=True)
class LossConfig:
    """Per-modality loss configuration (lives in the modality YAML)."""
    reweight: bool = False
    reweight_min_w: float = 0.005
    reweight_det_vocab_dim: int = 4000   # only used when reweight=True for coordinate codebooks


@dataclass(frozen=True)
class ModalitySpec:
    """
    Config-driven definition of a single modality.

    Every tunable per-modality knob lives here so that the training script,
    model forward, and dataset packing can all be driven from one YAML.
    """

    name: str
    id: int
    start_token_key: str
    end_token_key: str
    kind: str = "text"  # text | image | codebook

    # Token range ``(base, length)`` for CE-loss vocab slicing.
    # Codebook modalities get their own range from the tokenizer; text
    # modalities are auto-assigned ``(0, text_vocab_end)`` in
    # ``ModalityRegistry.from_config``.  Image modalities (ViT/VAE)
    # never produce CE tokens, so they keep ``None``.
    code_token_range: Optional[TokenRange] = None

    # Explicit, possibly **non-contiguous** set of CE-loss token ids. Used for
    # modalities whose alphabet is dispersed across the vocab and therefore
    # cannot be expressed as a single ``(base, length)`` range (e.g. cocodet,
    # which reuses det's low-vocab coordinate tokens + new high-vocab class/end
    # tokens). When set, the model does gather-CE over exactly these ids instead
    # of the contiguous-slice path. ``None`` for every existing modality.
    code_token_ids: Optional[Tuple[int, ...]] = None

    # Learnable positional embedding for code tokens.
    pos_embed_size: Optional[int] = None
    apply_pos_embed_in_forward: bool = False

    # Image representation flags (used by dataset packing and inferencer).
    represent_vit: bool = True
    represent_vae: bool = False

    # Which other modalities may condition this one during training.
    # ``None`` means "use all available"; an explicit list restricts.
    conditions: Optional[Tuple[str, ...]] = None
    # Optional sampling probabilities aligned with ``conditions`` order.
    # If omitted, the dataset uses a uniform distribution.
    condition_probs: Optional[Tuple[float, ...]] = None

    # Per-modality loss parameters (CE smoothing / reweighting).
    loss: LossConfig = LossConfig()

    # Override the model attribute name for the pos-embed (for checkpoint compat).
    # Defaults to ``name`` (e.g. modality "dino" → attr ``dino_pos_embed``).
    # Set to e.g. "grounding" for det → ``grounding_pos_embed``.
    pos_embed_name: Optional[str] = None

    # 2D spatial shape for codebook modalities (h, w).
    # None → 1D sequential (uses text region of build_2d_rope, gets sequential positions).
    # Set e.g. (32, 32) for dinolocal, (28, 28) for clip → gets 2D RoPE same as VAE latents.
    codebook_spatial_shape: Optional[Tuple[int, int]] = None

    # External tokenizer for inference decoding/encoding (e.g. DINO VQVAE).
    external_tokenizer_repo: Optional[str] = None
    external_tokenizer_kind: str = "vqvae"

    # ── Inference-time configuration ─────────────────────────────────────
    # These fields drive the unified inference pipeline so that the
    # inferencer does not need per-modality if/elif chains.

    # Decode method: "auto" resolves from ``kind`` (image→image, text→text).
    # Explicit values: "image", "text", "detection", "dino".
    inference_decode_method: str = "auto"

    # Maximum AR tokens for text / codebook decoding (None → inferencer default).
    inference_max_tokens: Optional[int] = None

    # Which CFG unconditional context to use when this modality is a *target*:
    #   "text"  → cfg_text_context (drop text, keep image)   e.g. detection
    #   "img"   → cfg_img_context  (drop image, keep text)   e.g. dino
    #   "both"  → dual CFG with both contexts                e.g. image generation
    #   "none"  → no CFG                                     e.g. plain text
    inference_cfg_uncond: str = "auto"

    # Whether to prepend "[start {name} x10]" instruction before generation.
    inference_add_instruction: bool = True

    # Default CFG image scale for this modality when it is the *target*.
    # ``None`` → use the global default from the base config.
    inference_cfg_img_scale: Optional[float] = None


class ModalityRegistry:
    """Runtime registry consumed by dataset packing, model forward, and inferencer."""

    def __init__(self, specs: Iterable[ModalitySpec]):
        specs = list(specs)
        if len(specs) == 0:
            raise ValueError("ModalityRegistry requires at least one ModalitySpec.")

        self._by_name: Dict[str, ModalitySpec] = {}
        self._by_id: Dict[int, ModalitySpec] = {}
        for s in specs:
            if s.name in self._by_name:
                raise ValueError(f"Duplicate modality name: {s.name}")
            if s.id in self._by_id:
                raise ValueError(f"Duplicate modality id: {s.id}")
            self._by_name[s.name] = s
            self._by_id[s.id] = s

    # ── lookups ──────────────────────────────────────────────────────────

    @property
    def specs(self) -> List[ModalitySpec]:
        return list(self._by_name.values())

    def get(self, name: str) -> ModalitySpec:
        return self._by_name[name]

    def get_by_id(self, modality_id: int) -> ModalitySpec:
        return self._by_id[modality_id]

    def name_to_id(self) -> Dict[str, int]:
        return {k: v.id for k, v in self._by_name.items()}

    def id_to_name(self) -> Dict[int, str]:
        return {k: v.name for k, v in self._by_id.items()}

    def modality_name(self, modality_id: int) -> str:
        spec = self._by_id.get(modality_id)
        return spec.name if spec is not None else f"unknown_{modality_id}"

    def start_token_key(self, name: str) -> str:
        return self._by_name[name].start_token_key

    def end_token_key(self, name: str) -> str:
        return self._by_name[name].end_token_key

    def start_token_id(self, new_token_ids: Mapping[str, Any], name: str) -> int:
        return int(new_token_ids[self.start_token_key(name)])

    def end_token_id(self, new_token_ids: Mapping[str, Any], name: str) -> int:
        return int(new_token_ids[self.end_token_key(name)])

    def code_token_range(self, name: str) -> Optional[TokenRange]:
        return self._by_name[name].code_token_range

    def has_codebook_modalities(self) -> bool:
        """True if any registered modality is ``kind == 'codebook'``."""
        return any(s.kind == "codebook" for s in self._by_name.values())

    def conditions_for(self, target_modality: str) -> Optional[List[str]]:
        """
        Return the allowed conditioning modalities for *target_modality*.

        ``None`` means "no restriction — use all available".
        """
        spec = self._by_name.get(target_modality)
        if spec is None or spec.conditions is None:
            return None
        return list(spec.conditions)

    def condition_probs_for(self, target_modality: str) -> Optional[List[float]]:
        """
        Return optional condition sampling probabilities for *target_modality*.

        The returned probabilities align with ``conditions_for(target_modality)``.
        ``None`` means "use uniform condition sampling".
        """
        spec = self._by_name.get(target_modality)
        if spec is None or spec.condition_probs is None:
            return None
        return list(spec.condition_probs)

    def resolve_decode_method(self, name: str) -> str:
        """Return the concrete decode method for *name* (resolves ``"auto"``)."""
        spec = self._by_name[name]
        if spec.inference_decode_method != "auto":
            return spec.inference_decode_method
        if spec.kind == "image":
            return "image"
        if spec.kind == "text":
            return "text"
        # codebook — guess from name
        if "det" in spec.name:
            return "detection"
        if spec.name == "dinolocal":
            return "dinolocal"
        if "dino" in spec.name:
            return "dino"
        return "text"

    def resolve_cfg_uncond(self, name: str) -> str:
        """Return the concrete CFG-uncond context type (resolves ``"auto"``)."""
        spec = self._by_name[name]
        if spec.inference_cfg_uncond != "auto":
            return spec.inference_cfg_uncond
        dm = self.resolve_decode_method(name)
        if dm == "image":
            return "both"
        if dm == "detection":
            return "text"
        if dm in ("dino", "dinolocal"):
            return "img"
        return "none"

    def resolve_cfg_img_scale(self, name: str, default: float = 2.0) -> float:
        """Return the per-modality CFG image scale, or *default* if not specified."""
        spec = self._by_name[name]
        if spec.inference_cfg_img_scale is not None:
            return spec.inference_cfg_img_scale
        return default

    def needs_external_tokenizer(self, name: str) -> bool:
        """Return True if the modality has an external tokenizer (e.g. DINO VQVAE)."""
        spec = self._by_name.get(name)
        return spec is not None and spec.external_tokenizer_repo is not None

    def modalities_with_forward_pos_embed(self) -> List[ModalitySpec]:
        return [
            s
            for s in self._by_name.values()
            if s.pos_embed_size is not None and s.apply_pos_embed_in_forward
        ]

    # ── construction from YAML dict ─────────────────────────────────────

    @staticmethod
    def from_config(
        cfg: Any,
        *,
        token_ranges: Optional[Mapping[str, TokenRange]] = None,
        code_token_ids: Optional[Mapping[str, List[int]]] = None,
    ) -> "ModalityRegistry":
        """
        Build a registry from the parsed YAML config dict.

        Expected structure::

            cfg["modalities"]: list[{name, id, start_token_key, end_token_key, kind, ...}]

        ``token_ranges`` maps modality name → ``(base, length)`` and is typically
        computed after tokenizer token additions.
        """
        if cfg is None:
            raise ValueError("cfg is required")

        # Accept either ``{"modalities": [...]}`` or a bare list.
        modalities = cfg.get("modalities") if isinstance(cfg, dict) else cfg
        if modalities is None:
            raise ValueError("No 'modalities' key found in cfg")

        specs: List[ModalitySpec] = []
        token_ranges = dict(token_ranges or {})
        code_token_ids = dict(code_token_ids or {})

        for m in modalities:
            name = str(m["name"])

            # Resolve code_token_range: prefer runtime-computed, fall back to cfg.
            _range = token_ranges.get(name)
            if _range is None and m.get("code_token_range") is not None:
                _range = tuple(m["code_token_range"])

            # Resolve explicit (possibly dispersed) code_token_ids, if any.
            _ids = code_token_ids.get(name)
            if _ids is None and m.get("code_token_ids") is not None:
                _ids = m["code_token_ids"]
            _ids = tuple(int(i) for i in _ids) if _ids is not None else None

            # pos_embed_size may be None.
            _pos = m.get("pos_embed_size")
            pos_embed_size = int(_pos) if _pos is not None else None

            # Parse conditions (list of strings or None).
            _conds = m.get("conditions")
            conditions = tuple(_conds) if _conds is not None else None
            _cond_probs = m.get("condition_probs")
            condition_probs = tuple(float(x) for x in _cond_probs) if _cond_probs is not None else None
            if condition_probs is not None:
                if conditions is None:
                    raise ValueError(
                        f"Modality '{name}' sets 'condition_probs' without 'conditions'."
                    )
                if len(condition_probs) != len(conditions):
                    raise ValueError(
                        f"Modality '{name}' has {len(conditions)} conditions but "
                        f"{len(condition_probs)} condition_probs."
                    )
                if any(p < 0.0 for p in condition_probs):
                    raise ValueError(f"Modality '{name}' has negative values in condition_probs.")
                probs_sum = float(sum(condition_probs))
                if abs(probs_sum - 1.0) > 1e-6:
                    raise ValueError(
                        f"Modality '{name}' condition_probs must sum to 1.0, got {probs_sum}."
                    )

            # Parse per-modality loss config.
            _loss_dict = m.get("loss") or {}
            loss_cfg = LossConfig(
                reweight=bool(_loss_dict.get("reweight", False)),
                reweight_min_w=float(_loss_dict.get("reweight_min_w", 0.005)),
                reweight_det_vocab_dim=int(_loss_dict.get("reweight_det_vocab_dim", 4000)),
            )

            _pe_name = m.get("pos_embed_name")
            pe_name = str(_pe_name) if _pe_name is not None else None

            # 2D spatial shape for codebook modalities.
            _shape = m.get("codebook_spatial_shape")
            codebook_spatial_shape = tuple(int(x) for x in _shape) if _shape is not None else None

            # Inference-time fields (optional in YAML; sensible defaults).
            _infer = m.get("inference") or {}
            _infer_decode = str(_infer.get("decode_method", m.get("inference_decode_method", "auto")))
            _infer_max = _infer.get("max_tokens", m.get("inference_max_tokens"))
            _infer_cfg = str(_infer.get("cfg_uncond", m.get("inference_cfg_uncond", "auto")))
            _infer_add_instr = bool(_infer.get("add_instruction", m.get("inference_add_instruction", True)))
            _infer_cfg_img = _infer.get("cfg_img_scale", m.get("inference_cfg_img_scale"))
            _infer_cfg_img = float(_infer_cfg_img) if _infer_cfg_img is not None else None

            spec = ModalitySpec(
                name=name,
                id=int(m["id"]),
                start_token_key=str(m["start_token_key"]),
                end_token_key=str(m["end_token_key"]),
                kind=str(m.get("kind", "text")),
                code_token_range=_range,
                code_token_ids=_ids,
                pos_embed_size=pos_embed_size,
                apply_pos_embed_in_forward=bool(m.get("apply_pos_embed_in_forward", False)),
                represent_vit=bool(m.get("represent_vit", True)),
                represent_vae=bool(m.get("represent_vae", False)),
                conditions=conditions,
                condition_probs=condition_probs,
                loss=loss_cfg,
                pos_embed_name=pe_name,
                codebook_spatial_shape=codebook_spatial_shape,
                external_tokenizer_repo=m.get("external_tokenizer_repo"),
                external_tokenizer_kind=str(m.get("external_tokenizer_kind", "vqvae")),
                inference_decode_method=_infer_decode,
                inference_max_tokens=int(_infer_max) if _infer_max is not None else None,
                inference_cfg_uncond=_infer_cfg,
                inference_add_instruction=_infer_add_instr,
                inference_cfg_img_scale=_infer_cfg_img,
            )
            specs.append(spec)

        # ── Auto-assign text token range to text-kind modalities ─────────
        # Find text_vocab_end = min base across all codebook ranges.
        text_vocab_end: Optional[int] = None
        for s in specs:
            if s.code_token_range is not None:
                cb_base, _ = s.code_token_range
                text_vocab_end = int(cb_base) if text_vocab_end is None else min(text_vocab_end, int(cb_base))

        if text_vocab_end is not None:
            text_range: TokenRange = (0, text_vocab_end)
            specs = [
                dc_replace(s, code_token_range=text_range)
                if s.kind == "text" and s.code_token_range is None else s
                for s in specs
            ]

        return ModalityRegistry(specs)


def infer_contiguous_token_range(token_ids: List[int]) -> TokenRange:
    """Given a list of token IDs, infer ``(base, length)`` and validate contiguity."""
    if len(token_ids) == 0:
        raise ValueError("token_ids cannot be empty")
    token_ids_sorted = sorted(int(x) for x in token_ids)
    base = token_ids_sorted[0]
    length = len(token_ids_sorted)
    expected = list(range(base, base + length))
    if token_ids_sorted != expected:
        raise ValueError(
            "Token IDs are not contiguous; cannot represent as a range efficiently. "
            f"base={base}, length={length}, first10={token_ids_sorted[:10]}"
        )
    return base, length
