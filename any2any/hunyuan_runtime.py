"""Validated runtime defaults for the released MODUS-Hunyuan checkpoint."""

from __future__ import annotations

import os


# These values describe the serialization and image geometry used to train the
# release checkpoint.  They are defaults, not global package side effects: the
# function is called only after ``--backend hunyuan`` is selected, and an
# explicit user setting remains authoritative.
_RUNTIME_DEFAULTS = {
    "HUNYUAN_TP_SIZE": "1",
    "HUNYUAN_REPLICATED_DP_MODE": "mean",
    "HUNYUAN_WIRE_FORMAT_V2": "1",
    "HUNYUAN_STRICT_NATIVE_IMAGE_BLOCK": "1",
    "HUNYUAN_NATIVE_CHAT_SERIALIZATION": "1",
    "HUNYUAN_INFER_NATIVE_CHAT": "1",
    "HUNYUAN_TRAIN_TIMESTEP_SCALE": "1000",
    "HUNYUAN_INFER_VIT_MAX_PATCH": "64",
    "HUNYUAN_INFER_VIT_MAX_SIZE": "512",
    "HUNYUAN_INFER_VIT_TRAIN_MAX": "512",
    "HUNYUAN_INFER_VIT_TRAIN_MIN": "224",
    "HUNYUAN_INFER_VAE_MAX_SIZE": "1024",
    "HUNYUAN_INFER_VAE_MIN_SIZE": "512",
    "HUNYUAN_INFER_MAX_LATENT_SIZE": "64",
    "HUNYUAN_INFER_IMAGE_ROPE_BLOCK": "0",
    "HUNYUAN_INFER_PIPE_PARITY": "1",
    "HUNYUAN_INFER_VIT_2D_ROPE": "1",
    "HUNYUAN_INFER_PREFILL_FULL_INCLUSIVE": "1",
    "HUNYUAN_TIMESTEP_SLOT_FILL": "1",
    "HUNYUAN_DISABLE_AR_2D_ROPE": "0",
    "HUNYUAN_USE_FORWARD_CODEBOOK_POS_EMBED": "1",
    "MODUS_NO_MEAN_RESIZING": "1",
}


def configure_hunyuan_runtime() -> dict[str, str]:
    """Apply and return the effective Hunyuan inference environment.

    Silent environment drift is especially dangerous for wire format and
    positional encoding.  Reject conflicting inherited values by default;
    controlled A/B diagnostics may opt out explicitly.
    """
    allow_override = os.environ.get("MODUS_HUNYUAN_ALLOW_RUNTIME_OVERRIDE", "0") == "1"
    for key, value in _RUNTIME_DEFAULTS.items():
        existing = os.environ.get(key)
        if existing is not None and existing != value and not allow_override:
            raise RuntimeError(
                f"{key}={existing!r} conflicts with the released checkpoint "
                f"profile ({value!r}); set MODUS_HUNYUAN_ALLOW_RUNTIME_OVERRIDE=1 "
                "only for an intentional diagnostic."
            )
        os.environ.setdefault(key, value)
    return {key: os.environ[key] for key in _RUNTIME_DEFAULTS}
