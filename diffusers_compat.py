"""diffusers 0.31 compatibility shim for the 4M (fourm) VQ tokenizers.

Import this module **first**, before any import that pulls in `fourm` (the
eval/online-val codebooks). It rewires two symbols that diffusers moved
between 0.20 (what fourm was written against) and 0.31 (what the training
stack pins, because huggingface-hub >= 0.28 removed `cached_download` which
diffusers 0.20 imports):

  1. `diffusers.models.unet_2d_blocks`  ->  `diffusers.models.unets.unet_2d_blocks`
     (fourm: fourm/vq/models/{uvit,lm_models}.py)
  2. `diffusers.utils.randn_tensor`     restored from `diffusers.utils.torch_utils`
     (fourm: fourm/vq/scheduling/scheduling_{ddpm,ddim,pndm}.py)

These are the only two fourm imports that moved (verified by scanning every
`from diffusers ...` in fourm/vq against diffusers 0.31). Everything else
(configuration_utils, modeling_utils, models.embeddings/resnet/controlnet,
schedulers.scheduling_utils, StableDiffusionPipeline, ...) is import-stable.

Usage — make it the FIRST line of any fourm-touching entry point:

    import diffusers_compat  # noqa: F401  (must precede any fourm import)
    ...
    from fourm.vq.vqvae import VQVAE

No-op (and silent) if diffusers isn't installed, or is a version where the
symbols already live in the expected place. Safe to import many times.
"""

from __future__ import annotations

import sys

_applied = False


def apply() -> bool:
    """Install the shims. Returns True if diffusers was found, else False."""
    global _applied
    if _applied:
        return True

    try:
        import diffusers  # noqa: F401
    except Exception:
        # diffusers not installed in this interpreter (e.g. login node). The
        # caller's fourm import would fail anyway; nothing to shim.
        return False

    # 1) randn_tensor moved diffusers.utils -> diffusers.utils.torch_utils.
    #    Restore it as an attribute on diffusers.utils so fourm's
    #    `from diffusers.utils import randn_tensor` resolves.
    try:
        import diffusers.utils as _u
        if not hasattr(_u, "randn_tensor"):
            from diffusers.utils.torch_utils import randn_tensor as _rt
            _u.randn_tensor = _rt
    except Exception:
        pass

    # 2) unet_2d_blocks moved into the diffusers.models.unets subpackage.
    #    Alias the old dotted path in sys.modules so fourm's
    #    `from diffusers.models.unet_2d_blocks import ...` resolves.
    try:
        import diffusers.models.unets.unet_2d_blocks as _b
        sys.modules.setdefault("diffusers.models.unet_2d_blocks", _b)
    except Exception:
        pass

    _applied = True
    return True


# Apply on import so `import diffusers_compat` is all a caller needs.
apply()
