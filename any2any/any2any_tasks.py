"""
Reusable any2any demo tasks.

Entry points (argparse or hydra) should stay thin and public-release friendly and import from here.
"""

from __future__ import annotations

import os
import re
import json
import sys
from typing import Any, Optional, Union, cast, Dict, Callable, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# diffusers 0.31 compat for fourm VQ tokenizers — single source of truth in
# diffusers_compat.py (repo root). Must run before the fourm import below.
try:
    import diffusers_compat  # noqa: F401
except Exception:
    pass

try:
    from fourm.vq.vqvae import VQVAE
except ImportError:
    VQVAE = None

from data.data_utils import pil_img2rgb
from data.transforms import ImageTransform
from .inferencer import InterleaveInferencer, pca_apply_sklearn, pca_image_sklearn


# ── Shared helpers ────────────────────────────────────────────────────────────

def _set_seed(seed: int):
    """Set random seed for reproducibility (no-op if seed <= 0)."""
    if seed > 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def _load_image(path_or_image) -> Image.Image:
    """Load and normalise an image to RGB PIL.Image."""
    if isinstance(path_or_image, str):
        image = Image.open(path_or_image)
    else:
        image = path_or_image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return pil_img2rgb(image)


def _run_inference_loop(
    inferencer,
    inference_hyper: dict,
    *,
    image: Optional[Image.Image] = None,
    text: Optional[str] = None,
    understanding_output: bool = False,
    num_samples: int = 5,
    callback: Callable[[Dict[str, Any], int], None],
):
    """Core inference loop: call inferencer *num_samples* times and invoke *callback* for each result.

    Handles device patching of the hyper dict on every iteration (the inferencer
    may consume/move tensors during a call).
    """
    model_device = next(inferencer.model.parameters()).device
    for i in range(num_samples):
        patched_hyper = cast(Dict[str, Any], move_tensors_to_device(inference_hyper, model_device))
        kwargs: Dict[str, Any] = {}
        if image is not None:
            kwargs['image'] = image
        if text is not None:
            kwargs['text'] = text
        if understanding_output:
            kwargs['understanding_output'] = True
        result = inferencer(**kwargs, **patched_hyper)
        callback(result, i)


def _safe_name_from_prompt(p: str, max_len: int = 80) -> str:
    """Sanitize a prompt string into a filesystem-safe filename fragment."""
    import hashlib
    sanitized = re.sub(r"\s+", "_", p.strip())
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", sanitized)
    sanitized = re.sub(r"_{2,}", "_", sanitized).strip("_")
    if len(sanitized) > max_len:
        digest = hashlib.md5(p.encode("utf-8")).hexdigest()[:8]
        cut = max_len - 9
        if cut < 1:
            cut = max_len
        sanitized = f"{sanitized[:cut]}_{digest}"
    return sanitized or "prompt"


# ── Unified generation entry point ───────────────────────────────────────────

def generate_any(
    inferencer,
    output_path: str,
    *,
    prompt: str = "",
    input_image=None,
    condition: list = ("caption",),
    target: str = "rgb",
    intermediate: Optional[Union[str, List[str]]] = None,
    cfg_text_scale: float = 4.0,
    cfg_img_scale: float = 2.0,
    num_timesteps: int = 50,
    seed: int = 0,
    use_instruction: bool = True,
    use_target_instruction: bool = True,
    use_condition_instruction: bool = False,
    use_intermediate_instruction: bool = True,
    do_modality_norm: bool = False,
    use_det_image: bool = False,
    use_gt_dino_condition: bool = False,
    use_gt_dinolocal_condition: bool = False,
    use_gt_clip_condition: bool = False,
    use_gt_imagebind_condition: bool = False,
    use_gt_imagebindlocal_condition: bool = False,
    image_size: int = 1024,
    num_samples: int = 2,
    seg_category: Optional[str] = None,
    det_categories: Optional[List[str]] = None,
    cond_feature_mode: str = "both",
) -> List[Dict[str, Any]]:
    """Unified inference for image-generating pipelines.

    Handles: text→image, image→image (edit), image→seg, and chained variants
    (text→intermediate→image, image→intermediate→image).

    Returns the list of raw result dicts from the inferencer.
    """
    _set_seed(seed)
    condition = list(condition)
    image = _load_image(input_image) if input_image is not None else None
    has_image = image is not None
    if isinstance(intermediate, list):
        _filtered = [m for m in intermediate if isinstance(m, str) and m.lower() not in ('', 'none')]
        is_chained = len(_filtered) > 0
        intermediate = _filtered if is_chained else None
    else:
        is_chained = intermediate is not None and str(intermediate).lower() not in ('', 'none')

    # Auto-determine cfg_interval / cfg_renorm_type from whether we have an image condition
    cfg_interval = [0.0, 1.0] if has_image else [0.4, 1.0]
    cfg_renorm_type = "text_channel" if has_image else "global"

    modality_type_dict: Dict[str, Any] = {"condition": condition, "target": target}
    if is_chained:
        modality_type_dict["intermediate"] = intermediate

    # Env override for renorm probing (codex review suggested disabling renorm
    # via cfg_renorm_min=1.0 to test if it's suppressing high-freq).
    import os as _os
    cfg_interval_env = _os.environ.get("CFG_INTERVAL")
    if cfg_interval_env:
        cfg_interval = [float(x) for x in cfg_interval_env.split(",", 1)]
    cfg_renorm_min = float(_os.environ.get("CFG_RENORM_MIN", "0.0"))
    cfg_renorm_type = _os.environ.get("CFG_RENORM_TYPE", cfg_renorm_type)
    hyper: Dict[str, Any] = dict(
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        cfg_interval=cfg_interval,
        timestep_shift=3.0,
        num_timesteps=num_timesteps,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
        image_shapes=(image_size, image_size),
        modality_type_dict=modality_type_dict,
        use_instruction=use_instruction,
        use_target_instruction=use_target_instruction,
        use_condition_instruction=use_condition_instruction,
        do_modality_norm=do_modality_norm,
        use_det_image=use_det_image,
        use_gt_dino_condition=use_gt_dino_condition,
        use_gt_dinolocal_condition=use_gt_dinolocal_condition,
        use_gt_clip_condition=use_gt_clip_condition,
        use_gt_imagebind_condition=use_gt_imagebind_condition,
        use_gt_imagebindlocal_condition=use_gt_imagebindlocal_condition,
        cond_feature_mode=cond_feature_mode,
    )
    if is_chained:
        hyper['chained_inference'] = True
        hyper['use_intermediate_instruction'] = use_intermediate_instruction
    # seg_category / det_categories must reach the inferencer for DIRECT generation
    # too (e.g. RGB->seg with a UI category), not only chained inference.
    if seg_category is not None:
        hyper['seg_category'] = seg_category
    if det_categories is not None:
        hyper['det_categories'] = det_categories

    results: List[Dict[str, Any]] = []
    text_arg = prompt if prompt else None

    def on_result(result, i):
        results.append(result)

    print("Running inference...")
    _run_inference_loop(
        inferencer, hyper,
        image=image, text=text_arg,
        num_samples=num_samples, callback=on_result,
    )
    return results


# ── Visualisation / post-processing ──────────────────────────────────────────

def compute_cosine_similarity_heatmap(feat1, feat2, save_path: str):
    """
    Compute cosine similarity between two feature maps at each spatial position and save as a heatmap.

    Args:
        feat1: torch.Tensor of shape [B, C, H, W] - ground truth features
        feat2: torch.Tensor of shape [B, C, H, W] - generated features
        save_path: path to save the cosine similarity heatmap
    """
    feat1 = feat1.cpu()
    feat2 = feat2.cpu()

    b, c, h, w = feat1.shape
    feat1_flat = feat1.view(b, c, -1)
    feat2_flat = feat2.view(b, c, -1)

    feat1_norm = torch.nn.functional.normalize(feat1_flat, p=2, dim=1)
    feat2_norm = torch.nn.functional.normalize(feat2_flat, p=2, dim=1)
    cos_sim = torch.sum(feat1_norm * feat2_norm, dim=1)
    cos_sim_spatial = cos_sim.view(b, h, w)

    cos_sim_vis = cos_sim_spatial[0].numpy()

    plt.figure(figsize=(10, 8))
    plt.imshow(cos_sim_vis, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Cosine Similarity")
    plt.title("Cosine Similarity Heatmap")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"Cosine similarity heatmap saved to {save_path}")
    print(f"Mean cosine similarity: {cos_sim_vis.mean():.4f}")
    print(f"Std cosine similarity: {cos_sim_vis.std():.4f}")


# ── Inferencer factory ────────────────────────────────────────────────────────

def create_inferencer(
    model,
    vae_model,
    tokenizer,
    new_token_ids,
    dino_tokenizer=None,       # BC: old callers pass tokenizer object positionally or as keyword.
    dinolocal_tokenizer=None,  # BC: old callers pass tokenizer object positionally or as keyword.
    *,
    modality_registry=None,
    use_dino_tokenizer: bool = False,
    use_dinolocal_tokenizer: bool = False,
    use_clip_tokenizer: bool = False,
    use_imagebind_tokenizer: bool = False,
    use_imagebindlocal_tokenizer: bool = False,
):
    """Create the inferencer from loaded components.

    Backward compat: old callers may pass ``dino_tokenizer`` / ``dinolocal_tokenizer``
    as positional or keyword args (5th and 6th).  When provided, they are used directly
    and the config-driven loading path is skipped for those two.
    """
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type == "janus_flow":
        vae_transform = ImageTransform(512, 16, 16, max_pixels=512 * 512)
        vit_transform = ImageTransform(384, 224, 16, max_pixels=16 * 16 * 24 * 24)
    elif hasattr(model, "hunyuan_model"):
        vae_max_size = int(os.environ.get("HUNYUAN_INFER_VAE_MAX_SIZE", "1024"))
        vae_min_size = int(os.environ.get("HUNYUAN_INFER_VAE_MIN_SIZE", "512"))
        vae_transform = ImageTransform(vae_max_size, vae_min_size, 16)
        # Hunyuan training clamps ViT input to vit_max_num_patch_per_side=16 (256x256 max);
        # mirror that here so inference sees the same input-size distribution as training.
        vit_transform = ImageTransform(256, 224, 16, max_pixels=16 * 16 * 16 * 16)
    else:
        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)

    # Config-driven external tokenizers (e.g. DINO/DINOLOCAL/CLIP/ImageBind VQVAE quantizers).
    # Training data may be pre-tokenized; these are mainly for inference-time encode/decode.
    model_device = next(model.parameters()).device

    clip_tokenizer = None
    imagebind_tokenizer = None
    imagebindlocal_tokenizer = None

    # If caller passed tokenizer objects directly (old API), use them as-is for dino/dinolocal.
    if dino_tokenizer is not None or dinolocal_tokenizer is not None:
        pass  # Skip config-driven loading for dino/dinolocal.
    elif modality_registry is not None:
        if VQVAE is None and (use_dino_tokenizer or use_dinolocal_tokenizer):
            raise ImportError("fourm.vq.vqvae is required for DINO external tokenizers")
        for name, want_flag in (("dino", use_dino_tokenizer), ("dinolocal", use_dinolocal_tokenizer)):
            if not want_flag:
                continue
            try:
                spec = modality_registry.get(name)
            except Exception:
                continue
            repo = getattr(spec, "external_tokenizer_repo", None)
            kind = getattr(spec, "external_tokenizer_kind", "vqvae")
            if repo is None:
                continue
            if kind != "vqvae":
                raise ValueError(f"Unsupported external_tokenizer_kind='{kind}' for modality '{name}'")
            tok = VQVAE.from_pretrained(str(repo)).eval().to(model_device)
            if name == "dino":
                dino_tokenizer = tok
            elif name == "dinolocal":
                dinolocal_tokenizer = tok
    else:
        # Backward compat: explicit flags still load fixed repos if no registry is provided.
        if use_dino_tokenizer:
            if VQVAE is None:
                raise ImportError("fourm.vq.vqvae is required for DINO external tokenizers")
            dino_tokenizer = VQVAE.from_pretrained("EPFL-VILAB/4M_tokenizers_DINOv2-B14-global_8k_16_224").eval().to(model_device)
        if use_dinolocal_tokenizer:
            if VQVAE is None:
                raise ImportError("fourm.vq.vqvae is required for DINOLOCAL external tokenizers")
            dinolocal_tokenizer = VQVAE.from_pretrained("EPFL-VILAB/4M_tokenizers_DINOv2-B14_8k_224-448").eval().to(model_device)

    # Config-driven loading for clip / imagebind / imagebindlocal external tokenizers.
    if modality_registry is not None:
        if VQVAE is None and (use_clip_tokenizer or use_imagebind_tokenizer or use_imagebindlocal_tokenizer):
            raise ImportError("fourm.vq.vqvae is required for external VQ tokenizers")
        for name, want_flag in (
            ("clip", use_clip_tokenizer),
            ("imagebind", use_imagebind_tokenizer),
            ("imagebindlocal", use_imagebindlocal_tokenizer),
        ):
            if not want_flag:
                continue
            try:
                spec = modality_registry.get(name)
            except Exception:
                continue
            repo = getattr(spec, "external_tokenizer_repo", None)
            kind = getattr(spec, "external_tokenizer_kind", "vqvae")
            if repo is None:
                continue
            if kind != "vqvae":
                raise ValueError(f"Unsupported external_tokenizer_kind='{kind}' for modality '{name}'")
            tok = VQVAE.from_pretrained(str(repo)).eval().to(model_device)
            if name == "clip":
                clip_tokenizer = tok
            elif name == "imagebind":
                imagebind_tokenizer = tok
            elif name == "imagebindlocal":
                imagebindlocal_tokenizer = tok

    return InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
        dino_tokenizer=dino_tokenizer,
        dinolocal_tokenizer=dinolocal_tokenizer,
        clip_tokenizer=clip_tokenizer,
        imagebind_tokenizer=imagebind_tokenizer,
        imagebindlocal_tokenizer=imagebindlocal_tokenizer,
        modality_registry=modality_registry,
    )


def move_tensors_to_device(obj: Any, device):
    """Recursively move all tensors in a dict/list to the specified device."""
    if isinstance(obj, dict):
        return {k: move_tensors_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_tensors_to_device(v, device) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    return obj


# ══════════════════════════════════════════════════════════════════════════════
# Unified task entry point
#
# A single function that handles ALL inference modes, dispatching to the
# correct pre/post-processing for each target modality.  The old per-task
# functions below are kept as backward-compatible thin wrappers.
# ══════════════════════════════════════════════════════════════════════════════

# Registry of per-modality post-processing callbacks.
# Each callback receives (result_dict, sample_index, context_dict) and returns
# any save-worthy artifact.  ``context_dict`` carries shared state
# (output_path, condition, target, prompt, image, …).

_POST_PROCESSORS: Dict[str, Callable] = {}


def _register_postprocessor(name: str):
    """Decorator to register a post-processor for a decode method."""
    def decorator(fn):
        _POST_PROCESSORS[name] = fn
        return fn
    return decorator


def _draw_prompt_bbox(image: Image.Image, prompt: str, label: str = "") -> Image.Image:
    """Parse bbox tokens from a det prompt and draw the box on *image*.

    Expects coordinate tokens like ``<|x1_100|><|y1_100|><|x2_400|><|y2_450|>``
    where values are in the [0, 999] quantized range.
    Returns a new image with the bbox drawn (original is not modified).
    """
    coord_pattern = re.compile(r"<\|([xy][12])_(\d+)\|>")
    matches = coord_pattern.findall(prompt)
    if len(matches) < 4:
        return image

    coords = {k: int(v) for k, v in matches[:4]}
    if not all(k in coords for k in ("x1", "y1", "x2", "y2")):
        return image

    w, h = image.size
    def to_px(v, max_px):
        return int(round(v * (max_px - 1) / 999.0))

    px1, py1 = to_px(coords["x1"], w), to_px(coords["y1"], h)
    px2, py2 = to_px(coords["x2"], w), to_px(coords["y2"], h)

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([px1, py1, px2, py2], outline="red", width=3)

    if label:
        font = ImageFont.load_default()
        text_y = max(2, py1 - 14)
        tb = draw.textbbox((px1, text_y), label, font=font)
        draw.rectangle([tb[0] - 1, tb[1] - 1, tb[2] + 3, tb[3] + 1], fill="red")
        draw.text((px1, text_y), label, fill="white", font=font)

    return overlay


@_register_postprocessor("image")
def _postprocess_image(result, i, ctx):
    """Save a generated image.  If the condition is 'det', also save a copy
    with the input bbox drawn on top."""
    raw = result["image"]
    # Inferencer order for chained: [intermediate_1, intermediate_2, ..., final].
    if isinstance(raw, (list, tuple)):
        intermediate_images = list(raw[:-1])
        out_image = raw[-1]
    else:
        intermediate_images = []
        out_image = raw

    condition = ctx["condition"]
    target = ctx["target"]
    intermediate = ctx.get("intermediate")
    num_timesteps = ctx.get("num_timesteps", "")

    # Normalize the intermediate-modality list.  Filter codebook intermediates
    # (dino/det) — they decode to token strings, not PIL images, and the
    # inferencer's __call__ only forwards Image.Image instances into
    # result["image"].  So intermediate_images is aligned with the *image-like*
    # subset of intermediate names.
    if isinstance(intermediate, list):
        intermediate_names_all = list(intermediate)
    elif isinstance(intermediate, str) and intermediate.lower() not in ("", "none"):
        intermediate_names_all = [intermediate]
    else:
        intermediate_names_all = []
    _NON_IMAGE_DM = {"dino", "dinolocal", "det", "detection", "clip",
                     "imagebind", "imagebindlocal", "text"}
    inferencer = ctx.get("_inferencer")  # optional — set by run_inference_task
    image_intermediate_names = []
    for name in intermediate_names_all:
        dm = None
        if inferencer is not None:
            try:
                dm = inferencer._resolve_decode_method(name)
            except Exception:
                dm = None
        if dm in _NON_IMAGE_DM:
            continue
        image_intermediate_names.append(name)

    # Build save name with chain info when present.
    name_parts = [condition[0]]
    if intermediate_names_all:
        name_parts.append("+".join(intermediate_names_all))
    name_parts.append(target)

    suffix = f"timestep_{num_timesteps}_" if ctx.get("has_image") and not intermediate_names_all else ""
    save_name = "2".join(name_parts) + f"_{suffix}{i}.png"
    save_path = os.path.join(ctx["output_path"], save_name)

    out_image.save(save_path)
    print(f"Image saved to {save_path}")

    # Save text intermediates (e.g. caption) when requested.  result["text"]
    # is a string when there's exactly one text intermediate, a list when
    # there are multiple, or None when there are none.
    if ctx.get("save_intermediate"):
        text_inter = result.get("text")
        if text_inter:
            text_list = text_inter if isinstance(text_inter, list) else [text_inter]
            text_inter_names = [
                n for n in intermediate_names_all if n not in image_intermediate_names
            ]
            for k, txt in enumerate(text_list):
                tname = text_inter_names[k] if k < len(text_inter_names) else f"text{k}"
                txt_path = os.path.join(
                    ctx["output_path"],
                    save_name.replace(".png", f"_intermediate_{tname}.txt"),
                )
                try:
                    with open(txt_path, "w") as f:
                        f.write(txt)
                    print(f"Intermediate ({tname}) saved to {txt_path}")
                except Exception as e:
                    print(f"[warn] failed to save intermediate {tname}: {e}")

    # Save every PIL Pass-1 intermediate (depth/normal/seg/...) when requested.
    # Codebook intermediates (dino/det) don't appear in intermediate_images.
    if ctx.get("save_intermediate") and intermediate_images:
        names_for_pils = (
            image_intermediate_names
            if len(image_intermediate_names) == len(intermediate_images)
            else [f"inter{k}" for k in range(len(intermediate_images))]
        )
        for img, name in zip(intermediate_images, names_for_pils):
            inter_name = save_name.replace(".png", f"_intermediate_{name}.png")
            inter_path = os.path.join(ctx["output_path"], inter_name)
            try:
                img.save(inter_path)
                print(f"Intermediate ({name}) saved to {inter_path}")
            except Exception as e:
                print(f"[warn] failed to save intermediate {name}: {e}")

    # Save input image alongside when present (so chained_edit / image-conditioned
    # runs can be inspected end-to-end).
    if ctx.get("save_intermediate") and ctx.get("image") is not None:
        in_name = save_name.replace(".png", "_input.png")
        in_path = os.path.join(ctx["output_path"], in_name)
        try:
            ctx["image"].save(in_path)
            print(f"Input image saved to {in_path}")
        except Exception as e:
            print(f"[warn] failed to save input image: {e}")

    # For dino/dinolocal -> rgb tests, save side-by-side comparison with input RGB.
    if condition and condition[0] in ("dino", "dinolocal") and target == "rgb":
        input_image = ctx.get("image")
        if input_image is not None:
            ref = pil_img2rgb(input_image)
            gen = pil_img2rgb(out_image)
            panels = [ref]

            # For dinolocal conditioning, include GT DINO-local PCA in the middle panel.
            if condition[0] == "dinolocal":
                cond_pca_image = ctx.get("cond_dinolocal_pca_image")
                if cond_pca_image is not None:
                    panels.append(pil_img2rgb(cond_pca_image))

            panels.append(gen)

            # Force all panels to input RGB resolution, then concatenate horizontally.
            target_w, target_h = ref.width, ref.height
            resized = [
                img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                for img in panels
            ]

            gap = 0
            total_w = sum(img.width for img in resized) + gap * (len(resized) - 1)
            canvas = Image.new("RGB", (total_w, target_h), (255, 255, 255))
            x = 0
            for img in resized:
                canvas.paste(img, (x, 0))
                x += img.width + gap
            compare_path = save_path.replace(".png", "_compare.png")
            canvas.save(compare_path)
            print(f"{condition[0]}2rgb comparison saved to {compare_path}")

    # If condition is det, also save a version with the input bbox overlaid.
    if "det" in condition:
        prompt = ctx.get("prompt", "")
        # Extract the phrase (before <sep>) for the label.
        phrase = prompt.split("<sep>")[0].strip() if "<sep>" in prompt else ""
        bbox_image = _draw_prompt_bbox(out_image, prompt, label=phrase)
        bbox_save_name = save_name.replace(".png", "_bbox.png")
        bbox_save_path = os.path.join(ctx["output_path"], bbox_save_name)
        bbox_image.save(bbox_save_path)
        print(f"  + bbox overlay saved to {bbox_save_path}")


@_register_postprocessor("detection")
def _postprocess_detection(result, i, ctx):
    """Parse detection coordinates, draw bounding box, save."""
    result_text = result["text"]
    condition = ctx["condition"]
    target = ctx["target"]
    image = ctx["image"]
    safe_prompt = _safe_name_from_prompt(ctx.get("prompt", ""), max_len=80)
    save_path = os.path.join(ctx["output_path"], f"{condition[0]}2{target}_{safe_prompt}_{i}.png")

    result_image = image.copy()
    coords = re.findall(r"<\|([xy]\d)_(\d+)\|>", result_text)
    coord_dict = {k: int(v) for k, v in coords}
    x1, y1, x2, y2 = coord_dict["x1"], coord_dict["y1"], coord_dict["x2"], coord_dict["y2"]

    def to_px(v, max_px):
        return int(round(v * (max_px - 1) / 999.0))

    w, h = image.size
    px1, py1, px2, py2 = to_px(x1, w), to_px(y1, h), to_px(x2, w), to_px(y2, h)

    draw = ImageDraw.Draw(result_image)
    draw.rectangle([px1, py1, px2, py2], outline="red", width=5)
    result_image.save(save_path)
    print(f"Detection image saved to {save_path}")


@_register_postprocessor("dino")
def _postprocess_dino(result, i, ctx):
    """Save DINO PCA visualization and cosine similarity."""
    target = ctx["target"]
    condition = ctx["condition"]

    dino_feat = result["dino_feat"]
    gt_global_feat = ctx.get("gt_global_feat")

    if gt_global_feat is not None:
        # 2026-05-14: ensure both tensors on same device (was: dino_feat on cuda,
        # gt_global_feat on cpu → RuntimeError in cosine_similarity)
        _gt = gt_global_feat.to(dino_feat.device).to(dino_feat.dtype)
        cos_sim = F.cosine_similarity(dino_feat, _gt, dim=-1)
        cos_sim_value = float(cos_sim.mean().item())
        print(f"Global DINO cosine similarity: {cos_sim_value:.6f}")
        # Use the same key the dl evals use, so external watchers/parsers pick it up.
        print(f"Mean cosine similarity: {cos_sim_value:.4f}")
        metric_path = os.path.join(ctx["output_path"], f"{condition[0]}2{target}_metrics.jsonl")
        with open(metric_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sample_index": i, "cosine_similarity": cos_sim_value}) + "\n")
        print(f"DINO metrics appended to {metric_path}")

    result_image = result.get("image")
    if result_image is None and result.get("dino_feat") is not None:
        # Global DINO has no spatial map → render a radial "fingerprint" of the
        # generated feature vector so the demo shows something (not blank).
        result_image = _global_feat_fingerprint(result["dino_feat"])
    if result_image is not None:
        save_path = os.path.join(ctx["output_path"], f"{condition[0]}2{target}_{i}.png")
        result_image.save(save_path)
        print(f"DINO image saved to {save_path}")


@_register_postprocessor("dinolocal")
def _postprocess_dinolocal(result, i, ctx):
    """Save DINO-local PCA visualization and cosine similarity heatmap."""
    target = ctx["target"]
    condition = ctx["condition"]

    dino_feat = result["dino_feat"]
    gt_feat_map = ctx.get("gt_feat_map")
    pca = ctx.get("pca")

    if gt_feat_map is not None:
        cos_sim_save_path = os.path.join(ctx["output_path"], f"{condition[0]}2{target}_cos_sim_{i}.png")
        compute_cosine_similarity_heatmap(gt_feat_map, dino_feat, cos_sim_save_path)

    if pca is not None:
        result_image = pca_apply_sklearn(dino_feat, pca)
        save_path = os.path.join(ctx["output_path"], f"{condition[0]}2{target}_{i}.png")
        result_image.save(save_path)
        print(f"DINO-local image saved to {save_path}")
    else:
        try:
            feat_pca, _ = pca_image_sklearn(dino_feat)
            save_path = os.path.join(ctx["output_path"], f"{condition[0]}2{target}_pca_{i}.png")
            feat_pca.save(save_path)
            print(f"DINO-local PCA image saved to {save_path}")
        except Exception as e:
            print(f"DINO-local postprocess: could not generate PCA visualization: {e}")
        tensor_path = os.path.join(ctx["output_path"], f"{condition[0]}2{target}_{i}.pt")
        torch.save(dino_feat, tensor_path)
        print(f"DINO-local raw features saved to {tensor_path}")


@_register_postprocessor("clip")
def _postprocess_clip(result, i, ctx):
    """Save CLIP PCA visualization and cosine similarity heatmap."""
    target = ctx["target"]
    condition = ctx["condition"]

    clip_feat = result["dino_feat"]  # codebook output reuses 'dino_feat' key
    gt_feat_map = ctx.get("gt_feat_map")
    pca = ctx.get("pca")

    if gt_feat_map is not None:
        cos_sim_save_path = os.path.join(ctx["output_path"], f"{condition[0]}2{target}_cos_sim_{i}.png")
        compute_cosine_similarity_heatmap(gt_feat_map, clip_feat, cos_sim_save_path)

    if pca is not None:
        result_image = pca_apply_sklearn(clip_feat, pca)
        save_path = os.path.join(ctx["output_path"], f"{condition[0]}2{target}_{i}.png")
        result_image.save(save_path)
        print(f"CLIP image saved to {save_path}")
    else:
        # Fallback: try PCA from scratch
        try:
            feat_pca, _ = pca_image_sklearn(clip_feat)
            save_path = os.path.join(ctx["output_path"], f"{condition[0]}2{target}_pca_{i}.png")
            feat_pca.save(save_path)
            print(f"CLIP PCA image saved to {save_path}")
        except Exception as e:
            print(f"CLIP postprocess: could not generate PCA visualization: {e}")
            save_path = os.path.join(ctx["output_path"], f"{condition[0]}2{target}_{i}.pt")
            torch.save(clip_feat, save_path)
            print(f"CLIP raw features saved to {save_path}")


@_register_postprocessor("imagebind")
def _postprocess_imagebind(result, i, ctx):
    """Save ImageBind (global) codebook output."""
    target = ctx["target"]
    condition = ctx["condition"]

    feat = result["dino_feat"]  # codebook output reuses 'dino_feat' key
    gt_global_feat = ctx.get("gt_global_feat")

    if gt_global_feat is not None:
        _gt = gt_global_feat.to(feat.device).to(feat.dtype)
        cos_sim = F.cosine_similarity(feat, _gt, dim=-1)
        cos_sim_value = float(cos_sim.mean().item())
        print(f"Global ImageBind cosine similarity: {cos_sim_value:.6f}")
        # Same key the dl evals use, so watchers/parsers pick it up.
        print(f"Mean cosine similarity: {cos_sim_value:.4f}")
        metric_path = os.path.join(ctx["output_path"], f"{condition[0]}2{target}_metrics.jsonl")
        with open(metric_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sample_index": i, "cosine_similarity": cos_sim_value}) + "\n")
        print(f"ImageBind metrics appended to {metric_path}")

    result_image = result.get("image")
    if result_image is None and feat is not None:
        # Global ImageBind has no spatial map → radial "fingerprint" viz.
        result_image = _global_feat_fingerprint(feat)
    if result_image is not None:
        save_path = os.path.join(ctx["output_path"], f"{condition[0]}2{target}_{i}.png")
        result_image.save(save_path)
        print(f"ImageBind image saved to {save_path}")


@_register_postprocessor("imagebindlocal")
def _postprocess_imagebindlocal(result, i, ctx):
    """Save ImageBind-local PCA visualization and cosine similarity heatmap."""
    target = ctx["target"]
    condition = ctx["condition"]

    ib_feat = result["dino_feat"]  # codebook output reuses 'dino_feat' key
    gt_feat_map = ctx.get("gt_feat_map")
    pca = ctx.get("pca")

    if gt_feat_map is not None:
        cos_sim_save_path = os.path.join(ctx["output_path"], f"{condition[0]}2{target}_cos_sim_{i}.png")
        compute_cosine_similarity_heatmap(gt_feat_map, ib_feat, cos_sim_save_path)

    if pca is not None:
        result_image = pca_apply_sklearn(ib_feat, pca)
        save_path = os.path.join(ctx["output_path"], f"{condition[0]}2{target}_{i}.png")
        result_image.save(save_path)
        print(f"ImageBind-local image saved to {save_path}")
    else:
        # Fallback: try PCA from scratch
        try:
            feat_pca, _ = pca_image_sklearn(ib_feat)
            save_path = os.path.join(ctx["output_path"], f"{condition[0]}2{target}_pca_{i}.png")
            feat_pca.save(save_path)
            print(f"ImageBind-local PCA image saved to {save_path}")
        except Exception as e:
            print(f"ImageBind-local postprocess: could not generate PCA visualization: {e}")
            save_path = os.path.join(ctx["output_path"], f"{condition[0]}2{target}_{i}.pt")
            torch.save(ib_feat, save_path)
            print(f"ImageBind-local raw features saved to {save_path}")


# Colormap for the global-feature "fingerprint" viz (matches _hero_16mod).
_FP_CX = np.array([0, .25, .5, .75, 1.])
_FP_CR = np.array([48, 33, 94, 253, 165])
_FP_CG = np.array([18, 144, 201, 231, 0])
_FP_CB = np.array([59, 141, 98, 37, 38])


def _global_feat_fingerprint(feat, cell: int = 320) -> Image.Image:
    """Render a global feature vector as a radial 'fingerprint' PIL image."""
    a = feat.detach().float().cpu().reshape(-1).numpy()
    r = 6
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / 2.0) ** 2)
    k /= k.sum()
    a = np.convolve(np.pad(a, r, mode="reflect"), k, mode="valid")
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    v = np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)
    n = v.size
    yy, xx = np.mgrid[0:cell, 0:cell]
    c = cell / 2.0
    ang = (np.arctan2(yy - c, xx - c) + np.pi) / (2 * np.pi)
    rad = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / c
    idx = np.clip((ang * (n - 1)).astype(int), 0, n - 1)
    vv = v[idx]
    rgb = np.stack(
        [np.interp(vv, _FP_CX, _FP_CR), np.interp(vv, _FP_CX, _FP_CG), np.interp(vv, _FP_CX, _FP_CB)],
        -1,
    ).astype(np.uint8)
    rgb[rad > 1.0] = 255
    return Image.fromarray(rgb, "RGB")


# COCO-80 class names (contiguous 0-indexed; the cocodet head emits this index,
# e.g. 0 -> person, 46 -> banana, 74 -> clock).
_COCO80 = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


@_register_postprocessor("cocodet")
def _postprocess_cocodet(result, i, ctx):
    """Draw the generated COCO detection boxes (with category name) on the input."""
    condition = ctx["condition"]
    target = ctx["target"]
    image = ctx.get("image")
    boxes = result.get("cocodet_boxes") or []
    # The cocodet head has a failure mode where it spams many low-confidence boxes
    # of one class (e.g. 199 "chair"). Keep only reasonably-confident boxes, drop
    # degenerate full-frame ones, and cap at the top-K by score so the overlay
    # stays legible instead of a red grid.
    _MIN_SCORE = float(os.environ.get("MODUS_COCODET_MIN_SCORE", "0.05"))
    _MAX_BOXES = int(os.environ.get("MODUS_COCODET_MAX_BOXES", "12"))
    boxes = [b for b in boxes if b.get("bbox")
             and not ((b["bbox"][2] - b["bbox"][0]) >= 0.97 and (b["bbox"][3] - b["bbox"][1]) >= 0.97)
             and (b.get("score") is None or b.get("score", 0) >= _MIN_SCORE)]
    boxes.sort(key=lambda b: b.get("score", 0), reverse=True)
    boxes = boxes[:_MAX_BOXES]
    canvas = (image.convert("RGB").copy() if image is not None
              else Image.new("RGB", (512, 512), (40, 40, 40)))
    W, H = canvas.size
    draw = ImageDraw.Draw(canvas)
    n_drawn = 0
    for b in boxes:
        bb = b.get("bbox")
        if not bb:
            continue
        x1, y1, x2, y2 = (bb[0] * W, bb[1] * H, bb[2] * W, bb[3] * H)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        lab = b.get("label")
        name = _COCO80[lab] if isinstance(lab, int) and 0 <= lab < len(_COCO80) else str(lab)
        score = b.get("score")
        tag = f"{name} {score:.2f}" if isinstance(score, (int, float)) else name
        draw.text((x1 + 2, y1 + 2), tag, fill="yellow")
        n_drawn += 1
    save_path = os.path.join(ctx["output_path"], f"{condition[0]}2{target}_{i}.png")
    canvas.save(save_path)
    print(f"cocodet image saved to {save_path} ({n_drawn}/{len(boxes)} boxes drawn)")


def run_inference_task(
    *,
    inferencer,
    output_path: str,
    mode: str = "generate",
    prompt: str = "",
    input_image=None,
    condition=("caption",),
    target: str = "rgb",
    intermediate: Optional[Union[str, List[str]]] = None,
    cfg_text_scale: float = 4.0,
    cfg_img_scale: float = 2.0,
    num_timesteps: int = 50,
    seed: int = 0,
    use_instruction: bool = True,
    use_target_instruction: bool = True,
    use_condition_instruction: bool = False,
    use_intermediate_instruction: bool = True,
    do_modality_norm: bool = False,
    use_det_image: bool = False,
    image_size: int = 1024,
    do_sample: bool = False,
    temperature: float = 0.003,
    text_temperature: float = 0.95,
    top_k: int = 0,
    top_p: float = 1.0,
    num_samples: int = 2,
    use_gt_dino_condition: bool = False,
    use_gt_dinolocal_condition: bool = False,
    use_gt_clip_condition: bool = False,
    use_gt_imagebind_condition: bool = False,
    use_gt_imagebindlocal_condition: bool = False,
    save_intermediate: bool = False,
    seg_category: Optional[str] = None,
    det_categories: Optional[List[str]] = None,
):
    """Single entry-point for all inference tasks.

    Pre-processing (GT feature computation for DINO) and post-processing
    (bbox drawing, PCA) are dispatched via the ``_POST_PROCESSORS`` registry.
    """
    condition = list(condition)
    image = _load_image(input_image) if input_image is not None else None

    # ── Determine decode method for post-processing dispatch ──────────────
    decode_method = _infer_decode_method(target, inferencer)

    # ── Pre-processing (per decode method) ────────────────────────────────
    ctx: Dict[str, Any] = dict(
        output_path=output_path, condition=condition, target=target,
        intermediate=intermediate, prompt=prompt, image=image,
        num_timesteps=num_timesteps, has_image=image is not None,
        save_intermediate=save_intermediate,
        _inferencer=inferencer,  # used by postprocessor to resolve modality decode methods
    )
    if decode_method == "dino":
        ctx.update(_preprocess_dino(image))
    elif decode_method == "dinolocal":
        ctx.update(_preprocess_dinolocal(image, output_path, condition, target))
    elif decode_method == "clip":
        ctx.update(_preprocess_clip(image, output_path, condition, target))
    elif decode_method == "imagebindlocal":
        ctx.update(_preprocess_imagebindlocal(image, output_path, condition, target))
    elif decode_method == "imagebind":
        ctx.update(_preprocess_imagebind(image))

    # For dinolocal -> rgb visualization, prepare GT dinolocal PCA panel.
    if image is not None and condition and condition[0] == "dinolocal" and target == "rgb":
        cond_ctx = _preprocess_dinolocal(image, output_path, condition, target)
        cond_gt_feat_map = cond_ctx.get("gt_feat_map")
        cond_pca = cond_ctx.get("pca")
        if cond_gt_feat_map is not None and cond_pca is not None:
            # Upsample local feature map to RGB resolution for clearer spatial alignment.
            upsampled_feat_map = F.interpolate(
                cond_gt_feat_map.float(),
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            )
            ctx["cond_dinolocal_pca_image"] = pca_apply_sklearn(upsampled_feat_map, cond_pca)

    # ── Run inference ─────────────────────────────────────────────────────
    if decode_method in ("image",):
        # Image-producing pipelines (generate, edit, segment, chained).
        results = generate_any(
            inferencer, output_path,
            prompt=prompt, input_image=input_image,
            condition=condition, target=target, intermediate=intermediate,
            cfg_text_scale=cfg_text_scale, cfg_img_scale=cfg_img_scale,
            num_timesteps=num_timesteps, seed=seed,
            use_instruction=use_instruction,
            use_target_instruction=use_target_instruction,
            use_condition_instruction=use_condition_instruction,
            use_intermediate_instruction=use_intermediate_instruction,
            do_modality_norm=do_modality_norm, use_det_image=use_det_image,
            use_gt_dino_condition=use_gt_dino_condition,
            use_gt_dinolocal_condition=use_gt_dinolocal_condition,
            use_gt_clip_condition=use_gt_clip_condition,
            use_gt_imagebind_condition=use_gt_imagebind_condition,
            use_gt_imagebindlocal_condition=use_gt_imagebindlocal_condition,
            image_size=image_size, num_samples=num_samples,
            seg_category=seg_category,
            det_categories=det_categories,
        )
        postprocessor = _POST_PROCESSORS.get(decode_method, _POST_PROCESSORS["image"])
        for i, result in enumerate(results):
            postprocessor(result, i, ctx)
    else:
        # Understanding pipelines (detection, dino, dinolocal, text).
        _set_seed(seed)

        inference_hyper = _build_understanding_hyper(
            decode_method=decode_method,
            condition=condition, target=target,
            cfg_text_scale=cfg_text_scale, cfg_img_scale=cfg_img_scale,
            use_instruction=use_instruction,
            use_target_instruction=use_target_instruction,
            use_condition_instruction=use_condition_instruction,
            do_modality_norm=do_modality_norm,
            do_sample=do_sample, temperature=temperature,
            text_temperature=text_temperature,
            top_k=top_k, top_p=top_p,
            use_gt_dino_condition=use_gt_dino_condition,
            use_gt_dinolocal_condition=use_gt_dinolocal_condition,
            use_gt_clip_condition=use_gt_clip_condition,
            use_gt_imagebind_condition=use_gt_imagebind_condition,
            use_gt_imagebindlocal_condition=use_gt_imagebindlocal_condition,
        )
        print(f"Running {decode_method} inference...")

        postprocessor = _POST_PROCESSORS.get(decode_method, lambda r, i, c: None)

        def on_result(result, i):
            postprocessor(result, i, ctx)

        _run_inference_loop(
            inferencer, inference_hyper,
            image=image, text=prompt,
            understanding_output=True,
            num_samples=num_samples,
            callback=on_result,
        )


# ── Internal helpers for run_inference_task ───────────────────────────────────

def _infer_decode_method(target: str, inferencer) -> str:
    """Resolve decode method from target name and inferencer registry."""
    if inferencer.modality_registry is not None:
        try:
            return inferencer.modality_registry.resolve_decode_method(target)
        except (KeyError, AttributeError):
            pass
    if "det" in target:
        return "detection"
    if target == "dinolocal":
        return "dinolocal"
    if "dino" in target:
        return "dino"
    return "image"


def _build_understanding_hyper(
    *,
    decode_method: str,
    condition, target,
    cfg_text_scale, cfg_img_scale,
    use_instruction, use_target_instruction, use_condition_instruction,
    do_modality_norm,
    do_sample, temperature,
    text_temperature: float = 0.95,
    top_k: int = 0,
    top_p: float = 1.0,
    use_gt_dino_condition: bool = False,
    use_gt_dinolocal_condition: bool = False,
    use_gt_clip_condition: bool = False,
    use_gt_imagebind_condition: bool = False,
    use_gt_imagebindlocal_condition: bool = False,
) -> dict:
    """Build inference_hyper dict for understanding tasks."""
    hyper: Dict[str, Any] = dict(
        modality_type_dict={"condition": condition + ["text"], "target": [target]},
        use_instruction=use_instruction,
        do_modality_norm=do_modality_norm,
        use_target_instruction=use_target_instruction,
        use_condition_instruction=use_condition_instruction,
        use_gt_dino_condition=use_gt_dino_condition,
        use_gt_dinolocal_condition=use_gt_dinolocal_condition,
        use_gt_clip_condition=use_gt_clip_condition,
        use_gt_imagebind_condition=use_gt_imagebind_condition,
        use_gt_imagebindlocal_condition=use_gt_imagebindlocal_condition,
    )

    if decode_method == "detection":
        hyper.update(
            max_think_token_n=1000,
            cfg_text_scale=cfg_text_scale,
            do_sample=do_sample,
            text_temperature=temperature,
        )
    elif decode_method in ("dino", "dinolocal", "clip", "imagebind", "imagebindlocal"):
        _codebook_max_tokens = {
            "dino": 17, "dinolocal": 1025,
            "clip": 785, "imagebind": 17, "imagebindlocal": 1025,
        }
        max_tok = _codebook_max_tokens.get(decode_method, 1025)
        hyper.update(
            max_think_token_n=max_tok,
            do_sample=do_sample,
            text_temperature=text_temperature,
            dino_pca=None,
            top_k=top_k,
            top_p=top_p,
            cfg_img_scale=cfg_img_scale,
        )
        hyper["modality_type_dict"]["condition"] = condition

    return hyper


def _preprocess_dino(image) -> Dict[str, Any]:
    """Compute GT DINO features for cosine-similarity comparison."""
    if image is None:
        return {}
    image_dino = image.resize((224, 224), Image.Resampling.LANCZOS)
    image_tensor = torch.from_numpy(np.array(image_dino)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    image_tensor = image_tensor.unsqueeze(0).to("cuda")
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval().to("cuda")
    with torch.no_grad():
        outputs = dinov2(image_tensor, is_training=True)
        if "x_norm_clstoken" in outputs:
            gt_global_feat = outputs["x_norm_clstoken"]
        else:
            gt_global_feat = None
            print("Warning: 'x_norm_clstoken' not found in DINOv2 outputs; skipping global feature save.")
    return {"gt_global_feat": gt_global_feat}


def _preprocess_dinolocal(image, output_path, condition, target) -> Dict[str, Any]:
    """Compute GT DINO-local features for cosine-similarity comparison."""
    if image is None:
        return {}
    image_dino = image.resize((448, 448), Image.Resampling.LANCZOS)
    image_tensor = torch.from_numpy(np.array(image_dino)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    image_tensor = image_tensor.unsqueeze(0).to("cuda")
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval().to("cuda")
    with torch.no_grad():
        outputs = dinov2(image_tensor, is_training=True)
        gt_feat = outputs["x_norm_patchtokens"]
        gt_feat_map = gt_feat.transpose(1, 2).reshape(1, 768, 32, 32)
        gt_feat_map_vis, pca = pca_image_sklearn(gt_feat_map)
        gt_feat_map_vis.save(os.path.join(output_path, f"{condition[0]}2{target}_gt_feat_map_vis.png"))
    return {"gt_feat_map": gt_feat_map, "pca": pca}


def _preprocess_clip(image, output_path, condition, target) -> Dict[str, Any]:
    """Compute GT CLIP ViT-B/16 local features at 448x448 for cosine-similarity comparison.

    Follows the same pipeline as tokenizer_clip.py:
    - Uses ``clip.load("ViT-B/16")`` (the ``clip`` package).
    - Resizes to 448x448, skips Resize/CenterCrop from clip_processor, keeps ToTensor + Normalize.
    - Runs the visual trunk manually, projects patch tokens from 768-d to 512-d via visual.proj.
    - Feature map: [B, 512, 28, 28].
    """
    if image is None:
        return {}
    try:
        import clip as clip_pkg
        import torchvision.transforms as transforms
    except ImportError:
        print("Warning: clip package not installed; skipping CLIP GT feature extraction.")
        return {}

    img_448 = image.resize((448, 448), Image.Resampling.LANCZOS)

    clip_model, clip_preprocess = clip_pkg.load("ViT-B/16", device="cuda")
    clip_model.eval()

    # Skip Resize(224) and CenterCrop(224); keep ToTensor + Normalize from clip_preprocess
    clip_no_resize = transforms.Compose(clip_preprocess.transforms[2:])
    pixel_values = clip_no_resize(img_448).unsqueeze(0).to("cuda")  # [1, 3, 448, 448]

    with torch.no_grad():
        visual = clip_model.visual
        pixel_values = pixel_values.to(dtype=visual.conv1.weight.dtype)
        x = visual.conv1(pixel_values)  # [B, width, grid, grid]
        grid_h, grid_w = x.shape[2], x.shape[3]  # 28, 28
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, grid*grid, width]
        x = torch.cat([visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1), x], dim=1)

        # Interpolate position embeddings for 448 resolution
        pos_emb = visual.positional_embedding.to(x.dtype)
        if pos_emb.dim() == 2:
            pos_emb = pos_emb.unsqueeze(0)  # [1, L, C]
        if pos_emb.size(1) != x.size(1):
            pos_emb = torch.nn.functional.interpolate(
                pos_emb.permute(0, 2, 1), size=x.size(1), mode="linear", align_corners=False
            ).permute(0, 2, 1)
        x = x + pos_emb

        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = visual.ln_post(x)

        # Extract patch tokens (skip CLS), project 768-d -> 512-d
        patch_tokens = x[:, 1:, :]  # [B, 784, 768]
        if hasattr(visual, "proj") and visual.proj is not None:
            patch_tokens = patch_tokens @ visual.proj  # [B, 784, 512]
        else:
            patch_tokens = patch_tokens[..., :512]

        C = patch_tokens.shape[-1]  # 512
        gt_feat_map = patch_tokens.transpose(1, 2).reshape(1, C, grid_h, grid_w)  # [1, 512, 28, 28]
        gt_feat_map_vis, pca = pca_image_sklearn(gt_feat_map.float())
        gt_feat_map_vis.save(os.path.join(output_path, f"{condition[0]}2{target}_gt_feat_map_vis.png"))

    del clip_model
    torch.cuda.empty_cache()
    return {"gt_feat_map": gt_feat_map.float(), "pca": pca}


def _preprocess_imagebind(image) -> Dict[str, Any]:
    """Compute GT ImageBind (global) CLS feature for cosine-similarity comparison.

    Mirrors tokenizer_imagebind.py's global path: 224 input -> ImageBind
    vision trunk -> CLS token (trunk_out[:, 0, :]) -> [1, 1280]. This is the
    feature the global imagebind codebook is trained against. Returns the same
    `gt_global_feat` key `_postprocess_imagebind` consumes (analogous to
    `_preprocess_dino`).
    """
    if image is None:
        return {}
    try:
        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType
        import torchvision.transforms as transforms
    except ImportError:
        print("Warning: imagebind not installed; skipping ImageBind GT feature extraction.")
        return {"gt_global_feat": None}

    IMAGEBIND_MEAN = (0.48145466, 0.4578275, 0.40821073)
    IMAGEBIND_STD = (0.26862954, 0.26130258, 0.27577711)

    # Global path uses 224 input (tokenizer_imagebind.py:511).
    ib_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGEBIND_MEAN, std=IMAGEBIND_STD),
    ])
    pixel_values = ib_transform(image).unsqueeze(0).to("cuda")  # [1, 3, 224, 224]

    ib_model = imagebind_model.imagebind_huge(pretrained=True).eval().to("cuda")
    with torch.no_grad():
        modality_value = ib_model.modality_preprocessors[ModalityType.VISION](vision=pixel_values)
        trunk_inputs = modality_value["trunk"]
        trunk_out = ib_model.modality_trunks[ModalityType.VISION](**trunk_inputs)  # [B, 1+N, 1280]
        cls_token = trunk_out[:, 0, :].float()  # [1, 1280] — global CLS
    del ib_model
    torch.cuda.empty_cache()
    return {"gt_global_feat": cls_token}


def _preprocess_imagebindlocal(image, output_path, condition, target) -> Dict[str, Any]:
    """Compute GT ImageBind ViT-H/14 local features at 448x448 for cosine-similarity comparison.

    Follows the same pipeline as tokenizer_imagebind.py:
    - Uses ``imagebind_model.imagebind_huge(pretrained=True)``.
    - Resize to 448, apply ImageBind vision transform (CLIP-style normalization).
    - Run modality_preprocessors -> modality_trunks -> extract patch tokens.
    - Feature map: [B, 1280, 32, 32].
    """
    if image is None:
        return {}
    try:
        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType
        import torchvision.transforms as transforms
    except ImportError:
        print("Warning: imagebind not installed; skipping ImageBind GT feature extraction.")
        return {}

    # ImageBind vision normalization (same as CLIP)
    IMAGEBIND_MEAN = (0.48145466, 0.4578275, 0.40821073)
    IMAGEBIND_STD = (0.26862954, 0.26130258, 0.27577711)

    img_448 = image.resize((448, 448), Image.Resampling.LANCZOS)
    ib_transform = transforms.Compose([
        transforms.Resize(448, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGEBIND_MEAN, std=IMAGEBIND_STD),
    ])
    pixel_values = ib_transform(img_448).unsqueeze(0).to("cuda")  # [1, 3, 448, 448]

    ib_model = imagebind_model.imagebind_huge(pretrained=True)
    ib_model = ib_model.eval().to("cuda")

    with torch.no_grad():
        # Same forward as imagebind_vision_forward() in tokenizer_imagebind.py
        modality_value = ib_model.modality_preprocessors[ModalityType.VISION](vision=pixel_values)
        trunk_inputs = modality_value["trunk"]
        trunk_out = ib_model.modality_trunks[ModalityType.VISION](**trunk_inputs)  # [B, 1+N, 1280]

        patch_tokens = trunk_out[:, 1:, :]  # [B, N, 1280] — skip CLS
        B, N, C = patch_tokens.shape
        S = int(N ** 0.5)  # 32
        gt_feat_map = patch_tokens.transpose(1, 2).reshape(B, C, S, S)  # [B, 1280, 32, 32]

        gt_feat_map_vis, pca = pca_image_sklearn(gt_feat_map.float())
        gt_feat_map_vis.save(os.path.join(output_path, f"{condition[0]}2{target}_gt_feat_map_vis.png"))

    del ib_model
    torch.cuda.empty_cache()
    return {"gt_feat_map": gt_feat_map.float(), "pca": pca}
