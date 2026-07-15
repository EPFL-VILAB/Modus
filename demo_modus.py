#!/usr/bin/env python3
"""
MODUS any-to-any Gradio demo (3 tabs).

This is a thin UI over the existing demo backend
(``demo_my/load_any2any.py`` + ``demo_my/any2any_tasks.py``).  The model is
loaded ONCE at startup and shared by all three tabs.

Tabs
----
1. Any-to-Any   — pick one CONDITION modality (image upload or caption text)
                  and a multi-select set of TARGET modalities; every checked
                  target is generated (condition -> target) and shown in a
                  gallery.  4M-style.
2. Chained      — condition -> intermediate -> target.  The intermediate is
                  restricted to a "bridge" set so the chain is never
                  meaningless.  Both the intermediate and the final result are
                  shown.  Recommended-chain quick-pick buttons fill the three
                  selectors.
3. Representation Analysis
                  — RGB->Depth or RGB->Normal run THREE times with the input
                  conditioned on {ViT only, VAE only, ViT+VAE}, shown
                  side-by-side.  Uses the new ``cond_feature_mode`` plumbing.

Running it
----------
1. Edit the CONFIG constants below (or set the matching env vars):
       CHECKPOINT_PATH   — training checkpoint dir (model.safetensors / ema.safetensors)
       MODEL_PATH        — base model folder (BAGEL-7B-MoT)
       MODEL_NAME        — usually "bagel_from_json"
       MODALITY_CONFIG   — conf/modalities/*.yaml
   Defaults are read from conf/inference/base.yaml where possible.
2. Launch on a GPU node:
       python demo_modus.py --port 7860
   then open http://<node>:7860 .
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from any2any.any2any_tasks import (  # noqa: E402
    create_inferencer,
    generate_any,
    run_inference_task,
    _build_understanding_hyper,
    _run_inference_loop,
    _set_seed,
    _load_image,
)
from any2any.load_any2any import (  # noqa: E402
    load_any2any_model_training_checkpoint,
)


# ── CONFIG (edit these) ───────────────────────────────────────────────────────
# Defaults mirror conf/inference/base.yaml.  Env vars take precedence so the
# user can override without editing the file.
CHECKPOINT_PATH = os.environ.get("MODUS_DEMO_CHECKPOINT", "")  # REQUIRED
MODEL_PATH = os.environ.get("BAGEL_MODEL_PATH", "models/BAGEL-7B-MoT")
MODEL_NAME = os.environ.get("MODUS_DEMO_MODEL_NAME", "bagel_from_json")
MODALITY_CONFIG = os.environ.get(
    "MODUS_DEMO_MODALITY_CONFIG", "conf/modalities/instruction_16mod_stage2.yaml"
)
USE_EMA = os.environ.get("MODUS_DEMO_USE_EMA", "0") == "1"
# Generation resolution for image-like targets (512 is ~3-4x faster / less GPU
# time than 1024, with little quality loss on depth/normal/seg/etc.).
GEN_IMAGE_SIZE = int(os.environ.get("MODUS_DEMO_IMAGE_SIZE", "512"))


# ── Modality sets (from the repo training configs; fallbacks) ─────────────────
# Targets the model can produce.
# The 14 target modalities of the 16-mod model (all modalities except the
# instruction-only `text` and the legacy `det`; `cocodet` is the real detection).
GENERATABLE_TARGETS = [
    "rgb", "caption", "depth", "normal", "canny", "seg", "samseg", "samedge",
    "cocodet", "dino", "dinolocal", "clip", "imagebind", "imagebindlocal",
]
# Image-like targets (produce a displayable RGB image directly).
IMAGE_LIKE = {"rgb", "depth", "normal", "canny", "seg", "samseg", "samedge"}
# Modalities good as a chained intermediate ("bridge" set).
BRIDGE_MODALITIES = ["rgb", "depth", "normal", "seg", "canny", "caption"]
# Conditions that are provided as free text rather than an image.
TEXT_CONDITIONS = {"caption", "text"}

# Human-readable labels for the modality codenames. Shown everywhere in the UI;
# the underlying dropdown/checkbox VALUE stays the codename (via (label, value)
# tuples), so nothing downstream changes.
DISPLAY_NAMES = {
    "rgb": "RGB image",
    "caption": "Caption (text)",
    "text": "Text",
    "depth": "Depth",
    "normal": "Surface normals",
    "canny": "Canny edges",
    "seg": "Semantic segmentation",
    "samseg": "SAM segmentation",
    "samedge": "SAM edge",
    "cocodet": "Object detection (COCO)",
    "det": "Grounding boxes",
    "dino": "DINOv2 features",
    "dinolocal": "DINOv2 patch features",
    "clip": "CLIP features",
    "imagebind": "ImageBind features",
    "imagebindlocal": "ImageBind patch features",
}


def _label(code: str) -> str:
    """Human-readable label for a modality codename."""
    return DISPLAY_NAMES.get(code, code)


def _labeled(codes: List[str]) -> List[Tuple[str, str]]:
    """[(display_label, codename), ...] for gradio dropdown/checkbox choices."""
    return [(_label(c), c) for c in codes]


def _resolve_path(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    p = str(p)
    return p if p.startswith("/") else os.path.join(REPO_ROOT, p)


# ── Model holder (loaded once, lazily on first request if startup fails) ──────
class _ModelHolder:
    def __init__(self):
        self.inferencer = None
        self.modality_registry = None
        self.load_error: Optional[str] = None
        # Feature targets disabled because their external tokenizers (fourm)
        # are unavailable in this environment.
        self.disabled_targets: set = set()

    def ensure_loaded(self):
        if self.inferencer is not None:
            return
        if not CHECKPOINT_PATH:
            raise RuntimeError(
                "CHECKPOINT_PATH is not set. Edit the CONFIG block at the top of "
                "demo_modus.py or set MODUS_DEMO_CHECKPOINT."
            )
        print(f"[demo] loading model from checkpoint={CHECKPOINT_PATH}")
        model, vae_model, tokenizer, new_token_ids, modality_registry = (
            load_any2any_model_training_checkpoint(
                checkpoint_path=_resolve_path(CHECKPOINT_PATH),
                model_path=_resolve_path(MODEL_PATH),
                model_name=MODEL_NAME,
                init_on_gpu=True,
                use_ema=USE_EMA,
                modality_config_path=_resolve_path(MODALITY_CONFIG),
            )
        )
        # Auto-enable any external tokenizers the loaded checkpoint declares, so
        # feature targets (dino/clip/imagebind) decode at inference.
        ext_flags = {}
        for name in ("dino", "dinolocal", "clip", "imagebind", "imagebindlocal"):
            try:
                if modality_registry.needs_external_tokenizer(name):
                    ext_flags[f"use_{name}_tokenizer"] = True
            except Exception:
                pass
        try:
            self.inferencer = create_inferencer(
                model=model,
                vae_model=vae_model,
                tokenizer=tokenizer,
                new_token_ids=new_token_ids,
                modality_registry=modality_registry,
                **ext_flags,
            )
        except Exception as e:
            # External feature tokenizers need the `fourm` package AND a runtime
            # download of each VQVAE from the Hub. If fourm is missing OR any
            # tokenizer fails to build (network, device, etc.), degrade gracefully:
            # load without them and hide the feature targets, so the other targets
            # keep working instead of taking down the whole model load.
            print(f"[demo] external feature tokenizers unavailable ({e}); "
                  f"disabling feature targets (dino/clip/imagebind).")
            self.disabled_targets = {
                "dino", "dinolocal", "clip", "imagebind", "imagebindlocal",
            }
            self.inferencer = create_inferencer(
                model=model,
                vae_model=vae_model,
                tokenizer=tokenizer,
                new_token_ids=new_token_ids,
                modality_registry=modality_registry,
            )
        self.modality_registry = modality_registry
        print("[demo] model ready.")

    def available_modalities(self) -> List[str]:
        if self.modality_registry is None:
            return []
        try:
            return list(self.modality_registry.name_to_id().keys())
        except Exception:
            return []


HOLDER = _ModelHolder()


def _supported_targets() -> List[str]:
    """Generatable targets intersected with what the loaded checkpoint supports,
    minus any feature targets disabled because their tokenizers are missing."""
    avail = set(HOLDER.available_modalities())
    pool = GENERATABLE_TARGETS if not avail else [m for m in GENERATABLE_TARGETS if m in avail]
    return [m for m in pool if m not in HOLDER.disabled_targets]


def _supported_conditions() -> List[str]:
    avail = HOLDER.available_modalities()
    if not avail:
        # Fallback: generatable set is a reasonable condition list too.
        avail = list(GENERATABLE_TARGETS)
    # 'text' and 'caption' are the same text-prompt condition here; keep only
    # 'caption' so the dropdown does not offer a duplicate.
    return [m for m in avail if m != "text"]


def _supported_bridges() -> List[str]:
    avail = set(HOLDER.available_modalities())
    if not avail:
        return list(BRIDGE_MODALITIES)
    return [m for m in BRIDGE_MODALITIES if m in avail]


# ── Core task runner ──────────────────────────────────────────────────────────
def _decode_method(target: str) -> str:
    try:
        return HOLDER.inferencer._resolve_decode_method(target)
    except Exception:
        return "image"


def _collect_pngs(out_dir: str) -> List[str]:
    paths = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.lower().endswith(".png"):
            paths.append(os.path.join(out_dir, fn))
    return paths


def run_task(
    *,
    condition: str,
    target: str,
    input_image=None,
    prompt: str = "",
    intermediate: Optional[str] = None,
    cfg_text_scale: float = 4.0,
    cfg_img_scale: float = 2.0,
    num_timesteps: int = 50,
    seed: int = 0,
    top_p: float = 1.0,
    top_k: int = 0,
    cond_feature_mode: str = "both",
    seg_category: str = "person",
) -> Tuple[List[Tuple[Image.Image, str]], str]:
    """Run a single (condition -> [intermediate ->] target) task.

    Returns (named_images, text).  ``named_images`` is a list of
    ``(PIL image, filename)`` tuples (final + intermediate PNGs); ``text`` is
    any text output (caption/grounding) or "".  The filename lets callers tell
    the final result from intermediate panels.
    """
    HOLDER.ensure_loaded()
    decode_method = _decode_method(target)
    is_text_target = decode_method in ("text", "caption", "grounding")

    cfg_img = cfg_img_scale
    try:
        cfg_img = HOLDER.modality_registry.resolve_cfg_img_scale(target, default=cfg_img_scale)
    except Exception:
        pass

    with tempfile.TemporaryDirectory(prefix="modus_demo_") as out_dir:
        if is_text_target and intermediate:
            # Chained text target (e.g. rgb -> normal -> caption): the chained
            # path lives in generate_any/unified_inference, NOT the understanding
            # loop.  Capture the returned text (and any intermediate PNGs).
            results = generate_any(
                HOLDER.inferencer, out_dir,
                prompt=prompt, input_image=input_image,
                condition=[condition], target=target, intermediate=intermediate,
                cfg_text_scale=cfg_text_scale, cfg_img_scale=cfg_img,
                num_timesteps=num_timesteps, seed=seed, num_samples=1,
                image_size=GEN_IMAGE_SIZE,
            )
            texts: List[str] = []
            for r in results:
                t = r.get("text")
                if isinstance(t, list):
                    texts.extend([x for x in t if isinstance(x, str)])
                elif isinstance(t, str):
                    texts.append(t)
            return [], "\n".join(texts)

        if is_text_target:
            # Direct text output (caption / grounding / vqa): run the
            # understanding loop directly so we capture the returned string.
            _set_seed(seed)
            image = _load_image(input_image) if input_image is not None else None
            hyper = _build_understanding_hyper(
                decode_method=decode_method,
                condition=[condition], target=target,
                cfg_text_scale=cfg_text_scale, cfg_img_scale=cfg_img,
                use_instruction=True, use_target_instruction=True,
                use_condition_instruction=False, do_modality_norm=False,
                do_sample=(top_p < 1.0 or top_k > 0),
                temperature=0.003, text_temperature=0.95,
                top_k=top_k, top_p=top_p,
            )
            texts: List[str] = []

            def _on_result(result, i):
                t = result.get("text")
                if isinstance(t, list):
                    texts.extend([x for x in t if isinstance(x, str)])
                elif isinstance(t, str):
                    texts.append(t)

            _run_inference_loop(
                HOLDER.inferencer, hyper,
                image=image, text=prompt,
                understanding_output=True, num_samples=1,
                callback=_on_result,
            )
            return [], "\n".join(texts)

        # Image / feature-visualisation targets: reuse run_inference_task, which
        # decodes + saves displayable PNGs (and intermediate PNGs when chained).
        run_inference_task(
            inferencer=HOLDER.inferencer,
            output_path=out_dir,
            prompt=prompt,
            input_image=input_image,
            condition=[condition],
            target=target,
            intermediate=intermediate,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img,
            num_timesteps=num_timesteps,
            seed=seed,
            top_k=top_k,
            top_p=top_p,
            num_samples=1,
            image_size=GEN_IMAGE_SIZE,
            seg_category=seg_category,
            save_intermediate=intermediate is not None,
        )
        named = [
            (Image.open(p).copy(), os.path.basename(p))
            for p in _collect_pngs(out_dir)
        ]
        return named, ""


# ── Tab 3 needs cond_feature_mode threaded through generate_any directly ──────
def run_representation_task(
    *,
    target: str,
    input_image,
    cfg_text_scale: float,
    cfg_img_scale: float,
    num_timesteps: int,
    seed: int,
    cond_feature_mode: str,
) -> Image.Image:
    """RGB -> {depth|normal} with a chosen condition-feature mode.

    Image-like target so the inferencer returns a PIL image in result['image'].
    """
    HOLDER.ensure_loaded()
    _set_seed(seed)
    cfg_img = cfg_img_scale
    try:
        cfg_img = HOLDER.modality_registry.resolve_cfg_img_scale(target, default=cfg_img_scale)
    except Exception:
        pass
    results = generate_any(
        HOLDER.inferencer, tempfile.gettempdir(),
        prompt="", input_image=input_image,
        condition=["rgb"], target=target, intermediate=None,
        cfg_text_scale=cfg_text_scale, cfg_img_scale=cfg_img,
        num_timesteps=num_timesteps, seed=seed, num_samples=1,
        cond_feature_mode=cond_feature_mode, image_size=GEN_IMAGE_SIZE,
    )
    raw = results[0]["image"]
    if isinstance(raw, (list, tuple)):
        raw = raw[-1]
    return raw


# ── Gradio callbacks ──────────────────────────────────────────────────────────
def _is_text_condition(condition: str) -> bool:
    return condition in TEXT_CONDITIONS


def _toggle_condition_input(condition: str):
    """Show image upload for image conditions, textbox for text conditions."""
    text_cond = _is_text_condition(condition)
    return (
        gr.update(visible=not text_cond),  # image input
        gr.update(visible=text_cond),      # text input
    )


def tab1_generate(
    condition, targets, image_in, text_in,
    seed, randomize, cfg_text, cfg_img, num_timesteps, top_p, top_k, seg_cat,
):
    if not targets:
        return [], "Select at least one target modality."
    text_cond = _is_text_condition(condition)
    if text_cond and not (text_in and text_in.strip()):
        return [], f"Condition '{condition}' needs text input."
    if not text_cond and image_in is None:
        return [], f"Condition '{condition}' needs an image input."

    if randomize:
        seed = int.from_bytes(os.urandom(2), "little")

    gallery: List[Tuple[Any, str]] = []
    status_lines: List[str] = []
    _SKIP = ("_intermediate_", "_input", "_compare", "_gt_feat_map", "_bbox", "_cos_sim")
    for tgt in targets:
        try:
            named, text = run_task(
                condition=condition, target=tgt,
                input_image=None if text_cond else image_in,
                prompt=(text_in or "") if text_cond else "",
                cfg_text_scale=cfg_text, cfg_img_scale=cfg_img,
                num_timesteps=int(num_timesteps), seed=int(seed),
                top_p=top_p, top_k=int(top_k),
                seg_category=(seg_cat or "person").strip(),
            )
            if text:
                status_lines.append(f"{tgt}: {text}")
            primary = [(im, fn) for im, fn in named if not any(s in fn for s in _SKIP)]
            if not primary:
                primary = named
            for k, (im, _fn) in enumerate(primary):
                label = tgt if len(primary) == 1 else f"{tgt} [{k}]"
                gallery.append((im, label))
        except Exception as e:
            traceback.print_exc()
            status_lines.append(f"{tgt}: ERROR {e}")
    return gallery, "\n".join(status_lines) if status_lines else "Done."


def tab2_generate(
    condition, intermediate, target, image_in, text_in,
    seed, randomize, cfg_text, cfg_img, num_timesteps, top_p, top_k,
):
    text_cond = _is_text_condition(condition)
    if text_cond and not (text_in and text_in.strip()):
        return None, "", "", f"Condition '{condition}' needs text input."
    if not text_cond and image_in is None:
        return None, "", "", f"Condition '{condition}' needs an image input."
    if randomize:
        seed = int.from_bytes(os.urandom(2), "little")

    inter = None if (not intermediate or intermediate.lower() == "none") else intermediate
    try:
        named, text = run_task(
            condition=condition, target=target,
            input_image=None if text_cond else image_in,
            prompt=(text_in or "") if text_cond else "",
            intermediate=inter,
            cfg_text_scale=cfg_text, cfg_img_scale=cfg_img,
            num_timesteps=int(num_timesteps), seed=int(seed),
            top_p=top_p, top_k=int(top_k),
        )
    except Exception as e:
        traceback.print_exc()
        return None, None, "", f"ERROR: {e}"

    # run_inference_task saves the final image plus intermediate PNGs
    # (named *_intermediate_*.png).  Split them by filename for labelling.
    final_img = None
    inter_img = None
    _SKIP = ("_input", "_compare", "_gt_feat_map", "_bbox", "_cos_sim")
    for im, fn in named:
        if "_intermediate_" in fn:
            if inter_img is None:
                inter_img = im
        elif not any(s in fn for s in _SKIP):
            if final_img is None:
                final_img = im

    if target in ("caption", "grounding", "text"):
        # Text target: there is no final image; show the text instead.
        status = f"final ({target}): {text}" if text else "Done."
        return None, inter_img, status, status
    status = f"intermediate text: {text}" if text else "Done."
    return final_img, inter_img, status, status


def tab3_generate(target, image_in, seed, cfg_text, cfg_img):
    if image_in is None:
        return None, None, None, "Provide an RGB image."
    outs = {}
    for mode in ("vit", "vae", "both"):
        try:
            outs[mode] = run_representation_task(
                target=target, input_image=image_in,
                cfg_text_scale=cfg_text, cfg_img_scale=cfg_img,
                num_timesteps=30, seed=int(seed), cond_feature_mode=mode,
            )
        except Exception as e:
            traceback.print_exc()
            outs[mode] = None
    status = "Done." if all(v is not None for v in outs.values()) else "One or more modes failed (see logs)."
    return outs["vit"], outs["vae"], outs["both"], status


# ── UI ────────────────────────────────────────────────────────────────────────
# Per-example segmentation category: selecting an example binds a sensible 'seg'
# target category (the image's main object) into the seg-category box, so a 'seg'
# generation on that example is meaningful out of the box.
_EXAMPLE_SEG_CATEGORY = {
    "01_mall.jpg": "person",
    "02_red_barn.jpg": "window",
    "03_kremlin_clock.jpg": "building",
    "04_dragon_boat.jpg": "boat",
    "05_red_carpet.jpg": "person",
    "06_train.jpg": "train",
    "07_city_bus.jpg": "bus",
    "08_food_market.jpg": "person",
    "09_waterfall.jpg": "water",
    "10_village_car.jpg": "car",
}


def _example_images() -> List[str]:
    d = os.path.join(REPO_ROOT, "test_images")
    if not os.path.isdir(d):
        return []
    return [
        os.path.join(d, f) for f in sorted(os.listdir(d))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and "_seg." not in f.lower()  # exclude the precomputed seg previews
    ]


# Modalities shown in the offline precompute grid for each example image, in
# order. Each is stored next to the example as ``<stem>_<modality>.jpg``.
PRECOMPUTE_GRID_MODALITIES = [
    "depth", "normal", "seg", "canny", "cocodet",
    "dinolocal", "clip", "imagebind",
]


def _precompute_grid_for(image_path: str) -> List[Tuple[str, str]]:
    """[(precomputed_image_path, display_label), ...] of the offline RGB→X previews
    that sit next to *image_path* as ``<stem>_<modality>.jpg``. Zero GPU — this is
    a static preview of what the model produces for this example image."""
    stem, _ext = os.path.splitext(image_path)
    grid = []
    for mod in PRECOMPUTE_GRID_MODALITIES:
        prev = f"{stem}_{mod}.jpg"
        if os.path.exists(prev):
            grid.append((prev, _label(mod)))
    return grid


def _example_rows() -> List[List[Any]]:
    """[[image_path, seg_category, precompute_grid], ...] — selecting a row fills the
    input image, its bound seg category, AND shows the offline-precomputed grid of
    every modality for that image (no GPU needed)."""
    rows = []
    for p in _example_images():
        cat = _EXAMPLE_SEG_CATEGORY.get(os.path.basename(p), "person")
        rows.append([p, cat, _precompute_grid_for(p)])
    return rows


def _advanced_controls():
    """Shared advanced accordion. Returns the control components."""
    with gr.Accordion("Advanced", open=False):
        with gr.Row():
            seed = gr.Number(value=0, label="Seed", precision=0)
            randomize = gr.Checkbox(value=False, label="Randomize seed")
        with gr.Row():
            cfg_text = gr.Slider(
                0.0, 10.0, value=4.0, step=0.5, label="Text guidance strength",
                info="How strongly to follow the text/caption condition (classifier-free "
                     "guidance, cfg_text_scale). Higher = more faithful to the text.")
            cfg_img = gr.Slider(
                0.0, 10.0, value=2.0, step=0.5, label="Image guidance strength",
                info="How strongly to follow the input image condition (classifier-free "
                     "guidance, cfg_img_scale). Higher = more faithful to the input image.")
        with gr.Row():
            num_timesteps = gr.Slider(
                10, 100, value=30, step=5, label="Diffusion steps",
                info="Denoising steps for image outputs (depth, surface normals, etc.). "
                     "More steps = higher quality but slower.")
            top_p = gr.Slider(
                0.0, 1.0, value=1.0, step=0.05, label="Top-p (nucleus sampling)",
                info="For token outputs (caption, detection): sample from the smallest set "
                     "of tokens whose probabilities sum to at least p. 1.0 = disabled.")
            top_k = gr.Number(
                value=0, label="Top-k sampling", precision=0,
                info="For token outputs: sample only from the k most likely tokens. 0 = disabled.")
    return seed, randomize, cfg_text, cfg_img, num_timesteps, top_p, top_k


def build_ui() -> gr.Blocks:
    cond_choices = _supported_conditions()
    target_choices = _supported_targets()
    bridge_choices = _supported_bridges()
    default_cond = "rgb" if "rgb" in cond_choices else (cond_choices[0] if cond_choices else "rgb")
    default_targets = [t for t in ("depth", "normal", "seg") if t in target_choices]

    with gr.Blocks(title="MODUS any-to-any") as demo:
        gr.Markdown(
            "# MODUS: Decoder-only Any-to-Any Modeling of Diverse Modalities\n"
            "MODUS is a single **decoder-only** model trained jointly on many "
            "modalities (RGB, depth, surface normals, segmentation, detection, "
            "edges, captions, and learned features such as DINO / CLIP / ImageBind). "
            "Given any of them as input, it can generate any of the others. "
            "This demo shows three ways to use it.\n\n"
            "🌐 [Project page](https://modus-multimodal.epfl.ch/) · "
            "📄 [Paper](https://storage.googleapis.com/multimodal_modus/static/modus_paper.pdf)\n"
            "*EPFL · Apple · University of Copenhagen · CUHK · University of Geneva · Lambda AI*"
        )
        if HOLDER.load_error:
            gr.Markdown(f"**Model failed to load at startup:** {HOLDER.load_error}\n\nIt will retry on first request.")

        # ── Tab 1: Any-to-Any ────────────────────────────────────────────────
        with gr.Tab("Any-to-Any"):
            gr.Markdown(
                "### Any-to-Any Generation\n"
                "Pick one input modality (an image or a caption) and generate any set "
                "of target modalities from it. Each target is generated on its own, "
                "conditioned only on the input and not on the other targets."
            )
            with gr.Row():
                with gr.Column():
                    t1_cond = gr.Dropdown(_labeled(cond_choices), value=default_cond, label="Input modality")
                    t1_image = gr.Image(type="pil", label="Input image", visible=not _is_text_condition(default_cond))
                    t1_text = gr.Textbox(label="Input text (caption)", visible=_is_text_condition(default_cond))
                    t1_targets = gr.CheckboxGroup(_labeled(target_choices), value=default_targets, label="Output modalities to generate")
                    t1_seg_cat = gr.Textbox(value="person", label="Segmentation category (used when Semantic segmentation is an output)")
                    t1_seed, t1_rand, t1_cfgt, t1_cfgi, t1_steps, t1_topp, t1_topk = _advanced_controls()
                    t1_btn = gr.Button("Generate", variant="primary")
                    t1_precompute = gr.Gallery(
                        label="Precomputed outputs for this example — RGB → every modality (no GPU)",
                        columns=4, height="auto", allow_preview=True, show_label=True,
                    )
                    ex = _example_rows()
                    if ex:
                        # Clicking an example fills the input image + its bound seg
                        # category AND shows the offline precompute grid: what the
                        # model produces for that image across every modality, with
                        # no GPU cost. Run "Generate" only for your own images.
                        t1_examples = gr.Gallery(
                            value=[r[0] for r in ex], label="Example images (click to load)",
                            columns=5, height="auto", allow_preview=False, show_label=True,
                        )

                        def _pick_example(evt: gr.SelectData):
                            r = ex[evt.index]
                            return r[0], r[1], r[2]

                        t1_examples.select(
                            _pick_example, None, [t1_image, t1_seg_cat, t1_precompute],
                        )
                with gr.Column():
                    t1_gallery = gr.Gallery(label="Results", columns=4)
                    t1_status = gr.Markdown()
            t1_cond.change(_toggle_condition_input, t1_cond, [t1_image, t1_text])
            t1_btn.click(
                tab1_generate,
                [t1_cond, t1_targets, t1_image, t1_text,
                 t1_seed, t1_rand, t1_cfgt, t1_cfgi, t1_steps, t1_topp, t1_topk, t1_seg_cat],
                [t1_gallery, t1_status],
            )

        # ── Tab 2: Chained ───────────────────────────────────────────────────
        with gr.Tab("Chained Prediction"):
            gr.Markdown(
                "### Chained Prediction\n"
                "Generate modalities one after another, where each one is conditioned "
                "on the input and on every modality generated before it (for example "
                "RGB, then depth, then surface normals). Because of this self-conditioning "
                "the outputs are mutually consistent, unlike generating each one "
                "independently. Pick a condition, an intermediate (bridge) modality, and "
                "a final target, or use a quick-pick chain below."
            )
            with gr.Row():
                with gr.Column():
                    t2_cond = gr.Dropdown(_labeled(cond_choices), value=default_cond, label="Input modality")
                    t2_inter = gr.Dropdown(_labeled(bridge_choices), value=(bridge_choices[0] if bridge_choices else None), label="Intermediate (bridge) modality")
                    t2_target = gr.Dropdown(_labeled(target_choices), value=(target_choices[0] if target_choices else None), label="Final output modality")
                    t2_image = gr.Image(type="pil", label="Input image", visible=not _is_text_condition(default_cond))
                    t2_text = gr.Textbox(label="Input text (caption)", visible=_is_text_condition(default_cond))
                    ex2 = _example_images()
                    if ex2:
                        gr.Examples(ex2, inputs=t2_image, label="Example images")
                    with gr.Row():
                        qp1 = gr.Button("Caption → RGB → Depth")
                        qp2 = gr.Button("RGB → Depth → Surface normals")
                        qp3 = gr.Button("RGB → Surface normals → Caption")
                    t2_seed, t2_rand, t2_cfgt, t2_cfgi, t2_steps, t2_topp, t2_topk = _advanced_controls()
                    t2_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    t2_final = gr.Image(label="Final output")
                    t2_inter_out = gr.Image(label="Intermediate output")
                    t2_text_out = gr.Markdown()
                    t2_status = gr.Markdown()
            t2_cond.change(_toggle_condition_input, t2_cond, [t2_image, t2_text])

            def _fill_chain(c, m, t):
                return (
                    gr.update(value=c), gr.update(value=m), gr.update(value=t),
                    *_toggle_condition_input(c),
                )
            qp1.click(lambda: _fill_chain("caption", "rgb", "depth"), None, [t2_cond, t2_inter, t2_target, t2_image, t2_text])
            qp2.click(lambda: _fill_chain("rgb", "depth", "normal"), None, [t2_cond, t2_inter, t2_target, t2_image, t2_text])
            qp3.click(lambda: _fill_chain("rgb", "normal", "caption"), None, [t2_cond, t2_inter, t2_target, t2_image, t2_text])

            t2_btn.click(
                tab2_generate,
                [t2_cond, t2_inter, t2_target, t2_image, t2_text,
                 t2_seed, t2_rand, t2_cfgt, t2_cfgi, t2_steps, t2_topp, t2_topk],
                [t2_final, t2_inter_out, t2_text_out, t2_status],
            )

        # ── Tab 3: Representation Analysis ───────────────────────────────────
        with gr.Tab("ViT vs VAE Features"):
            gr.Markdown(
                "### Input Features: ViT vs VAE\n"
                "MODUS can represent an input image with two kinds of features: **ViT** "
                "features (from a semantic encoder) and **VAE** features (from a "
                "reconstruction encoder). The same RGB→target is run three times, feeding "
                "the input as ViT features only, VAE features only, or both, so you can "
                "see how each representation changes the generated output."
            )
            with gr.Row():
                with gr.Column():
                    t3_task = gr.Radio(_labeled(["depth", "normal"]), value="depth", label="Output (RGB → ?)")
                    t3_image = gr.Image(type="pil", label="RGB input")
                    ex3 = _example_images()
                    if ex3:
                        gr.Examples(ex3, inputs=t3_image, label="Example images")
                    with gr.Row():
                        t3_seed = gr.Number(value=0, label="Seed", precision=0)
                        t3_cfgt = gr.Slider(0.0, 10.0, value=4.0, step=0.5, label="Text guidance strength",
                            info="Classifier-free guidance for text (cfg_text_scale).")
                        t3_cfgi = gr.Slider(0.0, 10.0, value=2.0, step=0.5, label="Image guidance strength",
                            info="Classifier-free guidance for the input image (cfg_img_scale).")
                    t3_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    with gr.Row():
                        t3_vit = gr.Image(label="ViT only")
                        t3_vae = gr.Image(label="VAE only")
                        t3_both = gr.Image(label="ViT+VAE")
                    t3_status = gr.Markdown()
            t3_btn.click(
                tab3_generate,
                [t3_task, t3_image, t3_seed, t3_cfgt, t3_cfgi],
                [t3_vit, t3_vae, t3_both, t3_status],
            )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    # Try eager load so the UI shows a clear error if config is wrong; the app
    # still launches and retries lazily on first request.
    try:
        HOLDER.ensure_loaded()
    except Exception as e:
        HOLDER.load_error = str(e)
        print(f"[demo] startup model load failed: {e}")
        traceback.print_exc()

    demo = build_ui()
    demo.launch(server_name=args.server_name, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
