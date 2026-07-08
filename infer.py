#!/usr/bin/env python3
"""
Unified inference entry-point — config-free modality routing.

Usage:

    # RGB → Depth
    python infer.py --condition rgb --target depth \\
        checkpoint_path=/path/to/ckpt input_image=test.jpg

    # RGB → DINO-local
    python infer.py --condition rgb --target dinolocal \\
        checkpoint_path=/path/to/ckpt input_image=test.jpg

    # Text → Image
    python infer.py --condition caption --target image \\
        checkpoint_path=/path/to/ckpt prompt="a cat sitting on a mat"

    # Chained: Text → Depth → Image
    python infer.py --condition caption --target image --intermediate depth \\
        checkpoint_path=/path/to/ckpt prompt="a house by the lake"

    # Chained: RGB → Depth → Image (edit)
    python infer.py --condition rgb --target image --intermediate depth \\
        checkpoint_path=/path/to/ckpt input_image=test.jpg

    # Override any default:
    python infer.py --condition rgb --target dinolocal \\
        checkpoint_path=/path/to/ckpt cfg_text_scale=6.0 num_timesteps=100

All model/hyperparameter defaults come from conf/inference/base.yaml.
Modality-specific behaviour (decode method, external tokenizers, CFG scales)
is auto-derived from conf/modalities/instruction_16mod_stage2.yaml.
"""

import os
import sys


def _parse_args():
    """Parse --condition/--target/--intermediate and key=value overrides from CLI."""
    try:
        from omegaconf import OmegaConf
    except ImportError as e:
        raise RuntimeError("infer.py requires omegaconf to be installed.") from e

    repo_root = os.path.dirname(os.path.abspath(__file__))

    condition = None
    target = None
    intermediate = None
    num_samples = 2
    args = list(sys.argv[1:])
    cleaned = []
    i = 0
    while i < len(args):
        if args[i] == "--condition" and i + 1 < len(args):
            condition = args[i + 1]
            i += 2
        elif args[i] == "--target" and i + 1 < len(args):
            target = args[i + 1]
            i += 2
        elif args[i] == "--intermediate" and i + 1 < len(args):
            intermediate = args[i + 1]
            i += 2
        elif args[i] == "--num_samples" and i + 1 < len(args):
            num_samples = int(args[i + 1])
            i += 2
        else:
            cleaned.append(args[i])
            i += 1

    if not condition or not target:
        _print_usage()
        sys.exit(1)

    # Load base config
    base_yaml = os.path.join(repo_root, "conf", "inference", "base.yaml")
    if not os.path.exists(base_yaml):
        raise FileNotFoundError(f"Base config not found: {base_yaml}")
    base_cfg = OmegaConf.load(base_yaml)

    # Apply key=value overrides
    overrides = [a for a in cleaned if "=" in a and not a.startswith("--")]
    cli_override_keys = {a.split("=", 1)[0] for a in overrides}
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        base_cfg = OmegaConf.merge(base_cfg, override_cfg)

    cfg = OmegaConf.to_container(base_cfg, resolve=True)

    # Inject condition/target/intermediate from CLI
    cfg["condition"] = [condition]
    cfg["target"] = target
    cfg["intermediate"] = intermediate
    cfg["num_samples"] = num_samples
    cfg["_cli_override_keys"] = cli_override_keys
    return cfg, repo_root


def _print_usage():
    print("Usage:")
    print("  python infer.py --condition <modality> --target <modality> [--intermediate <modality>] [key=value ...]")
    print()
    print("Examples:")
    print("  python infer.py --condition rgb --target depth checkpoint_path=/path/to/ckpt input_image=test.jpg")
    print("  python infer.py --condition rgb --target dinolocal checkpoint_path=/path/to/ckpt input_image=test.jpg")
    print("  python infer.py --condition caption --target image checkpoint_path=/path/to/ckpt prompt='a cat'")
    print("  python infer.py --condition caption --target image --intermediate depth checkpoint_path=/path/to/ckpt prompt='a house'")


def main():
    cfg, repo_root = _parse_args()

    # Add repo root to path for imports
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from any2any.any2any_tasks import (
        create_inferencer,
        run_inference_task,
    )
    from any2any.load_any2any import (
        load_any2any_model_hf,
        load_any2any_model_training_checkpoint,
    )

    # Resolve paths relative to repo root
    def resolve_path(p):
        if p is None:
            return None
        p = str(p)
        if p.startswith("/"):
            return p
        return os.path.join(repo_root, p)

    fmt = cfg["format"]

    # ── Load MODUS any2any model ──────────────────────────────────────────
    modality_config_path = resolve_path(cfg["modality_config"])

    if fmt == "hf":
        model, vae_model, tokenizer, new_token_ids, modality_registry = load_any2any_model_hf(
            model_path=resolve_path(cfg["model_path"]),
            model_name=cfg.get("model_name", "bagel_from_json"),
            init_on_gpu=cfg["init_on_gpu"],
            modality_config_path=modality_config_path,
        )
    elif fmt == "training":
        checkpoint_path = cfg.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError("checkpoint_path is required when format=training")
        model, vae_model, tokenizer, new_token_ids, modality_registry = load_any2any_model_training_checkpoint(
            checkpoint_path=resolve_path(checkpoint_path),
            model_path=resolve_path(cfg["model_path"]),
            model_name=cfg.get("model_name", "bagel_from_json"),
            init_on_gpu=cfg["init_on_gpu"],
            use_ema=cfg.get("use_ema", False),
            modality_config_path=modality_config_path,
        )
    else:
        raise ValueError(f"Unknown format: {fmt}")

    # ── Auto-derive settings from modality registry ───────────────────────
    condition = cfg["condition"]
    target = cfg["target"]
    intermediate = cfg.get("intermediate")

    # Auto-determine which external tokenizers to load
    use_dino = use_dinolocal = use_clip = use_imagebind = use_imagebindlocal = False
    all_modalities = list(condition) + [target] + ([intermediate] if intermediate else [])
    for mod_name in all_modalities:
        if mod_name and modality_registry.needs_external_tokenizer(mod_name):
            if mod_name == "dino":
                use_dino = True
            elif mod_name == "dinolocal":
                use_dinolocal = True
            elif mod_name == "clip":
                use_clip = True
            elif mod_name == "imagebind":
                use_imagebind = True
            elif mod_name == "imagebindlocal":
                use_imagebindlocal = True

    # Auto-determine cfg_img_scale from modality config, unless explicitly overridden
    cli_override_keys = cfg.get("_cli_override_keys", set())
    cfg_img_scale = cfg.get("cfg_img_scale", 2.0)
    if "cfg_img_scale" not in cli_override_keys:
        try:
            cfg_img_scale = modality_registry.resolve_cfg_img_scale(target, default=cfg_img_scale)
        except KeyError:
            pass

    # Auto-determine output directory
    output_dir = cfg.get("output_dir", "demo_outputs")
    if output_dir == "demo_outputs":
        chain_parts = [condition[0]]
        if intermediate and str(intermediate).lower() not in ("", "none"):
            chain_parts.append(intermediate)
        chain_parts.append(target)
        output_dir = os.path.join("demo_outputs", "2".join(chain_parts))
    output_dir = resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ── Create inferencer ─────────────────────────────────────────────────
    inferencer = create_inferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
        modality_registry=modality_registry,
        use_dino_tokenizer=use_dino,
        use_dinolocal_tokenizer=use_dinolocal,
        use_clip_tokenizer=use_clip,
        use_imagebind_tokenizer=use_imagebind,
        use_imagebindlocal_tokenizer=use_imagebindlocal,
    )

    # ── Run inference ─────────────────────────────────────────────────────
    prompt = cfg.get("prompt", "")
    raw_input_image = cfg.get("input_image")

    # Expand input_image: comma-separated list, directory of rgb_*.png,
    # or single path. Each image gets its own output subdir.
    image_list = []
    if raw_input_image:
        if "," in str(raw_input_image):
            image_list = [resolve_path(p.strip()) for p in str(raw_input_image).split(",") if p.strip()]
        else:
            resolved = resolve_path(raw_input_image)
            if resolved and os.path.isdir(resolved):
                import glob
                image_list = sorted(glob.glob(os.path.join(resolved, "**", "rgb_*.png"), recursive=True))
            else:
                image_list = [resolved]

    print(f"  condition:    {condition}")
    print(f"  target:       {target}")
    print(f"  intermediate: {intermediate}")
    print(f"  cfg_img_scale:{cfg_img_scale}")
    print(f"  output_dir:   {output_dir}")
    print(f"  num_samples:  {cfg.get('num_samples', 2)}")
    print(f"  num_images:   {len(image_list)}")
    print()

    for img_idx, img_path in enumerate(image_list or [None]):
        if img_path and len(image_list) > 1:
            img_stem = os.path.splitext(os.path.basename(img_path))[0]
            parent = os.path.basename(os.path.dirname(img_path))
            this_output = os.path.join(output_dir, f"{parent}__{img_stem}")
            os.makedirs(this_output, exist_ok=True)
            print(f"[{img_idx+1}/{len(image_list)}] image: {img_path}")
        else:
            this_output = output_dir

        run_inference_task(
            inferencer=inferencer,
            output_path=this_output,
            mode="generate",  # ignored; dispatch is driven by target modality
            prompt=prompt,
            input_image=img_path,
            condition=condition,
            target=target,
            intermediate=intermediate,
            cfg_text_scale=cfg["cfg_text_scale"],
            cfg_img_scale=cfg_img_scale,
            num_timesteps=cfg["num_timesteps"],
            seed=cfg["seed"],
            use_instruction=cfg["use_instruction"],
            use_target_instruction=cfg["use_target_instruction"],
            use_condition_instruction=cfg["use_condition_instruction"],
            use_intermediate_instruction=cfg.get("use_intermediate_instruction", True),
            do_modality_norm=cfg.get("do_modality_norm", False),
            use_det_image=cfg.get("use_det_image", False),
            image_size=cfg.get("image_size", 1024),
            do_sample=cfg.get("do_sample", False),
            temperature=cfg.get("temperature", 0.003),
            text_temperature=cfg.get("text_temperature", 0.95),
            top_k=cfg.get("top_k", 0),
            top_p=cfg.get("top_p", 1.0),
            num_samples=cfg.get("num_samples", 2),
            use_gt_dino_condition=cfg.get("use_gt_dino_condition", False),
            use_gt_dinolocal_condition=cfg.get("use_gt_dinolocal_condition", False),
        )

    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
