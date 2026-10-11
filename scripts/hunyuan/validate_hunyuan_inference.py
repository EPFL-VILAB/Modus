#!/usr/bin/env python3
"""Deterministic smoke/parity probe for a raw or exported Hunyuan checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from any2any.hunyuan_runtime import configure_hunyuan_runtime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("training", "hf"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--modality-config", required=True)
    parser.add_argument("--input-image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--targets", default="depth,caption,dinolocal")
    parser.add_argument("--rope-modes", default="0,1")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    runtime = configure_hunyuan_runtime()
    from any2any.hunyuan_loader import (
        load_any2any_model_hf,
        load_any2any_model_training_checkpoint,
    )
    from any2any.hunyuan_tasks import create_inferencer, run_inference_task

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.format == "training":
        if not args.base_model:
            raise ValueError("--base-model is required for format=training")
        components = load_any2any_model_training_checkpoint(
            checkpoint_path=args.checkpoint,
            model_path=args.base_model,
            model_name="hunyuan_image_3",
            init_on_gpu=True,
            use_ema=args.use_ema,
            modality_config_path=args.modality_config,
            train_config_path="conf/inference/hunyuan_model.yaml",
        )
    else:
        components = load_any2any_model_hf(
            model_path=args.checkpoint,
            model_name="hunyuan_image_3",
            init_on_gpu=True,
            modality_config_path=args.modality_config,
        )
    model, vae, tokenizer, new_ids, registry = components
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    inferencer = create_inferencer(
        model,
        vae,
        tokenizer,
        new_ids,
        modality_registry=registry,
        use_dinolocal_tokenizer="dinolocal" in targets,
    )
    rope_modes = [item.strip() for item in args.rope_modes.split(",") if item.strip()]
    if any(item not in {"0", "1"} for item in rope_modes):
        raise ValueError("--rope-modes accepts only comma-separated 0 and 1")
    for rope_mode in rope_modes:
        os.environ["HUNYUAN_INFER_IMAGE_ROPE_BLOCK"] = rope_mode
        for index, target in enumerate(targets):
            target_output = output / f"rope{rope_mode}" / target
            target_output.mkdir(parents=True, exist_ok=True)
            run_inference_task(
                inferencer=inferencer,
                output_path=str(target_output),
                input_image=args.input_image,
                condition=("rgb",),
                target=target,
                cfg_text_scale=4.0,
                cfg_img_scale=1.0,
                num_timesteps=20,
                seed=20261004 + index,
                use_instruction=True,
                use_target_instruction=True,
                use_condition_instruction=False,
                image_size=512,
                num_samples=1,
            )

    files = {
        str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "report.json"
    }
    report = {
        "result": "PASS",
        "format": args.format,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "targets": targets,
        "rope_modes": rope_modes,
        "runtime": runtime,
        "files": files,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
