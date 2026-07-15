#!/usr/bin/env bash
#
# Universal inference script — one script for ALL modality pairs.
#
# No per-task config YAML needed. Just specify condition, target,
# and (optionally) intermediate modality.
#
# ─── Basic usage ───────────────────────────────────────────────────────────────
#
#   bash scripts/inference.sh <checkpoint_path> --condition <mod> --target <mod> [options]
#
# ─── Examples ──────────────────────────────────────────────────────────────────
#
#   # RGB → Depth
#   bash scripts/inference.sh /path/to/ckpt --condition rgb --target depth \
#       input_image=test_images/01_basil_cathedral.jpg
#
#   # RGB → DINO-local
#   bash scripts/inference.sh /path/to/ckpt --condition rgb --target dinolocal \
#       input_image=test_images/01_basil_cathedral.jpg
#
#   # RGB → Segmentation
#   bash scripts/inference.sh /path/to/ckpt --condition rgb --target seg \
#       input_image=test_images/01_basil_cathedral.jpg
#
#   # RGB → Surface normals
#   bash scripts/inference.sh /path/to/ckpt --condition rgb --target normal \
#       input_image=test_images/01_basil_cathedral.jpg
#
#   # RGB → Detection
#   bash scripts/inference.sh /path/to/ckpt --condition rgb --target det \
#       input_image=test_images/01_basil_cathedral.jpg prompt="detect the chair"
#
#   # Text → Image
#   bash scripts/inference.sh /path/to/ckpt --condition caption --target image \
#       prompt="a cat sitting on a mat"
#
#   # Chained: Text → Depth → Image
#   bash scripts/inference.sh /path/to/ckpt --condition caption --target image \
#       --intermediate depth prompt="a house by the lake"
#
#   # Override hyperparams:
#   bash scripts/inference.sh /path/to/ckpt --condition rgb --target dinolocal \
#       input_image=test_images/01_basil_cathedral.jpg cfg_text_scale=6.0 num_timesteps=100
#
set -euo pipefail

checkpoint_path="${1:-}"
shift 1 2>/dev/null || true

if [[ -z "${checkpoint_path}" ]]; then
  echo "ERROR: checkpoint_path is required as first argument"
  echo ""
  echo "Usage: bash scripts/inference.sh <checkpoint_path> --condition <mod> --target <mod> [options]"
  echo ""
  echo "Examples:"
  echo "  bash scripts/inference.sh /path/to/ckpt --condition rgb --target depth input_image=test_images/01_basil_cathedral.jpg"
  echo "  bash scripts/inference.sh /path/to/ckpt --condition rgb --target dinolocal input_image=test_images/01_basil_cathedral.jpg"
  echo "  bash scripts/inference.sh /path/to/ckpt --condition caption --target image prompt='a cat'"
  exit 1
fi

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)"

python infer.py \
  checkpoint_path="${checkpoint_path}" \
  "$@"

