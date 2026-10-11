"""
HunyuanImage-3.0 VAE adapter for MODUS training pipeline.

Two classes:
  - HunyuanVAEConfig: duck-types AutoEncoderParams so build_train_dataloader
    can read .downsample and .z_channels.
  - HunyuanVAEAdapter: exposes .encode() returning a plain float tensor,
    matching BAGEL's AutoEncoder.encode() signature.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
from torch import Tensor


class HunyuanVAEConfig:
    """
    Duck-types AutoEncoderParams so that build_train_dataloader can read
    vae_config.downsample and vae_config.z_channels.

    Values come from the HunyuanImage-3.0 VAE config:
      ffactor_spatial = 16 (spatial downsampling)
      latent_channels = 32
      scaling_factor  = 0.562679178327931
    """
    downsample: int = 16
    z_channels: int = 32
    scale_factor: float = 0.562679178327931


class HunyuanVAEAdapter(nn.Module):
    """
    Thin wrapper around AutoencoderKLConv3D that exposes the same
    ``encode(x) -> Tensor`` interface as BAGEL's AutoEncoder.

    The underlying VAE returns an AutoencoderKLOutput with a
    DiagonalGaussianDistribution; we sample from it, squeeze the
    temporal dimension (T=1 for images), and apply the scaling factor.

    Input:
        x: (B, 3, H, W)  – padded batch of images (uint8 pixels normalised to [-1, 1]
                            or whatever the dataset pipeline provides)

    Output:
        z: (B, 32, H/16, W/16)  – scaled VAE latents
    """

    def __init__(self, vae: nn.Module, scale_factor: float = 0.562679178327931):
        super().__init__()
        self.vae = vae
        self.scale_factor = scale_factor
        self.source_model_path: str | None = None

    def _has_meta_params(self) -> bool:
        return any(getattr(p, "is_meta", False) for p in self.vae.parameters())

    def _reload_vae_from_source(self, *, device, dtype) -> None:
        if not self.source_model_path:
            raise RuntimeError("VAE has meta parameters but source_model_path is not set for reload.")

        from modeling.hunyuan_image_3.src.autoencoder_kl_3d import AutoencoderKLConv3D
        from safetensors.torch import load_file

        cfg_path = os.path.join(self.source_model_path, "config.json")
        with open(cfg_path, "r") as f:
            full_cfg = json.load(f)
        vae_cfg_dict = {k: v for k, v in full_cfg.get("vae", {}).items() if not k.startswith("_")}
        vae = AutoencoderKLConv3D(**vae_cfg_dict)

        index_path = os.path.join(self.source_model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                weight_map = json.load(f).get("weight_map", {})
            shard_paths = [
                os.path.join(self.source_model_path, s)
                for s in sorted({v for k, v in weight_map.items() if k.startswith("vae.")})
            ]
        else:
            shard_paths = []

        prefix = "vae."
        vae_state = {}
        for shard_path in shard_paths:
            shard = load_file(shard_path, device="cpu")
            for k, v in shard.items():
                if k.startswith(prefix):
                    vae_state[k[len(prefix):]] = v
            del shard

        if not vae_state:
            raise RuntimeError(f"No VAE weights found under {self.source_model_path}")

        vae.load_state_dict(vae_state, strict=False)
        vae = vae.eval().to(device=device, dtype=dtype)
        for p in vae.parameters():
            p.requires_grad = False
        self.vae = vae

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode a batch of images to latents.

        Args:
            x: (B, 3, H, W) float tensor

        Returns:
            z: (B, 32, H//16, W//16) float tensor
        """
        with torch.no_grad():
            if self._has_meta_params():
                self._reload_vae_from_source(device=x.device, dtype=x.dtype)
            out = self.vae.encode(x)           # AutoencoderKLOutput
            z = out.latent_dist.sample()        # (B, 32, 1, H/16, W/16)
            z = z.squeeze(2) * self.scale_factor  # (B, 32, H/16, W/16)
        return z

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latents back to images. Inverse of encode().

        Args:
            z: (B, 32, H//16, W//16) float tensor — same shape encode() returns.

        Returns:
            x: (B, 3, H, W) float tensor in approximately [-1, 1] range
               (inferencer applies the *0.5+0.5 remap to [0,1]).
        """
        with torch.no_grad():
            if self._has_meta_params():
                self._reload_vae_from_source(device=z.device, dtype=z.dtype)
            # Undo encode's scaling + T-squeeze: (B, C, H, W) → (B, C, 1, H, W).
            z = (z / self.scale_factor).unsqueeze(2)
            out = self.vae.decode(z)           # DecoderOutput
            x = out.sample                      # (B, 3, 1, H, W)
            x = x.squeeze(2)                    # (B, 3, H, W)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.encode(x)
