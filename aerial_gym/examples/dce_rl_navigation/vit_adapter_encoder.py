"""
vit_adapter_encoder.py — Inference wrapper for a TorchScript-exported ViT+adapter pipeline.

Mirrors the encode() interface of VAEImageEncoder so it can be used as a drop-in
replacement for image-latent encoding in NavigationTask subclasses.

The TorchScript model is produced by sampl_geometic_head/export/export_vit_adapter.py.
It expects raw float [0, 1] RGB images; ImageNet normalisation is baked into the model.

No resize is needed at inference time: the sim camera (lmf2_with_rgbd_camera)
outputs H=240 x W=320 images, which matches the training resolution of the adapter.
"""

import json
import os

import torch
import torch.nn.functional as F
from torch import Tensor


class ViTAdapterEncoder:
    """
    Loads a TorchScript-exported ViT+adapter pipeline and exposes an
    encode() method compatible with VAEImageEncoder.

    Args:
        model_path:    Path to the .pt TorchScript file
                       (e.g. vit_adapter_pipeline_240x320.pt).
        metadata_path: Path to the accompanying metadata.json.
        device:        Torch device string (e.g. "cuda:0").
    """

    def __init__(self, model_path: str, metadata_path: str, device: str = "cuda:0"):
        self.device = torch.device(device)

        print(f"[ViTAdapterEncoder] Loading model: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ViT adapter model not found: {model_path}")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        print(f"[ViTAdapterEncoder] Loading metadata: {metadata_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"ViT adapter metadata not found: {metadata_path}")
        with open(metadata_path) as f:
            meta = json.load(f)

        self.latent_dim    = int(meta["latent_dim"])
        self.target_height = int(meta["image_height"])   # expected H (240)
        self.target_width  = int(meta["image_width"])    # expected W (320)

        print(
            f"[ViTAdapterEncoder] Ready | "
            f"latent_dim={self.latent_dim} | "
            f"expected input HxW={self.target_height}x{self.target_width} | "
            f"device={self.device}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, rgb_images: Tensor) -> Tensor:
        """
        Encode a batch of RGB images to the 64-dim latent space.

        Args:
            rgb_images: float [0, 1] tensor in one of these layouts:
                        - (B, 1, H, W, 3)  — raw from obs_dict["rgb_pixels"]
                        - (B, H, W, 3)     — already stripped of sensor dim

        Returns:
            (B, latent_dim) float tensor on self.device.
        """
        with torch.no_grad():
            rgb_images = rgb_images.to(self.device)

            # Strip the extra sensor dimension if present: (B,1,H,W,3) → (B,H,W,3)
            if rgb_images.ndim == 5:
                rgb_images = rgb_images[:, 0]

            # (B, H, W, 3) → (B, 3, H, W)
            x = rgb_images.permute(0, 3, 1, 2).contiguous()

            # Resize if the sim camera resolution changed from the training resolution.
            # Under normal circumstances (sim at 240x320, training at 240x320) this
            # is a no-op, but it guards against mis-matched configs.
            if x.shape[2] != self.target_height or x.shape[3] != self.target_width:
                x = F.interpolate(
                    x,
                    size=(self.target_height, self.target_width),
                    mode="bilinear",
                    align_corners=False,
                )

            return self.model(x)   # (B, latent_dim)

    def get_latent_dims_size(self) -> int:
        """Matches the VAEImageEncoder API used in NavigationTask."""
        return self.latent_dim
