"""
dce_vit_comparison_task.py — Dual-pipeline navigation task for DCE vs ViT comparison.

Runs two parallel drones in the same Matterport scene with the same goal:
  - Env 0 (VAE_ENV_ID=0): depth image → frozen VAE  → 64-dim latent  (DCE pipeline)
  - Env 1 (VIT_ENV_ID=1): RGB  image → ViT+adapter  → 64-dim latent  (ViT pipeline)

Both drones use the same RL policy weights and the same 81-dim observation format.
Their goals are kept in sync after every reset so the comparison is fair.

The task inherits DCE_RL_Navigation_Task, keeping all reward shaping, curriculum,
navmesh goal sampling, and action transformation logic unchanged.
"""

import torch
import torch.nn.functional as F

from aerial_gym.examples.dce_rl_navigation.dce_navigation_task import DCE_RL_Navigation_Task
from aerial_gym.examples.dce_rl_navigation.vit_adapter_encoder import ViTAdapterEncoder
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)


# ---------------------------------------------------------------------------
# Comparison task configuration
# ---------------------------------------------------------------------------

class ComparisonTaskConfig:
    """
    Mixin / base override that sets num_envs=2.
    Combine with MatterportDCEEvalTaskConfig:

        class MyComparisonCfg(ComparisonTaskConfig, MatterportDCEEvalTaskConfig):
            pass
    """
    num_envs = 2


# ---------------------------------------------------------------------------
# Dual-pipeline task
# ---------------------------------------------------------------------------

class DCEViTComparisonTask(DCE_RL_Navigation_Task):
    """
    Navigation task that runs two robots side-by-side, encoding images with
    different pipelines:
      - Env 0 (VAE_ENV_ID): depth → VAE encoder  → image_latents[0]
      - Env 1 (VIT_ENV_ID): RGB  → ViT+adapter   → image_latents[1]

    Both environments navigate to the same goal (goals are synchronised on
    every reset so comparisons remain fair).

    Args:
        task_config:       Task configuration with num_envs=2.
        vit_model_path:    Path to the TorchScript vit_adapter_pipeline_HxW.pt file.
        vit_metadata_path: Path to the accompanying metadata.json.
    """

    VAE_ENV_ID: int = 0
    VIT_ENV_ID: int = 1

    def __init__(self, task_config, **kwargs):
        # Enforce num_envs=2 so the parent's cap-at-16 logic keeps it at 2.
        task_config.num_envs = 2
        super().__init__(task_config=task_config, **kwargs)

        # ViT model paths are stored on the task_config so they can be set
        # before task registration without modifying the task_registry API.
        vit_model_path    = getattr(task_config, "vit_model_path",    None)
        vit_metadata_path = getattr(task_config, "vit_metadata_path", None)

        if not vit_model_path or not vit_metadata_path:
            raise ValueError(
                "DCEViTComparisonTask requires task_config.vit_model_path and "
                "task_config.vit_metadata_path to be set before task creation."
            )

        logger.warning(
            "DCEViTComparisonTask: env %d uses VAE (depth), env %d uses ViT+adapter (RGB).",
            self.VAE_ENV_ID,
            self.VIT_ENV_ID,
        )

        self.vit_encoder = ViTAdapterEncoder(
            model_path=vit_model_path,
            metadata_path=vit_metadata_path,
            device=str(self.device),
        )

    # ------------------------------------------------------------------
    # Per-env image encoding (overrides NavigationTask)
    # ------------------------------------------------------------------

    def process_image_observation(self) -> None:
        """
        Route each environment's image through its respective encoder.

        Env 0 (VAE):  depth_range_pixels → VAE.encode()  → image_latents[0]
        Env 1 (ViT):  rgb_pixels         → ViT.encode()  → image_latents[1]

        Both encoders respect the encode_every_n_steps cadence from vae_config.
        """
        encode_every = max(
            1,
            int(getattr(self.task_config.vae_config, "encode_every_n_steps", 1)),
        )
        if (self.num_task_steps - 1) % encode_every != 0:
            return   # reuse previous latents on skipped steps

        # --- Env 0: depth → VAE ---
        depth_obs = self.obs_dict["depth_range_pixels"][[self.VAE_ENV_ID]].squeeze(1)
        # The frozen VAE expects depth at 135×240; apply min-pool if the camera
        # delivers a different resolution (shaded RGBD camera gives 240×320).
        if depth_obs.shape[-2:] != (135, 240):
            depth_obs = -F.adaptive_max_pool2d(
                -depth_obs.unsqueeze(1), (135, 240)
            ).squeeze(1)
        self.image_latents[[self.VAE_ENV_ID]] = self.vae_model.encode(depth_obs)

        # --- Env 1: RGB → ViT+adapter ---
        rgb_obs = self.obs_dict["rgb_pixels"][[self.VIT_ENV_ID]]  # (1, 1, H, W, 3)
        self.image_latents[[self.VIT_ENV_ID]] = self.vit_encoder.encode(rgb_obs)

    # ------------------------------------------------------------------
    # Goal synchronisation — both drones always target the same point
    # ------------------------------------------------------------------

    def _sync_goals(self) -> None:
        """Copy env 0's goal to env 1 so both drones navigate to the same target."""
        self.target_position[self.VIT_ENV_ID] = self.target_position[self.VAE_ENV_ID]

    def reset(self):
        result = super().reset()
        self._sync_goals()
        return result

    def reset_idx(self, env_ids) -> None:
        super().reset_idx(env_ids)
        # Synchronise after every partial or full reset.  Copying env 0's goal to
        # env 1 is always safe: if env 0 also just reset, it has a fresh goal; if
        # only env 1 reset, it inherits env 0's current goal.  Either way the
        # comparison objective is consistent.
        self._sync_goals()
