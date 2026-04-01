"""
matterport_dce_task_config.py — Task configurations for DCE RL navigation in Matterport scenes.

Three config classes, each building on the previous:

  MatterportVAETaskConfig   — VAE depth pipeline, lmf2 robot (depth-only, memory efficient).
                              Matches the dce_nn_navigation.py training pipeline exactly.
  MatterportViTTaskConfig   — ViT+adapter RGB pipeline, lmf2_rgb_only robot (no depth tensor).
  MatterportComparisonTaskConfig — Two-env comparison (env0=VAE, env1=ViT), RGBD robot.
"""

import torch
from aerial_gym.config.task_config.navigation_task_config import task_config as _BaseCfg


class MatterportVAETaskConfig(_BaseCfg):
    """
    Matterport scene navigation with the VAE depth pipeline.

    Robot: lmf2 (BaseDepthCameraConfig, 135x240, max_range=10m, normalized=[0,1]).
    This is memory-efficient (no RGB tensor) and matches the training pipeline used in
    dce_nn_navigation.py / DCE_RL_Navigation_Task exactly:

      Camera output → 135x240 depth (no resize needed)
      VAEImageEncoder.encode():
        unsqueeze(1)              (N,135,240) → (N,1,135,240)
        interpolate nearest       → (N,1,270,480)  [vae_config.image_res]
        VAE encode                → (N,64) sampled latent
      Observation: 81-dim [state(17) | latent(64)]
    """

    seed            = 42
    sim_name        = "base_sim"
    env_name        = "matterport_glb_env"
    robot_name      = "lmf2"                   # depth-only: no RGB tensor overhead
    controller_name = "lmf2_velocity_control"
    args            = {}
    num_envs        = 32
    use_warp        = True
    headless        = True
    device          = "cuda:0"

    observation_space_dim            = 81       # 17 state + 64 image latent
    privileged_observation_space_dim = 0
    action_space_dim                 = 3        # set by MatterportDCENavigationTask.__init__
    episode_len_steps                = 1000
    return_state_before_reset        = False

    target_min_ratio = [0.10, 0.10, 0.10]
    target_max_ratio = [0.90, 0.90, 0.90]

    dce_pipeline_type = "vae"

    # Execution cadence (steps = integer multiples of the physics dt).
    # Set these to emulate real onboard frequencies relative to the simulator timestep.
    # e.g. dt=0.01 s → perception_every_n_steps=2 → ~50 Hz perception
    #                 → policy_every_n_steps=2    → ~50 Hz policy
    perception_every_n_steps = 4   # run image encoder every N task steps
    policy_every_n_steps     = 4   # run RL policy every N task steps

    class navmesh_sampling:
        enable                     = True
        edge_padding               = 1.5
        spawn_height_offset_range  = [0.8, 1.8]
        goal_height_offset_range   = [0.8, 1.8]
        goal_min_separation        = 2.0
        goal_max_separation        = 10.0
        max_pair_sampling_attempts = 30

    class vae_config(_BaseCfg.vae_config):
        pass   # inherits use_vae=True, latent_dims=64, image_res=(270,480), model paths, etc.

    class curriculum(_BaseCfg.curriculum):
        min_level = 36             # matches DCE_RL_Navigation_Task standard

    @staticmethod
    def action_transformation_function(action):
        """3-dim DCE policy output → 4-dim velocity command (body frame)."""
        c = torch.clamp(action, -1.0, 1.0)
        c[:, 0] += 1.0             # shift forward component from [-1,1] to [0,2]
        out = torch.zeros((c.shape[0], 4), device=action.device, requires_grad=False)
        tilt = torch.pi / 4
        spd  = 2.0
        yr   = torch.pi / 3
        out[:, 0] = c[:, 0] * torch.cos(tilt * c[:, 1]) * spd / 2.0
        out[:, 1] = 0.0
        out[:, 2] = c[:, 0] * torch.sin(tilt * c[:, 1]) * spd / 2.0
        out[:, 3] = c[:, 2] * yr
        return out


class MatterportViTTaskConfig(MatterportVAETaskConfig):
    """
    Matterport scene navigation with the ViT+adapter RGB pipeline.

    Robot: lmf2_rgb_only (RGBOnlyCameraConfig, 240x320, calculate_depth=False).
    Only rgb_pixels are rendered; depth_range_pixels tensor is not populated.

    vit_config.model_path and vit_config.metadata_path must be set before task creation:
        MatterportViTTaskConfig.vit_config.model_path    = "/path/to/model.pt"
        MatterportViTTaskConfig.vit_config.metadata_path = "/path/to/metadata.json"
    """

    robot_name        = "lmf2_rgb_only"
    dce_pipeline_type = "vit"

    class vae_config(MatterportVAETaskConfig.vae_config):
        use_vae = False            # suppress VAE weight loading; image_latents still allocated

    class vit_config:
        model_path    = ""         # set at runtime
        metadata_path = ""         # set at runtime


class MatterportComparisonTaskConfig(MatterportVAETaskConfig):
    """
    Two-env side-by-side comparison: env 0 uses VAE (depth), env 1 uses ViT (RGB).

    Robot: lmf2_with_shaded_rgbd_camera provides both depth and RGB channels,
    which are needed simultaneously for the comparison.

    vit_config.model_path and vit_config.metadata_path must be set before task creation.
    """

    robot_name        = "lmf2_with_shaded_rgbd_camera"
    num_envs          = 2
    headless          = False
    dce_pipeline_type = "comparison"

    class vit_config:
        model_path    = ""         # set at runtime
        metadata_path = ""         # set at runtime
