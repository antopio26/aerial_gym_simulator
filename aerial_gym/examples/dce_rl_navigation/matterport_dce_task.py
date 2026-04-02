"""
matterport_dce_task.py — DCE navigation tasks for Matterport scenes.

Provides two reusable pipeline classes and two task classes:

  VAEPipeline               — depth → VAE → 64-dim latent
  ViTPipeline               — RGB  → ViT+adapter → 64-dim latent

  MatterportDCENavigationTask  — NavigationTask with pluggable pipeline, no env cap.
                                  Supports both "vae" and "vit" via task_config.dce_pipeline_type.
  MatterportComparisonTask     — Two-env variant: env0=VAE, env1=ViT, goals kept in sync.

Both task classes use the DCE observation format (81-dim) defined in DCE_RL_Navigation_Task
and are compatible with the same Sample Factory policy checkpoint.

VAE DEPTH PIPELINE — provenance and equivalence with dce_nn_navigation.py:
  Training used DCE_RL_Navigation_Task with lmf2 (BaseDepthCameraConfig):
    Camera: 135x240, max_range=10m, normalize_range=True → depth in [0, 1]
    NavigationTask.process_image_observation():
      min-pool to (135, 240) — no-op since camera is already 135x240
    VAEImageEncoder.encode():
      unsqueeze(1)        (N,135,240) → (N,1,135,240)
      interpolate nearest              → (N,1,270,480)  [vae_config.image_res]
      VAE encode                       → (N,64) sampled latent
  Eval uses lmf2_with_rgbd_camera (RGBDCameraConfig):
    Camera: 240x320, depth_max_range=10m — same normalization
    VAEPipeline.encode(): min-pool (240,320)→(135,240), then same VAEImageEncoder path
  Both cameras use max_range=10m → identical normalized depth values → pipelines ARE equivalent.
"""

import torch
import torch.nn.functional as F

from aerial_gym.examples.dce_rl_navigation.vit_adapter_encoder import ViTAdapterEncoder
from aerial_gym.task.navigation_task.navigation_task import NavigationTask
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import get_euler_xyz_tensor, quat_rotate_inverse

logger = CustomLogger(__name__)


# ---------------------------------------------------------------------------
# TorchScript helper
# ---------------------------------------------------------------------------

@torch.jit.script
def ssa(a: torch.Tensor) -> torch.Tensor:
    """Smallest signed angle — wraps to (−π, π]."""
    return torch.remainder(a + torch.pi, 2 * torch.pi) - torch.pi


# ---------------------------------------------------------------------------
# Pipeline classes
# ---------------------------------------------------------------------------

class VAEPipeline:
    """Wraps an already-loaded VAEImageEncoder for use in MatterportDCENavigationTask.

    The encoder is passed in (not created here) so the weights are not loaded twice —
    NavigationTask.__init__ already loads them when vae_config.use_vae=True.
    """

    def __init__(self, vae_encoder):
        self.encoder = vae_encoder

    def encode(self, obs_dict: dict, env_ids=None) -> torch.Tensor:
        """
        Encode depth images to 64-dim latents.

        Args:
            obs_dict:  Task observation dict; must contain "depth_range_pixels" (N,1,H,W).
            env_ids:   Optional LongTensor — slice to these envs before encoding.

        Returns:
            (N, 64) or (len(env_ids), 64) float tensor.
        """
        depth = obs_dict["depth_range_pixels"].squeeze(1)   # (N, H, W)
        if env_ids is not None:
            depth = depth[env_ids]
        # Min-pool to the VAE's expected input size if the camera resolution differs.
        # lmf2 natively outputs 135x240 (no-op); shaded-RGBD outputs 240x320 and needs pooling.
        if depth.shape[-2:] != (135, 240):
            depth = -F.adaptive_max_pool2d(-depth.unsqueeze(1), (135, 240)).squeeze(1)
        return self.encoder.encode(depth)   # (N, 64)


class ViTPipeline:
    """Wraps ViTAdapterEncoder for use in MatterportDCENavigationTask."""

    def __init__(self, vit_config, device: str):
        self.encoder = ViTAdapterEncoder(
            model_path=vit_config.model_path,
            metadata_path=vit_config.metadata_path,
            device=device,
        )

    def encode(self, obs_dict: dict, env_ids=None) -> torch.Tensor:
        """
        Encode RGB images to 64-dim latents.

        Args:
            obs_dict:  Task observation dict; must contain "rgb_pixels" (N,1,H,W,3).
            env_ids:   Optional LongTensor — slice to these envs before encoding.

        Returns:
            (N, 64) or (len(env_ids), 64) float tensor.
        """
        rgb = obs_dict["rgb_pixels"]        # (N, 1, H, W, 3)
        if env_ids is not None:
            rgb = rgb[env_ids]
        return self.encoder.encode(rgb)     # (N, 64)


# ---------------------------------------------------------------------------
# Task classes
# ---------------------------------------------------------------------------

class MatterportDCENavigationTask(NavigationTask):
    """
    DCE RL navigation task for Matterport scenes.

    Differences from DCE_RL_Navigation_Task:
      - No 16-env cap — supports arbitrary num_envs for parallel training.
      - Pluggable image pipeline: "vae" or "vit" selected via task_config.dce_pipeline_type.
      - process_image_observation() routes through self._pipeline instead of calling
        self.vae_model directly, keeping the interface clean regardless of pipeline.

    Compatible with the same 81-dim observation format and Sample Factory policy weights
    as DCE_RL_Navigation_Task.
    """

    def __init__(self, task_config, **kwargs):
        # Set action space before super().__init__ builds gymnasium spaces.
        task_config.action_space_dim = 3

        super().__init__(task_config=task_config, **kwargs)
        # After super(): self.device, self.vae_model, self.image_latents all exist.

        # Initialize the goal sampler and disable navmesh sampling by default since not all envs support it.
        self.navmesh_goal_sampling_enabled = False
        self.goal_navmesh_sampler = None
        self._setup_navmesh_goal_sampling()

        pipeline_type = getattr(task_config, "dce_pipeline_type", "vae")
        if pipeline_type == "vit":
            self._pipeline = ViTPipeline(task_config.vit_config, str(self.device))
        else:
            # "vae" (or "comparison" — comparison subclass builds ViT separately).
            # Reuse the VAEImageEncoder already loaded by NavigationTask to avoid
            # loading the ~44 MB weights twice.
            self._pipeline = VAEPipeline(self.vae_model)

        # Execution cadence — read from task_config, fall back to vae_config for compat.
        self._perception_every_n = int(
            getattr(task_config, "perception_every_n_steps",
                    getattr(task_config.vae_config, "encode_every_n_steps", 1))
        )
        self._policy_every_n = int(getattr(task_config, "policy_every_n_steps", 1))

        # Log frequencies relative to sim dt.
        sim_dt = float(self.sim_env.sim_config.sim.dt)
        logger.warning(
            "MatterportDCENavigationTask | pipeline=%-10s  num_envs=%d\n"
            "  sim dt          : %.4f s  (%.1f Hz)\n"
            "  perception step : every %d steps  → %.1f Hz\n"
            "  policy step     : every %d steps  → %.1f Hz",
            pipeline_type,
            self.num_envs,
            sim_dt, 1.0 / sim_dt,
            self._perception_every_n, 1.0 / (sim_dt * self._perception_every_n),
            self._policy_every_n,     1.0 / (sim_dt * self._policy_every_n),
        )

    def reset_idx(self, env_ids):
        # Convert env_ids to a LongTensor on the correct device and dtype if needed.
        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        if len(env_ids) == 0:
            self.infos = {}
            return

        # Navmesh goal sampling for reset: sample goals for these envs and update self.target_position
        if self.navmesh_goal_sampling_enabled:
            goal_world = self._sample_navmesh_goals(env_ids)
            if goal_world is not None:
                self.target_position[env_ids] = goal_world
                self.infos = {}
                return

        # If navmesh sampling is not enabled or fails, fall back to default reset behavior (e.g. random goal sampling).
        return super().reset_idx(env_ids)
    
    # ------------------------------------------------------------------
    # Navmesh goal sampling
    # ------------------------------------------------------------------

    def _setup_navmesh_goal_sampling(self):
        nav_cfg = getattr(self.task_config, "navmesh_sampling", None)
        if nav_cfg is None or not getattr(nav_cfg, "enable", False):
            return

        self.goal_navmesh_sampler = getattr(self.sim_env, "navmesh_sampler", None)
        if self.goal_navmesh_sampler is None or not self.goal_navmesh_sampler.enabled:
            logger.warning(
                "Task navmesh goal sampling enabled, but env navmesh sampler is not active. "
                "Enable navmesh_sampling in env config as well."
            )
            return

        self.navmesh_goal_sampling_enabled = True
        logger.info("Enabled task navmesh goal sampling using env-level navmesh sampler.")

        # Apply task-config navmesh_sampling overrides to the env-level sampler's nav_cfg.
        # The sampler was built from env_cfg, so without this its internal parameters
        # (edge_padding, spawn_height_offset_range, max_bound_resample_rounds, …) come
        # from the env config and the task config values are silently ignored.
        _SAMPLER_FIELDS = (
            "edge_padding",
            "spawn_height_offset_range",
            "max_bound_resample_rounds",
            "enforce_env_bounds",
            "zero_velocity_on_spawn",
        )
        for field in _SAMPLER_FIELDS:
            if hasattr(nav_cfg, field):
                setattr(self.goal_navmesh_sampler.nav_cfg, field, getattr(nav_cfg, field))
                logger.info(
                    "Navmesh sampler override: %s = %s (from task config)",
                    field, getattr(nav_cfg, field),
                )

    def _sample_navmesh_goals(self, env_ids):
        if self.goal_navmesh_sampler is None:
            return None

        env_ids = env_ids.to(device=self.device, dtype=torch.long)

        nav_cfg = self.task_config.navmesh_sampling
        goal_h = tuple(getattr(nav_cfg, "goal_height_offset_range", [0.0, 0.0]))
        goals = self.goal_navmesh_sampler.sample_world_points(
            env_ids=env_ids,
            height_offset_range=goal_h,
        )
        if goals is None:
            return None

        min_sep = float(getattr(nav_cfg, "goal_min_separation", 0.0) or 0.0)
        max_sep_cfg = getattr(nav_cfg, "goal_max_separation", None)
        max_sep = float(max_sep_cfg) if max_sep_cfg is not None else None
        max_attempts = int(getattr(nav_cfg, "max_pair_sampling_attempts", 4))

        planar_axes = (0, 1)
        navmesh_obj = getattr(self.goal_navmesh_sampler, "navmesh", None)
        if navmesh_obj is not None:
            navmesh_planar_axes = getattr(navmesh_obj, "planar_axes", None)
            if navmesh_planar_axes is not None and len(navmesh_planar_axes) == 2:
                planar_axes = (int(navmesh_planar_axes[0]), int(navmesh_planar_axes[1]))

        if min_sep > 0.0 or max_sep is not None:
            robot_pos = self.obs_dict["robot_position"][env_ids]
            for _ in range(max_attempts):
                planar_dist = torch.norm(
                    goals[:, [planar_axes[0], planar_axes[1]]]
                    - robot_pos[:, [planar_axes[0], planar_axes[1]]],
                    dim=1,
                )
                invalid = planar_dist < min_sep
                if max_sep is not None:
                    invalid = torch.logical_or(invalid, planar_dist > max_sep)
                if not torch.any(invalid):
                    break
                if invalid.device != env_ids.device:
                    invalid = invalid.to(env_ids.device)
                invalid_env_ids = env_ids[invalid]
                resampled = self.goal_navmesh_sampler.sample_world_points(
                    env_ids=invalid_env_ids,
                    height_offset_range=goal_h,
                )
                if resampled is None:
                    break
                goals[invalid] = resampled

        return goals

    # ------------------------------------------------------------------
    # Step — action repeat for policy cadence
    # ------------------------------------------------------------------

    def step(self, actions):
        """Step the simulation, repeating actions for policy_every_n_steps sub-steps.

        When policy_every_n_steps=1 (default) this is a direct pass-through to the base
        class.  When >1, the action is held constant for N physics steps so that the
        effective policy frequency is 1/(dt * policy_every_n_steps) Hz.
        """
        if self._policy_every_n <= 1:
            return super().step(actions)
        # Sub-step loop: repeat action N times, accumulate returns from the last sub-step.
        for _ in range(self._policy_every_n):
            result = super().step(actions)
        return result

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    def process_image_observation(self) -> None:
        """Encode images with the active pipeline, respecting perception_every_n_steps."""
        if (self.num_task_steps - 1) % self._perception_every_n != 0:
            return   # reuse previous latents on skipped steps
        self.image_latents[:] = self._pipeline.encode(self.obs_dict)

    # ------------------------------------------------------------------
    # Observation assembly (DCE format — mirrors DCE_RL_Navigation_Task)
    # ------------------------------------------------------------------

    def process_obs_for_task(self) -> None:
        """
        Assemble the 81-dim DCE observation vector.

          [0:3]   unit vector to goal in vehicle frame
          [3]     distance to goal / 5.0
          [4:6]   euler roll, pitch
          [6]     0 (reserved)
          [7:10]  body linear velocity
          [10:13] body angular velocity
          [13:17] previous actions
          [17:81] image latents (64-dim)
        """
        vec_to_target = quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"],
            self.target_position - self.obs_dict["robot_position"],
        )
        dist_to_tgt = torch.norm(vec_to_target, dim=1)
        self.task_obs["observations"][:, 0:3]  = vec_to_target / dist_to_tgt.unsqueeze(1)
        self.task_obs["observations"][:, 3]    = dist_to_tgt / 5.0
        euler = ssa(get_euler_xyz_tensor(self.obs_dict["robot_vehicle_orientation"]))
        self.task_obs["observations"][:, 4:6]  = euler[:, 0:2]
        self.task_obs["observations"][:, 6]    = 0.0
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 10:13]= self.obs_dict["robot_body_angvel"]
        self.task_obs["observations"][:, 13:17]= self.obs_dict["robot_actions"]
        self.task_obs["observations"][:, 17:81]= self.image_latents


class MatterportComparisonTask(MatterportDCENavigationTask):
    """
    Two-env side-by-side comparison task.

      Env 0 (VAE_ENV_ID): depth image → VAEPipeline  → image_latents[0]
      Env 1 (VIT_ENV_ID): RGB  image → ViTPipeline   → image_latents[1]

    Both envs use the same RL policy weights and the same goal position so the
    comparison is fair.  Goals are synchronised after every reset.
    """

    VAE_ENV_ID: int = 0
    VIT_ENV_ID: int = 1

    def __init__(self, task_config, **kwargs):
        task_config.num_envs = 2
        # Let super() build the VAE pipeline for env 0.
        super().__init__(task_config=task_config, **kwargs)

        # Build ViT pipeline for env 1.
        self._vit_pipeline = ViTPipeline(task_config.vit_config, str(self.device))

        logger.warning(
            "MatterportComparisonTask | env %d: VAE (depth) | env %d: ViT (RGB)",
            self.VAE_ENV_ID,
            self.VIT_ENV_ID,
        )

    def process_image_observation(self) -> None:
        """Route env 0 through VAE and env 1 through ViT with shared encode cadence."""
        if (self.num_task_steps - 1) % self._perception_every_n != 0:
            return

        vae_ids = torch.tensor([self.VAE_ENV_ID], device=self.device, dtype=torch.long)
        vit_ids = torch.tensor([self.VIT_ENV_ID], device=self.device, dtype=torch.long)
        self.image_latents[vae_ids] = self._pipeline.encode(self.obs_dict, env_ids=vae_ids)
        self.image_latents[vit_ids] = self._vit_pipeline.encode(self.obs_dict, env_ids=vit_ids)

    def _sync_goals(self) -> None:
        """Copy env 0's goal to env 1 so both drones navigate to the same target."""
        self.target_position[self.VIT_ENV_ID] = self.target_position[self.VAE_ENV_ID]

    def reset(self):
        result = super().reset()
        self._sync_goals()
        return result

    def reset_idx(self, env_ids) -> None:
        super().reset_idx(env_ids)
        self._sync_goals()
