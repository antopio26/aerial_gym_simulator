"""Multi-scene DCE navigation task.

Extends MatterportDCENavigationTask to support:
  - Loading N coexisting scenes simultaneously (via MultiSceneEnvManager).
  - Per-env spawn/goal sampling routed to the correct scene's navmesh.
  - Optional scene rotation (sync mode): after a threshold the simulation
    is torn down and rebuilt with a fresh set of scenes.
  - Disabling rotation entirely (scenes loaded once and kept forever).
"""

import torch

from aerial_gym.env_manager.multi_scene_env_manager import MultiSceneEnvManager
from aerial_gym.env_manager.scene_pool import ScenePool
from aerial_gym.examples.dce_rl_navigation.matterport_dce_task import (
    MatterportDCENavigationTask,
)
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)


class MultiSceneDCENavigationTask(MatterportDCENavigationTask):
    """DCE navigation across multiple coexisting Matterport scenes.

    The task operates in two modes depending on ``scene_rotation.enable``:

    **No rotation** (default):
        N scenes are loaded once; training proceeds indefinitely on them.

    **Sync rotation**:
        After ``switch_after_steps`` task steps (or ``switch_after_rollouts``
        completed episodes), the task waits for every env to finish its
        current episode, then destroys the simulation and rebuilds it with a
        new set of N scenes drawn from the scene pool.
    """

    def __init__(self, task_config, **kwargs):
        # --- Scene pool discovery ---
        pool_cfg = task_config.scene_rotation if hasattr(task_config, "scene_rotation") else None
        env_cfg = self._resolve_env_cfg(task_config)
        sp_cfg = getattr(env_cfg, "scene_pool", None)
        if sp_cfg is None:
            raise ValueError(
                "MultiSceneDCENavigationTask requires a scene_pool section in the env config."
            )
        self._scene_pool = ScenePool(sp_cfg.base_folder)
        self._num_coexisting = int(sp_cfg.num_coexisting_scenes)
        self._scene_spacing_margin = float(getattr(sp_cfg, "scene_spacing_margin", 50.0))

        # Pick initial scenes.
        self._active_scene_infos = self._scene_pool.sample(self._num_coexisting)
        self._active_scene_ids = [s.scene_id for s in self._active_scene_infos]

        # --- Rotation config ---
        self._rotation_enabled = False
        self._switch_after_steps = None
        self._switch_after_rollouts = None
        if pool_cfg is not None and getattr(pool_cfg, "enable", False):
            self._rotation_enabled = True
            self._switch_after_steps = getattr(pool_cfg, "switch_after_steps", None)
            self._switch_after_rollouts = getattr(pool_cfg, "switch_after_rollouts", None)
            if self._switch_after_steps is None and self._switch_after_rollouts is None:
                raise ValueError(
                    "scene_rotation.enable=True but neither switch_after_steps "
                    "nor switch_after_rollouts is set."
                )
        self._rollout_counter = 0
        self._waiting_for_drain = False

        # Store env_cfg for rebuilds.
        self._env_cfg_class = env_cfg

        # super().__init__ will call _build_sim_env() → MultiSceneEnvManager.
        super().__init__(task_config=task_config, **kwargs)

        logger.warning(
            "MultiSceneDCENavigationTask | %d scenes, rotation=%s",
            self._num_coexisting,
            "sync" if self._rotation_enabled else "disabled",
        )

    # ------------------------------------------------------------------
    # Sim env creation (called by NavigationTask.__init__)
    # ------------------------------------------------------------------

    def _build_sim_env(self):
        """Create a MultiSceneEnvManager with the current active scenes."""
        return MultiSceneEnvManager(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            device=self.device,
            scene_infos=self._active_scene_infos,
            scene_spacing_margin=self._scene_spacing_margin,
            args=self.task_config.args,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )

    # ------------------------------------------------------------------
    # Goal sampling — route to per-env scene navmesh
    # ------------------------------------------------------------------

    def _sample_navmesh_goals(self, env_ids):
        """Override: delegate to the multi-scene navmesh sampler."""
        if self.goal_navmesh_sampler is None:
            return None

        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        nav_cfg = self.task_config.navmesh_sampling
        goal_h = tuple(getattr(nav_cfg, "goal_height_offset_range", [0.0, 0.0]))
        goal_edge_padding = float(getattr(nav_cfg, "edge_padding", 0.0))

        goals = self.goal_navmesh_sampler.sample_world_points(
            env_ids=env_ids,
            height_offset_range=goal_h,
            edge_padding=goal_edge_padding,
        )
        if goals is None:
            return None

        # Separation filtering — same logic as parent, using per-env positions.
        min_sep = float(getattr(nav_cfg, "goal_min_separation", 0.0) or 0.0)
        max_sep_cfg = getattr(nav_cfg, "goal_max_separation", None)
        max_sep = float(max_sep_cfg) if max_sep_cfg is not None else None
        max_attempts = int(getattr(nav_cfg, "max_pair_sampling_attempts", 4))

        if min_sep > 0.0 or max_sep is not None:
            robot_pos = self.obs_dict["robot_position"][env_ids]
            for _ in range(max_attempts):
                dist = torch.norm(goals - robot_pos, dim=1)
                invalid = dist < min_sep
                if max_sep is not None:
                    invalid = torch.logical_or(invalid, dist > max_sep)
                if not torch.any(invalid):
                    break
                invalid_env_ids = env_ids[invalid.to(env_ids.device)]
                resampled = self.goal_navmesh_sampler.sample_world_points(
                    env_ids=invalid_env_ids,
                    height_offset_range=goal_h,
                    edge_padding=goal_edge_padding,
                )
                if resampled is None:
                    break
                goals[invalid] = resampled

        return goals

    # ------------------------------------------------------------------
    # Step — scene rotation check
    # ------------------------------------------------------------------

    def step(self, actions):
        result = super().step(actions)

        if not self._rotation_enabled:
            return result

        # Count completed rollouts from this step.
        terminated_or_truncated = torch.logical_or(
            self.terminations > 0, self.truncations > 0
        )
        self._rollout_counter += int(terminated_or_truncated.sum().item())

        # Check swap trigger.
        should_swap = False
        if self._switch_after_steps is not None:
            if self.num_task_steps > 0 and self.num_task_steps % self._switch_after_steps == 0:
                should_swap = True
        if self._switch_after_rollouts is not None:
            if self._rollout_counter >= self._switch_after_rollouts:
                should_swap = True

        if should_swap and not self._waiting_for_drain:
            self._waiting_for_drain = True
            logger.info(
                "Scene rotation triggered at step %d (rollouts=%d). "
                "Waiting for all episodes to finish...",
                self.num_task_steps, self._rollout_counter,
            )

        if self._waiting_for_drain:
            # In sync mode: check if all envs have terminated/truncated since
            # we started waiting.  We detect this by checking if sim_steps are
            # low (just been reset) for all envs.
            all_fresh = torch.all(self.sim_env.sim_steps <= 1)
            if all_fresh:
                self._execute_sync_swap()

        return result

    # ------------------------------------------------------------------
    # Sync swap
    # ------------------------------------------------------------------

    def _execute_sync_swap(self):
        """Destroy sim, pick new scenes, rebuild, rebind tensors."""
        logger.warning(
            "Executing sync scene swap: replacing %s",
            [s.scene_id for s in self._active_scene_infos],
        )

        # Pick new scenes (avoid repeating the same set when possible).
        new_scenes = self._scene_pool.sample(
            self._num_coexisting, exclude=self._active_scene_ids
        )
        self._active_scene_infos = new_scenes
        self._active_scene_ids = [s.scene_id for s in new_scenes]

        logger.warning("New scenes: %s", self._active_scene_ids)

        # Destroy old simulation.
        del self.sim_env
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Rebuild.
        self.sim_env = self._build_sim_env()
        self._rebind_after_swap()

        # Reset counters.
        self._rollout_counter = 0
        self._waiting_for_drain = False

        # Full reset.
        self.sim_env.reset()

        # Re-setup navmesh goal sampling for the new sim_env.
        self.goal_navmesh_sampler = None
        self.navmesh_goal_sampling_enabled = False
        self._setup_navmesh_goal_sampling()

        logger.warning("Sync scene swap complete.")

    def _rebind_after_swap(self):
        """Re-link task tensor references to the new sim_env."""
        self.obs_dict = self.sim_env.get_obs()
        if "curriculum_level" not in self.obs_dict:
            self.obs_dict["curriculum_level"] = self.curriculum_level
        self.obs_dict["num_obstacles_in_env"] = self.curriculum_level

        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.num_envs, device=self.device)

        self.target_position = torch.zeros(
            (self.num_envs, 3), device=self.device, requires_grad=False
        )
        self.pos_error_vehicle_frame_prev = torch.zeros_like(self.target_position)
        self.pos_error_vehicle_frame = torch.zeros_like(self.target_position)

        self.image_latents = torch.zeros(
            (self.num_envs, self.task_config.vae_config.latent_dims),
            device=self.device,
            requires_grad=False,
        )

        self.task_obs = {
            "observations": torch.zeros(
                (self.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_env_cfg(task_config):
        """Resolve the env config class from the registry."""
        from aerial_gym.registry.env_registry import env_config_registry
        return env_config_registry.make_env(task_config.env_name)
