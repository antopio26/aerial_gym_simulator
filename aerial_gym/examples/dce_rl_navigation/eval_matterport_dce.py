"""
Evaluation script: DCE RL navigation policy in a Matterport 3D environment.

Features:
  - Navmesh-based robot spawn and goal selection (safe, floor-level positions)
  - Pretrained DCE navigation policy inference (Sample Factory checkpoint)
  - Isaac Gym viewer showing the 3D scene, goal crosshair, and trajectory trail
  - Real-time OpenCV window with RGB and depth camera streams side-by-side

Usage (from the dce_rl_navigation directory):
  ./run_matterport_dce_eval.sh

Or directly:
  python3 eval_matterport_dce.py \\
      --train_dir=$(pwd)/selected_network \\
      --experiment=selected_network \\
      --env=test --obs_key=observations --load_checkpoint_kind=best
"""

# isort: off
import isaacgym  # must be imported before torch

# isort: on
import argparse
from collections import deque

import cv2
import numpy as np
import torch

from aerial_gym import AERIAL_GYM_DIRECTORY
from aerial_gym.config.env_config.matterport_glb_env import MatterportGLBEnvCfg
from aerial_gym.config.task_config.navigation_task_config import task_config as _BaseCfg
from aerial_gym.examples.dce_rl_navigation.dce_navigation_task import DCE_RL_Navigation_Task
from aerial_gym.examples.dce_rl_navigation.sf_inference_class import NN_Inference_Class
from aerial_gym.registry.task_registry import task_registry
from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym import (
    parse_aerialgym_cfg,
)
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)


# ---------------------------------------------------------------------------
# Custom task configuration
# ---------------------------------------------------------------------------

class MatterportDCEEvalTaskConfig(_BaseCfg):
    """
    Single-env task config for DCE policy evaluation in a Matterport scene.

    Key overrides vs. the base navigation_task_config:
      - env_name       → matterport_glb_env
            - robot_name     → lmf2_with_shaded_rgbd_camera  (original DCE robot + RGBD)
            - controller_name→ lmf2_velocity_control
      - num_envs       → 1
      - headless       → False (viewer enabled)
      - navmesh_sampling.enable → True (goal sampling from navmesh)
      - curriculum.min_level → 36 (matches DCE_RL_Navigation_Task default)
      - episode_len_steps → 500 (longer for evaluation)
    """

    seed = 42
    sim_name = "base_sim"
    env_name = "matterport_glb_env"
    robot_name = "lmf2_with_shaded_rgbd_camera"
    controller_name = "lmf2_velocity_control"
    args = {}
    num_envs = 1
    use_warp = True
    headless = False
    device = "cuda:0"

    observation_space_dim = 81  # 17 state + 64 VAE latent (unchanged from training)
    privileged_observation_space_dim = 0
    action_space_dim = 3  # DCE uses 3-dim actions; re-applied by DCE_RL_Navigation_Task
    episode_len_steps = 1000
    return_state_before_reset = False

    # Fallback ratio-based goal bounds (used when navmesh sampling fails)
    target_min_ratio = [0.10, 0.10, 0.10]
    target_max_ratio = [0.90, 0.90, 0.90]

    class navmesh_sampling:
        enable = True
        goal_height_offset_range = [0.15, 0.40]
        goal_min_separation = 3.0
        goal_max_separation = None
        max_pair_sampling_attempts = 6

    class vae_config(_BaseCfg.vae_config):
        # Inherited: use_vae=True, latent_dims=64, image_res=(270,480)
        # The shaded RGBD camera now follows base depth camera resolution (135,240).
        # VAE input can still be resized to image_res by the existing pipeline.
        pass

    class curriculum(_BaseCfg.curriculum):
        min_level = 36  # matches DCE_RL_Navigation_Task.__init__ override

    # Redefine so the device reference points to THIS class (not _BaseCfg).
    def action_transformation_function(action):
        clamped_action = torch.clamp(action, -1.0, 1.0)
        max_speed = 2.0
        max_yawrate = torch.pi / 3
        max_inclination_angle = torch.pi / 4
        clamped_action[:, 0] += 1.0
        processed_action = torch.zeros(
            (clamped_action.shape[0], 4),
            device=MatterportDCEEvalTaskConfig.device,
            requires_grad=False,
        )
        processed_action[:, 0] = (
            clamped_action[:, 0]
            * torch.cos(max_inclination_angle * clamped_action[:, 1])
            * max_speed
            / 2.0
        )
        processed_action[:, 1] = 0.0
        processed_action[:, 2] = (
            clamped_action[:, 0]
            * torch.sin(max_inclination_angle * clamped_action[:, 1])
            * max_speed
            / 2.0
        )
        processed_action[:, 3] = clamped_action[:, 2] * max_yawrate
        return processed_action


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_eval_args():
    """Parse script-specific args not consumed by Sample Factory."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--scene_file",
        default=None,
        help="Explicit path to a .glb scene file (overrides default).",
    )
    p.add_argument(
        "--scene_folder",
        default=None,
        help="Folder containing a .glb scene file (auto-detected).",
    )
    p.add_argument(
        "--max_episodes",
        type=int,
        default=50,
        help="Stop after this many episodes.",
    )
    p.add_argument(
        "--traj_len",
        type=int,
        default=200,
        help="Maximum number of trajectory waypoints shown in the viewer.",
    )
    p.add_argument(
        "--vis_every",
        type=int,
        default=4,
        help="Update the matplotlib window every N simulation steps.",
    )
    p.add_argument(
        "--policy_every",
        type=int,
        default=4,
        help="Run policy inference every N physics steps (hold last action in between).",
    )
    known, _ = p.parse_known_args()
    return known


# ---------------------------------------------------------------------------
# Environment / navmesh setup
# ---------------------------------------------------------------------------

def _resolve_glb(scene_file=None, scene_folder=None):
    """
    Return the absolute path to a .glb scene file.
    Resolution order:
      1. --scene_file (absolute or relative to AERIAL_GYM_DIRECTORY)
      2. --scene_folder  → first .glb inside it
         folder may be: absolute path | relative to AERIAL_GYM_DIRECTORY |
                        name under AERIAL_GYM_DIRECTORY/resources/envs/
      3. Default from MatterportGLBEnvCfg.static_scene.file
         (made absolute against AERIAL_GYM_DIRECTORY)
    Raises FileNotFoundError with a list of discovered .glb files if nothing works.
    """
    import glob
    import os

    res_envs = os.path.join(AERIAL_GYM_DIRECTORY, "resources/envs")

    if scene_file is not None:
        p = scene_file if os.path.isabs(scene_file) else os.path.join(AERIAL_GYM_DIRECTORY, scene_file)
        if not os.path.exists(p):
            raise FileNotFoundError(f"--scene_file not found: {p}")
        return p

    if scene_folder is not None:
        for candidate_dir in [
            scene_folder,
            os.path.join(AERIAL_GYM_DIRECTORY, scene_folder),
            os.path.join(res_envs, scene_folder),
        ]:
            glbs = sorted(glob.glob(os.path.join(candidate_dir, "*.glb")))
            if glbs:
                return glbs[0]
        raise FileNotFoundError(f"No .glb found for --scene_folder '{scene_folder}'")

    # Default path from config
    default = MatterportGLBEnvCfg.static_scene.file
    p = default if os.path.isabs(default) else os.path.join(AERIAL_GYM_DIRECTORY, default)
    if os.path.exists(p):
        return p

    # Nothing worked — scan and report
    found = sorted(glob.glob(os.path.join(res_envs, "**/*.glb"), recursive=True))
    hint = ("\n  Available scenes:\n    " + "\n    ".join(found)) if found else "\n  No .glb files found under resources/envs/."
    raise FileNotFoundError(
        f"Default scene not found: {p}\n"
        f"Pass --scene_file or --scene_folder to specify the location.{hint}"
    )


def setup_navmesh(scene_file=None, scene_folder=None):
    """Resolve scene, enable navmesh spawn (env) and goal sampling (task)."""
    glb_path = _resolve_glb(scene_file=scene_file, scene_folder=scene_folder)
    MatterportGLBEnvCfg.static_scene.file = glb_path
    logger.warning("Matterport scene: %s", glb_path)

    # navmesh_file=None → NavMeshSpawnSampler auto-resolves from the scene path
    MatterportGLBEnvCfg.navmesh_sampling.enable = True
    MatterportGLBEnvCfg.navmesh_sampling.navmesh_file = None
    MatterportDCEEvalTaskConfig.navmesh_sampling.enable = True
    logger.warning("Navmesh sampling enabled.")


# ---------------------------------------------------------------------------
# NN model construction
# ---------------------------------------------------------------------------

def build_nn_model(num_envs):
    cfg = parse_aerialgym_cfg(evaluation=True)
    model = NN_Inference_Class(
        num_envs=num_envs,
        num_actions=3,
        num_obs=81,
        cfg=cfg,
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# OpenCV camera display
# ---------------------------------------------------------------------------

CV_WIN = "DCE Navigation | RGB                    Depth"


def build_display():
    """Create a named OpenCV window. Returns nothing — state is in the window."""
    cv2.namedWindow(CV_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CV_WIN, 960, 270)
    # Show a black placeholder so the window appears immediately
    cv2.imshow(CV_WIN, np.zeros((270, 960, 3), dtype=np.uint8))
    cv2.waitKey(1)


def update_display(obs_dict):
    """
    Compose RGB | Depth side-by-side and push to the OpenCV window.
    RGB:   float [0,1] (H,W,3) → uint8 BGR
    Depth: float [0,1] (H,W)   → COLORMAP_PLASMA uint8 BGR
    """
    has_rgb   = "rgb_pixels"         in obs_dict
    has_depth = "depth_range_pixels" in obs_dict

    if has_rgb:
        rgb_np  = obs_dict["rgb_pixels"][0, 0].cpu().numpy()          # (H,W,3) RGB float
        rgb_u8  = np.clip(rgb_np * 255.0, 0, 255).astype(np.uint8)
        rgb_bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    else:
        rgb_bgr = np.zeros((270, 480, 3), dtype=np.uint8)

    if has_depth:
        depth_np  = obs_dict["depth_range_pixels"][0, 0].cpu().numpy()  # (H,W) float
        depth_u8  = np.clip(depth_np * 255.0, 0, 255).astype(np.uint8)
        depth_bgr = cv2.applyColorMap(depth_u8, cv2.COLORMAP_PLASMA)
    else:
        depth_bgr = np.zeros((270, 480, 3), dtype=np.uint8)

    # Make both panels the same height before concatenating
    h = max(rgb_bgr.shape[0], depth_bgr.shape[0])
    if rgb_bgr.shape[0] != h:
        rgb_bgr   = cv2.resize(rgb_bgr,   (rgb_bgr.shape[1],   h))
    if depth_bgr.shape[0] != h:
        depth_bgr = cv2.resize(depth_bgr, (depth_bgr.shape[1], h))

    combined = np.concatenate([rgb_bgr, depth_bgr], axis=1)
    cv2.imshow(CV_WIN, combined)
    cv2.waitKey(1)


# ---------------------------------------------------------------------------
# Isaac Gym debug line drawing
# ---------------------------------------------------------------------------

def _get_viewer_handles(rl_task):
    """
    Return (gym, viewer_handle, env_handle_0) or (None, None, None) if headless.
    Access path:
      rl_task.sim_env          → EnvManager
      .IGE_env                 → IsaacGymEnv
      .viewer                  → IGEViewerControl  (None when headless)
      .viewer (inner attr)     → raw gymapi viewer handle
      .gym                     → gymapi gym object
      .env_handles             → list of environment handles
    """
    ige = rl_task.sim_env.IGE_env
    viewer_ctrl = ige.viewer
    if viewer_ctrl is None or viewer_ctrl.viewer is None:
        return None, None, None
    return viewer_ctrl.gym, viewer_ctrl.viewer, ige.env_handles[0]


def draw_debug(rl_task, traj_list, goal_np, crosshair_half=0.3):
    """
    Draw goal crosshair (3 coloured axes) and trajectory trail in the viewer.

    traj_list  : list of (3,) numpy arrays, most-recent last
    goal_np    : (3,) numpy array, world-frame goal position
    """
    gym, viewer, env_handle = _get_viewer_handles(rl_task)
    if gym is None:
        return

    gym.clear_lines(viewer)

    lines = []   # each entry: [x0,y0,z0, x1,y1,z1]
    colors = []  # each entry: [r,g,b]

    # --- Goal crosshair ---
    gx, gy, gz = float(goal_np[0]), float(goal_np[1]), float(goal_np[2])
    h = crosshair_half
    lines.append([gx - h, gy, gz, gx + h, gy, gz])
    colors.append([1.0, 0.0, 0.0])  # X arm — red
    lines.append([gx, gy - h, gz, gx, gy + h, gz])
    colors.append([0.0, 1.0, 0.0])  # Y arm — green
    lines.append([gx, gy, gz - h, gx, gy, gz + h])
    colors.append([0.0, 0.0, 1.0])  # Z arm — blue

    # --- Trajectory trail ---
    n = len(traj_list)
    for k in range(n - 1):
        p0, p1 = traj_list[k], traj_list[k + 1]
        lines.append([p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]])
        alpha = (k + 1) / max(n - 1, 1)
        colors.append([0.0, alpha, alpha])  # cyan gradient (dim → bright)

    if not lines:
        return

    verts_np = np.array(lines, dtype=np.float32)    # (N, 6)
    colors_np = np.array(colors, dtype=np.float32)  # (N, 3)
    gym.add_lines(viewer, env_handle, len(lines), verts_np, colors_np)


# ---------------------------------------------------------------------------
# Episode statistics
# ---------------------------------------------------------------------------

class EpisodeStats:
    def __init__(self):
        self.successes = 0
        self.crashes = 0
        self.timeouts = 0
        self.episodes = 0

    def record(self, termination, truncation, infos):
        if termination[0]:
            self.crashes += 1
            self.episodes += 1
        elif truncation[0]:
            if "successes" in infos and infos["successes"][0]:
                self.successes += 1
            else:
                self.timeouts += 1
            self.episodes += 1

    def log(self):
        if self.episodes == 0:
            return
        logger.warning(
            "Episodes: %d | Successes: %d (%.0f%%) | Crashes: %d (%.0f%%) | "
            "Timeouts: %d (%.0f%%)",
            self.episodes,
            self.successes, 100.0 * self.successes / self.episodes,
            self.crashes,   100.0 * self.crashes   / self.episodes,
            self.timeouts,  100.0 * self.timeouts   / self.episodes,
        )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(eval_args):
    policy_every = max(1, int(eval_args.policy_every))

    # 1. Mutate env config before the task (and thus EnvManager) is built
    setup_navmesh(
        scene_file=eval_args.scene_file,
        scene_folder=eval_args.scene_folder,
    )

    # 2. Register the custom task
    task_registry.register_task(
        "matterport_dce_eval_task",
        DCE_RL_Navigation_Task,
        MatterportDCEEvalTaskConfig,
    )

    # 3. Build the task (num_envs=1 already enforced by config)
    rl_task = task_registry.make_task(
        "matterport_dce_eval_task",
        seed=42,
        use_warp=True,
        headless=False,
    )
    logger.warning("Task created. num_envs=%d", rl_task.num_envs)
    logger.warning("Policy inference frequency: every %d physics steps", policy_every)

    # 4. Build the NN policy (parse_aerialgym_cfg reads CLI args here)
    nn_model = build_nn_model(rl_task.num_envs)
    nn_model.reset(torch.arange(rl_task.num_envs))

    # 5. Set up the OpenCV camera window
    build_display()

    # 6. Trajectory ring buffer
    traj_buf = deque(maxlen=eval_args.traj_len)

    # 7. Episode stats
    stats = EpisodeStats()

    # 8. Initial environment reset
    rl_task.reset()
    command_actions = torch.zeros(
        (rl_task.num_envs, rl_task.task_config.action_space_dim),
        device=MatterportDCEEvalTaskConfig.device,
    )

    max_steps = eval_args.max_episodes * MatterportDCEEvalTaskConfig.episode_len_steps

    # 9. Main loop
    for step_i in range(max_steps):
        # Step the physics + sensor pipeline
        obs, rewards, termination, truncation, infos = rl_task.step(command_actions)

        # Policy inference every N steps, holding last action in between.
        if step_i % policy_every == 0:
            obs["obs"] = obs["observations"]
            action = nn_model.get_action(obs)
            action = torch.as_tensor(action, device=command_actions.device).expand(
                rl_task.num_envs, -1
            )
            command_actions[:] = action

        # Track trajectory for env 0
        robot_pos = rl_task.obs_dict["robot_position"][0].cpu().numpy().copy()
        traj_buf.append(robot_pos)

        # Draw goal marker + trajectory in the Isaac Gym viewer every step
        goal_np = rl_task.target_position[0].cpu().numpy()
        draw_debug(rl_task, list(traj_buf), goal_np)

        # Update camera feed every vis_every steps
        if step_i % eval_args.vis_every == 0:
            update_display(rl_task.obs_dict)

        # Handle episode resets
        any_done = torch.any(termination + truncation)
        if any_done:
            reset_ids = (termination + truncation).nonzero(as_tuple=True)
            stats.record(termination, truncation, infos)

            if termination[0]:
                logger.warning("Episode %d: CRASH", stats.episodes)
            elif truncation[0]:
                dist = torch.norm(
                    rl_task.target_position[0] - rl_task.obs_dict["robot_position"][0]
                ).item()
                logger.warning(
                    "Episode %d: %s (final dist to goal: %.2f m)",
                    stats.episodes,
                    "SUCCESS" if (infos.get("successes", torch.zeros(1))[0]) else "TIMEOUT",
                    dist,
                )

            nn_model.reset(reset_ids)
            traj_buf.clear()

            if stats.episodes % 10 == 0:
                stats.log()

            if stats.episodes >= eval_args.max_episodes:
                break

    # Final summary
    stats.log()
    logger.warning("Evaluation complete after %d steps.", step_i + 1)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    eval_args = parse_eval_args()
    run_evaluation(eval_args)
