"""
Evaluation script: DCE RL navigation policy in a Matterport 3D environment.

Supports two pipelines selectable via --pipeline:

  dce (default)
    Single drone.  Depth image → frozen VAE → 64-dim latent → RL policy.
    Reproduces the original DCE evaluation exactly.

  comparison
    Two parallel drones with the same spawn point and goal:
      - Drone 0: depth → VAE    (DCE pipeline)
      - Drone 1: RGB  → ViT+adapter  (ViT pipeline)
    Both use the same RL policy weights.  Per-pipeline episode statistics
    are tracked and printed separately.

Common features:
  - Navmesh-based robot spawn and goal selection (safe, floor-level positions)
  - Isaac Gym 3D viewer (shows drone 0 / DCE drone in both modes)
  - OpenCV camera window:
      dce mode:        RGB | Depth  (side-by-side, 1 row)
      comparison mode: 2x2 grid — top row = DCE drone, bottom row = ViT drone
                       Panels for the unused modality are dimmed and labelled.

Usage:
  # DCE-only (original behaviour)
  python eval_matterport_dce.py \\
      --train_dir=$(pwd)/selected_network \\
      --experiment=selected_network \\
      --env=test --obs_key=observations --load_checkpoint_kind=best

  # Comparison mode
  python eval_matterport_dce.py \\
      --pipeline=comparison \\
      --vit_model_path=/path/to/vit_adapter_pipeline_240x320.pt \\
      --vit_metadata=/path/to/metadata.json \\
      --train_dir=$(pwd)/selected_network \\
      --experiment=selected_network \\
      --env=test --obs_key=observations --load_checkpoint_kind=best
"""

# isort: off
import isaacgym  # must be imported before torch
# isort: on

import argparse
import cv2
import numpy as np
import torch

from aerial_gym import AERIAL_GYM_DIRECTORY
from aerial_gym.config.env_config.matterport_glb_env import MatterportGLBEnvCfg
from aerial_gym.config.task_config.matterport_dce_task_config import (
    MatterportVAETaskConfig,
    MatterportComparisonTaskConfig,
)
from aerial_gym.examples.dce_rl_navigation.matterport_dce_task import (
    MatterportDCENavigationTask,
    MatterportComparisonTask,
)
from aerial_gym.examples.dce_rl_navigation.sf_inference_class import NN_Inference_Class
from aerial_gym.registry.task_registry import task_registry
from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym import (
    parse_aerialgym_cfg,
)
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_eval_args() -> argparse.Namespace:
    """Parse script-specific args; leaves Sample Factory args untouched in sys.argv."""
    import sys

    p = argparse.ArgumentParser(add_help=False)

    # --- Pipeline selection ---
    p.add_argument(
        "--pipeline",
        default="dce",
        choices=["dce", "comparison"],
        help="Evaluation pipeline: 'dce' (default) or 'comparison' (DCE vs ViT side-by-side).",
    )

    # --- ViT pipeline (comparison mode) ---
    p.add_argument(
        "--vit_model_path",
        default=None,
        help="Path to the TorchScript ViT+adapter model (.pt). Required for --pipeline=comparison.",
    )
    p.add_argument(
        "--vit_metadata",
        default=None,
        help="Path to metadata.json produced by export_vit_adapter.py. Required for --pipeline=comparison.",
    )

    # --- Scene / environment ---
    p.add_argument(
        "--scene_file",
        default=None,
        help="Explicit path to a .glb scene file (overrides default).",
    )
    p.add_argument(
        "--scene_folder",
        default="00807-rsggHU7g7dh",
        help="Folder containing a .glb scene file (auto-detected).",
    )
    p.add_argument(
        "--scene_scale",
        type=float,
        default=None,
        help="Absolute scale applied to the mesh and navmesh (overrides the config default).",
    )
    p.add_argument(
        "--texture_atlas_tile_size",
        type=int,
        default=1024,
        help="Matterport texture atlas tile size. Lower values reduce GPU memory.",
    )

    # --- Episode control ---
    p.add_argument(
        "--max_episodes",
        type=int,
        default=500,
        help="Stop after this many episodes.",
    )
    p.add_argument(
        "--max_goal_spawn_dz",
        type=float,
        default=0.5,
        help="Maximum allowed |goal_z - spawn_z| during goal regeneration.",
    )

    # --- Timing / cadence ---
    p.add_argument(
        "--policy_every",
        type=int,
        default=None,
        help=(
            "Run RL policy every N physics steps (action repeat). "
            "Overrides task_config.policy_every_n_steps. "
            "Default: use value from task config."
        ),
    )
    p.add_argument(
        "--perception_every",
        type=int,
        default=None,
        help=(
            "Encode image latent every N task steps (reuse previous latent in between). "
            "Overrides task_config.perception_every_n_steps. "
            "Default: use value from task config."
        ),
    )
    p.add_argument(
        "--display_every",
        type=int,
        default=4,
        help="Update the OpenCV camera window every N simulation steps.",
    )
    p.add_argument(
        "--vis_every",
        type=int,
        default=4,
        help="Update the Isaac Gym debug lines (goal + trail) every N simulation steps.",
    )
    p.add_argument(
        "--viewer_every",
        type=int,
        default=1,
        help="Render Isaac viewer every N simulation steps.",
    )

    # --- Viewer options ---
    p.add_argument(
        "--sync_frame_time",
        action="store_true",
        default=False,
        help="Enable Isaac Gym frame sync.",
    )
    p.add_argument(
        "--disable_viewer_sync",
        action="store_true",
        default=False,
        help="Disable viewer graphics sync before draw.",
    )

    known, unknown = p.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown
    return known


# ===========================================================================
# Scene / navmesh setup
# ===========================================================================

def _resolve_glb(scene_file=None, scene_folder=None) -> str:
    """
    Return the absolute path to a .glb scene file.

    Resolution order:
      1. --scene_file (absolute or relative to AERIAL_GYM_DIRECTORY)
      2. --scene_folder → first .glb inside it
         (absolute | relative to AERIAL_GYM_DIRECTORY | name under resources/envs/val)
      3. Default from MatterportGLBEnvCfg.static_scene.file
    """
    import glob
    import os

    res_envs = os.path.join(AERIAL_GYM_DIRECTORY, "resources/envs/val")

    if scene_file is not None:
        p = scene_file if os.path.isabs(scene_file) else os.path.join(AERIAL_GYM_DIRECTORY, scene_file)
        if not os.path.exists(p):
            raise FileNotFoundError(f"--scene_file not found: {p}")
        return p

    if scene_folder is not None:
        for candidate in [
            scene_folder,
            os.path.join(AERIAL_GYM_DIRECTORY, scene_folder),
            os.path.join(res_envs, scene_folder),
        ]:
            glbs = sorted(glob.glob(os.path.join(candidate, "*.glb")))
            if glbs:
                return glbs[0]
        raise FileNotFoundError(f"No .glb found for --scene_folder '{scene_folder}'")

    default = MatterportGLBEnvCfg.static_scene.file
    p = default if os.path.isabs(default) else os.path.join(AERIAL_GYM_DIRECTORY, default)
    if os.path.exists(p):
        return p

    found = sorted(glob.glob(os.path.join(res_envs, "**/*.glb"), recursive=True))
    hint = ("\n  Available:\n    " + "\n    ".join(found)) if found else "\n  No .glb files found."
    raise FileNotFoundError(f"Default scene not found: {p}{hint}")


def setup_scene(eval_args) -> None:
    """
    Apply scene / navmesh configuration to the global MatterportGLBEnvCfg and
    the task config classes before the environment is created.
    """
    glb_path = _resolve_glb(
        scene_file=eval_args.scene_file,
        scene_folder=eval_args.scene_folder,
    )
    MatterportGLBEnvCfg.static_scene.file = glb_path
    MatterportGLBEnvCfg.static_scene.texture_atlas_tile_size = max(
        128, int(eval_args.texture_atlas_tile_size)
    )
    MatterportGLBEnvCfg.env.render_viewer_every_n_steps = max(1, int(eval_args.viewer_every))

    scene_scale = float(eval_args.scene_scale) if eval_args.scene_scale is not None else MatterportGLBEnvCfg.static_scene.scale
    MatterportGLBEnvCfg.static_scene.scale = scene_scale

    logger.warning("Matterport scene: %s", glb_path)
    logger.warning(
        "Scene config | viewer_every=%d  texture_atlas_tile=%d  scale=%.4f",
        MatterportGLBEnvCfg.env.render_viewer_every_n_steps,
        MatterportGLBEnvCfg.static_scene.texture_atlas_tile_size,
        scene_scale,
    )

    # Navmesh: env-level spawn sampling
    MatterportGLBEnvCfg.navmesh_sampling.enable = True
    MatterportGLBEnvCfg.navmesh_sampling.navmesh_file = None
    MatterportGLBEnvCfg.navmesh_sampling.spawn_height_offset_range = list(
        MatterportVAETaskConfig.navmesh_sampling.spawn_height_offset_range
    )

    # Enable task-level navmesh goal sampling for both pipeline configs.
    for cfg_cls in (MatterportVAETaskConfig, MatterportComparisonTaskConfig):
        cfg_cls.navmesh_sampling.enable = True

    logger.warning("Navmesh sampling enabled.")
    logger.warning(
        "Height ranges | spawn(env)=%s  goal(task)=%s",
        MatterportGLBEnvCfg.navmesh_sampling.spawn_height_offset_range,
        MatterportVAETaskConfig.navmesh_sampling.goal_height_offset_range,
    )

    # Warn if height ranges are incompatible with the Z-tolerance
    spawn_h = MatterportGLBEnvCfg.navmesh_sampling.spawn_height_offset_range
    goal_h  = MatterportVAETaskConfig.navmesh_sampling.goal_height_offset_range
    dz      = float(eval_args.max_goal_spawn_dz)
    if not ((spawn_h[0] - dz <= goal_h[1]) and (goal_h[0] - dz <= spawn_h[1])):
        logger.warning(
            "Height ranges may be incompatible with max_goal_spawn_dz=%.2f. "
            "Spawn=%s  Goal=%s",
            dz, spawn_h, goal_h,
        )


# ===========================================================================
# Task creation
# ===========================================================================

def make_task(eval_args, pipeline: str):
    """Register and instantiate the navigation task for the chosen pipeline."""
    if pipeline == "comparison":
        if not eval_args.vit_model_path or not eval_args.vit_metadata:
            raise ValueError(
                "--pipeline=comparison requires --vit_model_path and --vit_metadata."
            )
        MatterportComparisonTaskConfig.vit_config.model_path    = eval_args.vit_model_path
        MatterportComparisonTaskConfig.vit_config.metadata_path = eval_args.vit_metadata
        task_registry.register_task(
            "matterport_comparison_task",
            MatterportComparisonTask,
            MatterportComparisonTaskConfig,
        )
        rl_task = task_registry.make_task(
            "matterport_comparison_task",
            seed=42,
            use_warp=True,
            headless=False,
        )
    else:  # "dce"
        task_registry.register_task(
            "matterport_dce_eval_task",
            MatterportDCENavigationTask,
            MatterportVAETaskConfig,
        )
        rl_task = task_registry.make_task(
            "matterport_dce_eval_task",
            seed=42,
            use_warp=True,
            headless=False,
        )

    logger.warning("Task created | pipeline=%s  num_envs=%d", pipeline, rl_task.num_envs)
    return rl_task


# ===========================================================================
# RL policy
# ===========================================================================

def build_nn_model(num_envs: int) -> NN_Inference_Class:
    """Build the Sample Factory RL policy (shared across all envs / pipelines)."""
    cfg = parse_aerialgym_cfg(evaluation=True)
    model = NN_Inference_Class(
        num_envs=num_envs,
        num_actions=3,
        num_obs=81,
        cfg=cfg,
    )
    model.eval()
    return model


# ===========================================================================
# Viewer configuration
# ===========================================================================

def configure_runtime_viewer(rl_task, eval_args) -> None:
    viewer_ctrl = rl_task.sim_env.IGE_env.viewer
    if viewer_ctrl is None:
        return
    viewer_ctrl.sync_frame_time      = bool(eval_args.sync_frame_time)
    viewer_ctrl.enable_viewer_sync   = not bool(eval_args.disable_viewer_sync)
    logger.warning(
        "Viewer config | sync_frame_time=%s  enable_viewer_sync=%s",
        viewer_ctrl.sync_frame_time,
        viewer_ctrl.enable_viewer_sync,
    )


# ===========================================================================
# OpenCV camera display
# ===========================================================================

_CV_WIN_DCE        = "DCE Navigation | RGB          Depth"
_CV_WIN_COMPARISON = "DCE vs ViT | DCE: RGB   Depth  |  ViT: RGB   Depth"

# Dimming and desaturation for "dimmed / not-used" panels
_DIM_FACTOR   = 0.5   # multiply pixel values by this to darken


def _to_bgr_u8(tensor_hw, colormap=None) -> np.ndarray:
    """Convert a (H, W) float [0,1] tensor to a BGR uint8 image, optionally with colormap."""
    arr = np.clip(tensor_hw.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    if colormap is not None:
        return cv2.applyColorMap(arr, colormap)
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)


def _rgb_to_bgr_u8(tensor_hwc) -> np.ndarray:
    """Convert a (H, W, 3) float [0,1] RGB tensor to BGR uint8."""
    arr = np.clip(tensor_hwc.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _dim_panel(img: np.ndarray, label: str, sat_reduction: float = 0.70) -> np.ndarray:
    """
    Dim and desaturate an image to indicate it is the *unused* modality.
    A text label is drawn in the top-left corner.
    """
    # Convert BGR to HSV to reduce saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= (1.0 - sat_reduction)
    desat_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    out = (desat_img.astype(np.float32) * _DIM_FACTOR).clip(0, 255).astype(np.uint8)
    cv2.putText(out, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return out


def _label_panel(img: np.ndarray, label: str, custom_text=None, custom_color=(255, 255, 255)) -> np.ndarray:
    """Add a small text label to the top-left of an active (undimmed) panel."""
    out = img.copy()
    cv2.putText(out, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if custom_text:
        # Add success/crash text in the center
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        text_size = cv2.getTextSize(custom_text, font, font_scale, thickness)[0]
        text_x = (out.shape[1] - text_size[0]) // 2
        text_y = (out.shape[0] + text_size[1]) // 2
        cv2.putText(out, custom_text, (text_x, text_y), font, font_scale, custom_color, thickness, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------

def build_dce_display() -> None:
    cv2.namedWindow(_CV_WIN_DCE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_CV_WIN_DCE, 960, 270)
    cv2.imshow(_CV_WIN_DCE, np.zeros((270, 960, 3), dtype=np.uint8))
    cv2.waitKey(1)


def build_comparison_display() -> None:
    cv2.namedWindow(_CV_WIN_COMPARISON, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_CV_WIN_COMPARISON, 960, 540)   # 2 rows x 270 px
    cv2.imshow(_CV_WIN_COMPARISON, np.zeros((540, 960, 3), dtype=np.uint8))
    cv2.waitKey(1)


# --------------------------------------------------------------------------

def _get_rgb_depth(obs_dict, env_id: int):
    """
    Extract RGB and depth panels for env `env_id` from obs_dict.
    Returns (rgb_bgr, depth_bgr) as uint8 BGR images, or black fallbacks.
    """
    FALLBACK_RGB   = np.zeros((270, 480, 3), dtype=np.uint8)
    FALLBACK_DEPTH = np.zeros((270, 480, 3), dtype=np.uint8)

    try:
        rgb_bgr = (
            _rgb_to_bgr_u8(obs_dict["rgb_pixels"][env_id, 0])
            if "rgb_pixels" in obs_dict else FALLBACK_RGB
        )
    except Exception:
        rgb_bgr = FALLBACK_RGB

    try:
        depth_bgr = (
            _to_bgr_u8(obs_dict["depth_range_pixels"][env_id, 0], colormap=cv2.COLORMAP_PLASMA)
            if "depth_range_pixels" in obs_dict else FALLBACK_DEPTH
        )
    except Exception:
        depth_bgr = FALLBACK_DEPTH

    # Ensure both panels have the same height for horizontal concatenation
    h = max(rgb_bgr.shape[0], depth_bgr.shape[0])
    if rgb_bgr.shape[0]   != h:
        rgb_bgr   = cv2.resize(rgb_bgr,   (rgb_bgr.shape[1],   h))
    if depth_bgr.shape[0] != h:
        depth_bgr = cv2.resize(depth_bgr, (depth_bgr.shape[1], h))

    return rgb_bgr, depth_bgr


def update_dce_display(obs_dict) -> None:
    """Single-drone DCE display: RGB | Depth side-by-side."""
    try:
        if cv2.getWindowProperty(_CV_WIN_DCE, cv2.WND_PROP_VISIBLE) < 1:
            return
    except cv2.error:
        return

    rgb_bgr, depth_bgr = _get_rgb_depth(obs_dict, env_id=0)
    cv2.imshow(_CV_WIN_DCE, np.concatenate([rgb_bgr, depth_bgr], axis=1))
    cv2.waitKey(1)


def update_comparison_display(obs_dict, statuses=None) -> None:
    """
    2x2 comparison display.

    Row 0 (DCE drone):
      - Left:  DCE RGB    → dimmed (DCE does not use RGB)
      - Right: DCE Depth  → active (this is what DCE uses)

    Row 1 (ViT drone):
      - Left:  ViT RGB    → active (this is what ViT uses)
      - Right: ViT Depth  → dimmed (ViT does not use depth)
    """
    if statuses is None:
        statuses = [None, None]

    try:
        if cv2.getWindowProperty(_CV_WIN_COMPARISON, cv2.WND_PROP_VISIBLE) < 1:
            return
    except cv2.error:
        return

    def get_status_text_color(status):
        if status == "SUCCESS": return status, (0, 255, 0)
        if status == "CRASH": return status, (0, 0, 255)
        if status == "TIMEOUT": return status, (0, 165, 255)  # Orange
        return None, (255, 255, 255)

    dce_text, dce_color = get_status_text_color(statuses[0])
    vit_text, vit_color = get_status_text_color(statuses[1])

    # --- DCE drone (env 0) ---
    dce_rgb, dce_depth = _get_rgb_depth(obs_dict, env_id=0)
    dce_rgb_panel   = _dim_panel(dce_rgb, "DCE: RGB (unused)")
    dce_depth_panel = _label_panel(dce_depth, "DCE: Depth (used)", dce_text, dce_color)
    dce_row = np.concatenate([dce_rgb_panel, dce_depth_panel], axis=1)

    # --- ViT drone (env 1) ---
    vit_rgb, vit_depth = _get_rgb_depth(obs_dict, env_id=1)
    vit_rgb_panel   = _label_panel(vit_rgb, "ViT: RGB (used)", vit_text, vit_color)
    vit_depth_panel = _dim_panel(vit_depth, "ViT: Depth (unused)")
    vit_row = np.concatenate([vit_rgb_panel, vit_depth_panel], axis=1)

    # Stack rows; resize ViT row to match DCE row width if needed
    w = dce_row.shape[1]
    if vit_row.shape[1] != w:
        vit_row = cv2.resize(vit_row, (w, vit_row.shape[0]))

    cv2.imshow(_CV_WIN_COMPARISON, np.concatenate([dce_row, vit_row], axis=0))
    cv2.waitKey(1)


# ===========================================================================
# Isaac Gym debug lines (goal cross + trajectory trail)
# ===========================================================================

def _get_viewer_handles(rl_task):
    """Return (gym, viewer_handle, env_handle_0) or (None, None, None) if headless."""
    ige        = rl_task.sim_env.IGE_env
    viewer_ctrl = ige.viewer
    if viewer_ctrl is None or viewer_ctrl.viewer is None:
        return None, None, None
    return viewer_ctrl.gym, viewer_ctrl.viewer, ige.env_handles[0]


_GOAL_VERTS  = None
_GOAL_COLORS = None


def draw_debug(rl_task, goal_np, cross_half: float = 0.3) -> None:
    """Draw a red cross at the goal."""
    global _GOAL_VERTS, _GOAL_COLORS

    gym, viewer, env_handle = _get_viewer_handles(rl_task)
    if gym is None:
        return

    gx, gy, gz = float(goal_np[0]), float(goal_np[1]), float(goal_np[2])
    h = cross_half

    if _GOAL_VERTS is None:
        _GOAL_COLORS = np.array([[1., 0., 0.]] * 3, dtype=np.float32)
        _GOAL_VERTS  = np.zeros((3, 6), dtype=np.float32)

    _GOAL_VERTS[0] = [gx - h, gy, gz, gx + h, gy, gz]
    _GOAL_VERTS[1] = [gx, gy - h, gz, gx, gy + h, gz]
    _GOAL_VERTS[2] = [gx, gy, gz - h, gx, gy, gz + h]

    gym.clear_lines(viewer)
    gym.add_lines(viewer, env_handle, 3, _GOAL_VERTS, _GOAL_COLORS)


# ===========================================================================
# Episode statistics
# ===========================================================================

class EpisodeStats:
    """Tracks successes, crashes, and timeouts for a single pipeline."""

    def __init__(self):
        self.successes = 0
        self.crashes   = 0
        self.timeouts  = 0
        self.episodes  = 0

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

    def log(self, label: str = "") -> None:
        if self.episodes == 0:
            return
        prefix = f"[{label}] " if label else ""
        logger.warning(
            "%sEpisodes: %d | Successes: %d (%.0f%%) | Crashes: %d (%.0f%%) | "
            "Timeouts: %d (%.0f%%)",
            prefix,
            self.episodes,
            self.successes, 100.0 * self.successes / self.episodes,
            self.crashes,   100.0 * self.crashes   / self.episodes,
            self.timeouts,  100.0 * self.timeouts   / self.episodes,
        )


class ComparisonEpisodeStats:
    """Per-pipeline stats for comparison mode."""

    def __init__(self):
        self.dce = EpisodeStats()
        self.vit = EpisodeStats()

    @property
    def episodes(self) -> int:
        return self.dce.episodes

    def _slice_infos(self, infos: dict, env_id: int) -> dict:
        """Return a copy of infos with all tensor values sliced to [env_id:env_id+1]."""
        out = {}
        for k, v in infos.items():
            if isinstance(v, torch.Tensor):
                out[k] = v[[env_id]]
            else:
                out[k] = v
        return out

    def record(self, termination, truncation, infos) -> None:
        self.dce.record(
            termination[[MatterportComparisonTask.VAE_ENV_ID]],
            truncation[[MatterportComparisonTask.VAE_ENV_ID]],
            self._slice_infos(infos, MatterportComparisonTask.VAE_ENV_ID),
        )
        self.vit.record(
            termination[[MatterportComparisonTask.VIT_ENV_ID]],
            truncation[[MatterportComparisonTask.VIT_ENV_ID]],
            self._slice_infos(infos, MatterportComparisonTask.VIT_ENV_ID),
        )

    def log(self) -> None:
        self.dce.log(label="DCE / VAE")
        self.vit.log(label="ViT / Adapter")


# ===========================================================================
# Goal regeneration helpers
# ===========================================================================

def _regenerate_goal_with_height_tolerance(rl_task, max_dz: float = 0.5, max_attempts: int = 20) -> bool:
    """
    Resample the goal for env 0 until |goal_z - spawn_z| <= max_dz.
    In comparison mode the goal is immediately copied to env 1 by the task's
    reset_idx / _sync_goals methods, so no extra handling is needed here.
    """
    spawn_z = float(rl_task.obs_dict["robot_position"][0, 2].item())

    def _ok():
        return abs(float(rl_task.target_position[0, 2].item()) - spawn_z) <= max_dz

    if _ok():
        return True

    env_ids = torch.tensor([0], dtype=torch.long, device=rl_task.target_position.device)
    for _ in range(max_attempts):
        if getattr(rl_task, "navmesh_goal_sampling_enabled", False) and getattr(
            rl_task, "goal_navmesh_sampler", None
        ) is not None:
            goals = rl_task._sample_navmesh_goals(env_ids)
            if goals is not None:
                rl_task.target_position[env_ids] = goals
            else:
                rl_task.reset_idx(env_ids)
        else:
            rl_task.reset_idx(env_ids)

        if _ok():
            return True

    logger.warning(
        "Could not regenerate goal within %.2f m Z tolerance after %d attempts.",
        max_dz, max_attempts,
    )
    return False


def _log_spawn_goal(rl_task, episode: int) -> None:
    spawn = rl_task.obs_dict["robot_position"][0].cpu().numpy()
    goal  = rl_task.target_position[0].cpu().numpy()
    logger.warning(
        "Episode %d | Spawn [%.2f %.2f %.2f]  Goal [%.2f %.2f %.2f]  Dist %.2f m",
        episode,
        spawn[0], spawn[1], spawn[2],
        goal[0],  goal[1],  goal[2],
        float(np.linalg.norm(goal - spawn)),
    )


# ===========================================================================
# Per-episode reset logic (shared between DCE and comparison loops)
# ===========================================================================

def _handle_episode_reset(
    rl_task,
    nn_model:   NN_Inference_Class,
    stats,
    termination,
    truncation,
    infos,
    eval_args,
) -> None:
    """
    Called when any environment has terminated or truncated.
    Updates stats, resets RNN states, regenerates the goal.
    """
    done      = termination | truncation
    reset_ids = done.nonzero(as_tuple=True)

    stats.record(termination, truncation, infos)

    # Log outcome for env 0 (the reference / DCE drone in all modes)
    if termination[0]:
        logger.warning("Episode %d: CRASH", stats.episodes)
    elif truncation[0]:
        dist = torch.norm(
            rl_task.target_position[0] - rl_task.obs_dict["robot_position"][0]
        ).item()
        success = bool(infos.get("successes", torch.zeros(1))[0])
        logger.warning(
            "Episode %d: %s (dist to goal: %.2f m)",
            stats.episodes,
            "SUCCESS" if success else "TIMEOUT",
            dist,
        )

    nn_model.reset(reset_ids)

    _regenerate_goal_with_height_tolerance(
        rl_task, max_dz=float(eval_args.max_goal_spawn_dz)
    )
    _log_spawn_goal(rl_task, episode=stats.episodes)


# ===========================================================================
# Main evaluation loops
# ===========================================================================

def _run_dce_loop(eval_args, rl_task, nn_model: NN_Inference_Class) -> None:
    """
    Main step loop for DCE-only mode.

    VAE depth pipeline (matches dce_nn_navigation.py training exactly):
      lmf2 / BaseDepthCameraConfig → 135x240, max_range=10m, normalized [0,1]
      VAEPipeline.encode(): min-pool no-op → VAEImageEncoder interp to (270,480) → 64-dim latent
    """
    display_every = max(1, int(eval_args.display_every))
    max_steps     = eval_args.max_episodes * MatterportVAETaskConfig.episode_len_steps

    stats = EpisodeStats()

    build_dce_display()
    rl_task.reset()
    _regenerate_goal_with_height_tolerance(rl_task, max_dz=float(eval_args.max_goal_spawn_dz))
    _log_spawn_goal(rl_task, episode=0)

    command_actions = torch.zeros(
        (rl_task.num_envs, rl_task.task_config.action_space_dim),
        device=rl_task.device,
    )

    for step_i in range(max_steps):
        # Task handles action repeat (policy_every_n_steps) and perception cadence internally.
        obs, rewards, termination, truncation, infos = rl_task.step(command_actions)

        # Early success logic (independent of training env logic)
        dist = torch.norm(rl_task.target_position - rl_task.obs_dict["robot_position"], dim=1)
        early_success = (dist < 1.0) & (~termination.bool())
        truncation = truncation.bool() | early_success.bool()
        if "successes" not in infos:
            infos["successes"] = torch.zeros_like(truncation, dtype=torch.bool)
        if isinstance(infos["successes"], torch.Tensor):
            infos["successes"] = infos["successes"].bool() | early_success.bool()

        obs["obs"] = obs["observations"]
        action = nn_model.get_action(obs)
        command_actions[:] = torch.as_tensor(action, device=command_actions.device).expand(
            rl_task.num_envs, -1
        )

        if step_i % eval_args.vis_every == 0:
            draw_debug(rl_task, rl_task.target_position[0].cpu().numpy())

        if step_i % display_every == 0:
            update_dce_display(rl_task.obs_dict)

        done = termination | truncation
        if done.any():
            _handle_episode_reset(
                rl_task, nn_model,
                stats, termination, truncation, infos, eval_args,
            )
            if stats.episodes % 10 == 0:
                stats.log(label="DCE")
            if stats.episodes >= eval_args.max_episodes:
                break

    stats.log(label="DCE")
    logger.warning("Evaluation complete after %d steps.", step_i + 1)


def _run_comparison_loop(eval_args, rl_task, nn_model: NN_Inference_Class) -> None:
    """Main step loop for comparison mode (two parallel drones)."""
    display_every = max(1, int(eval_args.display_every))
    max_ep_steps  = MatterportVAETaskConfig.episode_len_steps

    # ---------------------------------------------------------
    # DISABLE AUTO-RESET AND CURRICULUM UPDATES
    # We want them to wait for each other, so we manage resets manually.
    if hasattr(rl_task.sim_env, "reset_terminated_and_truncated_envs"):
        rl_task.sim_env.reset_terminated_and_truncated_envs = lambda: torch.tensor([], dtype=torch.long, device=rl_task.device)
    if hasattr(rl_task, "check_and_update_curriculum_level"):
        rl_task.check_and_update_curriculum_level = lambda *args, **kwargs: None
    # ---------------------------------------------------------

    stats = ComparisonEpisodeStats()

    def sync_robots_and_goal():
        rl_task.target_position[1] = rl_task.target_position[0]
        state_tensor = rl_task.sim_env.global_tensor_dict["robot_state_tensor"]
        state_tensor[1] = state_tensor[0].clone()
        # Force state write to sim
        if hasattr(rl_task.sim_env, "IGE_env") and hasattr(rl_task.sim_env.IGE_env, "write_to_sim"):
            rl_task.sim_env.IGE_env.write_to_sim()

        # Optional: manually invoke update_states to ensure derived quantities (Euler angles, etc.) don't lag
        if hasattr(rl_task.sim_env.robot_manager, "robot") and hasattr(rl_task.sim_env.robot_manager.robot, "update_states"):
            rl_task.sim_env.robot_manager.robot.update_states()

        for key, tensor in rl_task.obs_dict.items():
            if isinstance(tensor, torch.Tensor) and tensor.shape[0] == rl_task.num_envs:
                tensor[1] = tensor[0].clone()

    build_comparison_display()

    # Force initial simulation reset to seed valid robot states before our manual sync
    rl_task.sim_env.reset_idx(torch.arange(rl_task.num_envs, device=rl_task.device))
    rl_task.reset()

    _regenerate_goal_with_height_tolerance(rl_task, max_dz=float(eval_args.max_goal_spawn_dz))
    sync_robots_and_goal()
    _log_spawn_goal(rl_task, episode=0)

    command_actions = torch.zeros(
        (rl_task.num_envs, rl_task.task_config.action_space_dim),
        device=MatterportVAETaskConfig.device,
    )

    drones_done = torch.zeros(rl_task.num_envs, dtype=torch.bool, device=rl_task.device)
    drone_statuses = [None] * rl_task.num_envs
    episode_step = 0
    step_i = 0

    while stats.episodes < eval_args.max_episodes:
        obs, _, termination, truncation, infos = rl_task.step(command_actions)
        episode_step += 1

        if episode_step >= max_ep_steps:
             truncation[:] = True

        # Early success logic (independent of training env logic)
        dist = torch.norm(rl_task.target_position - rl_task.obs_dict["robot_position"], dim=1)
        early_success = (dist < 1.0) & (~termination.bool())
        truncation = truncation.bool() | early_success.bool()
        if "successes" not in infos:
            infos["successes"] = torch.zeros_like(truncation, dtype=torch.bool)
        if isinstance(infos["successes"], torch.Tensor):
            infos["successes"] = infos["successes"].bool() | early_success.bool()

        # Task handles action repeat (policy_every_n_steps) internally.
        obs["obs"] = obs["observations"]
        action = nn_model.get_action(obs)
        command_actions[:] = torch.as_tensor(action, device=command_actions.device).expand(
            rl_task.num_envs, -1
        )

        # Force action to 0 for done drones so they just wait
        command_actions[drones_done] = 0.0

        if step_i % eval_args.vis_every == 0:
            draw_debug(rl_task, rl_task.target_position[0].cpu().numpy())

        if step_i % display_every == 0:
            update_comparison_display(rl_task.obs_dict, drone_statuses)

        done = termination | truncation
        newly_done = done & (~drones_done)

        if newly_done.any():
            stats.record(newly_done & termination, newly_done & truncation, infos)

            for d_idx in range(rl_task.num_envs):
                if newly_done[d_idx]:
                    succ = bool(infos.get("successes", torch.zeros(rl_task.num_envs))[d_idx])
                    if termination[d_idx]:
                        drone_statuses[d_idx] = "CRASH"
                    elif succ:
                        drone_statuses[d_idx] = "SUCCESS"
                    else:
                        drone_statuses[d_idx] = "TIMEOUT"

            if newly_done[0]:
                logger.warning("Episode %d (DCE/Env0): %s", stats.episodes, drone_statuses[0])
            if newly_done[1]:
                logger.warning("Episode %d (ViT/Env1): %s", stats.episodes, drone_statuses[1])

            drones_done |= newly_done

        if drones_done.all():
            if stats.episodes % 10 == 0:
                stats.log()

            episode_step = 0
            drones_done[:] = False
            drone_statuses = [None] * rl_task.num_envs

            # Force sim level reset first
            rl_task.sim_env.reset_idx(torch.arange(rl_task.num_envs, device=rl_task.device))
            rl_task.reset()

            nn_model.reset(torch.arange(rl_task.num_envs))
            _regenerate_goal_with_height_tolerance(rl_task, max_dz=float(eval_args.max_goal_spawn_dz))
            sync_robots_and_goal()
            _log_spawn_goal(rl_task, episode=stats.episodes)

            command_actions[:] = 0.0

        step_i += 1

    stats.log()
    logger.warning("Evaluation complete after %d steps.", step_i)


# ===========================================================================
# Entry point
# ===========================================================================

def run_evaluation(eval_args) -> None:
    pipeline = eval_args.pipeline

    # Override task config cadence from CLI args (if provided).
    # task_config defaults (perception_every_n_steps=2, policy_every_n_steps=2) are used otherwise.
    for cfg in (MatterportVAETaskConfig, MatterportComparisonTaskConfig):
        if eval_args.perception_every is not None:
            cfg.perception_every_n_steps = max(1, int(eval_args.perception_every))
        if eval_args.policy_every is not None:
            cfg.policy_every_n_steps = max(1, int(eval_args.policy_every))

    # 1. Mutate env + task configs (must happen before task/env creation)
    setup_scene(eval_args)

    # 2. Build task
    rl_task = make_task(eval_args, pipeline)

    # Cadence is logged by MatterportDCENavigationTask.__init__ with Hz values.
    if MatterportGLBEnvCfg.env.render_viewer_every_n_steps > 1:
        logger.warning(
            "Viewer cadence throttled (viewer_every=%d). Use --viewer_every=1 for fluid interaction.",
            MatterportGLBEnvCfg.env.render_viewer_every_n_steps,
        )
    configure_runtime_viewer(rl_task, eval_args)

    # 3. Build RL policy
    nn_model = build_nn_model(rl_task.num_envs)
    nn_model.reset(torch.arange(rl_task.num_envs))

    # 4. Run the appropriate loop
    if pipeline == "comparison":
        _run_comparison_loop(eval_args, rl_task, nn_model)
    else:
        _run_dce_loop(eval_args, rl_task, nn_model)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    eval_args = parse_eval_args()
    run_evaluation(eval_args)
