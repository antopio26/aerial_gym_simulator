import math
import os
import random

import numpy as np
from PIL import Image
import trimesh as tm

from aerial_gym.config.env_config.matterport_glb_env import MatterportGLBEnvCfg
from aerial_gym.config.sensor_config.camera_config.shaded_rgbd_camera_config import (
    ShadedRGBDCameraConfig,
)
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.logging import CustomLogger
import torch


logger = CustomLogger(__name__)


def _parse_vec3(raw: str, name: str) -> np.ndarray:
    vals = [v.strip() for v in raw.split(",") if v.strip() != ""]
    if len(vals) != 3:
        raise ValueError(f"{name} must contain exactly 3 comma-separated values, got: {raw}")
    return np.array([float(vals[0]), float(vals[1]), float(vals[2])], dtype=np.float32)


def _apply_spawn_region_from_env(cfg_cls, prefix: str):
    """Configure env spawn bounds from env vars.

    Supported options (prefix = AERIAL_GYM_DENSE):
    - <prefix>_SPAWN_CENTER=x,y,z with <prefix>_SPAWN_BOUNDS=dx,dy,dz (half-extents)
    - <prefix>_SPAWN_LOWER=x,y,z with <prefix>_SPAWN_UPPER=x,y,z
    - <prefix>_SPAWN_FROM_SCENE_BOUNDS=1 (+ margin/offset knobs)
    """
    lower = None
    upper = None

    if os.getenv(f"{prefix}_SPAWN_FROM_SCENE_BOUNDS", "0") == "1":
        scene_file = cfg_cls.static_scene.file
        scene = tm.load(scene_file, force="scene")
        b = np.asarray(scene.bounds, dtype=np.float32)
        scale = float(getattr(cfg_cls.static_scene, "scale", 1.0))
        translation = np.asarray(
            getattr(cfg_cls.static_scene, "translation", [0.0, 0.0, 0.0]), dtype=np.float32
        )
        b = b * scale + translation[None, :]

        xy_margin = float(os.getenv(f"{prefix}_SCENE_XY_MARGIN", "0.4"))
        z_min_offset = float(os.getenv(f"{prefix}_SCENE_Z_MIN_OFFSET", "1.0"))
        z_max_offset = float(os.getenv(f"{prefix}_SCENE_Z_MAX_OFFSET", "0.6"))

        lower = np.array(
            [b[0, 0] + xy_margin, b[0, 1] + xy_margin, b[0, 2] + z_min_offset],
            dtype=np.float32,
        )
        upper = np.array(
            [b[1, 0] - xy_margin, b[1, 1] - xy_margin, b[1, 2] + z_max_offset],
            dtype=np.float32,
        )

    center_raw = os.getenv(f"{prefix}_SPAWN_CENTER", None)
    bounds_raw = os.getenv(f"{prefix}_SPAWN_BOUNDS", os.getenv(f"{prefix}_SPAWN_HALF_EXTENTS", None))
    if center_raw is not None and bounds_raw is not None:
        center = _parse_vec3(center_raw, f"{prefix}_SPAWN_CENTER")
        half = _parse_vec3(bounds_raw, f"{prefix}_SPAWN_BOUNDS")
        lower = center - half
        upper = center + half

    lower_raw = os.getenv(f"{prefix}_SPAWN_LOWER", None)
    upper_raw = os.getenv(f"{prefix}_SPAWN_UPPER", None)
    if lower_raw is not None and upper_raw is not None:
        lower = _parse_vec3(lower_raw, f"{prefix}_SPAWN_LOWER")
        upper = _parse_vec3(upper_raw, f"{prefix}_SPAWN_UPPER")

    if lower is None or upper is None:
        return

    if np.any(upper <= lower):
        raise ValueError(
            f"Invalid spawn bounds for {prefix}: lower={lower.tolist()} upper={upper.tolist()}"
        )

    cfg_cls.env.lower_bound_min = lower.tolist()
    cfg_cls.env.lower_bound_max = lower.tolist()
    cfg_cls.env.upper_bound_min = upper.tolist()
    cfg_cls.env.upper_bound_max = upper.tolist()

    logger.warning(
        "Configured spawn region from %s: lower=%s upper=%s",
        prefix,
        np.array2string(lower, precision=3),
        np.array2string(upper, precision=3),
    )


def _tile_images_grid(images_u8: np.ndarray) -> np.ndarray:
    """Tile N images into a near-square grid. Input shape: (N, H, W, C)."""
    n, h, w, c = images_u8.shape
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    grid = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
    for i in range(n):
        r = i // cols
        col = i % cols
        grid[r * h:(r + 1) * h, col * w:(col + 1) * w, :] = images_u8[i]
    return grid


if __name__ == "__main__":
    logger.warning("Running dense Matterport GLB viewer demo with close env spacing.")

    num_envs = int(os.getenv("AERIAL_GYM_DENSE_NUM_ENVS", "16"))
    env_spacing = float(os.getenv("AERIAL_GYM_DENSE_ENV_SPACING", "0.5"))
    num_steps = int(os.getenv("AERIAL_GYM_DEMO_STEPS", "900"))
    capture_every = int(os.getenv("AERIAL_GYM_DEMO_CAPTURE_EVERY", "4"))
    headless = os.getenv("AERIAL_GYM_DEMO_HEADLESS", "1") == "1"
    enable_lighting = os.getenv("AERIAL_GYM_TEXTURE_LIGHTING", "0") == "1"
    debug_uv_checker = os.getenv("AERIAL_GYM_DEBUG_UV_CHECKER", "0") == "1"

    # Keep this scene setup identical to matterport_glb_viewer_demo.py, except for spacing.
    MatterportGLBEnvCfg.env.env_spacing = env_spacing
    _apply_spawn_region_from_env(MatterportGLBEnvCfg, prefix="AERIAL_GYM_DENSE")
    ShadedRGBDCameraConfig.enable_lighting = enable_lighting
    ShadedRGBDCameraConfig.debug_uv_checker = debug_uv_checker

    if debug_uv_checker:
        mode_name = "uv_checker"
    else:
        mode_name = "lit" if enable_lighting else "texture_only"

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="matterport_glb_env",
        robot_name="base_quadrotor_with_shaded_rgbd_camera",
        controller_name="lee_velocity_control",
        args=None,
        device="cuda:0",
        num_envs=num_envs,
        headless=headless,
        use_warp=True,
    )
    env_manager.cfg.env.render_viewer_every_n_steps = 1

    actions = torch.zeros((env_manager.num_envs, 4), device="cuda:0")
    env_manager.reset()

    rgb_frames = []
    env_phase = torch.linspace(0.0, 2.0 * np.pi, env_manager.num_envs, device="cuda:0")

    for step in range(num_steps):
        t = float(step)
        t_tensor = torch.tensor(t, device="cuda:0")

        # Same velocity-control style as the original demo, with small per-env phase offsets
        # so replicated envs are visually easier to distinguish in the tiled GIF.
        actions[:, 0] = 0.55 + 0.20 * torch.sin(0.020 * t_tensor + env_phase)
        actions[:, 1] = 0.18 * torch.sin(0.011 * t_tensor + 0.5 * env_phase)
        actions[:, 2] = 0.10 * torch.cos(0.015 * t_tensor + 0.7 * env_phase)
        actions[:, 3] = 0.30 * torch.sin(0.008 * t_tensor + 0.9 * env_phase)

        env_manager.step(actions=actions)
        env_manager.render(render_components="sensors")
        env_manager.reset_terminated_and_truncated_envs()

        if step % capture_every == 0:
            # rgb_pixels shape: (num_envs, num_sensors, H, W, 3)
            rgb = env_manager.global_tensor_dict["rgb_pixels"][:, 0].detach().cpu().numpy()
            rgb_u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
            tiled = _tile_images_grid(rgb_u8)
            rgb_frames.append(Image.fromarray(tiled))

    out_dir = os.path.join(os.path.dirname(__file__), "stored_data")
    os.makedirs(out_dir, exist_ok=True)
    gif_path = os.path.join(
        out_dir,
        f"matterport_glb_viewer_dense_{num_envs}envs_spacing{env_spacing:.2f}_{mode_name}.gif",
    )

    if len(rgb_frames) == 0:
        raise RuntimeError("No RGB frames were captured during dense viewer demo.")

    rgb_frames[0].save(
        gif_path,
        save_all=True,
        append_images=rgb_frames[1:],
        duration=60,
        loop=0,
    )
    logger.warning("Saved dense viewer GIF: %s", gif_path)
