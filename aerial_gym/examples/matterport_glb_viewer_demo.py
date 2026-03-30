import os
import random

import numpy as np
from PIL import Image

from aerial_gym.benchmark.benchmark_utils import tile_images_grid
from aerial_gym.config.env_config.matterport_glb_env import MatterportGLBEnvCfg
from aerial_gym.benchmark.matterport_spawn_helpers import (
    apply_navmesh_config_from_env,
    apply_spawn_region_from_env,
    resolve_scene_bundle_from_env,
)
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.logging import CustomLogger
import torch

from aerial_gym.config.sensor_config.camera_config.shaded_rgbd_camera_config import (
    ShadedRGBDCameraConfig,
)


logger = CustomLogger(__name__)


if __name__ == "__main__":
    num_envs = int(os.getenv("AERIAL_GYM_DEMO_NUM_ENVS", "1"))
    env_spacing = float(os.getenv("AERIAL_GYM_DEMO_ENV_SPACING", "0.0" if num_envs == 1 else "0.5"))
    num_steps = int(os.getenv("AERIAL_GYM_DEMO_STEPS", "1200"))
    capture_every = int(os.getenv("AERIAL_GYM_DEMO_CAPTURE_EVERY", "4"))
    headless = os.getenv("AERIAL_GYM_DEMO_HEADLESS", "0") == "1"
    enable_lighting = os.getenv("AERIAL_GYM_TEXTURE_LIGHTING", "0") == "1"
    debug_uv_checker = os.getenv("AERIAL_GYM_DEBUG_UV_CHECKER", "0") == "1"
    controller_name = os.getenv("AERIAL_GYM_DEMO_CONTROLLER", "lee_position_control")

    logger.warning(
        "Running Matterport GLB viewer demo: num_envs=%d, env_spacing=%.2f",
        num_envs,
        env_spacing,
    )

    if num_envs > 1:
        MatterportGLBEnvCfg.env.env_spacing = env_spacing

    prefix = "AERIAL_GYM_DEMO"
    scene_bundle = resolve_scene_bundle_from_env(
        MatterportGLBEnvCfg, prefix=prefix, logger=logger,
    )
    apply_spawn_region_from_env(MatterportGLBEnvCfg, prefix=prefix, logger=logger)
    apply_navmesh_config_from_env(
        MatterportGLBEnvCfg, prefix=prefix, scene_bundle=scene_bundle, logger=logger,
    )
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
        controller_name=controller_name,
        args=None,
        device="cuda:0",
        num_envs=num_envs,
        headless=headless,
        use_warp=True,
    )
    env_manager.cfg.env.render_viewer_every_n_steps = 1

    actions = torch.zeros((env_manager.num_envs, 4), device="cuda:0")
    env_manager.reset()
    robot_position = env_manager.global_tensor_dict["robot_position"]
    robot_euler_angles = env_manager.global_tensor_dict.get("robot_euler_angles", None)
    if "env_origins" in env_manager.global_tensor_dict:
        env_origins = env_manager.global_tensor_dict["env_origins"]
    elif hasattr(env_manager, "env_origins"):
        env_origins = env_manager.env_origins
    else:
        env_origins = torch.zeros_like(robot_position)

    anchor_local_pos = torch.zeros((env_manager.num_envs, 3), device="cuda:0")
    anchor_yaw = torch.zeros(env_manager.num_envs, device="cuda:0")

    def refresh_anchors(env_ids=None):
        local_pos = robot_position - env_origins
        if env_ids is None:
            anchor_local_pos[:] = local_pos
            if robot_euler_angles is not None:
                anchor_yaw[:] = robot_euler_angles[:, 2]
            else:
                anchor_yaw[:] = 0.0
            return

        if len(env_ids) == 0:
            return
        anchor_local_pos[env_ids] = local_pos[env_ids]
        if robot_euler_angles is not None:
            anchor_yaw[env_ids] = robot_euler_angles[env_ids, 2]
        else:
            anchor_yaw[env_ids] = 0.0

    refresh_anchors()

    # Per-env phase offset for multi-env sinusoidal motion variety.
    env_phase = torch.linspace(0.0, 2.0 * np.pi, env_manager.num_envs, device="cuda:0")

    rgb_frames = []
    for step in range(num_steps):
        t = float(step)
        if num_envs > 1:
            actions[:, 0] = anchor_local_pos[:, 0] + 0.55 + 0.20 * torch.sin(0.020 * t + env_phase)
            actions[:, 1] = anchor_local_pos[:, 1] + 0.18 * torch.sin(0.011 * t + 0.5 * env_phase)
            actions[:, 2] = anchor_local_pos[:, 2] + 0.10 * torch.cos(0.015 * t + 0.7 * env_phase)
            actions[:, 3] = anchor_yaw + 0.30 * torch.sin(0.008 * t + 0.9 * env_phase)
        else:
            actions[:, 0] = anchor_local_pos[:, 0] + 0.55 + 0.20 * np.sin(0.020 * t)
            actions[:, 1] = anchor_local_pos[:, 1] + 0.18 * np.sin(0.011 * t)
            actions[:, 2] = anchor_local_pos[:, 2] + 0.10 * np.cos(0.015 * t)
            actions[:, 3] = anchor_yaw + 0.30 * np.sin(0.008 * t)

        env_manager.step(actions=actions)
        env_manager.render(render_components="sensors")
        reset_env_ids = env_manager.reset_terminated_and_truncated_envs()
        refresh_anchors(reset_env_ids)
        if len(reset_env_ids) > 0:
            actions[reset_env_ids, 0:3] = anchor_local_pos[reset_env_ids]
            actions[reset_env_ids, 3] = anchor_yaw[reset_env_ids]

        if step % capture_every == 0:
            rgb = env_manager.global_tensor_dict["rgb_pixels"][:, 0].detach().cpu().numpy()
            rgb_u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
            if num_envs > 1:
                rgb_frames.append(Image.fromarray(tile_images_grid(rgb_u8)))
            else:
                rgb_frames.append(Image.fromarray(rgb_u8[0]))

    out_dir = os.path.join(os.path.dirname(__file__), "stored_data")
    os.makedirs(out_dir, exist_ok=True)
    if num_envs > 1:
        gif_name = f"matterport_glb_viewer_{num_envs}envs_spacing{env_spacing:.2f}_{mode_name}.gif"
    else:
        gif_name = f"matterport_glb_viewer_demo_{mode_name}.gif"
    gif_path = os.path.join(out_dir, gif_name)

    if len(rgb_frames) == 0:
        raise RuntimeError("No RGB frames were captured during the viewer demo.")

    rgb_frames[0].save(
        gif_path,
        save_all=True,
        append_images=rgb_frames[1:],
        duration=60,
        loop=0,
    )
    logger.warning("Saved viewer demo GIF: %s", gif_path)
