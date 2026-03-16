import os
import random

import numpy as np
from PIL import Image

from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.logging import CustomLogger
import torch


logger = CustomLogger(__name__)


if __name__ == "__main__":
    logger.warning("Running Matterport GLB viewer demo with moving quadrotor.")

    num_steps = int(os.getenv("AERIAL_GYM_DEMO_STEPS", "600"))
    capture_every = int(os.getenv("AERIAL_GYM_DEMO_CAPTURE_EVERY", "4"))
    headless = os.getenv("AERIAL_GYM_DEMO_HEADLESS", "0") == "1"

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
        num_envs=1,
        headless=headless,
        use_warp=True,
    )
    env_manager.cfg.env.render_viewer_every_n_steps = 1

    actions = torch.zeros((env_manager.num_envs, 4), device="cuda:0")
    env_manager.reset()

    rgb_frames = []
    for step in range(num_steps):
        t = float(step)
        actions[:, 0] = 0.55 + 0.20 * np.sin(0.020 * t)
        actions[:, 1] = 0.18 * np.sin(0.011 * t)
        actions[:, 2] = 0.10 * np.cos(0.015 * t)
        actions[:, 3] = 0.30 * np.sin(0.008 * t)

        env_manager.step(actions=actions)
        env_manager.render(render_components="sensors")
        env_manager.reset_terminated_and_truncated_envs()

        if step % capture_every == 0:
            rgb = env_manager.global_tensor_dict["rgb_pixels"][0, 0].detach().cpu().numpy()
            rgb_u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
            rgb_frames.append(Image.fromarray(rgb_u8))

    out_dir = os.path.join(os.path.dirname(__file__), "stored_data")
    os.makedirs(out_dir, exist_ok=True)
    gif_path = os.path.join(out_dir, "matterport_glb_viewer_demo.gif")

    if len(rgb_frames) == 0:
        raise RuntimeError("No RGB frames were captured during the viewer demo.")

    rgb_frames[0].save(
        gif_path,
        save_all=True,
        append_images=rgb_frames[1:],
        duration=60,
        loop=0,
    )
    logger.warning(f"Saved viewer demo GIF: {gif_path}")