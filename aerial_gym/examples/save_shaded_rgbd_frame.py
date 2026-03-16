import os
import random
import numpy as np
from PIL import Image

from aerial_gym.utils.logging import CustomLogger
from aerial_gym.sim.sim_builder import SimBuilder
import torch

logger = CustomLogger(__name__)


if __name__ == "__main__":
    logger.warning("Running shaded RGBD frame capture example.")

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="env_with_obstacles",
        robot_name="base_quadrotor_with_shaded_rgbd_camera",
        controller_name="lee_velocity_control",
        args=None,
        device="cuda:0",
        num_envs=1,
        headless=True,
        use_warp=True,
    )

    actions = torch.zeros((env_manager.num_envs, 4), device="cuda:0")
    actions[:, 3] = 0.1

    env_manager.reset()
    for i in range(20):
        env_manager.step(actions=actions)
        env_manager.render(render_components="sensors")
        env_manager.reset_terminated_and_truncated_envs()

    rgb_tensor = env_manager.global_tensor_dict.get("rgb_pixels", None)
    depth_tensor = env_manager.global_tensor_dict.get("depth_range_pixels", None)

    if rgb_tensor is None or depth_tensor is None:
        raise RuntimeError("Shaded RGBD tensors were not created. Check robot/sensor configuration.")

    logger.info(f"rgb_pixels shape: {tuple(rgb_tensor.shape)}")
    logger.info(f"depth_range_pixels shape: {tuple(depth_tensor.shape)}")
    logger.info("rgb sum: %.4f" % rgb_tensor.sum().item())
    logger.info(
        "depth stats: min=%.4f max=%.4f" % (depth_tensor.min().item(), depth_tensor.max().item())
    )

    if rgb_tensor.sum().item() <= 0.0:
        raise RuntimeError("RGB tensor is empty after sensor capture. Warp sensors may not have been rendered.")

    rgb = rgb_tensor[0, 0].detach().cpu().numpy()
    depth = depth_tensor[0, 0].detach().cpu().numpy()

    rgb_u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)

    depth_valid = depth[np.isfinite(depth)]
    if depth_valid.size == 0:
        depth_u8 = np.zeros_like(depth, dtype=np.uint8)
    else:
        d_min = np.min(depth_valid)
        d_max = np.max(depth_valid)
        if d_max - d_min < 1e-6:
            depth_u8 = np.zeros_like(depth, dtype=np.uint8)
        else:
            depth_norm = (depth - d_min) / (d_max - d_min)
            depth_u8 = np.clip(depth_norm * 255.0, 0.0, 255.0).astype(np.uint8)

    out_dir = os.path.join(os.path.dirname(__file__), "stored_data")
    os.makedirs(out_dir, exist_ok=True)

    rgb_path = os.path.join(out_dir, "shaded_rgb_frame.png")
    depth_path = os.path.join(out_dir, "shaded_depth_frame.png")

    Image.fromarray(rgb_u8).save(rgb_path)
    Image.fromarray(depth_u8).save(depth_path)

    logger.warning(f"Saved RGB frame: {rgb_path}")
    logger.warning(f"Saved depth frame: {depth_path}")
