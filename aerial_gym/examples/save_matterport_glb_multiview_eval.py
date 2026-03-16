import os
import random

import numpy as np
from PIL import Image, ImageDraw

from aerial_gym.config.sensor_config.camera_config.shaded_rgbd_camera_config import (
    ShadedRGBDCameraConfig,
)
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.logging import CustomLogger
import torch


logger = CustomLogger(__name__)


def _normalize(vec):
    norm = np.linalg.norm(vec)
    if norm < 1.0e-8:
        return vec
    return vec / norm


def _rotation_matrix_to_quaternion(matrix):
    trace = float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2])
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (matrix[2, 1] - matrix[1, 2]) * s
        qy = (matrix[0, 2] - matrix[2, 0]) * s
        qz = (matrix[1, 0] - matrix[0, 1]) * s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        qw = (matrix[2, 1] - matrix[1, 2]) / s
        qx = 0.25 * s
        qy = (matrix[0, 1] + matrix[1, 0]) / s
        qz = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        qw = (matrix[0, 2] - matrix[2, 0]) / s
        qx = (matrix[0, 1] + matrix[1, 0]) / s
        qy = 0.25 * s
        qz = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        qw = (matrix[1, 0] - matrix[0, 1]) / s
        qx = (matrix[0, 2] + matrix[2, 0]) / s
        qy = (matrix[1, 2] + matrix[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float32)
    quat /= np.linalg.norm(quat)
    return quat


def _look_at_quaternion(camera_pos, target_pos, world_up=np.array([0.0, 0.0, 1.0], dtype=np.float32)):
    forward = _normalize(target_pos - camera_pos)
    right = _normalize(np.cross(world_up, forward))
    if np.linalg.norm(right) < 1.0e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    up = _normalize(np.cross(forward, right))
    rotation = np.stack([right, up, forward], axis=1)
    return _rotation_matrix_to_quaternion(rotation)


def _save_contact_sheet(images, labels, out_path, columns=3):
    if not images:
        raise RuntimeError("No images available for contact sheet.")

    width, height = images[0].size
    rows = (len(images) + columns - 1) // columns
    label_h = 24
    canvas = Image.new("RGB", (columns * width, rows * (height + label_h)), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    for idx, (image, label) in enumerate(zip(images, labels)):
        row = idx // columns
        col = idx % columns
        x = col * width
        y = row * (height + label_h)
        canvas.paste(image, (x, y))
        draw.text((x + 6, y + height + 4), label, fill=(240, 240, 240))

    canvas.save(out_path)


if __name__ == "__main__":
    logger.warning("Running Matterport GLB multi-view evaluation capture.")

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    enable_lighting = os.getenv("AERIAL_GYM_TEXTURE_LIGHTING", "0") == "1"
    debug_uv_checker = os.getenv("AERIAL_GYM_DEBUG_UV_CHECKER", "0") == "1"
    ShadedRGBDCameraConfig.enable_lighting = enable_lighting
    ShadedRGBDCameraConfig.debug_uv_checker = debug_uv_checker
    ShadedRGBDCameraConfig.max_range = float(os.getenv("AERIAL_GYM_EVAL_MAX_RANGE", "80.0"))
    ShadedRGBDCameraConfig.min_range = float(os.getenv("AERIAL_GYM_EVAL_MIN_RANGE", "0.2"))
    if debug_uv_checker:
        mode_name = "uv_checker"
    else:
        mode_name = "lit" if enable_lighting else "texture_only"
    ShadedRGBDCameraConfig.width = int(os.getenv("AERIAL_GYM_EVAL_WIDTH", "640"))
    ShadedRGBDCameraConfig.height = int(os.getenv("AERIAL_GYM_EVAL_HEIGHT", "360"))

    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="matterport_glb_env",
        robot_name="base_quadrotor_with_shaded_rgbd_camera",
        controller_name="lee_velocity_control",
        args=None,
        device="cuda:0",
        num_envs=1,
        headless=True,
        use_warp=True,
    )
    env_manager.reset()

    scene_bounds = env_manager.static_scene.bounds
    scene_min = scene_bounds[0].astype(np.float32)
    scene_max = scene_bounds[1].astype(np.float32)
    scene_center = 0.5 * (scene_min + scene_max)
    scene_extent = scene_max - scene_min
    radius = 0.95 * float(np.linalg.norm(scene_extent[:2])) + 2.0
    target = scene_center + np.array([0.0, 0.0, 0.15 * scene_extent[2]], dtype=np.float32)
    mid_z = float(scene_center[2] + 0.35 * scene_extent[2] + 0.5)
    high_z = float(scene_center[2] + 0.95 * scene_extent[2] + 1.0)

    view_specs = [
        (np.array([scene_center[0] + radius, scene_center[1], mid_z], dtype=np.float32), "east"),
        (np.array([scene_center[0] - radius, scene_center[1], mid_z], dtype=np.float32), "west"),
        (np.array([scene_center[0], scene_center[1] + radius, mid_z], dtype=np.float32), "north"),
        (np.array([scene_center[0], scene_center[1] - radius, mid_z], dtype=np.float32), "south"),
        (np.array([scene_center[0] + 0.75 * radius, scene_center[1] + 0.75 * radius, high_z], dtype=np.float32), "northeast_high"),
        (np.array([scene_center[0] - 0.75 * radius, scene_center[1] - 0.75 * radius, high_z], dtype=np.float32), "southwest_high"),
    ]

    warp_sensor = env_manager.robot_manager.warp_sensor
    warp_sensor.sensor.capture()

    images = []
    labels = []
    out_dir = os.path.join(os.path.dirname(__file__), "stored_data")
    os.makedirs(out_dir, exist_ok=True)

    for camera_pos_np, label in view_specs:
        camera_quat_np = _look_at_quaternion(camera_pos_np, target)
        warp_sensor.sensor_position[0, 0] = torch.tensor(camera_pos_np, device="cuda:0")
        warp_sensor.sensor_orientation[0, 0] = torch.tensor(camera_quat_np, device="cuda:0")
        warp_sensor.sensor.capture()

        rgb = env_manager.global_tensor_dict["rgb_pixels"][0, 0].detach().cpu().numpy()
        rgb_u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        image = Image.fromarray(rgb_u8)
        images.append(image)
        labels.append(label)
        image.save(os.path.join(out_dir, f"matterport_glb_{mode_name}_{label}.png"))

    contact_sheet_path = os.path.join(out_dir, f"matterport_glb_multiview_{mode_name}.png")
    _save_contact_sheet(images, labels, contact_sheet_path)
    logger.warning(f"Saved multi-view contact sheet: {contact_sheet_path}")