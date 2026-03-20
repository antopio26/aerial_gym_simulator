import gc
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
from PIL import Image

from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import (
    BaseDepthCameraConfig,
)
from aerial_gym.config.env_config.matterport_glb_env import MatterportGLBEnvCfg
from aerial_gym.config.sensor_config.camera_config.shaded_rgbd_camera_config import (
    ShadedRGBDCameraConfig,
)
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.logging import CustomLogger
import torch
import trimesh as tm


logger = CustomLogger(__name__)


def _parse_env_counts(raw: str):
    counts = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            counts.append(int(token))
    if not counts:
        raise ValueError("AERIAL_GYM_BENCH_ENVS must contain at least one integer.")
    return counts


def _configure_camera_classes(width: int, height: int, max_range: float, enable_lighting: bool):
    # Keep both camera pipelines at identical resolution/range for fair benchmarking.
    BaseDepthCameraConfig.width = width
    BaseDepthCameraConfig.height = height
    BaseDepthCameraConfig.max_range = max_range
    BaseDepthCameraConfig.segmentation_camera = False
    BaseDepthCameraConfig.calculate_depth = True

    ShadedRGBDCameraConfig.width = width
    ShadedRGBDCameraConfig.height = height
    ShadedRGBDCameraConfig.max_range = max_range
    ShadedRGBDCameraConfig.enable_lighting = enable_lighting
    ShadedRGBDCameraConfig.debug_uv_checker = False


def _parse_vec3(raw: str, name: str) -> np.ndarray:
    vals = [v.strip() for v in raw.split(",") if v.strip() != ""]
    if len(vals) != 3:
        raise ValueError(f"{name} must contain exactly 3 comma-separated values, got: {raw}")
    return np.array([float(vals[0]), float(vals[1]), float(vals[2])], dtype=np.float32)


def _apply_spawn_region_from_env(cfg_cls, prefix: str):
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
        return None

    if np.any(upper <= lower):
        raise ValueError(
            f"Invalid spawn bounds for {prefix}: lower={lower.tolist()} upper={upper.tolist()}"
        )

    cfg_cls.env.lower_bound_min = lower.tolist()
    cfg_cls.env.lower_bound_max = lower.tolist()
    cfg_cls.env.upper_bound_min = upper.tolist()
    cfg_cls.env.upper_bound_max = upper.tolist()

    logger.warning(
        "Configured benchmark spawn region from %s: lower=%s upper=%s",
        prefix,
        np.array2string(lower, precision=3),
        np.array2string(upper, precision=3),
    )
    return {
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "source_prefix": prefix,
    }


def _tile_images_grid(images_u8: np.ndarray) -> np.ndarray:
    """Tile N images into a near-square grid. Input: (N,H,W,C) uint8."""
    n, h, w, c = images_u8.shape
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    grid = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
    for i in range(n):
        r = i // cols
        col = i % cols
        grid[r * h:(r + 1) * h, col * w:(col + 1) * w, :] = images_u8[i]
    return grid


def _collect_grid_frame(env_manager) -> Image.Image:
    rgb_tensor = env_manager.global_tensor_dict.get("rgb_pixels", None)
    if rgb_tensor is not None:
        rgb = rgb_tensor[:, 0].detach().cpu().numpy()  # (N, H, W, 3)
        rgb_u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        grid_u8 = _tile_images_grid(rgb_u8)
        return Image.fromarray(grid_u8)

    depth_tensor = env_manager.global_tensor_dict.get("depth_range_pixels", None)
    if depth_tensor is not None:
        depth = depth_tensor[:, 0].detach().cpu().numpy()  # (N, H, W)
        finite = np.isfinite(depth)
        if np.any(finite):
            d_min = float(np.min(depth[finite]))
            d_max = float(np.max(depth[finite]))
            if d_max - d_min > 1.0e-6:
                depth_norm = (depth - d_min) / (d_max - d_min)
            else:
                depth_norm = np.zeros_like(depth, dtype=np.float32)
        else:
            depth_norm = np.zeros_like(depth, dtype=np.float32)
        depth_u8 = np.clip(depth_norm * 255.0, 0.0, 255.0).astype(np.uint8)
        depth_rgb_u8 = np.repeat(depth_u8[..., None], 3, axis=-1)
        grid_u8 = _tile_images_grid(depth_rgb_u8)
        return Image.fromarray(grid_u8)

    raise RuntimeError("Neither rgb_pixels nor depth_range_pixels is available for GIF capture.")


def _maybe_save_case_gif(
    frames,
    robot_name: str,
    num_envs: int,
    gif_duration_ms: int,
):
    if len(frames) == 0:
        return None
    out_dir = os.path.join(os.path.dirname(__file__), "stored_data", "benchmark_gifs")
    os.makedirs(out_dir, exist_ok=True)
    gif_path = os.path.join(out_dir, f"benchmark_{robot_name}_{num_envs}envs.gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=gif_duration_ms,
        loop=0,
    )
    return gif_path


def _run_case(
    robot_name: str,
    controller_name: str,
    num_envs: int,
    warmup_steps: int,
    bench_steps: int,
    device: str,
    headless: bool,
    save_gif: bool,
    gif_every: int,
    gif_max_frames: int,
    gif_duration_ms: int,
    render_every: int,
):
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="matterport_glb_env",
        robot_name=robot_name,
        controller_name=controller_name,
        args=None,
        device=device,
        num_envs=num_envs,
        headless=headless,
        use_warp=True,
    )

    actions = torch.zeros((env_manager.num_envs, 4), device=device)
    env_manager.reset()
    frames = []
    total_steps = 0
    render_count = 0
    render_every = max(int(render_every), 1)

    robot_position = env_manager.global_tensor_dict["robot_position"]
    robot_euler_angles = env_manager.global_tensor_dict.get("robot_euler_angles", None)

    def maybe_capture_frame():
        nonlocal total_steps
        if (not save_gif) or gif_every <= 0 or len(frames) >= gif_max_frames:
            return
        if total_steps % gif_every != 0:
            return
        frames.append(_collect_grid_frame(env_manager))

    with torch.no_grad():
        for _ in range(warmup_steps):
            # Lee position controller expects [x, y, z, yaw]. Holding current pose
            # avoids fall/reset churn and keeps benchmark focused on sensor/render cost.
            actions[:, 0:3] = robot_position
            if robot_euler_angles is not None:
                actions[:, 3] = robot_euler_angles[:, 2]
            else:
                actions[:, 3] = 0.0
            env_manager.step(actions=actions)
            if total_steps % render_every == 0:
                env_manager.render(render_components="sensors")
                render_count += 1
            env_manager.reset_terminated_and_truncated_envs()
            maybe_capture_frame()
            total_steps += 1

        render_count = 0
        start = time.time()
        for _ in range(bench_steps):
            actions[:, 0:3] = robot_position
            if robot_euler_angles is not None:
                actions[:, 3] = robot_euler_angles[:, 2]
            else:
                actions[:, 3] = 0.0
            env_manager.step(actions=actions)
            if total_steps % render_every == 0:
                env_manager.render(render_components="sensors")
                render_count += 1
            env_manager.reset_terminated_and_truncated_envs()
            maybe_capture_frame()
            total_steps += 1
        elapsed = time.time() - start

    sim_dt = float(env_manager.sim_config.sim.dt)
    fps = (bench_steps * num_envs) / max(elapsed, 1.0e-9)
    rtf = (bench_steps * num_envs * sim_dt) / max(elapsed, 1.0e-9)
    fps_per_env = fps / max(num_envs, 1)
    rtf_per_env = rtf / max(num_envs, 1)
    rendered_fps = (render_count * num_envs) / max(elapsed, 1.0e-9)
    rendered_fps_per_env = rendered_fps / max(num_envs, 1)
    render_dt = sim_dt * render_every
    rendered_rtf = (render_count * num_envs * render_dt) / max(elapsed, 1.0e-9)
    rendered_rtf_per_env = rendered_rtf / max(num_envs, 1)

    del env_manager
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gif_path = _maybe_save_case_gif(
        frames=frames,
        robot_name=robot_name,
        num_envs=num_envs,
        gif_duration_ms=gif_duration_ms,
    )

    return {
        "robot_name": robot_name,
        "controller_name": controller_name,
        "num_envs": num_envs,
        "elapsed_s": elapsed,
        "fps": fps,
        "fps_per_env": fps_per_env,
        "real_time_speedup": rtf,
        "real_time_speedup_per_env": rtf_per_env,
        "render_count": render_count,
        "render_every": render_every,
        "rendered_fps": rendered_fps,
        "rendered_fps_per_env": rendered_fps_per_env,
        "rendered_real_time_speedup": rendered_rtf,
        "rendered_real_time_speedup_per_env": rendered_rtf_per_env,
        "gif_path": gif_path,
    }


def _run_case_subprocess(
    robot_name: str,
    controller_name: str,
    num_envs: int,
    warmup_steps: int,
    bench_steps: int,
    width: int,
    height: int,
    max_range: float,
    enable_lighting: bool,
    device: str,
    headless: bool,
    save_gif: bool,
    gif_every: int,
    gif_max_frames: int,
    gif_duration_ms: int,
    render_every: int,
):
    env = os.environ.copy()
    env.update(
        {
            "AERIAL_GYM_BENCH_CHILD": "1",
            "AERIAL_GYM_BENCH_ROBOT_NAME": robot_name,
            "AERIAL_GYM_BENCH_CONTROLLER_NAME": controller_name,
            "AERIAL_GYM_BENCH_NUM_ENVS": str(num_envs),
            "AERIAL_GYM_BENCH_WARMUP_STEPS": str(warmup_steps),
            "AERIAL_GYM_BENCH_STEPS": str(bench_steps),
            "AERIAL_GYM_BENCH_CAM_WIDTH": str(width),
            "AERIAL_GYM_BENCH_CAM_HEIGHT": str(height),
            "AERIAL_GYM_BENCH_MAX_RANGE": str(max_range),
            "AERIAL_GYM_BENCH_RGBD_LIGHTING": "1" if enable_lighting else "0",
            "AERIAL_GYM_BENCH_DEVICE": device,
            "AERIAL_GYM_BENCH_HEADLESS": "1" if headless else "0",
            "AERIAL_GYM_BENCH_SAVE_GIFS": "1" if save_gif else "0",
            "AERIAL_GYM_BENCH_GIF_EVERY": str(gif_every),
            "AERIAL_GYM_BENCH_GIF_MAX_FRAMES": str(gif_max_frames),
            "AERIAL_GYM_BENCH_GIF_DURATION_MS": str(gif_duration_ms),
            "AERIAL_GYM_BENCH_RENDER_EVERY": str(render_every),
        }
    )

    cmd = [sys.executable, __file__]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Child benchmark failed for robot={robot_name}, num_envs={num_envs}.\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:") :].strip())

    raise RuntimeError(
        f"Child benchmark did not emit RESULT_JSON for robot={robot_name}, num_envs={num_envs}.\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )


def _print_table(depth_rows, rgbd_rows):
    logger.warning("\n=== Matterport Parallel Benchmark: Depth-only vs RGBD ===")
    print(
        f"{'envs':>6} | {'depth FPS':>12} | {'rgbd FPS':>12} | {'d FPS/env':>10} | {'r FPS/env':>10} | {'rgbd/depth':>11} | "
        f"{'depth RTF':>10} | {'rgbd RTF':>10}"
    )
    print("-" * 110)
    for d, r in zip(depth_rows, rgbd_rows):
        ratio = r["fps"] / max(d["fps"], 1.0e-9)
        print(
            f"{d['num_envs']:6d} | {d['fps']:12.2f} | {r['fps']:12.2f} | {d['fps_per_env']:10.2f} | {r['fps_per_env']:10.2f} | {ratio:11.3f} | "
            f"{d['real_time_speedup']:10.2f} | {r['real_time_speedup']:10.2f}"
        )


if __name__ == "__main__":
    if os.getenv("AERIAL_GYM_BENCH_CHILD", "0") == "1":
        robot_name = os.getenv("AERIAL_GYM_BENCH_ROBOT_NAME", "base_quadrotor_with_camera")
        controller_name = os.getenv("AERIAL_GYM_BENCH_CONTROLLER_NAME", "lee_position_control")
        num_envs = int(os.getenv("AERIAL_GYM_BENCH_NUM_ENVS", "1"))
        warmup_steps = int(os.getenv("AERIAL_GYM_BENCH_WARMUP_STEPS", "100"))
        bench_steps = int(os.getenv("AERIAL_GYM_BENCH_STEPS", "300"))
        width = int(os.getenv("AERIAL_GYM_BENCH_CAM_WIDTH", "240"))
        height = int(os.getenv("AERIAL_GYM_BENCH_CAM_HEIGHT", "135"))
        max_range = float(os.getenv("AERIAL_GYM_BENCH_MAX_RANGE", "80.0"))
        enable_lighting = os.getenv("AERIAL_GYM_BENCH_RGBD_LIGHTING", "0") == "1"
        headless = os.getenv("AERIAL_GYM_BENCH_HEADLESS", "1") == "1"
        device = os.getenv("AERIAL_GYM_BENCH_DEVICE", "cuda:0")
        save_gif = os.getenv("AERIAL_GYM_BENCH_SAVE_GIFS", "0") == "1"
        gif_every = int(os.getenv("AERIAL_GYM_BENCH_GIF_EVERY", "5"))
        gif_max_frames = int(os.getenv("AERIAL_GYM_BENCH_GIF_MAX_FRAMES", "300"))
        gif_duration_ms = int(os.getenv("AERIAL_GYM_BENCH_GIF_DURATION_MS", "60"))
        render_every = int(os.getenv("AERIAL_GYM_BENCH_RENDER_EVERY", "1"))

        _configure_camera_classes(
            width=width,
            height=height,
            max_range=max_range,
            enable_lighting=enable_lighting,
        )
        _apply_spawn_region_from_env(MatterportGLBEnvCfg, prefix="AERIAL_GYM_BENCH")
        row = _run_case(
            robot_name=robot_name,
            controller_name=controller_name,
            num_envs=num_envs,
            warmup_steps=warmup_steps,
            bench_steps=bench_steps,
            device=device,
            headless=headless,
            save_gif=save_gif,
            gif_every=gif_every,
            gif_max_frames=gif_max_frames,
            gif_duration_ms=gif_duration_ms,
            render_every=render_every,
        )
        print("RESULT_JSON:" + json.dumps(row))
        raise SystemExit(0)

    seed = int(os.getenv("AERIAL_GYM_BENCH_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    env_counts = _parse_env_counts(os.getenv("AERIAL_GYM_BENCH_ENVS", "1,2,4,8,16"))
    warmup_steps = int(os.getenv("AERIAL_GYM_BENCH_WARMUP_STEPS", "100"))
    bench_steps = int(os.getenv("AERIAL_GYM_BENCH_STEPS", "300"))
    width = int(os.getenv("AERIAL_GYM_BENCH_CAM_WIDTH", "240"))
    height = int(os.getenv("AERIAL_GYM_BENCH_CAM_HEIGHT", "135"))
    max_range = float(os.getenv("AERIAL_GYM_BENCH_MAX_RANGE", "80.0"))
    enable_lighting = os.getenv("AERIAL_GYM_BENCH_RGBD_LIGHTING", "0") == "1"
    headless = os.getenv("AERIAL_GYM_BENCH_HEADLESS", "1") == "1"
    device = os.getenv("AERIAL_GYM_BENCH_DEVICE", "cuda:0")
    save_gifs = os.getenv("AERIAL_GYM_BENCH_SAVE_GIFS", "0") == "1"
    gif_every = int(os.getenv("AERIAL_GYM_BENCH_GIF_EVERY", "5"))
    gif_max_frames = int(os.getenv("AERIAL_GYM_BENCH_GIF_MAX_FRAMES", "300"))
    gif_duration_ms = int(os.getenv("AERIAL_GYM_BENCH_GIF_DURATION_MS", "60"))
    render_every = int(os.getenv("AERIAL_GYM_BENCH_RENDER_EVERY", "1"))
    controller_name = os.getenv("AERIAL_GYM_BENCH_CONTROLLER_NAME", "lee_position_control")

    _configure_camera_classes(
        width=width,
        height=height,
        max_range=max_range,
        enable_lighting=enable_lighting,
    )
    spawn_region = _apply_spawn_region_from_env(MatterportGLBEnvCfg, prefix="AERIAL_GYM_BENCH")

    logger.warning(
        "Running benchmark on env counts %s, warmup=%d, bench=%d, resolution=%dx%d, max_range=%.1f",
        env_counts,
        warmup_steps,
        bench_steps,
        width,
        height,
        max_range,
    )

    depth_rows = []
    rgbd_rows = []

    for n in env_counts:
        logger.warning("Depth-only case: num_envs=%d", n)
        depth_rows.append(
            _run_case_subprocess(
                robot_name="base_quadrotor_with_camera",
                controller_name=controller_name,
                num_envs=n,
                warmup_steps=warmup_steps,
                bench_steps=bench_steps,
                width=width,
                height=height,
                max_range=max_range,
                enable_lighting=enable_lighting,
                device=device,
                headless=headless,
                save_gif=save_gifs,
                gif_every=gif_every,
                gif_max_frames=gif_max_frames,
                gif_duration_ms=gif_duration_ms,
                render_every=render_every,
            )
        )

        logger.warning("Shaded RGBD case: num_envs=%d", n)
        rgbd_rows.append(
            _run_case_subprocess(
                robot_name="base_quadrotor_with_shaded_rgbd_camera",
                controller_name=controller_name,
                num_envs=n,
                warmup_steps=warmup_steps,
                bench_steps=bench_steps,
                width=width,
                height=height,
                max_range=max_range,
                enable_lighting=enable_lighting,
                device=device,
                headless=headless,
                save_gif=save_gifs,
                gif_every=gif_every,
                gif_max_frames=gif_max_frames,
                gif_duration_ms=gif_duration_ms,
                render_every=render_every,
            )
        )

    _print_table(depth_rows, rgbd_rows)

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "settings": {
            "env_counts": env_counts,
            "warmup_steps": warmup_steps,
            "bench_steps": bench_steps,
            "width": width,
            "height": height,
            "max_range": max_range,
            "enable_lighting": enable_lighting,
            "controller_name": controller_name,
            "render_every": render_every,
            "headless": headless,
            "device": device,
            "save_gifs": save_gifs,
            "gif_every": gif_every,
            "gif_max_frames": gif_max_frames,
            "gif_duration_ms": gif_duration_ms,
            "spawn_region": spawn_region,
        },
        "depth_only": depth_rows,
        "shaded_rgbd": rgbd_rows,
    }

    out_dir = os.path.join(os.path.dirname(__file__), "stored_data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "matterport_depth_vs_rgbd_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.warning("Saved benchmark report: %s", out_path)
