"""Shared utilities for benchmark scripts."""

import os
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base*, mutating *base* in place."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def tile_images_grid(images_u8: np.ndarray) -> np.ndarray:
    """Tile N images into a near-square grid. Input shape: (N, H, W, C)."""
    n, h, w, c = images_u8.shape
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    grid = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
    for i in range(n):
        r = i // cols
        col = i % cols
        grid[r * h : (r + 1) * h, col * w : (col + 1) * w, :] = images_u8[i]
    return grid


def collect_grid_frame(env_manager) -> Image.Image:
    """Capture a tiled grid frame from an env_manager (RGB preferred, depth fallback)."""
    rgb_tensor = env_manager.global_tensor_dict.get("rgb_pixels", None)
    if rgb_tensor is not None:
        rgb = rgb_tensor[:, 0].detach().cpu().numpy()
        rgb_u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(tile_images_grid(rgb_u8))

    depth_tensor = env_manager.global_tensor_dict.get("depth_range_pixels", None)
    if depth_tensor is not None:
        depth = depth_tensor[:, 0].detach().cpu().numpy()
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
        return Image.fromarray(tile_images_grid(np.repeat(depth_u8[..., None], 3, axis=-1)))

    raise RuntimeError("Neither rgb_pixels nor depth_range_pixels is available for frame capture.")


def maybe_save_case_gif(
    frames,
    label: str,
    num_envs: int,
    gif_duration_ms: int,
) -> Optional[str]:
    """Save captured frames as an animated GIF. Returns the path or None."""
    if len(frames) == 0:
        return None
    out_dir = os.path.join(os.path.dirname(__file__), "stored_data", "benchmark_gifs")
    os.makedirs(out_dir, exist_ok=True)
    gif_path = os.path.join(out_dir, f"benchmark_{label}_{num_envs}envs.gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=gif_duration_ms,
        loop=0,
    )
    return gif_path


def compute_benchmark_metrics(
    bench_steps: int,
    num_envs: int,
    elapsed: float,
    sim_dt: float,
    render_count: int,
    render_every: int,
) -> dict:
    """Compute FPS and real-time factor metrics from a benchmark run."""
    fps = (bench_steps * num_envs) / max(elapsed, 1.0e-9)
    rtf = (bench_steps * num_envs * sim_dt) / max(elapsed, 1.0e-9)
    fps_per_env = fps / max(num_envs, 1)
    rtf_per_env = rtf / max(num_envs, 1)
    rendered_fps = (render_count * num_envs) / max(elapsed, 1.0e-9)
    rendered_fps_per_env = rendered_fps / max(num_envs, 1)
    render_dt = sim_dt * render_every
    rendered_rtf = (render_count * num_envs * render_dt) / max(elapsed, 1.0e-9)
    rendered_rtf_per_env = rendered_rtf / max(num_envs, 1)
    return {
        "fps": fps,
        "fps_per_env": fps_per_env,
        "real_time_speedup": rtf,
        "real_time_speedup_per_env": rtf_per_env,
        "rendered_fps": rendered_fps,
        "rendered_fps_per_env": rendered_fps_per_env,
        "rendered_real_time_speedup": rendered_rtf,
        "rendered_real_time_speedup_per_env": rendered_rtf_per_env,
    }
